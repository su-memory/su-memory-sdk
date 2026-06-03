#!/usr/bin/env python3
"""
LoCoMo Benchmark Runner for su-memory
=====================================
LoCoMo (Long Conversational Memory) 是 ACL 2024 发布的超长对话记忆评测基准。
数据集 ``snap-research/locomo`` 包含 10 个超长对话，每个跨越多达 35 个会话、
约 300 轮以上，配套 QA 问答、事件总结、跨会话推理与时间因果四大子任务。

任务说明：
- ``qa``:               基于对话历史回答问题（EM / F1）
- ``event_summary``:    事件总结任务（ROUGE-L F1）
- ``cross_session``:    跨会话推理任务（Accuracy）
- ``temporal_causal``:  时间因果推理任务（Accuracy）

参考：
- LoCoMo: https://snap-research.github.io/locomo/
- HF id:  snap-research/locomo
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Path setup ----------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
sys.path.insert(0, str(_THIS_FILE.parent.parent))
sys.path.insert(0, str(_THIS_FILE.parent.parent / "src"))

from benchmarks.config import (  # noqa: E402
    BACKENDS,
    BenchmarkResult,
    COMPETITOR_SCORES,
    DATASETS,
    compute_f1,
    compute_rouge_l,
    ensure_data_dir,
    exact_match,
    semantic_match,
)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _safe_get(obj: Any, *keys: str, default: Any = None) -> Any:
    """从 dict / 对象中按多个候选键查找首个存在的字段。"""

    if obj is None:
        return default
    for key in keys:
        if isinstance(obj, dict) and key in obj and obj[key] is not None:
            return obj[key]
        if hasattr(obj, key) and getattr(obj, key) is not None:
            return getattr(obj, key)
    return default


def _coerce_text(value: Any) -> str:
    """将任意结构强制转为字符串内容。"""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(_coerce_text(v) for v in value)
    if isinstance(value, dict):
        for k in ("text", "content", "answer", "value", "summary"):
            if k in value:
                return _coerce_text(value[k])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _chunk_text(text: str, chunk_size: int = 200) -> list[str]:
    """按字符数粗分块（保留句末空白）。"""

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size
    return chunks


# ---------------------------------------------------------------------------
# 主运行器
# ---------------------------------------------------------------------------

class LoCoMoRunner:
    """LoCoMo 超长对话记忆评测器。

    支持 ``ollama`` / ``sbert`` 等 ``benchmarks.config.BACKENDS`` 中定义的后端，
    以及 QA / 事件总结 / 跨会话推理 / 时间因果四个子任务。
    """

    DEFAULT_TASKS: tuple[str, ...] = (
        "qa",
        "event_summary",
        "cross_session",
        "temporal_causal",
    )

    # ---- 初始化 ----------------------------------------------------------

    def __init__(
        self,
        backend: str = "ollama",
        storage_path: str | None = None,
        verbose: bool = False,
    ) -> None:
        if backend not in BACKENDS:
            raise ValueError(
                f"Unknown backend '{backend}'. Available: {list(BACKENDS.keys())}"
            )
        self.backend: str = backend
        self.backend_cfg: dict[str, Any] = BACKENDS[backend]
        self.verbose: bool = verbose
        self.storage_path: str = storage_path or f"/tmp/su-memory-bench/locomo-{backend}"
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        # 缓存数据集，避免重复加载
        self._dataset_cache: Any = None

    # ---- 数据加载 --------------------------------------------------------

    def load_dataset(self, force_reload: bool = False) -> Any:
        """加载 ``snap-research/locomo`` 数据集。

        优先使用 HuggingFace ``datasets`` 库；若网络不可用，回退到本地缓存目录
        ``benchmarks/data/locomo/`` 下的 JSON 文件。
        """

        if self._dataset_cache is not None and not force_reload:
            return self._dataset_cache

        cache_dir = ensure_data_dir("locomo")
        hf_id = DATASETS["locomo"]["hf_id"]

        try:
            from datasets import load_dataset

            if self.verbose:
                print(f"[LoCoMo] Loading {hf_id} (cache_dir={cache_dir})...")
            ds = load_dataset(hf_id, cache_dir=cache_dir)
            self._dataset_cache = ds
            return ds
        except Exception as exc:
            if self.verbose:
                print(f"[LoCoMo] HF load failed ({exc}); falling back to local files.")
            local_data = self._load_local(cache_dir)
            if local_data is None:
                raise RuntimeError(
                    f"Failed to load LoCoMo dataset from HF '{hf_id}' and no local "
                    f"cache found in {cache_dir}."
                ) from exc
            self._dataset_cache = local_data
            return local_data

    @staticmethod
    def _load_local(cache_dir: str) -> list[dict] | None:
        """从本地 JSON 缓存读取（支持 ``locomo10.json`` 格式）。"""

        path = Path(cache_dir)
        for fname in DATASETS["locomo"]["files"]:
            f = path / fname
            if f.exists():
                with f.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                return data if isinstance(data, list) else [data]
        # 兜底：扫描目录任何 .json
        for f in path.glob("*.json"):
            with f.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, list) else [data]
        return None

    # ---- 数据规范化 ------------------------------------------------------

    @staticmethod
    def iter_conversations(dataset: Any) -> Iterable[dict[str, Any]]:
        """将异构格式（HF DatasetDict / list / dict）统一迭代为对话样本字典。"""

        if dataset is None:
            return
        # HuggingFace DatasetDict
        if hasattr(dataset, "keys") and hasattr(dataset, "__getitem__") and not isinstance(
            dataset, (list, dict)
        ):
            try:
                for split in dataset.keys():  # type: ignore[attr-defined]
                    for sample in dataset[split]:
                        yield sample
                return
            except Exception:
                pass
        if isinstance(dataset, list):
            for sample in dataset:
                yield sample
            return
        if isinstance(dataset, dict):
            yield dataset
            return
        # HuggingFace Dataset 单 split
        try:
            for sample in dataset:  # type: ignore[not-an-iterable]
                yield sample
        except Exception:
            return

    @staticmethod
    def normalize_conversation(sample: dict[str, Any]) -> list[dict[str, Any]]:
        """将一个 LoCoMo 样本拍平为统一的 ``turns`` 列表。

        支持两种常见结构：
        1. ``conversation = {"session_1": [{"speaker", "text"}, ...], ...}``
        2. ``conversation = [{"role"|"speaker", "content"|"text"}, ...]``

        Returns:
            ``[{"role", "content", "session_id", "turn_index", "timestamp"}, ...]``
        """

        conv = _safe_get(sample, "conversation", "dialog", "dialogue", "messages",
                         default=None)
        if conv is None:
            return []

        turns: list[dict[str, Any]] = []

        if isinstance(conv, dict):
            # 形如 {session_1: [...], session_2: [...], speaker_a/speaker_b: ...}
            session_keys = sorted(
                [k for k in conv.keys() if isinstance(k, str) and k.startswith("session")],
                key=lambda x: int("".join(c for c in x if c.isdigit()) or 0),
            )
            global_idx = 0
            for sid in session_keys:
                session_turns = conv[sid]
                if not isinstance(session_turns, list):
                    continue
                for local_idx, t in enumerate(session_turns):
                    role = _safe_get(t, "speaker", "role", "name", default="user")
                    content = _coerce_text(
                        _safe_get(t, "text", "content", "utterance", default=t)
                    )
                    timestamp = _safe_get(
                        t, "timestamp", "time", "date", "dia_id", default=None
                    )
                    turns.append({
                        "role": str(role),
                        "content": content,
                        "session_id": sid,
                        "turn_index": global_idx,
                        "session_turn_index": local_idx,
                        "timestamp": timestamp,
                    })
                    global_idx += 1
            if turns:
                return turns

        if isinstance(conv, list):
            for idx, t in enumerate(conv):
                role = _safe_get(t, "role", "speaker", "name", default="user")
                content = _coerce_text(_safe_get(t, "content", "text", default=t))
                timestamp = _safe_get(t, "timestamp", "time", "date", default=None)
                session_id = _safe_get(t, "session_id", "session", default="session_1")
                turns.append({
                    "role": str(role),
                    "content": content,
                    "session_id": str(session_id),
                    "turn_index": idx,
                    "session_turn_index": idx,
                    "timestamp": timestamp,
                })
        return turns

    # ---- Memory 客户端 --------------------------------------------------

    def _new_memory(self):
        """根据 backend 创建一个全新的 SuMemoryLitePro 实例（清空旧目录）。"""

        try:
            from su_memory.sdk.lite_pro import SuMemoryLitePro
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "Cannot import su_memory.sdk.lite_pro. Run `pip install -e .` "
                "in the su-memory-sdk root."
            ) from exc

        # 清理旧持久化数据，确保隔离
        if Path(self.storage_path).exists():
            shutil.rmtree(self.storage_path, ignore_errors=True)
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)

        backend_type = self.backend_cfg.get("type", "ollama")
        if backend_type == "ollama":
            embedding_backend = "ollama"
        elif backend_type == "sentence-transformers":
            embedding_backend = "sentence_transformers"
            os.environ.setdefault("SU_MEMORY_EMBEDDING_MODEL", self.backend_cfg["model"])
        else:
            embedding_backend = "auto"

        try:
            return SuMemoryLitePro(
                storage_path=self.storage_path,
                enable_vector=True,
                embedding_backend=embedding_backend,
            )
        except TypeError:
            # 兼容旧签名
            return SuMemoryLitePro(
                storage_path=self.storage_path,
                enable_vector=True,
            )

    # ---- 注入对话 --------------------------------------------------------

    def ingest_conversation(
        self,
        conversation: list[dict[str, Any]],
        memory: Any,
        chunk_size: int = 200,
        sample_id: str | None = None,
    ) -> dict[str, Any]:
        """将超长对话注入 su-memory（带时间戳元数据）。

        Args:
            conversation: 经 :py:meth:`normalize_conversation` 拍平的轮次列表。
            memory: ``SuMemoryLitePro`` 实例。
            chunk_size: chunk 字符数（默认 200）。
            sample_id: 对话样本 ID，写入元数据用于检索去重。

        Returns:
            写入统计：``{"turns", "chunks", "total_add_ms"}``
        """

        chunk_count = 0
        total_add_ms = 0.0
        base_ts = int(time.time())

        for turn in conversation:
            content_full = f"[{turn['role']}] {turn['content']}"
            chunks = _chunk_text(content_full, chunk_size=chunk_size)
            ts = turn.get("timestamp")
            if ts is None:
                # 估算时间戳：每轮间隔 60s
                ts = base_ts + turn["turn_index"] * 60

            for chunk_idx, chunk in enumerate(chunks):
                metadata = {
                    "turn_index": turn["turn_index"],
                    "session_id": turn["session_id"],
                    "session_turn_index": turn["session_turn_index"],
                    "timestamp": ts,
                    "role": turn["role"],
                    "chunk_index": chunk_idx,
                    "sample_id": sample_id,
                }
                t0 = time.perf_counter()
                try:
                    memory.add(content=chunk, metadata=metadata)
                except TypeError:
                    memory.add(chunk, metadata=metadata)
                except Exception as exc:
                    if self.verbose:
                        print(f"  [ingest] add failed: {exc}")
                    continue
                total_add_ms += (time.perf_counter() - t0) * 1000.0
                chunk_count += 1

        return {
            "turns": len(conversation),
            "chunks": chunk_count,
            "total_add_ms": total_add_ms,
        }

    # ---- 查询辅助 --------------------------------------------------------

    @staticmethod
    def _result_content(item: Any) -> str:
        """从 query 返回项中提取文本内容。"""

        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            return str(item.get("content") or item.get("text") or "")
        return str(getattr(item, "content", "") or getattr(item, "text", "") or "")

    def _query(self, memory: Any, q: str, top_k: int = 5) -> list[Any]:
        """统一的查询封装。"""

        try:
            return list(memory.query(q, top_k=top_k))
        except TypeError:
            return list(memory.query(q))
        except Exception as exc:
            if self.verbose:
                print(f"  [query] failed: {exc}")
            return []

    @staticmethod
    def _extract_qa_pairs(sample: dict[str, Any]) -> list[dict[str, Any]]:
        """从样本提取 QA 列表，兼容多种字段命名。"""

        qa = _safe_get(sample, "qa", "qas", "questions", "question_answers", default=[])
        if isinstance(qa, dict):
            qa = [qa]
        if not isinstance(qa, list):
            return []
        out: list[dict[str, Any]] = []
        for item in qa:
            question = _safe_get(item, "question", "q", "query")
            answer = _safe_get(item, "answer", "a", "gold_answer", "ground_truth")
            if not question:
                continue
            out.append({
                "question": _coerce_text(question),
                "answer": _coerce_text(answer),
                "category": _safe_get(item, "category", "type", default=""),
                "evidence": _safe_get(item, "evidence", "support", default=None),
            })
        return out

    # ---- 任务 1: QA -----------------------------------------------------

    def run_qa_task(
        self,
        dataset: Any,
        max_conversations: int | None = None,
    ) -> BenchmarkResult:
        """QA 问答：对每个对话注入后回答其 QA 列表，使用 EM 与 F1。"""

        result = BenchmarkResult(benchmark_name="locomo-qa", backend=self.backend)
        em_total = 0
        f1_sum = 0.0
        n_q = 0
        recall_at_1 = recall_at_3 = recall_at_5 = 0
        query_times: list[float] = []
        add_times: list[float] = []
        total_chunks = 0
        category_scores: dict[str, list[float]] = {}

        for conv_idx, sample in enumerate(self.iter_conversations(dataset)):
            if max_conversations and conv_idx >= max_conversations:
                break
            sample_id = str(_safe_get(sample, "sample_id", "id", default=conv_idx))
            turns = self.normalize_conversation(sample)
            qa_pairs = self._extract_qa_pairs(sample)
            if not turns or not qa_pairs:
                if self.verbose:
                    print(f"  [qa] skip sample {sample_id}: turns={len(turns)} "
                          f"qa={len(qa_pairs)}")
                continue

            memory = self._new_memory()
            stats = self.ingest_conversation(turns, memory, sample_id=sample_id)
            total_chunks += stats["chunks"]
            if stats["chunks"]:
                add_times.append(stats["total_add_ms"] / stats["chunks"])

            if self.verbose:
                print(f"[qa] sample {sample_id}: turns={stats['turns']} "
                      f"chunks={stats['chunks']} qa={len(qa_pairs)}")

            for qa in qa_pairs:
                t0 = time.perf_counter()
                hits = self._query(memory, qa["question"], top_k=5)
                query_times.append((time.perf_counter() - t0) * 1000.0)

                contents = [self._result_content(h) for h in hits]
                joined = " ".join(contents)
                gold = qa["answer"]

                em = 1 if exact_match(joined, gold) or any(
                    gold and gold.lower() in c.lower() for c in contents
                ) else 0
                f1 = compute_f1(joined, gold)

                em_total += em
                f1_sum += f1
                n_q += 1

                # Recall@k —— 子串匹配
                if gold:
                    g = gold.lower()
                    for k_idx, content in enumerate(contents[:5]):
                        if g in content.lower():
                            if k_idx < 5:
                                recall_at_5 += 1
                            if k_idx < 3:
                                recall_at_3 += 1
                            if k_idx < 1:
                                recall_at_1 += 1
                            break

                cat = qa.get("category") or "uncategorized"
                category_scores.setdefault(cat, []).append(f1)

            try:
                memory.clear()
            except Exception:
                pass

        result.total_questions = n_q
        result.correct = em_total
        result.accuracy = em_total / n_q if n_q else 0.0
        result.f1_score = f1_sum / n_q if n_q else 0.0
        result.recall_at_1 = recall_at_1 / n_q if n_q else 0.0
        result.recall_at_3 = recall_at_3 / n_q if n_q else 0.0
        result.recall_at_5 = recall_at_5 / n_q if n_q else 0.0
        result.avg_query_time_ms = sum(query_times) / len(query_times) if query_times else 0.0
        result.avg_add_time_ms = sum(add_times) / len(add_times) if add_times else 0.0
        result.dimension_scores = {
            cat: sum(scores) / len(scores) for cat, scores in category_scores.items()
            if scores
        }
        result.metadata = {
            "task": "qa",
            "total_chunks": total_chunks,
            "conversations_evaluated": min(
                max_conversations or 10**9, conv_idx + 1 if n_q else 0
            ),
        }
        return result

    # ---- 任务 2: 事件总结 ------------------------------------------------

    def run_event_summarization(
        self,
        dataset: Any,
        max_conversations: int | None = None,
    ) -> BenchmarkResult:
        """事件总结：拼接召回的 top-k 对话片段作为预测，使用 ROUGE-L 评分。"""

        result = BenchmarkResult(
            benchmark_name="locomo-event-summary", backend=self.backend
        )
        rouge_sum = 0.0
        n = 0
        query_times: list[float] = []

        for conv_idx, sample in enumerate(self.iter_conversations(dataset)):
            if max_conversations and conv_idx >= max_conversations:
                break
            turns = self.normalize_conversation(sample)
            events = _safe_get(
                sample, "event_summary", "events", "summary", "session_summaries",
                default=None,
            )
            if not turns or not events:
                continue

            # 规范化 events 为 [(query, gold_summary), ...]
            event_pairs: list[tuple[str, str]] = []
            if isinstance(events, dict):
                for key, val in events.items():
                    event_pairs.append((str(key), _coerce_text(val)))
            elif isinstance(events, list):
                for ev in events:
                    q = _coerce_text(
                        _safe_get(ev, "topic", "title", "question", "session", default="")
                    )
                    a = _coerce_text(
                        _safe_get(ev, "summary", "answer", "text", "events", default=ev)
                    )
                    event_pairs.append((q or a[:50], a))
            else:
                continue

            memory = self._new_memory()
            sample_id = str(_safe_get(sample, "sample_id", "id", default=conv_idx))
            self.ingest_conversation(turns, memory, sample_id=sample_id)

            for query, gold in event_pairs:
                if not gold:
                    continue
                t0 = time.perf_counter()
                hits = self._query(memory, query or gold[:50], top_k=10)
                query_times.append((time.perf_counter() - t0) * 1000.0)
                pred = " ".join(self._result_content(h) for h in hits)
                rouge_sum += compute_rouge_l(pred, gold)
                n += 1

            try:
                memory.clear()
            except Exception:
                pass

        result.total_questions = n
        result.rouge_l = rouge_sum / n if n else 0.0
        result.f1_score = result.rouge_l
        result.avg_query_time_ms = sum(query_times) / len(query_times) if query_times else 0.0
        result.metadata = {"task": "event_summary"}
        return result

    # ---- 任务 3: 跨会话推理 ----------------------------------------------

    def run_cross_session(
        self,
        dataset: Any,
        max_conversations: int | None = None,
    ) -> BenchmarkResult:
        """跨会话推理：聚焦 ``category in {multi-hop, multi_session, cross_session}`` 的 QA。"""

        result = BenchmarkResult(
            benchmark_name="locomo-cross-session", backend=self.backend
        )
        em_total = 0
        n = 0
        query_times: list[float] = []
        f1_sum = 0.0

        target_cats = {
            "multi-hop", "multi_hop", "multihop",
            "cross_session", "cross-session", "multi_session",
            "multi-session",
        }

        for conv_idx, sample in enumerate(self.iter_conversations(dataset)):
            if max_conversations and conv_idx >= max_conversations:
                break
            turns = self.normalize_conversation(sample)
            qa_pairs = self._extract_qa_pairs(sample)
            cross_qa = [
                qa for qa in qa_pairs
                if str(qa.get("category", "")).lower().replace(" ", "_") in target_cats
            ]
            # 若数据集没有显式分类，则启发式选取 evidence 跨多个 session 的 QA
            if not cross_qa:
                cross_qa = [
                    qa for qa in qa_pairs
                    if isinstance(qa.get("evidence"), list) and len(qa["evidence"]) >= 2
                ]
            if not turns or not cross_qa:
                continue

            memory = self._new_memory()
            sample_id = str(_safe_get(sample, "sample_id", "id", default=conv_idx))
            self.ingest_conversation(turns, memory, sample_id=sample_id)

            for qa in cross_qa:
                t0 = time.perf_counter()
                hits = self._query(memory, qa["question"], top_k=10)
                query_times.append((time.perf_counter() - t0) * 1000.0)
                contents = [self._result_content(h) for h in hits]
                joined = " ".join(contents)
                gold = qa["answer"]
                if gold and (
                    any(gold.lower() in c.lower() for c in contents)
                    or semantic_match(joined, gold, threshold=0.6)
                ):
                    em_total += 1
                f1_sum += compute_f1(joined, gold)
                n += 1

            try:
                memory.clear()
            except Exception:
                pass

        result.total_questions = n
        result.correct = em_total
        result.accuracy = em_total / n if n else 0.0
        result.f1_score = f1_sum / n if n else 0.0
        result.avg_query_time_ms = sum(query_times) / len(query_times) if query_times else 0.0
        result.metadata = {"task": "cross_session"}
        return result

    # ---- 任务 4: 时间因果 ------------------------------------------------

    def run_temporal_causal(
        self,
        dataset: Any,
        max_conversations: int | None = None,
    ) -> BenchmarkResult:
        """时间因果推理：聚焦 ``category in {temporal, causal, temporal_reasoning}``。"""

        result = BenchmarkResult(
            benchmark_name="locomo-temporal-causal", backend=self.backend
        )
        em_total = 0
        f1_sum = 0.0
        n = 0
        query_times: list[float] = []

        target_cats = {
            "temporal", "temporal_reasoning", "time", "causal",
            "causality", "temporal-causal", "when",
        }

        for conv_idx, sample in enumerate(self.iter_conversations(dataset)):
            if max_conversations and conv_idx >= max_conversations:
                break
            turns = self.normalize_conversation(sample)
            qa_pairs = self._extract_qa_pairs(sample)
            target_qa = [
                qa for qa in qa_pairs
                if any(t in str(qa.get("category", "")).lower() for t in target_cats)
            ]
            if not turns or not target_qa:
                continue

            memory = self._new_memory()
            sample_id = str(_safe_get(sample, "sample_id", "id", default=conv_idx))
            self.ingest_conversation(turns, memory, sample_id=sample_id)

            for qa in target_qa:
                t0 = time.perf_counter()
                hits = self._query(memory, qa["question"], top_k=10)
                query_times.append((time.perf_counter() - t0) * 1000.0)
                contents = [self._result_content(h) for h in hits]
                joined = " ".join(contents)
                gold = qa["answer"]
                if gold and any(gold.lower() in c.lower() for c in contents):
                    em_total += 1
                f1_sum += compute_f1(joined, gold)
                n += 1

            try:
                memory.clear()
            except Exception:
                pass

        result.total_questions = n
        result.correct = em_total
        result.accuracy = em_total / n if n else 0.0
        result.f1_score = f1_sum / n if n else 0.0
        result.avg_query_time_ms = sum(query_times) / len(query_times) if query_times else 0.0
        result.metadata = {"task": "temporal_causal"}
        return result

    # ---- 总入口 ----------------------------------------------------------

    def run(
        self,
        tasks: list[str] | None = None,
        max_conversations: int | None = None,
    ) -> dict[str, BenchmarkResult]:
        """运行全部或指定任务。

        Args:
            tasks: 任务列表，可选 ``qa`` / ``event_summary`` / ``cross_session``
                / ``temporal_causal`` / ``all``，``None`` 等价于 ``all``。
            max_conversations: 限制评测的对话数（调试用）。

        Returns:
            ``{task_name: BenchmarkResult}``。
        """

        tasks = tasks or list(self.DEFAULT_TASKS)
        if "all" in tasks:
            tasks = list(self.DEFAULT_TASKS)

        unknown = [t for t in tasks if t not in self.DEFAULT_TASKS]
        if unknown:
            raise ValueError(
                f"Unknown tasks {unknown}. Valid: {self.DEFAULT_TASKS}"
            )

        dataset = self.load_dataset()
        results: dict[str, BenchmarkResult] = {}

        dispatch = {
            "qa": self.run_qa_task,
            "event_summary": self.run_event_summarization,
            "cross_session": self.run_cross_session,
            "temporal_causal": self.run_temporal_causal,
        }
        for task in tasks:
            if self.verbose:
                print(f"\n========== Running task: {task} ==========")
            t0 = time.time()
            results[task] = dispatch[task](dataset, max_conversations=max_conversations)
            if self.verbose:
                r = results[task]
                print(f"[{task}] n={r.total_questions} "
                      f"acc={r.accuracy:.4f} f1={r.f1_score:.4f} "
                      f"rouge_l={r.rouge_l:.4f} "
                      f"({time.time() - t0:.1f}s)")
        return results


# ---------------------------------------------------------------------------
# 报告 & 持久化
# ---------------------------------------------------------------------------

def format_report(all_results: dict[str, dict[str, BenchmarkResult]]) -> str:
    """生成多 backend × 多 task 的对比报告。"""

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("  su-memory — LoCoMo Benchmark Report")
    lines.append("=" * 78)

    for backend, task_results in all_results.items():
        cfg = BACKENDS.get(backend, {})
        lines.append("")
        lines.append(f"  Backend: {backend}  ({cfg.get('name', backend)})")
        lines.append("  " + "-" * 60)
        lines.append(f"  {'Task':<22}{'N':>6}{'Acc':>10}{'F1':>10}{'ROUGE-L':>12}"
                     f"{'AvgQ (ms)':>12}")
        for task, r in task_results.items():
            lines.append(
                f"  {task:<22}{r.total_questions:>6}{r.accuracy:>10.2%}"
                f"{r.f1_score:>10.2%}{r.rouge_l:>12.2%}"
                f"{r.avg_query_time_ms:>12.1f}"
            )

    # 竞品对比（locomo_f1）
    lines.append("")
    lines.append("  --- Competitor Comparison (locomo_f1) ---")
    lines.append(f"  {'System':<20}{'F1':>10}")
    lines.append("  " + "-" * 32)
    for name, scores in COMPETITOR_SCORES.items():
        v = scores.get("locomo_f1")
        lines.append(f"  {name:<20}{('  N/A' if v is None else f'{v:>10.2%}')}")

    lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)


def save_results(
    all_results: dict[str, dict[str, BenchmarkResult]], output_path: str
) -> str:
    """将所有结果保存为单个 JSON。"""

    payload: dict[str, Any] = {}
    for backend, task_results in all_results.items():
        payload[backend] = {
            task: r.to_dict() for task, r in task_results.items()
        }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="su-memory LoCoMo Benchmark Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backend", default="ollama",
        choices=list(BACKENDS.keys()) + ["both"],
        help="Embedding backend; 'both' runs ollama and sbert sequentially.",
    )
    parser.add_argument(
        "--tasks", default="all",
        help="Comma-separated tasks: qa,event_summary,cross_session,temporal_causal,all",
    )
    parser.add_argument(
        "--max-conversations", type=int, default=None,
        help="Limit number of conversations per task (debug aid).",
    )
    parser.add_argument(
        "--storage-path", default=None,
        help="Storage directory for su-memory persistence (auto if omitted).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (omitted = no file written).",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if args.backend == "both":
        backends = ["ollama", "sbert"]
    else:
        backends = [args.backend]

    all_results: dict[str, dict[str, BenchmarkResult]] = {}
    for backend in backends:
        runner = LoCoMoRunner(
            backend=backend,
            storage_path=args.storage_path,
            verbose=args.verbose,
        )
        try:
            all_results[backend] = runner.run(
                tasks=tasks, max_conversations=args.max_conversations
            )
        except Exception as exc:
            print(f"[locomo] backend={backend} failed: {exc}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            all_results[backend] = {}

    print(format_report(all_results))

    if args.output:
        path = save_results(all_results, args.output)
        print(f"\nResults saved to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
