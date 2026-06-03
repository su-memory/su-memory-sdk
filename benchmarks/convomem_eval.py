#!/usr/bin/env python3
"""
ConvoMem Benchmark Runner for su-memory
=======================================
Evaluates su-memory against the Salesforce ConvoMem benchmark:

- 75,336 QA pairs across 6 evidence categories
- Conversation-level memory retention from 2 to 300 dialogues
- Single-message vs multi-message evidence support
- Knowledge-update / abstention / preference / multi-hop reasoning

Reference:
    https://huggingface.co/datasets/Salesforce/ConvoMem

Usage:
    python benchmarks/convomem_eval.py --backend ollama --max-samples 100
    python benchmarks/convomem_eval.py --backend both --categories all --verbose
    python benchmarks/convomem_eval.py --backend ollama \\
        --categories user_facts,changing_facts --context-sizes 2,10,50
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Local imports — sys.path bootstrap so the script works both as a module and
# when launched directly via ``python benchmarks/convomem_eval.py``.
# ---------------------------------------------------------------------------

_BENCH_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BENCH_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.config import (  # noqa: E402
    BACKENDS,
    BenchmarkResult,
    COMPETITOR_SCORES,
    DATASETS,
    compute_recall_at_k,
    ensure_data_dir,
    load_hf_dataset,
    semantic_match,
)

try:  # pragma: no cover - 依赖检查
    from su_memory.sdk.lite_pro import SuMemoryLitePro
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "无法导入 SuMemoryLitePro，请确认 src/su_memory 已安装或 PYTHONPATH 正确。"
    ) from exc


# ---------------------------------------------------------------------------
# 1. 常量定义
# ---------------------------------------------------------------------------

EVIDENCE_CATEGORIES: tuple[str, ...] = (
    "user_facts",
    "assistant_facts",
    "changing_facts",
    "abstention",
    "preferences",
    "implicit_connections",
)

DEFAULT_CONTEXT_SIZES: tuple[int, ...] = (2, 4, 6, 10, 20, 30, 50, 70, 100, 150, 200, 300)

# 表示"未提及/不知道"的常见关键词，用于 abstention 类型评估
ABSTENTION_KEYWORDS: tuple[str, ...] = (
    "未提及",
    "未提到",
    "没有提及",
    "不知道",
    "不清楚",
    "无相关信息",
    "无法确定",
    "not mentioned",
    "not specified",
    "no information",
    "unknown",
    "do not know",
    "cannot determine",
    "n/a",
)


# ---------------------------------------------------------------------------
# 2. ConvoMemRunner
# ---------------------------------------------------------------------------

@dataclass
class _CategoryStats:
    """单个证据类型的累计统计，用于聚合到 BenchmarkResult。"""

    total: int = 0
    correct: int = 0
    recall_at_1: int = 0
    recall_at_3: int = 0
    recall_at_5: int = 0
    single_message_total: int = 0
    single_message_correct: int = 0
    multi_message_total: int = 0
    multi_message_correct: int = 0
    query_time_ms: float = 0.0
    add_time_ms: float = 0.0
    add_count: int = 0
    by_size: dict[int, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


class ConvoMemRunner:
    """ConvoMem 对话记忆评测器。

    Args:
        backend: 嵌入后端（``ollama`` / ``sbert`` / ``sbert-mpnet``）。
        storage_path: 持久化目录；为 ``None`` 时使用临时目录。
        verbose: 是否打印调试信息。
    """

    benchmark_name: str = "convomem"

    def __init__(
        self,
        backend: str = "ollama",
        storage_path: str | None = None,
        verbose: bool = False,
    ) -> None:
        if backend not in BACKENDS:
            raise ValueError(
                f"未知 backend '{backend}'，可选: {list(BACKENDS.keys())}"
            )
        self.backend = backend
        self.verbose = verbose
        self.storage_root = (
            Path(storage_path)
            if storage_path
            else Path(tempfile.gettempdir()) / "su-memory-bench" / "convomem"
        )
        self.storage_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------
    # 目录命名 → EVIDENCE_CATEGORIES 映射
    _CONVOMEM_DIR_TO_CATEGORY: dict[str, str] = {
        "abstention_evidence": "abstention",
        "assistant_facts_evidence": "assistant_facts",
        "changing_evidence": "changing_facts",
        "implicit_connection_evidence": "implicit_connections",
        "preference_evidence": "preferences",
        "user_evidence": "user_facts",
    }

    def load_dataset(self, split: str | None = None) -> Any:
        """加载 ``Salesforce/ConvoMem`` 数据集。

        由于 ConvoMem 数据集的 JSON 文件结构异构（不同文件列不一致），
        ``datasets.load_dataset`` 总会因 schema mismatch 失败。因此本方法
        **优先**从 HuggingFace Hub 本地缓存快照直接解析 JSON 文件，
        仅当快照不存在时才尝试通过 ``load_dataset`` 触发下载。

        Args:
            split: 可选切分（``train`` / ``test``）。``None`` 返回 DatasetDict。

        Returns:
            ``datasets.Dataset`` 或 ``datasets.DatasetDict`` 对象，或本地回退
            加载得到的 ``list[dict]``。
        """

        cfg = DATASETS["convomem"]
        cache_dir = ensure_data_dir(self.benchmark_name)
        if self.verbose:
            print(f"[ConvoMem] 加载数据集 {cfg['hf_id']} → {cache_dir}")

        # 优先尝试本地快照（HF Hub 缓存或 benchmarks/data/convomem）
        local = self._load_local_snapshot(cache_dir)
        if local:
            if self.verbose:
                print(f"[ConvoMem] 已从本地快照加载 {len(local)} 个样本")
            return local

        # 本地快照为空 → 尝试通过 HF datasets 库下载（会触发缓存）
        if self.verbose:
            print("[ConvoMem] 本地快照为空，尝试通过 HF datasets 下载…")
        try:
            ds = load_hf_dataset(cfg["hf_id"], split=split, cache_dir=cache_dir)
            return ds
        except Exception as exc:
            if self.verbose:
                print(f"[ConvoMem] HF load_dataset 失败 ({exc})")
            # 下载后再次尝试本地快照（下载可能已缓存原始文件）
            local = self._load_local_snapshot(cache_dir)
            if local:
                if self.verbose:
                    print(f"[ConvoMem] 下载后从本地快照加载 {len(local)} 个样本")
                return local
            raise RuntimeError(
                f"Failed to load ConvoMem dataset: {exc}. "
                f"Ensure the dataset is cached locally or accessible from HuggingFace."
            ) from exc

    def _load_local_snapshot(self, cache_dir: str) -> list[dict[str, Any]]:
        """扫描本地 HF 缓存或 ``benchmarks/data/convomem`` 下的原始 JSON 文件。

        ConvoMem 在 HF 上的 layout 为目录树而非 parquet/jsonl，``datasets.load_dataset``
        无法直接加载。本方法将原始 JSON 展开为评测器期望的样本字典：

        ``{question, answer, category, conversations[{messages:[{role,content}]}],
           evidence_count, multi_message, context_size}``
        """

        candidate_roots: list[Path] = []
        # 1. 本地 benchmarks/data/convomem 目录
        candidate_roots.append(Path(cache_dir))
        # 2. HF hub 默认缓存目录
        hub_root = Path(
            os.environ.get("HF_HOME")
            or (Path.home() / ".cache" / "huggingface")
        )
        snap_dir = hub_root / "hub" / "datasets--Salesforce--ConvoMem" / "snapshots"
        if snap_dir.exists():
            for snap in sorted(snap_dir.iterdir()):
                if snap.is_dir():
                    candidate_roots.append(snap)

        samples: list[dict[str, Any]] = []
        for root in candidate_roots:
            mixed_dir = root / "core_benchmark" / "pre_mixed_testcases"
            if not mixed_dir.exists():
                continue
            for cat_dir in sorted(mixed_dir.iterdir()):
                if not cat_dir.is_dir():
                    continue
                category = self._CONVOMEM_DIR_TO_CATEGORY.get(cat_dir.name)
                if category is None:
                    continue
                for ev_dir in sorted(cat_dir.iterdir()):
                    if not ev_dir.is_dir():
                        continue
                    # 目录形如 ``2_evidence`` / ``6_evidence``
                    try:
                        ev_count = int(ev_dir.name.split("_")[0])
                    except Exception:
                        ev_count = 1
                    for jf in sorted(ev_dir.glob("batched_*.json")):
                        try:
                            with jf.open("r", encoding="utf-8") as fh:
                                payload = json.load(fh)
                        except Exception:
                            continue
                        if not isinstance(payload, list):
                            continue
                        for testcase in payload:
                            if not isinstance(testcase, dict):
                                continue
                            ctx_size = testcase.get("contextSize") or len(
                                testcase.get("conversations") or []
                            )
                            for ev in testcase.get("evidenceItems") or []:
                                if not isinstance(ev, dict):
                                    continue
                                question = str(ev.get("question") or "").strip()
                                answer = str(ev.get("answer") or "").strip()
                                if not question or not answer:
                                    continue
                                convs = ev.get("conversations") or testcase.get(
                                    "conversations"
                                ) or []
                                norm_convs: list[dict[str, Any]] = []
                                for cv in convs:
                                    if not isinstance(cv, dict):
                                        continue
                                    raw_msgs = cv.get("messages") or []
                                    norm_msgs = []
                                    for m in raw_msgs:
                                        if not isinstance(m, dict):
                                            continue
                                        role = str(
                                            m.get("role")
                                            or m.get("speaker")
                                            or "user"
                                        )
                                        content = str(
                                            m.get("content")
                                            or m.get("text")
                                            or ""
                                        )
                                        if content:
                                            norm_msgs.append(
                                                {"role": role, "content": content}
                                            )
                                    if norm_msgs:
                                        norm_convs.append(
                                            {
                                                "id": str(cv.get("id", "")),
                                                "messages": norm_msgs,
                                            }
                                        )
                                samples.append(
                                    {
                                        "question": question,
                                        "answer": answer,
                                        "category": category,
                                        "conversations": norm_convs,
                                        "evidence_count": ev_count,
                                        "multi_message": ev_count > 1,
                                        "context_size": int(ctx_size or len(norm_convs) or 1),
                                    }
                                )
            if samples:
                break  # 已从此 root 加载到样本，无需再扫描后续 root
        return samples

    # ------------------------------------------------------------------
    # 数据样本归一化
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
        """从样本中提取对话消息列表，兼容多种字段命名。

        优先级：
            1. ``conversations: [{messages: [...]}, ...]``
            2. ``conversation: [{role, content}, ...]``
            3. ``history`` / ``dialogue`` / ``messages``
        """

        msgs: list[dict[str, str]] = []
        convs = sample.get("conversations") or sample.get("conversation")
        if isinstance(convs, list):
            for conv in convs:
                if isinstance(conv, dict) and isinstance(conv.get("messages"), list):
                    for m in conv["messages"]:
                        if isinstance(m, dict) and "content" in m:
                            msgs.append(
                                {
                                    "role": str(m.get("role", "user")),
                                    "content": str(m.get("content", "")),
                                    "conversation_id": str(conv.get("id", "")),
                                }
                            )
                elif isinstance(conv, dict) and "content" in conv:
                    msgs.append(
                        {
                            "role": str(conv.get("role", "user")),
                            "content": str(conv.get("content", "")),
                            "conversation_id": "",
                        }
                    )
            if msgs:
                return msgs

        for key in ("history", "dialogue", "messages"):
            seq = sample.get(key)
            if isinstance(seq, list):
                for m in seq:
                    if isinstance(m, dict) and "content" in m:
                        msgs.append(
                            {
                                "role": str(m.get("role", "user")),
                                "content": str(m.get("content", "")),
                                "conversation_id": str(m.get("conversation_id", "")),
                            }
                        )
                    elif isinstance(m, str):
                        msgs.append({"role": "user", "content": m, "conversation_id": ""})
                if msgs:
                    return msgs

        # 兜底：用 context 字段拆分
        ctx = sample.get("context")
        if isinstance(ctx, str) and ctx:
            for line in ctx.splitlines():
                line = line.strip()
                if line:
                    msgs.append({"role": "user", "content": line, "conversation_id": ""})
        return msgs

    @staticmethod
    def _extract_answer(sample: dict[str, Any]) -> str:
        """提取标准答案字段。"""

        for key in ("answer", "gold_answer", "ground_truth", "label", "target"):
            v = sample.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, list) and v:
                return str(v[0]).strip()
        return ""

    @staticmethod
    def _extract_question(sample: dict[str, Any]) -> str:
        """提取查询问题字段。"""

        for key in ("question", "query", "prompt"):
            v = sample.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    @staticmethod
    def _extract_category(sample: dict[str, Any]) -> str:
        """提取证据类型字段，归一化为已知类型。"""

        for key in ("category", "evidence_type", "type", "label_type"):
            v = sample.get(key)
            if isinstance(v, str) and v.strip():
                norm = v.strip().lower().replace("-", "_").replace(" ", "_")
                if norm in EVIDENCE_CATEGORIES:
                    return norm
        return "user_facts"

    @staticmethod
    def _extract_context_size(sample: dict[str, Any], messages: list[dict]) -> int:
        """提取上下文规模（对话数量）。"""

        for key in ("context_size", "n_conversations", "num_conversations", "size"):
            v = sample.get(key)
            if isinstance(v, int) and v > 0:
                return int(v)
            if isinstance(v, str) and v.isdigit():
                return int(v)
        # fallback: 用唯一 conversation_id 数量；若无则用消息数 / 2
        ids = {m.get("conversation_id") for m in messages if m.get("conversation_id")}
        if ids:
            return len(ids)
        return max(1, len(messages) // 2)

    @staticmethod
    def _is_multi_message_evidence(sample: dict[str, Any]) -> bool:
        """判断是否多消息证据。"""

        v = sample.get("evidence_count") or sample.get("num_evidence_messages")
        if isinstance(v, int):
            return v > 1
        flag = sample.get("multi_message") or sample.get("is_multi_message")
        if isinstance(flag, bool):
            return flag
        evidences = sample.get("evidences") or sample.get("evidence")
        if isinstance(evidences, list):
            return len(evidences) > 1
        return False

    # ------------------------------------------------------------------
    # 摄入 & 查询
    # ------------------------------------------------------------------
    def _new_memory(self, label: str) -> tuple[SuMemoryLitePro, Path]:
        """为单个样本创建一个全新的 SuMemoryLitePro 实例。"""

        path = self.storage_root / f"{label}_{int(time.time() * 1000)}"
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        memory = SuMemoryLitePro(
            storage_path=str(path),
            embedding_backend=self.backend,
            enable_vector=True,
        )
        return memory, path

    def ingest_conversations(
        self,
        messages: list[dict[str, str]],
        memory: SuMemoryLitePro,
    ) -> tuple[float, int]:
        """将对话消息逐条写入 su-memory。

        Returns:
            ``(total_add_time_ms, add_count)`` 累计耗时和写入条数。
        """

        total_ms = 0.0
        count = 0
        for idx, msg in enumerate(messages):
            content = msg.get("content", "").strip()
            if not content:
                continue
            role = msg.get("role", "user")
            conv_id = msg.get("conversation_id", "")
            text = f"[{role}] {content}"
            t0 = time.time()
            try:
                memory.add(
                    content=text,
                    metadata={
                        "conversation_id": conv_id,
                        "role": role,
                        "turn_index": idx,
                    },
                )
            except Exception as exc:  # pragma: no cover
                if self.verbose:
                    print(f"  [warn] add failed: {exc}")
                continue
            total_ms += (time.time() - t0) * 1000.0
            count += 1
        return total_ms, count

    @staticmethod
    def _result_text(item: Any) -> str:
        """从查询结果里提取文本字段，兼容字典/对象。"""

        if isinstance(item, str):
            return item
        for attr in ("content", "text"):
            val = getattr(item, attr, None)
            if isinstance(val, str):
                return val
            if isinstance(item, dict):
                v = item.get(attr)
                if isinstance(v, str):
                    return v
        return str(item)

    @staticmethod
    def _is_answer_in_results(results: Iterable[Any], answer: str) -> bool:
        """判断答案是否出现在任意 top-K 结果文本中。"""

        if not answer:
            return False
        norm_ans = answer.strip().lower()
        for r in results:
            text = ConvoMemRunner._result_text(r).lower()
            if norm_ans in text:
                return True
            # fallback: 语义匹配（仅在直接命中失败时）
            try:
                if semantic_match(text, answer, threshold=0.78):
                    return True
            except Exception:
                pass
        return False

    @staticmethod
    def _is_abstention_correct(results: Iterable[Any], answer: str) -> bool:
        """abstention 类型的判定：

        - 若标准答案已是 "未提及" 等关键词，且检索结果未提供其他实体信息，视为正确；
        - 否则若答案中关键实体不出现在 top-K 结果文本，则视为正确（系统正确放弃）。
        """

        result_list = list(results)
        joined = " ".join(ConvoMemRunner._result_text(r).lower() for r in result_list)

        if not answer:
            return len(result_list) == 0 or all(
                not ConvoMemRunner._result_text(r).strip() for r in result_list
            )

        norm = answer.strip().lower()
        if any(kw in norm for kw in ABSTENTION_KEYWORDS):
            # gold 即"未提及"，只要结果里也没有具体实体信息（或为空）即算正确
            return not joined.strip() or any(kw in joined for kw in ABSTENTION_KEYWORDS)

        # gold 不是 abstention 关键词时：top-K 中未出现答案关键 token 即视为正确放弃
        return norm not in joined

    def _evaluate_sample(
        self,
        sample: dict[str, Any],
        stats: _CategoryStats,
    ) -> bool:
        """评估单个样本。返回是否答对（abstention 也按此累计）。"""

        messages = self._extract_messages(sample)
        question = self._extract_question(sample)
        answer = self._extract_answer(sample)
        category = self._extract_category(sample)
        ctx_size = self._extract_context_size(sample, messages)
        multi_msg = self._is_multi_message_evidence(sample)

        if not question or not messages:
            return False

        memory, path = self._new_memory(category)
        try:
            add_ms, add_n = self.ingest_conversations(messages, memory)
            stats.add_time_ms += add_ms
            stats.add_count += add_n

            t0 = time.time()
            try:
                results = memory.query(question, top_k=5)
            except Exception as exc:  # pragma: no cover
                if self.verbose:
                    print(f"  [warn] query failed: {exc}")
                results = []
            stats.query_time_ms += (time.time() - t0) * 1000.0

            results = list(results) if results else []

            if category == "abstention":
                correct = self._is_abstention_correct(results, answer)
                hit_at_1 = correct
                hit_at_3 = correct
                hit_at_5 = correct
            else:
                hit_at_1 = self._is_answer_in_results(results[:1], answer)
                hit_at_3 = self._is_answer_in_results(results[:3], answer)
                hit_at_5 = self._is_answer_in_results(results[:5], answer)
                correct = hit_at_5

            stats.total += 1
            stats.correct += int(correct)
            stats.recall_at_1 += int(hit_at_1)
            stats.recall_at_3 += int(hit_at_3)
            stats.recall_at_5 += int(hit_at_5)

            if multi_msg:
                stats.multi_message_total += 1
                stats.multi_message_correct += int(correct)
            else:
                stats.single_message_total += 1
                stats.single_message_correct += int(correct)

            bucket = stats.by_size[ctx_size]
            bucket[0] += 1
            bucket[1] += int(correct)
            return correct
        finally:
            try:
                memory.clear()
            except Exception:
                pass
            shutil.rmtree(path, ignore_errors=True)

    # ------------------------------------------------------------------
    # 类别评测
    # ------------------------------------------------------------------
    def _filter_samples(
        self,
        dataset: Any,
        category: str | None = None,
        context_sizes: list[int] | None = None,
        max_samples: int | None = None,
    ) -> list[dict[str, Any]]:
        """从 HuggingFace Dataset / DatasetDict 中筛选样本列表。"""

        # 展平 DatasetDict → 单一可迭代序列
        if hasattr(dataset, "keys") and not hasattr(dataset, "features"):
            iterable: Iterable[dict] = []
            for split_name in dataset.keys():
                iterable = list(iterable) + list(dataset[split_name])
        else:
            iterable = dataset

        sizes_set = set(context_sizes) if context_sizes else None
        out: list[dict[str, Any]] = []
        for sample in iterable:
            if not isinstance(sample, dict):
                try:
                    sample = dict(sample)
                except Exception:
                    continue
            if category is not None and self._extract_category(sample) != category:
                continue
            if sizes_set is not None:
                msgs = self._extract_messages(sample)
                if self._extract_context_size(sample, msgs) not in sizes_set:
                    continue
            out.append(sample)
            if max_samples is not None and len(out) >= max_samples:
                break
        return out

    def run_category(
        self,
        category: str,
        dataset: Any,
        context_sizes: list[int] | None = None,
        max_samples: int | None = None,
    ) -> BenchmarkResult:
        """运行单个证据类型的评测。"""

        if category not in EVIDENCE_CATEGORIES:
            raise ValueError(
                f"未知 category '{category}'，可选 {EVIDENCE_CATEGORIES}"
            )
        samples = self._filter_samples(
            dataset,
            category=category,
            context_sizes=context_sizes,
            max_samples=max_samples,
        )
        stats = _CategoryStats()
        if self.verbose:
            print(f"[ConvoMem][{category}] 样本数: {len(samples)}")
        for i, s in enumerate(samples, 1):
            self._evaluate_sample(s, stats)
            if self.verbose and i % 25 == 0:
                print(
                    f"  ... {i}/{len(samples)} 进行中 acc={stats.accuracy:.2%}"
                )

        result = BenchmarkResult(
            benchmark_name=self.benchmark_name,
            backend=self.backend,
            total_questions=stats.total,
            correct=stats.correct,
            accuracy=stats.accuracy,
            f1_score=stats.accuracy,
            recall_at_1=(stats.recall_at_1 / stats.total) if stats.total else 0.0,
            recall_at_3=(stats.recall_at_3 / stats.total) if stats.total else 0.0,
            recall_at_5=(stats.recall_at_5 / stats.total) if stats.total else 0.0,
            avg_query_time_ms=(stats.query_time_ms / stats.total) if stats.total else 0.0,
            avg_add_time_ms=(stats.add_time_ms / stats.add_count) if stats.add_count else 0.0,
            dimension_scores={
                "single_message_accuracy": (
                    stats.single_message_correct / stats.single_message_total
                    if stats.single_message_total
                    else 0.0
                ),
                "multi_message_accuracy": (
                    stats.multi_message_correct / stats.multi_message_total
                    if stats.multi_message_total
                    else 0.0
                ),
            },
            metadata={
                "category": category,
                "context_sizes": context_sizes,
                "max_samples": max_samples,
                "by_context_size": {
                    str(k): {
                        "total": v[0],
                        "correct": v[1],
                        "accuracy": (v[1] / v[0]) if v[0] else 0.0,
                    }
                    for k, v in sorted(stats.by_size.items())
                },
                "single_message_total": stats.single_message_total,
                "multi_message_total": stats.multi_message_total,
            },
        )
        return result

    # ------------------------------------------------------------------
    # 上下文规模分级
    # ------------------------------------------------------------------
    def run_scaling_test(
        self,
        dataset: Any,
        sizes: list[int] | None = None,
        max_per_size: int | None = 50,
    ) -> dict[str, float]:
        """测试不同对话数量下的准确率衰减。"""

        sizes = sizes or list(DEFAULT_CONTEXT_SIZES)
        results: dict[str, float] = {}
        for size in sizes:
            subset = self._filter_samples(
                dataset,
                category=None,
                context_sizes=[size],
                max_samples=max_per_size,
            )
            stats = _CategoryStats()
            for s in subset:
                self._evaluate_sample(s, stats)
            results[f"accuracy@{size}"] = stats.accuracy
            if self.verbose:
                print(
                    f"[ConvoMem][scaling] size={size:>3} n={stats.total:<4} "
                    f"acc={stats.accuracy:.2%}"
                )
        return results

    # ------------------------------------------------------------------
    # 总入口
    # ------------------------------------------------------------------
    def run(
        self,
        categories: list[str] | None = None,
        max_samples: int | None = None,
        context_sizes: list[int] | None = None,
        run_scaling: bool = False,
    ) -> dict[str, Any]:
        """运行全部或指定类型评测。

        Args:
            categories: 要评测的证据类型；``None`` 或 ``["all"]`` 表示全部。
            max_samples: 每类最大样本数。
            context_sizes: 限制评测的上下文规模。
            run_scaling: 是否额外执行规模分级测试。

        Returns:
            包含每类 BenchmarkResult、聚合结果和（可选）规模曲线的字典。
        """

        dataset = self.load_dataset()
        cats = (
            list(EVIDENCE_CATEGORIES)
            if not categories or categories == ["all"]
            else [c for c in categories if c in EVIDENCE_CATEGORIES]
        )
        per_category: dict[str, BenchmarkResult] = {}
        total_q = 0
        total_correct = 0
        weighted_r1 = weighted_r3 = weighted_r5 = 0.0
        total_query_ms = 0.0
        total_add_ms = 0.0
        total_add_n = 0
        sm_total = sm_correct = mm_total = mm_correct = 0

        for cat in cats:
            res = self.run_category(
                cat,
                dataset,
                context_sizes=context_sizes,
                max_samples=max_samples,
            )
            per_category[cat] = res
            total_q += res.total_questions
            total_correct += res.correct
            weighted_r1 += res.recall_at_1 * res.total_questions
            weighted_r3 += res.recall_at_3 * res.total_questions
            weighted_r5 += res.recall_at_5 * res.total_questions
            total_query_ms += res.avg_query_time_ms * res.total_questions
            sm_t = res.metadata.get("single_message_total", 0) or 0
            mm_t = res.metadata.get("multi_message_total", 0) or 0
            sm_total += sm_t
            mm_total += mm_t
            sm_correct += int(round(res.dimension_scores.get("single_message_accuracy", 0.0) * sm_t))
            mm_correct += int(round(res.dimension_scores.get("multi_message_accuracy", 0.0) * mm_t))
            total_add_ms += res.avg_add_time_ms  # 取均值再平均
            total_add_n += 1

        overall = BenchmarkResult(
            benchmark_name=self.benchmark_name,
            backend=self.backend,
            total_questions=total_q,
            correct=total_correct,
            accuracy=(total_correct / total_q) if total_q else 0.0,
            f1_score=(total_correct / total_q) if total_q else 0.0,
            recall_at_1=(weighted_r1 / total_q) if total_q else 0.0,
            recall_at_3=(weighted_r3 / total_q) if total_q else 0.0,
            recall_at_5=(weighted_r5 / total_q) if total_q else 0.0,
            avg_query_time_ms=(total_query_ms / total_q) if total_q else 0.0,
            avg_add_time_ms=(total_add_ms / total_add_n) if total_add_n else 0.0,
            dimension_scores={
                f"{cat}_accuracy": per_category[cat].accuracy for cat in cats
            } | {
                "single_message_accuracy": (sm_correct / sm_total) if sm_total else 0.0,
                "multi_message_accuracy": (mm_correct / mm_total) if mm_total else 0.0,
            },
            metadata={
                "categories": cats,
                "max_samples": max_samples,
                "context_sizes": context_sizes,
            },
        )

        out: dict[str, Any] = {
            "overall": overall,
            "per_category": per_category,
        }

        if run_scaling:
            out["scaling"] = self.run_scaling_test(
                dataset,
                sizes=context_sizes,
                max_per_size=max_samples,
            )

        return out

    # ------------------------------------------------------------------
    # 报告渲染
    # ------------------------------------------------------------------
    def format_report(self, summary: dict[str, Any]) -> str:
        """生成 ConvoMem 评测报告。"""

        overall: BenchmarkResult = summary["overall"]
        per_cat: dict[str, BenchmarkResult] = summary.get("per_category", {})
        scaling: dict[str, float] = summary.get("scaling", {})

        lines: list[str] = [
            "=" * 72,
            f"  su-memory — ConvoMem Benchmark Report  [backend={self.backend}]",
            "=" * 72,
            "",
            f"  Overall accuracy : {overall.accuracy:.2%}  ({overall.correct}/{overall.total_questions})",
            f"  Recall @ 1/3/5   : {overall.recall_at_1:.2%} / {overall.recall_at_3:.2%} / {overall.recall_at_5:.2%}",
            f"  Avg query latency: {overall.avg_query_time_ms:.1f} ms",
            f"  Avg add  latency : {overall.avg_add_time_ms:.1f} ms",
            "",
            "  --- Per-Category Accuracy ---",
        ]
        for cat, res in per_cat.items():
            lines.append(
                f"    {cat:<22} acc={res.accuracy:.2%}  n={res.total_questions:<5} "
                f"r@1={res.recall_at_1:.2%}  r@5={res.recall_at_5:.2%}"
            )

        sm = overall.dimension_scores.get("single_message_accuracy", 0.0)
        mm = overall.dimension_scores.get("multi_message_accuracy", 0.0)
        lines.extend(
            [
                "",
                "  --- Single vs Multi Message Evidence ---",
                f"    single_message       acc={sm:.2%}",
                f"    multi_message        acc={mm:.2%}",
            ]
        )

        if scaling:
            lines.extend(["", "  --- Context-Size Scaling ---"])
            for key, acc in scaling.items():
                lines.append(f"    {key:<14} acc={acc:.2%}")

        lines.extend(["", "  --- Competitor Comparison (convomem_accuracy) ---"])
        for name, scores in COMPETITOR_SCORES.items():
            v = scores.get("convomem_accuracy")
            tag = "N/A" if v is None else f"{v:.2%}"
            lines.append(f"    {name:<14} {tag}")
        lines.append(
            f"    {'su-memory':<14} {overall.accuracy:.2%}"
        )

        lines.append("=" * 72)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. CLI
# ---------------------------------------------------------------------------

def _parse_int_list(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [int(p) for p in parts] if parts else None


def _parse_str_list(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    raw = raw.strip()
    if raw.lower() == "all":
        return ["all"]
    return [p.strip() for p in raw.split(",") if p.strip()]


def _summary_to_serializable(summary: dict[str, Any]) -> dict[str, Any]:
    """将 ``run`` 返回的 summary 转为可 JSON 序列化的纯字典。"""

    out: dict[str, Any] = {}
    overall = summary.get("overall")
    if isinstance(overall, BenchmarkResult):
        out["overall"] = overall.to_dict()
    per_cat = summary.get("per_category", {})
    out["per_category"] = {
        cat: (res.to_dict() if isinstance(res, BenchmarkResult) else res)
        for cat, res in per_cat.items()
    }
    if "scaling" in summary:
        out["scaling"] = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in summary["scaling"].items()
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ConvoMem benchmark runner for su-memory",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "sbert", "sbert-mpnet", "both"],
        default="ollama",
        help="嵌入后端；both 同时跑 ollama 和 sbert",
    )
    parser.add_argument(
        "--categories",
        default="all",
        help="逗号分隔的证据类型或 all（默认 all）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="每个类别最多评测的样本数（调试用）",
    )
    parser.add_argument(
        "--context-sizes",
        default=None,
        help="逗号分隔的上下文规模列表，如 2,10,50",
    )
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="额外执行上下文规模分级测试",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="持久化目录（默认临时目录）",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="JSON 结果输出路径",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印详细进度",
    )
    args = parser.parse_args(argv)

    backends = ["ollama", "sbert"] if args.backend == "both" else [args.backend]
    categories = _parse_str_list(args.categories) or ["all"]
    context_sizes = _parse_int_list(args.context_sizes)

    aggregated: dict[str, Any] = {}
    for be in backends:
        runner = ConvoMemRunner(backend=be, storage_path=args.storage, verbose=args.verbose)
        summary = runner.run(
            categories=categories,
            max_samples=args.max_samples,
            context_sizes=context_sizes,
            run_scaling=args.scaling,
        )
        report = runner.format_report(summary)
        print(report)
        aggregated[be] = _summary_to_serializable(summary)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(aggregated, fh, indent=2, ensure_ascii=False)
        print(f"\nJSON saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
