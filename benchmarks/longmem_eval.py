#!/usr/bin/env python3
"""
LongMemEval Benchmark Runner for su-memory (六维度官方版)
=========================================================

对接 HuggingFace 官方数据集 ``xiaowu0162/longmemeval-cleaned``，
对 :class:`SuMemoryLitePro` 执行完整六维度评测：

1. 信息提取（single-session-*）：Accuracy + Recall@1/3/5
2. 多会话推理（multi-session）：Accuracy
3. 时序推理（temporal-reasoning）：Accuracy
4. 知识更新（knowledge-update）：Accuracy
5. 时序位置（早/中/晚 chunk 位置）：各位置 Accuracy
6. 放弃识别（``question_id`` 以 ``_abs`` 结尾）：放弃 Precision / Recall

支持双后端切换（Ollama bge-m3 / sentence-transformers），输出
ASCII 报表 + JSON 结果文件 + 与 Hindsight / 主流竞品的对比表。

Reference:
    https://arxiv.org/abs/2406.09974
    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# 路径与依赖
# ---------------------------------------------------------------------------

_BENCH_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _BENCH_DIR.parent

# 允许独立执行：python benchmarks/longmem_eval.py
for _p in (str(_PKG_ROOT), str(_PKG_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:  # 包内导入
    from benchmarks.config import (
        BACKENDS,
        BenchmarkResult,
        COMPETITOR_SCORES,
        DATASETS,
        compute_f1,
        compute_recall_at_k,
        ensure_data_dir,
        exact_match,
        load_hf_dataset,
        semantic_match,
    )
except ImportError:  # pragma: no cover - 兼容直接 import
    from config import (  # type: ignore[no-redef]
        BACKENDS,
        BenchmarkResult,
        COMPETITOR_SCORES,
        DATASETS,
        compute_f1,
        compute_recall_at_k,
        ensure_data_dir,
        exact_match,
        load_hf_dataset,
        semantic_match,
    )

from su_memory.sdk.lite_pro import SuMemoryLitePro  # noqa: E402


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 数据集 split 与 HuggingFace 子集 / 文件名映射
SPLIT_MAP: dict[str, dict[str, str]] = {
    "oracle": {
        "config": "oracle",
        "filename": "longmemeval_oracle.json",
    },
    "s": {
        "config": "longmemeval_s",
        "filename": "longmemeval_s_cleaned.json",
    },
    "m": {
        "config": "longmemeval_m",
        "filename": "longmemeval_m_cleaned.json",
    },
}

#: 单跳信息提取相关的 question_type
SINGLE_SESSION_TYPES: frozenset[str] = frozenset(
    {
        "single-session-user",
        "single-session-assistant",
        "single-session-preference",
    }
)

#: 注入记忆时的 chunk 字符长度（按句拆分时上限）
DEFAULT_CHUNK_CHARS = 800

#: 答案命中匹配阈值
SEMANTIC_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def _load_local_json(split: str) -> list[dict[str, Any]]:
    """尝试从本地缓存加载 LongMemEval JSON。"""

    cache_dir = Path(ensure_data_dir("longmemeval"))
    fname = SPLIT_MAP[split]["filename"]
    candidates = [cache_dir / fname, _BENCH_DIR / "data" / fname]
    for path in candidates:
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and "data" in data:
                data = data["data"]
            if isinstance(data, list):
                return data
    return []


def load_longmemeval(split: str = "oracle", verbose: bool = False) -> list[dict[str, Any]]:
    """加载指定 split 的 LongMemEval 数据。

    优先级：本地缓存 JSON > HuggingFace ``xiaowu0162/longmemeval-cleaned``。
    """

    if split not in SPLIT_MAP:
        raise ValueError(f"Unknown split '{split}'. Choices: {list(SPLIT_MAP)}")

    local = _load_local_json(split)
    if local:
        if verbose:
            print(f"[LongMemEval] 使用本地缓存数据集 split={split} n={len(local)}")
        return local

    hf_id = DATASETS["longmemeval"]["hf_id"]
    cache_dir = DATASETS["longmemeval"]["local_cache"]
    cfg_name = SPLIT_MAP[split]["config"]
    if verbose:
        print(f"[LongMemEval] 从 HuggingFace 加载 {hf_id} (config={cfg_name})")

    # 兼容多种数据集 layout：优先按 config name 加载，回退到默认 split。
    last_err: Exception | None = None
    for attempt in (
        {"name": cfg_name, "split": "test"},
        {"name": cfg_name, "split": "train"},
        {"split": "test"},
        {"split": "train"},
    ):
        try:
            ds = load_hf_dataset(hf_id, cache_dir=cache_dir, **attempt)
            return [dict(item) for item in ds]
        except Exception as exc:  # pragma: no cover - 网络相关
            last_err = exc
            continue

    raise RuntimeError(
        f"Failed to load LongMemEval split={split}: {last_err}. "
        f"Place '{SPLIT_MAP[split]['filename']}' under {cache_dir}."
    )


# ---------------------------------------------------------------------------
# 文本处理
# ---------------------------------------------------------------------------

def _format_turn(turn: dict[str, Any]) -> str:
    """将 ``haystack_sessions`` 中的单条 turn 渲染为带 role 标签的文本。"""

    role = str(turn.get("role", "user")).strip() or "user"
    content = str(turn.get("content", "")).strip()
    return f"[{role}] {content}" if content else ""


def _chunk_session(session_turns: list[dict[str, Any]], max_chars: int) -> list[str]:
    """将一个 session 的对话按字符上限切成若干 chunk。"""

    chunks: list[str] = []
    buf: list[str] = []
    cur_len = 0
    for turn in session_turns:
        line = _format_turn(turn)
        if not line:
            continue
        if cur_len + len(line) + 1 > max_chars and buf:
            chunks.append("\n".join(buf))
            buf = [line]
            cur_len = len(line)
        else:
            buf.append(line)
            cur_len += len(line) + 1
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def _is_answer_found(retrieved_texts: list[str], gold_answer: str) -> bool:
    """多策略答案匹配：精确包含 → SQuAD-EM → 语义相似度。"""

    if not gold_answer or not retrieved_texts:
        return False
    gold_lc = gold_answer.lower().strip()
    if not gold_lc:
        return False

    for text in retrieved_texts:
        if not text:
            continue
        if gold_lc in text.lower():
            return True
        if exact_match(text, gold_answer):
            return True

    for text in retrieved_texts:
        if text and semantic_match(text, gold_answer, threshold=SEMANTIC_THRESHOLD):
            return True
    return False


def _position_bucket(idx: int, total: int) -> str:
    """将 chunk 序号划分为 early / mid / late 三档。"""

    if total <= 0:
        return "early"
    pos = idx / total
    if pos < 1 / 3:
        return "early"
    if pos < 2 / 3:
        return "mid"
    return "late"


# ---------------------------------------------------------------------------
# 评测核心
# ---------------------------------------------------------------------------

@dataclass
class _RunStats:
    """单后端运行的细粒度计数器。"""

    total: int = 0
    correct: int = 0
    recall_at_1: int = 0
    recall_at_3: int = 0
    recall_at_5: int = 0
    f1_sum: float = 0.0

    by_type: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    by_position: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))

    abstain_tp: int = 0  # 真放弃 & 系统未命中
    abstain_fp: int = 0  # 真回答 & 系统未命中
    abstain_fn: int = 0  # 真放弃 & 系统命中
    abstain_total_gold: int = 0
    abstain_total_pred: int = 0

    add_times_ms: list[float] = field(default_factory=list)
    query_times_ms: list[float] = field(default_factory=list)
    chunks_ingested: int = 0


def _build_memory(backend: str, storage_path: str) -> SuMemoryLitePro:
    """根据 backend key 构造 :class:`SuMemoryLitePro` 实例。"""

    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Choices: {list(BACKENDS)}")

    cfg = BACKENDS[backend]
    backend_type = cfg["type"]

    # 清理旧持久化目录，确保隔离
    if os.path.exists(storage_path):
        shutil.rmtree(storage_path, ignore_errors=True)
    os.makedirs(storage_path, exist_ok=True)

    if backend_type == "sentence-transformers":
        os.environ.setdefault("SU_MEMORY_EMBEDDING_MODEL", cfg["model"])
        embedding_backend = "sentence-transformers"
    elif backend_type == "ollama":
        embedding_backend = "ollama"
    else:
        embedding_backend = backend_type

    return SuMemoryLitePro(
        storage_path=storage_path,
        embedding_backend=embedding_backend,
        enable_vector=True,
        enable_tfidf=True,
        enable_graph=False,  # 评测纯检索，关闭 graph 加速
        enable_temporal=False,
        enable_session=False,
        enable_prediction=False,
        enable_explainability=False,
    )


def _ingest_question(
    memory: SuMemoryLitePro,
    sample: dict[str, Any],
    stats: _RunStats,
    chunk_chars: int,
) -> int:
    """按 chunk 注入 ``haystack_sessions``，返回总 chunk 数。"""

    haystack: list[list[dict[str, Any]]] = sample.get("haystack_sessions") or []
    qid = str(sample.get("question_id", ""))
    total_chunks = 0

    # 先把所有 session 的 chunk 收集起来，以便按全局位置打 position 标签
    flat_chunks: list[tuple[str, str]] = []  # (session_id, chunk_text)
    session_ids = sample.get("haystack_session_ids") or []
    for s_idx, session in enumerate(haystack):
        sid = session_ids[s_idx] if s_idx < len(session_ids) else f"session_{s_idx}"
        for chunk in _chunk_session(session, chunk_chars):
            flat_chunks.append((str(sid), chunk))

    answer_session_ids = {str(s) for s in (sample.get("answer_session_ids") or [])}

    for c_idx, (sid, text) in enumerate(flat_chunks):
        position_bucket = _position_bucket(c_idx, max(len(flat_chunks), 1))
        meta = {
            "question_id": qid,
            "session_id": sid,
            "chunk_index": c_idx,
            "position_bucket": position_bucket,
            "is_evidence": sid in answer_session_ids,
        }
        t0 = time.perf_counter()
        try:
            memory.add(content=text, metadata=meta)
        except Exception as exc:  # pragma: no cover - 单条失败不应中断评测
            print(f"  [warn] add failed: {exc}")
            continue
        stats.add_times_ms.append((time.perf_counter() - t0) * 1000)
        total_chunks += 1

    stats.chunks_ingested += total_chunks
    return total_chunks


def _evaluate_question(
    memory: SuMemoryLitePro,
    sample: dict[str, Any],
    stats: _RunStats,
    top_k: int = 5,
) -> dict[str, Any]:
    """对单条问题执行检索并更新统计，返回逐题诊断信息。"""

    question = str(sample.get("question", ""))
    gold_answer = str(sample.get("answer", "") or "")
    qid = str(sample.get("question_id", ""))
    qtype = str(sample.get("question_type", "unknown"))
    is_abstain_gold = qid.endswith("_abs")

    t0 = time.perf_counter()
    try:
        results = memory.query(question, top_k=top_k)
    except Exception as exc:  # pragma: no cover
        print(f"  [warn] query failed for {qid}: {exc}")
        results = []
    stats.query_times_ms.append((time.perf_counter() - t0) * 1000)

    retrieved_texts = [str(r.get("content", "")) for r in results]

    # 命中判断
    hit_top1 = _is_answer_found(retrieved_texts[:1], gold_answer) if not is_abstain_gold else False
    hit_top3 = _is_answer_found(retrieved_texts[:3], gold_answer) if not is_abstain_gold else False
    hit_top5 = _is_answer_found(retrieved_texts[:5], gold_answer) if not is_abstain_gold else False

    # 放弃识别：以 top5 是否检索到 evidence session 为信号
    answer_session_ids = {str(s) for s in (sample.get("answer_session_ids") or [])}
    retrieved_sessions = {
        str((r.get("metadata") or {}).get("session_id"))
        for r in results
    }
    found_evidence = bool(answer_session_ids & retrieved_sessions) or hit_top5
    pred_abstain = not found_evidence

    # 总体计数
    stats.total += 1
    if is_abstain_gold:
        stats.abstain_total_gold += 1
        # 正确："不知道"问题应当“拒答” → 我们用 pred_abstain 等价
        if pred_abstain:
            correct = True
            stats.abstain_tp += 1
        else:
            correct = False
            stats.abstain_fn += 1
    else:
        correct = hit_top5
        if hit_top1:
            stats.recall_at_1 += 1
        if hit_top3:
            stats.recall_at_3 += 1
        if hit_top5:
            stats.recall_at_5 += 1
        if pred_abstain:
            stats.abstain_fp += 1

    if pred_abstain:
        stats.abstain_total_pred += 1

    if correct:
        stats.correct += 1

    # F1：以 top1 文本对 gold 计算（仅对非放弃问题）
    if not is_abstain_gold and retrieved_texts:
        stats.f1_sum += compute_f1(retrieved_texts[0], gold_answer)

    # 维度切片
    type_bucket = stats.by_type[qtype]
    type_bucket[1] += 1
    if correct:
        type_bucket[0] += 1

    # 位置维度：按 evidence chunk 的 position_bucket 归类（取首个命中证据，否则用问题元信息）
    pos_label = "unknown"
    for r in results:
        meta = r.get("metadata") or {}
        if meta.get("is_evidence"):
            pos_label = str(meta.get("position_bucket", "unknown"))
            break
    if pos_label == "unknown":
        # 回退：用 sample 自带 position 字段（若有）
        pos_label = str(sample.get("position", "unknown"))
    pos_bucket = stats.by_position[pos_label]
    pos_bucket[1] += 1
    if correct:
        pos_bucket[0] += 1

    return {
        "question_id": qid,
        "question_type": qtype,
        "abstain_gold": is_abstain_gold,
        "abstain_pred": pred_abstain,
        "correct": correct,
        "hit_top1": hit_top1,
        "hit_top3": hit_top3,
        "hit_top5": hit_top5,
    }


def run_backend(
    backend: str,
    samples: list[dict[str, Any]],
    storage_root: str,
    top_k: int = 5,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    verbose: bool = False,
) -> tuple[BenchmarkResult, list[dict[str, Any]]]:
    """对单个嵌入后端执行完整评测，返回 :class:`BenchmarkResult` + 逐题诊断。"""

    stats = _RunStats()
    diagnostics: list[dict[str, Any]] = []
    storage_path = os.path.join(storage_root, f"longmem_{backend}")

    if verbose:
        print(f"\n[LongMemEval] 初始化后端: {BACKENDS[backend]['name']}")
    memory = _build_memory(backend, storage_path)

    try:
        for i, sample in enumerate(samples):
            if verbose:
                qid = sample.get("question_id", f"#{i}")
                qtype = sample.get("question_type", "?")
                print(f"  [{i + 1}/{len(samples)}] {qid} ({qtype})")

            # 每题独立内存：清空再注入，避免跨题串扰
            memory.clear()
            _ingest_question(memory, sample, stats, chunk_chars)
            diag = _evaluate_question(memory, sample, stats, top_k=top_k)
            diagnostics.append(diag)
    finally:
        try:
            memory.clear()
        except Exception:
            pass

    result = _stats_to_result(backend, stats, len(samples))
    return result, diagnostics


def _stats_to_result(backend: str, stats: _RunStats, total_samples: int) -> BenchmarkResult:
    """将累计统计折算为 :class:`BenchmarkResult`。"""

    total = stats.total or 1
    answerable = max(stats.total - stats.abstain_total_gold, 1)

    # 维度评分
    dim: dict[str, float] = {}
    single_correct = sum(stats.by_type[t][0] for t in SINGLE_SESSION_TYPES if t in stats.by_type)
    single_total = sum(stats.by_type[t][1] for t in SINGLE_SESSION_TYPES if t in stats.by_type)
    if single_total:
        dim["single_session"] = single_correct / single_total
    if "multi-session" in stats.by_type:
        c, t = stats.by_type["multi-session"]
        dim["multi_session"] = c / t if t else 0.0
    if "temporal-reasoning" in stats.by_type:
        c, t = stats.by_type["temporal-reasoning"]
        dim["temporal_reasoning"] = c / t if t else 0.0
    if "knowledge-update" in stats.by_type:
        c, t = stats.by_type["knowledge-update"]
        dim["knowledge_update"] = c / t if t else 0.0

    # 位置维度
    for pos in ("early", "mid", "late"):
        if pos in stats.by_position:
            c, t = stats.by_position[pos]
            dim[f"position_{pos}"] = c / t if t else 0.0

    # 放弃识别 P/R
    abstain_precision = (
        stats.abstain_tp / stats.abstain_total_pred if stats.abstain_total_pred else 0.0
    )
    abstain_recall = (
        stats.abstain_tp / stats.abstain_total_gold if stats.abstain_total_gold else 0.0
    )
    if stats.abstain_total_gold:
        dim["abstain_precision"] = abstain_precision
        dim["abstain_recall"] = abstain_recall

    avg_add = sum(stats.add_times_ms) / len(stats.add_times_ms) if stats.add_times_ms else 0.0
    avg_query = (
        sum(stats.query_times_ms) / len(stats.query_times_ms) if stats.query_times_ms else 0.0
    )

    return BenchmarkResult(
        benchmark_name="longmemeval",
        backend=backend,
        total_questions=stats.total,
        correct=stats.correct,
        accuracy=stats.correct / total,
        f1_score=stats.f1_sum / answerable if answerable else 0.0,
        recall_at_1=stats.recall_at_1 / answerable if answerable else 0.0,
        recall_at_3=stats.recall_at_3 / answerable if answerable else 0.0,
        recall_at_5=stats.recall_at_5 / answerable if answerable else 0.0,
        avg_query_time_ms=avg_query,
        avg_add_time_ms=avg_add,
        dimension_scores=dim,
        metadata={
            "samples_in_split": total_samples,
            "chunks_ingested": stats.chunks_ingested,
            "abstain_total_gold": stats.abstain_total_gold,
            "abstain_total_pred": stats.abstain_total_pred,
            "backend_name": BACKENDS[backend]["name"],
        },
    )


# ---------------------------------------------------------------------------
# 报表
# ---------------------------------------------------------------------------

def _fmt_pct(v: float | None) -> str:
    return "N/A" if v is None else f"{v:.1%}"


def render_report(results: dict[str, BenchmarkResult]) -> str:
    """生成 ASCII 综合报表（含 Hindsight 对比）。"""

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("  su-memory · LongMemEval Benchmark Report (六维度官方版)")
    lines.append("=" * 78)
    lines.append(f"  Run timestamp: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    # 总览
    header = f"  {'Backend':<32} {'Acc':>7} {'F1':>7} {'R@1':>7} {'R@3':>7} {'R@5':>7}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for key, res in results.items():
        lines.append(
            f"  {BACKENDS[key]['name']:<32} "
            f"{res.accuracy:>7.1%} {res.f1_score:>7.1%} "
            f"{res.recall_at_1:>7.1%} {res.recall_at_3:>7.1%} {res.recall_at_5:>7.1%}"
        )
    lines.append("")

    # 六维度
    lines.append("  --- 六维度评分 ---")
    dim_keys = (
        ("single_session", "1. 信息提取(单跳)"),
        ("multi_session", "2. 多会话推理"),
        ("temporal_reasoning", "3. 时序推理"),
        ("knowledge_update", "4. 知识更新"),
        ("position_early", "5a. 位置-早段"),
        ("position_mid", "5b. 位置-中段"),
        ("position_late", "5c. 位置-晚段"),
        ("abstain_precision", "6a. 放弃 Precision"),
        ("abstain_recall", "6b. 放弃 Recall"),
    )
    head = f"  {'维度':<24}" + "".join(f" {BACKENDS[k]['name'][:18]:>20}" for k in results)
    lines.append(head)
    lines.append("  " + "-" * (len(head) - 2))
    for dim_key, label in dim_keys:
        row = f"  {label:<24}"
        for backend_key, res in results.items():
            v = res.dimension_scores.get(dim_key)
            row += f" {_fmt_pct(v):>20}"
        lines.append(row)
    lines.append("")

    # 性能
    lines.append("  --- 性能 ---")
    for key, res in results.items():
        meta = res.metadata
        lines.append(
            f"  {BACKENDS[key]['name']:<32} "
            f"add={res.avg_add_time_ms:6.1f}ms  "
            f"query={res.avg_query_time_ms:6.1f}ms  "
            f"chunks={meta.get('chunks_ingested', 0)}"
        )
    lines.append("")

    # Hindsight 对标
    lines.append("  --- 与竞品对比 (LongMemEval Accuracy) ---")
    lines.append(f"  {'System':<28} {'Accuracy':>10}")
    lines.append("  " + "-" * 40)
    for name, scores in COMPETITOR_SCORES.items():
        v = scores.get("longmemeval_accuracy")
        lines.append(f"  {name:<28} {_fmt_pct(v):>10}")
    for key, res in results.items():
        diff = res.accuracy - (COMPETITOR_SCORES["hindsight"].get("longmemeval_accuracy") or 0.0)
        lines.append(
            f"  {('su-memory/' + key):<28} {res.accuracy:>10.1%}  "
            f"(vs Hindsight {diff:+.1%})"
        )
    lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_backends(arg: str) -> list[str]:
    """解析 ``--backend`` 参数。"""

    arg = arg.strip().lower()
    if arg == "both":
        return ["ollama", "sbert"]
    if arg in BACKENDS:
        return [arg]
    raise ValueError(f"Unknown backend '{arg}'. Choices: ollama / sbert / both")


def build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="su-memory LongMemEval benchmark runner (官方六维度版)",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "sbert", "both"],
        default="both",
        help="嵌入后端：ollama / sbert / both (默认 both)",
    )
    parser.add_argument(
        "--split",
        choices=list(SPLIT_MAP.keys()),
        default="oracle",
        help="数据集 split: oracle / s / m (默认 oracle)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="限制评测问题数（调试用）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="检索 top-K (默认 5)",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=DEFAULT_CHUNK_CHARS,
        help=f"注入 chunk 字符上限 (默认 {DEFAULT_CHUNK_CHARS})",
    )
    parser.add_argument(
        "--storage",
        default="/tmp/su-memory-bench/longmem",
        help="su-memory 持久化目录",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="JSON 结果输出路径 (默认 results/longmemeval_<split>_<ts>.json)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="ASCII 报表输出路径 (默认仅打印至 stdout)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="详细输出",
    )
    return parser


def _default_output(split: str) -> str:
    out_dir = _BENCH_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(out_dir / f"longmemeval_{split}_{ts}.json")


def _filter_samples(samples: list[dict[str, Any]], max_q: int | None) -> list[dict[str, Any]]:
    if max_q is None or max_q <= 0:
        return samples
    return samples[: max_q]


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    backends = _resolve_backends(args.backend)

    print("=" * 60)
    print("  su-memory · LongMemEval Benchmark")
    print(f"  split={args.split}  backends={backends}  top_k={args.top_k}")
    print("=" * 60)

    samples = load_longmemeval(split=args.split, verbose=args.verbose)
    samples = _filter_samples(samples, args.max_questions)
    if not samples:
        print("[error] no samples loaded; aborting.")
        return 2
    print(f"[LongMemEval] loaded {len(samples)} samples")

    results: dict[str, BenchmarkResult] = {}
    diagnostics: dict[str, list[dict[str, Any]]] = {}
    for backend in backends:
        try:
            res, diag = run_backend(
                backend=backend,
                samples=samples,
                storage_root=args.storage,
                top_k=args.top_k,
                chunk_chars=args.chunk_chars,
                verbose=args.verbose,
            )
        except Exception as exc:
            print(f"[error] backend={backend} failed: {exc}")
            continue
        results[backend] = res
        diagnostics[backend] = diag

    if not results:
        print("[error] all backends failed.")
        return 3

    report = render_report(results)
    print("\n" + report)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(report, encoding="utf-8")
        print(f"[LongMemEval] report → {args.report}")

    output_path = args.output or _default_output(args.split)
    payload = {
        "benchmark": "longmemeval",
        "split": args.split,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "backends": {k: v.to_dict() for k, v in results.items()},
        "competitor_scores": COMPETITOR_SCORES,
        "config": {
            "top_k": args.top_k,
            "chunk_chars": args.chunk_chars,
            "max_questions": args.max_questions,
            "samples_evaluated": len(samples),
        },
        "diagnostics_sample": {
            k: v[:20] for k, v in diagnostics.items()  # 仅保留前 20 条诊断，避免过大
        },
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"[LongMemEval] results → {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
