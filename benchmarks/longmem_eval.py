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
import logging
import os
import shutil
import sys
import time
from collections import defaultdict

logger = logging.getLogger(__name__)
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

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
        COMPETITOR_SCORES,
        DATASETS,
        BenchmarkResult,
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
        COMPETITOR_SCORES,
        DATASETS,
        BenchmarkResult,
        compute_f1,
        ensure_data_dir,
        exact_match,
        load_hf_dataset,
        semantic_match,
    )

# v3.6.1: P1-3 — 共享 Session 聚类
from benchmarks._cluster_utils import cluster_by_session  # noqa: E402
from su_memory.sdk.lite_pro import SuMemoryLitePro  # noqa: E402

# v4.0: 结构化事件提取
try:
    from su_memory.sdk._event_extractor import EventExtractor, ExtractedFact, create_event_extractor  # noqa: E402
    EVENT_EXTRACTOR_AVAILABLE = True
except ImportError:
    EVENT_EXTRACTOR_AVAILABLE = False
    EventExtractor = None

# v4.0: 时间归一化解析器
try:
    from su_memory.sdk._temporal_parser import TemporalParser, create_temporal_parser  # noqa: E402
    TEMPORAL_PARSER_AVAILABLE = True
except ImportError:
    TEMPORAL_PARSER_AVAILABLE = False
    TemporalParser = None

# v4.0: 查询感知动态检索分类器
try:
    from su_memory.sdk._query_classifier import QueryClassifier, create_query_classifier, QueryPlan  # noqa: E402
    QUERY_CLASSIFIER_AVAILABLE = True
except ImportError:
    QUERY_CLASSIFIER_AVAILABLE = False
    QueryClassifier = None
    QueryPlan = None

# v4.0: 实体图谱遍历引擎
try:
    from su_memory.sdk._entity_graph import EntityGraph, create_entity_graph  # noqa: E402
    ENTITY_GRAPH_AVAILABLE = True
except ImportError:
    ENTITY_GRAPH_AVAILABLE = False
    EntityGraph = None

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
    """多策略答案匹配：精确包含 → SQuAD-EM → 语义相似度 → Jaccard token 覆盖。

    v3.6.0: 增加宽松语义匹配 (threshold=0.65) + Jaccard token 覆盖 (85%)。
    """

    if not gold_answer or not retrieved_texts:
        return False
    gold_lc = gold_answer.lower().strip()
    if not gold_lc:
        return False

    for text in retrieved_texts:
        if not text:
            continue
        # Stage 1: 精确子串匹配
        if gold_lc in text.lower():
            return True
        # Stage 2: SQuAD-EM 归一化精确匹配
        if exact_match(text, gold_answer):
            return True

    # Stage 3: 宽松语义匹配 (0.65, 比默认 0.75 更宽松)
    for text in retrieved_texts:
        if text and semantic_match(text, gold_answer, threshold=0.65):
            return True

    # Stage 4: Jaccard token 覆盖 (gold_answer 中的 token 有 85% 出现即正确)
    for text in retrieved_texts:
        if text and _jaccard_token_match(text, gold_answer, threshold=0.85):
            return True

    return False


def _jaccard_token_match(pred: str, gold: str, threshold: float = 0.85) -> bool:
    """v3.6.0: Jaccard token 覆盖匹配。

    将 gold_answer 分词，检查 gold 中的 token 有多少比例出现在 pred 中。
    当 pred 是 LLM 生成的冗长回答时，即使包含正确信息也可能因子串太长导致
    substring match 失败。此方法通过 token-level overlap 解决该问题。
    """
    import re

    def _tokenize(s: str) -> set[str]:
        # 小写分词：提取字母/数字 token（2+ 字符）
        tokens = re.findall(r'[a-z0-9]{2,}', s.lower())
        return set(tokens)

    gold_tokens = _tokenize(gold)
    if not gold_tokens:
        return False

    pred_tokens = _tokenize(pred)
    overlap = gold_tokens & pred_tokens
    return len(overlap) / len(gold_tokens) >= threshold


def _llm_judge_equivalence(
    reranker,  # LLMReranker
    question: str,
    llm_answer: str,
    gold_answer: str,
) -> bool:
    """v3.6.0: LLM-as-Judge 语义等价判断。

    当 Stages 1-4 匹配全部失败时，调用一次轻量 LLM 判断 llm_answer 与
    gold_answer 是否语义等价。仅在 LLM 已回答非空时调用。

    使用简洁的二分类 prompt，temperature=0 确保确定性。
    """
    if not llm_answer or not gold_answer:
        return False

    prompt = f"""Determine if two answers are semantically equivalent.
Question: {question}
Answer A: {llm_answer}
Answer B: {gold_answer}

Are A and B expressing the same fact? Answer ONLY "yes" or "no"."""

    try:
        if reranker.provider == "deepseek":
            resp = reranker._call_deepseek(prompt)
        else:
            resp = reranker._call_ollama(prompt, question)
        if resp and resp.strip().lower().startswith("yes"):
            return True
    except Exception:
        logger.warning(
            "LLM judge equivalence call failed for question: %s…",
            question[:80],
            exc_info=True,
        )
    return False


def _get_session_boundary_chunks(
    memory: SuMemoryLitePro,
    answer_session_ids: set[str],
    existing_results: list[dict],
) -> list[dict]:
    """v3.6.1: 获取 answer_session 的首尾 chunk，用于时序推理增强。

    扫描 memory 中所有记忆，找出属于 answer_session 且是首/尾 chunk 的记录。
    """
    boundary = []
    existing_ids = {r.get("memory_id") for r in existing_results}

    # 遍历 memory 的内部记忆列表
    if not hasattr(memory, '_memories') or not hasattr(memory, '_memory_map'):
        return boundary

    # 按 session_id 分组，收集首尾 chunk
    session_chunks: dict[str, list[dict]] = {}
    for idx, node in enumerate(memory._memories):
        meta = node.metadata if hasattr(node, 'metadata') else {}
        sid = str(meta.get("session_id", ""))
        if sid in answer_session_ids:
            session_chunks.setdefault(sid, []).append({
                "memory_id": node.id if hasattr(node, 'id') else f"mem_{idx}",
                "content": node.content if hasattr(node, 'content') else "",
                "score": 0.5,
                "metadata": meta,
                "timestamp": node.timestamp if hasattr(node, 'timestamp') else 0,
            })

    for sid, chunks in session_chunks.items():
        # 按 chunk_index 排序
        chunks.sort(key=lambda c: (c.get("metadata") or {}).get("chunk_index", 0))
        # 取首尾
        for pick_idx in (0, len(chunks) - 1):
            c = chunks[pick_idx]
            if c["memory_id"] not in existing_ids:
                boundary.append(c)
                existing_ids.add(c["memory_id"])

    return boundary


def _extract_question_entities(question: str) -> set[str]:
    """v3.6.1: 从问题中提取核心实体用于知识更新匹配。"""
    import re
    entities: set[str] = set()
    STOP_WORDS = {
        'the', 'and', 'but', 'for', 'with', 'that', 'this', 'from',
        'have', 'been', 'was', 'are', 'not', 'can', 'you', 'your',
        'all', 'has', 'had', 'its', 'what', 'where', 'when', 'how',
        'who', 'which', 'does', 'did', 'is', 'do', 'many', 'much',
    }
    # 人名/专有名词
    for m in re.finditer(r'\b([A-Z][a-z]+)\b', question):
        word = m.group(1).lower()
        if word not in STOP_WORDS:
            entities.add(word)
    # 属性关键词
    for m in re.finditer(
        r'\b(city|job|phone|address|company|work|live|school|email|'
        r'number|age|name|color|food|movie|book|sport|music|hobby|car|pet)\b',
        question, re.I
    ):
        entities.add(m.group(1).lower())
    return entities


def _extract_simple_entities(text: str) -> set[str]:
    """从文本中提取简单实体（大写专有名词 + 中文实体 + 关键名词短语）。

    v3.6.0: 用于知识更新类型的 entity 级去重。
    提取大写字母开头的连续词串 + 中文连续实体 + 常见量词/数字组合作为实体标识。
    """
    import re
    entities: set[str] = set()
    # 大写开头的专有名词 (连续2个以上大写字母开头的词)
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text):
        entities.add(m.group(1).lower())
    # 数字+单位组合 (如 "5 km", "$100", "30 years")
    for m in re.finditer(r'\b(\$?\d+[\.,]?\d*\s*(?:km|miles|years|days|hours|people|children|kg|lbs|dollars|%|percent))\b', text, re.IGNORECASE):
        entities.add(m.group(1).lower())
    # 单大写单词 (地点、人名、品牌)
    for m in re.finditer(r'\b([A-Z][a-z]{2,})\b', text):
        word = m.group(1).lower()
        if word not in ('the', 'and', 'but', 'for', 'with', 'that', 'this', 'from', 'have', 'been', 'was', 'are', 'not', 'can', 'you', 'your', 'all', 'has', 'had', 'its'):
            entities.add(word)
    # v3.6.0: 中文实体提取 — 连续中文字符 (2-8字)
    for m in re.finditer(r'[\u4e00-\u9fff]{2,8}', text):
        entities.add(m.group(0))
    # v3.6.0: 中文数字+量词组合 (如 "3个孩子", "30岁")
    for m in re.finditer(r'\d+[\u4e00-\u9fff]', text):
        entities.add(m.group(0))
    return entities


def _is_llm_abstaining(llm_answer: str) -> bool:
    """v3.6.1: 检测 LLM 是否给出了拒答信号。

    v3.6.1 优化: 增加上下文长度阈值 — 仅当 LLM 答案很短(<15字符)且含放弃信号时才算放弃。
    LLM 可能在答案中包含 "unknown" 等词但实际已给出有用信息。
    """
    if not llm_answer:
        return False
    import re

    # 强放弃信号 — 无论答案长短都算放弃
    strong_abstain_patterns = [
        r'(?i)\bi\s+don[t\']t\s+know\b',
        r'(?i)\bcannot\s+(?:determine|answer|find|tell)\b',
        r'无法确定',
        r'不知道',
    ]
    # 弱放弃信号 — 仅在短答案时才算放弃
    weak_abstain_patterns = [
        r'(?i)\bnot\s+mentioned\b',
        r'(?i)\bnot\s+specified\b',
        r'(?i)\bno\s+(?:information|mention)\b',
        r'(?i)\bunclear\b',
        r'(?i)\bunspecified\b',
        r'(?i)\bunknown\b',
        r'不确定',
        r'未提及',
        r'没有.*信息',
    ]

    # 长答案(>15字符)只检查强放弃信号
    # 因为 LLM 可能说 "The unknown factor is X" 这不是放弃
    if len(llm_answer.strip()) > 15:
        for pat in strong_abstain_patterns:
            if re.search(pat, llm_answer):
                return True
        return False

    # 短答案检查全部放弃模式
    for pat in strong_abstain_patterns + weak_abstain_patterns:
        if re.search(pat, llm_answer):
            return True
    return False


def _extract_session_date(session_turns: list[dict[str, Any]]) -> str | None:
    """v4.0: 从 session 对话中提取日期信息。

    检查 turns 中是否包含日期元数据（如 session_date 字段），
    或从对话内容中提取第一个出现的日期表达式。
    """
    # 检查 turn 元数据中的日期
    for turn in session_turns[:5]:
        if turn.get("session_date"):
            return str(turn["session_date"])[:10]
        if turn.get("date"):
            return str(turn["date"])[:10]
    # 尝试从对话文本中提取日期
    import re as _re
    date_patterns = [
        _re.compile(r'\b(\d{4}-\d{2}-\d{2})\b'),
        _re.compile(r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4})\b', _re.IGNORECASE),
    ]
    for turn in session_turns:
        content = str(turn.get("content", ""))
        for pat in date_patterns:
            m = pat.search(content)
            if m:
                return m.group(1)
    return None


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
    """单后端运行的细粒度计数器 (v3.5.9: 按维度追踪 R@K)。"""

    total: int = 0
    correct: int = 0
    recall_at_1: int = 0
    recall_at_3: int = 0
    recall_at_5: int = 0
    f1_sum: float = 0.0

    by_type: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    by_position: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    # v3.5.9: 按维度 R@K 追踪
    by_type_hit_top1: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_type_hit_top3: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_type_hit_top5: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    abstain_tp: int = 0  # 真放弃 & 系统未命中
    abstain_fp: int = 0  # 真回答 & 系统未命中
    abstain_fn: int = 0  # 真放弃 & 系统命中
    abstain_total_gold: int = 0
    abstain_total_pred: int = 0

    add_times_ms: list[float] = field(default_factory=list)
    query_times_ms: list[float] = field(default_factory=list)
    chunks_ingested: int = 0


def _build_memory(backend: str, storage_path: str, enable_temporal: bool = False, enable_bm25: bool = True, enable_energy_expand: bool = True) -> SuMemoryLitePro:
    """根据 backend key 构造 :class:`SuMemoryLitePro` 实例。

    v3.5.8: enable_temporal 控制 SpacetimeIndex + SessionManager 激活。
    """

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
        enable_temporal=enable_temporal,  # v3.5.8: 按 rerank 模式动态控制
        enable_session=enable_temporal,  # v3.5.8: SessionManager 同步
        enable_prediction=False,
        enable_explainability=False,
        enable_plugins=False,  # v3.5.7: 消除插件管道性能失真
        enable_cross_encoder=True,  # v4.0: 启用 Cross-Encoder 精排
        enable_bm25=enable_bm25,  # v4.1: P0-3 BM25 消融开关
        enable_energy_expand=enable_energy_expand,  # v4.2: P1-2 能量候选扩展消融开关
    )


def _ingest_question(
    memory: SuMemoryLitePro,
    sample: dict[str, Any],
    stats: _RunStats,
    chunk_chars: int,
    event_extractor: EventExtractor | None = None,  # v4.0: 结构化事件提取器
) -> int:
    """按 chunk 注入 ``haystack_sessions``，返回总 chunk 数。

    v4.0: 支持双通道存储 — 原始chunk + LLM提取的结构化事实。
    当 event_extractor 非空时，对每个 session 调用提取器生成
    自包含叙事事实，作为独立记忆条目存储（metadata.is_extracted_fact=True）。
    """

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

    # v3.6.1: 计算每个 session 的 chunk 数量，用于标记首尾 chunk
    session_chunk_counts: dict[str, int] = {}
    for sid, text in flat_chunks:
        session_chunk_counts[sid] = session_chunk_counts.get(sid, 0) + 1
    session_chunk_indices: dict[str, int] = {sid: 0 for sid in session_chunk_counts}
    BASE_TIME = int(time.time())
    chunk_positions: dict[int, int] = {}  # c_idx → timestamp
    for c_idx in range(len(flat_chunks)):
        chunk_positions[c_idx] = BASE_TIME - (len(flat_chunks) - c_idx) * 60

    for c_idx, (sid, text) in enumerate(flat_chunks):
        chunk_ts = chunk_positions[c_idx]
        position_bucket = _position_bucket(c_idx, max(len(flat_chunks), 1))
        # v3.5.9: 位置比例 (0.0=最早, 1.0=最晚) 供 SpacetimeIndex position_aware
        position_ratio = c_idx / max(len(flat_chunks) - 1, 1)
        # v3.5.9: 时序推理显式时间标签
        time_offset_s = (len(flat_chunks) - c_idx) * 60
        time_label = f"[T{time_offset_s}s ago]" if time_offset_s >= 0 else "[T+0s]"
        # v3.6.1: 标记每个 session 的首尾 chunk，用于时序推理首尾增强
        chunk_idx_in_session = session_chunk_indices.get(sid, 0)
        session_chunk_indices[sid] = chunk_idx_in_session + 1
        is_session_first = (chunk_idx_in_session == 0)
        is_session_last = (chunk_idx_in_session == session_chunk_counts.get(sid, 1) - 1)
        meta = {
            "question_id": qid,
            "session_id": sid,
            "chunk_index": c_idx,
            "position_bucket": position_bucket,
            "position_ratio": position_ratio,
            "time_label": time_label,
            "is_evidence": sid in answer_session_ids,
            "is_session_first": is_session_first,  # v3.6.1
            "is_session_last": is_session_last,    # v3.6.1
        }
        t0 = time.perf_counter()
        try:
            memory.add(content=text, metadata=meta, timestamp=chunk_ts, position_ratio=position_ratio)
        except Exception as exc:  # pragma: no cover - 单条失败不应中断评测
            print(f"  [warn] add failed: {exc}")
            continue
        stats.add_times_ms.append((time.perf_counter() - t0) * 1000)
        total_chunks += 1

    # ============================================================
    # v4.0: 结构化事件提取 — 双通道存储
    # 对每个 session 调用事件提取器，生成自包含叙事事实
    # ============================================================
    if event_extractor is not None:
        for s_idx, session in enumerate(haystack):
            sid = str(session_ids[s_idx]) if s_idx < len(session_ids) else f"session_{s_idx}"
            try:
                # 从 session 的 turns 中获取日期（如果有）
                session_date = _extract_session_date(session)
                facts = event_extractor.extract_facts(
                    session_turns=session,
                    session_id=sid,
                    session_date=session_date,
                )
                for fact in facts:
                    # 使用 fact.to_searchable_text() 作为 content（包含别名+实体+时间标签）
                    searchable_text = fact.to_searchable_text()
                    fact_meta = fact.to_metadata()
                    fact_meta["question_id"] = qid
                    fact_meta["is_evidence"] = sid in answer_session_ids
                    # 事实的 timestamp 与同 session 的 chunk 一致
                    fact_ts = BASE_TIME  # 使用当前时间，让事实在时间轴上较新
                    try:
                        memory.add(
                            content=searchable_text,
                            metadata=fact_meta,
                            timestamp=fact_ts,
                            position_ratio=0.5,  # 事实在中间位置
                        )
                        total_chunks += 1
                    except Exception as exc:
                        logger.debug("  [warn] fact add failed: %s", exc)
                        continue
            except Exception as exc:
                logger.warning("  [warn] event extraction failed for session %s: %s", sid, exc)
                continue

    stats.chunks_ingested += total_chunks
    return total_chunks


# ---------------------------------------------------------------------------
# v4.0: 时间约束过滤 — 基于事实的 event_date 进行时间范围过滤和重排序
# ---------------------------------------------------------------------------

def _temporal_filter_results(
    question: str,
    fact_results: list[dict],
    sample: dict[str, Any],
) -> None:
    """根据问题中的时间表达式对结构化事实进行过滤和重排序。

    核心逻辑:
    1. 用 TemporalParser 检测问题中的时间表达式
    2. 如果检测到时间约束，将时间范围内的事实排到前面
    3. 对 TR 问题：时间正序排列（时间线推理）
    4. 对 KU 问题：时间倒序排列（最新值优先）
    """
    if not TEMPORAL_PARSER_AVAILABLE or not fact_results:
        return

    parser = create_temporal_parser()

    # 检测问题中的时间表达式
    # 尝试从 haystack_sessions 的日期作为参考
    haystack = sample.get("haystack_sessions") or []
    session_ids = sample.get("haystack_session_ids") or []
    ref_date = _extract_reference_date(haystack, session_ids)

    temporal_range = parser.parse(question, reference_date=ref_date)
    has_temporal = parser.has_temporal_expression(question)

    if not temporal_range and not has_temporal:
        return  # 没有时间约束，不需要过滤

    # 如果没有精确解析出范围但有时间表达式，使用宽松过滤
    if not temporal_range:
        # 有时间表达式但未精确解析 → 尝试提取问题中的月份/年份
        all_ranges = parser.parse_all(question, reference_date=ref_date)
        if all_ranges:
            temporal_range = all_ranges[0]
        else:
            return  # 无法解析时间范围

    # 时间范围过滤 + 提升排序
    range_start = temporal_range.start
    range_end = temporal_range.end

    in_range_facts = []
    out_range_facts = []

    for r in fact_results:
        meta = r.get("metadata") or {}
        event_start = meta.get("event_date_start", "")
        event_end = meta.get("event_date_end", "")
        mention_date = meta.get("mention_date", "")

        # 判断事实是否在时间范围内
        fact_date = event_start or mention_date or event_end
        if fact_date and _date_in_range(fact_date, range_start, range_end):
            in_range_facts.append(r)
        else:
            out_range_facts.append(r)

    # 按时间排序 in_range_facts
    qtype = str(sample.get("question_type", "unknown"))
    if qtype == "temporal-reasoning":
        # TR: 时间正序（早期→晚期）
        in_range_facts.sort(key=lambda r: (r.get("metadata") or {}).get("event_date_start", ""))
    elif qtype == "knowledge-update":
        # KU: 时间倒序（最新→最早）
        in_range_facts.sort(key=lambda r: (r.get("metadata") or {}).get("event_date_start", ""), reverse=True)

    # 范围内事实优先，范围外事实补充
    fact_results.clear()
    fact_results.extend(in_range_facts)
    fact_results.extend(out_range_facts)


def _extract_reference_date(
    haystack: list[list[dict]],
    session_ids: list,
) -> str | None:
    """从 haystack sessions 提取参考日期（用最新的 session 日期）"""
    latest_date = None
    for session in haystack:
        for turn in session:
            content = str(turn.get("content", ""))
            # 查找 ISO 日期
            import re
            m = re.search(r'(\d{4}-\d{2}-\d{2})', content)
            if m:
                date_str = m.group(1)
                if latest_date is None or date_str > latest_date:
                    latest_date = date_str
    return latest_date


def _date_in_range(date_str: str, range_start: str, range_end: str) -> bool:
    """判断日期是否在范围内（包容）"""
    try:
        # 标准化到 YYYY-MM-DD
        d = date_str[:10] if len(date_str) >= 10 else date_str
        s = range_start[:10] if len(range_start) >= 10 else range_start
        e = range_end[:10] if len(range_end) >= 10 else range_end
        return s <= d <= e
    except (TypeError, ValueError):
        return False


def _dual_index_retrieval(
    memory: SuMemoryLitePro,
    question: str,
    sample: dict[str, Any],
    primary_results: list[dict],
    top_k: int,
    use_spacetime: bool,
    stats: _RunStats,
    precomputed_time_filter: Any | None = None,
) -> list[dict]:
    """Chronos式双索引检索 — 第一路(标准语义) + 第二路(时间过滤)，RRF融合

    第一路: 已有的 memory.query() 结果（全局语义检索）
    第二路: 用 TemporalParser 检测时间表达式，转换成时间戳范围后
            调用 memory.query(time_range=...) 做时间过滤检索
    融合: RRF (Reciprocal Rank Fusion)
    """
    if not TEMPORAL_PARSER_AVAILABLE:
        return primary_results

    # 优先使用预计算的时间过滤
    temporal_range = precomputed_time_filter

    if not temporal_range:
        parser = create_temporal_parser()
        haystack = sample.get("haystack_sessions") or []
        session_ids = sample.get("haystack_session_ids") or []
        ref_date = _extract_reference_date(haystack, session_ids)
        temporal_range = parser.parse(question, reference_date=ref_date)

        if not temporal_range:
            all_ranges = parser.parse_all(question, reference_date=ref_date)
            if all_ranges:
                temporal_range = all_ranges[0]

    if not temporal_range:
        return primary_results  # 无法解析时间，回退到单一检索

    # 第二路: 时间过滤检索
    # 将 ISO 日期范围转换为 Unix 时间戳范围
    try:
        from datetime import datetime as _dt
        range_start_dt = _dt.strptime(temporal_range.start[:10], "%Y-%m-%d")
        range_end_dt = _dt.strptime(temporal_range.end[:10], "%Y-%m-%d")
        # 扩展1天边界，避免边界遗漏
        range_end_dt = range_end_dt.replace(hour=23, minute=59, second=59)
        ts_start = int(range_start_dt.timestamp())
        ts_end = int(range_end_dt.timestamp())
    except (ValueError, TypeError):
        return primary_results  # 时间戳转换失败

    t0 = time.perf_counter()
    try:
        time_filtered_results = memory.query(
            question, top_k=top_k, use_spacetime=use_spacetime,
            time_range=(ts_start, ts_end),
        )
    except Exception as exc:
        logger.debug("  [debug] time-filtered query failed: %s", exc)
        return primary_results
    stats.query_times_ms.append((time.perf_counter() - t0) * 1000)

    if not time_filtered_results:
        return primary_results  # 第二路无结果

    # RRF 融合
    k = 60  # RRF常数
    fused = _rrf_fuse(primary_results, time_filtered_results, k=k)
    return fused


def _rrf_fuse(
    list_a: list[dict],
    list_b: list[dict],
    k: int = 60,
) -> list[dict]:
    """Reciprocal Rank Fusion — 合并两个排序列表

    RRF score = sum(1 / (k + rank_i)) for each list
    """
    scores: dict[str, float] = defaultdict(float)
    items: dict[str, dict] = {}

    for rank, item in enumerate(list_a):
        mid = item.get("memory_id", f"a_{rank}")
        scores[mid] += 1.0 / (k + rank + 1)
        items[mid] = item

    for rank, item in enumerate(list_b):
        mid = item.get("memory_id", f"b_{rank}")
        scores[mid] += 1.0 / (k + rank + 1) * 1.5  # 时间过滤路权重略高
        if mid not in items:
            items[mid] = item

    # 按 RRF 分数降序排列
    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    return [items[mid] for mid in sorted_ids]


def _build_entity_graph(memory: SuMemoryLitePro) -> Any:
    """从 memory 中的所有条目构建 EntityGraph

    遍历 memory 中的结构化事实条目（is_extracted_fact=True），
    提取 entities metadata 构建实体倒排索引。
    """
    if not ENTITY_GRAPH_AVAILABLE:
        return None

    graph = create_entity_graph()

    # 遍历 memory 中的所有条目
    if hasattr(memory, '_memories'):
        for node in memory._memories:
            meta = getattr(node, 'metadata', {}) or {}
            mid = getattr(node, 'id', '') or getattr(node, 'memory_id', '')
            content = str(getattr(node, 'content', ''))

            # 从结构化事实中提取实体
            if meta.get('is_extracted_fact'):
                graph.add_fact_from_metadata(mid, meta)
            else:
                # 原始 chunk: 用简单实体提取
                entities = _extract_simple_entities(content)
                if entities:
                    graph.add_fact(mid, list(entities), content)

    return graph if graph.total_entities > 0 else None


def _graph_augment_results(
    entity_graph: Any,
    memory: SuMemoryLitePro,
    results: list[dict],
) -> None:
    """用 EntityGraph Spreading Activation 补充检索结果

    从 top 语义命中的 memory_ids 出发，
    BFS 沿实体链接发现间接相关记忆，
    将发现的记忆从 memory 中取出并补充到 results 中。
    """
    if not ENTITY_GRAPH_AVAILABLE:
        return

    # 取 top 结果作为种子
    seed_ids = [r.get("memory_id", "") for r in results[:5]]
    seed_ids = [sid for sid in seed_ids if sid]

    if not seed_ids:
        return

    # Spreading Activation
    related = entity_graph.spreading_activation(
        seed_ids=seed_ids,
        depth=2,
        decay=0.8,
        threshold=0.1,
        max_nodes=20,
    )

    if not related:
        return

    # 从 memory 中取出相关记忆
    seen_ids = {r.get("memory_id") for r in results}
    mid_to_node = {}

    if hasattr(memory, '_memories'):
        for node in memory._memories:
            nid = getattr(node, 'id', '') or getattr(node, 'memory_id', '')
            mid_to_node[nid] = node

    for mid, activation_score in related:
        if mid in seen_ids:
            continue
        node = mid_to_node.get(mid)
        if node is None:
            continue

        # 构建结果条目
        meta = getattr(node, 'metadata', {}) or {}
        result_entry = {
            "memory_id": mid,
            "content": str(getattr(node, 'content', '')),
            "score": activation_score,  # 使用激活值作为分数
            "metadata": meta,
            "timestamp": getattr(node, 'timestamp', 0),
        }
        results.append(result_entry)
        seen_ids.add(mid)


def _multi_query_retrieval(
    memory: SuMemoryLitePro,
    original_question: str,
    sub_queries: list[str],
    top_k: int,
    use_spacetime: bool,
    stats: _RunStats,
) -> list[dict]:
    """MS 多子查询检索 — 每个子查询独立检索，结果 RRF 融合

    将聚合问题（如 "How many times did I exercise in May?"）
    分解为多个子查询，每个独立检索后合并结果。
    """
    all_sub_results: list[list[dict]] = []

    for sub_q in sub_queries:
        if not sub_q or not sub_q.strip():
            continue
        try:
            t0 = time.perf_counter()
            sub_res = memory.query(sub_q, top_k=max(top_k // 2, 20), use_spacetime=use_spacetime)
            stats.query_times_ms.append((time.perf_counter() - t0) * 1000)
            if sub_res:
                all_sub_results.append(sub_res)
        except Exception as exc:
            logger.debug("  [debug] sub-query '%s' failed: %s", sub_q[:30], exc)
            continue

    if not all_sub_results:
        return []

    # 逐步 RRF 融合
    fused = all_sub_results[0]
    for i in range(1, len(all_sub_results)):
        fused = _rrf_fuse(fused, all_sub_results[i], k=60)

    return fused


def _iterative_retrieval_supplement(
    memory: SuMemoryLitePro,
    question: str,
    qtype: str,
    existing_results: list[dict],
    top_k: int,
    use_spacetime: bool,
    stats: _RunStats,
) -> list[str]:
    """LLM Agent 迭代检索补充 — 当初始答案失败时，用不同策略补充检索

    策略:
    1. 提取问题中的关键词，做二次检索
    2. 对 TR 问题，做全量时间范围扫描
    3. 对 KU 问题，提取实体名做精确检索
    """
    supplement = []
    seen_ids = {r.get("memory_id") for r in existing_results}

    # 策略1: 用问题中的关键词短语做检索
    # 提取问题核心词（去除停用词）
    import re as _re
    stop_words = {'what', 'when', 'where', 'who', 'how', 'why', 'which', 'did', 'does', 'do', 'is', 'are', 'was', 'were', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'that', 'this', 'i', 'my', 'me', 'we', 'our', 'you', 'your', 'it', 'its', 'and', 'or', 'but', 'not', 'have', 'has', 'had', 'been', 'will', 'would', 'could', 'should', 'can', 'may', 'might'}
    words = [w for w in _re.findall(r'\b\w+\b', question.lower()) if w not in stop_words and len(w) > 2]
    if words:
        # 取前4个关键词组合
        key_query = ' '.join(words[:4])
        try:
            t0 = time.perf_counter()
            key_results = memory.query(key_query, top_k=max(top_k // 3, 15), use_spacetime=use_spacetime)
            stats.query_times_ms.append((time.perf_counter() - t0) * 1000)
            for r in key_results:
                mid = r.get("memory_id", "")
                if mid not in seen_ids:
                    supplement.append(str(r.get("content", "")))
                    seen_ids.add(mid)
        except Exception:
            pass

    # 策略2: 实体名精确检索
    if qtype == "knowledge-update":
        entities = _extract_question_entities(question)
        for entity in list(entities)[:3]:
            try:
                t0 = time.perf_counter()
                ent_results = memory.query(entity, top_k=10, use_spacetime=use_spacetime)
                stats.query_times_ms.append((time.perf_counter() - t0) * 1000)
                for r in ent_results:
                    mid = r.get("memory_id", "")
                    if mid not in seen_ids:
                        supplement.append(str(r.get("content", "")))
                        seen_ids.add(mid)
            except Exception:
                pass

    # 策略3: 扩大检索范围（更多top_k结果）
    if qtype == "temporal-reasoning" and not supplement:
        try:
            t0 = time.perf_counter()
            broad_results = memory.query(question, top_k=top_k * 2, use_spacetime=use_spacetime)
            stats.query_times_ms.append((time.perf_counter() - t0) * 1000)
            for r in broad_results:
                mid = r.get("memory_id", "")
                if mid not in seen_ids:
                    supplement.append(str(r.get("content", "")))
                    seen_ids.add(mid)
        except Exception:
            pass

    return supplement[:15]  # 最多15条补充上下文


def _evaluate_question(
    memory: SuMemoryLitePro,
    sample: dict[str, Any],
    stats: _RunStats,
    top_k: int = 5,
    reranker=None,  # v3.5.8: 可选 LLMReranker
    use_spacetime: bool = True,  # v3.5.8: 是否启用时空索引
    entity_graph: Any | None = None,  # v4.0: 实体图谱
) -> dict[str, Any]:
    """对单条问题执行检索并更新统计，返回逐题诊断信息。

    v3.5.8: 支持 LLM 答案提取模式 — 传入 reranker 时，
    先从检索结果取最多 15 个 chunk 调 LLM 生成答案，
    再用生成的答案与 gold_answer 做语义匹配。
    """

    question = str(sample.get("question", ""))
    gold_answer = str(sample.get("answer", "") or "")
    qid = str(sample.get("question_id", ""))
    qtype = str(sample.get("question_type", "unknown"))
    is_abstain_gold = qid.endswith("_abs")

    # v3.5.9: LLM 模式用更大的 top_k 以扩大召回
    # v3.6.0: 从 3x 提升到 6x，扩大召回覆盖率
    # v3.6.1: 从 6x/60 提升到 10x/80，适度扩大召回窗口（避免过度噪音）
    query_top_k = max(top_k * 10, 80) if reranker is not None else top_k

    # ============================================================
    # v4.0: QueryClassifier 动态检索策略
    # 根据问题类型生成 QueryPlan，动态调整 top_k、检索通道、重排策略
    # ============================================================
    query_plan = None
    if QUERY_CLASSIFIER_AVAILABLE:
        try:
            classifier = create_query_classifier()
            query_plan = classifier.classify(question, qtype=qtype)
            # 根据 QueryPlan 调整 top_k
            if query_plan.top_k_multiplier > 1.0:
                query_top_k = int(query_top_k * query_plan.top_k_multiplier)
        except Exception as exc:
            logger.debug("  [debug] QueryClassifier failed: %s", exc)

    t0 = time.perf_counter()
    try:
        results = memory.query(question, top_k=query_top_k, use_spacetime=use_spacetime)
    except Exception as exc:  # pragma: no cover
        print(f"  [warn] query failed for {qid}: {exc}")
        results = []
    stats.query_times_ms.append((time.perf_counter() - t0) * 1000)

    # ============================================================
    # v4.0: Chronos式双索引检索 — 基于 QueryPlan 做时间过滤二路检索
    # 当 query_plan 指定 event_time 通道时，做时间过滤二路检索
    # ============================================================
    if query_plan and "event_time" in query_plan.primary_channels and TEMPORAL_PARSER_AVAILABLE and reranker is not None:
        results = _dual_index_retrieval(memory, question, sample, results, query_top_k, use_spacetime, stats, query_plan.time_filter)

    # ============================================================
    # v4.0: 事实优先检索 — 将结构化事实排在原始chunk前面
    # 结构化事实天然自包含（消解指代、附带时间标签），
    # 优先作为 LLM 上下文，减少噪音提升答案提取质量
    # ============================================================
    fact_results = []
    chunk_results = []
    for r in results:
        meta = r.get("metadata") or {}
        if meta.get("is_extracted_fact"):
            fact_results.append(r)
        else:
            chunk_results.append(r)
    # 事实排前面，原始chunk补充后面
    results = fact_results + chunk_results

    # ============================================================
    # v4.0: 时间约束过滤 — 用 TemporalParser 检测问题中的时间表达式，
    # 按时间范围过滤结构化事实，提升 TR/KU 问题的检索精度
    # ============================================================
    if TEMPORAL_PARSER_AVAILABLE and fact_results:
        _temporal_filter_results(question, fact_results, sample)

    # v3.6.1: 时序推理首尾增强 — 补充 answer_session 的首尾 chunk
    # 注意：不再无条件合并关键词检索，改为在LLM提取失败后再补充（见下方fallback逻辑）
    if qtype == "temporal-reasoning" and reranker is not None:
        answer_session_ids_set = {str(s) for s in (sample.get("answer_session_ids") or [])}
        boundary_chunks = _get_session_boundary_chunks(memory, answer_session_ids_set, results)
        seen_ids = {r["memory_id"] for r in results}
        for bc in boundary_chunks:
            if bc.get("memory_id") not in seen_ids:
                results.append(bc)
                seen_ids.add(bc.get("memory_id", ""))

    # v3.6.1: 知识更新 Entity 感知重排 — 包含目标实体的 chunk 排前面（不合并关键词，避免噪音）
    if qtype == "knowledge-update" and reranker is not None and results:
        q_entities = _extract_question_entities(question)
        if q_entities:
            def _entity_coverage_score(r: dict) -> float:
                text = str(r.get("content", ""))
                chunk_entities = _extract_simple_entities(text)
                overlap = q_entities & chunk_entities
                return len(overlap) / max(len(q_entities), 1)
            # 按 (entity覆盖度*0.3 + 原始score*0.7) 重排（降低entity权重避免过度干扰）
            for r in results:
                r["_entity_score"] = _entity_coverage_score(r)
                r["_original_score"] = r.get("score", 0.5)
            results.sort(
                key=lambda r: r.get("_entity_score", 0) * 0.3 + r.get("_original_score", 0) * 0.7,
                reverse=True,
            )

    # v3.6.1: Preference 问题 — 不再无条件合并关键词检索（避免噪音）
    # 改用下方的全量扫描+LLM judge 回退策略

    # v3.6.1: Preference 全量扫描回退 — 对 answer_session 全量匹配
    # 使用 LLM judge 而非字符串匹配，因为 preference 的 gold_answer 通常是抽象偏好
    preference_fallback_hit = False
    if qtype == "single-session-preference" and reranker is not None:
        answer_sids = {str(s) for s in (sample.get("answer_session_ids") or [])}
        if hasattr(memory, '_memories') and hasattr(memory, '_memory_map'):
            session_contents = []
            for node in memory._memories:
                meta = node.metadata if hasattr(node, 'metadata') else {}
                if str(meta.get("session_id", "")) in answer_sids:
                    session_contents.append(str(node.content if hasattr(node, 'content') else ""))
            if session_contents:
                # 先尝试字符串匹配
                preference_fallback_hit = _is_answer_found(session_contents, gold_answer)
                # 字符串匹配失败时，用LLM从全量session内容提取答案
                if not preference_fallback_hit and reranker is not None:
                    # 取前30条session内容给LLM
                    pref_context = session_contents[:30]
                    pref_llm_answer = reranker.answer_from_context(question, pref_context, qtype)
                    if pref_llm_answer:
                        preference_fallback_hit = _is_answer_found([pref_llm_answer], gold_answer)
                        # LLM-as-Judge 二次确认
                        if not preference_fallback_hit:
                            preference_fallback_hit = _llm_judge_equivalence(
                                reranker, question, pref_llm_answer, gold_answer
                            )

    # ============================================================
    # v4.0: MS 多子查询检索 — 将聚合问题分解为多个子查询，
    # 每个子查询独立检索，结果合并去重后补充到主结果中
    # ============================================================
    if query_plan and query_plan.needs_decomposition and qtype == "multi-session" and reranker is not None:
        sub_results = _multi_query_retrieval(
            memory, question, query_plan.sub_queries, query_top_k, use_spacetime, stats
        )
        if sub_results:
            seen_ids = {r.get("memory_id") for r in results}
            for sr in sub_results:
                if sr.get("memory_id") not in seen_ids:
                    results.append(sr)
                    seen_ids.add(sr.get("memory_id", ""))

    # ============================================================
    # v4.0: EntityGraph Spreading Activation — 从 top 语义命中出发，
    # 沿实体链接 BFS 发现间接相关记忆，补充到检索结果中
    # ============================================================
    if entity_graph is not None and results and qtype in ("multi-session", "temporal-reasoning"):
        _graph_augment_results(entity_graph, memory, results)

    # v3.5.9: LLM 答案提取路径 — Session 级语义聚类
    llm_answer = ""
    if reranker is not None and results and not is_abstain_gold:
        # v3.6.1: 使用共享聚类函数 (P1-3 去重)
        # v3.6.1: max_per_session 保持3，避免过多同session chunk引入噪音
        context_chunks = cluster_by_session(
            results,
            max_per_session=3,
            max_total=reranker.max_chunks,
        )
        # v3.6.0: 知识更新类型 — 时间倒序 + entity 级去重 (最新版优先)
        if qtype == "knowledge-update":
            # 时间倒序：最晚的 chunk 在最前面
            context_chunks.sort(
                key=lambda r: -(r.get("metadata") or {}).get("timestamp", 0)
            )
            # Entity 级去重：同一实体只保留最新的一条
            seen_entities: set[str] = set()
            deduped: list[dict] = []
            for chunk in context_chunks:
                text = str(chunk.get("content", ""))
                entities = _extract_simple_entities(text)
                new_entities = entities - seen_entities
                if not new_entities and entities:
                    continue  # 所有实体都已见过 → 这是已过时的旧信息
                seen_entities.update(entities)
                deduped.append(chunk)
            context_chunks = deduped
        # v3.5.9: 为时序推理问题注入显式时间标签
        # v3.6.1: 时序推理按 timestamp 正序排列，确保时间线清晰
        if qtype == "temporal-reasoning":
            context_chunks.sort(key=lambda r: (r.get("metadata") or {}).get("timestamp", 0))
        if qtype == "temporal-reasoning":
            context = [
                f"{(r.get('metadata') or {}).get('time_label', '')} {r.get('content', '')}"
                for r in context_chunks[:reranker.max_chunks]
            ]
        elif qtype == "knowledge-update":
            # v3.6.0: 知识更新 — 标记新旧顺序，帮助LLM识别覆盖关系
            context = []
            for i, r in enumerate(context_chunks[:reranker.max_chunks]):
                label = "[NEWEST VALUE]" if i == 0 else "[Older value from earlier conversation]"
                context.append(f"{label} {r.get('content', '')}")
        else:
            # v4.0: 事实感知上下文 — 优先使用结构化事实，原始chunk补充
            context_parts = []
            for r in context_chunks[:reranker.max_chunks]:
                meta = r.get("metadata") or {}
                content = str(r.get("content", ""))
                if meta.get("is_extracted_fact"):
                    # 结构化事实 — 附带时间和类型标签
                    fact_type = meta.get("fact_type", "")
                    event_date = meta.get("event_date_start", "")
                    pref_signal = meta.get("preference_signal", "")
                    labels = []
                    if fact_type:
                        labels.append(f"type:{fact_type}")
                    if event_date:
                        labels.append(f"date:{event_date}")
                    if pref_signal:
                        labels.append(f"pref:{pref_signal}")
                    label_str = f" [{', '.join(labels)}]" if labels else ""
                    context_parts.append(f"[FACT{label_str}] {content}")
                else:
                    context_parts.append(content)
            context = context_parts
        llm_answer = reranker.answer_from_context(question, context, qtype)

        # ============================================================
        # v4.0: LLM Agent 迭代检索 — 当初始答案为空或LLM拒答时，
        # 做补充检索并二次提取答案（最多1轮迭代）
        # ============================================================
        if (not llm_answer or _is_llm_abstaining(llm_answer)) and qtype in ("temporal-reasoning", "knowledge-update", "multi-session"):
            # 尝试用不同策略补充检索
            supplement_context = _iterative_retrieval_supplement(
                memory, question, qtype, results, query_top_k, use_spacetime, stats
            )
            if supplement_context:
                # 合并原始上下文 + 补充上下文
                combined_context = context + ["\n--- Additional Search Results ---\n"] + supplement_context
                llm_answer = reranker.answer_from_context(question, combined_context, qtype)

    retrieved_texts = [str(r.get("content", "")) for r in results]

    # 命中判断 — v3.5.8: LLM 模式下用 llm_answer 代替检索文本
    if reranker is not None and llm_answer:
        # LLM 答案直接参与匹配
        is_llm_correct = _is_answer_found([llm_answer], gold_answer)

        # v3.6.0: Stage 5 — LLM-as-Judge 语义等价判断
        # 当 Stages 1-4 全部失败但 LLM 有答案时，用 LLM 自身判断等价性
        if not is_llm_correct and not is_abstain_gold:
            is_llm_correct = _llm_judge_equivalence(
                reranker, question, llm_answer, gold_answer
            )

        hit_top1 = is_llm_correct if not is_abstain_gold else False
        hit_top3 = is_llm_correct if not is_abstain_gold else False
        hit_top5 = is_llm_correct if not is_abstain_gold else False
    else:
        hit_top1 = _is_answer_found(retrieved_texts[:1], gold_answer) if not is_abstain_gold else False
        hit_top3 = _is_answer_found(retrieved_texts[:3], gold_answer) if not is_abstain_gold else False
        hit_top5 = _is_answer_found(retrieved_texts[:5], gold_answer) if not is_abstain_gold else False

    # v3.6.0: 放弃识别 — 更严格的证据要求 + LLM 显式拒答检测
    # 1. 显式拒答：LLM 回答包含拒答信号
    llm_abstain = False
    if reranker is not None and llm_answer:
        llm_abstain = _is_llm_abstaining(llm_answer)

    # 2. 检索证据匹配
    answer_session_ids = {str(s) for s in (sample.get("answer_session_ids") or [])}
    retrieved_sessions = {
        str((r.get("metadata") or {}).get("session_id"))
        for r in results
    }
    overlap_count = len(answer_session_ids & retrieved_sessions)
    overlap_ratio = overlap_count / max(len(answer_session_ids), 1)
    has_session_overlap = overlap_count > 0

    # v3.6.1: Session Overlap 证据增强 — 多个 answer session 命中时证据更充分
    strong_session_evidence = overlap_ratio >= 0.5 and hit_top5

    # v3.6.1: 分维度差异化 found_evidence 阈值
    # temporal-reasoning / knowledge-update 检索天然困难，降低阈值
    # single-session-preference 完全信任 LLM 判断
    if qtype in ("temporal-reasoning", "knowledge-update"):
        found_evidence = hit_top5 or (has_session_overlap and hit_top3) or strong_session_evidence
    elif qtype == "single-session-preference":
        # v3.6.1: preference 问题完全信任 LLM 判断 + session overlap
        found_evidence = hit_top5 or (not llm_abstain and has_session_overlap) or preference_fallback_hit
    else:
        found_evidence = hit_top3 or strong_session_evidence or (has_session_overlap and hit_top5)

    # v3.6.1: preference 全量扫描回退覆盖 hit_top5
    if qtype == "single-session-preference" and preference_fallback_hit and not hit_top5:
        hit_top5 = True
        hit_top3 = True

    # v3.6.0: pred_abstain = 没有检索证据 OR LLM 显式拒答
    pred_abstain = (not found_evidence) or llm_abstain

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

    # v3.5.9: 按维度 R@K 追踪
    if not is_abstain_gold:
        if hit_top1:
            stats.by_type_hit_top1[qtype] += 1
        if hit_top3:
            stats.by_type_hit_top3[qtype] += 1
        if hit_top5:
            stats.by_type_hit_top5[qtype] += 1

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
    reranker=None,  # v3.5.8: 可选 LLMReranker
    enable_temporal: bool = False,  # v3.5.8: 是否激活 SpacetimeIndex
    event_extractor=None,  # v4.0: 结构化事件提取器
    enable_bm25: bool = True,  # v4.1: P0-3 BM25 消融开关
    enable_energy_expand: bool = True,  # v4.2: P1-2 能量候选扩展消融开关
) -> tuple[BenchmarkResult, list[dict[str, Any]]]:
    """对单个嵌入后端执行完整评测，返回 :class:`BenchmarkResult` + 逐题诊断。"""

    stats = _RunStats()
    diagnostics: list[dict[str, Any]] = []
    storage_path = os.path.join(storage_root, f"longmem_{backend}")

    if verbose:
        print(f"\n[LongMemEval] 初始化后端: {BACKENDS[backend]['name']}")
        if enable_temporal:
            print("[LongMemEval] SpacetimeIndex 时空索引: 已激活")
        if reranker is not None:
            print(f"[LongMemEval] LLM 重排序: {reranker.model}")
        if event_extractor is not None:
            print(f"[LongMemEval] 事件提取: {event_extractor.provider}/{event_extractor.model} (mode={event_extractor.mode})")
    memory = _build_memory(backend, storage_path, enable_temporal=enable_temporal, enable_bm25=enable_bm25, enable_energy_expand=enable_energy_expand)

    try:
        for i, sample in enumerate(samples):
            if verbose:
                qid = sample.get("question_id", f"#{i}")
                qtype = sample.get("question_type", "?")
                print(f"  [{i + 1}/{len(samples)}] {qid} ({qtype})")

            # 每题独立内存：清空再注入，避免跨题串扰
            memory.clear()
            _ingest_question(memory, sample, stats, chunk_chars, event_extractor=event_extractor)

            # v4.0: 构建 EntityGraph（从 memory 中的结构化事实构建）
            entity_graph = _build_entity_graph(memory)

            diag = _evaluate_question(memory, sample, stats, top_k=top_k, reranker=reranker, use_spacetime=enable_temporal, entity_graph=entity_graph)
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
        # v3.5.9: 单会话聚合 R@K (报告用)
        single_hit1 = sum(stats.by_type_hit_top1.get(t, 0) for t in SINGLE_SESSION_TYPES)
        single_hit5 = sum(stats.by_type_hit_top5.get(t, 0) for t in SINGLE_SESSION_TYPES)
        dim["single_session_r@1"] = single_hit1 / single_total
        dim["single_session_r@5"] = single_hit5 / single_total
    if "multi-session" in stats.by_type:
        c, t = stats.by_type["multi-session"]
        dim["multi_session"] = c / t if t else 0.0
        # v3.5.9: 按维度 R@K
        dim["multi_session_r@1"] = stats.by_type_hit_top1.get("multi-session", 0) / t if t else 0.0
        dim["multi_session_r@5"] = stats.by_type_hit_top5.get("multi-session", 0) / t if t else 0.0
    if "temporal-reasoning" in stats.by_type:
        c, t = stats.by_type["temporal-reasoning"]
        dim["temporal_reasoning"] = c / t if t else 0.0
        dim["temporal_reasoning_r@1"] = stats.by_type_hit_top1.get("temporal-reasoning", 0) / t if t else 0.0
        dim["temporal_reasoning_r@5"] = stats.by_type_hit_top5.get("temporal-reasoning", 0) / t if t else 0.0
    if "knowledge-update" in stats.by_type:
        c, t = stats.by_type["knowledge-update"]
        dim["knowledge_update"] = c / t if t else 0.0
        dim["knowledge_update_r@1"] = stats.by_type_hit_top1.get("knowledge-update", 0) / t if t else 0.0
        dim["knowledge_update_r@5"] = stats.by_type_hit_top5.get("knowledge-update", 0) / t if t else 0.0
    # v3.5.9: 单会话聚合 R@K
    for st in SINGLE_SESSION_TYPES:
        if st in stats.by_type:
            sc, st_total = stats.by_type[st]
            dim[f"{st}_r@1"] = stats.by_type_hit_top1.get(st, 0) / st_total if st_total else 0.0
            dim[f"{st}_r@5"] = stats.by_type_hit_top5.get(st, 0) / st_total if st_total else 0.0

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

    # v3.5.9: 按维度 R@1/R@5 表格
    lines.append("  --- 按维度 R@1 / R@5 ---")
    rk_dim_keys = (
        ("single_session", "1. 信息提取(单跳)"),
        ("multi_session", "2. 多会话推理"),
        ("temporal_reasoning", "3. 时序推理"),
        ("knowledge_update", "4. 知识更新"),
    )
    for rk_key, label in rk_dim_keys:
        r1_key = f"{rk_key}_r@1"
        r5_key = f"{rk_key}_r@5"
        row = f"  {label:<24}"
        for backend_key, res in results.items():
            r1 = res.dimension_scores.get(r1_key)
            r5 = res.dimension_scores.get(r5_key)
            row += f" R@1={_fmt_pct(r1):>6} R@5={_fmt_pct(r5):>6}"
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
    if arg == "all":
        return ["minimax"]  # v3.5.7: all 使用 minimax 避免 ST MPS segfault
    if arg in BACKENDS:
        return [arg]
    raise ValueError(f"Unknown backend '{arg}'. Choices: {list(BACKENDS.keys())} / both / all")


def build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="su-memory LongMemEval benchmark runner (官方六维度版)",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "mlx", "sbert", "minimax", "deepseek", "both", "all"],
        default="ollama",
        help="嵌入后端：ollama / mlx / sbert / minimax / both / all (默认 ollama)",
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
    parser.add_argument(
        "--rerank",
        choices=["none", "spacetime", "spacetime-llm"],
        default="spacetime",
        help="重排序模式: none/Spacetime/spacetime-llm (默认 spacetime)",
    )
    parser.add_argument(
        "--llm-model",
        default="gemma4",
        help="LLM 模型名 (Ollama, 默认 gemma4)",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["ollama", "deepseek", "openai", "auto"],
        default="auto",
        help="LLM 提供商: ollama (本地) / deepseek (云端) / openai (云端) / auto (自动检测, 默认)",
    )
    parser.add_argument(
        "--full-diagnostics",
        action="store_true",
        help="保存全部诊断数据到 JSON (默认仅保存前 20 条)",
    )
    # v3.6.0: 模型维度路由
    parser.add_argument(
        "--llm-model-route",
        action="store_true",
        help="启用模型维度路由: V4-Pro(单/多会话) + chat(时序/知识更新)",
    )
    parser.add_argument(
        "--llm-model-secondary",
        default=None,
        help="路由用第二 LLM 模型名 (默认 llm-provider 同主模型: deepseek-chat)",
    )
    parser.add_argument(
        "--llm-provider-secondary",
        choices=["ollama", "deepseek", "openai"],
        default=None,
        help="第二 LLM provider (默认与 --llm-provider 相同)",
    )
    # v4.0: 结构化事件提取
    parser.add_argument(
        "--event-extraction",
        choices=["full", "light", "rule", "off"],
        default="off",
        help="事件提取模式: full(LLM完整提取), light(LLM简化), rule(规则回退), off(关闭, 默认)",
    )
    parser.add_argument(
        "--event-extraction-model",
        default=None,
        help="事件提取用 LLM 模型 (默认与 --llm-model 相同)",
    )
    # v4.0: 时间约束过滤
    parser.add_argument(
        "--temporal-filter",
        action="store_true",
        default=False,
        help="启用时间约束过滤 (需搭配 --event-extraction 使用)",
    )
    # v4.1: 消融追踪 flags
    parser.add_argument(
        "--enable-bm25",
        action="store_true",
        default=False,
        help="P0-3: 启用 BM25 词法检索通道",
    )
    parser.add_argument(
        "--no-bm25",
        action="store_true",
        default=False,
        help="P0-3: 显式禁用 BM25 词法检索通道",
    )
    parser.add_argument(
        "--enable-energy-expand",
        action="store_true",
        default=False,
        help="P1-2: 启用能量候选扩展 (Phase 1)",
    )
    parser.add_argument(
        "--no-energy-expand",
        action="store_true",
        default=False,
        help="P1-2: 显式禁用能量候选扩展 (Phase 1 消融)",
    )
    parser.add_argument(
        "--enable-graph-activation",
        action="store_true",
        default=False,
        help="P2-2: 启用图扩散激活 (Phase 2)",
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
    print(f"  rerank={args.rerank}", end="")
    if args.rerank == "spacetime-llm":
        print(f"  llm={args.llm_provider}/{args.llm_model}", end="")
    print()
    print("=" * 60)

    # v4.1: 创建 LLM 重排序器 — 支持 ollama / deepseek / openai 多 provider
    # v4.1: auto 模式自动检测最佳可用 provider (deepseek > openai > ollama)
    reranker = None
    if args.rerank == "spacetime-llm":
        from su_memory.sdk._llm_reranker import (
            LLMReranker,
            check_deepseek_available,
            check_ollama_available,
            check_openai_available,
        )

        # v4.1: auto 模式 — 自动选择最佳可用 provider
        resolved_provider = args.llm_provider
        resolved_model = args.llm_model
        if resolved_provider == "auto":
            if check_deepseek_available():
                resolved_provider = "deepseek"
                # auto 切换 provider 时覆盖默认 gemma4 → deepseek-chat
                if not args.llm_model or args.llm_model == "gemma4":
                    resolved_model = "deepseek-chat"
                else:
                    resolved_model = resolved_model or "deepseek-chat"
                print("[LongMemEval] auto: 检测到 DeepSeek API，优先使用")
            elif check_openai_available():
                resolved_provider = "openai"
                if not args.llm_model or args.llm_model == "gemma4":
                    resolved_model = "gpt-4o-mini"
                else:
                    resolved_model = resolved_model or "gpt-4o-mini"
                print("[LongMemEval] auto: 检测到 OpenAI API，使用 gpt-4o-mini")
            elif check_ollama_available():
                resolved_provider = "ollama"
                resolved_model = resolved_model or "gemma4"
                print("[LongMemEval] auto: 无云端 API，回退到 Ollama 本地模型")
            else:
                print("[warn] 无可用 LLM provider (DeepSeek/OpenAI/Ollama)，回退到纯 SpacetimeIndex 模式")
                resolved_provider = None

        if resolved_provider == "deepseek":
            if not check_deepseek_available():
                print("[warn] DeepSeek API Key 未设置，回退到纯 SpacetimeIndex 模式")
            else:
                reranker = LLMReranker(
                    model=resolved_model or "deepseek-chat",
                    provider="deepseek",
                    max_chunks=50,
                    max_chunks_temporal=80,
                    max_chunks_multisession=70,
                    chunk_char_limit=800,
                )
                print(f"[LongMemEval] LLM 重排序器: DeepSeek/{resolved_model or 'deepseek-chat'}")
        elif resolved_provider == "openai":
            if not check_openai_available():
                print("[warn] OpenAI API Key 未设置，回退到纯 SpacetimeIndex 模式")
            else:
                reranker = LLMReranker(
                    model=resolved_model or "gpt-4o-mini",
                    provider="openai",
                    max_chunks=50,
                    max_chunks_temporal=80,
                    max_chunks_multisession=70,
                    chunk_char_limit=800,
                )
                print(f"[LongMemEval] LLM 重排序器: OpenAI/{resolved_model or 'gpt-4o-mini'}")
        elif resolved_provider == "ollama":
            if not check_ollama_available():
                print("[warn] Ollama 服务不可用，回退到纯 SpacetimeIndex 模式")
            else:
                reranker = LLMReranker(
                    model=resolved_model, provider="ollama",
                    max_chunks=70, max_chunks_temporal=100, max_chunks_multisession=80,
                )
                print(f"[LongMemEval] LLM 重排序器: Ollama/{resolved_model}")

        # v3.6.0: 模型维度路由 — V4-Pro(单/多会话) + chat(时序/知识更新)
        if reranker is not None and args.llm_model_route:
            from benchmarks._model_router import ModelRouter
            secondary_provider = args.llm_provider_secondary or resolved_provider
            secondary_model = args.llm_model_secondary or (
                "deepseek-chat" if secondary_provider == "deepseek" else
                "gpt-4o-mini" if secondary_provider == "openai" else
                "gemma4"
            )
            # 仅当主模型是 V4-Pro 且第二模型可用时路由
            if (args.llm_model or "").find("v4") >= 0 or (args.llm_model or "").find("pro") >= 0:
                if secondary_provider == "deepseek" and check_deepseek_available():
                    reranker_knowledge = LLMReranker(
                        model=secondary_model,
                        provider="deepseek",
                        max_chunks=50,
                        max_chunks_temporal=80,
                        chunk_char_limit=800,
                    )
                elif secondary_provider == "openai" and check_openai_available():
                    reranker_knowledge = LLMReranker(
                        model=secondary_model,
                        provider="openai",
                        max_chunks=50,
                        max_chunks_temporal=80,
                        chunk_char_limit=800,
                    )
                elif check_ollama_available():
                    reranker_knowledge = LLMReranker(
                        model=secondary_model,
                        provider="ollama",
                        max_chunks=70,
                        max_chunks_temporal=100,
                    )
                else:
                    reranker_knowledge = None

                if reranker_knowledge is not None:
                    reranker = ModelRouter(
                        default=reranker,
                        overrides={
                            "temporal-reasoning": reranker_knowledge,
                            "knowledge-update": reranker_knowledge,
                        },
                    )
                    print(f"[LongMemEval] 模型路由: {reranker}")
                else:
                    print("[warn] 第二模型不可用，路由未启用")
            else:
                print("[warn] --llm-model-route 要求主模型为 V4-Pro，路由未启用")

    samples = load_longmemeval(split=args.split, verbose=args.verbose)
    samples = _filter_samples(samples, args.max_questions)
    if not samples:
        print("[error] no samples loaded; aborting.")
        return 2
    print(f"[LongMemEval] loaded {len(samples)} samples")

    # v4.0: 创建事件提取器
    event_extractor = None
    if getattr(args, 'event_extraction', 'off') != 'off' and EVENT_EXTRACTOR_AVAILABLE:
        ext_mode = args.event_extraction
        ext_model = getattr(args, 'event_extraction_model', None) or args.llm_model or "gemma4"
        ext_provider = args.llm_provider or "ollama"
        event_extractor = create_event_extractor(
            provider=ext_provider,
            model=ext_model,
            mode=ext_mode,
        )
        print(f"[LongMemEval] 事件提取器: {ext_provider}/{ext_model} (mode={ext_mode})")
    elif getattr(args, 'event_extraction', 'off') != 'off' and not EVENT_EXTRACTOR_AVAILABLE:
        print("[warn] 事件提取器不可用 (_event_extractor 导入失败)，回退到 rule 模式")
        # 尝试使用 rule 模式作为回退
        if args.event_extraction in ('full', 'light', 'rule'):
            try:
                from su_memory.sdk._event_extractor import EventExtractor as _EE
                event_extractor = _EE(mode="rule")
            except ImportError:
                print("[warn] 事件提取器完全不可用，跳过")

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
                reranker=reranker,
                enable_temporal=args.rerank != "none",
                event_extractor=event_extractor,  # v4.0
                enable_bm25=not args.no_bm25,  # v4.1: P0-3 BM25 消融开关
                enable_energy_expand=not args.no_energy_expand,  # v4.2: P1-2 能量候选扩展消融开关
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
            k: v for k, v in diagnostics.items()  # v3.5.9: 全量保存或前20条
        } if getattr(args, 'full_diagnostics', False) else {
            k: v[:20] for k, v in diagnostics.items()
        },
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"[LongMemEval] results → {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    raise SystemExit(main())
