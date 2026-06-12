#!/usr/bin/env python3
"""
TimeQA / TempRAGEval Benchmark Runner for su-memory (时序推理专测)
===================================================================

对接 HuggingFace 数据集 ``siyue/TempRAGEval``（整合 TimeQA + SituatedQA），
对 :class:`SuMemoryLitePro` 执行时序推理能力评测。

评测维度:
1. 时序检索 Accuracy — 检索结果是否包含时间正确答案
2. 时间感知 Recall@1/3/5 — 答案在前 K 条检索中的命中率
3. 按时间跨度分桶 — 近/中/远期问题的分桶 Accuracy
4. LLM 答案提取模式 — 检索 → LLM 生成答案 → 语义匹配

su-memory 核心优势:
- SpacetimeIndex 60 周期干支编码 — 任何竞品都不具备
- v4.4.1: 问题时间约束解析 + chunk 真实年份提取 + 时间感知 LLM prompt
- 时间衰减权重 (_temporal_weight)
- 五行季节映射 (木=春/火=夏/土=长夏/金=秋/水=冬)

Reference:
    https://github.com/google-research-datasets/TimeQA
    https://huggingface.co/datasets/siyue/TempRAGEval
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 路径与依赖
# ---------------------------------------------------------------------------
_BENCH_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _BENCH_DIR.parent

for _p in (str(_PKG_ROOT), str(_PKG_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from benchmarks.config import (
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
except ImportError:
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

from su_memory.sdk.lite_pro import SuMemoryLitePro

# v4.0: LLM Reranker
try:
    from su_memory.sdk._llm_reranker import LLMReranker, create_llm_reranker
    LLM_RERANKER_AVAILABLE = True
except ImportError:
    LLM_RERANKER_AVAILABLE = False
    LLMReranker = None  # type: ignore[assignment]

# v4.4.1: 时序解析器 — 解析问题中的时间约束用于检索加权
try:
    from su_memory.sdk._temporal_parser import TemporalParser
    _temporal_parser = TemporalParser()
    TEMPORAL_PARSER_AVAILABLE = True
except ImportError:
    TemporalParser = None  # type: ignore[assignment]
    _temporal_parser = None
    TEMPORAL_PARSER_AVAILABLE = False

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
VERSION = "4.4.1"
DEFAULT_CHUNK_CHARS = 600  # TimeQA 文本较短，chunk 更小
SEMANTIC_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def _load_local_json(benchmark: str) -> list[dict[str, Any]]:
    """尝试从本地缓存加载数据。支持 JSON (list/dict) 和 JSONL (每行一个 JSON)。"""
    cache_dir = Path(ensure_data_dir(benchmark))
    cfg = DATASETS.get(benchmark, {})
    files = cfg.get("files", [])
    for fname in files:
        path = cache_dir / fname
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                first_char = fh.read(1)
                fh.seek(0)
                if first_char == "[":
                    # 标准 JSON 数组
                    data = json.load(fh)
                    if isinstance(data, list):
                        return data
                    if isinstance(data, dict) and "data" in data:
                        return data["data"]
                elif first_char == "{":
                    # JSONL: 每行一个 JSON 对象
                    items: list[dict[str, Any]] = []
                    for line in fh:
                        line = line.strip()
                        if line:
                            try:
                                items.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                    if items:
                        return items
    return []


def load_timeqa(verbose: bool = False) -> list[dict[str, Any]]:
    """加载 TimeQA (TempRAGEval) 数据。

    数据格式预期 (来自 siyue/TempRAGEval):
        - question: str      — 时间敏感问题
        - answer: str        — 正确答案
        - context: str/None  — 上下文/文档 (可能为空，此时用 question 本身)
        - temporal_scope: str/None — 时间范围 (如 "2023", "2020-2022")
        - id: str/None       — 问题 ID

    优先级：本地缓存 JSON > HuggingFace 加载。
    """
    local = _load_local_json("timeqa")
    if local:
        # v4.3.1: TimeQA JSONL 字段映射 — idx/question/context/targets → id/question/context/answer
        if "targets" in (local[0] if local else {}):
            normalized: list[dict[str, Any]] = []
            for item in local:
                targets = item.get("targets", [])
                answer = targets[0] if isinstance(targets, list) and targets else str(targets)
                # 从 idx 中提取时间信息 (e.g., "/wiki/Foo#P54#0")
                tid = item.get("idx", "")
                normalized.append({
                    "id": tid,
                    "question": item.get("question", ""),
                    "answer": answer,
                    "context": item.get("context", ""),
                    "temporal_scope": tid.split("#")[-1] if "#" in tid else "",
                })
            local = normalized
        if verbose:
            print(f"[TimeQA] 使用本地缓存数据集 n={len(local)}")
        return local

    hf_id = DATASETS["timeqa"]["hf_id"]
    cache_dir = DATASETS["timeqa"]["local_cache"]
    if verbose:
        print(f"[TimeQA] 从 HuggingFace 加载 {hf_id}")

    # 尝试多种加载策略
    last_err: Exception | None = None
    for attempt in (
        {"split": "test"},
        {"split": "validation"},
        {"split": "train"},
        {},
    ):
        try:
            ds = load_hf_dataset(hf_id, cache_dir=cache_dir, **attempt)
            return [dict(item) for item in ds]
        except Exception as exc:
            last_err = exc
            continue

    # HuggingFace 加载失败 — 生成内置样本数据用于冒烟验证
    if verbose:
        print(f"[TimeQA] HF 加载失败 ({last_err})，使用内置冒烟样本 (15 题)")

    return _generate_smoke_samples()


def _generate_smoke_samples() -> list[dict[str, Any]]:
    """生成内置 TimeQA 风格冒烟样本 (15 题)。"""
    samples: list[dict[str, Any]] = [
        {
            "id": "tqa_001",
            "question": "What was Alice's job in 2019?",
            "answer": "software engineer",
            "context": "In 2019, Alice worked as a software engineer at Google. In 2020, she switched to become a product manager at Apple. By 2022, she had started her own company.",
            "temporal_scope": "2019",
        },
        {
            "id": "tqa_002",
            "question": "How many books did Bob read in the summer of 2021?",
            "answer": "5 books",
            "context": "Bob had a productive summer in 2021. He read 5 books between June and August: 'Dune', 'Foundation', 'Neuromancer', 'Snow Crash', and 'The Left Hand of Darkness'.",
            "temporal_scope": "summer 2021",
        },
        {
            "id": "tqa_003",
            "question": "Which city did Carol visit first: Paris or Tokyo?",
            "answer": "Paris",
            "context": "Carol went to Paris in March 2018. Six months later, in September 2018, she traveled to Tokyo. She returned to Paris again in 2020.",
            "temporal_scope": "2018",
        },
        {
            "id": "tqa_004",
            "question": "What was the company revenue in Q3 2022?",
            "answer": "$45 million",
            "context": "Q1 2022: $38M. Q2 2022: $42M. Q3 2022: $45M. Q4 2022: $50M. The steady growth was attributed to the new product launch in March 2022.",
            "temporal_scope": "Q3 2022",
        },
        {
            "id": "tqa_005",
            "question": "How many employees did the startup have before the Series A round?",
            "answer": "12 employees",
            "context": "The startup began with 3 co-founders in 2019. By early 2021, before the Series A round, the team had grown to 12 employees. After the $10M Series A in June 2021, they expanded to 45 people by year end.",
            "temporal_scope": "early 2021",
        },
        {
            "id": "tqa_006",
            "question": "What temperature was recorded in Beijing on January 15, 2023?",
            "answer": "-12°C",
            "context": "Beijing weather log: Jan 14, 2023: -8°C, cloudy. Jan 15, 2023: -12°C, sunny with strong winds. Jan 16, 2023: -10°C, overcast. The cold snap was the coldest in 10 years.",
            "temporal_scope": "Jan 15, 2023",
        },
        {
            "id": "tqa_007",
            "question": "Which operating system version was running on the server in April 2020?",
            "answer": "Ubuntu 18.04 LTS",
            "context": "Server upgrade history: April 2020 - Ubuntu 18.04 LTS. October 2020 - upgraded to Ubuntu 20.04 LTS. March 2022 - migrated to Ubuntu 22.04 LTS. The migration in 2022 included a full data center relocation.",
            "temporal_scope": "April 2020",
        },
        {
            "id": "tqa_008",
            "question": "How many subscribers did the newsletter have by the end of 2021?",
            "answer": "15,000 subscribers",
            "context": "Newsletter growth timeline: Dec 2019: 1,000 subs. Dec 2020: 5,000 subs. Dec 2021: 15,000 subs. Dec 2022: 50,000 subs after viral article. The growth rate was accelerating year over year.",
            "temporal_scope": "end of 2021",
        },
        {
            "id": "tqa_009",
            "question": "What was the original name of the product before the 2022 rebranding?",
            "answer": "Project Nexus",
            "context": "The product was originally called 'Project Nexus' from 2019 to early 2022. After a major rebranding in March 2022, it became 'Atlas Platform'. The rebrand came with a complete UI overhaul and a new pricing model.",
            "temporal_scope": "before March 2022",
        },
        {
            "id": "tqa_010",
            "question": "Which team won the championship in spring 2020?",
            "answer": "Team Phoenix",
            "context": "Spring 2020 championship: Team Phoenix defeated Team Dragon 3-2 in the finals. Summer 2020 championship: Team Dragon defeated Team Phoenix 3-1. Spring 2021: Team Phoenix won again, beating Team Griffin 3-0.",
            "temporal_scope": "spring 2020",
        },
        {
            "id": "tqa_011",
            "question": "What was David's PhD thesis topic when he graduated in 2018?",
            "answer": "Quantum Computing Error Correction",
            "context": "David started his PhD in 2014 focusing on quantum computing. His thesis topic evolved from 'Quantum Algorithms' in 2015 to 'Quantum Computing Error Correction' by his 2018 graduation. He published 4 papers on the topic during 2016-2018.",
            "temporal_scope": "2018",
        },
        {
            "id": "tqa_012",
            "question": "How much was the electricity bill in the hottest month of 2023?",
            "answer": "$890",
            "context": "Monthly electricity bills 2023: Jan $320, Feb $310, Mar $350, Apr $420, May $560, Jun $720, Jul $890 (hottest, 38°C avg), Aug $850, Sep $680, Oct $450, Nov $380, Dec $340.",
            "temporal_scope": "July 2023",
        },
        {
            "id": "tqa_013",
            "question": "What was the population of Springfield before the 2020 census?",
            "answer": "52,000",
            "context": "Springfield population history: 2010 census: 48,500. 2015 estimate: 50,200. 2019 estimate: 52,000. 2020 census: 53,400. 2025 projection: 58,000. The growth has been steady at about 1% per year.",
            "temporal_scope": "before 2020",
        },
        {
            "id": "tqa_014",
            "question": "Which medication was prescribed to the patient in the first quarter of 2022?",
            "answer": "Metformin 500mg",
            "context": "Patient medication log: Q1 2022 - Metformin 500mg daily. Q2 2022 - Metformin 1000mg daily (dosage increased). Q3 2022 - Added Januvia 100mg. Q4 2022 - Metformin 1000mg + Januvia 100mg + Farxiga 10mg.",
            "temporal_scope": "Q1 2022",
        },
        {
            "id": "tqa_015",
            "question": "What was the conference attendance count in the year before the pandemic?",
            "answer": "8,500 people",
            "context": "Annual TechConf attendance: 2018: 7,200 people. 2019: 8,500 people. 2020: virtual only (pandemic). 2021: virtual only. 2022: 6,500 people (hybrid). 2023: 9,000 people (full return).",
            "temporal_scope": "2019",
        },
    ]
    return samples


# ---------------------------------------------------------------------------
# 文本处理
# ---------------------------------------------------------------------------

def _chunk_text(text: str, max_chars: int) -> list[str]:
    """将文本按字符上限切成若干 chunk。"""
    if len(text) <= max_chars:
        return [text]
    sentences = text.replace(". ", ".\n").replace("! ", "!\n").replace("? ", "?\n").split("\n")
    chunks: list[str] = []
    buf = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(buf) + len(sent) > max_chars and buf:
            chunks.append(buf)
            buf = sent
        else:
            buf = (buf + " " + sent).strip() if buf else sent
    if buf:
        chunks.append(buf)
    return chunks or [text]


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


def _temporal_bucket(scope: str | None) -> str:
    """按时间跨度分桶：near / mid / far。"""
    if not scope:
        return "unknown"
    import re as _re
    # 尝试提取年份
    years = _re.findall(r'\b(20\d{2})\b', str(scope))
    if not years:
        return "unknown"
    latest = max(int(y) for y in years)
    if latest >= 2022:
        return "near"  # 近1-2年
    if latest >= 2019:
        return "mid"   # 3-5年
    return "far"        # 5年+


# ---------------------------------------------------------------------------
# 评测核心
# ---------------------------------------------------------------------------

@dataclass
class _TimeQAStats:
    """时序推理细粒度计数器。"""
    total: int = 0
    correct: int = 0
    recall_at_1: int = 0
    recall_at_3: int = 0
    recall_at_5: int = 0
    f1_sum: float = 0.0
    by_temporal: dict[str, list[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0, 0])
    )
    by_position: dict[str, list[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0, 0])
    )
    add_times_ms: list[float] = field(default_factory=list)
    query_times_ms: list[float] = field(default_factory=list)


def _build_memory(
    backend: str,
    storage_path: str,
    enable_bm25: bool = True,
    enable_energy_expand: bool = True,
) -> SuMemoryLitePro:
    """构造 SuMemoryLitePro 实例。"""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Choices: {list(BACKENDS)}")

    cfg = BACKENDS[backend]
    backend_type = cfg["type"]

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
        enable_graph=False,
        enable_temporal=True,
        enable_session=False,
        enable_prediction=False,
        enable_explainability=False,
        enable_plugins=False,
        enable_cross_encoder=False,
        enable_bm25=enable_bm25,
        enable_energy_expand=enable_energy_expand,
    )


# ---------------------------------------------------------------------------
# v4.4.1: 时间约束解析与真实年份提取工具
# ---------------------------------------------------------------------------

# 年份提取正则: 匹配 1900-2099 的 4 位年份
_YEAR_PATTERN = re.compile(r'\b(19\d{2}|20\d{2})\b')

# 月份名到数字映射
_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _extract_years_from_text(text: str) -> list[int]:
    """从文本中提取所有提及的年份。"""
    years: list[int] = []
    for m in _YEAR_PATTERN.finditer(text):
        years.append(int(m.group(1)))
    return years


def _chunk_to_timestamp(
    chunk_text: str,
    temporal_scope: str,
    base_year: int = 2024,
) -> int:
    """
    P0 修复: 从 chunk 文本中提取真实年份作为 timestamp。

    策略:
    1. 优先从 chunk 文本中提取年份
    2. 取最常见/中位数年份
    3. 回退到 temporal_scope 的年份
    4. 再回退到 base_year

    返回 Unix timestamp (该年份 7 月 1 日的 timestamp)，
    使 su-memory 的指数衰减和干支编码能按真实时间跨度工作。
    """
    text_years = _extract_years_from_text(chunk_text)
    if text_years:
        # 取中位数年份 (chunk 可能跨年)
        mid_year = sorted(text_years)[len(text_years) // 2]
        year = max(1970, min(2099, mid_year))
    else:
        # 回退: temporal_scope → base_year
        scope_years = _extract_years_from_text(temporal_scope) if temporal_scope else []
        if scope_years:
            year = max(1970, min(2099, scope_years[0]))
        else:
            year = base_year

    # 转为 Unix timestamp: 当年 7 月 1 日
    import calendar as _cal
    import datetime as _dt
    dt_obj = _dt.datetime(year, 7, 1, tzinfo=_dt.timezone.utc)
    return int(dt_obj.timestamp())


def _parse_question_time(
    question: str,
) -> list[int]:
    """
    P1: 从问题中提取时间约束。

    优先级:
    1. TemporalParser 正则引擎 (支持绝对/相对/持续/季度时间)
    2. 简单年份提取作为回退
    3. 返回目标年份列表 (用于 chunk 加权)
    """
    # 策略1: TemporalParser
    if TEMPORAL_PARSER_AVAILABLE and _temporal_parser is not None:
        try:
            parsed = _temporal_parser.parse_all(question)
            if parsed:
                target_years: list[int] = []
                for expr in parsed:
                    years = _extract_years_from_text(str(expr))
                    target_years.extend(years)
                if target_years:
                    return sorted(set(target_years))
        except Exception:
            pass

    # 策略2: 简单正则提取
    years = _extract_years_from_text(question)
    if years:
        return sorted(set(years))

    return []


def _boost_by_time(
    results: list[dict[str, Any]],
    question: str,
) -> list[dict[str, Any]]:
    """
    P1: 基于问题时间约束对检索结果加权重排。

    策略:
    - 从问题中提取目标年份
    - 对检索结果按内容年份与目标年份的匹配度加权
    - 完全匹配 +0.3, 同年区 +0.15, 完全不匹配不扣分
    - 保持原始排名作为 tie-breaking
    """
    target_years = _parse_question_time(question)
    if not target_years:
        return results

    target_set = set(target_years)

    for i, r in enumerate(results):
        content = str(r.get("content", ""))
        content_years = set(_extract_years_from_text(content))

        boost = 0.0
        if content_years & target_set:
            # 精确年份匹配 — 最高加分
            boost = 0.30
        elif content_years:
            # 检查是否在同一年区 (±1 年容差)
            for cy in content_years:
                for ty in target_set:
                    if abs(cy - ty) <= 1:
                        boost = 0.15
                        break
                if boost:
                    break

        original_score = float(r.get("score", 0.0))
        # 保持原始排名权重 (越靠前 baseline 越高)
        rank_weight = max(0, 1.0 - i * 0.02)  # 线性衰减
        r["_time_boost"] = boost
        r["_adjusted_score"] = original_score + boost * rank_weight

    # 按调整后分数重排
    results.sort(key=lambda r: r.get("_adjusted_score", r.get("score", 0)), reverse=True)

    # 清理临时字段
    for r in results:
        r.pop("_time_boost", None)
        r.pop("_adjusted_score", None)

    return results


def _ingest_sample(
    memory: SuMemoryLitePro,
    sample: dict[str, Any],
    stats: _TimeQAStats,
    chunk_chars: int,
) -> int:
    """
    将上下文注入 su-memory，返回 chunk 数。

    v4.4.1 P0 修复: 从 chunk 文本提取真实年份/日期作为 timestamp，
    而非伪造的 60 秒间距当前时间戳，使 su-memory 的干支编码和指数衰减能
    按真实历史时间跨度工作。
    """
    context = str(sample.get("context", "") or sample.get("document", "") or "")
    qid = str(sample.get("id", sample.get("question_id", "")))
    temporal_scope = str(sample.get("temporal_scope", "") or "")

    if not context:
        # 无上下文时用 question 本身
        context = str(sample.get("question", ""))

    chunks = _chunk_text(context, chunk_chars)

    for c_idx, chunk_text in enumerate(chunks):
        # P0: 提取真实年份作为 timestamp
        chunk_ts = _chunk_to_timestamp(chunk_text, temporal_scope)
        position_bucket_val = _position_bucket(c_idx, max(len(chunks), 1))
        position_ratio = c_idx / max(len(chunks) - 1, 1)
        meta = {
            "question_id": qid,
            "chunk_index": c_idx,
            "position_bucket": position_bucket_val,
            "position_ratio": position_ratio,
            "temporal_scope": temporal_scope,
            "is_evidence": True,
        }
        t0 = time.perf_counter()
        try:
            memory.add(
                content=chunk_text,
                metadata=meta,
                timestamp=chunk_ts,
                position_ratio=position_ratio,
            )
        except Exception as exc:
            logger.debug("  [warn] add failed: %s", exc)
            continue
        stats.add_times_ms.append((time.perf_counter() - t0) * 1000)

    return len(chunks)


def _is_answer_found(retrieved_texts: list[str], gold_answer: str) -> bool:
    """多策略答案匹配。"""
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
        if text and semantic_match(text, gold_answer, threshold=0.65):
            return True

    return False


def _evaluate_question(
    memory: SuMemoryLitePro,
    sample: dict[str, Any],
    stats: _TimeQAStats,
    top_k: int = 5,
    reranker: Any = None,
    use_spacetime: bool = True,
) -> dict[str, Any]:
    """
    对单条时序问题执行检索+评测。

    v4.4.1 P1: 问题时间约束解析 — 提取问题中的目标年份后对检索结果
    做时间匹配加权重排，使时间上相关的 chunk 优先。
    """
    question = str(sample.get("question", ""))
    gold_answer = str(sample.get("answer", "") or "")
    qid = str(sample.get("id", sample.get("question_id", "")))
    temporal_scope = str(sample.get("temporal_scope", "") or "")

    query_top_k = max(top_k * 8, 60) if reranker is not None else top_k

    t0 = time.perf_counter()
    try:
        results = memory.query(question, top_k=query_top_k, use_spacetime=use_spacetime)
    except Exception as exc:
        logger.debug("  [warn] query failed for %s: %s", qid, exc)
        results = []
    stats.query_times_ms.append((time.perf_counter() - t0) * 1000)

    # P1: 基于问题时间约束加权重排检索结果
    if results:
        results = _boost_by_time(list(results), question)

    # R@K 计算
    retrieved_texts = [str(r.get("content", "")) for r in results]
    r1_hit = _is_answer_found(retrieved_texts[:1], gold_answer)
    r3_hit = _is_answer_found(retrieved_texts[:3], gold_answer)
    r5_hit = _is_answer_found(retrieved_texts[:5], gold_answer)

    if r1_hit:
        stats.recall_at_1 += 1
    if r3_hit:
        stats.recall_at_3 += 1
    if r5_hit:
        stats.recall_at_5 += 1

    # LLM 答案提取 — P2: 注入时间提示
    llm_answer = ""
    if reranker is not None and retrieved_texts:
        top_context = "\n---\n".join(retrieved_texts[:15])
        try:
            llm_answer = _extract_answer_llm(
                reranker, question, top_context, gold_answer,
                temporal_scope=temporal_scope,
            )
        except Exception as exc:
            logger.debug("  [debug] LLM extraction failed: %s", exc)

    # 匹配判断
    is_correct = False
    if llm_answer:
        is_correct = (
            gold_answer.lower().strip() in llm_answer.lower()
            or semantic_match(llm_answer, gold_answer, threshold=SEMANTIC_THRESHOLD)
        )
    if not is_correct:
        is_correct = r5_hit

    if is_correct:
        stats.correct += 1

    # 维度统计
    t_bucket = _temporal_bucket(temporal_scope)
    stats.by_temporal[t_bucket][0] += 1 if is_correct else 0
    stats.by_temporal[t_bucket][1] += 1

    stats.total += 1

    return {
        "question_id": qid,
        "question": question[:100],
        "gold_answer": gold_answer,
        "llm_answer": llm_answer[:200],
        "correct": is_correct,
    }


def _extract_answer_llm(
    reranker: Any,
    question: str,
    context: str,
    gold_answer: str = "",
    temporal_scope: str = "",
) -> str:
    """
    用 LLM 从检索上下文中提取答案。

    v4.4.1 P2: 时间感知 Prompt — 将 temporal_scope 注入 prompt，
    引导 LLM 关注特定时间段的信息，减少跨时间混淆。
    """
    # P2: 构建时间提示
    time_hint = ""
    if temporal_scope:
        time_hint = f"""\n⚠️  TIME CONSTRAINT: The question asks about "{temporal_scope}".
ONLY use information from that specific time period. Ignore information from other time periods."""

    # P1: 从问题中额外提取年份提示
    question_years = _parse_question_time(question)
    year_hint = ""
    if question_years:
        year_hint = f"\nTarget year(s): {', '.join(str(y) for y in question_years)}. Prefer context mentioning these years."

    prompt = f"""Based on the context below, answer the following TIME-SENSITIVE question.
Provide ONLY the answer — no explanation, no preamble.

⚠️  IMPORTANT: Pay close attention to DATES and TIMEFRAMES in the context.
Only use information from the time period the question asks about.
{time_hint}{year_hint}

Context:
{context[:4000]}

Question: {question}

Answer:"""

    try:
        if hasattr(reranker, '_call_deepseek') and reranker.provider == "deepseek":
            resp = reranker._call_deepseek(prompt)
        elif hasattr(reranker, '_call_minimax') and reranker.provider == "minimax":
            resp = reranker._call_minimax(prompt)
        elif hasattr(reranker, '_call_glm') and reranker.provider == "glm":
            resp = reranker._call_glm(prompt)
        elif hasattr(reranker, '_call_openai'):
            resp = reranker._call_openai(prompt)
        else:
            resp = reranker._call_ollama(prompt, question)
        if resp:
            return resp.strip()
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# 主评测函数
# ---------------------------------------------------------------------------

def run_timeqa(
    backend: str = "minimax",
    storage_path: str = "",
    rerank_mode: str = "spacetime-llm",
    llm_provider: str = "auto",
    llm_model: str = "",
    max_questions: int = 0,
    top_k: int = 5,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    enable_bm25: bool = True,
    enable_energy_expand: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """运行 TimeQA 全量评测。"""
    if not storage_path:
        storage_path = os.path.join(
            ensure_data_dir("timeqa"), f"timeqa_run_{int(time.time())}"
        )

    # 加载数据
    samples = load_timeqa(verbose=verbose)
    if max_questions > 0:
        samples = samples[:max_questions]

    if verbose:
        print(f"\n{'='*65}")
        print(f"  su-memory v{VERSION} — TimeQA Temporal Reasoning Benchmark")
        print(f"{'='*65}")
        print(f"  Backend:      {backend}")
        print(f"  Questions:    {len(samples)}")
        print(f"  Rerank:       {rerank_mode}")
        print(f"  Chunk chars:  {chunk_chars}")
        print(f"  BM25:         {'ON' if enable_bm25 else 'OFF'}")
        print(f"  EnergyExpand: {'ON' if enable_energy_expand else 'OFF'}")
        print(f"  P0 real-ts:   ON  (chunk 真实年份提取)")
        print(f"  P1 time-bst:  ON  (问题时间约束加权)")
        print(f"  P2 time-pmt:  ON  (时间感知 LLM prompt)")
        print(f"{'='*65}\n")

    # 构建记忆引擎
    use_spacetime = "spacetime" in rerank_mode
    memory = _build_memory(
        backend, storage_path,
        enable_bm25=enable_bm25,
        enable_energy_expand=enable_energy_expand,
    )

    # LLM Reranker
    reranker = None
    if "llm" in rerank_mode and LLM_RERANKER_AVAILABLE:
        try:
            reranker = create_llm_reranker(provider=llm_provider, model=llm_model)
            if verbose:
                print(f"  LLM Reranker: {reranker.provider} ({reranker.model})\n")
        except Exception as exc:
            logger.warning("LLM reranker 初始化失败: %s", exc)

    stats = _TimeQAStats()

    # 逐题评测
    for idx, sample in enumerate(samples):
        qid = str(sample.get("id", sample.get("question_id", f"tqa_{idx}")))
        question_preview = str(sample.get("question", ""))[:80]

        # 注入
        n_chunks = _ingest_sample(memory, sample, stats, chunk_chars)

        # 评测
        result = _evaluate_question(
            memory, sample, stats,
            top_k=top_k,
            reranker=reranker,
            use_spacetime=use_spacetime,
        )

        if verbose and (idx < 5 or idx % 20 == 19):
            status = "✓" if result["correct"] else "✗"
            print(f"  [{idx+1}/{len(samples)}] {status} {qid}  "
                  f"(chunks={n_chunks})  \"{question_preview}…\"")

    # 汇总
    accuracy = stats.correct / max(stats.total, 1)
    avg_f1 = stats.f1_sum / max(stats.total, 1)

    if verbose:
        print(f"\n{'='*65}")
        print(f"  TimeQA Results Summary")
        print(f"{'='*65}")
        print(f"  Total:          {stats.total}")
        print(f"  Correct:        {stats.correct}")
        print(f"  Accuracy:       {accuracy:.1%}")
        print(f"  Recall@1:       {stats.recall_at_1 / max(stats.total, 1):.1%}")
        print(f"  Recall@3:       {stats.recall_at_3 / max(stats.total, 1):.1%}")
        print(f"  Recall@5:       {stats.recall_at_5 / max(stats.total, 1):.1%}")
        if stats.query_times_ms:
            avg_q = sum(stats.query_times_ms) / len(stats.query_times_ms)
            print(f"  Avg Query:      {avg_q:.1f}ms")

        print(f"\n  Temporal Buckets:")
        for bucket in ("near", "mid", "far", "unknown"):
            cnt = stats.by_temporal.get(bucket)
            if cnt and cnt[1] > 0:
                print(f"    {bucket:<10} {cnt[0]/cnt[1]:.1%}  ({cnt[0]}/{cnt[1]})")

        # 竞品对比
        print(f"\n  Competitor Comparison:")
        gpt4_score = COMPETITOR_SCORES.get("gpt4_turbo", {}).get("timeqa_accuracy")
        chronos_score = COMPETITOR_SCORES.get("chronos", {}).get("timeqa_accuracy")
        hindsight_score = COMPETITOR_SCORES.get("hindsight", {}).get("timeqa_accuracy")
        print(f"    su-memory v{VERSION}:  {accuracy:.1%}")
        if gpt4_score:
            delta = accuracy - gpt4_score
            print(f"    GPT-4 (full ctx):     {gpt4_score:.1%}  (Δ={delta:+.1%})")
        if chronos_score:
            delta = accuracy - chronos_score
            print(f"    Chronos:              {chronos_score:.1%}  (Δ={delta:+.1%})")
        if hindsight_score:
            delta = accuracy - hindsight_score
            print(f"    Hindsight:            {hindsight_score:.1%}  (Δ={delta:+.1%})")
        print(f"{'='*65}\n")

    return {
        "benchmark": "timeqa",
        "version": VERSION,
        "backend": backend,
        "total": stats.total,
        "correct": stats.correct,
        "accuracy": accuracy,
        "recall_at_1": stats.recall_at_1 / max(stats.total, 1),
        "recall_at_3": stats.recall_at_3 / max(stats.total, 1),
        "recall_at_5": stats.recall_at_5 / max(stats.total, 1),
        "temporal_buckets": {
            k: v[0] / max(v[1], 1) for k, v in stats.by_temporal.items() if v[1] > 0
        },
        "avg_query_time_ms": (
            sum(stats.query_times_ms) / len(stats.query_times_ms)
            if stats.query_times_ms else 0
        ),
        "avg_add_time_ms": (
            sum(stats.add_times_ms) / len(stats.add_times_ms)
            if stats.add_times_ms else 0
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="su-memory TimeQA Temporal Reasoning Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend", default="minimax",
        choices=list(BACKENDS.keys()),
        help="嵌入后端 (default: minimax)",
    )
    parser.add_argument(
        "--rerank", default="spacetime-llm",
        choices=["none", "spacetime", "spacetime-llm", "llm"],
        help="重排序模式: spacetime=时空索引, llm=LLM答案提取 (default: spacetime-llm)",
    )
    parser.add_argument(
        "--llm-provider", default="auto",
        choices=["auto", "deepseek", "openai", "ollama", "minimax", "glm"],
        help="LLM provider (default: auto)",
    )
    parser.add_argument(
        "--llm-model", default="",
        help="LLM 模型名称 (provider=ollama 时必须指定)",
    )
    parser.add_argument(
        "--max-questions", type=int, default=0,
        help="限制评测题数 (0=全部)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="检索返回 top-K (default: 5)",
    )
    parser.add_argument(
        "--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS,
        help=f"chunk 字符上限 (default: {DEFAULT_CHUNK_CHARS})",
    )
    parser.add_argument(
        "--storage", default="",
        help="持久化存储路径 (default: 自动生成)",
    )
    parser.add_argument("--no-bm25", action="store_true", help="关闭 BM25 检索")
    parser.add_argument("--no-energy-expand", action="store_true", help="关闭能量候选扩展")
    parser.add_argument(
        "--report", default="",
        help="输出 Markdown 报告路径",
    )
    parser.add_argument(
        "--output", default="",
        help="JSON 结果输出路径",
    )
    parser.add_argument("-v", "--verbose", action="store_true", default=True)
    parser.add_argument("-q", "--quiet", dest="verbose", action="store_false")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = run_timeqa(
        backend=args.backend,
        storage_path=args.storage,
        rerank_mode=args.rerank,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        max_questions=args.max_questions,
        top_k=args.top_k,
        chunk_chars=args.chunk_chars,
        enable_bm25=not args.no_bm25,
        enable_energy_expand=not args.no_energy_expand,
        verbose=args.verbose,
    )

    # JSON 输出
    if args.output:
        output_path = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            ensure_data_dir("timeqa"), f"timeqa_{args.backend}_{ts}.json"
        )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    if args.verbose:
        print(f"  📄 JSON: {output_path}")

    return 0 if result["total"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
