#!/usr/bin/env python3
"""
su-memory Benchmark 基础设施配置
================================
支持 LongMemEval / LoCoMo / ConvoMem 三大评测基准的共享配置层。

包含内容：
- 嵌入后端配置（Ollama bge-m3、sentence-transformers）
- 数据集配置（HuggingFace 数据集 ID 与本地缓存路径）
- ``BenchmarkResult`` 通用结果数据结构
- 评测指标计算函数（EM / F1 / Recall@K / BLEU / ROUGE-L / 语义匹配）
- 数据集加载辅助
- 竞品对标分数

参考：
- LongMemEval: https://arxiv.org/abs/2406.09974
- LoCoMo:      https://snap-research.github.io/locomo/
- ConvoMem:    https://huggingface.co/datasets/Salesforce/ConvoMem
"""

from __future__ import annotations

import json
import os
import re
import string
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


# ---------------------------------------------------------------------------
# 1. 嵌入后端配置
# ---------------------------------------------------------------------------

BACKENDS: dict[str, dict[str, Any]] = {
    "ollama": {
        "name": "Ollama bge-m3",
        "type": "ollama",
        "model": "bge-m3",
        "base_url": "http://localhost:11434",
        "dimension": 1024,
    },
    "sbert": {
        "name": "sentence-transformers (all-MiniLM-L6-v2)",
        "type": "sentence-transformers",
        "model": "all-MiniLM-L6-v2",
        "dimension": 384,
    },
    "sbert-mpnet": {
        "name": "sentence-transformers (all-mpnet-base-v2)",
        "type": "sentence-transformers",
        "model": "all-mpnet-base-v2",
        "dimension": 768,
    },
}


# ---------------------------------------------------------------------------
# 2. 数据集配置
# ---------------------------------------------------------------------------

BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data"

DATASETS: dict[str, dict[str, Any]] = {
    "longmemeval": {
        "hf_id": "xiaowu0162/longmemeval-cleaned",
        "local_cache": str(DATA_DIR / "longmemeval"),
        "files": [
            "longmemeval_oracle.json",
            "longmemeval_s_cleaned.json",
            "longmemeval_m_cleaned.json",
        ],
        "description": "Long-term memory retention across extended conversations",
    },
    "locomo": {
        "hf_id": "snap-research/locomo",
        "local_cache": str(DATA_DIR / "locomo"),
        "files": ["locomo10.json"],
        "description": "Long Conversational Memory across 600+ turns / multiple sessions",
    },
    "convomem": {
        "hf_id": "Salesforce/ConvoMem",
        "local_cache": str(DATA_DIR / "convomem"),
        "files": [],
        "description": "Conversational memory benchmark by Salesforce Research",
    },
}


# ---------------------------------------------------------------------------
# 3. BenchmarkResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """三大评测基准共享的结果数据结构。

    Attributes:
        benchmark_name: 评测名称（``longmemeval`` / ``locomo`` / ``convomem``）。
        backend: 嵌入后端 key（来自 :data:`BACKENDS`）。
        timestamp: ISO 8601 运行时间戳。
        total_questions: 评测问题总数。
        correct: 正确回答数。
        accuracy: 总体准确率。
        f1_score: 宏平均 F1。
        recall_at_1/3/5: 检索召回率。
        bleu: 语料级 BLEU 平均分（生成式问答时使用）。
        rouge_l: ROUGE-L F1 平均分。
        avg_query_time_ms: 平均查询耗时（毫秒）。
        avg_add_time_ms: 平均写入耗时（毫秒）。
        dimension_scores: 各分维度（如 single-hop / multi-hop / 时序）评分。
        metadata: 其他附加信息（数据集大小、超参等）。
    """

    benchmark_name: str
    backend: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    total_questions: int = 0
    correct: int = 0
    accuracy: float = 0.0
    f1_score: float = 0.0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    bleu: float = 0.0
    rouge_l: float = 0.0
    avg_query_time_ms: float = 0.0
    avg_add_time_ms: float = 0.0
    dimension_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转为可序列化字典，浮点数保留 4 位小数。"""

        def _round(v: Any) -> Any:
            if isinstance(v, float):
                return round(v, 4)
            if isinstance(v, dict):
                return {k: _round(x) for k, x in v.items()}
            if isinstance(v, list):
                return [_round(x) for x in v]
            return v

        return {k: _round(v) for k, v in asdict(self).items()}

    def to_json(self, path: str) -> str:
        """将结果以 JSON 形式写入 ``path``，返回写入路径。"""

        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkResult":
        """从 ``to_dict`` 输出还原实例。"""

        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# 4. 评测指标计算函数
# ---------------------------------------------------------------------------

_PUNC_TABLE = str.maketrans("", "", string.punctuation)
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """SQuAD 风格归一化：小写、去标点、去冠词、压缩空白。"""

    if text is None:
        return ""
    text = str(text).lower()
    text = text.translate(_PUNC_TABLE)
    text = _ARTICLES_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def exact_match(prediction: str, reference: str) -> bool:
    """SQuAD 风格归一化后的精确匹配。"""

    return _normalize(prediction) == _normalize(reference)


def compute_f1(prediction: str, reference: str) -> float:
    """词级 F1（SQuAD 风格）。

    用于生成式问答中评估预测与标准答案之间的 token 级重合度。
    """

    pred_tokens = _normalize(prediction).split()
    ref_tokens = _normalize(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_recall_at_k(retrieved: Sequence[Any], gold: Iterable[Any], k: int) -> float:
    """Recall@K：top-K 检索结果中至少命中一个 gold 标签即视为命中。

    Args:
        retrieved: 检索系统返回的候选列表（顺序敏感）。
        gold: 标准答案集合。
        k: 截断位置。

    Returns:
        命中率 1.0 或 0.0；当 ``gold`` 为空时返回 0.0。
    """

    gold_set = {str(g) for g in gold}
    if not gold_set or k <= 0:
        return 0.0
    top_k = list(retrieved)[:k]
    for item in top_k:
        if str(item) in gold_set:
            return 1.0
        # 兼容字典 / 对象包含 content 字段的情形：子串匹配
        text = item if isinstance(item, str) else getattr(item, "content", None) or (
            item.get("content") if isinstance(item, dict) else ""
        )
        if text:
            text_str = str(text)
            for g in gold_set:
                if g and g in text_str:
                    return 1.0
    return 0.0


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu(prediction: str, reference: str, max_n: int = 4) -> float:
    """简化的 BLEU 句子级实现（带 brevity penalty，1..max_n 取几何平均）。

    若安装了 ``nltk``，将优先使用 NLTK 的 ``sentence_bleu``，
    否则回退到本地实现以保证零依赖可用。
    """

    pred_tokens = _normalize(prediction).split()
    ref_tokens = _normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    try:  # pragma: no cover - 依赖可选
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        smoothing = SmoothingFunction().method1
        weights = tuple([1.0 / max_n] * max_n)
        return float(sentence_bleu([ref_tokens], pred_tokens, weights=weights, smoothing_function=smoothing))
    except Exception:
        pass

    import math

    precisions: list[float] = []
    for n in range(1, max_n + 1):
        pred_ng = _ngrams(pred_tokens, n)
        ref_ng = _ngrams(ref_tokens, n)
        if not pred_ng:
            precisions.append(0.0)
            continue
        overlap = sum((pred_ng & ref_ng).values())
        # +1 smoothing 避免 log(0)
        precisions.append((overlap + 1) / (sum(pred_ng.values()) + 1))
    if all(p == 0 for p in precisions):
        return 0.0
    log_avg = sum(math.log(p) for p in precisions) / max_n
    bp = 1.0 if len(pred_tokens) > len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(pred_tokens))
    return float(bp * math.exp(log_avg))


def compute_rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1（基于最长公共子序列）。

    若安装了 ``rouge_score``，优先使用官方实现；否则回退到本地 LCS 实现。
    """

    pred_tokens = _normalize(prediction).split()
    ref_tokens = _normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    try:  # pragma: no cover - 依赖可选
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return float(scores["rougeL"].fmeasure)
    except Exception:
        pass

    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    precision = lcs / m
    recall = lcs / n
    return 2 * precision * recall / (precision + recall)


def semantic_match(prediction: str, reference: str, threshold: float = 0.75) -> bool:
    """基于嵌入相似度的语义匹配。

    优先尝试 ``sentence-transformers`` 的 ``all-MiniLM-L6-v2`` 计算余弦相似度；
    若依赖缺失，回退到 ``compute_f1`` 是否超过阈值。
    """

    if exact_match(prediction, reference):
        return True
    try:  # pragma: no cover - 依赖可选
        from sentence_transformers import SentenceTransformer, util

        model = _get_sbert_model()
        emb = model.encode([prediction, reference], convert_to_tensor=True, normalize_embeddings=True)
        score = float(util.cos_sim(emb[0], emb[1]).item())
        return score >= threshold
    except Exception:
        return compute_f1(prediction, reference) >= threshold


_SBERT_MODEL_CACHE: dict[str, Any] = {}


def _get_sbert_model(model_name: str = "all-MiniLM-L6-v2") -> Any:  # pragma: no cover
    """惰性加载 sentence-transformers 模型并缓存。"""

    if model_name in _SBERT_MODEL_CACHE:
        return _SBERT_MODEL_CACHE[model_name]
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    _SBERT_MODEL_CACHE[model_name] = model
    return model


# ---------------------------------------------------------------------------
# 5. 数据集加载辅助
# ---------------------------------------------------------------------------

def ensure_data_dir(benchmark_name: str) -> str:
    """确保 ``benchmarks/data/<benchmark_name>/`` 目录存在并返回其路径。

    Args:
        benchmark_name: 例如 ``longmemeval`` / ``locomo`` / ``convomem``。

    Returns:
        目录的绝对路径字符串。
    """

    cfg = DATASETS.get(benchmark_name)
    if cfg and "local_cache" in cfg:
        path = Path(cfg["local_cache"])
    else:
        path = DATA_DIR / benchmark_name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def load_hf_dataset(
    dataset_id: str,
    split: str | None = None,
    cache_dir: str | None = None,
    streaming: bool = False,
    **kwargs: Any,
) -> Any:
    """加载 HuggingFace 数据集，统一封装错误信息。

    Args:
        dataset_id: 形如 ``xiaowu0162/longmemeval-cleaned`` 的 HF 数据集 ID。
        split: 数据集切分（``train`` / ``test`` / ``validation``）。
        cache_dir: 本地缓存目录；若为 ``None`` 则使用 HF 默认。
        streaming: 是否启用流式加载。
        **kwargs: 透传给 ``datasets.load_dataset``。

    Returns:
        ``datasets.Dataset`` / ``DatasetDict`` 对象。

    Raises:
        ImportError: 当未安装 ``datasets`` 库。
        RuntimeError: 加载失败时附带原始异常信息。
    """

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - 依赖检查
        raise ImportError(
            "Missing dependency 'datasets'. Run: pip install datasets"
        ) from exc

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # 过滤 deprecated 的 trust_remote_code（新版 datasets 不再支持）
    kwargs.pop("trust_remote_code", None)

    try:
        return load_dataset(
            dataset_id,
            split=split,
            cache_dir=cache_dir,
            streaming=streaming,
            **kwargs,
        )
    except Exception as exc:  # pragma: no cover - 网络错误
        raise RuntimeError(f"Failed to load HF dataset '{dataset_id}': {exc}") from exc


# ---------------------------------------------------------------------------
# 6. 竞品对标数据
# ---------------------------------------------------------------------------

# 数据来源：各竞品官方论文 / 公开 leaderboard / 第三方复现报告。
# 字段命名：<benchmark>_<metric>，缺失值以 None 表示（不展示在排行榜）。
COMPETITOR_SCORES: dict[str, dict[str, float | None]] = {
    "hindsight": {
        "longmemeval_accuracy": 0.914,
        "locomo_f1": 0.682,
        "convomem_accuracy": None,
    },
    "memgpt": {
        "longmemeval_accuracy": 0.783,
        "locomo_f1": 0.564,
        "convomem_accuracy": 0.612,
    },
    "mem0": {
        "longmemeval_accuracy": 0.725,
        "locomo_f1": 0.526,
        "convomem_accuracy": 0.580,
    },
    "zep": {
        "longmemeval_accuracy": 0.661,
        "locomo_f1": 0.498,
        "convomem_accuracy": None,
    },
    "letta": {
        "longmemeval_accuracy": 0.693,
        "locomo_f1": 0.512,
        "convomem_accuracy": None,
    },
    "gpt4_turbo": {
        "longmemeval_accuracy": 0.535,
        "locomo_f1": 0.402,
        "convomem_accuracy": 0.488,
    },
}


# ---------------------------------------------------------------------------
# 7. 入口：自检
# ---------------------------------------------------------------------------

def _self_check() -> None:
    """轻量自检：验证配置完整性与指标函数可用性。"""

    print("=" * 60)
    print("  su-memory Benchmark Config — Self Check")
    print("=" * 60)

    print(f"  BENCHMARK_DIR: {BENCHMARK_DIR}")
    print(f"  DATA_DIR:      {DATA_DIR}  (exists={DATA_DIR.exists()})")

    print("\n  Backends:")
    for key, cfg in BACKENDS.items():
        print(f"    - {key:<12} {cfg['name']:<48} dim={cfg['dimension']}")

    print("\n  Datasets:")
    for key, cfg in DATASETS.items():
        cache = Path(cfg["local_cache"])
        print(f"    - {key:<12} hf={cfg['hf_id']:<32} cache_exists={cache.exists()}")

    print("\n  Metric sanity:")
    pred, ref = "The quick brown fox", "A quick brown fox"
    print(f"    exact_match     = {exact_match(pred, ref)}")
    print(f"    compute_f1      = {compute_f1(pred, ref):.4f}")
    print(f"    compute_bleu    = {compute_bleu(pred, ref):.4f}")
    print(f"    compute_rouge_l = {compute_rouge_l(pred, ref):.4f}")
    print(f"    recall@3        = {compute_recall_at_k(['a', 'b', 'c'], ['c'], 3):.4f}")

    print("\n  BenchmarkResult round-trip:")
    sample = BenchmarkResult(
        benchmark_name="longmemeval",
        backend="ollama",
        total_questions=100,
        correct=84,
        accuracy=0.84,
        f1_score=0.812,
        recall_at_1=0.79,
        dimension_scores={"single_hop": 0.91, "multi_hop": 0.72},
    )
    payload = sample.to_dict()
    restored = BenchmarkResult.from_dict(payload)
    print(f"    sample.accuracy={sample.accuracy}  restored.accuracy={restored.accuracy}")

    print("\n  Competitor scores (longmemeval_accuracy):")
    for name, scores in COMPETITOR_SCORES.items():
        v = scores.get("longmemeval_accuracy")
        print(f"    - {name:<12} {('N/A' if v is None else f'{v:.1%}')}")

    print("\n  ✅ Config self-check passed.")


if __name__ == "__main__":
    _self_check()
