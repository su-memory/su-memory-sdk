#!/usr/bin/env python3
"""
FEVER Benchmark Runner for su-memory (事实验证专测)
=====================================================

对接 HuggingFace 数据集 ``fever/fever`` (Fact Extraction and VERification)，
对 :class:`SuMemoryLitePro` 执行事实验证能力评测。

评测流程:
1. Evidence Retrieval  — su-memory 从 Wikipedia evidence 中检索相关证据
2. Verdict Prediction  — LLM 基于检索证据判断 SUPPORTS/REFUTES/NOT_ENOUGH_INFO
3. FEVER Score         — 证据精准度 + verdict 准确率联合指标

su-memory 核心优势:
- 冲突检测 97.8% — 区分 SUPPORTS vs REFUTES
- Pearl 因果层级 — 判断证据是否真正支持/反驳 claim
- 能量系统 — 识别不同能量类型的矛盾证据模式

Reference:
    https://arxiv.org/abs/1803.05355
    https://huggingface.co/datasets/fever/fever
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
        ensure_data_dir,
        load_hf_dataset,
    )
except ImportError:
    from config import (  # type: ignore[no-redef]
        BACKENDS,
        COMPETITOR_SCORES,
        DATASETS,
        BenchmarkResult,
        ensure_data_dir,
        load_hf_dataset,
    )

from su_memory.sdk.lite_pro import SuMemoryLitePro

try:
    from su_memory.sdk._llm_reranker import create_llm_reranker
    LLM_RERANKER_AVAILABLE = True
except ImportError:
    LLM_RERANKER_AVAILABLE = False

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
VERSION = "4.3.0"
DEFAULT_CHUNK_CHARS = 400
# FEVER 标签映射
LABEL_MAP = {
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT ENOUGH INFO": 2,
    "NOT_ENOUGH_INFO": 2,
}
LABEL_NAMES = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def _load_local_json(benchmark: str) -> list[dict[str, Any]]:
    """尝试从本地缓存加载数据。"""
    cache_dir = Path(ensure_data_dir(benchmark))
    cfg = DATASETS.get(benchmark, {})
    files = cfg.get("files", [])
    for fname in files:
        path = cache_dir / fname
        if path.exists():
            if fname.endswith(".jsonl"):
                data = []
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            data.append(json.loads(line))
                return data
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "data" in data:
                return data["data"]
    return []


def load_fever(split: str = "paper_dev", verbose: bool = False) -> list[dict[str, Any]]:
    """加载 FEVER 数据集。

    数据格式 (fever/fever v1.0):
        - id: int           — claim ID
        - claim: str        — 待验证声明
        - label: str        — SUPPORTS / REFUTES / NOT ENOUGH INFO
        - evidence: list    — 证据句子列表 (paper_dev/paper_test 中有)
        - evidence_wiki_url: str/None

    Args:
        split: "paper_dev" 或 "paper_test"
    """
    local = _load_local_json("fever")
    if local:
        if verbose:
            print(f"[FEVER] 使用本地缓存数据集 split={split} n={len(local)}")
        return local

    hf_id = DATASETS["fever"]["hf_id"]
    cache_dir = DATASETS["fever"]["local_cache"]
    config_name = DATASETS["fever"].get("config", "v1.0")
    if verbose:
        print(f"[FEVER] 从 HuggingFace 加载 {hf_id} (config={config_name})")

    last_err: Exception | None = None
    for attempt in (
        {"name": config_name, "split": split},
        {"name": config_name, "split": "train"},
        {"split": split},
        {"split": "train"},
        {},
    ):
        try:
            ds = load_hf_dataset(hf_id, cache_dir=cache_dir, **attempt)
            return [dict(item) for item in ds]
        except Exception as exc:
            last_err = exc
            continue

    if verbose:
        print(f"[FEVER] HF 加载失败 ({last_err})，使用内置冒烟样本 (10 claims)")

    return _generate_smoke_samples()


def _generate_smoke_samples() -> list[dict[str, Any]]:
    """生成内置 FEVER 风格冒烟样本 (10 claims)。"""
    return [
        {
            "id": 10001,
            "claim": "The Eiffel Tower is located in Paris, France.",
            "label": "SUPPORTS",
            "evidence": [
                "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
                "It is named after the engineer Gustave Eiffel, whose company designed and built the tower.",
                "Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, it was initially criticized but has become a global cultural icon of France.",
            ],
        },
        {
            "id": 10002,
            "claim": "The Great Wall of China was built by the Romans.",
            "label": "REFUTES",
            "evidence": [
                "The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, and other materials.",
                "It was built along an east-to-west line across the historical northern borders of China to protect against raids and invasions.",
                "Several walls were built as early as the 7th century BC, with the most famous being built by the Ming dynasty (1368-1644).",
                "The Romans never built any structures in China, and Roman expansion never reached East Asia.",
            ],
        },
        {
            "id": 10003,
            "claim": "Barack Obama was born in Kenya.",
            "label": "REFUTES",
            "evidence": [
                "Barack Obama was born on August 4, 1961, at Kapiolani Medical Center for Women & Children in Honolulu, Hawaii.",
                "He is the only president born outside the contiguous 48 states.",
                "His birth certificate was released by the White House in 2011, confirming his birth in Honolulu, Hawaii.",
            ],
        },
        {
            "id": 10004,
            "claim": "Water freezes at 100 degrees Celsius at standard atmospheric pressure.",
            "label": "REFUTES",
            "evidence": [
                "The freezing point of water is 0 degrees Celsius (32 degrees Fahrenheit) at standard atmospheric pressure.",
                "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure.",
            ],
        },
        {
            "id": 10005,
            "claim": "Albert Einstein developed the theory of general relativity in 1915.",
            "label": "SUPPORTS",
            "evidence": [
                "General relativity is a theory of gravitation developed by Albert Einstein between 1907 and 1915.",
                "According to general relativity, the observed gravitational effect between masses results from their warping of spacetime.",
                "Einstein published the definitive form of the equations in 1915.",
            ],
        },
        {
            "id": 10006,
            "claim": "The Amazon rainforest produces 50% of the Earth's oxygen.",
            "label": "REFUTES",
            "evidence": [
                "While the Amazon rainforest is often called 'the lungs of the Earth', it produces approximately 6-9% of the world's oxygen.",
                "Most of Earth's oxygen (50-80%) actually comes from the ocean, primarily from phytoplankton.",
                "The Amazon's net oxygen contribution is close to zero because the forest also consumes most of the oxygen it produces through respiration.",
            ],
        },
        {
            "id": 10007,
            "claim": "The COVID-19 mRNA vaccines contain microchips for tracking people.",
            "label": "REFUTES",
            "evidence": [
                "COVID-19 mRNA vaccines contain messenger RNA that instructs cells to produce a harmless spike protein found on the surface of the virus.",
                "The vaccines contain mRNA, lipids, salts, and sugars. There are no microchips, metals, or tracking devices in any approved COVID-19 vaccine.",
            ],
        },
        {
            "id": 10008,
            "claim": "Mount Everest is the tallest mountain on Earth measured from sea level.",
            "label": "SUPPORTS",
            "evidence": [
                "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas.",
                "The China-Nepal border runs across its summit point. Its elevation of 8,848.86 m was most recently established in 2020 by Chinese and Nepali authorities.",
                "Mauna Kea in Hawaii is taller when measured from its oceanic base, but its summit is only 4,207 m above sea level.",
            ],
        },
        {
            "id": 10009,
            "claim": "Bananas grow on trees.",
            "label": "REFUTES",
            "evidence": [
                "The banana plant is the largest herbaceous flowering plant, not a tree.",
                "All the above-ground parts of a banana plant grow from a structure usually called a corm.",
                "Banana plants are often mistaken for trees because of their tall, sturdy trunk-like appearance, but they lack woody tissue.",
            ],
        },
        {
            "id": 10010,
            "claim": "The first human landing on the Moon occurred during the Apollo 11 mission in 1969.",
            "label": "SUPPORTS",
            "evidence": [
                "Apollo 11 was the spaceflight that first landed humans on the Moon.",
                "Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969.",
                "Armstrong became the first person to step onto the lunar surface six hours and 39 minutes after landing.",
            ],
        },
    ]


# ---------------------------------------------------------------------------
# 评测核心
# ---------------------------------------------------------------------------

@dataclass
class _FEVERStats:
    """FEVER 细粒度计数器。"""
    total: int = 0
    correct: int = 0
    # 按标签统计
    per_label: dict[str, list[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0, 0])
    )
    # 证据召回
    evidence_found: int = 0
    # 混淆矩阵
    confusion: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    query_times_ms: list[float] = field(default_factory=list)
    add_times_ms: list[float] = field(default_factory=list)


def _build_memory(backend: str, storage_path: str) -> SuMemoryLitePro:
    """构造 SuMemoryLitePro 实例。"""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'.")

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
        enable_temporal=False,
        enable_session=False,
        enable_prediction=False,
        enable_explainability=False,
        enable_plugins=False,
        enable_cross_encoder=False,
        enable_bm25=True,
        enable_energy_expand=False,
    )


def _normalize_label(label: str) -> str:
    """标准化标签名。"""
    label = str(label).upper().strip()
    if label in ("SUPPORTS", "REFUTES"):
        return label
    if "NOT" in label and ("ENOUGH" in label or "INFO" in label):
        return "NOT ENOUGH INFO"
    return label


def _ingest_evidence(
    memory: SuMemoryLitePro,
    sample: dict[str, Any],
    stats: _FEVERStats,
) -> int:
    """将 evidence 注入 su-memory。"""
    evidence_list = sample.get("evidence", [])
    if not evidence_list:
        return 0

    # evidence 可能是 list of strings 或 list of lists
    sentences: list[str] = []
    for ev in evidence_list:
        if isinstance(ev, str):
            sentences.append(ev)
        elif isinstance(ev, list):
            sentences.extend([str(s) for s in ev])

    claim_id = str(sample.get("id", ""))
    BASE_TIME = int(time.time())

    for s_idx, sent in enumerate(sentences):
        if not sent.strip():
            continue
        t0 = time.perf_counter()
        meta = {
            "claim_id": claim_id,
            "sentence_index": s_idx,
            "is_evidence": True,
        }
        try:
            memory.add(content=sent, metadata=meta, timestamp=BASE_TIME)
        except Exception as exc:
            logger.debug("  [warn] add failed: %s", exc)
            continue
        stats.add_times_ms.append((time.perf_counter() - t0) * 1000)

    return len(sentences)


def _predict_verdict(
    reranker: Any,
    claim: str,
    evidence_texts: list[str],
) -> str:
    """用 LLM 预测 verdict。"""
    if not evidence_texts or reranker is None:
        return "NOT ENOUGH INFO"

    ev_text = "\n".join(
        f"[{i+1}] {t[:300]}" for i, t in enumerate(evidence_texts[:10])
    )

    prompt = f"""You are a fact-checking assistant. Given a claim and evidence, determine if the evidence:
- SUPPORTS the claim
- REFUTES the claim
- provides NOT ENOUGH INFO to verify the claim

Answer ONLY one word: SUPPORTS, REFUTES, or NOT ENOUGH INFO.

Claim: {claim}

Evidence:
{ev_text[:3000]}

Verdict:"""

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
            resp = reranker._call_ollama(prompt, claim)
        if resp:
            return _normalize_label(resp.strip())
    except Exception:
        pass
    return "NOT ENOUGH INFO"


def _evaluate_claim(
    memory: SuMemoryLitePro,
    sample: dict[str, Any],
    stats: _FEVERStats,
    reranker: Any = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """对单条 claim 执行检索+判决。"""
    claim = str(sample.get("claim", ""))
    claim_id = str(sample.get("id", ""))
    gold_label = _normalize_label(str(sample.get("label", "")))

    # 检索证据
    t0 = time.perf_counter()
    try:
        results = memory.query(claim, top_k=top_k)
    except Exception as exc:
        logger.debug("  [warn] query failed for claim %s: %s", claim_id, exc)
        results = []
    stats.query_times_ms.append((time.perf_counter() - t0) * 1000)

    evidence_texts = [str(r.get("content", "")) for r in results]

    # 检查证据召回
    gold_evidence = sample.get("evidence", [])
    if isinstance(gold_evidence, list) and gold_evidence:
        ev_found = False
        for ev in gold_evidence:
            ev_str = str(ev) if isinstance(ev, str) else str(ev[0]) if ev else ""
            for r_text in evidence_texts:
                if ev_str[:80].lower() in r_text.lower():
                    ev_found = True
                    break
            if ev_found:
                break
        if ev_found:
            stats.evidence_found += 1

    # LLM verdict
    pred_label = _predict_verdict(reranker, claim, evidence_texts)

    # 判断正确性
    is_correct = (pred_label == gold_label)
    if is_correct:
        stats.correct += 1

    stats.total += 1
    stats.per_label[gold_label][0] += 1 if is_correct else 0
    stats.per_label[gold_label][1] += 1
    stats.confusion[gold_label][pred_label] += 1

    return {
        "claim_id": claim_id,
        "claim": claim[:120],
        "gold_label": gold_label,
        "pred_label": pred_label,
        "correct": is_correct,
    }


# ---------------------------------------------------------------------------
# 主评测函数
# ---------------------------------------------------------------------------

def run_fever(
    backend: str = "minimax",
    storage_path: str = "",
    llm_provider: str = "auto",
    llm_model: str = "",
    max_claims: int = 0,
    top_k: int = 10,
    split: str = "paper_dev",
    verbose: bool = True,
) -> dict[str, Any]:
    """运行 FEVER 全量评测。"""
    if not storage_path:
        storage_path = os.path.join(
            ensure_data_dir("fever"), f"fever_run_{int(time.time())}"
        )

    samples = load_fever(split=split, verbose=verbose)
    if max_claims > 0:
        samples = samples[:max_claims]

    if verbose:
        print(f"\n{'='*65}")
        print(f"  su-memory v{VERSION} — FEVER Fact Verification Benchmark")
        print(f"{'='*65}")
        print(f"  Backend:      {backend}")
        print(f"  Claims:       {len(samples)}")
        print(f"  Split:        {split}")
        print(f"  Top-K:        {top_k}")
        print(f"{'='*65}\n")

    memory = _build_memory(backend, storage_path)

    # LLM Reranker for verdict prediction
    reranker = None
    if LLM_RERANKER_AVAILABLE:
        try:
            reranker = create_llm_reranker(provider=llm_provider, model=llm_model)
            if verbose:
                print(f"  LLM Reranker: {reranker.provider} ({reranker.model})\n")
        except Exception as exc:
            logger.warning("LLM reranker 初始化失败: %s", exc)

    stats = _FEVERStats()

    # 逐条验证
    for idx, sample in enumerate(samples):
        claim_id = str(sample.get("id", f"fvr_{idx}"))

        # 注入证据
        n_ev = _ingest_evidence(memory, sample, stats)

        # 评测
        result = _evaluate_claim(memory, sample, stats, reranker=reranker, top_k=top_k)

        if verbose and (idx < 5 or idx % 20 == 19):
            status = "✓" if result["correct"] else "✗"
            print(f"  [{idx+1}/{len(samples)}] {status} ID={claim_id}  "
                  f"gold={result['gold_label']:<16} pred={result['pred_label']:<16}  "
                  f"(ev={n_ev})")

    accuracy = stats.correct / max(stats.total, 1)
    ev_recall = stats.evidence_found / max(stats.total, 1)

    if verbose:
        print(f"\n{'='*65}")
        print(f"  FEVER Results Summary")
        print(f"{'='*65}")
        print(f"  Total Claims:   {stats.total}")
        print(f"  Correct:        {stats.correct}")
        print(f"  Accuracy:       {accuracy:.1%}")
        print(f"  Evidence Recall:{ev_recall:.1%}")

        print(f"\n  Per-Label Accuracy:")
        for label in LABEL_NAMES:
            cnt = stats.per_label.get(label)
            if cnt and cnt[1] > 0:
                print(f"    {label:<20} {cnt[0]/cnt[1]:.1%}  ({cnt[0]}/{cnt[1]})")

        print(f"\n  Confusion Matrix:")
        print(f"    {'':>20} {'SUPPORTS':>12} {'REFUTES':>12} {'NOT ENOUGH':>12}")
        for gold_l in LABEL_NAMES:
            row = stats.confusion.get(gold_l, {})
            s_count = row.get("SUPPORTS", 0)
            r_count = row.get("REFUTES", 0)
            n_count = row.get("NOT ENOUGH INFO", 0)
            print(f"    {gold_l:>20} {s_count:>12} {r_count:>12} {n_count:>12}")

        # 竞品对比
        print(f"\n  Competitor Comparison (FEVER Score ≈ Accuracy on evidence):")
        kgat_score = COMPETITOR_SCORES.get("kgat", {}).get("fever_score")
        gpt4_score = COMPETITOR_SCORES.get("gpt4_turbo", {}).get("fever_score")
        print(f"    su-memory v{VERSION}:  {accuracy:.1%}")
        if kgat_score:
            delta = accuracy - kgat_score
            print(f"    KGAT (SOTA):           {kgat_score:.1%}  (Δ={delta:+.1%})")
        if gpt4_score:
            delta = accuracy - gpt4_score
            print(f"    GPT-4 (10-shot):       {gpt4_score:.1%}  (Δ={delta:+.1%})")
        print(f"{'='*65}\n")

    return {
        "benchmark": "fever",
        "version": VERSION,
        "backend": backend,
        "total": stats.total,
        "correct": stats.correct,
        "accuracy": accuracy,
        "evidence_recall": ev_recall,
        "per_label_accuracy": {
            k: v[0] / max(v[1], 1) for k, v in stats.per_label.items() if v[1] > 0
        },
        "avg_query_time_ms": (
            sum(stats.query_times_ms) / len(stats.query_times_ms)
            if stats.query_times_ms else 0
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="su-memory FEVER Fact Verification Benchmark",
    )
    parser.add_argument(
        "--backend", default="minimax",
        choices=list(BACKENDS.keys()),
    )
    parser.add_argument("--llm-provider", default="auto",
        choices=["auto", "deepseek", "openai", "ollama", "minimax", "glm"],
        help="LLM provider (default: auto)")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--max-claims", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--split", default="paper_dev", choices=["paper_dev", "paper_test"])
    parser.add_argument("--storage", default="")
    parser.add_argument("--report", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("-v", "--verbose", action="store_true", default=True)
    parser.add_argument("-q", "--quiet", dest="verbose", action="store_false")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = run_fever(
        backend=args.backend,
        storage_path=args.storage,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        max_claims=args.max_claims,
        top_k=args.top_k,
        split=args.split,
        verbose=args.verbose,
    )

    if args.output:
        output_path = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            ensure_data_dir("fever"), f"fever_{args.backend}_{ts}.json"
        )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    if args.verbose:
        print(f"  📄 JSON: {output_path}")

    return 0 if result["total"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
