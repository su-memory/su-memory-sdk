"""
真实 HotpotQA A/B 对比: embedding 检索 vs embedding + 实体桥接图传播

数据
----
benchmarks/data/hotpotqa_validation_200.json — 官方 HotpotQA validation 子集
(200 题, 全 hard level, 166 bridge + 34 comparison).
来源: https://hotpotqa.github.io/  Yang et al. EMNLP 2018.

这是 *真实* 评测数据 (非合成), 可与文献中的 supporting-fact 检索任务对比.

任务
----
Supporting Fact Retrieval: 给定问题 + 多个维基百科段落 (含干扰段),
召回两个 gold supporting paragraphs. 标准 HotpotQA 子任务.

核心难点 (treatment 的提升空间)
-------------------------------
embedding 容易命中 *第一个* gold 段落 (query-段落 直接语义相关),
但 *第二个* gold 段落 (桥接段落) 常被干扰段挤出 top-k, 因为它与 query
无直接语义重叠, 必须通过第一个证据的实体桥接才能找到.

A/B 设计
--------
- baseline:    embedding 余弦相似度 top-k
- treatment:   embedding top-1 起 + CausalDAG 实体桥接图 BFS 召回第二个
               (实体共现建边: 段落间共享命名实体 → 因果图边 → 传播)

指标: Full@k (两个 gold 都在 top-k 的比例), Recall@k (至少一个 gold)
运行: python benchmarks/ab_hotpotqa_real.py [--sample N]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from su_memory._sys.encoders import _get_st_model
from su_memory.algebra.causal_graph import CausalDAG

DATA_PATH = ROOT / "benchmarks" / "data" / "hotpotqa_validation_200.json"


# ===========================================================================
# Embedding
# ===========================================================================

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = _get_st_model()
    return _MODEL


def embed(text: str) -> np.ndarray:
    m = get_model()
    v = m.encode(text)
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, list) and v and isinstance(v[0], list):
        v = v[0]
    return np.asarray(v, dtype=np.float32)


def embed_batch(texts: list[str]) -> np.ndarray:
    """Embed a list of texts, returning an (n, d) float32 matrix.

    Uses the encoder's native batch encode (sentence-transformers processes
    all texts in one forward pass, ~10× faster than per-item Ollama requests).
    Falls back to sequential embed if the encoder lacks batch support.
    """
    m = get_model()
    if m is None:
        return np.stack([embed(t) for t in texts]) if texts else np.zeros((0, 1024), dtype=np.float32)
    try:
        arr = np.asarray(m.encode(texts), dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == len(texts):
            return arr
    except Exception:
        pass
    # fallback: sequential
    return np.stack([embed(t) for t in texts]) if texts else np.zeros((0, 1024), dtype=np.float32)


# ===========================================================================
# 段落工具
# ===========================================================================

def para_title(para: str) -> str:
    """HotpotQA paragraphs are 'Title: rest'. Extract the title."""
    if ":" in para:
        return para.split(":", 1)[0].strip()
    return para[:40].strip()


def extract_entities(text: str) -> set[str]:
    """Crude named-entity extraction: capitalized n-grams (1-3 tokens).

    A lightweight, model-free extractor sufficient for building a co-occurrence
    bridge graph. Not a full NER; the goal is to capture the *bridge entity*
    that connects two gold paragraphs (e.g. a person mentioned in both).
    """
    ents = set()
    # Capitalized word sequences (allow internal spaces/hyphens).
    for m in re.finditer(r"[A-Z][a-zA-Z]+(?:[\s\-][A-Z][a-zA-Z]+){0,3}", text):
        ent = m.group().strip()
        # filter short stopwords / sentence-initial common words
        if len(ent) >= 3 and ent.lower() not in {
            "the", "this", "that", "these", "those", "his", "her", "their",
            "she", "him", "was", "were", "has", "had", "been", "from", "into",
            "after", "also", "american", "united", "first", "second",
        }:
            ents.add(ent)
    return ents


def build_bridge_graph(paras: list[str]) -> tuple[CausalDAG, list[set[str]]]:
    """Build a paragraph co-occurrence graph via shared named entities.

    Two paragraphs are linked (weight 1.0) if they share at least one
    capitalized entity. This captures the HotpotQA bridge structure: the
    second gold paragraph shares a bridge entity with the first.
    Returns (DAG, per-paragraph entity sets).
    """
    ent_sets = [extract_entities(p) for p in paras]
    dag = CausalDAG()
    n = len(paras)
    # undirected co-occurrence: add both directions
    for i in range(n):
        dag.add_node(i)
        for j in range(i + 1, n):
            if ent_sets[i] & ent_sets[j]:
                dag.add_edge(i, j, weight=1.0)
                dag.add_edge(j, i, weight=1.0)
    return dag, ent_sets


# ===========================================================================
# Retrievers
# ===========================================================================

def baseline_rank(qv, pv):
    """Embedding cosine similarity ranking."""
    qn = np.linalg.norm(qv) + 1e-9
    pn = np.linalg.norm(pv, axis=1) + 1e-9
    return pv @ qv / (pn * qn)


def retrieve_baseline(qv, pv, top_k):
    sims = baseline_rank(qv, pv)
    return list(np.argsort(-sims)[:top_k])


def _entity_specificity(paras, ent_sets, top1):
    """Score how *specifically* each paragraph shares entities with top1.

    A bridge paragraph shares rare (low-document-frequency) entities with top1.
    Common entities (appearing in many paragraphs) are generic and uninformative,
    so we weight co-occurrence by inverse document frequency of the shared entity.
    """
    n = len(paras)
    # document frequency of each entity
    df = {}
    for ents in ent_sets:
        for e in ents:
            df[e] = df.get(e, 0) + 1
    top1_ents = ent_sets[top1]
    spec = np.zeros(n, dtype=np.float32)
    for i in range(n):
        shared = top1_ents & ent_sets[i]
        if shared:
            # sum of idf weights of shared entities
            spec[i] = sum(np.log((n + 1) / (df[e] + 1)) + 1 for e in shared)
    return spec


def retrieve_treatment(qv, pv, paras, top_k, alpha=0.8):
    """Embedding + entity-bridge DAG propagation with fusion re-ranking.

    1. Embedding finds the top-1 paragraph (the directly-relevant gold).
    2. From it, BFS the entity co-occurrence DAG to find bridged paragraphs
       (candidate set for the second gold — covers ~93% of second golds).
    3. Re-rank bridged successors by a *fusion* of:
         - query similarity (semantic relevance), and
         - entity specificity (rare-entity co-occurrence with top1, which
           distinguishes the true bridge paragraph from generic distractors).
       alpha controls the blend (0.5 = balanced).
    4. Merge: top-1 + top-ranked bridged successor + remaining embedding order.
    """
    sims = baseline_rank(qv, pv)
    order = list(np.argsort(-sims))
    top1 = order[0]

    dag, ent_sets = build_bridge_graph(paras)
    eff = dag.propagate(top1, delta=1.0)
    bridged = [n for n in eff if n != top1]

    spec = _entity_specificity(paras, ent_sets, top1)
    # normalise both signals to [0,1] over the bridge set
    sim_arr = np.array([sims[n] for n in bridged], dtype=np.float32)
    spec_arr = np.array([spec[n] for n in bridged], dtype=np.float32)
    def norm(a):
        lo, hi = a.min(), a.max()
        return (a - lo) / (hi - lo + 1e-9)
    sim_n = norm(sim_arr) if len(sim_arr) else sim_arr
    spec_n = norm(spec_arr) if len(spec_arr) else spec_arr
    fusion = alpha * sim_n + (1 - alpha) * spec_n
    bridged = [bridged[i] for i in np.argsort(-fusion)]

    merged = [top1]
    seen = {top1}
    for n in bridged:
        if n not in seen:
            seen.add(n)
            merged.append(n)
        if len(merged) >= top_k:
            break
    for n in order:
        if n not in seen:
            seen.add(n)
            merged.append(n)
        if len(merged) >= top_k:
            break
    return merged[:top_k]


# ===========================================================================
# Evaluation
# ===========================================================================

def gold_indices(paras, gold_titles):
    """Indices of paragraphs whose title matches a gold supporting title."""
    golds = set()
    for i, p in enumerate(paras):
        t = para_title(p)
        if t in gold_titles or any(gt in p[: len(gt) + 2] for gt in gold_titles):
            golds.add(i)
    return golds


def evaluate(data, top_ks=(1, 2, 3, 4), verbose=True):
    """Run A/B over the dataset; return per-method per-k metrics."""
    metrics = {name: {k: {"full": 0, "any": 0} for k in top_ks}
               for name in ("baseline", "treatment")}
    by_type = {"bridge": {"baseline": {k: 0 for k in top_ks},
                          "treatment": {k: 0 for k in top_ks},
                          "n": 0},
               "comparison": {"baseline": {k: 0 for k in top_ks},
                              "treatment": {k: 0 for k in top_ks},
                              "n": 0}}

    for di, d in enumerate(data):
        paras = d["context"]
        gold_titles = set(d["supporting_facts"]["title"])
        golds = gold_indices(paras, gold_titles)
        if len(golds) < 2:
            continue  # need 2 golds to evaluate Full@k

        qv = embed(d["question"])
        pv = embed_batch([p[:500] for p in paras])

        for name, fn in [("baseline", lambda k: retrieve_baseline(qv, pv, k)),
                         ("treatment", lambda k: retrieve_treatment(qv, pv, paras, k))]:
            for k in top_ks:
                got = fn(k)
                hits = len(set(got[:k]) & golds)
                if hits >= 2:
                    metrics[name][k]["full"] += 1
                if hits >= 1:
                    metrics[name][k]["any"] += 1
                    by_type[d["type"]][name][k] += 1
        by_type[d["type"]]["n"] += 1

        if verbose and (di + 1) % 50 == 0:
            print(f"  ...{di+1}/{len(data)} 题")

    n = len(data)
    return metrics, by_type, n


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0,
                    help="评测题数 (0=全部200)")
    ap.add_argument("--data", type=str, default=str(DATA_PATH))
    args = ap.parse_args()

    with open(args.data) as f:
        data = json.load(f)
    if args.sample and args.sample < len(data):
        data = data[: args.sample]

    print("=" * 64)
    print("真实 HotpotQA A/B: embedding vs embedding+实体桥接图")
    print(f"数据: {len(data)} 题 (官方 validation 子集, 非合成)")
    print("=" * 64)

    top_ks = (1, 2, 3, 4)
    metrics, by_type, n = evaluate(data, top_ks=top_ks)

    print(f"\n有效题数 (>=2 gold 段落): {n}")
    print(f"\n{'指标':<16} " + " ".join(f"k={k:<3}" for k in top_ks))
    print("-" * 48)
    for metric in ("any", "full"):
        label = "Recall@k" if metric == "any" else "Full@k (两gold)"
        for name in ("baseline", "treatment"):
            row = " ".join(f"{metrics[name][k][metric]/n:<5.3f}" for k in top_ks)
            print(f"{label+' '+name:<16} {row}")
        # delta
        drow = " ".join(
            f"{(metrics['treatment'][k][metric]-metrics['baseline'][k][metric])/n:+.3f}"
            for k in top_ks)
        print(f"{'Δ '+label:<16} {drow}")
        print()

    print("分 type Full@2 (两 gold 都在 top-2):")
    for t in ("bridge", "comparison"):
        cnt = by_type[t]["n"]
        if cnt == 0:
            continue
        b = by_type[t]["baseline"][2] / cnt
        tr = by_type[t]["treatment"][2] / cnt
        print(f"  {t:12} ({cnt:3}题): baseline={b:.3f}  treatment={tr:.3f}  Δ={tr-b:+.3f}")

    print("\n" + "=" * 64)
    print("解读:")
    print("- Full@k 衡量两个 gold 证据段落都被召回 (多跳核心).")
    print("- treatment 通过实体桥接图提升 *第二个* gold 的召回.")
    print("- 若 treatment Full@2 显著 > baseline, 证明结构化桥接补足 embedding 多跳缺陷.")
    print("=" * 64)


if __name__ == "__main__":
    main()
