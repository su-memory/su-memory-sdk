"""
algebra 层消融实验 — 为专利创造性提供技术效果证据。

不依赖 reader，测纯检索质量：
  - supporting-fact 标题召回率 (Full@k): 召回的 top-k 段落是否覆盖全部 gold 段落
  - 桥接发现率: bridge 题（需跨段落推理）的第二证据召回成功率

配置:
  A. baseline:    仅 direct (纯向量余弦)
  B. +title:      direct + title-bridge (传统实体→标题匹配)
  C. +causaldag:  direct + entity-bridge (CausalDAG 罕见实体桥接) ← 本专利核心
  D. full:        三路融合 (direct + title + CausalDAG)

运行: python benchmarks/ablation_algebra.py [--sample N] [--top-k K]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from su_memory.sdk.multi_hop_reader import MultiHopReader
from su_memory.sdk._bridge_recall import extract_entities


def para_title(p: str) -> str:
    """提取段落标题 (HotpotQA 格式: 'Title: content')."""
    if ":" in p[:80]:
        return p.split(":")[0].strip().lower()
    return ""


def gold_titles(item: dict) -> set[str]:
    """从 supporting_facts 提取 gold 段落标题集合。"""
    sf = item.get("supporting_facts", {})
    return set(t.strip().lower() for t in sf.get("title", []))


def all_titles(item: dict) -> list[str]:
    """context 中所有段落标题。"""
    return [para_title(p) for p in item.get("context", [])]


def recall_at_k(retrieved_titles: list[str], gold: set[str], k: int) -> float:
    """Full@k: top-k 召回是否覆盖全部 gold（多跳需全召回）。"""
    if not gold:
        return 0.0
    topk = set(retrieved_titles[:k])
    return 1.0 if gold <= topk else len(gold & topk) / len(gold)


class AblationReader:
    """可配置的三路融合检索器，支持逐路开关。"""

    def __init__(self, embed_fn, embed_batch_fn, use_title=True, use_causaldag=True):
        self._embed = embed_fn
        self._embed_batch = embed_batch_fn
        self.use_title = use_title
        self.use_causaldag = use_causaldag

    def retrieve(self, query: str, paragraphs: list[str], top_k: int = 5):
        """返回排序后的段落索引列表 + 每段的标题。"""
        if not paragraphs:
            return []
        paras = [p[:600] for p in paragraphs]
        qv = self._embed(query)
        pv = self._embed_batch(paras)
        sims = (pv @ qv) / (np.linalg.norm(pv, axis=1) * np.linalg.norm(qv) + 1e-9)
        order = list(np.argsort(-sims))
        top1 = order[0]

        sources = [order[1:]]  # direct 剩余

        # 路2: title-bridge
        if self.use_title:
            hop1_ents = extract_entities(paras[top1])
            titles = [para_title(p) for p in paras]
            title_hits = [
                i for i, t in enumerate(titles)
                if i != top1 and any(t == e or t in e or e in t for e in hop1_ents)
            ]
            sources.append(title_hits)
        else:
            sources.append([])

        # 路3: entity-bridge (CausalDAG)
        if self.use_causaldag:
            bridge = self._entity_bridge(paras, top1)
            sources.append(bridge)
        else:
            sources.append([])

        # 交错融合
        union = [top1]
        seen = {top1}
        si = 0
        guard = 0
        while len(union) < max(top_k, 4) and any(sources) and guard < 1000:
            guard += 1
            src = sources[si % len(sources)]
            if src:
                i = src.pop(0)
                if i not in seen:
                    seen.add(i)
                    union.append(i)
            si += 1
        for i in order:
            if len(union) >= top_k:
                break
            if i not in seen:
                seen.add(i)
                union.append(i)
        return union

    def _entity_bridge(self, paras, top1):
        """CausalDAG 罕见实体桥接（IDF 加权, DF<=3）。"""
        import math
        from su_memory.algebra.causal_graph import CausalDAG
        ent_sets = [extract_entities(p) for p in paras]
        n = len(paras)
        df = {}
        for ents in ent_sets:
            for e in ents:
                df[e] = df.get(e, 0) + 1
        dag = CausalDAG()
        for i in range(n):
            dag.add_node(i)
        inv = {}
        for i, ents in enumerate(ent_sets):
            for e in ents:
                inv.setdefault(e, []).append(i)
        for docs in inv.values():
            for a in range(len(docs)):
                for b in range(a + 1, len(docs)):
                    dag.add_edge(docs[a], docs[b], 1.0)
                    dag.add_edge(docs[b], docs[a], 1.0)
        eff = dag.propagate(top1, 1.0)
        seed_ents = ent_sets[top1]
        scored = []
        for j in eff:
            if j == top1:
                continue
            shared = seed_ents & ent_sets[j]
            if not shared:
                continue
            rare = [e for e in shared if df[e] <= 3]
            if not rare:
                continue
            spec = sum(math.log((n + 1) / (df[e] + 1)) + 1 for e in rare)
            scored.append((j, spec))
        scored.sort(key=lambda x: -x[1])
        return [j for j, _ in scored]


def make_embed_fns(use_real=False):
    """构造 embedding 函数。默认 hash（快速），--real-embed 用 bge-m3。"""
    if use_real:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("BAAI/bge-small-en-v1.5")
            def embed(text):
                return model.encode(text, normalize_embeddings=True)
            def embed_batch(texts):
                return model.encode(texts, normalize_embeddings=True, batch_size=32)
            print("[embed] bge-small-en-v1.5")
            return embed, embed_batch
        except Exception as e:
            print(f"[embed] bge加载失败({e}), 回退hash")
    # 回退: 简单 hash embedding (512维)
    print("[embed] 回退 hash embedding (512维)")
    def hash_embed(text):
        h = np.zeros(512)
        for tok in re.findall(r"\w+", text.lower()):
            idx = hash(tok) % 512
            h[idx] += 1.0
        n = np.linalg.norm(h)
        return h / n if n > 0 else h
    def hash_batch(texts):
        return np.array([hash_embed(t) for t in texts])
    return hash_embed, hash_batch


def run_ablation(data, top_k, embed_fn, embed_batch_fn):
    """跑4个配置，返回各配置的 Full@k 和 bridge 发现率。"""
    configs = {
        "A.baseline(仅direct)": AblationReader(embed_fn, embed_batch_fn, False, False),
        "B.+title(传统桥接)": AblationReader(embed_fn, embed_batch_fn, True, False),
        "C.+causaldag(本专利)": AblationReader(embed_fn, embed_batch_fn, False, True),
        "D.full(三路融合)": AblationReader(embed_fn, embed_batch_fn, True, True),
    }
    results = {name: {"full_recall": [], "bridge_recall": [], "comp_recall": []}
               for name in configs}

    for idx, item in enumerate(data):
        q = item["question"]
        paras = item["context"]
        gold = gold_titles(item)
        titles = all_titles(item)
        qtype = item.get("type", "")
        if not gold:
            continue
        for name, reader in configs.items():
            ranked = reader.retrieve(q, paras, top_k=top_k)
            retrieved_titles = [titles[i] if i < len(titles) else "" for i in ranked]
            r = recall_at_k(retrieved_titles, gold, top_k)
            results[name]["full_recall"].append(r)
            if qtype == "bridge":
                results[name]["bridge_recall"].append(r)
            elif qtype == "comparison":
                results[name]["comp_recall"].append(r)
        if (idx + 1) % 20 == 0:
            print(f"  已处理 {idx+1}/{len(data)}")

    return results


def main():
    ap = argparse.ArgumentParser(description="algebra 层消融实验")
    ap.add_argument("--sample", type=int, default=50, help="评测题数")
    ap.add_argument("--top-k", type=int, default=5, help="召回数")
    ap.add_argument("--data", default="benchmarks/data/hotpotqa_validation_200.json")
    ap.add_argument("--real-embed", action="store_true", help="用bge-small(慢但准)")
    args = ap.parse_args()

    data_path = ROOT / args.data
    with open(data_path) as f:
        data = json.load(f)
    if args.sample > 0:
        data = data[:args.sample]
    print(f"数据集: {data_path.name}, 评测 {len(data)} 题, top_k={args.top_k}\n")

    embed_fn, embed_batch_fn = make_embed_fns(use_real=args.real_embed)

    t0 = time.time()
    results = run_ablation(data, args.top_k, embed_fn, embed_batch_fn)
    elapsed = time.time() - t0

    print(f"\n{'='*65}")
    print(f"消融实验结果 (top_k={args.top_k}, {len(data)}题, {elapsed:.1f}s)")
    print(f"{'='*65}")
    print(f"{'配置':<28} {'Full@k':>8} {'bridge':>8} {'comparison':>10}")
    print(f"{'-'*65}")
    for name, res in results.items():
        full = np.mean(res["full_recall"]) * 100 if res["full_recall"] else 0
        br = np.mean(res["bridge_recall"]) * 100 if res["bridge_recall"] else 0
        cp = np.mean(res["comp_recall"]) * 100 if res["comp_recall"] else 0
        print(f"{name:<28} {full:>7.1f}% {br:>7.1f}% {cp:>9.1f}%")

    # 保存结果
    out = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "sample": len(data), "top_k": args.top_k, "elapsed_s": round(elapsed, 1),
        "results": {name: {
            "full_recall_pct": round(np.mean(r["full_recall"]) * 100, 1) if r["full_recall"] else 0,
            "bridge_recall_pct": round(np.mean(r["bridge_recall"]) * 100, 1) if r["bridge_recall"] else 0,
            "comparison_recall_pct": round(np.mean(r["comp_recall"]) * 100, 1) if r["comp_recall"] else 0,
            "n_bridge": len(r["bridge_recall"]), "n_comparison": len(r["comp_recall"]),
        } for name, r in results.items()}
    }
    out_path = ROOT / "benchmarks/results/ablation_algebra.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
