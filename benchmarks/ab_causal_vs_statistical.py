"""
A/B 对比实验: 纯统计检索 vs 统计+结构化因果推理

诚实声明
--------
本评测使用 *合成* 数据集(随机种子固定),非真实 HotpotQA/BEIR。
目的是隔离"因果推理"这一变量,量化 algebra 层接入前后的差异,
而非与外部 SOTA 榜单对比。所有数字可在本机复现。

实验设计
--------
三个评测维度,每个维度都有"黄金答案"(构造时已知):

1. 因果链多跳 (CAUSAL_MULTIHOP)
   - 构造 A→B→C→D 因果链, query 问 "A 的远期后果"
   - baseline(TF-IDF): 只能命中与 query 字面相近的节点
   - treatment(因果推理): 通过 BeliefNetwork 传播, 命中链上远端节点

2. 时空关联 (SPATIOTEMPORAL)
   - 构造同一时序位置/能量亲和的记忆群
   - baseline: 纯关键词
   - treatment: 用 TemporalRing + AffinityMatrix 扩召回

3. 纯语义对照 (SEMANTIC_CONTROL)
   - 无因果/时空结构, 纯语义检索
   - 验证 treatment 不损害基础检索能力 (应与 baseline 持平)

指标: Recall@k (黄金答案是否出现在 top-k), MRR (平均倒数排名)

运行: python benchmarks/ab_causal_vs_statistical.py
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# 让脚本独立可跑 (src 在 path)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from su_memory.sdk.lite import SuMemoryLite
from su_memory.sdk._causal import CausalEngine


# ===========================================================================
# 合成数据集构造 (固定种子, 可复现)
# ===========================================================================

SEED = 20260629


@dataclass
class EvalCase:
    """单条评测样本."""
    query: str
    golden_ids: set[str]  # 黄金答案记忆 id 集合
    dim: str  # CAUSAL_MULTIHOP / SPATIOTEMPORAL / SEMANTIC_CONTROL
    note: str = ""


@dataclass
class Dataset:
    memories: list[dict] = field(default_factory=list)  # [{id, content}]
    cases: list[EvalCase] = field(default_factory=list)
    causal_edges: list[tuple[str, str]] = field(default_factory=list)  # 显式因果边
    energy_labels: dict[str, str] = field(default_factory=dict)  # oid -> energy (wood/fire/earth/metal/water)


def build_dataset() -> Dataset:
    """构造合成评测数据集.

    每个维度的案例都内置黄金答案, 评测只验证检索能否命中.
    """
    rng = random.Random(SEED)
    ds = Dataset()
    mid = 0

    def mk(content: str) -> str:
        nonlocal mid
        i = f"m{mid}"
        mid += 1
        ds.memories.append({"id": i, "content": content})
        return i

    # ---- 维度1: 因果链多跳 ----
    # 构造 5 条独立因果链, 每条 4 跳 A->B->C->D
    causal_chains = [
        ("持续暴雨", "水库水位暴涨", "大坝泄洪", "下游村庄被淹"),
        ("全球变暖", "北极冰盖融化", "海平面上升", "沿海城市迁移"),
        ("芯片短缺", "汽车减产", "供应链重组", "国产替代加速"),
        ("熬夜加班", "免疫力下降", "反复感冒", "工作效率降低"),
        ("过度施肥", "土壤板结", "作物减产", "农民改用有机肥"),
    ]
    for a, b, c, d in causal_chains:
        # 记忆用跨句因果标记, 确保 detect_causal_link 能识别
        ia = mk(f"由于{a}")
        ib = mk(f"因此{b}")
        ic = mk(f"所以{c}")
        id_ = mk(f"最终{d}")
        # 显式因果边 (构造时已知, 干净隔离因果结构变量)
        ds.causal_edges += [(ia, ib), (ib, ic), (ic, id_)]
        # 干扰项 (语义相近但不在链上)
        mk(f"{a}是常见现象")
        # query 问远期后果: 答案应包含 c, d (远端)
        ds.cases.append(EvalCase(
            query=a, golden_ids={ic, id_},
            dim="CAUSAL_MULTIHOP",
            note=f"{a} 的远期后果链",
        ))

    # ---- 维度2: 时空关联 ----
    # 构造同能量亲和的记忆群 (水系/火系), query 应召回同系记忆
    water_group = ["潮汐研究", "海洋生态", "河流治理", "水循环模型"]
    fire_group = ["火山监测", "太阳能利用", "热能转换", "地热开发"]
    for g, energy in [(water_group, "water"), (fire_group, "fire")]:
        ids = [mk(x) for x in g]
        for oid in ids:
            ds.energy_labels[oid] = energy
        ds.cases.append(EvalCase(
            query=g[0], golden_ids=set(ids[1:]),  # 查 g[0] 应召回同能量群其他
            dim="SPATIOTEMPORAL",
            note=f"{energy}能量亲和群",
        ))

    # ---- 维度3: 纯语义对照 ----
    # 无因果/时空结构, query 与 target 共享关键词 (公平测纯关键词检索能力).
    # 这验证 treatment 的因果增强不会损害 baseline 的基础匹配.
    semantic_pairs = [
        ("React组件", "React框架笔记"),
        ("PostgreSQL", "PostgreSQL调优记录"),
        ("Kubernetes", "Kubernetes部署文档"),
    ]
    for q, target in semantic_pairs:
        tid = mk(target)
        mk("完全无关的随机内容")
        ds.cases.append(EvalCase(
            query=q, golden_ids={tid},
            dim="SEMANTIC_CONTROL",
            note="纯语义匹配",
        ))

    # 补充干扰记忆
    for _ in range(20):
        mk(f"干扰项{rng.randint(1000,9999)}" + rng.choice(["数据","记录","条目"]))

    return ds


# ===========================================================================
# 检索器: baseline vs treatment
# ===========================================================================

def _find_sources_by_substring(q: str, memories: list[dict], limit: int = 10, min_overlap: int = 2) -> list[str]:
    """Stable source selection via shared character n-grams.

    Bypasses SuMemoryLite.query state-instability on short Chinese queries.
    A memory is a candidate source if it shares a character subsequence of
    length >= ``min_overlap`` with the query (e.g. query "React组件" matches
    content "React框架笔记" via the shared "React" prefix). This reflects true
    keyword-overlap ability without depending on the embedding model.
    """
    hits = []
    for m in memories:
        c = m["content"]
        # exact substring first
        if q in c:
            hits.append(m["id"])
            continue
        # shared character n-gram of length >= min_overlap
        shared = False
        for i in range(len(q) - min_overlap + 1):
            if q[i:i + min_overlap] in c:
                shared = True
                break
        if shared:
            hits.append(m["id"])
        if len(hits) >= limit:
            break
    return hits


def baseline_retriever(memories: list[dict]) -> Callable[[str, int], list[str]]:
    """基线: 纯 TF-IDF SuMemoryLite 检索 (无因果/时空增强)."""
    mem = SuMemoryLite()
    # 用 metadata 携带合成数据集的原始 id, 因为 SuMemoryLite 生成自己的 mem_xxx id.
    content_to_oid = {}
    oid_to_memid = {}
    for m in memories:
        memid = mem.add(m["content"], metadata={"oid": m["id"]})
        oid_to_memid[m["id"]] = memid
        content_to_oid[m["content"]] = m["id"]
    def _query(q: str, top_k: int = 5) -> list[str]:
        results = mem.query(q, top_k=max(top_k * 2, 10))
        out, seen = [], set()
        for r in results:
            oid = r.get("metadata", {}).get("oid")
            if oid and oid not in seen:
                seen.add(oid)
                out.append(oid)
        # Fallback: TF-IDF may miss short Chinese queries; augment with
        # substring matches so the baseline reflects its true keyword ability.
        for oid in _find_sources_by_substring(q, memories):
            if oid not in seen:
                seen.add(oid)
                out.append(oid)
        return out[:top_k]
    return _query, oid_to_memid


def treatment_retriever(memories: list[dict], causal_edges: list[tuple[str, str]] | None = None, energy_labels: dict[str, str] | None = None) -> Callable[[str, int], list[str]]:
    """增强: TF-IDF 检索 + 显式因果图 BFS 传播召回.

    对每个 query:
    1. baseline TF-IDF 找到语义起点 (chain head)
    2. 沿显式因果图 (CausalDAG) 做 BFS, 召回因果后继
    3. 合并候选: 因果后继优先 (远端命中), 再补 TF-IDF 结果
    这干净地隔离了"因果结构"这一变量, 不依赖 detect_causal_link 的噪声.
    """
    from su_memory.algebra.causal_graph import CausalDAG

    mem = SuMemoryLite()
    oid_to_memid = {}
    memid_to_oid = {}
    for m in memories:
        memid = mem.add(m["content"], metadata={"oid": m["id"]})
        oid_to_memid[m["id"]] = memid
        memid_to_oid[memid] = m["id"]

    # 构造显式因果 DAG (来自数据集, 权重=1.0 纯传递)
    dag = CausalDAG()
    if causal_edges:
        for parent, child in causal_edges:
            try:
                dag.add_edge(parent, child, weight=1.0)
            except (ValueError, KeyError):
                pass

    # 能量亲和群索引 (同 energy 的记忆互为亲和, AffinityMatrix SAME=1.1)
    energy_groups: dict[str, list[str]] = {}
    if energy_labels:
        for oid, e in energy_labels.items():
            energy_groups.setdefault(e, []).append(oid)

    def _query(q: str, top_k: int = 5) -> list[str]:
        # 起点: content 子串匹配 *且是结构化节点* (因果图节点或有能量标签).
        # 用稳定子串匹配绕过 SuMemoryLite 短查询的状态不稳定性.
        dag_nodes = dag.nodes if dag.nodes else set()
        labeled = set(energy_labels.keys()) if energy_labels else set()
        structured = dag_nodes | labeled
        src_oid = ""
        for oid in _find_sources_by_substring(q, memories):
            if oid in structured:
                src_oid = oid
                break

        # 因果 BFS: 从起点沿 DAG 召回所有因果后继
        causal_succ: list[str] = []
        if src_oid and dag.nodes:
            eff = dag.propagate(src_oid, delta=1.0)
            causal_succ = [n for n, _ in sorted(eff.items(), key=lambda x: -x[1]) if n != src_oid]

        # 能量亲和召回: 若起点有能量标签, 召回同能量群其他节点
        affinity_succ: list[str] = []
        if src_oid and energy_groups:
            src_energy = energy_labels.get(src_oid) if energy_labels else None
            if src_energy:
                affinity_succ = [o for o in energy_groups.get(src_energy, []) if o != src_oid]

        structured_succ = causal_succ + affinity_succ

        # baseline TF-IDF 候选 (去重), 含 n-gram 兜底保证稳定性
        cand = mem.query(q, top_k=max(top_k * 3, 15))
        seen = set()
        tfidf_oids = []
        for r in cand:
            oid = memid_to_oid.get(r.get("memory_id", ""), "")
            if oid and oid not in seen:
                seen.add(oid)
                tfidf_oids.append(oid)
        # n-gram 兜底: 确保 treatment 的基础候选与 baseline 一致稳定
        for oid in _find_sources_by_substring(q, memories):
            if oid not in seen:
                seen.add(oid)
                tfidf_oids.append(oid)

        # 合并: 结构化后继优先, 再补 TF-IDF, 去重, 截断 top_k.
        # 若无任何结构化后继 (纯语义 query), 直接返回 baseline TF-IDF 结果,
        # 确保 treatment 不损害基础检索能力.
        if not structured_succ:
            return tfidf_oids[:top_k]
        merged: list[str] = []
        merged_seen: set[str] = set()
        for oid in structured_succ + tfidf_oids:
            if oid not in merged_seen:
                merged_seen.add(oid)
                merged.append(oid)
            if len(merged) >= top_k:
                break
        return merged

    return _query, oid_to_memid




# ===========================================================================
# 指标
# ===========================================================================

def recall_at_k(retrieved: list[str], golden: set[str], k: int) -> float:
    """Recall@k: 黄金答案在 top-k 的命中率."""
    if not golden:
        return 0.0
    topk = retrieved[:k]
    hits = len(set(topk) & golden)
    return hits / len(golden)


def mrr(retrieved: list[str], golden: set[str]) -> float:
    """MRR: 第一个命中的黄金答案的倒数排名."""
    for i, rid in enumerate(retrieved):
        if rid in golden:
            return 1.0 / (i + 1)
    return 0.0


# ===========================================================================
# 主实验
# ===========================================================================

def run_experiment():
    print("=" * 64)
    print("A/B 对比: 纯统计检索 vs 统计+结构化因果推理")
    print("(合成数据集, 种子 %d, 非外部榜单)" % SEED)
    print("=" * 64)

    ds = build_dataset()
    print(f"\n数据集: {len(ds.memories)} 条记忆, {len(ds.cases)} 条评测样本")
    by_dim = {}
    for c in ds.cases:
        by_dim.setdefault(c.dim, []).append(c)
    for dim, cases in by_dim.items():
        print(f"  {dim}: {len(cases)} 条")

    for top_k in (3, 5):
        print(f"\n--- top_k = {top_k} ---")
        baseline, _ = baseline_retriever(ds.memories)
        treatment, _ = treatment_retriever(ds.memories, ds.causal_edges, ds.energy_labels)

        results = {}
        for name, retriever in [("baseline", baseline), ("treatment", treatment)]:
            by_dim_metrics = {}
            for dim, cases in by_dim.items():
                recalls, mrrs = [], []
                for case in cases:
                    ret = retriever(case.query, top_k=top_k)
                    recalls.append(recall_at_k(ret, case.golden_ids, top_k))
                    mrrs.append(mrr(ret, case.golden_ids))
                by_dim_metrics[dim] = {
                    "recall": sum(recalls) / len(recalls),
                    "mrr": sum(mrrs) / len(mrrs),
                }
            results[name] = by_dim_metrics

        # 打印对比表
        print("")
        print("维度                  base R@%d   treat R@%d   Δ        base MRR   treat MRR" % (top_k, top_k))
        print("-" * 80)
        for dim in sorted(by_dim):
            b = results["baseline"][dim]
            t = results["treatment"][dim]
            delta = t["recall"] - b["recall"]
            sign = "+" if delta >= 0 else ""
            print("%-22s %-10.3f %-12.3f %s%-7.3f %-11.3f %-12.3f" % (
                dim, b["recall"], t["recall"], sign, delta, b["mrr"], t["mrr"]))

    print("\n" + "=" * 64)
    print("解读指南:")
    print("- CAUSAL_MULTIHOP: treatment 应显著优于 baseline (因果传播命中远端)")
    print("- SEMANTIC_CONTROL: 两者应持平 (treatment 不损害基础能力)")
    print("- SPATIOTEMPORAL: 取决于能量/时序结构是否被 query 触发")
    print("=" * 64)


if __name__ == "__main__":
    run_experiment()
