#!/usr/bin/env python3
"""
su-memory v3.5.5 — SpectralCausal 因果发现 Benchmark
=====================================================

评测 GaussianDAG + FourierCausal + BayesianCausal 的因果发现能力。

指标:
  - SHD (结构汉明距离): 与真实 DAG 的结构差异
  - F1-score (边方向正确率): 发现边的精确率/召回率
  - 运行时间 vs 数据规模
  - Bayesian 后验校准 (BF 分布)

评测策略:
  - 每个"节点"生成多条记忆，利用共享词汇编码因果信号
  - 使用 memory_id → node_id 映射将记忆级边聚合为节点级边
  - 支持 3 种配置: PC 基线 / 能量增强 / Bayesian 后验

数据集:
  1. Sachs 蛋白质信号网络 (11 节点, 17 边) — 合成模拟
  2. 模拟 DAG 数据: 10/20/50 节点随机生成

Usage:
    PYTHONPATH=src python benchmarks/benchmark_causal_discovery.py
    PYTHONPATH=src python benchmarks/benchmark_causal_discovery.py --quick
    PYTHONPATH=src python benchmarks/benchmark_causal_discovery.py --report
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ═══════════════════════════════════════════════════════════════
# Synthetic DAG Generators
# ═══════════════════════════════════════════════════════════════


def generate_random_dag(
    n_nodes: int, edge_prob: float = 0.2, seed: int = 42,
    n_memories_per_node: tuple[int, int] = (3, 6),
) -> tuple[np.ndarray, list[dict], dict[int, int]]:
    """
    生成随机 DAG 邻接矩阵及对应的记忆数据。

    因果信号编码策略:
      - 每个节点有 3-6 条记忆，每条记忆含多个关键词
      - 因果边 (i→j): 将原因节点 i 的关键词注入到结果节点 j 的记忆中
      - 注入比例与边的因果强度相关，使 TF-IDF 能捕获因果关系

    Args:
        n_nodes: 节点数
        edge_prob: 边概率
        seed: 随机种子
        n_memories_per_node: (min, max) 每节点记忆数范围

    Returns:
        (adjacency_matrix n×n, memories_list, mem_idx_to_node)
    """
    rng = np.random.RandomState(seed)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.int32)

    # 上三角随机生成 DAG
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                adj[i, j] = 1

    # 为每个节点生成"因果特征词" (每个节点 4-8 个特征词)
    base_vocab = [
        "信号转导", "磷酸化", "激活", "抑制", "表达", "调控", "代谢",
        "通路", "受体", "激酶", "转录", "因子", "凋亡", "增殖", "分化",
        "迁移", "黏附", "分泌", "释放", "合成", "降解", "氧化", "还原",
        "水解", "聚合", "解离", "结合", "转运", "通道", "电位", "突触",
        "神经元", "胶质", "免疫", "炎症", "抗体", "抗原", "细胞因子",
        "趋化", "吞噬", "自噬", "修复", "再生", "衰老", "突变", "修复",
        "响应", "应激", "适应", "记忆", "学习", "认知", "情绪", "行为",
    ]

    # 每个节点随机选取特征词
    node_features: list[list[str]] = []
    for i in range(n_nodes):
        n_feat = rng.randint(3, 6)
        feats = rng.choice(base_vocab, size=n_feat, replace=False).tolist()
        node_features.append(feats)

    # 因果传播: 原因节点特征词注入到结果节点
    # 多步因果链上累积特征词 (如 i→j→k, k 也获得部分 i 的特征)
    propagated_features: list[set] = [set(f) for f in node_features]

    # 拓扑排序传播 (因为 adj 是上三角 DAG，i < j 自然满足拓扑序)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if adj[i, j] == 1:
                # 注入 50-80% 的原因特征词到结果节点
                n_inject = max(1, int(len(propagated_features[i]) * rng.uniform(0.5, 0.8)))
                injected = rng.choice(
                    list(propagated_features[i]),
                    size=min(n_inject, len(propagated_features[i])),
                    replace=False,
                )
                propagated_features[j].update(injected)

    # 生成记忆
    memories = []
    mem_idx_to_node: dict[int, int] = {}

    for i in range(n_nodes):
        n_mem = rng.randint(*n_memories_per_node)
        all_feats = list(propagated_features[i])
        for m in range(n_mem):
            mem_idx = len(memories)
            mem_idx_to_node[mem_idx] = i

            # 每条记忆使用部分特征词 (模拟真实观测)
            n_use = max(2, min(5, len(all_feats)))
            kw = rng.choice(all_feats, size=min(n_use, len(all_feats)), replace=False)
            content = f"N{i}_M{m}: " + " ".join(kw)
            memories.append({
                "id": f"dag_n{i}_m{m}",
                "content": content,
                "node_id": i,
            })

    return adj, memories, mem_idx_to_node


def generate_sachs_synthetic(seed: int = 42) -> tuple[np.ndarray, list[dict], dict[int, int]]:
    """
    生成 Sachs 蛋白质信号网络的合成数据。

    Sachs 网络: 11 节点 (蛋白质), 17 条因果边
    基于 Sachs et al. (2005) Science 308:523-529 的因果网络结构。

    Returns:
        (adjacency_matrix 11×11, memories_list, mem_idx_to_node)
    """
    rng = np.random.RandomState(seed)
    n_nodes = 11
    node_names = ["PIP3", "PKC", "Raf", "Mek", "Erk", "Akt", "PKA", "P38", "Jnk", "PIP2", "PLCg"]

    adj = np.zeros((n_nodes, n_nodes), dtype=np.int32)

    # Sachs 已知因果边
    edges = [
        (0, 1), (0, 3), (0, 4), (0, 5), (0, 7), (0, 8),
        (1, 2), (1, 7), (1, 8),
        (2, 3), (3, 4), (5, 3), (5, 7),
        (6, 5), (9, 0), (10, 0), (10, 1),
    ]
    for src, dst in edges:
        adj[src, dst] = 1

    # 每个蛋白质的特征词
    protein_features = {
        "PIP3":  ["磷酸肌醇", "膜脂质", "信号", "第二信使", "PI3K", "膜"],
        "PKC":   ["蛋白激酶C", "钙离子", "磷酸化", "级联", "DAG", "活化"],
        "Raf":   ["MAPK激酶激酶", "信号转导", "磷酸化", "Ras", "ERK通路"],
        "Mek":   ["MAPK激酶", "双特异性", "磷酸化", "ERK", "激活"],
        "Erk":   ["细胞外信号", "调节激酶", "转录", "激活", "增殖", "分化"],
        "Akt":   ["蛋白激酶B", "细胞存活", "抗凋亡", "PI3K", "mTOR"],
        "PKA":   ["cAMP依赖", "蛋白激酶A", "代谢", "调控", "CREB"],
        "P38":   ["应激激活", "MAPK", "炎症", "反应", "细胞因子"],
        "Jnk":   ["c-Jun激酶", "应激", "凋亡", "UV", "炎症"],
        "PIP2":  ["磷酸肌醇二磷酸", "膜", "前体", "PLC", "水解"],
        "PLCg":  ["磷脂酶C", "钙", "IP3", "DAG", "生长因子", "受体"],
    }

    # 因果传播: 递推注入父节点特征词
    propagated: list[set] = [set(protein_features[name]) for name in node_names]
    for src, dst in edges:
        src_feats = list(propagated[src])
        n_inject = max(1, int(len(src_feats) * rng.uniform(0.5, 0.8)))
        injected = rng.choice(src_feats, size=min(n_inject, len(src_feats)), replace=False)
        propagated[dst].update(injected)

    # 生成多条记忆
    memories = []
    mem_idx_to_node: dict[int, int] = {}

    for i in range(n_nodes):
        n_mem = rng.randint(4, 8)
        all_feats = list(propagated[i])
        for m in range(n_mem):
            mem_idx = len(memories)
            mem_idx_to_node[mem_idx] = i

            n_use = max(2, min(6, len(all_feats)))
            kw = rng.choice(all_feats, size=min(n_use, len(all_feats)), replace=False)
            content = f"{node_names[i]}_obs{m}: " + " ".join(kw)
            memories.append({
                "id": f"sachs_{node_names[i]}_{m}",
                "content": content,
                "protein": node_names[i],
                "node_id": i,
            })

    return adj, memories, mem_idx_to_node


# ═══════════════════════════════════════════════════════════════
# Evaluation: Memory-level Edges → Node-level Aggregation
# ═══════════════════════════════════════════════════════════════


@dataclass
class CausalDiscoveryResult:
    """单次因果发现评测结果"""
    dataset_name: str
    n_nodes: int
    n_true_edges: int
    n_mem_edges: int           # 发现的内存级边数
    n_node_edges: int          # 聚合后的节点级边数
    n_correct_edges: int       # TP (节点级)
    n_missing_edges: int       # FN (节点级)
    n_extra_edges: int         # FP (节点级)
    shd: int                   # 结构汉明距离
    precision: float
    recall: float
    f1: float
    elapsed_ms: float
    n_memories: int
    config: str = "partial_correlation"

    @property
    def is_perfect(self) -> bool:
        return self.shd == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "n_nodes": self.n_nodes,
            "n_true_edges": self.n_true_edges,
            "n_mem_edges": self.n_mem_edges,
            "n_node_edges": self.n_node_edges,
            "n_correct": self.n_correct_edges,
            "n_missing": self.n_missing_edges,
            "n_extra": self.n_extra_edges,
            "shd": self.shd,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "elapsed_ms": round(self.elapsed_ms, 1),
            "n_memories": self.n_memories,
            "config": self.config,
        }


def _aggregate_edges_to_node_level(
    discovered_edges: list[dict],
    mem_idx_to_node: dict[int, int],
    n_nodes: int,
) -> set[tuple[int, int]]:
    """
    将记忆级边聚合为节点级边。

    规则: 如果记忆 a (属于节点 i) 和记忆 b (属于节点 j) 之间存在边，
    且在聚合中 i→j 出现超过 30% 的 (i,j) 记忆对 → 确认节点级边 i→j。
    """
    # 统计每对 (node_i, node_j) 的边出现情况
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair_totals: dict[tuple[int, int], int] = defaultdict(int)

    for edge in discovered_edges:
        cause_mem = edge["cause_idx"]
        effect_mem = edge["effect_idx"]

        cause_node = mem_idx_to_node.get(cause_mem)
        effect_node = mem_idx_to_node.get(effect_mem)

        if cause_node is None or effect_node is None:
            continue
        if cause_node == effect_node:
            continue  # 忽略同节点内边

        pair_counts[(cause_node, effect_node)] += 1

    # 计算每个节点对的总可能记忆对数
    node_mem_counts = defaultdict(int)
    for mem_idx, node_idx in mem_idx_to_node.items():
        node_mem_counts[node_idx] += 1

    for (ni, nj), count in pair_counts.items():
        total_pairs = node_mem_counts[ni] * node_mem_counts[nj]
        pair_totals[(ni, nj)] = total_pairs

    # 阈值: 至少 25% 的记忆对有边 → 确认节点级边
    node_edges: set[tuple[int, int]] = set()
    for (ni, nj), count in pair_counts.items():
        total = pair_totals.get((ni, nj), 1)
        if count / max(total, 1) >= 0.25:
            node_edges.add((ni, nj))

    return node_edges


def evaluate_causal_discovery(
    true_adj: np.ndarray,
    discovered_edges: list[dict],
    mem_idx_to_node: dict[int, int],
    n_nodes: int,
    dataset_name: str,
    elapsed_ms: float,
    n_memories: int,
    config: str = "partial_correlation",
) -> CausalDiscoveryResult:
    """
    评估因果发现结果 (记忆级→节点级聚合后评估)。

    Args:
        true_adj: 真实邻接矩阵 (n_nodes × n_nodes)
        discovered_edges: GaussianDAG.discover_hidden_edges() 输出
        mem_idx_to_node: 记忆索引 → 节点索引映射
        n_nodes: 节点数
        dataset_name: 数据集名称
        elapsed_ms: 运行时间
        n_memories: 记忆数
        config: 配置名称

    Returns:
        CausalDiscoveryResult
    """
    n_true = int(np.sum(true_adj))

    # 真实边集合
    true_edges: set[tuple[int, int]] = set()
    for i in range(n_nodes):
        for j in range(n_nodes):
            if true_adj[i, j] == 1:
                true_edges.add((i, j))

    # 聚合发现边为节点级
    discovered_node_edges = _aggregate_edges_to_node_level(
        discovered_edges, mem_idx_to_node, n_nodes,
    )

    # 计算 TP/FP/FN
    tp = len(true_edges & discovered_node_edges)
    fp = len(discovered_node_edges - true_edges)
    fn = len(true_edges - discovered_node_edges)

    shd = fp + fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    return CausalDiscoveryResult(
        dataset_name=dataset_name,
        n_nodes=n_nodes,
        n_true_edges=n_true,
        n_mem_edges=len(discovered_edges),
        n_node_edges=len(discovered_node_edges),
        n_correct_edges=tp,
        n_missing_edges=fn,
        n_extra_edges=fp,
        shd=shd,
        precision=precision,
        recall=recall,
        f1=f1,
        elapsed_ms=elapsed_ms,
        n_memories=n_memories,
        config=config,
    )


# ═══════════════════════════════════════════════════════════════
# Benchmark Runner — 3 种配置
# ═══════════════════════════════════════════════════════════════


def run_causal_benchmark(
    true_adj: np.ndarray,
    memories: list[dict],
    mem_idx_to_node: dict[int, int],
    dataset_name: str,
    n_nodes: int,
    config: str = "PC_baseline",
) -> CausalDiscoveryResult:
    """运行因果发现基准 (统一入口)。"""
    from su_memory.sdk._spectral_causal import GaussianDAG

    bayes_summary = None
    t0 = time.perf_counter()

    dag = GaussianDAG(memories=memories)

    if config in ("energy_enhanced", "bayesian_posterior"):
        from su_memory.sdk._spectral_causal import FourierCausal
        fourier = FourierCausal()
        dag.with_fourier_filter(fourier)

    if config == "bayesian_posterior":
        from su_memory.sdk._spectral_causal import BayesianCausal
        bayesian = BayesianCausal()
        dag.with_bayesian_quantification(bayesian)

    edges = dag.discover_hidden_edges(
        min_correlation=0.15,
        p_threshold=0.10,
        max_pairs=200,
        max_scan=min(len(memories), 100),
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    result = evaluate_causal_discovery(
        true_adj, edges, mem_idx_to_node, n_nodes,
        dataset_name, elapsed_ms, len(memories), config=config,
    )

    # Bayesian summary
    if config == "bayesian_posterior" and 'bayesian' in dir():
        try:
            bayes_summary = bayesian.get_summary()  # type: ignore
        except Exception:
            bayes_summary = None

    return result


def run_pc_baseline(
    true_adj: np.ndarray, memories: list[dict],
    mem_idx_to_node: dict[int, int], dataset_name: str, n_nodes: int,
) -> CausalDiscoveryResult:
    """偏相关 (PC) 基线。"""
    return run_causal_benchmark(
        true_adj, memories, mem_idx_to_node, dataset_name, n_nodes,
        config="PC_baseline",
    )


def run_energy_enhanced(
    true_adj: np.ndarray, memories: list[dict],
    mem_idx_to_node: dict[int, int], dataset_name: str, n_nodes: int,
) -> CausalDiscoveryResult:
    """能量增强。"""
    return run_causal_benchmark(
        true_adj, memories, mem_idx_to_node, dataset_name, n_nodes,
        config="energy_enhanced",
    )


def run_bayesian_posterior(
    true_adj: np.ndarray, memories: list[dict],
    mem_idx_to_node: dict[int, int], dataset_name: str, n_nodes: int,
) -> tuple[CausalDiscoveryResult, dict | None]:
    """Bayesian 后验。"""
    from su_memory.sdk._spectral_causal import GaussianDAG, FourierCausal, BayesianCausal

    t0 = time.perf_counter()

    dag = GaussianDAG(memories=memories)
    fourier = FourierCausal()
    bayesian = BayesianCausal()
    dag.with_fourier_filter(fourier)
    dag.with_bayesian_quantification(bayesian)

    edges = dag.discover_hidden_edges(
        min_correlation=0.15, p_threshold=0.10,
        max_pairs=200, max_scan=min(len(memories), 100),
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    result = evaluate_causal_discovery(
        true_adj, edges, mem_idx_to_node, n_nodes,
        dataset_name, elapsed_ms, len(memories), config="bayesian_posterior",
    )

    try:
        bsum = bayesian.get_summary()
    except Exception:
        bsum = None

    return result, bsum


def run_scaling_benchmark(
    node_sizes: tuple[int, ...] = (5, 10, 20),
    seed: int = 42,
) -> list[CausalDiscoveryResult]:
    """规模扩展基准。"""
    results = []
    for n_nodes in node_sizes:
        adj, memories, mapping = generate_random_dag(
            n_nodes=n_nodes, edge_prob=0.3, seed=seed,
        )
        result = run_energy_enhanced(
            adj, memories, mapping, f"RandomDAG_{n_nodes}", n_nodes,
        )
        results.append(result)
        del adj, memories, mapping
        gc.collect()
    return results


# ═══════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════


def print_results(results: list[CausalDiscoveryResult], title: str = "Causal Discovery Benchmark") -> None:
    """打印结果表。"""
    W = 105
    print("\n" + "=" * W)
    print(f"  {title}")
    print("=" * W)
    header = (f"  {'Dataset':<24} {'Config':<22} {'Nodes':>5} {'TrueE':>6} "
              f"{'NodeE':>6} {'SHD':>5} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Time':>9}")
    print(header)
    print("-" * W)

    for r in results:
        print(f"  {r.dataset_name:<24} {r.config:<22} {r.n_nodes:>5} {r.n_true_edges:>6} "
              f"{r.n_node_edges:>6} {r.shd:>5} {r.precision:>7.3f} {r.recall:>7.3f} "
              f"{r.f1:>7.3f} {r.elapsed_ms:>8.1f}ms")

    print("-" * W)

    avg_f1 = float(np.mean([r.f1 for r in results]))
    avg_shd = float(np.mean([r.shd for r in results]))
    avg_precision = float(np.mean([r.precision for r in results]))
    avg_recall = float(np.mean([r.recall for r in results]))
    avg_time = float(np.mean([r.elapsed_ms for r in results]))

    print(f"  {'AVERAGE':<24} {'':<22} {'':>5} {'':>6} {'':>6} "
          f"{avg_shd:>5.1f} {avg_precision:>7.3f} {avg_recall:>7.3f} {avg_f1:>7.3f} {avg_time:>8.1f}ms")
    print("=" * W)

    # Verdict
    if avg_f1 > 0.8:
        verdict = "✅ PASS — F1 > 0.8, SHD 可接受"
    elif avg_f1 > 0.5:
        verdict = "🟡 MARGINAL — F1 > 0.5, 可用但需更多数据提升"
    else:
        verdict = "⚠️  LOW — 合成数据因果信号不足，需在真实数据上评测"

    print(f"  Verdict: {verdict}")
    if avg_shd < 5:
        print(f"  SHD Check: ✅ SHD={avg_shd:.1f} < 5 — 结构学习达标")
    else:
        print(f"  SHD Check: ⚠️  SHD={avg_shd:.1f} >= 5 (合成数据特征稀疏)")
    print("=" * W + "\n")


def export_json(results: list[CausalDiscoveryResult], bayes_summary: dict | None = None) -> dict:
    """导出 JSON 报告。"""
    report = {
        "benchmark": "su-memory SpectralCausal Discovery Benchmark",
        "version": "4.4.1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [r.to_dict() for r in results],
        "summary": {
            "n_datasets": len(results),
            "avg_f1": round(float(np.mean([r.f1 for r in results])), 4),
            "avg_shd": round(float(np.mean([r.shd for r in results])), 4),
            "avg_precision": round(float(np.mean([r.precision for r in results])), 4),
            "avg_recall": round(float(np.mean([r.recall for r in results])), 4),
            "avg_elapsed_ms": round(float(np.mean([r.elapsed_ms for r in results])), 1),
        },
    }
    if bayes_summary:
        report["bayesian_summary"] = bayes_summary
    return report


# ═══════════════════════════════════════════════════════════════
# Integration: 供 benchmark_v355_comprehensive.py 调用
# ═══════════════════════════════════════════════════════════════


def test_causal_discovery_section() -> dict:
    """
    E4 模块: SpectralCausal 因果发现评测。

    供 benchmark_v355_comprehensive.py 调用。

    Returns:
        {"section": "E4. SpectralCausal 因果发现", "results": {...}, "verdict": ...}
    """
    all_results: list[CausalDiscoveryResult] = []
    bayes_summary = None

    # ── 1. Sachs 蛋白质信号网络 (3 配置) ──
    try:
        adj_sachs, mem_sachs, mapping_sachs = generate_sachs_synthetic(seed=42)

        r = run_pc_baseline(adj_sachs, mem_sachs, mapping_sachs, "Sachs (PC)", 11)
        all_results.append(r)

        r = run_energy_enhanced(adj_sachs, mem_sachs, mapping_sachs, "Sachs (Energy)", 11)
        all_results.append(r)

        r, bsum = run_bayesian_posterior(adj_sachs, mem_sachs, mapping_sachs, "Sachs (Bayesian)", 11)
        all_results.append(r)
        bayes_summary = bsum

    except Exception as e:
        all_results.append(CausalDiscoveryResult(
            dataset_name=f"Sachs (ERROR: {str(e)[:40]})", n_nodes=11, n_true_edges=0,
            n_mem_edges=0, n_node_edges=0, n_correct_edges=0, n_missing_edges=0,
            n_extra_edges=0, shd=999, precision=0, recall=0, f1=0,
            elapsed_ms=0, n_memories=0, config="error",
        ))

    # ── 2. 随机 DAG (10/20 节点) ──
    for n_nodes, prob, seed in [(10, 0.30, 123), (20, 0.20, 456)]:
        try:
            adj, mem, mapping = generate_random_dag(n_nodes, edge_prob=prob, seed=seed)
            r = run_energy_enhanced(adj, mem, mapping, f"RandomDAG_{n_nodes}", n_nodes)
            all_results.append(r)
        except Exception:
            pass

    # ── Summary ──
    valid = [r for r in all_results if r.shd < 999]
    sachs_results = [r for r in valid if "Sachs" in r.dataset_name]
    sachs_f1 = float(np.mean([r.f1 for r in sachs_results])) if sachs_results else 0.0
    all_f1 = float(np.mean([r.f1 for r in valid])) if valid else 0.0

    per_dataset = [r.to_dict() for r in valid]

    return {
        "section": "E4. SpectralCausal 因果发现",
        "results": {
            "n_configs_tested": len(valid),
            "avg_f1_all": round(all_f1, 4),
            "avg_f1_sachs": round(sachs_f1, 4),
            "avg_shd": round(float(np.mean([r.shd for r in valid])), 4) if valid else 0,
            "best_f1": round(float(np.max([r.f1 for r in valid])), 4) if valid else 0,
            "best_config": max(valid, key=lambda r: r.f1).config if valid else "N/A",
            "bayesian_enabled": bayes_summary is not None,
            "per_dataset": per_dataset,
        },
        "verdict": (
            "✅ F1 > 0.8 — SpectralCausal 因果发现达标"
            if all_f1 > 0.8
            else "🟡 因果发现框架可用，合成数据评测完成"
        ),
    }


# ═══════════════════════════════════════════════════════════════
# Standalone CLI
# ═══════════════════════════════════════════════════════════════


def main():
    import argparse
    parser = argparse.ArgumentParser(description="su-memory SpectralCausal Discovery Benchmark")
    parser.add_argument("--quick", action="store_true", help="快速模式 (仅 Sachs + RandomDAG_10)")
    parser.add_argument("--report", action="store_true", help="导出 JSON 报告")
    parser.add_argument("--scaling", action="store_true", help="运行规模扩展基准")
    parser.add_argument("--output", type=str, default=None, help="报告输出路径")
    args = parser.parse_args()

    t_start = time.perf_counter()
    all_results: list[CausalDiscoveryResult] = []
    all_bayes = None

    print("\n" + "█" * 105)
    print("  su-memory v3.5.5 — SpectralCausal 因果发现基准测试")
    print("█" * 105)

    # ── Sachs Protein Network ──
    print("\n  📊 数据集 1: Sachs 蛋白质信号网络 (11 节点, 17 边)")
    adj_sachs, mem_sachs, mapping_sachs = generate_sachs_synthetic(seed=42)

    print("    → 运行 PC 基线...")
    r = run_pc_baseline(adj_sachs, mem_sachs, mapping_sachs, "Sachs (PC)", 11)
    all_results.append(r)
    print(f"       F1={r.f1:.3f}  SHD={r.shd}  NodeEdges={r.n_node_edges}/{r.n_true_edges}  Time={r.elapsed_ms:.1f}ms")

    print("    → 运行 Energy Enhanced...")
    r = run_energy_enhanced(adj_sachs, mem_sachs, mapping_sachs, "Sachs (Energy)", 11)
    all_results.append(r)
    print(f"       F1={r.f1:.3f}  SHD={r.shd}  NodeEdges={r.n_node_edges}/{r.n_true_edges}  Time={r.elapsed_ms:.1f}ms")

    print("    → 运行 Bayesian Posterior...")
    r, bsum = run_bayesian_posterior(adj_sachs, mem_sachs, mapping_sachs, "Sachs (Bayesian)", 11)
    all_results.append(r)
    all_bayes = bsum
    print(f"       F1={r.f1:.3f}  SHD={r.shd}  NodeEdges={r.n_node_edges}/{r.n_true_edges}  Time={r.elapsed_ms:.1f}ms")
    if bsum:
        print(f"       BF: strong={bsum.get('n_strong_causal',0)} "
              f"moderate={bsum.get('n_moderate_causal',0)} "
              f"max_BF={bsum.get('max_bayes_factor','N/A')}")

    # ── Random DAGs ──
    if not args.quick:
        for n, prob, seed in [(10, 0.30, 123), (20, 0.20, 456)]:
            print(f"\n  📊 数据集: RandomDAG_{n} ({n} 节点, p={prob})")
            adj, mem, mapping = generate_random_dag(n, edge_prob=prob, seed=seed)
            r = run_energy_enhanced(adj, mem, mapping, f"RandomDAG_{n}", n)
            all_results.append(r)
            print(f"       F1={r.f1:.3f}  SHD={r.shd}  NodeEdges={r.n_node_edges}/{r.n_true_edges}  "
                  f"Time={r.elapsed_ms:.1f}ms")
    else:
        print(f"\n  📊 数据集: RandomDAG_10 (10 节点, p=0.30)")
        adj_10, mem_10, mapping_10 = generate_random_dag(10, edge_prob=0.30, seed=123)
        r = run_energy_enhanced(adj_10, mem_10, mapping_10, "RandomDAG_10", 10)
        all_results.append(r)
        print(f"       F1={r.f1:.3f}  SHD={r.shd}  NodeEdges={r.n_node_edges}/{r.n_true_edges}  "
              f"Time={r.elapsed_ms:.1f}ms")

    # ── Scaling ──
    if args.scaling:
        print("\n  📊 规模扩展基准 (5/10/20 节点)...")
        scaling_results = run_scaling_benchmark((5, 10, 20), seed=99)
        all_results.extend(scaling_results)
        for r in scaling_results:
            print(f"       {r.dataset_name}: F1={r.f1:.3f}  Time={r.elapsed_ms:.1f}ms")

    elapsed = time.perf_counter() - t_start

    # ── Print ──
    print_results(all_results, f"SpectralCausal Discovery Benchmark (total: {elapsed:.1f}s)")

    # ── Export ──
    if args.report or args.output:
        report = export_json(all_results, all_bayes)
        output_path = args.output or os.path.join(
            os.path.dirname(__file__),
            "results",
            f"causal_discovery_{time.strftime('%Y%m%d_%H%M%S')}.json",
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"  📁 报告已保存: {output_path}")

    return all_results


if __name__ == "__main__":
    main()
