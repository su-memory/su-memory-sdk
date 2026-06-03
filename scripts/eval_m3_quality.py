#!/usr/bin/env python3
"""
M3 端到端训练质量评估脚本
==========================

对比 untrained vs trained 的 GNN 预测质量，验证训练是否真正
提升了邻接矩阵预测能力。

评估指标:
- MAE (Mean Absolute Error): mean(|A_pred - A_target|)
- RMSE (Root Mean Squared Error): sqrt(mean((A_pred - A_target)^2))
- Edge F1: 阈值化边的 precision/recall/F1
- Jaccard (IoU): |edges_pred ∩ edges_target| / |edges_pred ∪ edges_target|
- Spearman ρ: A_pred 与 A_target 值的秩相关系数
- Normalized Frobenius: ||A_pred - A_target||_F / ||A_target||_F
- Edge Density: 预测与目标图的边密度对比

用法:
    python scripts/eval_m3_quality.py
"""

import sys
import os
import time
import numpy as np

# 确保项目根目录及 src/ 在 sys.path 中，支持包导入
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from dataclasses import dataclass, field


# =============================================================================
# 指标计算
# =============================================================================


@dataclass
class QualityMetrics:
    """单对预测的质量指标。"""
    mae: float = 0.0
    rmse: float = 0.0
    edge_precision: float = 0.0
    edge_recall: float = 0.0
    edge_f1: float = 0.0
    jaccard: float = 0.0
    spearman_rho: float = 0.0
    frobenius_norm: float = 0.0
    pred_density: float = 0.0
    target_density: float = 0.0
    n_nodes: int = 0

    @classmethod
    def compute(
        cls,
        A_pred: np.ndarray,
        A_target: np.ndarray,
        edge_threshold: float = 0.1,
    ) -> "QualityMetrics":
        """计算单对预测的所有指标。"""
        A_pred = np.asarray(A_pred, dtype=np.float64)
        A_target = np.asarray(A_target, dtype=np.float64)
        n = A_pred.shape[0]

        if n == 0:
            return cls()

        # ── MAE / RMSE ──
        diff = A_pred - A_target
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))

        # ── 阈值化二值边（使用绝对值，因为 ρ 可为负）──
        pred_binary = (np.abs(A_pred) >= edge_threshold).astype(np.int32)
        target_binary = (np.abs(A_target) >= edge_threshold).astype(np.int32)

        # 排除对角线
        np.fill_diagonal(pred_binary, 0)
        np.fill_diagonal(target_binary, 0)

        tp = int(np.sum(pred_binary * target_binary))
        fp = int(np.sum(pred_binary * (1 - target_binary)))
        fn = int(np.sum((1 - pred_binary) * target_binary))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        jaccard = tp / max(tp + fp + fn, 1)

        # ── Spearman ρ ──
        pred_flat = A_pred.flatten()
        target_flat = A_target.flatten()
        if np.std(pred_flat) > 1e-10 and np.std(target_flat) > 1e-10:
            from scipy.stats import spearmanr
            try:
                rho, _ = spearmanr(pred_flat, target_flat)
                rho = float(rho) if not np.isnan(rho) else 0.0
            except Exception:
                rho = 0.0
        else:
            rho = 0.0

        # ── Normalized Frobenius ──
        frob = float(np.linalg.norm(diff, "fro"))
        target_frob = float(np.linalg.norm(A_target, "fro"))
        frob_norm = frob / max(target_frob, 1e-10)

        # ── 边密度 ──
        pred_density = float(np.sum(pred_binary)) / max(n * (n - 1), 1)
        target_density = float(np.sum(target_binary)) / max(n * (n - 1), 1)

        return cls(
            mae=round(mae, 6),
            rmse=round(rmse, 6),
            edge_precision=round(precision, 4),
            edge_recall=round(recall, 4),
            edge_f1=round(f1, 4),
            jaccard=round(jaccard, 4),
            spearman_rho=round(rho, 4),
            frobenius_norm=round(frob_norm, 4),
            pred_density=round(pred_density, 4),
            target_density=round(target_density, 4),
            n_nodes=n,
        )


@dataclass
class AggregateMetrics:
    """聚合指标（mean ± std over all pairs）。"""
    n_pairs: int = 0

    mae: tuple[float, float] = (0.0, 0.0)
    rmse: tuple[float, float] = (0.0, 0.0)
    edge_f1: tuple[float, float] = (0.0, 0.0)
    edge_precision: tuple[float, float] = (0.0, 0.0)
    edge_recall: tuple[float, float] = (0.0, 0.0)
    jaccard: tuple[float, float] = (0.0, 0.0)
    spearman_rho: tuple[float, float] = (0.0, 0.0)
    frobenius_norm: tuple[float, float] = (0.0, 0.0)
    pred_density: tuple[float, float] = (0.0, 0.0)
    target_density: tuple[float, float] = (0.0, 0.0)
    valid_ratio: float = 0.0  # fraction of pairs with n_nodes > 0

    @classmethod
    def from_list(cls, metrics_list: list[QualityMetrics]) -> "AggregateMetrics":
        """从指标列表计算聚合统计。"""
        valid = [m for m in metrics_list if m.n_nodes > 0]

        def _agg(attr: str) -> tuple[float, float]:
            vals = [getattr(m, attr) for m in valid]
            if not vals:
                return (0.0, 0.0)
            return (round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4))

        return cls(
            n_pairs=len(metrics_list),
            mae=_agg("mae"),
            rmse=_agg("rmse"),
            edge_f1=_agg("edge_f1"),
            edge_precision=_agg("edge_precision"),
            edge_recall=_agg("edge_recall"),
            jaccard=_agg("jaccard"),
            spearman_rho=_agg("spearman_rho"),
            frobenius_norm=_agg("frobenius_norm"),
            pred_density=_agg("pred_density"),
            target_density=_agg("target_density"),
            valid_ratio=round(len(valid) / max(len(metrics_list), 1), 4),
        )


# =============================================================================
# 预测收集
# =============================================================================


def collect_predictions(wm, dataset, encoder, predictor) -> list[QualityMetrics]:
    """
    对所有训练对运行 GAT→GNN 推理，收集 A_pred vs A_target 指标。

    使用 training_predict() 缓存 A_pred（不修改权重，仅做前向传播），
    直接获取完整邻接矩阵而非通过 _build_state 的阈值过滤版本。
    """
    from su_memory.sdk._jepa_gat_encoder import preprocess_memories_to_features, features_to_state
    from su_memory.sdk._jepa_gnn import align_adjacency
    from su_memory.sdk._world_model import CausalWorldModelState

    metrics_list: list[QualityMetrics] = []

    has_states = hasattr(dataset, 'state_pairs') and dataset.state_pairs

    for idx, (mem_t, _mem_t1) in enumerate(dataset.memory_pairs):
        # ── 目标状态（来自 state_pairs 缓存）──
        s_t1 = dataset.state_pairs[idx][1] if has_states and idx < len(dataset.state_pairs) else None

        if s_t1 is None:
            metrics_list.append(QualityMetrics())
            continue

        # ── GAT 编码 (推理模式) ──
        s_t_cache = dataset.state_pairs[idx][0] if has_states and idx < len(dataset.state_pairs) else None
        X, node_index, _ = preprocess_memories_to_features(wm, mem_t, state=s_t_cache)
        if X.shape[0] == 0:
            metrics_list.append(QualityMetrics())
            continue

        A_enc = encoder.gat_encoder.forward(X)

        # ── 构建编码状态 ──
        s_enc = features_to_state(A_enc, node_index, CausalWorldModelState())

        # ── GNN 预测 (training_predict 缓存完整 A_pred) ──
        predictor.training_predict(s_enc)
        A_pred = predictor.get_predicted_adj()
        pred_node_index = predictor.get_node_index()

        if A_pred is None or pred_node_index is None:
            metrics_list.append(QualityMetrics())
            continue

        # ── 获取 A_target ──
        A_target = align_adjacency(s_t1, pred_node_index)

        # ── 对齐形状 ──
        n = A_pred.shape[0]
        if A_target.shape[0] != n:
            padded = np.zeros((n, n), dtype=A_target.dtype)
            min_dim = min(A_target.shape[0], n)
            padded[:min_dim, :min_dim] = A_target[:min_dim, :min_dim]
            A_target = padded

        metrics = QualityMetrics.compute(A_pred, A_target)
        metrics_list.append(metrics)

    return metrics_list


# =============================================================================
# 对比报告
# =============================================================================


def compare_metrics(
    untrained: AggregateMetrics,
    trained: AggregateMetrics,
) -> str:
    """生成 untrained vs trained 对比报告。"""
    lines = []
    lines.append("=" * 70)
    lines.append("M3 训练质量对比: UNTRAINED → TRAINED")
    lines.append("=" * 70)
    lines.append(f"{'Metric':<25} {'Untrained':>16} {'Trained':>16} {'Δ%':>8}")
    lines.append("-" * 70)

    comparisons = [
        ("MAE (↓ better)", "mae", False),
        ("RMSE (↓ better)", "rmse", False),
        ("Edge F1 (↑ better)", "edge_f1", True),
        ("Edge Precision", "edge_precision", True),
        ("Edge Recall", "edge_recall", True),
        ("Jaccard/IoU (↑ better)", "jaccard", True),
        ("Spearman ρ (↑ better)", "spearman_rho", True),
        ("Norm. Frobenius (↓ better)", "frobenius_norm", False),
        ("Pred Edge Density", "pred_density", True),
        ("Target Edge Density", "target_density", True),
    ]

    improvements = []

    for label, attr, higher_better in comparisons:
        u_mean, u_std = getattr(untrained, attr)
        t_mean, t_std = getattr(trained, attr)

        if abs(u_mean) < 1e-10 and abs(t_mean) < 1e-10:
            delta_pct = 0.0
        elif abs(u_mean) < 1e-10:
            delta_pct = float("inf") if higher_better else float("-inf")
        else:
            delta_pct = (t_mean - u_mean) / abs(u_mean) * 100

        if higher_better:
            improved = t_mean > u_mean
        else:
            improved = t_mean < u_mean

        icon = "✅" if improved else "⚠️"
        delta_str = f"{delta_pct:+.1f}%" if abs(delta_pct) < 1e6 else "N/A"
        lines.append(
            f"{label:<25} {u_mean:>8.4f}±{u_std:.4f}  {t_mean:>8.4f}±{t_std:.4f}  "
            f"{delta_str:>8} {icon}"
        )
        improvements.append((label, improved, delta_pct))

    lines.append("-" * 70)
    lines.append(f"Valid pairs: {untrained.n_pairs} (valid ratio: {untrained.valid_ratio:.1%})")

    n_improved = sum(1 for _, imp, _ in improvements if imp)
    lines.append(f"Metrics improved: {n_improved}/{len(improvements)}")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================


def main():
    from su_memory.sdk._world_model import MCIWorldModel
    from su_memory.sdk._jepa_dataset import JEPADataset
    from su_memory.sdk._jepa_trainer import JEPATrainer

    # Import generate_causal_timeline from sibling script
    from scripts.train_m3_e2e import generate_causal_timeline

    # ── 1. 数据准备 ──
    print("[1/5] 生成时序数据...")
    memories = generate_causal_timeline(n_timesteps=30, n_memories_per_step=10)

    wm = MCIWorldModel()
    wm.initialize()
    dataset = JEPADataset.from_memories(wm, memories, window_size=10)
    print(f"      {dataset.n_pairs} 训练对, {dataset.n_memory_pairs} 记忆对")

    # ── 2. Untrained 基准 ──
    print("[2/5] 收集 UNTRAINED 预测...")
    wm_untrained = MCIWorldModel()
    wm_untrained.initialize()
    wm_untrained.enable_m3(encoder_key_dim=16, predictor_hidden_dim=16)

    # Re-run discover() to create states with the untrained wm
    ds_untrained = JEPADataset.from_memories(wm_untrained, memories, window_size=10)

    untrained_metrics = collect_predictions(
        wm_untrained, ds_untrained,
        wm_untrained._jepa_encoder, wm_untrained._jepa_predictor,
    )
    agg_untrained = AggregateMetrics.from_list(untrained_metrics)

    # ── 3. 训练 ──
    print("[3/5] 训练模型...")
    wm.enable_m3(encoder_key_dim=16, predictor_hidden_dim=16)
    ds_trained = JEPADataset.from_memories(wm, memories, window_size=10)

    t0 = time.time()
    trainer = JEPATrainer(
        encoder=wm._jepa_encoder,
        predictor=wm._jepa_predictor,
        dataset=ds_trained,
        alpha_energy=0.1,
        beta_cons=0.05,
    )
    stats = trainer.train(n_epochs=15, learning_rate=0.02)
    elapsed = time.time() - t0
    print(f"      训练完成: {elapsed:.1f}s, "
          f"Loss: {stats.loss_history[0]:.4f} → {stats.loss_history[-1]:.4f}")

    # ── 4. Trained 预测 ──
    print("[4/5] 收集 TRAINED 预测...")
    trained_metrics = collect_predictions(
        wm, ds_trained,
        wm._jepa_encoder, wm._jepa_predictor,
    )
    agg_trained = AggregateMetrics.from_list(trained_metrics)

    # ── 5. 对比报告 ──
    print("\n")
    print(compare_metrics(agg_untrained, agg_trained))

    # ── 逐对细节（前 5 对）──
    print("\n[Detail] 前 5 对边 F1 变化:")
    for i in range(min(5, len(untrained_metrics))):
        u = untrained_metrics[i]
        t = trained_metrics[i]
        if u.n_nodes == 0:
            continue
        delta = t.edge_f1 - u.edge_f1
        icon = "✅" if delta > 0 else ("⚠️" if delta < 0 else "➡️")
        print(f"  Pair {i}: F1 {u.edge_f1:.4f} → {t.edge_f1:.4f} "
              f"({delta:+.4f}) {icon}  | nodes={u.n_nodes}")


if __name__ == "__main__":
    main()
