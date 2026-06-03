"""
su-memory v3.6.0 — Energy Consistency Loss (M7)
==================================================

拓扑先验能量一致性损失函数，用于 QLoRA 参数化训练时的结构约束。

论文公式 (v3.6.0):
    L_total = L_SFT + α · L_energy

其中 L_energy 惩罚违反已知增强/抑制模式（拓扑先验）的预测。

核心组件:
- TopologicalEnergyMatrix: 基于Energy Types增强/抑制关系的能量矩阵
- EnergyConsistencyLoss: 结合 SFT loss + 拓扑能量损失

用法:
    from su_memory.sdk._energy_loss import EnergyConsistencyLoss

    energy_loss = EnergyConsistencyLoss(alpha=0.1)
    total_loss = energy_loss.compute(sft_loss, predictions, topological_matrix)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# 拓扑能量常数
# =============================================================================

# Five-category state system (Energy Types → five states)
FIVE_CATEGORICAL_STATES = ["semantic", "causal", "spacetime", "generative", "trust"]

# 增强关系：c_i → c_{(i+1) mod 5}（Hamiltonian 电路）
ENHANCE_EDGES = [
    ("semantic", "causal"),
    ("causal", "spacetime"),
    ("spacetime", "generative"),
    ("generative", "trust"),
    ("trust", "semantic"),
]

# 抑制关系：c_i → c_{(i+2) mod 5}（stride-2 有向环）
SUPPRESS_EDGES = [
    ("semantic", "spacetime"),
    ("causal", "generative"),
    ("spacetime", "trust"),
    ("generative", "semantic"),
    ("trust", "causal"),
]

# 关系类型 → 乘法强度因子 φ(r)
PHI_FACTORS = {
    "enhance": 1.2,
    "suppress": 0.8,
    "overconstraint": 0.6,
    "reverse": 0.4,
    "same": 1.1,
    "neutral": 1.0,
}


# =============================================================================
# TopologicalEnergyMatrix
# =============================================================================


@dataclass
class TopologicalEnergyMatrix:
    """
    拓扑先验能量矩阵。

    基于五范畴状态系统的 20 条有向边构建，
    为每条边分配拓扑能量值 E_topo(e) = -log φ(r)。

    矩阵 M[i][j] ∈ [0, 1]:
        - 增强边：高能量倾向 (0.7-1.0)
        - 抑制边：低能量倾向 (0.1-0.3)
        - 无关边：0.0
    """

    matrix: np.ndarray  # shape=(5, 5), float32
    state_index: dict[str, int]  # state_name → matrix index
    edge_types: dict[tuple[int, int], str]  # (i, j) → relation_type

    @classmethod
    def build(cls) -> TopologicalEnergyMatrix:
        """
        构建标准五范畴状态能量矩阵。

        Returns:
            TopologicalEnergyMatrix 实例
        """
        n = len(FIVE_CATEGORICAL_STATES)
        matrix = np.zeros((n, n), dtype=np.float32)
        state_index = {s: i for i, s in enumerate(FIVE_CATEGORICAL_STATES)}
        edge_types: dict[tuple[int, int], str] = {}

        # 增强边
        for src, dst in ENHANCE_EDGES:
            i, j = state_index[src], state_index[dst]
            matrix[i, j] = 0.8  # 增强关系高权重
            edge_types[(i, j)] = "enhance"

        # 抑制边
        for src, dst in SUPPRESS_EDGES:
            i, j = state_index[src], state_index[dst]
            matrix[i, j] = 0.2  # 抑制关系低权重
            edge_types[(i, j)] = "suppress"

        # 同状态：温和权重
        for i in range(n):
            matrix[i, i] = 0.5
            edge_types[(i, i)] = "same"

        return cls(matrix=matrix, state_index=state_index, edge_types=edge_types)

    def get_energy(self, src_state: str, dst_state: str) -> float:
        """查询两状态间的拓扑能量值。"""
        i = self.state_index.get(src_state)
        j = self.state_index.get(dst_state)
        if i is None or j is None:
            return 0.0
        return float(self.matrix[i, j])

    def get_relation_type(self, src_state: str, dst_state: str) -> str:
        """查询两状态间的拓扑关系类型。"""
        i = self.state_index.get(src_state)
        j = self.state_index.get(dst_state)
        if i is None or j is None:
            return "neutral"
        return self.edge_types.get((i, j), "neutral")

    def to_flat_vector(self) -> np.ndarray:
        """将 5×5 矩阵展平为 25 维向量（用于 MLX/Torch 训练）。"""
        return self.matrix.flatten().copy()

    def copy(self) -> TopologicalEnergyMatrix:
        """深拷贝。"""
        return TopologicalEnergyMatrix(
            matrix=self.matrix.copy(),
            state_index=dict(self.state_index),
            edge_types=dict(self.edge_types),
        )


# =============================================================================
# EnergyConsistencyLoss
# =============================================================================


class EnergyConsistencyLoss:
    """
    拓扑先验能量一致性损失。

    计算公式:
        L_energy = (1 / N) Σ_i Σ_j |pred_{ij} - M_{ij}| · w_{ij}

    其中:
        - pred_{ij}: 模型预测的状态 i→j 因果强度
        - M_{ij}:   拓扑能量矩阵期望值
        - w_{ij}:   边权重 (增强/抑制边的惩罚更高)

    总损失:
        L_total = L_SFT + α · L_energy
    """

    def __init__(
        self,
        topological: TopologicalEnergyMatrix | None = None,
        alpha: float = 0.1,
        edge_penalty_multiplier: float = 2.0,
    ):
        """
        Args:
            topological: 拓扑能量矩阵（None 时自动构建标准矩阵）
            alpha: 能量损失权重系数（默认 0.1）
            edge_penalty_multiplier: 增强/抑制边惩罚倍数
        """
        self._topo = topological or TopologicalEnergyMatrix.build()
        self.alpha = alpha
        self.edge_penalty_multiplier = edge_penalty_multiplier
        self._history: list[dict] = []

    @property
    def topological_matrix(self) -> TopologicalEnergyMatrix:
        return self._topo

    # -----------------------------------------------------------------
    # 核心计算
    # -----------------------------------------------------------------

    def compute(
        self,
        sft_loss: float,
        predictions: np.ndarray,
        topological: TopologicalEnergyMatrix | None = None,
    ) -> tuple[float, dict]:
        """
        计算总损失 = L_SFT + α · L_energy。

        Args:
            sft_loss: 标准语言模型损失（交叉熵）
            predictions: 模型预测的因果强度矩阵，shape=(5, 5) 或展平 (25,)
            topological: 可选的自定义拓扑矩阵

        Returns:
            (total_loss, diagnostics_dict)
        """
        topo = topological or self._topo

        # 确保 predictions 是 5×5 矩阵
        pred = np.asarray(predictions, dtype=np.float32)
        if pred.ndim == 1:
            pred = pred.reshape(5, 5)

        # ── 计算 L_energy ──
        n = pred.shape[0]
        energy_loss = 0.0
        n_edges = 0
        edge_losses: dict[str, float] = {"enhance": 0.0, "suppress": 0.0, "other": 0.0}
        edge_counts: dict[str, int] = {"enhance": 0, "suppress": 0, "other": 0}

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue  # 跳过自环
                pred_val = pred[i, j]
                target_val = topo.matrix[i, j]

                # 确定边类型
                rel = topo.edge_types.get((i, j), "neutral")
                if rel in ("enhance", "suppress"):
                    edge_type = rel
                else:
                    edge_type = "other"

                # 边惩罚权重
                weight = self.edge_penalty_multiplier if edge_type in ("enhance", "suppress") else 1.0

                # L1 距离
                diff = abs(pred_val - target_val)
                weighted_diff = diff * weight

                energy_loss += weighted_diff
                edge_losses[edge_type] += weighted_diff
                edge_counts[edge_type] += 1
                n_edges += 1

        # 归一化
        if n_edges > 0:
            energy_loss /= n_edges
            for k in edge_losses:
                if edge_counts[k] > 0:
                    edge_losses[k] /= edge_counts[k]

        total_loss = sft_loss + self.alpha * energy_loss

        # ── 诊断信息 ──
        diag = {
            "sft_loss": round(float(sft_loss), 6),
            "energy_loss": round(float(energy_loss), 6),
            "total_loss": round(float(total_loss), 6),
            "alpha": self.alpha,
            "n_edges": n_edges,
            "edge_losses": {k: round(float(v), 6) for k, v in edge_losses.items()},
            "edge_counts": dict(edge_counts),
        }

        self._history.append(diag)
        return total_loss, diag

    def compute_only_energy(
        self,
        predictions: np.ndarray,
        topological: TopologicalEnergyMatrix | None = None,
    ) -> float:
        """
        仅计算 L_energy（不与 SFT loss 结合时使用）。

        Args:
            predictions: 模型预测矩阵
            topological: 可选拓扑矩阵

        Returns:
            L_energy 值
        """
        _, diag = self.compute(sft_loss=0.0, predictions=predictions, topological=topological)
        return diag["energy_loss"]

    # -----------------------------------------------------------------
    # 推理时验证
    # -----------------------------------------------------------------

    def validate_prediction(
        self,
        src_state: str,
        dst_state: str,
        predicted_strength: float,
    ) -> dict:
        """
        验证单个因果预测是否与拓扑先验一致。

        Args:
            src_state: 原因状态
            dst_state: 效应状态
            predicted_strength: 模型预测强度 [0, 1]

        Returns:
            {
                "is_consistent": bool,
                "expected_range": (float, float),
                "deviation": float,
                "relation_type": str,
                "verdict": "confirmed" | "novel" | "suppressed" | "none",
            }
        """
        rel_type = self._topo.get_relation_type(src_state, dst_state)
        expected = self._topo.get_energy(src_state, dst_state)

        # 期望范围（±0.3 容差）
        expected_range = (max(0.0, expected - 0.3), min(1.0, expected + 0.3))
        deviation = abs(predicted_strength - expected)
        is_consistent = expected_range[0] <= predicted_strength <= expected_range[1]

        # 三重判定（基于关系类型，而非静态阈值）
        if rel_type in ("enhance", "same"):
            if is_consistent and predicted_strength > 0.5:
                verdict = "confirmed"
            elif not is_consistent and predicted_strength < 0.3:
                verdict = "suppressed"  # 预测与增强/同状态先验矛盾
            else:
                verdict = "none"
        elif rel_type == "suppress":
            if predicted_strength > 0.5:
                verdict = "novel"  # 模型发现抑制边之上存在强因果
            else:
                verdict = "none"
        else:
            verdict = "none"

        return {
            "is_consistent": is_consistent,
            "expected_range": expected_range,
            "deviation": round(float(deviation), 4),
            "relation_type": rel_type,
            "verdict": verdict,
        }

    # -----------------------------------------------------------------
    # 训练反馈
    # -----------------------------------------------------------------

    def get_history(self) -> list[dict]:
        """获取损失历史记录。"""
        return self._history.copy()

    def reset_history(self):
        """清空历史记录。"""
        self._history.clear()

    def get_trend(self) -> dict:
        """
        分析损失趋势。

        Returns:
            {
                "n_steps": int,
                "energy_loss_trend": "converging" | "diverging" | "stable" | "insufficient_data",
                "final_energy_loss": float,
            }
        """
        if len(self._history) < 5:
            return {
                "n_steps": len(self._history),
                "energy_loss_trend": "insufficient_data",
                "final_energy_loss": 0.0,
            }

        recent = self._history[-5:]
        energy_vals = [h["energy_loss"] for h in recent]

        if energy_vals[-1] < energy_vals[0] * 0.9:
            trend = "converging"
        elif energy_vals[-1] > energy_vals[0] * 1.1:
            trend = "diverging"
        else:
            trend = "stable"

        return {
            "n_steps": len(self._history),
            "energy_loss_trend": trend,
            "final_energy_loss": round(float(energy_vals[-1]), 6),
        }


    # -----------------------------------------------------------------
    # v4.0.0 JEPA: N×N 图结构能量损失
    # -----------------------------------------------------------------

    def compute_graph_energy(
        self,
        pred_state,
        actual_state,
    ) -> tuple[float, dict]:
        """
        计算 JEPA 图结构能量损失。

        将 5×5 拓扑先验扩展到 N×N 因果图:
        - 对 pred_state 中的每条因果边，用 _sys/_energy_relations 判定拓扑违规
        - 增强边但权重过低 → 惩罚
        - 抑制边但权重过高 → 惩罚
        - 返回归一化违规分数

        Args:
            pred_state: 预测的 CausalWorldModelState
            actual_state: 实际的 CausalWorldModelState

        Returns:
            (energy_loss, diagnostics)
        """
        try:
            from su_memory._sys._energy_relations import is_enhancing, is_suppressing
        except ImportError:
            return 0.0, {"note": "energy_relations_unavailable"}

        if not pred_state.causal_edges:
            return 0.0, {"n_edges": 0}

        violations = 0.0
        n_enhance_violations = 0
        n_suppress_violations = 0
        n_edges = len(pred_state.causal_edges)

        for edge in pred_state.causal_edges:
            energy_rel = edge.get("energy_relation", "neutral")
            rho = edge.get("rho", 0.0)

            if energy_rel == "enhance" and rho < 0.3:
                violations += (0.3 - rho)
                n_enhance_violations += 1
            elif energy_rel == "suppress" and rho > 0.7:
                violations += (rho - 0.7)
                n_suppress_violations += 1

        energy_loss = violations / max(n_edges, 1)

        diagnostics = {
            "n_edges": n_edges,
            "violations": round(float(violations), 6),
            "energy_loss": round(float(energy_loss), 6),
            "n_enhance_violations": n_enhance_violations,
            "n_suppress_violations": n_suppress_violations,
        }

        return energy_loss, diagnostics


# =============================================================================
# v4.0.0 JEPA: 图结构能量损失工厂函数
# =============================================================================


def compute_jepa_graph_energy(
    pred_state,
    actual_state,
    alpha: float = 0.1,
) -> float:
    """
    便捷函数：计算 JEPA 图能量损失。

    Args:
        pred_state: 预测状态
        actual_state: 实际状态
        alpha: 能量损失权重

    Returns:
        加权能量损失
    """
    loss = EnergyConsistencyLoss(alpha=alpha)
    energy, _ = loss.compute_graph_energy(pred_state, actual_state)
    return alpha * energy


# =============================================================================
# 工厂函数
# =============================================================================


def create_default_energy_loss(alpha: float = 0.1) -> EnergyConsistencyLoss:
    """创建使用标准拓扑矩阵的 EnergyConsistencyLoss 实例。"""
    return EnergyConsistencyLoss(
        topological=TopologicalEnergyMatrix.build(),
        alpha=alpha,
    )


def build_energy_matrix_from_energy_bus(energy_bus) -> TopologicalEnergyMatrix:
    """
    从 EnergyBus 实例动态构建能量矩阵。

    读取当前 EnergyBus 中各状态节点的强度值，
    与标准拓扑矩阵融合生成运行时能量矩阵。

    Args:
        energy_bus: EnergyBus 实例

    Returns:
        运行时能量矩阵
    """
    matrix = TopologicalEnergyMatrix.build()
    if energy_bus is None:
        return matrix

    try:
        core = getattr(energy_bus, "_energy_core", None)
        if core is None:
            return matrix

        # 读取节点当前强度值
        intensities: dict[str, float] = {}
        for state in FIVE_CATEGORICAL_STATES:
            try:
                node = getattr(core, f"_node_{state}", None)
                if node is not None:
                    val = float(getattr(node, "intensity", 0.5))
                    intensities[state] = val
            except Exception:
                intensities[state] = 0.5

        # 与标准矩阵加权融合（70% 标准 + 30% 运行时）
        for src in FIVE_CATEGORICAL_STATES:
            for dst in FIVE_CATEGORICAL_STATES:
                i = matrix.state_index[src]
                j = matrix.state_index[dst]
                base = matrix.matrix[i, j]
                runtime_influence = (intensities.get(src, 0.5) * intensities.get(dst, 0.5))
                matrix.matrix[i, j] = float(base * 0.7 + runtime_influence * 0.3)
    except Exception as e:
        logger.warning("从 EnergyBus 构建能量矩阵失败: %s，使用标准矩阵", e)

    return matrix
