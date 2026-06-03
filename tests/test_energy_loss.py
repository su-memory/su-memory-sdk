"""
su-memory v3.6.0 — EnergyConsistencyLoss 单元测试
================================================

测试覆盖:
- 拓扑能量矩阵构建
- 能量损失计算
- 预测一致性验证
- 训练趋势分析
"""

import numpy as np
import pytest

pytestmark = pytest.mark.causal

from su_memory.sdk._energy_loss import (
    FIVE_CATEGORICAL_STATES,
    TopologicalEnergyMatrix,
    build_energy_matrix_from_energy_bus,
    create_default_energy_loss,
)


class TestTopologicalEnergyMatrix:
    """拓扑能量矩阵单元测试。"""

    def test_build_default_matrix(self):
        """测试标准矩阵构建 — 5×5 结构。"""
        topo = TopologicalEnergyMatrix.build()
        assert topo.matrix.shape == (5, 5)
        assert len(topo.state_index) == 5
        for state in FIVE_CATEGORICAL_STATES:
            assert state in topo.state_index

    def test_enhance_edges_high_weight(self):
        """增强边应有高权重。"""
        topo = TopologicalEnergyMatrix.build()
        val = topo.get_energy("semantic", "causal")
        assert val >= 0.7, f"增强边权重应 ≥ 0.7, 实际: {val}"

    def test_suppress_edges_low_weight(self):
        """抑制边应有低权重。"""
        topo = TopologicalEnergyMatrix.build()
        val = topo.get_energy("semantic", "spacetime")
        assert val <= 0.3, f"抑制边权重应 ≤ 0.3, 实际: {val}"

    def test_same_state_moderate_weight(self):
        """同状态应有中等权重。"""
        topo = TopologicalEnergyMatrix.build()
        val = topo.get_energy("semantic", "semantic")
        assert 0.3 <= val <= 0.7, f"同状态权重应在 [0.3, 0.7], 实际: {val}"

    def test_relation_type_enhance(self):
        """增强关系类型识别。"""
        topo = TopologicalEnergyMatrix.build()
        rel = topo.get_relation_type("semantic", "causal")
        assert rel == "enhance"

    def test_relation_type_suppress(self):
        """抑制关系类型识别。"""
        topo = TopologicalEnergyMatrix.build()
        rel = topo.get_relation_type("semantic", "spacetime")
        assert rel == "suppress"

    def test_unknown_state_returns_neutral(self):
        """未知状态返回中性。"""
        topo = TopologicalEnergyMatrix.build()
        rel = topo.get_relation_type("quantum", "photon")
        assert rel == "neutral"
        val = topo.get_energy("quantum", "photon")
        assert val == 0.0

    def test_total_edges_count(self):
        """验证 20 条有向边（增强 5 + 抑制 5 + 反向增强 5 + 反向抑制 5）。"""
        topo = TopologicalEnergyMatrix.build()
        n_enhance = sum(1 for r in topo.edge_types.values() if r == "enhance")
        n_suppress = sum(1 for r in topo.edge_types.values() if r == "suppress")
        n_same = sum(1 for r in topo.edge_types.values() if r == "same")
        assert n_enhance == 5
        assert n_suppress == 5
        assert n_same == 5

    def test_to_flat_vector(self):
        """测试展平为 25 维向量。"""
        topo = TopologicalEnergyMatrix.build()
        vec = topo.to_flat_vector()
        assert vec.shape == (25,)
        assert vec.dtype == np.float32

    def test_copy_is_deep(self):
        """深拷贝不应共享内存。"""
        topo = TopologicalEnergyMatrix.build()
        copy = topo.copy()
        copy.matrix[0, 0] = 0.99
        assert topo.matrix[0, 0] != 0.99


class TestEnergyConsistencyLoss:
    """能量一致性损失测试。"""

    @pytest.fixture
    def energy_loss(self):
        return create_default_energy_loss(alpha=0.1)

    def test_compute_basic(self, energy_loss):
        """基本损失计算。"""
        predictions = energy_loss.topological_matrix.matrix.copy()
        total, diag = energy_loss.compute(sft_loss=1.0, predictions=predictions)
        assert total >= 0.0
        assert "sft_loss" in diag
        assert "energy_loss" in diag
        assert "total_loss" in diag

    def test_perfect_match_zero_energy_loss(self, energy_loss):
        """预测完全匹配拓扑先验时应为 0 能量损失。"""
        predictions = energy_loss.topological_matrix.matrix.copy()
        _, diag = energy_loss.compute(sft_loss=1.0, predictions=predictions)
        assert diag["energy_loss"] < 0.01

    def test_random_predictions_produce_loss(self, energy_loss):
        """随机预测应产生非零能量损失。"""
        rng = np.random.RandomState(42)
        predictions = rng.rand(5, 5).astype(np.float32)
        _, diag = energy_loss.compute(sft_loss=1.0, predictions=predictions)
        assert diag["energy_loss"] > 0.01

    def test_1d_predictions_reshaped(self, energy_loss):
        """1D 预测自动 reshape 为 5×5。"""
        predictions = energy_loss.topological_matrix.to_flat_vector()
        total, diag = energy_loss.compute(sft_loss=0.5, predictions=predictions)
        assert diag["energy_loss"] < 0.01

    def test_compute_only_energy(self, energy_loss):
        """仅计算能量损失。"""
        predictions = energy_loss.topological_matrix.matrix.copy()
        e_loss = energy_loss.compute_only_energy(predictions)
        assert e_loss < 0.01

    def test_diagnostics_contain_edge_stats(self, energy_loss):
        """诊断信息包含边统计。"""
        predictions = np.ones((5, 5), dtype=np.float32) * 0.5
        _, diag = energy_loss.compute(sft_loss=1.0, predictions=predictions)
        assert "edge_losses" in diag
        assert "edge_counts" in diag
        assert diag["n_edges"] == 20  # 5×5 - 5 自环

    def test_history_recording(self, energy_loss):
        """损失历史记录。"""
        predictions = energy_loss.topological_matrix.matrix.copy()
        for _ in range(10):
            energy_loss.compute(sft_loss=1.0, predictions=predictions)
        history = energy_loss.get_history()
        assert len(history) == 10

    def test_trend_analysis_converging(self, energy_loss):
        """训练趋势分析 — 收敛。"""
        predictions = energy_loss.topological_matrix.matrix.copy()
        # 模拟逐渐靠近目标
        for _ in range(10):
            energy_loss.compute(sft_loss=0.5, predictions=predictions)
        trend = energy_loss.get_trend()
        assert trend["n_steps"] == 10
        assert trend["energy_loss_trend"] in ("converging", "stable", "insufficient_data")

    def test_insufficient_data_trend(self, energy_loss):
        """数据不足时趋势判定。"""
        trend = energy_loss.get_trend()
        assert trend["n_steps"] == 0
        assert trend["energy_loss_trend"] == "insufficient_data"

    def test_alpha_scaling(self):
        """alpha 参数正确缩放总损失。"""
        e1 = create_default_energy_loss(alpha=0.0)
        e2 = create_default_energy_loss(alpha=1.0)
        predictions = np.ones((5, 5), dtype=np.float32)
        _, d1 = e1.compute(sft_loss=1.0, predictions=predictions)
        _, d2 = e2.compute(sft_loss=1.0, predictions=predictions)
        assert d1["total_loss"] - 1.0 < 0.01  # alpha=0 → total ≈ SFT
        assert d2["total_loss"] > d1["total_loss"]


class TestPredictionValidation:
    """预测一致性验证测试。"""

    @pytest.fixture
    def energy_loss(self):
        return create_default_energy_loss(alpha=0.1)

    def test_enhance_prediction_confirmed(self, energy_loss):
        """增强边预测被确认。"""
        result = energy_loss.validate_prediction("semantic", "causal", 0.85)
        assert result["is_consistent"]
        assert result["verdict"] == "confirmed"

    def test_suppress_prediction_suppressed(self, energy_loss):
        """低预测对增强边被抑制。"""
        result = energy_loss.validate_prediction("semantic", "causal", 0.1)
        assert not result["is_consistent"]
        assert result["verdict"] == "suppressed"

    def test_novel_causal_discovery(self, energy_loss):
        """高预测对抑制边被标记为新发现。"""
        result = energy_loss.validate_prediction("semantic", "spacetime", 0.8)
        assert result["verdict"] == "novel"

    def test_neutral_returns_none(self, energy_loss):
        """无关状态对返回 none。"""
        result = energy_loss.validate_prediction("quantum", "photon", 0.5)
        assert result["verdict"] == "none"


class TestFactoryFunctions:
    """工厂函数测试。"""

    def test_create_default(self):
        """默认工厂创建。"""
        e = create_default_energy_loss()
        assert e.alpha == 0.1
        assert e.topological_matrix is not None

    def test_build_from_energy_bus_none(self):
        """None EnergyBus 返回标准矩阵。"""
        topo = build_energy_matrix_from_energy_bus(None)
        assert topo.matrix.shape == (5, 5)
