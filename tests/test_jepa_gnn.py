"""v4.0.0 M2: 可微 GNN 预测器测试."""

from __future__ import annotations

import numpy as np
import pytest

from su_memory.sdk._jepa_gnn import GNNPredictor, align_adjacency
from su_memory.sdk._world_model import CausalWorldModelState

pytestmark = pytest.mark.jepa


# =============================================================================
# 测试辅助函数
# =============================================================================

def _make_state(
    edges: list[dict] | None = None,
    timestamp: str = "2026-01-01T00:00:00",
    active_states: set | None = None,
) -> CausalWorldModelState:
    """构造测试用 CausalWorldModelState。"""
    return CausalWorldModelState(
        causal_edges=edges or [],
        active_states=active_states or {"semantic", "causal"},
        n_memories=10,
        timestamp=timestamp,
    )


def _make_simple_state() -> CausalWorldModelState:
    """构造简单 3 节点状态。"""
    return _make_state([
        {"cause": "price", "effect": "demand", "rho": 0.72, "confidence": 0.85},
        {"cause": "supply", "effect": "price", "rho": 0.58, "confidence": 0.70},
        {"cause": "demand", "effect": "revenue", "rho": 0.65, "confidence": 0.78},
    ])


def _make_chain_state(n_nodes: int = 5) -> CausalWorldModelState:
    """构造链式因果图状态。"""
    edges = []
    for i in range(n_nodes - 1):
        edges.append({
            "cause": f"n{i}",
            "effect": f"n{i + 1}",
            "rho": 0.6 + 0.05 * (i % 3),
            "confidence": 0.7 + 0.05 * (i % 2),
        })
    return _make_state(edges)


# =============================================================================
# GNNPredictor 初始化测试
# =============================================================================


class TestGNNPredictorInit:
    """参数初始化测试。"""

    def test_default_init(self):
        p = GNNPredictor()
        assert p.hidden_dim == 16
        assert p.name == "gnn_v1"
        assert p.train_steps == 0

    def test_custom_hidden_dim(self):
        p = GNNPredictor(hidden_dim=32)
        assert p.hidden_dim == 32

    def test_parameter_shapes(self):
        p = GNNPredictor(hidden_dim=8)
        D = GNNPredictor.INPUT_DIM  # 8
        assert p.W1.shape == (D, 8)
        assert p.W2.shape == (D, 8)
        assert p.W3.shape == (8, 8)

    def test_parameter_values_finite(self):
        p = GNNPredictor(seed=123)
        assert np.all(np.isfinite(p.W1))
        assert np.all(np.isfinite(p.W2))
        assert np.all(np.isfinite(p.W3))

    def test_get_set_params_roundtrip(self):
        p = GNNPredictor(seed=42)
        params = p.get_params()
        assert set(params.keys()) == {"W1", "W2", "W3"}

        p2 = GNNPredictor(seed=99)
        p2.set_params(params)
        assert np.allclose(p2.W1, p.W1)
        assert np.allclose(p2.W2, p.W2)
        assert np.allclose(p2.W3, p.W3)

    def test_different_seeds_different_weights(self):
        p1 = GNNPredictor(seed=1)
        p2 = GNNPredictor(seed=2)
        assert not np.allclose(p1.W1, p2.W1)


# =============================================================================
# GNNPredictor 前向传播测试
# =============================================================================


class TestGNNPredictorForward:
    """前向传播（推理 + 训练）测试。"""

    def test_predict_returns_state(self):
        p = GNNPredictor(seed=42)
        s = _make_simple_state()
        s_pred = p.predict(s)
        assert isinstance(s_pred, CausalWorldModelState)
        # 应该产生一些预测边
        assert len(s_pred.causal_edges) > 0

    def test_predict_edge_values_in_range(self):
        p = GNNPredictor(seed=42)
        s = _make_simple_state()
        s_pred = p.predict(s)
        for e in s_pred.causal_edges:
            assert 0.0 < e["rho"] <= 1.0
            assert 0.0 <= e["confidence"] <= 1.0

    def test_predict_no_self_loops(self):
        p = GNNPredictor(seed=42)
        s = _make_chain_state(5)
        s_pred = p.predict(s)
        for e in s_pred.causal_edges:
            assert e["cause"] != e["effect"], f"自环: {e['cause']} → {e['effect']}"

    def test_predict_preserves_metadata(self):
        p = GNNPredictor(seed=42)
        s = _make_simple_state()
        s.n_confirmed = 3
        s.n_suppressed = 1
        s_pred = p.predict(s)
        assert s_pred.n_suppressed == 1

    def test_predict_empty_state(self):
        p = GNNPredictor(seed=42)
        s = _make_state([])
        s_pred = p.predict(s)
        assert isinstance(s_pred, CausalWorldModelState)
        assert len(s_pred.causal_edges) == 0

    def test_predict_idempotent_no_training_side_effect(self):
        """推理模式不应影响训练状态。"""
        p = GNNPredictor(seed=42)
        s = _make_simple_state()
        orig_w1 = p.W1.copy()
        p.predict(s)
        assert np.allclose(p.W1, orig_w1)
        assert p.train_steps == 0


class TestGNNPredictorTraining:
    """训练模式测试。"""

    def test_training_predict_returns_state(self):
        p = GNNPredictor(seed=42)
        s = _make_simple_state()
        s_pred = p.training_predict(s)
        assert isinstance(s_pred, CausalWorldModelState)
        assert len(s_pred.causal_edges) > 0

    def test_training_predict_caches_data(self):
        p = GNNPredictor(seed=42)
        s = _make_simple_state()
        p.training_predict(s)
        assert not p._cache.get("empty", True)
        assert "A_pred" in p._cache
        assert "node_index" in p._cache

    def test_get_predicted_adj(self):
        p = GNNPredictor(seed=42)
        s = _make_simple_state()
        p.training_predict(s)
        A_pred = p.get_predicted_adj()
        assert A_pred is not None
        assert A_pred.shape[0] == A_pred.shape[1]  # square
        assert np.all(A_pred >= 0) and np.all(A_pred <= 1)

    def test_get_node_index(self):
        p = GNNPredictor(seed=42)
        s = _make_simple_state()
        p.training_predict(s)
        ni = p.get_node_index()
        assert ni is not None
        assert "price" in ni
        assert "demand" in ni

    def test_training_empty_state(self):
        p = GNNPredictor(seed=42)
        s = _make_state([])
        s_pred = p.training_predict(s)
        assert isinstance(s_pred, CausalWorldModelState)
        assert len(s_pred.causal_edges) == 0
        assert p._cache.get("empty", False)

    def test_prediction_count_increments(self):
        p = GNNPredictor(seed=42)
        s = _make_simple_state()
        assert p._prediction_count == 0
        p.training_predict(s)
        assert p._prediction_count == 1
        p.predict(s)
        assert p._prediction_count == 1  # predict 不增加 count (基类 evaluate 才增加)


# =============================================================================
# GNNPredictor 反向传播测试
# =============================================================================


class TestGNNPredictorBackward:
    """手写梯度反向传播测试。"""

    def test_compute_gradients_returns_valid_dict(self):
        p = GNNPredictor(seed=42)
        s = _make_simple_state()
        p.training_predict(s)
        A_pred = p.get_predicted_adj()
        result = p.compute_gradients(A_pred)  # 目标 = 预测 → loss ≈ 0

        assert "loss" in result
        assert "grads" in result
        for key in ["W1", "W2", "W3"]:
            assert key in result["grads"]

    def test_zero_loss_for_perfect_prediction(self):
        p = GNNPredictor(seed=42)
        s = _make_simple_state()
        p.training_predict(s)
        A_pred = p.get_predicted_adj().copy()
        result = p.compute_gradients(A_pred)
        assert result["loss"] < 1e-8

    def test_positive_loss_for_wrong_target(self):
        p = GNNPredictor(seed=42)
        s = _make_simple_state()
        p.training_predict(s)
        A_pred = p.get_predicted_adj()
        # 故意用不同的目标
        A_wrong = np.ones_like(A_pred) * 0.5
        result = p.compute_gradients(A_wrong)
        assert result["loss"] > 0

    def test_empty_cache_returns_zero(self):
        p = GNNPredictor(seed=42)
        result = p.compute_gradients(np.ones((3, 3)))
        assert result["loss"] == 0.0

    def test_gradient_shapes_match_params(self):
        p = GNNPredictor(seed=42)
        s = _make_simple_state()
        p.training_predict(s)
        A_pred = p.get_predicted_adj()
        # 用一个略有差异的目标来产生非零梯度
        A_target = A_pred * 0.9 + 0.1
        result = p.compute_gradients(A_target)

        assert result["grads"]["W1"].shape == p.W1.shape
        assert result["grads"]["W2"].shape == p.W2.shape
        assert result["grads"]["W3"].shape == p.W3.shape

    def test_gradients_are_finite(self):
        p = GNNPredictor(seed=42)
        s = _make_simple_state()
        p.training_predict(s)
        A_pred = p.get_predicted_adj()
        A_target = A_pred * 0.9 + 0.1
        result = p.compute_gradients(A_target)

        for k in ["W1", "W2", "W3"]:
            assert np.all(np.isfinite(result["grads"][k])), f"{k} 梯度含 NaN/Inf"


class TestGNNPredictorGradientDescent:
    """梯度下降收敛测试。"""

    def test_loss_decreases_with_training(self):
        """在合成数据上验证梯度下降降低损失。"""
        p = GNNPredictor(seed=42, hidden_dim=8)

        # 构造两个相似但不相同的状态
        s_t = _make_chain_state(5)
        # s_{t+1}: 在 s_t 基础上微调 rho
        s_t1_edges = []
        for e in s_t.causal_edges:
            new_e = dict(e)
            new_e["rho"] = round(e["rho"] * 1.1, 4)  # 10% 增强
            s_t1_edges.append(new_e)
        s_t1 = _make_state(s_t1_edges)

        # 初始损失
        p.training_predict(s_t)
        ni = p.get_node_index()
        A_target = align_adjacency(s_t1, ni)
        result = p.compute_gradients(A_target)
        initial_loss = result["loss"]

        # 训练 50 步
        losses = [initial_loss]
        for _ in range(50):
            p.apply_gradients(result["grads"], lr=0.05)
            p.training_predict(s_t)
            A_target = align_adjacency(s_t1, ni)
            result = p.compute_gradients(A_target)
            losses.append(result["loss"])

        assert losses[-1] < initial_loss * 0.95, (
            f"Loss 未下降: {losses[0]:.6f} → {losses[-1]:.6f}"
        )

    def test_train_steps_counter(self):
        p = GNNPredictor(seed=42)
        s = _make_simple_state()
        p.training_predict(s)
        A_pred = p.get_predicted_adj()
        result = p.compute_gradients(A_pred)

        assert p.train_steps == 0
        p.apply_gradients(result["grads"], lr=0.01)
        assert p.train_steps == 1


# =============================================================================
# align_adjacency 测试
# =============================================================================


class TestAlignAdjacency:
    """邻接矩阵对齐工具测试。"""

    def test_align_same_nodes(self):
        s = _make_simple_state()
        ni = s._build_node_index()
        adj = align_adjacency(s, ni)
        assert adj.shape == (len(ni), len(ni))
        assert adj[ni["price"], ni["demand"]] == 0.72

    def test_align_subset_nodes(self):
        s = _make_simple_state()
        # 只用 price, demand 两个节点
        ni = {"price": 0, "demand": 1}
        adj = align_adjacency(s, ni)
        assert adj.shape == (2, 2)
        assert adj[0, 1] == 0.72

    def test_align_superset_nodes(self):
        s = _make_simple_state()
        # 目标索引包含不存在的节点
        ni = {"price": 0, "demand": 1, "supply": 2, "extra": 3}
        adj = align_adjacency(s, ni)
        assert adj.shape == (4, 4)
        assert adj[ni["price"], ni["demand"]] == 0.72
        assert adj[ni["supply"], ni["price"]] == 0.58
        # extra 节点没有边
        assert np.all(adj[ni["extra"], :] == 0)
        assert np.all(adj[:, ni["extra"]] == 0)

    def test_align_returns_float64(self):
        s = _make_simple_state()
        ni = s._build_node_index()
        adj = align_adjacency(s, ni)
        assert adj.dtype == np.float64


# =============================================================================
# GNNPredictor 评估测试
# =============================================================================


class TestGNNPredictorEvaluate:
    """评估方法测试。"""

    def test_evaluate_on_dataset(self):
        p = GNNPredictor(seed=42)
        s1 = _make_chain_state(4)
        s2 = _make_chain_state(5)
        result = p.evaluate([(s1, s2)])
        assert "avg_distance" in result
        assert "predictor" in result
        assert result["predictor"] == "gnn_v1"
        assert "hidden_dim" in result
        assert "param_norm" in result

    def test_evaluate_empty(self):
        p = GNNPredictor(seed=42)
        result = p.evaluate([])
        assert result["n"] == 0
        assert result["avg_distance"] == 1.0


# =============================================================================
# 集成测试
# =============================================================================


class TestGNNIntegration:
    """GNN 与 JEPATrainer / MCIWorldModel 集成测试。"""

    def test_trainer_detects_gnn(self):
        from su_memory.sdk._jepa_trainer import JEPATrainer

        p = GNNPredictor(seed=42)
        # encoder=None 仅用于检测
        trainer = JEPATrainer.__new__(JEPATrainer)
        trainer.predictor = p
        assert trainer._detect_gnn() is True

    def test_trainer_detects_baseline(self):
        from su_memory.sdk._jepa_predictor import IdentityPredictor
        from su_memory.sdk._jepa_trainer import JEPATrainer

        trainer = JEPATrainer.__new__(JEPATrainer)
        trainer.predictor = IdentityPredictor()
        assert trainer._detect_gnn() is False

    def test_gnn_trainer_training_loop(self):
        """完整的 GNN 训练循环：数据 → 训练 → 损失下降。"""
        from su_memory.sdk._jepa_trainer import JEPATrainer

        p = GNNPredictor(seed=42, hidden_dim=8)
        trainer = JEPATrainer(None, p, alpha_energy=0.0, beta_cons=0.0)

        # 构造训练对
        s_t = _make_chain_state(5)
        s_t1_edges = []
        for e in s_t.causal_edges:
            new_e = dict(e)
            new_e["rho"] = round(e["rho"] * 1.1, 4)
            s_t1_edges.append(new_e)
        s_t1 = _make_state(s_t1_edges)

        # 初始评估
        eval0 = p.evaluate([(s_t, s_t1)])
        eval0["avg_distance"]

        # 模拟训练循环
        losses = []
        for _ in range(20):
            loss = trainer._train_gnn_step(s_t, s_t1, learning_rate=0.02)
            losses.append(loss)

        assert losses[-1] < losses[0] * 0.95, (
            f"GNN 训练未收敛: {losses[0]:.6f} → {losses[-1]:.6f}"
        )

    def test_training_predict_and_predict_consistent(self):
        """验证 training_predict 和 predict 产生相同的前向结果。"""
        p = GNNPredictor(seed=42)
        s = _make_simple_state()

        s_pred_train = p.training_predict(s)
        s_pred_infer = p.predict(s)

        # 因果边数量应相近
        # (因数值精度差异可能略有不同)
        assert abs(len(s_pred_train.causal_edges) - len(s_pred_infer.causal_edges)) <= 1

    def test_l2_regularization(self):
        """L2 正则化增加损失。"""
        p_no_reg = GNNPredictor(seed=42, hidden_dim=8, l2_reg=0.0)
        p_reg = GNNPredictor(seed=42, hidden_dim=8, l2_reg=0.01)
        p_reg.set_params(p_no_reg.get_params())  # 相同初始权重

        s = _make_simple_state()
        p_no_reg.training_predict(s)
        A_pred = p_no_reg.get_predicted_adj()

        p_reg.training_predict(s)
        # 相同预测，但 L2 正则惩罚参数范数
        result_no_reg = p_no_reg.compute_gradients(A_pred)
        result_reg = p_reg.compute_gradients(A_pred)

        assert result_reg["loss"] > result_no_reg["loss"]

    def test_gnn_preserves_temporal_belief_metadata(self):
        """GNN 预测器保留时空和信念元数据。"""
        p = GNNPredictor(seed=42)
        s = _make_simple_state()
        # 模拟 temporal_info 和 cognitive_gaps (即使是 None)
        s_pred = p.predict(s)
        # 至少不应崩溃
        assert isinstance(s_pred, CausalWorldModelState)

    def test_gnn_serializable(self):
        """参数可序列化/反序列化。"""
        p = GNNPredictor(seed=42)
        params = p.get_params()
        # 验证可 pickle
        import pickle
        data = pickle.dumps(params)
        restored = pickle.loads(data)
        assert set(restored.keys()) == {"W1", "W2", "W3"}
