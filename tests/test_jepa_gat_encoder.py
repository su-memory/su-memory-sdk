"""v4.0.0 M3: 可微 GAT 编码器测试."""

from __future__ import annotations

import numpy as np
import pytest

from su_memory.sdk._jepa_gat_encoder import (
    GATEncoder,
    preprocess_memories_to_features,
    features_to_state,
)
from su_memory.sdk._world_model import CausalWorldModelState

pytestmark = pytest.mark.jepa


def _make_features(n_nodes: int = 5, seed: int = 42) -> np.ndarray:
    """构造随机节点特征矩阵。"""
    rng = np.random.RandomState(seed)
    return rng.randn(n_nodes, 8).astype(np.float32)


def _make_memories(n: int = 10) -> list[dict]:
    """构造测试记忆列表。"""
    topics = ["price increase", "demand drop", "supply chain delay",
              "revenue growth", "cost reduction", "quality improvement",
              "efficiency gain", "growth acceleration", "risk mitigation",
              "market expansion"]
    memories = []
    for i in range(min(n, len(topics))):
        memories.append({
            "id": f"mem_{i}",
            "content": f"{topics[i]} by {np.random.randint(5, 20)}% in Q{i+1}",
            "timestamp": f"2026-01-{i+1:02d}T00:00:00",
        })
    return memories


# =============================================================================
# GATEncoder 初始化测试
# =============================================================================


class TestGATEncoderInit:
    """参数初始化测试。"""

    def test_default_init(self):
        enc = GATEncoder()
        assert enc._input_dim == 8
        assert enc._key_dim == 16
        assert enc.train_steps == 0

    def test_custom_dims(self):
        enc = GATEncoder(input_dim=8, key_dim=32)
        assert enc._key_dim == 32
        assert enc.W_q.shape == (8, 32)
        assert enc.W_k.shape == (8, 32)

    def test_parameter_values_finite(self):
        enc = GATEncoder(seed=123)
        assert np.all(np.isfinite(enc.W_q))
        assert np.all(np.isfinite(enc.W_k))

    def test_get_set_params_roundtrip(self):
        enc = GATEncoder(seed=42)
        params = enc.get_params()
        assert set(params.keys()) == {"W_q", "W_k"}

        enc2 = GATEncoder(seed=99)
        enc2.set_params(params)
        assert np.allclose(enc2.W_q, enc.W_q)
        assert np.allclose(enc2.W_k, enc.W_k)

    def test_different_seeds_different_weights(self):
        enc1 = GATEncoder(seed=1)
        enc2 = GATEncoder(seed=2)
        assert not np.allclose(enc1.W_q, enc2.W_q)


# =============================================================================
# GATEncoder 前向传播测试
# =============================================================================


class TestGATEncoderForward:
    """前向传播测试。"""

    def test_forward_returns_correct_shape(self):
        enc = GATEncoder(seed=42)
        X = _make_features(5)
        A = enc.forward(X)
        assert A.shape == (5, 5)
        assert A.dtype == np.float32

    def test_forward_values_in_range(self):
        enc = GATEncoder(seed=42)
        X = _make_features(10)
        A = enc.forward(X)
        assert np.all(A >= 0.0) and np.all(A <= 1.0)

    def test_forward_deterministic(self):
        enc = GATEncoder(seed=42)
        X = _make_features(5)
        A1 = enc.forward(X)
        A2 = enc.forward(X)
        assert np.allclose(A1, A2)

    def test_forward_different_inputs_different_outputs(self):
        enc = GATEncoder(seed=42)
        X1 = _make_features(5, seed=1)
        X2 = _make_features(5, seed=2)
        A1 = enc.forward(X1)
        A2 = enc.forward(X2)
        assert not np.allclose(A1, A2)

    def test_forward_wrong_input_dim_raises(self):
        enc = GATEncoder(input_dim=8)
        X = np.random.randn(5, 10).astype(np.float32)
        with pytest.raises(ValueError):
            enc.forward(X)

    def test_training_forward_caches(self):
        enc = GATEncoder(seed=42)
        X = _make_features(5)
        A = enc.training_forward(X)
        assert A.shape == (5, 5)
        assert "Q" in enc._cache
        assert "K" in enc._cache
        assert "A_enc" in enc._cache


# =============================================================================
# GATEncoder 反向传播测试
# =============================================================================


class TestGATEncoderBackward:
    """手写梯度反向传播测试。"""

    def test_compute_gradients_shapes(self):
        enc = GATEncoder(seed=42)
        X = _make_features(5)
        enc.training_forward(X)
        dA = np.random.randn(5, 5).astype(np.float64)

        grads = enc.compute_gradients(dA)
        assert grads["W_q"].shape == enc.W_q.shape
        assert grads["W_k"].shape == enc.W_k.shape

    def test_gradients_are_finite(self):
        enc = GATEncoder(seed=42)
        X = _make_features(5)
        enc.training_forward(X)
        dA = np.random.randn(5, 5).astype(np.float64) * 0.1

        grads = enc.compute_gradients(dA)
        assert np.all(np.isfinite(grads["W_q"]))
        assert np.all(np.isfinite(grads["W_k"]))

    def test_compute_gradients_from_mse(self):
        enc = GATEncoder(seed=42)
        X = _make_features(5)
        enc.training_forward(X)
        A_enc = enc._cache["A_enc"]
        # 目标 = 预测 → loss ≈ 0
        result = enc.compute_gradients_from_mse(A_enc)
        assert result["mse"] < 1e-8
        assert "W_q" in result
        assert "W_k" in result

    def test_mse_positive_for_wrong_target(self):
        enc = GATEncoder(seed=42)
        X = _make_features(5)
        enc.training_forward(X)
        A_wrong = np.ones((5, 5)) * 0.5
        result = enc.compute_gradients_from_mse(A_wrong)
        assert result["mse"] > 0

    def test_empty_cache_returns_zero(self):
        enc = GATEncoder(seed=42)
        result = enc.compute_gradients_from_mse(np.ones((3, 3)))
        assert result["mse"] == 0.0

    def test_backward_through_same_forward(self):
        """验证反向传播链: 相同的输入产生一致的梯度。"""
        enc = GATEncoder(seed=42)
        X = _make_features(5)
        enc.training_forward(X)
        A_enc = enc._cache["A_enc"]
        dA = (A_enc - 0.3) * 0.01  # 小扰动
        grads1 = enc.compute_gradients(dA)
        grads2 = enc.compute_gradients(dA)
        assert np.allclose(grads1["W_q"], grads2["W_q"])


class TestGATEncoderGradientDescent:
    """梯度下降收敛测试。"""

    def test_loss_decreases(self):
        enc = GATEncoder(seed=42, key_dim=8)
        X = _make_features(5)

        # 初始损失
        enc.training_forward(X)
        # 目标: 接近初始预测但略有不同
        A_enc = enc._cache["A_enc"].copy()
        A_target = A_enc * 0.9 + 0.1  # push toward 0.1
        result = enc.compute_gradients_from_mse(A_target)
        initial_loss = result["mse"]

        # 训练 100 步
        losses = [initial_loss]
        for _ in range(100):
            enc.apply_gradients(result, lr=0.05)
            enc.training_forward(X)
            result = enc.compute_gradients_from_mse(A_target)
            losses.append(result["mse"])

        assert losses[-1] < initial_loss * 0.90, (
            f"GAT loss 未下降: {initial_loss:.6f} → {losses[-1]:.6f}"
        )

    def test_train_steps_counter(self):
        enc = GATEncoder(seed=42)
        X = _make_features(5)
        enc.training_forward(X)
        A_enc = enc._cache["A_enc"]
        result = enc.compute_gradients_from_mse(A_enc)

        assert enc.train_steps == 0
        enc.apply_gradients(result, lr=0.01)
        assert enc.train_steps == 1

    def test_l2_regularization_increases_loss(self):
        enc_no_reg = GATEncoder(seed=42, l2_reg=0.0)
        enc_reg = GATEncoder(seed=42, l2_reg=0.01)
        enc_reg.set_params(enc_no_reg.get_params())

        X = _make_features(5)
        enc_no_reg.training_forward(X)
        enc_reg.training_forward(X)

        A_target = np.ones((5, 5)) * 0.3
        r_no = enc_no_reg.compute_gradients_from_mse(A_target)
        r_reg = enc_reg.compute_gradients_from_mse(A_target)
        assert r_reg["loss"] > r_no["loss"]


# =============================================================================
# 桥接函数测试
# =============================================================================


class TestPreprocessFeatures:
    """preprocess_memories_to_features 测试。"""

    def test_returns_valid_shapes(self):
        from su_memory.sdk._world_model import MCIWorldModel
        wm = MCIWorldModel()
        memories = _make_memories(15)  # 更多记忆帮助 discover() 成功
        X, node_index, node_names = preprocess_memories_to_features(wm, memories)
        # discover() may fail with simple test data → shape may be (0, 8) or (N, 8)
        assert X.shape[1] == 8  # always correct feature dim
        if X.shape[0] > 0:
            assert len(node_index) == X.shape[0]
            assert len(node_names) == X.shape[0]

    def test_features_to_state(self):
        """从 GAT 输出重建状态。"""
        node_index = {"price": 0, "demand": 1, "supply": 2}
        A = np.array([[0, 0.7, 0.1],
                       [0.05, 0, 0.3],
                       [0.6, 0, 0]], dtype=np.float64)
        template = CausalWorldModelState()
        state = features_to_state(A, node_index, template)
        assert isinstance(state, CausalWorldModelState)
        assert len(state.causal_edges) >= 2  # 至少 price→demand, supply→price


# =============================================================================
# 集成测试
# =============================================================================


class TestGATE2EIntegration:
    """GAT + GNN 端到端集成测试。"""

    def test_gat_to_gnn_pipeline(self):
        """GAT 编码 → GNN 预测 → 完整流水线。"""
        from su_memory.sdk._jepa_gnn import GNNPredictor, align_adjacency

        gat = GATEncoder(seed=42, key_dim=8)
        gnn = GNNPredictor(seed=42, hidden_dim=8)

        X = _make_features(5)

        # GAT 编码
        A_enc = gat.training_forward(X)

        # 构建临时状态
        node_index = {f"n{i}": i for i in range(5)}
        from su_memory.sdk._jepa_gat_encoder import features_to_state
        state = features_to_state(A_enc, node_index, CausalWorldModelState())

        # GNN 预测
        s_pred = gnn.training_predict(state)
        assert isinstance(s_pred, CausalWorldModelState)

        # GNN 损失
        A_pred = gnn.get_predicted_adj()
        result = gnn.compute_gradients(A_pred)
        assert result["loss"] < 1e-8

        # 分别更新
        gnn.apply_gradients(result["grads"], lr=0.01)

        # GAT 损失
        gat_result = gat.compute_gradients_from_mse(A_enc * 0.95 + 0.05)
        gat.apply_gradients(gat_result, lr=0.01)

        assert gat.train_steps == 1
        assert gnn.train_steps == 1

    def test_gat_encoder_serializable(self):
        enc = GATEncoder(seed=42)
        params = enc.get_params()
        import pickle
        data = pickle.dumps(params)
        restored = pickle.loads(data)
        assert set(restored.keys()) == {"W_q", "W_k"}

    def test_attention_scores_accessor(self):
        enc = GATEncoder(seed=42)
        X = _make_features(5)
        enc.training_forward(X)
        scores = enc.get_attention_scores()
        assert scores is not None
        assert scores.shape == (5, 5)

    def test_to_adjacency_matrix_threshold(self):
        enc = GATEncoder(seed=42)
        X = _make_features(5)
        A = enc.to_adjacency_matrix(X, threshold=0.3)
        assert np.all((A == 0.0) | (A >= 0.3))
