"""
su-memory v4.0.0 M2 — GNN Predictor
====================================

可微 GNN 因果图预测器，用 numpy 手写梯度实现。

替代三个不可参基线（Identity/EnergyPropagation/BeliefPropagation），
提供端到端可训练的潜空间预测能力。

架构:
    H = relu(A @ X @ W1 + X @ W2)   # 消息传递编码 [N, H]
    A_pred = sigmoid(H @ W3 @ H.T)  # 边预测解码 [N, N]

参数:
    W1: [D=8, H]  输入变换（邻居信息）
    W2: [D=8, H]  输入变换（自信息）
    W3: [H, H]     边预测交互

训练:
    损失: MSE(A_pred, A_actual)
    优化: 手写梯度反向传播 → SGD

用法:
    from su_memory.sdk._jepa_gnn import GNNPredictor

    predictor = GNNPredictor(hidden_dim=16)
    s_pred = predictor.predict(s_t)                           # 推理
    predictor.training_predict(s_t)                           # 训练前向
    result = predictor.compute_gradients(A_actual)            # 损失 + 梯度
    predictor.apply_gradients(result["grads"], lr=0.01)       # 参数更新
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import numpy as np

from su_memory.sdk._jepa_predictor import JEPAPredictor
from su_memory.sdk._world_model import CausalWorldModelState

logger = logging.getLogger(__name__)


class GNNPredictor(JEPAPredictor):
    """
    可微 GNN 因果图预测器。

    单层消息传递 GNN:
    1. 消息聚合: 邻接矩阵加权邻居特征
    2. 节点更新: 自信息 + 邻居信息 → 隐状态
    3. 边解码: 隐状态对交互 → 预测邻接

    全流程手写梯度反向传播，零外部 autograd 依赖。
    """

    # 输入特征维度（与 CausalWorldModelState.to_node_feature_matrix() 一致）
    INPUT_DIM = 8

    def __init__(
        self,
        hidden_dim: int = 16,
        seed: int = 42,
        l2_reg: float = 0.0,
    ):
        """
        Args:
            hidden_dim: 隐层维度 H
            seed: 随机种子（权重初始化）
            l2_reg: L2 正则系数（0 = 无正则）
        """
        super().__init__(name="gnn_v1")
        self._hidden_dim = hidden_dim
        self._l2_reg = l2_reg
        self._rng = np.random.RandomState(seed)

        # ── 可训练参数 ──
        D = self.INPUT_DIM
        H = hidden_dim
        self.W1: np.ndarray = self._rng.randn(D, H).astype(np.float64) * np.sqrt(2.0 / (D + H))
        self.W2: np.ndarray = self._rng.randn(D, H).astype(np.float64) * np.sqrt(2.0 / (D + H))
        self.W3: np.ndarray = self._rng.randn(H, H).astype(np.float64) * np.sqrt(2.0 / (H + H))

        # ── 前向缓存（训练模式）──
        self._cache: dict = {}

        # ── 训练统计 ──
        self._train_steps: int = 0

    # =====================================================================
    # 参数访问
    # =====================================================================

    def get_params(self) -> dict[str, np.ndarray]:
        """返回可训练参数副本。"""
        return OrderedDict(
            W1=self.W1.copy(),
            W2=self.W2.copy(),
            W3=self.W3.copy(),
        )

    def set_params(self, params: dict[str, np.ndarray]):
        """从字典加载参数。"""
        if "W1" in params:
            self.W1 = np.asarray(params["W1"], dtype=np.float64)
        if "W2" in params:
            self.W2 = np.asarray(params["W2"], dtype=np.float64)
        if "W3" in params:
            self.W3 = np.asarray(params["W3"], dtype=np.float64)

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def train_steps(self) -> int:
        return self._train_steps

    # =====================================================================
    # JEPAPredictor 接口 — 推理
    # =====================================================================

    def predict(self, state: CausalWorldModelState) -> CausalWorldModelState:
        """
        推理模式前向传播（不缓存中间值，不追踪梯度）。

        Args:
            state: 当前因果世界状态 s_t

        Returns:
            预测的下一时刻状态 s_{t+1}
        """
        A, X, node_index = self._extract_graph(state)
        if A.shape[0] == 0:
            return CausalWorldModelState()

        A_pred = self._forward_inference(A, X)
        return self._build_state(A_pred, node_index, state)

    # =====================================================================
    # 训练接口
    # =====================================================================

    def training_predict(self, state: CausalWorldModelState) -> CausalWorldModelState:
        """
        训练模式前向传播（缓存中间值用于反向传播）。

        Args:
            state: 当前因果世界状态 s_t

        Returns:
            预测的下一时刻状态 s_{t+1}
        """
        A, X, node_index = self._extract_graph(state)
        if A.shape[0] == 0:
            self._cache = {"empty": True}
            return CausalWorldModelState()

        # 类型提升（float64 保证梯度精度）
        A = A.astype(np.float64)
        X = X.astype(np.float64)

        # ── 前向传播 ──
        Y = A @ X @ self.W1 + X @ self.W2           # [N, H]
        H = np.maximum(Y, 0.0)                        # relu
        Z = H @ self.W3 @ H.T                        # [N, N]
        Z_clipped = np.clip(Z, -15.0, 15.0)          # 数值稳定
        A_pred = 1.0 / (1.0 + np.exp(-Z_clipped))    # sigmoid

        # ── 缓存中间值 ──
        self._cache = {
            "empty": False,
            "A_input": A,
            "X_input": X,
            "Y": Y,
            "H": H,
            "Z": Z,
            "A_pred": A_pred,
            "node_index": node_index,
            "node_names": list(node_index.keys()),
        }

        self._prediction_count += 1
        return self._build_state(A_pred.astype(np.float32), node_index, state)

    def get_predicted_adj(self) -> np.ndarray | None:
        """返回最近一次 training_predict 的预测邻接矩阵。"""
        if self._cache.get("empty", True):
            return None
        return self._cache["A_pred"]

    def get_node_index(self) -> dict[str, int] | None:
        """返回最近一次 training_predict 的节点索引。"""
        if self._cache.get("empty", True):
            return None
        return self._cache["node_index"]

    def compute_gradients(
        self,
        A_target: np.ndarray,
    ) -> dict:
        """
        计算 MSE 损失 + 手写反向传播梯度。

        L = mean((A_pred - A_target)^2) + l2_reg * ||W||^2

        Args:
            A_target: 目标邻接矩阵 (float64, 与 A_pred 同 shape)

        Returns:
            {"loss": float, "grads": {"W1": ..., "W2": ..., "W3": ...}}
        """
        cache = self._cache
        if cache.get("empty", True):
            return {"loss": 0.0, "grads": {"W1": self.W1 * 0, "W2": self.W2 * 0, "W3": self.W3 * 0}}

        A_pred = cache["A_pred"]
        H_mat  = cache["H"]
        Y_mat  = cache["Y"]
        A_in   = cache["A_input"]
        X_in   = cache["X_input"]
        W1, W2, W3 = self.W1, self.W2, self.W3

        A_target = np.asarray(A_target, dtype=np.float64)
        N = A_pred.shape[0]

        # ── MSE 损失 ──
        diff = A_pred - A_target
        mse = float(np.mean(diff ** 2))

        # ── L2 正则 ──
        l2 = 0.0
        if self._l2_reg > 0:
            l2 = self._l2_reg * (float(np.sum(W1 ** 2)) + float(np.sum(W2 ** 2)) + float(np.sum(W3 ** 2)))

        loss = mse + l2

        # ── 反向传播 ──
        # dL/dA_pred = 2 * (A_pred - A_target) / N^2
        dA_pred = (2.0 / (N * N)) * diff

        # dL/dZ = dA_pred ⊙ sigmoid'(Z) = dA_pred ⊙ A_pred ⊙ (1 - A_pred)
        dZ = dA_pred * A_pred * (1.0 - A_pred)

        # dL/dH = dZ @ H @ W3^T + dZ^T @ H @ W3
        dH = dZ @ H_mat @ W3.T + dZ.T @ H_mat @ W3

        # dL/dY = dH ⊙ relu'(Y) = dH ⊙ (Y > 0)
        dY = dH * (Y_mat > 0).astype(np.float64)

        # dL/dW1 = (A @ X)^T @ dY
        dW1 = (A_in @ X_in).T @ dY
        # dL/dW2 = X^T @ dY
        dW2 = X_in.T @ dY
        # dL/dW3 = H^T @ dZ @ H
        dW3 = H_mat.T @ dZ @ H_mat

        # ── L2 正则梯度 ──
        if self._l2_reg > 0:
            dW1 = dW1 + 2.0 * self._l2_reg * W1
            dW2 = dW2 + 2.0 * self._l2_reg * W2
            dW3 = dW3 + 2.0 * self._l2_reg * W3

        return {
            "loss": loss,
            "mse": mse,
            "l2": l2,
            "grads": {"W1": dW1, "W2": dW2, "W3": dW3},
        }

    def apply_gradients(self, grads: dict[str, np.ndarray], lr: float = 0.01):
        """
        梯度下降参数更新。

        Args:
            grads: compute_gradients() 返回的梯度字典
            lr: 学习率
        """
        self.W1 -= lr * grads["W1"]
        self.W2 -= lr * grads["W2"]
        self.W3 -= lr * grads["W3"]
        self._train_steps += 1
        self._prediction_count += 1

    # =====================================================================
    # 评估 (覆盖基类，增加训练感知指标)
    # =====================================================================

    def evaluate(
        self,
        dataset: list[tuple[CausalWorldModelState, CausalWorldModelState]],
    ) -> dict:
        """
        评估预测精度。

        与基类相同，但额外报告 GNN 参数统计。
        """
        result = super().evaluate(dataset)
        result["predictor"] = "gnn_v1"
        result["hidden_dim"] = self._hidden_dim
        result["train_steps"] = self._train_steps
        result["param_norm"] = {
            "W1": round(float(np.linalg.norm(self.W1)), 4),
            "W2": round(float(np.linalg.norm(self.W2)), 4),
            "W3": round(float(np.linalg.norm(self.W3)), 4),
        }
        return result

    # =====================================================================
    # 内部方法
    # =====================================================================

    def _forward_inference(self, A: np.ndarray, X: np.ndarray) -> np.ndarray:
        """推理模式前向（float32，不做缓存）。"""
        A = A.astype(np.float64)
        X = X.astype(np.float64)
        Y = A @ X @ self.W1 + X @ self.W2
        H = np.maximum(Y, 0.0)
        Z = H @ self.W3 @ H.T
        Z_clipped = np.clip(Z, -15.0, 15.0)
        A_pred = 1.0 / (1.0 + np.exp(-Z_clipped))
        return A_pred.astype(np.float32)

    def _extract_graph(
        self, state: CausalWorldModelState,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        """从状态提取邻接矩阵、节点特征、节点索引。"""
        A = state.to_adjacency_matrix()
        X = state.to_node_feature_matrix()
        node_index = state._build_node_index()
        return A, X, node_index

    def _build_state(
        self,
        A_pred: np.ndarray,
        node_index: dict[str, int],
        template: CausalWorldModelState,
    ) -> CausalWorldModelState:
        """从预测邻接矩阵 + 节点索引重建 CausalWorldModelState。"""
        inv_index = {v: k for k, v in node_index.items()}
        N = A_pred.shape[0]
        causal_edges: list[dict] = []

        for i in range(N):
            for j in range(N):
                rho = float(A_pred[i, j])
                if rho > 0.05 and i != j:
                    cause_name = inv_index.get(i, f"n{i}")
                    effect_name = inv_index.get(j, f"n{j}")
                    edge = {
                        "cause": cause_name,
                        "effect": effect_name,
                        "rho": round(rho, 4),
                        "confidence": round(min(rho + 0.15, 1.0), 4),
                        "verdict": "novel" if rho > 0.5 else "none",
                        "energy_relation": "neutral",
                        "bayes_factor": 0.0,
                    }
                    causal_edges.append(edge)

        n_confirmed = sum(1 for e in causal_edges if e.get("verdict") == "confirmed")
        n_novel = sum(1 for e in causal_edges if e.get("verdict") == "novel")

        return CausalWorldModelState(
            causal_edges=causal_edges,
            active_states=set(template.active_states),
            n_confirmed=n_confirmed,
            n_novel=n_novel,
            n_suppressed=template.n_suppressed,
            n_memories=template.n_memories,
            timestamp=template.timestamp,
            counterfactual_graph=template.counterfactual_graph,
            do_interventions=list(template.do_interventions),
            temporal_info=template.temporal_info,
            belief_tracker=template.belief_tracker,
            cognitive_gaps=list(template.cognitive_gaps),
        )

    def __repr__(self) -> str:
        return (
            f"GNNPredictor(hidden_dim={self._hidden_dim}, "
            f"train_steps={self._train_steps})"
        )


# =============================================================================
# 对齐工具
# =============================================================================


def align_adjacency(
    state: CausalWorldModelState,
    target_node_index: dict[str, int],
) -> np.ndarray:
    """
    将状态的邻接矩阵对齐到目标节点索引。

    用于训练时将 s_{t+1} 的邻接矩阵对齐到 s_t 的节点索引，
    确保 A_pred 和 A_target 形状一致。

    Args:
        state: 源因果世界状态
        target_node_index: 目标节点索引 {"node_name": idx}

    Returns:
        shape=(N, N) 的 float64 邻接矩阵，N=len(target_node_index)
    """
    n = len(target_node_index)
    adj = np.zeros((n, n), dtype=np.float64)

    state._build_node_index()

    for e in state.causal_edges:
        cause = state._get_node_name(e, "cause")
        effect = state._get_node_name(e, "effect")
        rho = e.get("rho", 0.0)

        # 两边都必须出现在目标索引中
        if cause in target_node_index and effect in target_node_index:
            i = target_node_index[cause]
            j = target_node_index[effect]
            adj[i, j] = float(rho)

    return adj
