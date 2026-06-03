"""
su-memory v4.0.0 M3 — GAT Encoder
=================================

可微图注意力编码器，替代 discover() 中的统计因果推断层。

架构:
    X [N, D] → Q = X @ W_q, K = X @ W_k    # 可学习投影
    A_enc = sigmoid(Q @ K^T / √d_k)          # 注意力边预测 [N, N]

与 GNN 预测器串联形成端到端可微闭环:
    memories → preprocess → X → GATEncoder → A_enc → GNNPredictor → A_pred
                                        ↑____________________________↓
                                                   loss + backward

训练: 手写梯度反向传播，零外部 autograd 依赖。

用法:
    from su_memory.sdk._jepa_gat_encoder import GATEncoder

    encoder = GATEncoder(input_dim=8, key_dim=16)
    X = state.to_node_feature_matrix()      # [N, 8]
    A_enc = encoder.forward(X)              # [N, N]
    A_enc = encoder.training_forward(X)     # [N, N] + 缓存
    grads = encoder.backward(dA)            # 反向传播
    encoder.apply_gradients(grads, lr=0.01) # 参数更新
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import numpy as np

logger = logging.getLogger(__name__)


class GATEncoder:
    """
    可微图注意力编码器。

    单层 scaled dot-product attention:
    1. 投影: Q = X @ W_q, K = X @ W_k
    2. 注意力得分: S = Q @ K^T / √d_k
    3. 边预测: A_enc = σ(S)

    参数:
        W_q: [input_dim, key_dim]  查询投影
        W_k: [input_dim, key_dim]  键投影
        (总计 2 × input_dim × key_dim ≈ 256 参数)
    """

    def __init__(
        self,
        input_dim: int = 8,
        key_dim: int = 16,
        seed: int = 42,
        l2_reg: float = 0.0,
    ):
        """
        Args:
            input_dim: 输入节点特征维度 (与 to_node_feature_matrix() 一致)
            key_dim: 注意力键/查询维度 d_k
            seed: 随机种子
            l2_reg: L2 正则系数
        """
        self._input_dim = input_dim
        self._key_dim = key_dim
        self._l2_reg = l2_reg
        self._rng = np.random.RandomState(seed)

        # ── 可训练参数 ──
        D, K = input_dim, key_dim
        self.W_q: np.ndarray = self._rng.randn(D, K).astype(np.float64) * np.sqrt(2.0 / (D + K))
        self.W_k: np.ndarray = self._rng.randn(D, K).astype(np.float64) * np.sqrt(2.0 / (D + K))

        # ── 前向缓存 ──
        self._cache: dict = {}

        # ── 统计 ──
        self._train_steps: int = 0
        self._forward_count: int = 0

    # =====================================================================
    # 参数访问
    # =====================================================================

    def get_params(self) -> dict[str, np.ndarray]:
        """返回可训练参数副本。"""
        return OrderedDict(
            W_q=self.W_q.copy(),
            W_k=self.W_k.copy(),
        )

    def set_params(self, params: dict[str, np.ndarray]):
        """从字典加载参数。"""
        if "W_q" in params:
            self.W_q = np.asarray(params["W_q"], dtype=np.float64)
        if "W_k" in params:
            self.W_k = np.asarray(params["W_k"], dtype=np.float64)

    @property
    def key_dim(self) -> int:
        return self._key_dim

    @property
    def train_steps(self) -> int:
        return self._train_steps

    # =====================================================================
    # 推理
    # =====================================================================

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        推理模式前向（不缓存）。

        Args:
            X: [N, D] 节点特征矩阵 (float32 or float64)

        Returns:
            A_enc: [N, N] 边预测邻接矩阵 (float32)
        """
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape
        if D != self._input_dim:
            raise ValueError(f"输入维度 {D} != {self._input_dim}")

        scale = np.sqrt(float(self._key_dim))
        Q = X @ self.W_q              # [N, K]
        K = X @ self.W_k              # [N, K]
        S = Q @ K.T / scale           # [N, N]
        S_clipped = np.clip(S, -15.0, 15.0)
        A_enc = 1.0 / (1.0 + np.exp(-S_clipped))

        self._forward_count += 1
        return A_enc.astype(np.float32)

    # =====================================================================
    # 训练
    # =====================================================================

    def training_forward(self, X: np.ndarray) -> np.ndarray:
        """
        训练模式前向（缓存中间值用于反向传播）。

        Args:
            X: [N, D] 节点特征矩阵

        Returns:
            A_enc: [N, N] 边预测邻接矩阵 (float64)
        """
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape

        scale = np.sqrt(float(self._key_dim))
        Q = X @ self.W_q
        K = X @ self.W_k
        S = Q @ K.T / scale
        S_clipped = np.clip(S, -15.0, 15.0)
        A_enc = 1.0 / (1.0 + np.exp(-S_clipped))

        self._cache = {
            "X": X,
            "Q": Q,
            "K": K,
            "S": S,
            "A_enc": A_enc,
        }

        self._forward_count += 1
        return A_enc

    def compute_gradients(
        self,
        dA: np.ndarray,
        A_enc: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """
        从上游梯度计算参数梯度。

        dA = ∂L/∂A_enc, 从 GNN 预测器的损失反向传播得到。

        反向传播链:
            dA → sigmoid' → dS → dQ,dK → dW_q,dW_k

        Args:
            dA: [N, N] 上游梯度 ∂L/∂A_enc
            A_enc: 前向输出（可选，默认使用缓存）

        Returns:
            {"W_q": ..., "W_k": ..., "loss_component": ...}
        """
        cache = self._cache
        if not cache:
            return {"W_q": np.zeros_like(self.W_q), "W_k": np.zeros_like(self.W_k)}

        dA = np.asarray(dA, dtype=np.float64)
        A_enc = A_enc if A_enc is not None else cache["A_enc"]
        cache["S"]
        Q = cache["Q"]
        K = cache["K"]
        X = cache["X"]
        scale = np.sqrt(float(self._key_dim))

        # ── 1. sigmoid 反向 ──
        # dL/dS = dA ⊙ σ'(S) = dA ⊙ A_enc ⊙ (1 - A_enc)
        dS = dA * A_enc * (1.0 - A_enc)

        # ── 2. S = Q @ K^T / scale ──
        # dL/dQ = (dL/dS) @ K / scale
        dQ = dS @ K / scale
        # dL/dK = (dL/dS)^T @ Q / scale
        dK = dS.T @ Q / scale

        # ── 3. Q = X @ W_q, K = X @ W_k ──
        dW_q = X.T @ dQ
        dW_k = X.T @ dK

        # ── L2 正则梯度 ──
        if self._l2_reg > 0:
            dW_q = dW_q + 2.0 * self._l2_reg * self.W_q
            dW_k = dW_k + 2.0 * self._l2_reg * self.W_k

        return {"W_q": dW_q, "W_k": dW_k}

    def compute_gradients_from_mse(
        self,
        A_target: np.ndarray,
    ) -> dict:
        """
        直接从 MSE 损失计算梯度（不从上游接收 dA）。

        L = mean((A_enc - A_target)^2)

        用于独立训练 GAT 编码器或作为辅助损失。

        Args:
            A_target: [N, N] 目标邻接矩阵

        Returns:
            {"loss": float, "mse": float, "W_q": ..., "W_k": ...}
        """
        cache = self._cache
        if not cache:
            return {"loss": 0.0, "mse": 0.0,
                    "W_q": np.zeros_like(self.W_q), "W_k": np.zeros_like(self.W_k)}

        A_enc = cache["A_enc"]
        A_target = np.asarray(A_target, dtype=np.float64)
        N = A_enc.shape[0]

        diff = A_enc - A_target
        mse = float(np.mean(diff ** 2))

        l2 = 0.0
        if self._l2_reg > 0:
            l2 = self._l2_reg * (float(np.sum(self.W_q ** 2)) + float(np.sum(self.W_k ** 2)))

        loss = mse + l2

        # ── dA = 2 * (A_enc - A_target) / N^2 ──
        dA = (2.0 / (N * N)) * diff

        grads = self.compute_gradients(dA, A_enc)
        grads["loss"] = loss
        grads["mse"] = mse
        if self._l2_reg > 0:
            grads["l2"] = l2

        return grads

    def apply_gradients(
        self,
        grads: dict[str, np.ndarray],
        lr: float = 0.01,
    ):
        """
        梯度下降参数更新。

        Args:
            grads: compute_gradients() 返回的梯度字典
            lr: 学习率
        """
        self.W_q -= lr * grads.get("W_q", 0.0)
        self.W_k -= lr * grads.get("W_k", 0.0)
        self._train_steps += 1

    # =====================================================================
    # 工具方法
    # =====================================================================

    def get_attention_scores(self) -> np.ndarray | None:
        """返回缓存中的注意力得分 S = Q @ K^T / √d_k。"""
        if not self._cache:
            return None
        return self._cache.get("S")

    def to_adjacency_matrix(
        self,
        X: np.ndarray,
        threshold: float = 0.05,
    ) -> np.ndarray:
        """
        阈值化邻接矩阵（去除弱边）。

        Args:
            X: [N, D] 节点特征
            threshold: 边权重阈值

        Returns:
            thresholded adjacency
        """
        A = self.forward(X)
        A_thresh = A.copy()
        A_thresh[A_thresh < threshold] = 0.0
        return A_thresh

    def __repr__(self) -> str:
        return (
            f"GATEncoder(dim={self._input_dim}, key={self._key_dim}, "
            f"steps={self._train_steps})"
        )


# =============================================================================
# 可微编码器桥接
# =============================================================================


def preprocess_memories_to_features(
    world_model,
    memories: list[dict],
    state=None,
) -> tuple[np.ndarray, dict[str, int], list[str]]:
    """
    将记忆预处理为 GAT 编码器可用的节点特征。

    流程:
    1. 用 discover() 做一次快速的因果发现（不含 BayesianCausal 后验）
    2. 从产生的 state 提取节点特征矩阵和节点索引
    3. 节点特征作为 GAT 编码器的输入

    这一步是不可微的，但只用一次 —— 之后 GAT 编码器的参数
    在端到端训练中完全可微。

    Args:
        world_model: MCIWorldModel 实例
        memories: 记忆列表
        state: 预计算的 CausalWorldModelState（缓存优化，跳过 discover()）

    Returns:
        (X, node_index, node_names):
        - X: [N, 8] 节点特征矩阵
        - node_index: {"node_name": idx}
        - node_names: ["node1", "node2", ...]
    """
    # 使用缓存的 state 或重新运行 discover()
    if state is not None:
        pass  # 使用预计算状态
    else:
        state = world_model.discover(memories, verbose=False)

    X = state.to_node_feature_matrix()      # [N, 8]
    node_index = state._build_node_index()
    node_names = list(node_index.keys())

    return X, node_index, node_names


def features_to_state(
    A_enc: np.ndarray,
    node_index: dict[str, int],
    template_state,
) -> CausalWorldModelState:  # noqa: F821 (forward ref via __future__ annotations)
    """
    从 GAT 编码器的输出重建 CausalWorldModelState。

    Args:
        A_enc: [N, N] 预测邻接矩阵
        node_index: {"name": idx}
        template_state: 模板 CausalWorldModelState (for metadata)

    Returns:
        CausalWorldModelState
    """
    from su_memory.sdk._world_model import CausalWorldModelState

    inv_index = {v: k for k, v in node_index.items()}
    N = A_enc.shape[0]
    causal_edges: list[dict] = []

    for i in range(N):
        for j in range(N):
            rho = float(A_enc[i, j])
            if rho > 0.05 and i != j:
                edge = {
                    "cause": inv_index.get(i, f"n{i}"),
                    "effect": inv_index.get(j, f"n{j}"),
                    "rho": round(rho, 4),
                    "confidence": round(min(rho + 0.15, 1.0), 4),
                    "verdict": "novel" if rho > 0.5 else "none",
                    "energy_relation": "neutral",
                    "bayes_factor": 0.0,
                }
                causal_edges.append(edge)

    n_novel = sum(1 for e in causal_edges if e.get("verdict") == "novel")

    return CausalWorldModelState(
        causal_edges=causal_edges,
        active_states=set(template_state.active_states) if template_state else set(),
        n_confirmed=0,
        n_novel=n_novel,
        n_suppressed=template_state.n_suppressed if template_state else 0,
        n_memories=template_state.n_memories if template_state else 0,
        timestamp=template_state.timestamp if template_state else "",
        temporal_info=template_state.temporal_info if template_state else None,
        cognitive_gaps=list(template_state.cognitive_gaps) if template_state else [],
    )
