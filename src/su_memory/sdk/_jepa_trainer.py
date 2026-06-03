"""
su-memory v4.0.0 M2 — JEPA Trainer
===================================

JEPA 端到端训练循环，支持两类预测器:

1. 不可参基线 (Identity/EnergyPropagation/BeliefPropagation):
   训练损失 = state_distance(s_pred, s_actual) + α·L_energy + β·L_cons

2. 可微 GNN (GNNPredictor):
   训练损失 = MSE(A_pred, A_actual) + α·L_energy + β·L_cons
   反向传播: 手写梯度 → SGD 参数更新

用法:
    from su_memory.sdk._jepa_trainer import JEPATrainer

    trainer = JEPATrainer(encoder, predictor, dataset)
    stats = trainer.train(n_epochs=10)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from su_memory.sdk._world_model import CausalWorldModelState

logger = logging.getLogger(__name__)


@dataclass
class JEPATrainingStats:
    """JEPA 训练统计。"""

    n_epochs: int = 0
    n_pairs: int = 0
    final_loss: float = 0.0
    min_loss: float = float("inf")
    loss_history: list[float] = field(default_factory=list)
    alpha_energy: float = 0.1
    beta_cons: float = 0.05

    def to_dict(self) -> dict:
        return {
            "n_epochs": self.n_epochs,
            "n_pairs": self.n_pairs,
            "final_loss": round(self.final_loss, 6),
            "min_loss": round(self.min_loss, 6) if self.min_loss != float("inf") else None,
            "alpha_energy": self.alpha_energy,
            "beta_cons": self.beta_cons,
            "loss_trend": self._trend(),
        }

    def _trend(self) -> str:
        if len(self.loss_history) < 3:
            return "insufficient_data"
        if self.loss_history[-1] < self.loss_history[0] * 0.9:
            return "converging"
        elif self.loss_history[-1] > self.loss_history[0] * 1.05:
            return "diverging"
        return "stable"


class JEPATrainer:
    """
    JEPA 训练器：端到端训练循环。

    训练目标:
        minimize Σ [state_distance(predict(encode(mem_t)), encode(mem_{t+1}))
                     + α * energy_loss
                     + β * consistency_loss]
    """

    def __init__(
        self,
        encoder,
        predictor,
        dataset=None,
        alpha_energy: float = 0.1,
        beta_cons: float = 0.05,
    ):
        """
        Args:
            encoder: JEPAEncoder 实例
            predictor: JEPAPredictor 实例
            dataset: JEPADataset 实例（可选，train() 时传入）
            alpha_energy: 能量损失权重
            beta_cons: 能量守恒损失权重
        """
        self.encoder = encoder
        self.predictor = predictor
        self.dataset = dataset
        self.alpha_energy = alpha_energy
        self.beta_cons = beta_cons
        self._stats = JEPATrainingStats(
            alpha_energy=alpha_energy, beta_cons=beta_cons,
        )
        self._is_gnn = self._detect_gnn()
        self._is_e2e = self._detect_e2e()

    def _detect_gnn(self) -> bool:
        """检测预测器是否支持可微训练（M2 GNN）。"""
        return hasattr(self.predictor, 'training_predict') and \
               hasattr(self.predictor, 'compute_gradients') and \
               hasattr(self.predictor, 'apply_gradients')

    def _detect_e2e(self) -> bool:
        """
        检测是否支持端到端可微训练（M3 GAT + GNN）。

        条件:
        - encoder 有 training_encode 方法 (GAT 编码器)
        - predictor 有 training_predict 方法 (GNN 预测器)
        """
        if self.encoder is None:
            return False
        return hasattr(self.encoder, 'training_encode') and self._is_gnn

    @property
    def stats(self) -> JEPATrainingStats:
        return self._stats

    # -----------------------------------------------------------------
    # 训练
    # -----------------------------------------------------------------

    def train(
        self,
        dataset=None,
        n_epochs: int = 10,
        learning_rate: float = 0.01,
    ) -> JEPATrainingStats:
        """
        执行 JEPA 训练循环。

        Args:
            dataset: JEPADataset 实例（覆盖初始化时的 dataset）
            n_epochs: 训练轮数
            learning_rate: 学习率（用于 M2 阶段可训预测器；当前基线不可训时忽略）

        Returns:
            JEPATrainingStats 训练统计
        """
        ds = dataset or self.dataset
        if ds is None or len(ds) == 0:
            logger.warning("JEPA 训练数据为空")
            return self._stats

        self._stats = JEPATrainingStats(
            n_epochs=n_epochs,
            n_pairs=len(ds),
            alpha_energy=self.alpha_energy,
            beta_cons=self.beta_cons,
        )

        for epoch in range(n_epochs):
            epoch_losses: list[float] = []

            if self._is_e2e and hasattr(ds, 'memory_pairs') and ds.memory_pairs:
                # ── M3: 端到端 (GAT + GNN) ──
                has_state_cache = hasattr(ds, 'state_pairs') and ds.state_pairs
                for idx, (mem_t, mem_t1) in enumerate(ds.memory_pairs):
                    s_t1 = ds.pairs[idx][1] if idx < len(ds.pairs) else None
                    s_t = ds.state_pairs[idx][0] if has_state_cache and idx < len(ds.state_pairs) else None
                    loss = self._train_e2e_step(mem_t, mem_t1, s_t1, s_t=s_t, learning_rate=learning_rate)
                    epoch_losses.append(loss)
            else:
                for s_t, s_t1 in ds.pairs:
                    if self._is_gnn:
                        loss = self._train_gnn_step(s_t, s_t1, learning_rate)
                    else:
                        loss = self._compute_loss(s_t, s_t1)
                    epoch_losses.append(loss)

            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
            self._stats.loss_history.append(avg_loss)

            if avg_loss < self._stats.min_loss:
                self._stats.min_loss = avg_loss

            logger.info(
                "JEPA Epoch %d/%d | Loss: %.6f | Min: %.6f%s",
                epoch + 1, n_epochs, avg_loss, self._stats.min_loss,
                " [E2E]" if self._is_e2e else (" [GNN]" if self._is_gnn else ""),
            )

        self._stats.final_loss = (
            self._stats.loss_history[-1] if self._stats.loss_history else 0.0
        )
        self._stats.n_epochs = n_epochs
        self._stats.n_pairs = len(ds)

        return self._stats

    # -----------------------------------------------------------------
    # GNN 可微训练步 (M2)
    # -----------------------------------------------------------------

    def _train_gnn_step(
        self,
        s_t: CausalWorldModelState,
        s_t1: CausalWorldModelState,
        learning_rate: float = 0.01,
    ) -> float:
        """
        GNN 单步训练：前向 → 损失 → 反向 → 更新。

        Args:
            s_t: 当前状态
            s_t1: 目标状态
            learning_rate: 学习率

        Returns:
            总损失 float
        """
        from su_memory.sdk._jepa_gnn import align_adjacency

        # ── 1. 训练前向 ──
        s_pred = self.predictor.training_predict(s_t)

        # ── 2. 对齐 A_target ──
        node_index = self.predictor.get_node_index()
        if node_index is None:
            return 0.0
        A_target = np.abs(align_adjacency(s_t1, node_index))

        # ── 3. 损失 + 梯度 ──
        result = self.predictor.compute_gradients(A_target)
        gnn_loss = float(result["loss"])

        # ── 4. 正则化损失（能量 + 守恒）──
        l_energy = self._compute_energy_loss(s_pred, s_t1)
        l_cons = self._compute_consistency_loss(s_pred, s_t1)

        total_loss = gnn_loss + self.alpha_energy * l_energy + self.beta_cons * l_cons

        # ── 5. 参数更新 ──
        self.predictor.apply_gradients(result["grads"], lr=learning_rate)

        return total_loss

    # -----------------------------------------------------------------
    # 端到端可微训练步 (M3)
    # -----------------------------------------------------------------

    def _train_e2e_step(
        self,
        mem_t: list[dict],
        mem_t1: list[dict],
        s_t1: CausalWorldModelState | None,
        s_t: CausalWorldModelState | None = None,
        learning_rate: float = 0.01,
    ) -> float:
        """
        M3 端到端训练：GAT 编码器 → GNN 预测器 → 损失 → 反向传播。

        梯度流:
            dL/dW_pred ← dL/dA_pred ← GNN.backward()
            dL/dW_enc  ← dL/dA_enc  ← dA_pred/dA_enc chain rule
                    ← GAT.backward(dA_enc)

        Args:
            mem_t: t 时刻的记忆窗口
            mem_t1: t+1 时刻的记忆窗口
            s_t1: t+1 时刻的状态（用于目标 A_target）
            s_t: t 时刻的预计算状态（缓存，跳过 discover()）
            learning_rate: 学习率

        Returns:
            总损失 float
        """
        from su_memory.sdk._jepa_gnn import align_adjacency

        # ── 1. GAT 编码器前向 (训练模式) ──
        A_enc, node_index = self.encoder.training_encode(mem_t, state=s_t)
        if A_enc.shape[0] == 0 or not node_index:
            return 0.0

        # ── 2. 用 A_enc 构建临时状态传给 GNN ──
        # 从 node_index 构建 node_names
        sorted(node_index.keys(), key=lambda k: node_index[k])
        from su_memory.sdk._jepa_gat_encoder import features_to_state

        template_state = s_t1 if s_t1 else CausalWorldModelState()
        s_enc = features_to_state(A_enc, node_index, template_state)

        # ── 3. GNN 预测器前向 (训练模式) ──
        s_pred = self.predictor.training_predict(s_enc)

        # ── 4. 对齐 A_target ──
        pred_node_index = self.predictor.get_node_index()
        if pred_node_index is None:
            return 0.0

        A_target = None
        if s_t1 is not None:
            A_target = np.abs(align_adjacency(s_t1, pred_node_index))
        else:
            # 无目标状态时，用 A_enc 作为自监督目标
            A_target = np.abs(align_adjacency(s_enc, pred_node_index))

        # ── 5. GNN 损失 + 梯度 ──
        gnn_result = self.predictor.compute_gradients(A_target)
        gnn_loss = float(gnn_result["loss"])

        # ── 6. 反向传播到 GAT 编码器 ──
        # 从 GNN 预测器的前向缓存获取 dL/dA_enc
        # A_enc → GNN → A_pred → loss
        # dL/dA_enc = dL/dA_pred @ dA_pred/dH @ dH/dA_enc (via GNN chain rule)
        # 简化: 直接用 encoder.compute_gradients_from_mse(A_target)
        # 因为 A_enc 应该接近 A_target（编码器质量）
        if self.encoder.gat_encoder is not None:
            enc_result = self.encoder.gat_encoder.compute_gradients_from_mse(A_target)
            enc_loss = float(enc_result.get("mse", 0.0))
            self.encoder.gat_encoder.apply_gradients(enc_result, lr=learning_rate)
        else:
            enc_loss = 0.0

        # ── 7. GNN 参数更新 ──
        self.predictor.apply_gradients(gnn_result["grads"], lr=learning_rate)

        # ── 8. 正则化（能量 + 守恒）──
        l_energy = self._compute_energy_loss(s_pred, s_t1 if s_t1 else s_enc)
        l_cons = self._compute_consistency_loss(s_pred, s_t1 if s_t1 else s_enc)

        total_loss = gnn_loss + enc_loss + self.alpha_energy * l_energy + self.beta_cons * l_cons

        return total_loss

    def _compute_loss(
        self,
        s_t: CausalWorldModelState,
        s_t1: CausalWorldModelState,
    ) -> float:
        """
        计算单步 JEPA 训练损失。

        L_total = L_pred + α·L_energy + β·L_cons
        """
        # ── 主项: 预测误差 ──
        s_pred = self.predictor.predict(s_t)
        l_pred = s_pred.state_distance(s_t1)

        # ── 正则项: 能量一致性 ──
        l_energy = self._compute_energy_loss(s_pred, s_t1)

        # ── 正则项: 能量守恒 ──
        l_cons = self._compute_consistency_loss(s_pred, s_t1)

        return l_pred + self.alpha_energy * l_energy + self.beta_cons * l_cons

    def _compute_energy_loss(
        self,
        s_pred: CausalWorldModelState,
        s_actual: CausalWorldModelState,
    ) -> float:
        """
        计算图结构能量损失。

        对 pred 中的每条边，用 _sys/_energy_relations.py 判定是否违反增强/抑制模式。
        """
        try:
            from su_memory._sys._energy_relations import is_enhancing, is_suppressing
        except ImportError:
            return 0.0

        if not s_pred.causal_edges:
            return 0.0

        violations = 0.0
        n_edges = len(s_pred.causal_edges)

        for edge in s_pred.causal_edges:
            edge.get("cause", "")
            edge.get("effect", "")
            energy_rel = edge.get("energy_relation", "neutral")
            rho = edge.get("rho", 0.0)

            # 增强模式下的低权重惩罚
            if energy_rel == "enhance" and rho < 0.3:
                violations += (0.3 - rho) * 2.0
            # 抑制模式下的高权重惩罚
            elif energy_rel == "suppress" and rho > 0.7:
                violations += (rho - 0.7) * 2.0

        return violations / max(n_edges, 1)

    def _compute_consistency_loss(
        self,
        s_pred: CausalWorldModelState,
        s_actual: CausalWorldModelState,
    ) -> float:
        """计算能量总量守恒损失。"""
        pred_total = sum(abs(e.get("rho", 0.0)) for e in s_pred.causal_edges)
        actual_total = sum(abs(e.get("rho", 0.0)) for e in s_actual.causal_edges)
        max_total = max(pred_total, actual_total, 1e-10)
        return abs(pred_total - actual_total) / max_total

    # -----------------------------------------------------------------
    # 评估
    # -----------------------------------------------------------------

    def evaluate(self, dataset=None) -> dict:
        """
        在完整数据集上评估预测器精度。

        Args:
            dataset: JEPADataset 实例

        Returns:
            评估统计字典
        """
        ds = dataset or self.dataset
        if ds is None or len(ds) == 0:
            return {"error": "empty_dataset"}

        pairs_as_tuples = list(ds.pairs)
        result = self.predictor.evaluate(pairs_as_tuples)
        result["trainer_mode"] = "e2e" if self._is_e2e else ("gnn" if self._is_gnn else "baseline")
        return result
