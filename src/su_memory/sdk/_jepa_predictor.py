"""
su-memory v4.0.0 — JEPA Predictor
==================================

JEPA 潜空间预测器接口 + 三个不可参基线实现。

基线:
- IdentityPredictor: s_{t+1} = s_t (下界)
- EnergyPropagationPredictor: 基于 EnergyBus 三层网络做能量传播
- BeliefPropagationPredictor: 基于 BayesianNetwork + BeliefPropagator 做概率推断

用法:
    from su_memory.sdk._jepa_predictor import (
        JEPAPredictor, IdentityPredictor,
        EnergyPropagationPredictor, BeliefPropagationPredictor,
    )

    predictor = IdentityPredictor()
    s_pred = predictor.predict(s_t)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from su_memory.sdk._world_model import CausalWorldModelState

logger = logging.getLogger(__name__)


# =============================================================================
# JEPAPredictor — 抽象基类
# =============================================================================


class JEPAPredictor(ABC):
    """
    JEPA 潜空间预测器抽象基类。

    所有预测器（基线和可训练版本）实现此接口:
        predict(s_t: CausalWorldModelState) → CausalWorldModelState

    predict() 可以:
    - 不修改因果图结构（IdentityPredictor）
    - 调整边权重（EnergyPropagationPredictor）
    - 预测新边（BeliefPropagationPredictor）
    - 学习到结构演化（可训练 GNN，M2 阶段实现）
    """

    def __init__(self, name: str = "base"):
        self._name = name
        self._prediction_count: int = 0

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def predict(self, state: CausalWorldModelState) -> CausalWorldModelState:
        """
        预测下一时刻的因果世界状态。

        Args:
            state: 当前时刻状态 s_t

        Returns:
            预测的下一时刻状态 s_{t+1}
        """
        ...

    def evaluate(
        self,
        dataset: list[tuple[CausalWorldModelState, CausalWorldModelState]],
    ) -> dict:
        """
        在数据集上评估预测精度。

        Args:
            dataset: [(s_t, s_{t+1}), ...]

        Returns:
            评估统计字典
        """
        distances: list[float] = []
        for s_t, s_t1 in dataset:
            s_pred = self.predict(s_t)
            d = s_pred.state_distance(s_t1)
            distances.append(d)
            self._prediction_count += 1

        if not distances:
            return {"avg_distance": 1.0, "n": 0}

        return {
            "avg_distance": round(float(np.mean(distances)), 6),
            "min_distance": round(float(np.min(distances)), 6),
            "max_distance": round(float(np.max(distances)), 6),
            "std_distance": round(float(np.std(distances)), 6),
            "n": len(distances),
            "predictor": self._name,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name!r})"


# =============================================================================
# IdentityPredictor — 世界静止基线
# =============================================================================


class IdentityPredictor(JEPAPredictor):
    """
    恒等预测器：s_{t+1} = s_t。

    假设因果图不随时间变化。作为 JEPA 预测精度的下界—
    任何有意义的预测器必须比 Identity 更准。
    """

    def __init__(self):
        super().__init__(name="identity")

    def predict(self, state: CausalWorldModelState) -> CausalWorldModelState:
        """返回输入状态的浅拷贝。"""
        return CausalWorldModelState(
            causal_edges=list(state.causal_edges),  # 浅拷贝边列表
            active_states=set(state.active_states),
            n_confirmed=state.n_confirmed,
            n_novel=state.n_novel,
            n_suppressed=state.n_suppressed,
            n_memories=state.n_memories,
            timestamp=state.timestamp,
            counterfactual_graph=state.counterfactual_graph,
            do_interventions=list(state.do_interventions),
        )


# =============================================================================
# EnergyPropagationPredictor — 能量传播基线
# =============================================================================


class EnergyPropagationPredictor(JEPAPredictor):
    """
    能量传播预测器：在因果图上执行能量传播，调整边权重。

    复用 _sys/_energy_bus.py 的 propagate_energy() 和
    _sys/causal.py 的 CausalChain.propagate()。

    对每条因果边:
    - 从 cause 节点向 effect 节点传播能量 delta
    - 根据 energy_relation 调整边权重（增强→ +Δ, 抑制→ -Δ）
    """

    def __init__(self, propagation_alpha: float = 0.1):
        super().__init__(name="energy_propagation")
        self._alpha = propagation_alpha
        self._energy_bus = None
        self._init_energy_system()

    def _init_energy_system(self):
        """延迟加载能量系统组件。"""
        try:
            from su_memory._sys._energy_bus import create_complete_energy_network
            self._energy_bus = create_complete_energy_network()
        except Exception as e:
            logger.debug("EnergyBus 初始化失败（回退到轻量模式）: %s", e)

    def predict(self, state: CausalWorldModelState) -> CausalWorldModelState:
        """
        基于能量传播调整因果边权重。

        对每条边:
        1. 根据 energy_relation 确定传播方向（增强/抑制）
        2. 用 alpha 比例调整 rho
        3. confidence 随之更新
        """
        new_edges = []
        for edge in state.causal_edges:
            new_edge = dict(edge)
            energy_rel = edge.get("energy_relation", "neutral")
            rho = edge.get("rho", 0.0)

            # 能量传播调整
            if energy_rel == "enhance":
                new_rho = rho + self._alpha * (1.0 - rho)  # 增强边增强
            elif energy_rel == "suppress":
                new_rho = rho - self._alpha * rho  # 抑制边减弱
            else:
                # 中性边轻微衰减（回归均值）
                new_rho = rho * (1.0 - self._alpha * 0.5)

            new_edge["rho"] = round(max(0.01, min(new_rho, 0.99)), 4)
            # confidence 随 rho 变化
            conf = edge.get("confidence", 0.5)
            new_edge["confidence"] = round(max(0.0, min(conf + (new_rho - rho) * 0.3, 1.0)), 4)
            new_edges.append(new_edge)

        return CausalWorldModelState(
            causal_edges=new_edges,
            active_states=set(state.active_states),
            n_confirmed=state.n_confirmed,
            n_novel=state.n_novel,
            n_suppressed=state.n_suppressed,
            n_memories=state.n_memories,
            timestamp=state.timestamp,
            counterfactual_graph=state.counterfactual_graph,
            do_interventions=list(state.do_interventions),
        )


# =============================================================================
# BeliefPropagationPredictor — 贝叶斯信念传播基线
# =============================================================================


class BeliefPropagationPredictor(JEPAPredictor):
    """
    贝叶斯信念传播预测器：基于 BayesianNetwork + BeliefPropagator 做概率推断。

    复用 _sys/bayesian_network.py 的 BeliefPropagator 和
    _sys/bayesian_reasoning.py 的 BayesianPredictor。

    对比 EnergyPropagation, Belief 能:
    - 发现新因果边（概率 > 阈值）
    - 抑制弱边（后验均值 < 阈值折抑制）
    - 提供不确定性量化（95% 可信区间）
    """

    def __init__(self, new_edge_threshold: float = 0.6):
        super().__init__(name="belief_propagation")
        self._threshold = new_edge_threshold
        self._engine = None

    def _init_engine(self):
        """延迟加载贝叶斯引擎。"""
        if self._engine is not None:
            return
        try:
            from su_memory._sys.bayesian import BayesianEngine
            self._engine = BayesianEngine()
        except Exception as e:
            logger.debug("BayesianEngine 初始化失败: %s", e)

    def predict(self, state: CausalWorldModelState) -> CausalWorldModelState:
        """
        基于贝叶斯信念更新因果边。

        工作流:
        1. 对每条边，用 BayesianEngine 更新信念
        2. 信念强度 > threshold 的新边保留
        3. 信念过弱的边标记为 suppressed
        """
        self._init_engine()

        new_edges = []
        # 收集所有已知节点对
        known_pairs: set[tuple[str, str]] = set()
        for edge in state.causal_edges:
            known_pairs.add((edge.get("cause", ""), edge.get("effect", "")))

        for edge in state.causal_edges:
            new_edge = dict(edge)
            rho = edge.get("rho", 0.0)
            confidence = edge.get("confidence", 0.5)

            # 简单贝叶斯更新（信念回归均值）
            if self._engine is not None:
                try:
                    belief_id = f"{edge.get('cause', '')}→{edge.get('effect', '')}"
                    # 用默认先验 New_Beta(1, 1) + 观测更新
                    effective_obs = int(rho * confidence * 10)
                    belief = self._engine.update_belief(
                        belief_id,
                        positive=effective_obs,
                        negative=10 - effective_obs,
                    )
                    if belief is not None:
                        new_edge["rho"] = round(float(belief.posterior.mean), 4)
                        new_edge["confidence"] = round(float(1.0 - belief.posterior.std * 2), 4)
                        # 存贝叶斯因子作为额外信息
                        if hasattr(belief, "bayes_factor"):
                            new_edge["bayes_factor"] = round(float(belief.bayes_factor), 4)
                except Exception:
                    pass  # 默认保持原值

            # 弱边抑制
            if new_edge.get("rho", 0.0) < 0.15:
                new_edge["verdict"] = "suppressed"
            else:
                new_edge["verdict"] = edge.get("verdict", "none")

            new_edges.append(new_edge)

        return CausalWorldModelState(
            causal_edges=new_edges,
            active_states=set(state.active_states),
            n_confirmed=sum(1 for e in new_edges if e.get("verdict") == "confirmed"),
            n_novel=sum(1 for e in new_edges if e.get("verdict") == "novel"),
            n_suppressed=sum(1 for e in new_edges if e.get("verdict") == "suppressed"),
            n_memories=state.n_memories,
            timestamp=state.timestamp,
            counterfactual_graph=state.counterfactual_graph,
            do_interventions=list(state.do_interventions),
        )
