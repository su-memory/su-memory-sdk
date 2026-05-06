"""
贝叶斯推理统一集成模块

将 BayesianEngine + BayesianNetwork + EvidenceCollector 整合为单一入口，
并桥接现有的 BeliefTracker / PredictionModule / MetaCognition。

对外暴露：BayesianReasoningSystem
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import math
import time
import json

from .bayesian import (
    BayesianEngine,
    BetaDistribution,
    LikelihoodFunctions,
    BayesianBelief
)
from .bayesian_network import BayesianNetwork, BeliefPropagator
from .evidence import EvidenceCollector, EvidenceRecord, SourceProfile
from .states import BeliefTracker, BayesianBeliefTracker


# ============================================================
# 预测增强
# ============================================================

class BayesianPredictor:
    """
    贝叶斯预测器

    基于后验概率更新历史预测，提升预测准确度：
    1. 使用 Beta-Binomial 模型做事件概率预测
    2. 基于贝叶斯网络做因果预测
    3. 预测校准（Calibration）
    """

    def __init__(
        self,
        engine: BayesianEngine = None,
        network: BayesianNetwork = None,
        evidence: EvidenceCollector = None
    ):
        self._engine = engine or BayesianEngine()
        self._network = network or BayesianNetwork()
        self._evidence = evidence or EvidenceCollector(self._engine)

        # 预测历史（用于校准）
        self._prediction_history: List[Dict] = []
        self._calibration_curve: List[Tuple[float, float]] = []  # (predicted, actual)

    def predict_event_probability(
        self,
        event_id: str,
        context_evidence: Dict[str, bool] = None,
        use_network: bool = True
    ) -> Dict:
        """
        预测事件概率

        综合两个来源：
        1. 直接信念（BayesianEngine）
        2. 网络推断（BayesianNetwork）

        Args:
            event_id: 事件ID
            context_evidence: 上下文证据
            use_network: 是否使用贝叶斯网络

        Returns:
            {"probability": float, "uncertainty": float, "ci_95": tuple, "sources": [...]}
        """
        probability = 0.5
        uncertainty = 1.0
        sources = []

        # 1. 直接从引擎获取
        belief = self._engine.get_belief(event_id)
        if belief and belief.posterior.effective_sample_size > 2:
            prob_direct = belief.posterior.mean
            uncert_direct = belief.posterior.std
            sources.append({
                "source": "direct_belief",
                "probability": prob_direct,
                "weight": min(1.0, belief.posterior.effective_sample_size / 10)
            })
            probability = prob_direct
            uncertainty = uncert_direct

        # 2. 贝叶斯网络推断
        if use_network and self._network.get_node(event_id):
            evidence = context_evidence or {}
            posteriors = self._network.infer_posterior([event_id], evidence)
            posterior = posteriors.get(event_id)
            if posterior and posterior.effective_sample_size > 2:
                prob_net = posterior.mean
                uncert_net = posterior.std
                sources.append({
                    "source": "bayesian_network",
                    "probability": prob_net,
                    "weight": 0.5
                })

                # 融合：加权平均
                if len(sources) == 2:
                    w1 = sources[0]["weight"]
                    w2 = sources[1]["weight"]
                    total_w = w1 + w2
                    probability = (prob_direct * w1 + prob_net * w2) / total_w
                    uncertainty = (uncert_direct * w1 + uncert_net * w2) / total_w

        # 3. 证据叠加
        if context_evidence:
            evidence_str = self._evidence.compute_evidence_strength(event_id)
            if evidence_str["evidence_count"] > 0:
                sources.append({
                    "source": "recent_evidence",
                    "weighted_ratio": evidence_str["weighted_ratio"],
                    "weight": 0.3
                })

        # 校准
        if self._calibration_curve:
            probability = self._calibrate(probability)

        ci_95 = (
            max(0.0, probability - 1.96 * uncertainty),
            min(1.0, probability + 1.96 * uncertainty)
        )

        return {
            "event_id": event_id,
            "probability": probability,
            "uncertainty": uncertainty,
            "ci_95": ci_95,
            "sources": sources,
            "confidence_level": self._assess_confidence(uncertainty)
        }

    def record_prediction_outcome(
        self,
        event_id: str,
        predicted_prob: float,
        actual_outcome: bool
    ):
        """
        记录预测结果以校准模型

        Args:
            event_id: 事件ID
            predicted_prob: 预测的概率
            actual_outcome: 实际结果
        """
        self._prediction_history.append({
            "event_id": event_id,
            "predicted_prob": predicted_prob,
            "actual_outcome": actual_outcome,
            "timestamp": time.time(),
            "correct": (predicted_prob >= 0.5) == actual_outcome
        })

        # 更新校准曲线
        self._calibration_curve.append((predicted_prob, float(actual_outcome)))

        # 反馈到引擎
        self._engine.observe(
            belief_id=event_id,
            success=actual_outcome,
            weight=0.5,
            source="prediction_outcome"
        )

    def _calibrate(self, raw_prob: float) -> float:
        """
        校准概率（Platt Scaling 简化版）

        基于历史预测-实际数据校正预测概率
        """
        if len(self._calibration_curve) < 5:
            return raw_prob

        # 简单校准：如果系统整体偏乐观/悲观，进行调整
        total_pred = sum(p for p, _ in self._calibration_curve)
        total_actual = sum(a for _, a in self._calibration_curve)
        n = len(self._calibration_curve)

        if n == 0:
            return raw_prob

        bias = total_pred / n - total_actual / n

        # 向实际方向调整
        calibrated = raw_prob - bias * 0.5
        return max(0.0, min(1.0, calibrated))

    def _assess_confidence(self, uncertainty: float) -> str:
        """根据不确定性评估置信度等级"""
        if uncertainty < 0.1:
            return "high"
        elif uncertainty < 0.2:
            return "medium"
        elif uncertainty < 0.3:
            return "low"
        else:
            return "very_low"

    def get_calibration_report(self) -> Dict:
        """获取校准报告"""
        if len(self._prediction_history) < 5:
            return {"status": "insufficient_data", "count": len(self._prediction_history)}

        total = len(self._prediction_history)
        correct = sum(1 for p in self._prediction_history if p["correct"])
        accuracy = correct / max(total, 1)

        # Brier Score
        brier = sum(
            (p["predicted_prob"] - float(p["actual_outcome"])) ** 2
            for p in self._prediction_history
        ) / max(total, 1)

        # 校准度 (Calibration-in-the-large)
        mean_pred = sum(p["predicted_prob"] for p in self._prediction_history) / total
        mean_actual = sum(float(p["actual_outcome"]) for p in self._prediction_history) / total
        calibration_bias = mean_pred - mean_actual

        return {
            "total_predictions": total,
            "accuracy": accuracy,
            "brier_score": brier,
            "calibration_bias": calibration_bias,
            "status": (
                "well_calibrated" if abs(calibration_bias) < 0.1
                else "overconfident" if calibration_bias > 0.1
                else "underconfident"
            )
        }


# ============================================================
# 推理建议引擎
# ============================================================

class BayesianAdvisor:
    """
    贝叶斯建议引擎

    基于后验概率提供决策建议：
    1. 信息价值 (VoI) 计算
    2. 最优行动推荐
    3. 不确定性降序建议
    """

    def __init__(
        self,
        engine: BayesianEngine = None,
        network: BayesianNetwork = None,
        predictor: BayesianPredictor = None
    ):
        self._engine = engine or BayesianEngine()
        self._network = network or BayesianNetwork()
        self._predictor = predictor or BayesianPredictor(self._engine, self._network)

    def value_of_information(
        self,
        target_belief_id: str,
        candidate_evidence_sources: List[str],
        decision_threshold: float = 0.7
    ) -> List[Dict]:
        """
        计算信息的预期价值 (Expected Value of Information, EVoI)

        帮助决策：收集哪些证据最能降低不确定性？

        Args:
            target_belief_id: 目标信念
            candidate_evidence_sources: 候选证据源（对应的信念节点）
            decision_threshold: 决策阈值

        Returns:
            按信息价值排序的建议列表
        """
        results = []

        # 当前不确定性
        current_belief = self._engine.get_belief(target_belief_id)
        if not current_belief:
            return results

        current_uncertainty = current_belief.posterior.std
        current_mean = current_belief.posterior.mean

        for source_id in candidate_evidence_sources:
            # 查询因果强度
            causal = self._network.query_causal_strength(source_id, target_belief_id)
            causal_strength = abs(causal["causal_strength"]) if causal else 0.1

            # 估算：如果我们知道 source 的真实值，能减少多少不确定性？
            source_belief = self._engine.get_belief(source_id)
            source_uncertainty = source_belief.posterior.std if source_belief else 0.5

            # EVoI = |causal_strength| × source_uncertainty × (1 - |current_mean - 0.5|)
            evoi = causal_strength * source_uncertainty * (1 - abs(current_mean - 0.5) * 2)

            # 考虑获取该证据的成本（来源可靠性越低，成本越高）
            source_reliability = self._predictor._evidence.get_source_reliability(source_id) if hasattr(self._predictor, '_evidence') else 0.7
            net_voi = evoi * source_reliability

            results.append({
                "source_id": source_id,
                "causal_strength": causal_strength,
                "value_of_information": net_voi,
                "current_uncertainty_reduction": evoi,
                "source_reliability": source_reliability,
                "recommendation": (
                    "强烈建议收集" if net_voi > 0.3
                    else "建议收集" if net_voi > 0.15
                    else "可选收集" if net_voi > 0.05
                    else "可暂缓"
                )
            })

        results.sort(key=lambda x: x["value_of_information"], reverse=True)
        return results

    def recommend_actions(
        self,
        belief_id: str,
        action_options: List[Dict] = None
    ) -> List[Dict]:
        """
        基于后验概率推荐行动

        Args:
            belief_id: 目标信念
            action_options: [{"name": str, "if_true": float, "if_false": float}, ...]
                           每个 action 定义当信念为真/假时的收益

        Returns:
            按期望收益排序的行动建议
        """
        belief = self._engine.get_belief(belief_id)
        if not belief:
            return []

        prob = belief.posterior.mean

        if not action_options:
            # 默认行动选项
            action_options = [
                {"name": "confident_action", "if_true": 1.0, "if_false": -0.5,
                 "description": "当信念置信度高时执行"},
                {"name": "cautious_action", "if_true": 0.5, "if_false": -0.2,
                 "description": "谨慎行动"},
                {"name": "wait_and_collect", "if_true": 0.1, "if_false": 0.1,
                 "description": "等待更多证据"},
                {"name": "avoid_action", "if_true": -0.5, "if_false": 1.0,
                 "description": "当信念为假时执行"},
            ]

        results = []
        for action in action_options:
            expected_utility = prob * action["if_true"] + (1 - prob) * action["if_false"]

            # 风险 = 期望收益的方差
            variance = (
                prob * (action["if_true"] - expected_utility) ** 2 +
                (1 - prob) * (action["if_false"] - expected_utility) ** 2
            )

            results.append({
                "action_name": action["name"],
                "description": action.get("description", ""),
                "expected_utility": expected_utility,
                "risk": math.sqrt(variance),
                "recommended": expected_utility > 0,
                "confidence": min(1.0, max(0.0, expected_utility / (abs(expected_utility) + math.sqrt(variance))))
            })

        results.sort(key=lambda x: x["expected_utility"], reverse=True)
        return results

    def uncertainty_reduction_plan(
        self,
        top_k: int = 5
    ) -> List[Dict]:
        """
        不确定性降序计划

        找出最需要收集证据的信念（不确定性最高 + 影响最大）
        """
        plan = []

        for belief_id, belief in self._engine._beliefs.items():
            uncertainty = belief.posterior.std
            impact_factor = len(self._network.get_children(belief_id)) * 0.3 + 1.0
            evidence_strength = belief.posterior.effective_sample_size

            # 不确定性高 + 影响大 + 证据不足 → 高优先级
            priority = uncertainty * impact_factor / (math.log(evidence_strength + 1) + 1)

            plan.append({
                "belief_id": belief_id,
                "uncertainty": uncertainty,
                "confidence": belief.posterior.mean,
                "evidence_strength": evidence_strength,
                "impact_factor": impact_factor,
                "priority": priority,
                "suggestion": (
                    "急需更多证据" if priority > 0.3
                    else "建议补充证据" if priority > 0.15
                    else "证据相对充足"
                )
            })

        plan.sort(key=lambda x: x["priority"], reverse=True)
        return plan[:top_k]


# ============================================================
# 统一推理系统
# ============================================================

class BayesianReasoningSystem:
    """
    贝叶斯推理统一系统 - 顶层入口

    整合所有贝叶斯模块，提供统一的推理接口：

    1. BayesianEngine    — 概率信念管理
    2. BayesianNetwork   — 因果概率图模型
    3. EvidenceCollector — 证据收集与似然计算
    4. BayesianPredictor — 预测与校准
    5. BayesianAdvisor   — 决策建议

    典型使用流程：
    ```python
    brs = BayesianReasoningSystem()

    # 注册信念
    brs.register_belief("will_rain_tomorrow", prior=0.3)

    # 收集证据
    brs.collect_evidence("will_rain_tomorrow", positive=False, source="weather_app")

    # 建立因果网络
    brs.add_causal_link("humidity_high", "will_rain_tomorrow")

    # 预测
    prediction = brs.predict("will_rain_tomorrow")

    # 获取建议
    advice = brs.recommend("will_rain_tomorrow")
    ```
    """

    def __init__(
        self,
        name: str = "default",
        prior_type: str = "uniform",
        enable_network: bool = True,
        enable_predictor: bool = True,
        enable_advisor: bool = True,
    ):
        self.name = name

        # 核心引擎
        self.engine = BayesianEngine(default_prior_type=prior_type)
        self.tracker = BayesianBeliefTracker(prior_type=prior_type)
        self.tracker._engine = self.engine  # 共享引擎

        # 可选模块
        self.network = BayesianNetwork(name=name) if enable_network else None
        self.evidence_collector = EvidenceCollector(bayesian_engine=self.engine)
        self.predictor = (
            BayesianPredictor(self.engine, self.network, self.evidence_collector)
            if enable_predictor else None
        )
        self.advisor = (
            BayesianAdvisor(self.engine, self.network, self.predictor)
            if enable_advisor else None
        )

    # ---- 信念管理 ----

    def register_belief(
        self,
        belief_id: str,
        content: str = "",
        prior: float = None,
        category: str = "general",
        tags: List[str] = None
    ) -> Dict:
        """注册新信念"""
        belief = self.engine.register_belief(
            belief_id=belief_id,
            content_summary=content,
            prior_belief=prior,
            category=category,
            tags=tags or []
        )
        self.tracker.initialize(belief_id)
        return belief.to_dict()

    def get_belief(self, belief_id: str) -> Optional[Dict]:
        """获取信念详细信息"""
        belief = self.engine.get_belief(belief_id)
        return belief.to_dict() if belief else None

    # ---- 证据收集 ----

    def collect_evidence(
        self,
        belief_id: str,
        positive: bool = True,
        source: str = "user",
        source_type: str = "user_feedback",
        weight: float = 1.0,
        context: str = "",
    ) -> Dict:
        """收集证据并更新后验（含贝叶斯网络传播）"""
        record = self.evidence_collector.collect(
            belief_id=belief_id,
            is_positive=positive,
            source=source,
            source_type=source_type,
            weight=weight,
            context=context,
        )

        # 同步 tracker
        if positive:
            self.tracker.reinforce(belief_id, weight=weight)
        else:
            self.tracker.shake(belief_id)

        # 🔥 关键：通过贝叶斯网络传播证据
        # 将当前证据节点的状态传播给所有关联节点
        if self.network is not None:
            # 获取当前节点在网络中的邻居
            neighbors = self.network.get_neighbors(belief_id)
            if neighbors:
                # 使用信念传播推断所有邻居的后验
                evidence = {belief_id: positive}
                try:
                    posteriors = self.network.infer_posterior(
                        list(neighbors),
                        evidence
                    )
                    # 将传播结果写入引擎
                    for neighbor_id, posterior in posteriors.items():
                        neighbor_belief = self.engine.get_belief(neighbor_id)
                        if neighbor_belief:
                            # 融合：网络推断结果与现有信念取加权平均
                            existing_mean = neighbor_belief.posterior.mean
                            network_mean = posterior.mean
                            # 融合权重基于各自的证据强度
                            existing_w = neighbor_belief.posterior.effective_sample_size
                            network_w = posterior.effective_sample_size
                            total_w = existing_w + network_w
                            if total_w > 0:
                                fused_mean = (
                                    existing_mean * existing_w + network_mean * network_w
                                ) / total_w
                                # 更新信念（保持方向一致）
                                if fused_mean > existing_mean:
                                    self.engine.observe(
                                        neighbor_id, success=True,
                                        weight=network_w * 0.3,
                                        source=f"network_propagation_from_{belief_id}"
                                    )
                                elif fused_mean < existing_mean:
                                    self.engine.observe(
                                        neighbor_id, success=False,
                                        weight=network_w * 0.3,
                                        source=f"network_propagation_from_{belief_id}"
                                    )
                except Exception:
                    pass  # 网络推断失败时静默回退

        belief = self.engine.get_belief(belief_id)
        return {
            "evidence_id": record.evidence_id,
            "belief_id": belief_id,
            "posterior_mean": belief.posterior.mean if belief else None,
            "posterior_std": belief.posterior.std if belief else None,
            "stage": belief.get_stage() if belief else "unknown",
        }

    def verify_evidence(
        self,
        evidence_id: str,
        was_correct: bool
    ):
        """验证证据（反馈闭环）"""
        self.evidence_collector.verify_evidence(evidence_id, was_correct)

    # ---- 因果网络 ----

    def add_causal_link(
        self,
        cause_id: str,
        effect_id: str,
        initial_strength: float = 0.5,
        edge_type: str = "causal"
    ) -> Dict:
        """添加因果关系边"""
        if self.network is None:
            return {"error": "Network module not enabled"}

        try:
            edge = self.network.add_edge(
                parent_id=cause_id,
                child_id=effect_id,
                edge_type=edge_type,
                initial_strength=initial_strength
            )
            return {
                "status": "created",
                "parent": cause_id,
                "child": effect_id,
                "edge_type": edge_type,
            }
        except ValueError as e:
            return {"error": str(e)}

    def observe_causal(
        self,
        cause_id: str,
        effect_id: str,
        cause_state: bool,
        effect_state: bool,
        weight: float = 1.0
    ) -> Dict:
        """观测因果关系证据"""
        if self.network is None:
            return {"error": "Network module not enabled"}

        self.network.observe(cause_id, effect_id, cause_state, effect_state, weight)
        strength = self.network.query_causal_strength(cause_id, effect_id)
        return strength or {}

    # ---- 推理查询 ----

    def query(self, belief_id: str) -> Dict:
        """
        查询信念的完整贝叶斯状态

        Returns:
            {
                "confidence": float,        # 后验期望
                "uncertainty": float,       # 后验标准差
                "stage": str,               # 信念阶段
                "credible_interval_95": tuple,
                "evidence_strength": float,  # 有效样本量
                "hypothesis_test": dict,     # 假设检验结果
            }
        """
        belief = self.engine.get_belief(belief_id)
        if not belief:
            return {"error": f"Belief '{belief_id}' not found"}

        return {
            "belief_id": belief_id,
            "content": belief.content_summary,
            "confidence": belief.posterior.mean,
            "uncertainty": belief.posterior.std,
            "stage": belief.get_stage(),
            "credible_interval_95": belief.posterior.credible_interval(0.95),
            "evidence_strength": belief.posterior.effective_sample_size,
            "positive_evidence": belief.positive_evidence,
            "negative_evidence": belief.negative_evidence,
            "hypothesis_test": self.engine.hypothesis_test(belief_id),
        }

    def predict(
        self,
        event_id: str,
        context: Dict[str, bool] = None
    ) -> Dict:
        """贝叶斯预测"""
        if self.predictor is None:
            return {"error": "Predictor module not enabled"}
        return self.predictor.predict_event_probability(
            event_id, context_evidence=context
        )

    def record_outcome(
        self,
        event_id: str,
        predicted_prob: float,
        actual_outcome: bool
    ):
        """记录预测结果以校准"""
        if self.predictor:
            self.predictor.record_prediction_outcome(
                event_id, predicted_prob, actual_outcome
            )

    def recommend(self, belief_id: str, actions: List[Dict] = None) -> List[Dict]:
        """获取行动建议"""
        if self.advisor is None:
            return [{"error": "Advisor module not enabled"}]
        return self.advisor.recommend_actions(belief_id, actions)

    def value_of_info(
        self,
        target_id: str,
        sources: List[str]
    ) -> List[Dict]:
        """计算信息价值"""
        if self.advisor is None:
            return [{"error": "Advisor module not enabled"}]
        return self.advisor.value_of_information(target_id, sources)

    # ---- 诊断与报告 ----

    def get_diagnostic_report(self) -> Dict:
        """获取完整诊断报告"""
        return {
            "system": {
                "name": self.name,
                "total_beliefs": len(self.engine._beliefs),
                "total_evidence": self.evidence_collector._total_collected,
            },
            "engine": self.engine.get_statistics(),
            "network": self.network.get_statistics() if self.network else None,
            "evidence": self.evidence_collector.get_statistics(),
            "calibration": (
                self.predictor.get_calibration_report() if self.predictor else None
            ),
            "top_beliefs": self.engine.get_top_beliefs(5),
            "most_uncertain": self.engine.get_uncertain_beliefs(5),
            "source_rankings": self.evidence_collector.get_source_rankings(),
            "uncertainty_plan": (
                self.advisor.uncertainty_reduction_plan(5) if self.advisor else []
            ),
        }

    def get_accuracy_metrics(self) -> Dict:
        """
        获取准确度度量指标

        Returns:
            {
                "brier_score": float,         # Brier 评分（越低越好）
                "calibration_error": float,    # 校准误差
                "sharpness": float,            # 锐度（平均置信度）
                "resolution": float,           # 分辨能力
                "mean_confidence": float,
                "mean_uncertainty": float,
            }
        """
        beliefs = list(self.engine._beliefs.values())
        if not beliefs:
            return {"error": "No beliefs registered"}

        confidences = [b.posterior.mean for b in beliefs]
        uncertainties = [b.posterior.std for b in beliefs]
        n_effs = [b.posterior.effective_sample_size for b in beliefs]

        return {
            "mean_confidence": sum(confidences) / len(confidences),
            "mean_uncertainty": sum(uncertainties) / len(uncertainties),
            "mean_evidence_strength": sum(n_effs) / len(n_effs),
            "sharpness": sum(c * (1 - c) for c in confidences) / len(confidences),
            "high_confidence_ratio": sum(1 for c in confidences if c >= 0.8) / len(confidences),
            "calibration": (
                self.predictor.get_calibration_report() if self.predictor
                else {"status": "predictor_disabled"}
            ),
        }

    # ---- 持久化 ----

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "engine": self.engine.to_dict(),
            "network": self.network.to_dict() if self.network else {},
            "evidence": self.evidence_collector.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: Dict) -> 'BayesianReasoningSystem':
        system = cls(
            name=d.get("name", "default"),
            enable_network=bool(d.get("network")),
            enable_predictor=True,
            enable_advisor=True,
        )
        system.engine = BayesianEngine.from_dict(d.get("engine", {}))
        system.tracker._engine = system.engine
        system.evidence_collector = EvidenceCollector.from_dict(d.get("evidence", {}))
        if system.network and d.get("network"):
            system.network = BayesianNetwork.from_dict(d["network"])
        return system

    @classmethod
    def from_json(cls, json_str: str) -> 'BayesianReasoningSystem':
        return cls.from_dict(json.loads(json_str))

    def reset(self):
        """重置整个系统"""
        self.engine.reset()
        self.tracker = BayesianBeliefTracker()
        self.tracker._engine = self.engine
        self.evidence_collector.reset()
        if self.network:
            self.network.reset()
        if self.predictor:
            self.predictor._prediction_history.clear()
            self.predictor._calibration_curve.clear()
