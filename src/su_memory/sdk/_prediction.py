"""
_prediction — 预测模块（lite_pro.py 拆分）

PredictionModule: 基于时序模式的预测。依赖 TemporalSystem。
从 lite_pro.py 拆分，对外通过 lite_pro.py 再导出保持兼容。
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from su_memory.sdk._temporal_system import TemporalSystem


class PredictionModule:
    """
    时序预测模块
    基于历史记忆模式预测未来事件和趋势

    V1.7.7: 支持贝叶斯置信度增强（通过 enable_bayesian=True 启用）
    """

    def __init__(self, temporal_system: TemporalSystem = None, enable_bayesian: bool = True):
        self._temporal = temporal_system or TemporalSystem()
        self._pattern_cache: dict[str, list[float]] = defaultdict(list)
        self._event_sequences: list[dict] = []

        # 贝叶斯增强
        self._enable_bayesian = enable_bayesian
        self._bayesian_engine = None
        self._prediction_feedback: dict[str, dict] = {}  # {pred_type: {"success": n, "failure": n}}

        if enable_bayesian:
            try:
                from su_memory._sys.bayesian import BayesianEngine
                self._bayesian_engine = BayesianEngine(default_prior_type="weak")
                # 为每种预测类型注册信念
                for pred_type in ["enhancement_prediction", "suppression_warning",
                                  "frequency_prediction", "trend_prediction",
                                  "historical_causal", "energy_enhancement"]:
                    self._bayesian_engine.register_belief(
                        pred_type,
                        content_summary=f"{pred_type} prediction accuracy",
                        prior_belief=0.65,
                        prior_strength=3.0
                    )
            except ImportError:
                self._enable_bayesian = False

    def _get_confidence(self, pred_type: str, fallback: float = 0.7) -> float:
        """获取贝叶斯置信度（若启用），否则回退到固定值"""
        if self._enable_bayesian and self._bayesian_engine:
            belief = self._bayesian_engine.get_belief(pred_type)
            if belief and belief.posterior.effective_sample_size > 3:
                return belief.posterior.mean
        return fallback

    def feedback(self, pred_type: str, was_correct: bool):
        """提供预测反馈，更新贝叶斯先验"""
        if not self._enable_bayesian or not self._bayesian_engine:
            return
        self._bayesian_engine.observe(pred_type, success=was_correct, weight=1.0, source="prediction_feedback")

    def record_event(self, content: str, timestamp: int = None, metadata: dict = None):
        """
        Record event for subsequent prediction

        Args:
            content: Event content
            timestamp: Event timestamp
            metadata: Event metadata
        """
        ts = timestamp or int(time.time())
        energy_type = self._temporal.infer_energy_from_content(content)
        self._event_sequences.append({
            "content": content,
            "timestamp": ts,
            "metadata": metadata or {},
            "energy_type": energy_type
        })

        # Update pattern cache
        self._pattern_cache[energy_type].append(ts)

    def predict_next_events(self, current_context: str, top_k: int = 3) -> list[dict[str, Any]]:
        """
        预测下一个可能的事件

        Args:
            current_context: 当前上下文
            top_k: 返回预测数量

        Returns:
            List of predicted events with confidence
        """
        # 1. Based on energy enhancement prediction
        current_energy = self._temporal.infer_energy_from_content(current_context)
        current_time = int(time.time())

        predictions = []

        enhanced = self._temporal.ENERGY_ENHANCE.get(current_energy, "earth")
        enhanced_events = [e for e in self._event_sequences
                       if e["energy_type"] == enhanced and e["timestamp"] > current_time - 86400 * 30]

        if enhanced_events:
            predictions.append({
                "type": "enhancement_prediction",
                "content": f"{enhanced} related events may occur",
                "confidence": self._get_confidence("enhancement_prediction", 0.75),
                "confidence_source": "bayesian" if self._enable_bayesian else "heuristic",
                "basis": f"Current {current_energy} enhances {enhanced}, historically {enhanced} events are frequent"
            })

        # 2. Based on temporal pattern prediction
        suppressed = self._temporal.ENERGY_SUPPRESS.get(current_energy, "earth")
        predictions.append({
            "type": "suppression_warning",
            "content": f"Pay attention to {suppressed} related matters",
            "confidence": self._get_confidence("suppression_warning", 0.65),
            "confidence_source": "bayesian" if self._enable_bayesian else "heuristic",
            "basis": f"Current {current_energy} may be affected by {suppressed}"
        })

        # 3. Based on historical frequency prediction
        energy_counts = defaultdict(int)
        for e in self._event_sequences[-100:]:  # Recent 100
            if e["timestamp"] > current_time - 86400 * 7:
                energy_counts[e["energy_type"]] += 1

        if energy_counts:
            most_common = max(energy_counts, key=energy_counts.get)
            if most_common != current_energy:
                predictions.append({
                    "type": "frequency_prediction",
                    "content": f"Recent {most_common} type events are high frequency",
                    "confidence": self._get_confidence("frequency_prediction", 0.70),
                    "confidence_source": "bayesian" if self._enable_bayesian else "heuristic",
                    "basis": f"In past 7 days, {most_common} events appeared {energy_counts[most_common]} times"
                })

        # 按置信度排序
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        return predictions[:top_k]

    def predict_temporal_trend(self, metric: str = "activity", days: int = 7) -> dict[str, Any]:
        """
        预测时间趋势

        Args:
            metric: Metric type (activity/recency/energy)
            days: 预测天数

        Returns:
            趋势预测结果
        """
        current_time = int(time.time())

        if metric == "activity":
            # 活动趋势预测
            recent_events = [e for e in self._event_sequences
                           if e["timestamp"] > current_time - 86400 * 7]
            prev_events = [e for e in self._event_sequences
                          if current_time - 86400 * 14 < e["timestamp"] <= current_time - 86400 * 7]

            recent_count = len(recent_events)
            prev_count = len(prev_events)

            if prev_count > 0:
                change_rate = (recent_count - prev_count) / prev_count
            else:
                change_rate = 0.0

            # 预测趋势
            if change_rate > 0.2:
                trend = "上升"
                confidence = self._get_confidence("trend_prediction", min(0.9, 0.6 + abs(change_rate)))
            elif change_rate < -0.2:
                trend = "下降"
                confidence = self._get_confidence("trend_prediction", min(0.9, 0.6 + abs(change_rate)))
            else:
                trend = "平稳"
                confidence = self._get_confidence("trend_prediction", 0.75)

            return {
                "metric": "activity",
                "trend": trend,
                "change_rate": change_rate,
                "confidence": confidence,
                "recent_count": recent_count,
                "prev_count": prev_count,
                "prediction": f"未来{days}天活动量预计{trend}"
            }

        elif metric == "energy_type":
            # Energy trend prediction
            energy_distribution = defaultdict(int)
            for e in self._event_sequences[-50:]:
                if e["timestamp"] > current_time - 86400 * 14:
                    energy_distribution[e["energy_type"]] += 1

            # Current time period energy
            current_time_code = self._temporal.get_time_code(current_time)
            current_energy = current_time_code["energy_type"]

            return {
                "metric": "energy_type",
                "current_energy": current_energy,
                "distribution": dict(energy_distribution),
                "prediction": f"Current energy {current_energy}, recommend focusing on {self._temporal.ENERGY_ENHANCE.get(current_energy)} related"
            }

        return {"error": "Unknown metric"}

    def get_causal_predictions(self, cause_content: str) -> list[dict[str, Any]]:
        """
        基于因果关系预测结果

        Args:
            cause_content: 原因内容

        Returns:
            可能的结果列表
        """
        cause_energy = self._temporal.infer_energy_from_content(cause_content)

        # Causal keyword detection
        causal_keywords = {
            "如果": ["就", "那么", "则", "会"],
            "因为": ["所以", "因此", "导致", "使得"],
            "当": ["就", "便", "则"],
        }

        results = []

        for cause_kw, effect_kws in causal_keywords.items():
            if cause_kw in cause_content:
                # Find historical causal pairs
                for event in self._event_sequences:
                    for effect_kw in effect_kws:
                        if effect_kw in event["content"]:
                            confidence = self._get_confidence("historical_causal", 0.70)
                            results.append({
                                "cause": cause_content,
                                "effect": event["content"],
                                "confidence": round(confidence, 3),
                                "type": "historical_causal",
                                "confidence_source": "bayesian" if self._enable_bayesian else "heuristic"
                            })

        # Based on energy enhancement prediction
        enhanced = self._temporal.ENERGY_ENHANCE.get(cause_energy)
        if enhanced:
            confidence = self._get_confidence("energy_enhancement", 0.60)
            results.append({
                "cause": cause_content,
                "effect": f"May trigger {enhanced} related events",
                "confidence": round(confidence, 3),
                "type": "energy_enhancement",
                "basis": f"{cause_energy} enhances {enhanced}",
                "confidence_source": "bayesian" if self._enable_bayesian else "heuristic"
            })

        return results[:3]


