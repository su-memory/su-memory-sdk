"""
证据收集与似然函数计算机制

提供：
1. 多源证据收集框架
2. 证据可信度评估（来源可靠性）
3. 加权似然函数计算
4. 证据冲突检测
5. 自适应证据权重调整

对外暴露：EvidenceCollector
"""

from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import math
import time
import json

from .bayesian import BetaDistribution, BayesianEngine, LikelihoodFunctions


# ============================================================
# 数据结构
# ============================================================

@dataclass
class EvidenceRecord:
    """单条证据记录"""
    evidence_id: str
    belief_id: str              # 关联的信念ID
    source: str                 # 证据来源
    source_type: str            # "user_feedback" | "model_output" | "sensor" | "inference" | "external"
    is_positive: bool           # 正面/负面证据
    raw_weight: float = 1.0    # 原始权重
    calibrated_weight: float = 1.0  # 校准后权重（考虑来源可靠性）
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)
    context: str = ""           # 上下文描述


@dataclass
class SourceProfile:
    """
    证据来源画像

    每个来源维护自身的可靠性 Beta 分布：
    - reliability = P(证据正确 | 来源 = source)
    """
    source_id: str
    source_type: str
    reliability: BetaDistribution = field(default_factory=BetaDistribution.uniform)
    total_evidence: int = 0
    verified_evidence: int = 0     # 被验证为正确的证据数
    contradicted_evidence: int = 0  # 被验证为错误的证据数
    last_active: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

    @property
    def reliability_score(self) -> float:
        """来源可靠性分数"""
        return self.reliability.mean

    def update_reliability(self, was_correct: bool):
        """更新来源可靠性"""
        self.total_evidence += 1
        if was_correct:
            self.verified_evidence += 1
            self.reliability.alpha += 0.5
        else:
            self.contradicted_evidence += 1
            self.reliability.beta += 0.5


# ============================================================
# 证据收集器
# ============================================================

class EvidenceCollector:
    """
    证据收集器 - 对外唯一接口

    功能：
    1. 多源证据收集
    2. 来源可靠性评估
    3. 证据权重校准
    4. 似然函数计算
    5. 证据冲突检测
    6. 自适应权重调整
    """

    def __init__(
        self,
        bayesian_engine: BayesianEngine = None,
        max_history: int = 10000,
        default_source_reliability: float = 0.7,
    ):
        """
        Args:
            bayesian_engine: 关联的贝叶斯引擎
            max_history: 最大证据历史记录数
            default_source_reliability: 默认来源可靠性
        """
        self._engine = bayesian_engine or BayesianEngine()
        self._max_history = max_history
        self._default_reliability = default_source_reliability

        # 证据存储
        self._evidence_history: List[EvidenceRecord] = []
        self._evidence_index: Dict[str, List[int]] = defaultdict(list)  # belief_id → indices
        self._source_profiles: Dict[str, SourceProfile] = {}

        # 统计
        self._total_collected = 0
        self._conflicts_detected = 0

    # ---- 来源管理 ----

    def register_source(
        self,
        source_id: str,
        source_type: str = "unknown",
        initial_reliability: float = None,
    ) -> SourceProfile:
        """
        注册证据来源

        Args:
            source_id: 来源唯一标识
            source_type: 来源类型
            initial_reliability: 初始可靠性

        Returns:
            SourceProfile
        """
        if source_id in self._source_profiles:
            return self._source_profiles[source_id]

        rel = initial_reliability or self._default_reliability
        profile = SourceProfile(
            source_id=source_id,
            source_type=source_type,
            reliability=BetaDistribution.weak_informative(rel, strength=5.0)
        )
        self._source_profiles[source_id] = profile
        return profile

    def get_source_reliability(self, source_id: str) -> float:
        """获取来源可靠性"""
        profile = self._source_profiles.get(source_id)
        return profile.reliability_score if profile else self._default_reliability

    # ---- 证据收集 ----

    def collect(
        self,
        belief_id: str,
        is_positive: bool,
        source: str = "unknown",
        source_type: str = "unknown",
        weight: float = 1.0,
        context: str = "",
        metadata: Dict = None,
    ) -> EvidenceRecord:
        """
        收集一条证据

        流程：
        1. 记录原始证据
        2. 获取来源可靠性
        3. 校准权重 = 原始权重 × 来源可靠性
        4. 提交到 BayesianEngine 更新后验

        Args:
            belief_id: 关联信念ID
            is_positive: 是否为正面证据
            source: 证据来源
            source_type: 来源类型
            weight: 原始权重
            context: 上下文
            metadata: 元信息

        Returns:
            EvidenceRecord
        """
        # 确保来源已注册
        self.register_source(source, source_type)

        # 校准权重
        reliability = self.get_source_reliability(source)
        calibrated_weight = weight * reliability

        now = time.time()
        record = EvidenceRecord(
            evidence_id=f"ev_{now}_{self._total_collected}",
            belief_id=belief_id,
            source=source,
            source_type=source_type,
            is_positive=is_positive,
            raw_weight=weight,
            calibrated_weight=calibrated_weight,
            timestamp=now,
            metadata=metadata or {},
            context=context,
        )

        # 存储
        self._evidence_history.append(record)
        self._evidence_index[belief_id].append(len(self._evidence_history) - 1)
        self._total_collected += 1

        # 限制历史长度
        if len(self._evidence_history) > self._max_history:
            self._prune_history()

        # 提交到贝叶斯引擎
        self._engine.observe(
            belief_id=belief_id,
            success=is_positive,
            weight=calibrated_weight,
            source=source,
            note=context
        )

        return record

    def collect_batch(
        self,
        evidence_list: List[Dict]
    ) -> List[EvidenceRecord]:
        """批量收集证据"""
        records = []
        for ev in evidence_list:
            record = self.collect(
                belief_id=ev["belief_id"],
                is_positive=ev.get("is_positive", True),
                source=ev.get("source", "batch"),
                source_type=ev.get("source_type", "unknown"),
                weight=ev.get("weight", 1.0),
                context=ev.get("context", ""),
                metadata=ev.get("metadata")
            )
            records.append(record)
        return records

    # ---- 证据验证 ----

    def verify_evidence(
        self,
        evidence_id: str,
        was_correct: bool,
        ground_truth_source: str = "ground_truth"
    ):
        """
        验证证据是否正确（使用真实结果反馈）

        Args:
            evidence_id: 被验证的证据ID
            was_correct: 证据是否正确
            ground_truth_source: 验证来源
        """
        # 找到证据记录
        record = None
        for ev in self._evidence_history:
            if ev.evidence_id == evidence_id:
                record = ev
                break

        if not record:
            return

        # 更新来源可靠性
        profile = self._source_profiles.get(record.source)
        if profile:
            profile.update_reliability(was_correct)
            profile.last_active = time.time()

        # 如果不正确，标记为冲突
        if not was_correct:
            self._conflicts_detected += 1

            # 用 ground truth 更新信念（反方向修正）
            self._engine.observe(
                belief_id=record.belief_id,
                success=record.is_positive,  # ground truth 认为是正面？
                weight=1.5,  # ground truth 权重更高
                source=ground_truth_source,
                note=f"Verification of {evidence_id}"
            )

    # ---- 似然计算 ----

    def compute_likelihood(
        self,
        belief_id: str,
        hypothesis_value: float = None,
        time_window: float = None
    ) -> float:
        """
        计算给定信念的似然函数 P(Evidence | belief = hypothesis_value)

        Args:
            belief_id: 信念ID
            hypothesis_value: 假设的概率值（None=使用后验期望）
            time_window: 时间窗口（秒），None=使用全部证据

        Returns:
            对数似然值
        """
        belief = self._engine.get_belief(belief_id)
        if not belief:
            return 0.0

        h = hypothesis_value or belief.posterior.mean

        # 收集相关证据
        evidence_list = self.get_evidence_for_belief(belief_id, time_window)
        if not evidence_list:
            return 0.0

        return LikelihoodFunctions.weighted_likelihood(evidence_list, h)

    def compute_evidence_strength(
        self,
        belief_id: str,
        time_window: float = None
    ) -> Dict[str, float]:
        """
        计算证据强度摘要

        Returns:
            {
                "positive_weight": float,
                "negative_weight": float,
                "total_weight": float,
                "evidence_count": int,
                "weighted_ratio": float  # 加权正面/总比例
            }
        """
        evidence_list = self.get_evidence_for_belief(belief_id, time_window)

        pos_weight = sum(
            e.calibrated_weight for e in evidence_list if e.is_positive
        )
        neg_weight = sum(
            e.calibrated_weight for e in evidence_list if not e.is_positive
        )
        total_weight = pos_weight + neg_weight

        return {
            "positive_weight": pos_weight,
            "negative_weight": neg_weight,
            "total_weight": total_weight,
            "evidence_count": len(evidence_list),
            "weighted_ratio": pos_weight / max(total_weight, 1e-10)
        }

    # ---- 冲突检测 ----

    def detect_evidence_conflicts(
        self,
        belief_id: str,
        threshold: float = 0.3
    ) -> List[Dict]:
        """
        检测证据冲突

        当正面和负面证据权重都较大时，存在冲突:
        min(pos_weight, neg_weight) / total_weight > threshold

        Args:
            belief_id: 信念ID
            threshold: 冲突阈值

        Returns:
            [{"type": "conflict", "severity": float, ...}]
        """
        conflicts = []
        strength = self.compute_evidence_strength(belief_id)

        total = strength["total_weight"]
        if total < 4.0:  # 证据太少不判定
            return conflicts

        pos_ratio = strength["weighted_ratio"]

        # 如果正面和负面证据都在 30%-70% 之间，存在冲突
        if threshold <= pos_ratio <= (1 - threshold):
            severity = 1.0 - abs(pos_ratio - 0.5) * 2  # 0.5 时最严重
            conflicts.append({
                "type": "evidence_conflict",
                "belief_id": belief_id,
                "severity": severity,
                "positive_weight": strength["positive_weight"],
                "negative_weight": strength["negative_weight"],
                "recommendation": "建议收集更多证据以消除不确定性" if severity > 0.7
                else "存在轻度冲突，持续观察"
            })

        return conflicts

    def detect_cross_belief_conflicts(self) -> List[Dict]:
        """
        检测跨信念冲突（基于矛盾证据）
        """
        conflicts = []

        belief_ids = list(self._evidence_index.keys())
        for i, bid_a in enumerate(belief_ids):
            for bid_b in belief_ids[i + 1:]:
                # 检查是否存在 A 的正面证据 = B 的负面证据的模式
                ev_a = self.get_evidence_for_belief(bid_a)
                ev_b = self.get_evidence_for_belief(bid_b)

                if not ev_a or not ev_b:
                    continue

                pos_a = sum(1 for e in ev_a if e.is_positive)
                neg_a = len(ev_a) - pos_a
                pos_b = sum(1 for e in ev_b if e.is_positive)
                neg_b = len(ev_b) - pos_b

                # A正面多但B负面多 → 可能存在对立关系
                if pos_a > neg_a and pos_b < neg_b:
                    conflicts.append({
                        "type": "cross_belief_conflict",
                        "belief_a": bid_a,
                        "belief_b": bid_b,
                        "a_pos_ratio": pos_a / max(len(ev_a), 1),
                        "b_pos_ratio": pos_b / max(len(ev_b), 1),
                        "recommendation": "检查两个信念是否存在对立关系"
                    })

        return conflicts

    # ---- 自适应权重 ----

    def calibrate_all_sources(self):
        """
        重新校准所有来源的可靠性

        基于历史验证结果更新 SourceProfile
        """
        for profile in self._source_profiles.values():
            # 长期未被验证 → 轻微衰减
            days_inactive = (time.time() - profile.last_active) / (24 * 3600)
            if days_inactive > 30 and profile.total_evidence > 10:
                decay = min(0.3, (days_inactive - 30) * 0.01)
                profile.reliability.alpha = max(1.0, profile.reliability.alpha - decay)
                profile.reliability.beta += decay * 0.5

    def get_source_rankings(self) -> List[Dict]:
        """获取来源可靠性排名"""
        rankings = []
        for profile in self._source_profiles.values():
            if profile.total_evidence >= 3:  # 至少3条证据
                rankings.append({
                    "source_id": profile.source_id,
                    "source_type": profile.source_type,
                    "reliability": profile.reliability_score,
                    "total_evidence": profile.total_evidence,
                    "accuracy": (
                        profile.verified_evidence / max(profile.total_evidence, 1)
                    )
                })

        rankings.sort(key=lambda x: x["reliability"], reverse=True)
        return rankings

    # ---- 查询辅助 ----

    def get_evidence_for_belief(
        self,
        belief_id: str,
        time_window: float = None
    ) -> List[EvidenceRecord]:
        """获取某个信念的所有证据"""
        indices = self._evidence_index.get(belief_id, [])
        records = []

        for idx in indices:
            if idx < len(self._evidence_history):
                record = self._evidence_history[idx]
                if time_window is None or \
                   (time.time() - record.timestamp) <= time_window:
                    records.append(record)

        return records

    def get_recent_evidence(self, n: int = 50) -> List[EvidenceRecord]:
        """获取最近 N 条证据"""
        return self._evidence_history[-n:] if self._evidence_history else []

    def get_evidence_summary(self, belief_id: str) -> Dict:
        """获取信念的证据摘要"""
        records = self.get_evidence_for_belief(belief_id)
        strength = self.compute_evidence_strength(belief_id)

        # 按来源统计
        source_stats = defaultdict(lambda: {"positive": 0, "negative": 0, "total": 0})
        for r in records:
            stats = source_stats[r.source]
            if r.is_positive:
                stats["positive"] += 1
            else:
                stats["negative"] += 1
            stats["total"] += 1

        return {
            "belief_id": belief_id,
            "total_evidence": len(records),
            "strength": strength,
            "sources": {
                src: {
                    "positive": s["positive"],
                    "negative": s["negative"],
                    "total": s["total"],
                    "reliability": self.get_source_reliability(src)
                }
                for src, s in source_stats.items()
            },
            "conflicts": self.detect_evidence_conflicts(belief_id)
        }

    def get_statistics(self) -> Dict:
        """获取收集器统计信息"""
        return {
            "total_collected": self._total_collected,
            "history_size": len(self._evidence_history),
            "registered_sources": len(self._source_profiles),
            "conflicts_detected": self._conflicts_detected,
            "engine_stats": self._engine.get_statistics(),
            "top_sources": self.get_source_rankings()[:5]
        }

    # ---- 内部方法 ----

    def _prune_history(self):
        """裁剪证据历史（保留最近的）"""
        keep_count = self._max_history // 2
        removed = self._evidence_history[:-keep_count]
        self._evidence_history = self._evidence_history[-keep_count:]

        # 重建索引
        self._evidence_index.clear()
        for i, record in enumerate(self._evidence_history):
            self._evidence_index[record.belief_id].append(i)

    # ---- 持久化 ----

    def to_dict(self) -> Dict:
        return {
            "evidence_count": len(self._evidence_history),
            "total_collected": self._total_collected,
            "conflicts_detected": self._conflicts_detected,
            "sources": {
                sid: {
                    "source_type": p.source_type,
                    "reliability": p.reliability.to_dict(),
                    "total_evidence": p.total_evidence,
                    "verified_evidence": p.verified_evidence,
                    "contradicted_evidence": p.contradicted_evidence,
                }
                for sid, p in self._source_profiles.items()
            },
            "engine": self._engine.to_dict()
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: Dict) -> 'EvidenceCollector':
        engine = BayesianEngine.from_dict(d.get("engine", {}))
        collector = cls(bayesian_engine=engine)
        collector._total_collected = d.get("total_collected", 0)
        collector._conflicts_detected = d.get("conflicts_detected", 0)

        for sid, sd in d.get("sources", {}).items():
            collector._source_profiles[sid] = SourceProfile(
                source_id=sid,
                source_type=sd["source_type"],
                reliability=BetaDistribution.from_dict(sd["reliability"]),
                total_evidence=sd.get("total_evidence", 0),
                verified_evidence=sd.get("verified_evidence", 0),
                contradicted_evidence=sd.get("contradicted_evidence", 0),
            )

        return collector

    @classmethod
    def from_json(cls, json_str: str) -> 'EvidenceCollector':
        return cls.from_dict(json.loads(json_str))

    def reset(self):
        """重置收集器"""
        self._evidence_history.clear()
        self._evidence_index.clear()
        self._source_profiles.clear()
        self._total_collected = 0
        self._conflicts_detected = 0
        self._engine.reset()
