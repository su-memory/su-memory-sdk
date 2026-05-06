"""
信念演化追踪系统

Phase 1: 完整的信念生命周期管理
Phase 2: 贝叶斯后验概率更新引擎集成

对外暴露：BeliefTracker, BayesianBeliefTracker
内部实现：封装在su_core._internal中
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
import math

# 贝叶斯引擎导入
try:
    from .bayesian import BayesianEngine, BetaDistribution, BayesianBelief
except ImportError:
    BayesianEngine = None
    BetaDistribution = None
    BayesianBelief = None


class BeliefStage:
    """信念阶段"""
    COGNITION = "认知"        # 新记忆进入
    CONFIRM = "确认"         # 被反复引用
    REINFORCE = "强化"       # 成为坚定信念
    DECAY = "衰减"           # 久未引用
    SHAKE = "动摇"           # 被反驳
    RESHAPE = "重塑"         # 遗忘或更新


@dataclass
class BeliefState:
    """信念状态"""
    memory_id: str
    stage: str                      # 当前阶段
    confidence: float              # 置信度 0.0-1.0
    reinforce_count: int            # 被强化次数
    shake_count: int               # 被动摇次数
    last_reinforced: float         # 上次强化时间戳
    last_shaken: float             # 上次动摇时间戳
    created_at: float              # 创建时间
    transitions: List[str] = field(default_factory=list)  # 阶段转换历史


class BeliefTracker:
    """
    信念演化追踪器 - 对外唯一接口

    功能：
    1. 信念状态初始化
    2. 信念强化（被引用/确认）
    3. 信念动摇（被反驳/冲突）
    4. 阶段自动转换
    5. 生命周期查询

    对外隐藏：状态转换规则、阈值配置
    """

    # 阶段转换阈值
    REINFORCE_THRESHOLD = 3        # 强化3次 → 确认
    CONFIRM_CONFIDENCE = 0.7       # 置信度0.7 → 强化
    DECAY_DAYS = 30                # 30天未强化 → 衰减
    SHAKE_THRESHOLD = 2            # 动摇2次 + 置信度<0.5 → 重塑

    # 衰减配置
    DECAY_RATE_PER_DAY = 0.02       # 每天衰减2%
    MIN_CONFIDENCE = 0.1           # 最低置信度

    def __init__(self):
        # 信念状态存储
        self._beliefs: Dict[str, BeliefState] = {}

    def initialize(self, memory_id: str) -> BeliefState:
        """
        初始化新记忆的信念状态
        """
        now = time.time()

        state = BeliefState(
            memory_id=memory_id,
            stage=BeliefStage.COGNITION,
            confidence=0.5,
            reinforce_count=0,
            shake_count=0,
            last_reinforced=now,
            last_shaken=0,
            created_at=now,
            transitions=["认知"]
        )

        self._beliefs[memory_id] = state
        return state

    def reinforce(self, memory_id: str) -> BeliefState:
        """
        强化信念（记忆被引用/确认）

        Returns:
            更新后的信念状态
        """
        if memory_id not in self._beliefs:
            self.initialize(memory_id)

        state = self._beliefs[memory_id]
        now = time.time()

        # 更新强化计数
        state.reinforce_count += 1
        state.last_reinforced = now

        # 置信度提升（边际递减）
        confidence_boost = 0.1 / (1 + state.reinforce_count * 0.1)
        state.confidence = min(1.0, state.confidence + confidence_boost)

        # 阶段转换检查
        self._check_stage_transition(state, "reinforce")

        return state

    def shake(self, memory_id: str, conflict_with: str = None) -> BeliefState:
        """
        动摇信念（记忆被反驳/发现冲突）

        Args:
            memory_id: 被动摇的记忆ID
            conflict_with: 冲突的记忆ID
        """
        if memory_id not in self._beliefs:
            self.initialize(memory_id)

        state = self._beliefs[memory_id]
        now = time.time()

        # 更新动摇计数
        state.shake_count += 1
        state.last_shaken = now

        # 置信度下降
        state.confidence = max(0.0, state.confidence - 0.15)

        # 阶段转换检查
        self._check_stage_transition(state, "shake")

        return state

    def _check_stage_transition(self, state: BeliefState, trigger: str):
        """检查并执行阶段转换"""
        now = time.time()

        if trigger == "reinforce":
            if state.stage == BeliefStage.COGNITION and state.reinforce_count >= self.REINFORCE_THRESHOLD:
                state.stage = BeliefStage.CONFIRM
                state.transitions.append("确认")
            elif state.stage == BeliefStage.CONFIRM and state.confidence >= self.CONFIRM_CONFIDENCE:
                state.stage = BeliefStage.REINFORCE
                state.transitions.append("强化")

        elif trigger == "shake":
            if state.stage in [BeliefStage.REINFORCE, BeliefStage.CONFIRM]:
                if state.shake_count >= self.SHAKE_THRESHOLD and state.confidence < 0.5:
                    state.stage = BeliefStage.RESHAPE
                    state.transitions.append("重塑")
                elif state.confidence < 0.6:
                    state.stage = BeliefStage.SHAKE
                    state.transitions.append("动摇")

        # 检查衰减
        if state.stage in [BeliefStage.CONFIRM, BeliefStage.REINFORCE]:
            days_since_reinforce = (now - state.last_reinforced) / (24 * 3600)
            if days_since_reinforce > self.DECAY_DAYS:
                state.stage = BeliefStage.DECAY
                state.transitions.append("衰减")

    def apply_decay(self) -> List[str]:
        """
        全局衰减（定期调用）

        Returns:
            进入重塑阶段的记忆ID列表
        """
        now = time.time()
        reshaped = []

        for memory_id, state in self._beliefs.items():
            if state.stage not in [BeliefStage.DECAY, BeliefStage.SHAKE]:
                continue

            days_elapsed = (now - state.last_reinforced) / (24 * 3600)
            decay_amount = days_elapsed * self.DECAY_RATE_PER_DAY

            state.confidence = max(
                self.MIN_CONFIDENCE,
                state.confidence - decay_amount
            )

            # 置信度过低 → 进入重塑
            if state.confidence <= self.MIN_CONFIDENCE:
                state.stage = BeliefStage.RESHAPE
                reshaped.append(memory_id)

        return reshaped

    def get_state(self, memory_id: str) -> Optional[BeliefState]:
        """获取记忆的信念状态"""
        return self._beliefs.get(memory_id)

    def get_stage_distribution(self) -> Dict[str, int]:
        """获取信念阶段分布统计"""
        distribution = {}
        for state in self._beliefs.values():
            distribution[state.stage] = distribution.get(state.stage, 0) + 1
        return distribution

    def should_forget(self, memory_id: str) -> bool:
        """判断记忆是否应该被遗忘（进入死态）"""
        state = self._beliefs.get(memory_id)
        if not state:
            return False
        return state.stage == BeliefStage.RESHAPE and state.confidence <= self.MIN_CONFIDENCE


# ============================================================
# 贝叶斯信念追踪器 (Phase 2)
# ============================================================

class BayesianBeliefTracker:
    """
    贝叶斯信念追踪器

    基于 BayesianEngine 的后验概率更新算法，替代原始的 ad-hoc 置信度修正。

    核心改进：
    1. 使用 Beta 分布建模信念不确定性（而非点估计）
    2. 基于贝叶斯定理的证据更新（而非固定增量）
    3. 自然实现边际递减（更多证据 → 更小更新幅度）
    4. 提供完整的后验分布（均值、方差、置信区间）
    5. 支持贝叶斯因子比较

    API 兼容 BeliefTracker，可无缝替换。
    """

    # 阶段转换阈值
    REINFORCE_THRESHOLD = 3
    CONFIRM_CONFIDENCE = 0.7
    DECAY_DAYS = 30
    SHAKE_THRESHOLD = 2

    # 衰减配置
    DECAY_RATE_PER_DAY = 0.005  # 贝叶斯版本衰减更温和（每天0.5%）
    MIN_CONFIDENCE = 0.1

    def __init__(self, prior_type: str = "uniform"):
        """
        Args:
            prior_type: 先验类型 "uniform" | "jeffreys" | "weak"
        """
        self._engine = BayesianEngine(
            default_prior_type=prior_type,
            default_prior_strength=2.0
        ) if BayesianEngine else None
        self._transitions: Dict[str, List[str]] = {}  # memory_id → transition history
        self._initialized_at: Dict[str, float] = {}

    # ---- 状态管理（兼容 BeliefTracker）----

    def initialize(self, memory_id: str) -> 'BayesianBeliefState':
        """初始化新记忆的信念状态"""
        if self._engine is None:
            return self._fallback_initialize(memory_id)

        belief = self._engine.register_belief(memory_id)
        now = time.time()
        self._transitions[memory_id] = ["认知"]
        self._initialized_at[memory_id] = now

        return BayesianBeliefState(
            memory_id=memory_id,
            stage="认知",
            confidence=belief.posterior.mean,
            uncertainty=belief.posterior.std,
            reinforce_count=belief.positive_evidence,
            shake_count=belief.negative_evidence,
            last_reinforced=now,
            last_shaken=0,
            created_at=now,
            transitions=["认知"],
            alpha=belief.posterior.alpha,
            beta=belief.posterior.beta,
        )

    def _fallback_initialize(self, memory_id: str):
        """无贝叶斯引擎时的回退"""
        now = time.time()
        state = BayesianBeliefState(
            memory_id=memory_id,
            stage=BeliefStage.COGNITION,
            confidence=0.5,
            uncertainty=0.5,
            reinforce_count=0,
            shake_count=0,
            last_reinforced=now,
            last_shaken=0,
            created_at=now,
            transitions=["认知"],
            alpha=1.0,
            beta=1.0,
        )
        self._transitions[memory_id] = ["认知"]
        return state

    def reinforce(self, memory_id: str, weight: float = 1.0) -> 'BayesianBeliefState':
        """
        强化信念 — 使用贝叶斯后验更新

        Args:
            memory_id: 信念ID
            weight: 证据权重（1.0=标准, >1.0=强证据, <1.0=弱证据）

        Returns:
            更新后的信念状态
        """
        if self._engine is None:
            return self._fallback_reinforce(memory_id)

        belief = self._engine.observe(
            belief_id=memory_id,
            success=True,
            weight=weight,
            source="reinforce"
        )

        return self._build_state(belief, memory_id, trigger="reinforce")

    def _fallback_reinforce(self, memory_id: str):
        """无贝叶斯引擎时的回退"""
        if memory_id not in self._transitions:
            self._fallback_initialize(memory_id)
        # 使用简化逻辑
        return BayesianBeliefState(
            memory_id=memory_id,
            stage="认知",
            confidence=0.6,
            uncertainty=0.4,
            reinforce_count=1,
            shake_count=0,
            last_reinforced=time.time(),
            last_shaken=0,
            created_at=time.time(),
            transitions=self._transitions.get(memory_id, ["认知"]),
            alpha=2.0,
            beta=1.0,
        )

    def shake(self, memory_id: str, conflict_with: str = None) -> 'BayesianBeliefState':
        """
        动摇信念 — 使用贝叶斯后验更新

        Args:
            memory_id: 被动摇的记忆ID
            conflict_with: 冲突的记忆ID
        """
        if self._engine is None:
            return self._fallback_shake(memory_id, conflict_with)

        belief = self._engine.observe(
            belief_id=memory_id,
            success=False,
            weight=1.0,
            source="shake",
            note=f"conflict_with={conflict_with}" if conflict_with else ""
        )

        return self._build_state(belief, memory_id, trigger="shake")

    def _fallback_shake(self, memory_id: str, conflict_with: str = None):
        if memory_id not in self._transitions:
            self._fallback_initialize(memory_id)
        return BayesianBeliefState(
            memory_id=memory_id,
            stage="动摇",
            confidence=0.4,
            uncertainty=0.5,
            reinforce_count=0,
            shake_count=1,
            last_reinforced=0,
            last_shaken=time.time(),
            created_at=time.time(),
            transitions=self._transitions.get(memory_id, ["认知"]),
            alpha=1.0,
            beta=2.0,
        )

    def _build_state(
        self,
        belief: 'BayesianBelief',
        memory_id: str,
        trigger: str = None
    ) -> 'BayesianBeliefState':
        """从 BayesianBelief 构建 BayesianBeliefState"""
        # 阶段判定（基于后验统计量）
        stage = belief.get_stage()

        # 记录转换
        transitions = self._transitions.get(memory_id, ["认知"])
        if trigger == "reinforce" and stage not in ["cognition"]:
            last_stage = transitions[-1] if transitions else ""
            stage_map = {"confirm": "确认", "reinforce": "强化", "decay": "衰减"}
            mapped = stage_map.get(stage, "")
            if mapped and mapped != last_stage:
                transitions.append(mapped)
        elif trigger == "shake":
            last_stage = transitions[-1] if transitions else ""
            stage_map = {"shake": "动摇", "reshape": "重塑"}
            mapped = stage_map.get(stage, "")
            if mapped and mapped != last_stage:
                transitions.append(mapped)

        self._transitions[memory_id] = transitions

        return BayesianBeliefState(
            memory_id=memory_id,
            stage=stage,
            confidence=belief.posterior.mean,
            uncertainty=belief.posterior.std,
            reinforce_count=belief.positive_evidence,
            shake_count=belief.negative_evidence,
            last_reinforced=belief.last_updated,
            last_shaken=belief.last_updated if belief.negative_evidence > 0 else 0,
            created_at=belief.created_at,
            transitions=transitions,
            alpha=belief.posterior.alpha,
            beta=belief.posterior.beta,
            credible_interval_95=belief.posterior.credible_interval(0.95),
        )

    def apply_decay(self) -> List[str]:
        """
        全局衰减（定期调用）

        对超过 DECAY_DAYS 未更新的信念施加时间衰减

        Returns:
            进入重塑阶段的记忆ID列表
        """
        if self._engine is None:
            return []

        now = time.time()
        reshaped = []

        for belief_id, belief in self._engine._beliefs.items():
            days_elapsed = (now - belief.last_updated) / (24 * 3600)
            stage = belief.get_stage()

            # 只对确认/强化阶段的信念做衰减
            if stage not in ["confirm", "reinforce", "decay"]:
                continue

            if days_elapsed > self.DECAY_DAYS:
                # 衰减：降低有效证据权重（增加 beta 伪计数）
                decay_amount = (days_elapsed - self.DECAY_DAYS) * self.DECAY_RATE_PER_DAY
                belief.posterior.beta += decay_amount

                new_stage = belief.get_stage()
                if new_stage == "reshape":
                    reshaped.append(belief_id)

                # 记录转换
                transitions = self._transitions.get(belief_id, [])
                if "衰减" not in transitions:
                    transitions.append("衰减")
                self._transitions[belief_id] = transitions

        return reshaped

    def get_state(self, memory_id: str) -> Optional['BayesianBeliefState']:
        """获取记忆的信念状态"""
        if self._engine is None:
            return None

        belief = self._engine.get_belief(memory_id)
        if not belief:
            return None
        return self._build_state(belief, memory_id)

    def get_posterior(self, memory_id: str) -> Optional[Dict]:
        """
        获取完整的后验分布信息

        Returns:
            {"mean": float, "std": float, "ci_95": (lower, upper), "alpha": float, "beta": float}
        """
        if self._engine is None:
            return None
        belief = self._engine.get_belief(memory_id)
        if not belief:
            return None
        return belief.posterior.to_dict()

    def get_stage_distribution(self) -> Dict[str, int]:
        """获取信念阶段分布统计"""
        if self._engine is None:
            return {}
        return self._engine.get_stage_distribution()

    def should_forget(self, memory_id: str) -> bool:
        """判断记忆是否应该被遗忘"""
        if self._engine is None:
            return False
        belief = self._engine.get_belief(memory_id)
        if not belief:
            return False
        return belief.get_stage() == "reshape" and belief.posterior.mean <= self.MIN_CONFIDENCE

    # ---- 贝叶斯特有功能 ----

    def compare_beliefs(self, id_a: str, id_b: str) -> Optional[Dict]:
        """贝叶斯因子比较两个信念"""
        if self._engine is None:
            return None
        return self._engine.compare_beliefs(id_a, id_b)

    def hypothesis_test(self, memory_id: str, null_value: float = 0.5) -> Optional[Dict]:
        """贝叶斯假设检验"""
        if self._engine is None:
            return None
        return self._engine.hypothesis_test(memory_id, null_value)

    def get_top_beliefs(self, n: int = 10) -> List[Dict]:
        """获取置信度最高的信念"""
        if self._engine is None:
            return []
        return self._engine.get_top_beliefs(n)

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if self._engine is None:
            return {"error": "Bayesian engine not available"}
        return self._engine.get_statistics()


@dataclass
class BayesianBeliefState:
    """
    贝叶斯信念状态

    相比 BeliefState，增加了：
    - uncertainty: 后验标准差
    - alpha/beta: Beta 分布参数
    - credible_interval_95: 95% 置信区间
    """
    memory_id: str
    stage: str
    confidence: float          # 后验期望
    uncertainty: float = 0.5   # 后验标准差
    reinforce_count: int = 0
    shake_count: int = 0
    last_reinforced: float = 0
    last_shaken: float = 0
    created_at: float = 0
    transitions: List[str] = field(default_factory=list)

    # 贝叶斯特有
    alpha: float = 1.0   # Beta 分布的 α
    beta: float = 1.0    # Beta 分布的 β
    credible_interval_95: Tuple[float, float] = (0.0, 1.0)
