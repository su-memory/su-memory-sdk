"""
信念演化追踪系统

Phase 1: 完整的信念生命周期管理

对外暴露：BeliefTracker
内部实现：封装在su_core._internal中
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import time


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
