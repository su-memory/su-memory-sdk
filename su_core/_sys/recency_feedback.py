"""
时序感知 + 反馈调节系统

为 su-memory 增强时间维度召回能力：
1. 在 TemporalSystem 基础上增加双向反馈
2. 使用后 boost（positive feedback → 能量恢复）
3. 错误后 penalty（negative feedback → 能量衰减）
4. 与 ProgressiveDisclosure 联动

核心增强点：
- Hindsight 只有线性衰减（-0.02/天）和简单 boost（+0.1 on use）
- su-memory 有分段指数衰减 + 月令调制，扩展为反馈感知版本
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, List, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from su_memory._sys.chrono import TemporalSystem


# ========================
# 配置
# ========================

FEEDBACK_DECAY_PER_DAY = 0.02  # Hindsight 的固定衰减率
BOOST_ON_USE = 0.1  # Hindsight 的使用后 boost

# su-memory 扩展：连续反馈影响因子
CONTINUOUS_POSITIVE_THRESHOLD = 3  # 连续多少次正向才触发额外 boost
CONTINUOUS_NEGATIVE_PENALTY = 2   # 连续多少次负向才触发 penalty
POSITIVE_BOOST_MULTIPLIER = 1.15    # 连续正向后的额外乘数
NEGATIVE_PENALTY_MULTIPLIER = 0.85 # 连续负向后的额外乘数


# ========================
# 数据结构
# ========================

@dataclass
class FeedbackEvent:
    """单次反馈事件"""
    memory_id: str
    was_useful: bool
    timestamp: int  # Unix timestamp
    query_context: str = ""  # 触发反馈时的查询上下文


# ========================
# RecencyFeedbackSystem
# ========================

class RecencyFeedbackSystem:
    """
    反馈感知的时序系统

    在 TemporalSystem 基础上叠加用户反馈调节：

        base_score = temporal_system.calculate_time_decay(memory_timestamp, wuxing)
        feedback_modifier = feedback_system.get_feedback_modifier(memory_id)
        final_score = base_score * feedback_modifier

    使用方法：
        # 初始化时传入已有的 TemporalSystem
        feedback_sys = RecencyFeedbackSystem(temporal_system)

        # 记录反馈
        feedback_sys.record_feedback("mem-001", was_useful=True)

        # 计算最终得分
        score = feedback_sys.calculate_recency_score("mem-001", mem_timestamp, wuxing)
    """

    def __init__(
        self,
        temporal_system: "TemporalSystem",
        feedback_log_path: Optional[str] = None,
        max_history: int = 20,
    ):
        self._temporal = temporal_system
        self._feedback_log_path = feedback_log_path
        self._max_history = max_history

        # memory_id → [FeedbackEvent]
        self._feedback_log: Dict[str, List[FeedbackEvent]] = {}
        self._log_lock = threading.RLock()

    def record_feedback(
        self,
        memory_id: str,
        was_useful: bool,
        timestamp: Optional[int] = None,
        query_context: str = "",
    ) -> None:
        """
        记录用户对某记忆的反馈

        Args:
            memory_id: 记忆 ID
            was_useful: True = positive, False = negative
            timestamp: 反馈时间（默认当前）
            query_context: 触发反馈的查询（用于分析）
        """
        ts = timestamp or int(time.time())
        event = FeedbackEvent(
            memory_id=memory_id,
            was_useful=was_useful,
            timestamp=ts,
            query_context=query_context[:200],
        )

        with self._log_lock:
            self._feedback_log.setdefault(memory_id, [])
            self._feedback_log[memory_id].append(event)

            # 限制历史长度
            if len(self._feedback_log[memory_id]) > self._max_history:
                self._feedback_log[memory_id] = (
                    self._feedback_log[memory_id][-self._max_history:]
                )

        # 同步到文件
        self._persist_feedback(memory_id, event)

    def get_feedback_modifier(self, memory_id: str) -> float:
        """
        获取某记忆的反馈调节因子

        Returns:
            > 1.0: positive feedback boost
            < 1.0: negative feedback penalty
            = 1.0: no feedback recorded
        """
        events = self._get_recent_events(memory_id, limit=5)
        if not events:
            return 1.0

        positive_count = sum(1 for e in events if e.was_useful)
        negative_count = len(events) - positive_count
        total = len(events)

        # 全部正向
        if positive_count == total:
            if total >= CONTINUOUS_POSITIVE_THRESHOLD:
                return POSITIVE_BOOST_MULTIPLIER  # 1.15
            return 1.0 + (BOOST_ON_USE * positive_count / total)  # ~1.0-1.1

        # 全部负向
        if negative_count == total:
            if total >= CONTINUOUS_NEGATIVE_PENALTY:
                return NEGATIVE_PENALTY_MULTIPLIER  # 0.85
            return 1.0 - (FEEDBACK_DECAY_PER_DAY * negative_count / total)

        # 混合：按比例线性调整
        ratio = positive_count / total
        modifier = 0.85 + (0.30 * ratio)  # 0.85 ~ 1.15
        return modifier

    def calculate_recency_score(
        self,
        memory_id: str,
        memory_timestamp: int,
        memory_wuxing: str,
        query_timestamp: Optional[int] = None,
    ) -> float:
        """
        综合计算时效性得分

        = TemporalSystem 基础衰减 × 反馈调节因子

        Args:
            memory_id: 记忆 ID（用于查反馈历史）
            memory_timestamp: 记忆创建时间
            memory_wuxing: 记忆的五行属性（影响衰减速度）
            query_timestamp: 查询时间（默认当前）

        Returns:
            0.0 ~ 1.0 范围的时效性得分
        """
        q_ts = query_timestamp or int(time.time())

        # 1. TemporalSystem 基础衰减
        base_decay = self._temporal.calculate_time_decay(memory_timestamp, memory_wuxing)

        # 2. 近期性 boost（记忆越新 boost 越高，防止新记忆被旧记忆压制）
        recency_boost = self._temporal.calculate_recency_boost(
            memory_timestamp, q_ts, memory_wuxing
        )

        # 3. 反馈调节
        feedback_modifier = self.get_feedback_modifier(memory_id)

        # 综合
        final_score = base_decay * recency_boost * feedback_modifier
        return max(0.0, min(1.0, final_score))  # clamp to [0, 1]

    def get_feedback_trend(self, memory_id: str) -> str:
        """
        获取某记忆的反馈趋势

        Returns:
            "improving": 连续正向
            "declining": 连续负向
            "stable": 无明显趋势
            "mixed": 正负交替
        """
        events = self._get_recent_events(memory_id, limit=10)
        if len(events) < 2:
            return "stable"

        recent = [e.was_useful for e in events[-5:]]

        # 检查连续性
        if all(recent):
            return "improving"
        if not any(recent):
            return "declining"

        # 检查趋势（前半段 vs 后半段）
        mid = len(recent) // 2
        first_half = sum(1 for v in recent[:mid] if v)
        second_half = sum(1 for v in recent[mid:] if v)

        if second_half > first_half:
            return "improving"
        elif second_half < first_half:
            return "declining"
        return "mixed"

    def get_memory_summary(self, memory_id: str) -> Dict:
        """获取某记忆的时序+反馈综合摘要"""
        events = self._get_recent_events(memory_id, limit=10)
        positive = sum(1 for e in events if e.was_useful)
        negative = len(events) - positive

        return {
            "memory_id": memory_id,
            "feedback_count": len(events),
            "positive_count": positive,
            "negative_count": negative,
            "trend": self.get_feedback_trend(memory_id),
            "modifier": self.get_feedback_modifier(memory_id),
            "last_feedback": (
                events[-1].timestamp if events else None
            ),
        }

    def _get_recent_events(
        self,
        memory_id: str,
        limit: int = 5,
    ) -> List[FeedbackEvent]:
        """获取最近的反馈事件"""
        with self._log_lock:
            events = self._feedback_log.get(memory_id, [])
            return events[-limit:] if events else []

    def _persist_feedback(self, memory_id: str, event: FeedbackEvent) -> None:
        """持久化反馈到文件（append 模式）"""
        if not self._feedback_log_path:
            return
        try:
            from pathlib import Path
            path = Path(self._feedback_log_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                entry = {
                    "type": "feedback.recorded",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.{}Z".format(
                        int(time.time() * 1000) % 1000)),
                    "memory_id": memory_id,
                    "was_useful": event.was_useful,
                    "query_context": event.query_context,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def reset_memory_feedback(self, memory_id: str) -> None:
        """清除某记忆的反馈历史"""
        with self._log_lock:
            if memory_id in self._feedback_log:
                del self._feedback_log[memory_id]

    @property
    def memory_count(self) -> int:
        return len(self._feedback_log)

    def get_all_trends(self) -> Dict[str, str]:
        """批量获取所有有反馈的记忆的趋势"""
        return {
            mem_id: self.get_feedback_trend(mem_id)
            for mem_id, events in self._feedback_log.items()
            if events
        }


import json
