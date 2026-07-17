"""
feedback_trainer — 临床反馈训练飞轮

将临床反馈（rating/action）自动转化为记忆置信度训练信号，
形成"推荐→反馈→置信度更新→下次排序优化"的闭环。

⚠️ 项目区隔：训练目标是检索排序权重，不是世界模型参数（后者由 World Model 负责）。

工作原理：
  高质量反馈（rating≥4 + accept/modify）→ ConfidenceTracker.record_positive()
  低质量反馈（reject / rating≤2）       → ConfidenceTracker.record_negative()
  → 下次 query() 时，高置信度记忆排名↑，低置信度排名↓

闭环：
  营养方案推荐 → 用户反馈(rating) → feedback_trainer.train_from_feedback()
      → ConfidenceTracker 更新 α/β → query() 自动重排序 → 下次推荐优化

Example:
  >>> from su_memory.clinical import ConfidenceTracker, FeedbackTrainer
  >>> tracker = ConfidenceTracker(client)
  >>> trainer = FeedbackTrainer(tracker)
  >>> trainer.train_from_feedback(memory_id="mem_001", rating=5, action="accept")
  >>> # mem_001 置信度↑，下次检索排名提升
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from su_memory.clinical.confidence import ConfidenceTracker

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 反馈训练器
# ═══════════════════════════════════════════════════════════════════


class FeedbackTrainer:
    """临床反馈训练器 — 反馈 → 置信度 → 检索排序优化。

    用法：
        tracker = ConfidenceTracker(client)
        tracker.inject_hooks(client)
        trainer = FeedbackTrainer(tracker)
        trainer.train_from_feedback("mem_001", rating=5, action="accept")
    """

    # 反馈阈值配置
    RATING_POSITIVE_THRESHOLD = 4   # rating ≥ 4 视为正反馈
    RATING_NEGATIVE_THRESHOLD = 2   # rating ≤ 2 视为负反馈
    # 反馈权重映射（rating → 训练权重）
    RATING_WEIGHTS = {1: 2.0, 2: 1.5, 3: 0.5, 4: 1.0, 5: 2.0}

    def __init__(self, tracker: ConfidenceTracker):
        self._tracker = tracker
        self._total_trained = 0
        self._positive_count = 0
        self._negative_count = 0

    def train_from_feedback(
        self,
        memory_id: str,
        rating: int,
        action: str = "accept",
        source: str = "",
    ) -> float:
        """根据临床反馈训练记忆置信度。

        Args:
            memory_id: 被反馈的记忆 ID
            rating: 评分 1-5
            action: 反馈动作 accept/modify/reject
            source: 反馈来源标识（如 feedback_id）

        Returns:
            更新后的置信度
        """
        weight = self.RATING_WEIGHTS.get(rating, 1.0)
        source_tag = source or f"feedback_r{rating}_{action}"

        # reject 无论 rating 多少都是负反馈
        if action == "reject":
            confidence = self._tracker.record_negative(
                memory_id, source=source_tag, weight=weight
            )
            self._negative_count += 1
            logger.debug(
                "[FeedbackTrainer] 负反馈训练 mem=%s rating=%d → conf=%.3f",
                memory_id[:12], rating, confidence,
            )
            self._total_trained += 1
            return confidence

        # rating ≥ 4 + accept/modify → 正反馈
        if rating >= self.RATING_POSITIVE_THRESHOLD:
            confidence = self._tracker.record_positive(
                memory_id, source=source_tag, weight=weight
            )
            self._positive_count += 1
            logger.debug(
                "[FeedbackTrainer] 正反馈训练 mem=%s rating=%d → conf=%.3f",
                memory_id[:12], rating, confidence,
            )
            self._total_trained += 1
            return confidence

        # rating ≤ 2 → 负反馈
        if rating <= self.RATING_NEGATIVE_THRESHOLD:
            confidence = self._tracker.record_negative(
                memory_id, source=source_tag, weight=weight
            )
            self._negative_count += 1
            self._total_trained += 1
            return confidence

        # rating = 3（中性）→ 轻微负反馈（不够好但不完全否定）
        confidence = self._tracker.record_negative(
            memory_id, source=source_tag, weight=0.5
        )
        self._total_trained += 1
        return confidence

    def train_batch(
        self, feedbacks: list[dict]
    ) -> dict[str, float]:
        """批量训练。

        Args:
            feedbacks: [{"memory_id", "rating", "action", "source"}, ...]

        Returns:
            {memory_id: confidence} 更新后的置信度
        """
        results: dict[str, float] = {}
        for fb in feedbacks:
            mid = fb.get("memory_id", "")
            if not mid:
                continue
            conf = self.train_from_feedback(
                memory_id=mid,
                rating=fb.get("rating", 3),
                action=fb.get("action", "accept"),
                source=fb.get("source", ""),
            )
            results[mid] = conf
        return results

    def get_training_stats(self) -> dict:
        """获取训练统计"""
        return {
            "total_trained": self._total_trained,
            "positive": self._positive_count,
            "negative": self._negative_count,
            "positive_ratio": (
                self._positive_count / max(self._total_trained, 1)
            ),
        }
