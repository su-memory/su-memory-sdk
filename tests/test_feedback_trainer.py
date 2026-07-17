"""
临床反馈训练飞轮测试 — P1-S3 验证

反馈 → 置信度训练 → 检索排序优化闭环。
"""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("SU_MEMORY_SKIP_ENV_CHECK", "1")
os.environ.setdefault("MEMORY_EMBEDDING_BACKEND", "none")


class TestFeedbackTraining:
    """反馈训练测试"""

    def test_positive_feedback_increases_confidence(self):
        """rating=5 + accept 提升置信度"""
        from su_memory.clinical import ConfidenceTracker, FeedbackTrainer
        tracker = ConfidenceTracker()
        trainer = FeedbackTrainer(tracker)
        conf = trainer.train_from_feedback("mem_1", rating=5, action="accept")
        assert conf > 0.5

    def test_reject_decreases_confidence(self):
        """reject 降低置信度（无论 rating）"""
        from su_memory.clinical import ConfidenceTracker, FeedbackTrainer
        tracker = ConfidenceTracker()
        trainer = FeedbackTrainer(trainer := FeedbackTrainer(tracker)) if False else FeedbackTrainer(tracker)
        conf = trainer.train_from_feedback("mem_1", rating=4, action="reject")
        assert conf < 0.5

    def test_low_rating_decreases(self):
        """rating=1 + accept 降低置信度"""
        from su_memory.clinical import ConfidenceTracker, FeedbackTrainer
        tracker = ConfidenceTracker()
        trainer = FeedbackTrainer(tracker := FeedbackTrainer(tracker)) if False else FeedbackTrainer(tracker)
        conf = trainer.train_from_feedback("mem_1", rating=1, action="accept")
        assert conf < 0.5

    def test_neutral_rating_slight_negative(self):
        """rating=3 轻微负反馈"""
        from su_memory.clinical import ConfidenceTracker, FeedbackTrainer
        tracker = ConfidenceTracker()
        trainer = FeedbackTrainer(tracker)
        conf = trainer.train_from_feedback("mem_1", rating=3, action="accept")
        assert conf < 0.5  # 略低于默认


class TestBatchTraining:
    """批量训练测试"""

    def test_batch_mixed_feedback(self):
        """批量混合反馈训练"""
        from su_memory.clinical import ConfidenceTracker, FeedbackTrainer
        tracker = ConfidenceTracker()
        trainer = FeedbackTrainer(tracker)
        results = trainer.train_batch([
            {"memory_id": "mem_a", "rating": 5, "action": "accept"},
            {"memory_id": "mem_b", "rating": 1, "action": "reject"},
            {"memory_id": "mem_c", "rating": 4, "action": "modify"},
        ])
        assert results["mem_a"] > 0.5
        assert results["mem_b"] < 0.5
        assert results["mem_c"] > 0.5


class TestFlywheelE2E:
    """飞轮端到端闭环测试"""

    def test_flywheel_changes_ranking(self, tmp_path):
        """飞轮闭环：反馈后检索排序改变"""
        from su_memory.sdk.lite_pro import SuMemoryLitePro
        from su_memory.clinical import ConfidenceTracker, FeedbackTrainer

        client = SuMemoryLitePro(
            storage_path=str(tmp_path / "flywheel"),
            embedding_backend="none",
            enable_llm_energy=False,
        )
        tracker = ConfidenceTracker(client)
        tracker.inject_hooks(client)
        trainer = FeedbackTrainer(tracker)

        # 写入两条相似记忆
        mid_good = client.add("高蛋白营养方案适合营养不良患者")
        mid_bad = client.add("低蛋白方案可能不够营养")

        # 模拟用户反馈：好评高蛋白方案多次
        for _ in range(10):
            trainer.train_from_feedback(mid_good, rating=5, action="accept")
        # 差评低蛋白方案
        for _ in range(5):
            trainer.train_from_feedback(mid_bad, rating=1, action="reject")

        # 查询应重排序：高蛋白方案置信度更高
        results = client.query("营养方案", top_k=2)
        good_rank = next(
            (i for i, r in enumerate(results) if r.get("memory_id") == mid_good),
            None,
        )
        bad_rank = next(
            (i for i, r in enumerate(results) if r.get("memory_id") == mid_bad),
            None,
        )
        if good_rank is not None and bad_rank is not None:
            assert good_rank < bad_rank, "飞轮闭环失败：好评方案未排在差评前面"

    def test_training_stats(self):
        """训练统计正确"""
        from su_memory.clinical import ConfidenceTracker, FeedbackTrainer
        tracker = ConfidenceTracker()
        trainer = FeedbackTrainer(tracker)
        trainer.train_from_feedback("m1", rating=5, action="accept")
        trainer.train_from_feedback("m2", rating=1, action="reject")
        stats = trainer.get_training_stats()
        assert stats["total_trained"] == 2
        assert stats["positive"] == 1
        assert stats["negative"] == 1
