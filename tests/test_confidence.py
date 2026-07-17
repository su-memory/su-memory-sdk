"""
记忆置信度增强测试 — P1-S2 验证

贝叶斯 Beta-Binomial 模型评估记忆可靠性 + 检索重排序。
"""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("SU_MEMORY_SKIP_ENV_CHECK", "1")
os.environ.setdefault("MEMORY_EMBEDDING_BACKEND", "none")


class TestConfidenceModel:
    """贝叶斯置信度模型测试"""

    def test_new_memory_default_confidence(self):
        """新记忆默认置信度 = 0.5"""
        from su_memory.clinical.confidence import ConfidenceTracker
        tracker = ConfidenceTracker()
        assert tracker.get_confidence("mem_new") == 0.5

    def test_positive_feedback_increases(self):
        """正反馈提高置信度"""
        from su_memory.clinical.confidence import ConfidenceTracker
        tracker = ConfidenceTracker()
        tracker.record_positive("mem_1")
        assert tracker.get_confidence("mem_1") > 0.5

    def test_negative_feedback_decreases(self):
        """负反馈降低置信度"""
        from su_memory.clinical.confidence import ConfidenceTracker
        tracker = ConfidenceTracker()
        tracker.record_negative("mem_1")
        assert tracker.get_confidence("mem_1") < 0.5

    def test_multiple_positives_converge_high(self):
        """多次正反馈趋近高置信度"""
        from su_memory.clinical.confidence import ConfidenceTracker
        tracker = ConfidenceTracker()
        for _ in range(10):
            tracker.record_positive("mem_1")
        assert tracker.get_confidence("mem_1") > 0.85

    def test_mixed_feedback(self):
        """正负混合反馈（7正3负）"""
        from su_memory.clinical.confidence import ConfidenceTracker
        tracker = ConfidenceTracker()
        for _ in range(7):
            tracker.record_positive("mem_1")
        for _ in range(3):
            tracker.record_negative("mem_1")
        # α=8, β=4 → 8/12 = 0.667
        assert 0.6 < tracker.get_confidence("mem_1") < 0.75


class TestReranking:
    """置信度重排序测试"""

    def test_rerank_high_confidence_first(self):
        """高置信度记忆排名靠前"""
        from su_memory.clinical.confidence import ConfidenceTracker
        tracker = ConfidenceTracker()

        # 给 mem_a 多次正反馈使其置信度远高于 mem_b
        for _ in range(20):
            tracker.record_positive("mem_a")

        results = [
            {"memory_id": "mem_b", "score": 0.6, "content": "B"},
            {"memory_id": "mem_a", "score": 0.5, "content": "A"},
        ]
        reranked = tracker.rerank_by_confidence(results, blend=0.5)

        # mem_a 有高置信度，应排前面
        assert reranked[0]["memory_id"] == "mem_a"

    def test_rerank_preserves_order_with_no_feedback(self):
        """无反馈时保持原始排序"""
        from su_memory.clinical.confidence import ConfidenceTracker
        tracker = ConfidenceTracker()
        results = [
            {"memory_id": "mem_a", "score": 0.9},
            {"memory_id": "mem_b", "score": 0.5},
        ]
        reranked = tracker.rerank_by_confidence(results, blend=0.3)
        assert reranked[0]["memory_id"] == "mem_a"  # 原始高分仍在前

    def test_rerank_adds_confidence_field(self):
        """重排序后结果包含 confidence 字段"""
        from su_memory.clinical.confidence import ConfidenceTracker
        tracker = ConfidenceTracker()
        tracker.record_positive("mem_a")
        results = [{"memory_id": "mem_a", "score": 0.8}]
        reranked = tracker.rerank_by_confidence(results)
        assert "confidence" in reranked[0]


class TestHookInjection:
    """钩子注入测试"""

    def test_query_auto_reranks(self, tmp_path):
        """query 注入钩子后自动重排序"""
        from su_memory.sdk.lite_pro import SuMemoryLitePro
        from su_memory.clinical.confidence import ConfidenceTracker

        client = SuMemoryLitePro(
            storage_path=str(tmp_path / "conf_test"),
            embedding_backend="none",
            enable_llm_energy=False,
        )
        tracker = ConfidenceTracker(client)
        tracker.inject_hooks(client)

        mid = client.add("华法林抗凝治疗")
        # 给这条记忆正反馈
        for _ in range(5):
            tracker.record_positive(mid)

        # query 应自动重排序
        results = client.query("华法林", top_k=3)
        assert len(results) > 0
        assert "confidence" in results[0]

    def test_stats(self):
        """统计信息正确"""
        from su_memory.clinical.confidence import ConfidenceTracker
        tracker = ConfidenceTracker()
        for _ in range(5):
            tracker.record_positive("mem_high")
        for _ in range(3):
            tracker.record_negative("mem_low")

        stats = tracker.get_stats()
        assert stats["total"] == 2
        assert stats["avg_confidence"] > 0
        assert stats["high_confidence"] >= 1

class TestPersistence:
    """P0-1 置信度持久化测试"""

    def test_save_load_roundtrip(self, tmp_path):
        """save → load 应完整恢复记录"""
        from su_memory.clinical.confidence import ConfidenceTracker

        path = str(tmp_path / "conf.json")
        t1 = ConfidenceTracker()
        t1.record_positive("mem_001", source="feedback")
        t1.record_positive("mem_001", source="feedback")
        t1.record_negative("mem_002", source="feedback")
        t1.save(path)

        t2 = ConfidenceTracker(persist_path=path)
        stats = t2.get_stats()
        assert stats["total"] == 2
        # mem_001 有两次正例，置信度应高于 mem_002
        assert t2.get_confidence("mem_001") > t2.get_confidence("mem_002")

    def test_autosave_on_record(self, tmp_path):
        """配置 persist_path 时 record_positive 自动落盘"""
        import os
        from su_memory.clinical.confidence import ConfidenceTracker

        path = str(tmp_path / "auto.json")
        t = ConfidenceTracker(persist_path=path)
        t.record_positive("mem_x", source="feedback")

        assert os.path.exists(path)

    def test_client_auto_derives_persist_path(self, tmp_path):
        """ClinicalMemoryClient 应从 storage_path 自动推导 confidence 持久化路径"""
        import os
        from su_memory.clinical import ClinicalMemoryClient

        storage = str(tmp_path / "mem.db")
        client = ClinicalMemoryClient(
            storage_path=storage,
            embedding_backend="none",
            enable_confidence=True,
        )
        client.add_patient_event("P001", "测试记忆", "plan")
        hits = client.recall("P001", "测试")
        assert len(hits) >= 1
        # train_from_feedback 触发置信度记录
        client.train_from_feedback(hits[0]["memory_id"], rating=5, action="accept")
        client.close()

        # storage_path 本身是目录（lite_pro 用作存储根），confidence.json 在其下
        assert os.path.exists(os.path.join(storage, "confidence.json"))
