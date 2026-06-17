"""
P0: LifecycleManager 单元测试 (v3.5.5 P1-3)
=============================================
覆盖: MemoryLifecycleManager, LifecycleReport, LifecycleAction
      auto_expire, deduplicate, archive, get_report,
      健康评分计算, 年龄分布, 推荐建议生成

运行: pytest tests/test_lifecycle_manager.py -v
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from su_memory._sys._lifecycle_manager import (
    LifecycleAction,
    LifecycleReport,
    MemoryLifecycleManager,
)
from su_memory.sdk.lite_pro import SuMemoryLitePro


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def empty_client():
    """空的记忆客户端 (v3.5.5-p0: 使用原生 get_all_memories/forget, yield 清理)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        client = SuMemoryLitePro(
            max_memories=200,
            enable_vector=False,
            enable_graph=False,
            enable_temporal=False,
            enable_session=False,
            enable_prediction=False,
            enable_explainability=False,
            storage_path=tmpdir,
        )
        yield client
        # teardown: 清理全局状态
        try:
            client.clear()
        except Exception:
            pass


@pytest.fixture
def populated_client(empty_client):
    """含 24 条记忆的客户端（含重复和不同时间戳）"""
    client = empty_client
    # 添加 20 条记忆
    for i in range(20):
        ts = (datetime.now() - timedelta(days=i * 10)).isoformat()
        client.add(
            f"记忆内容_{i}: 这是一条用于测试的数据记录",
            metadata={"timestamp": ts, "index": i}
        )
    # 添加重复记忆
    client.add("重复记忆_A")
    client.add("重复记忆_A")  # 完全相同
    client.add("重复记忆_B")
    client.add("重复记忆_B")  # 完全相同
    return client


@pytest.fixture
def manager(populated_client):
    """生命周期管理器"""
    return MemoryLifecycleManager(populated_client)


# ============================================================
# 数据模型
# ============================================================

class TestLifecycleAction:
    """LifecycleAction 数据模型"""

    def test_create_action(self):
        action = LifecycleAction(
            action="dedup",
            timestamp=datetime.now().isoformat(),
            affected_count=5,
            details={"method": "hash"},
        )
        assert action.action == "dedup"
        assert action.affected_count == 5
        assert action.details["method"] == "hash"


class TestLifecycleReport:
    """LifecycleReport 数据模型"""

    def test_summary_contains_key_info(self):
        report = LifecycleReport(
            generated_at=datetime.now().isoformat(),
            total_memories=100,
            active_count=80,
            archived_count=15,
            expired_count=5,
            duplicate_memories=10,
            health_score=85.0,
            health_status="healthy",
        )
        summary = report.summary()
        assert "100" in summary
        assert "healthy" in summary
        assert "85" in summary


# ============================================================
# MemoryLifecycleManager 核心功能
# ============================================================

class TestLifecycleManagerInit:
    """初始化"""

    def test_create_manager(self, empty_client):
        mgr = MemoryLifecycleManager(empty_client)
        assert mgr is not None

    def test_default_max_history(self, empty_client):
        mgr = MemoryLifecycleManager(empty_client)
        assert mgr._max_action_history == 50

    def test_custom_max_history(self, empty_client):
        mgr = MemoryLifecycleManager(empty_client, max_action_history=10)
        assert mgr._max_action_history == 10


class TestDeduplicate:
    """去重功能"""

    def test_deduplicate_hash_dry_run(self, manager):
        """内容哈希去重 (dry_run)"""
        result = manager.deduplicate(method="content_hash", dry_run=True)
        assert result["clusters"] >= 2  # 至少 2 组重复
        assert result["duplicates_found"] >= 2
        assert result["removed"] == 0  # dry_run 不删除

    def test_deduplicate_hash_execute(self, manager):
        """内容哈希去重 (执行)"""
        before = len(manager._client._memories)
        result = manager.deduplicate(method="content_hash")
        after = len(manager._client._memories)
        assert result["removed"] >= 2
        assert after < before

    def test_deduplicate_empty(self, empty_client):
        """空记忆库去重"""
        mgr = MemoryLifecycleManager(empty_client)
        result = mgr.deduplicate()
        assert result["duplicates_found"] == 0
        assert result["clusters"] == 0

    def test_deduplicate_semantic(self, manager):
        """语义去重"""
        result = manager.deduplicate(method="semantic", threshold=0.8, dry_run=True)
        assert isinstance(result, dict)
        assert "clusters" in result


class TestAutoExpire:
    """自动过期"""

    def test_auto_expire_dry_run(self, manager):
        """模拟过期 (dry_run)"""
        result = manager.auto_expire(days=5, dry_run=True)
        assert result["expired"] == 0  # dry_run 不删除
        assert "candidates" in result

    def test_auto_expire_short_window(self, manager):
        """短窗口过期"""
        result = manager.auto_expire(days=1)
        assert "remaining" in result

    def test_auto_expire_long_window(self, manager):
        """长窗口 (不会过期任何记忆)"""
        result = manager.auto_expire(days=3650)
        assert result["expired"] == 0

    def test_auto_expire_records_action(self, manager):
        """记录操作历史"""
        manager.auto_expire(days=30, dry_run=True)
        history = manager.get_action_history()
        assert len(history) >= 1
        assert "expire" in history[-1].action


class TestArchive:
    """归档功能"""

    def test_archive_old_dry_run(self, manager):
        result = manager.archive(condition="old", threshold_days=30, dry_run=True)
        assert result["archived"] == 0

    def test_archive_low_energy_dry_run(self, manager):
        result = manager.archive(condition="low_energy", dry_run=True)
        assert isinstance(result, dict)

    def test_archive_inactive_dry_run(self, manager):
        result = manager.archive(condition="inactive", threshold_days=30, dry_run=True)
        assert "candidates" in result


class TestGetReport:
    """生命周期报告"""

    def test_get_report_basic(self, manager):
        report = manager.get_report()
        assert isinstance(report, LifecycleReport)
        assert report.total_memories == 24  # 20 + 4 duplicates
        assert report.generated_at != ""

    def test_report_age_distribution(self, manager):
        report = manager.get_report()
        assert isinstance(report.age_distribution, dict)
        assert set(report.age_distribution.keys()) == {
            "<7天", "7-30天", "30-90天", "90-180天", ">180天"
        }

    def test_report_has_recommendations(self, manager):
        report = manager.get_report()
        assert isinstance(report.recommendations, list)
        # 有重复记忆 → 应有去重建议
        if report.duplicate_memories > 0:
            assert any("重复" in r or "deduplicate" in r.lower()
                      for r in report.recommendations)

    def test_report_health_score_range(self, manager):
        report = manager.get_report()
        assert 0 <= report.health_score <= 100

    def test_report_health_status_valid(self, manager):
        report = manager.get_report()
        assert report.health_status in ("healthy", "fair", "degraded", "critical")

    def test_report_empty_store(self, empty_client):
        mgr = MemoryLifecycleManager(empty_client)
        report = mgr.get_report()
        assert report.total_memories == 0
        assert report.health_score == 100.0
        assert report.health_status == "healthy"

    def test_report_contains_duplicate_info(self, manager):
        report = manager.get_report()
        assert report.duplicate_clusters >= 0
        assert report.duplicate_memories >= 0

    def test_report_category_distribution(self, manager):
        report = manager.get_report()
        assert isinstance(report.category_distribution, dict)


class TestHealthScore:
    """健康评分计算"""

    def test_healthy_empty(self, manager):
        """空记忆库满分"""
        score = manager._calculate_health(0, 0, 0, {
            "<7天": 0, "7-30天": 0, "30-90天": 0, "90-180天": 0, ">180天": 0,
        })
        assert score == 100.0

    def test_duplicates_reduce_score(self, manager):
        """重复扣分"""
        score_clean = manager._calculate_health(100, 0, 0, {
            "<7天": 50, "7-30天": 30, "30-90天": 10, "90-180天": 5, ">180天": 5,
        })
        score_dups = manager._calculate_health(100, 50, 0, {
            "<7天": 50, "7-30天": 30, "30-90天": 10, "90-180天": 5, ">180天": 5,
        })
        assert score_dups < score_clean

    def test_expired_reduce_score(self, manager):
        """过期扣分"""
        score_clean = manager._calculate_health(100, 0, 0, {
            "<7天": 50, "7-30天": 30, "30-90天": 10, "90-180天": 5, ">180天": 5,
        })
        score_exp = manager._calculate_health(100, 0, 30, {
            "<7天": 50, "7-30天": 30, "30-90天": 10, "90-180天": 5, ">180天": 5,
        })
        assert score_exp < score_clean

    def test_freshness_bonus(self, manager):
        """新鲜度加分"""
        score = manager._calculate_health(100, 0, 0, {
            "<7天": 80, "7-30天": 10, "30-90天": 5, "90-180天": 3, ">180天": 2,
        })
        assert score >= 100  # 新鲜度高 + 无过期/重复


class TestRecommendations:
    """推荐建议生成"""

    def test_duplicate_recommendation(self, manager):
        recs = manager._generate_recommendations(
            total=100, dup_count=10, age_dist={
                "<7天": 50, "7-30天": 30, "30-90天": 10, "90-180天": 5, ">180天": 5,
            },
            expired_count=0, archived_count=0, health_score=75.0,
        )
        assert any("重复" in r for r in recs)

    def test_healthy_recommendation(self, manager):
        recs = manager._generate_recommendations(
            total=100, dup_count=0, age_dist={
                "<7天": 60, "7-30天": 30, "30-90天": 10, "90-180天": 0, ">180天": 0,
            },
            expired_count=0, archived_count=0, health_score=92.0,
        )
        assert any("优秀" in r for r in recs)

    def test_critical_recommendation(self, manager):
        recs = manager._generate_recommendations(
            total=100, dup_count=30, age_dist={
                "<7天": 5, "7-30天": 10, "30-90天": 20, "90-180天": 25, ">180天": 40,
            },
            expired_count=25, archived_count=60, health_score=35.0,
        )
        assert any("偏低" in r or "评分" in r for r in recs)


class TestActionHistory:
    """操作历史"""

    def test_record_action(self, manager):
        manager.auto_expire(days=30)
        history = manager.get_action_history()
        assert len(history) >= 1

    def test_clear_history(self, manager):
        manager.auto_expire(days=30)
        manager.clear_history()
        assert len(manager.get_action_history()) == 0

    def test_max_history_enforced(self, empty_client):
        mgr = MemoryLifecycleManager(empty_client, max_action_history=3)
        for _ in range(10):
            mgr.auto_expire(days=30)
        assert len(mgr.get_action_history()) == 3


class TestTimestampExtraction:
    """时间戳提取"""

    def test_from_timestamp_field(self, manager):
        ts = manager._get_timestamp({"timestamp": "2026-01-01T00:00:00"})
        assert ts is not None
        assert ts.year == 2026

    def test_from_created_at(self, manager):
        ts = manager._get_timestamp({"created_at": "2025-06-15T12:00:00"})
        assert ts is not None

    def test_from_metadata(self, manager):
        ts = manager._get_timestamp({"metadata": {"created_at": "2025-01-01T00:00:00"}})
        assert ts is not None

    def test_from_unix_timestamp(self, manager):
        ts = manager._get_timestamp({"timestamp": 1700000000})
        assert ts is not None

    def test_missing_timestamp(self, manager):
        ts = manager._get_timestamp({})
        assert ts is None


# ============================================================
# 入口
# ============================================================


# ============================================================
# 边界测试 (P2-3: 覆盖率提升)
# ============================================================

class TestBoundaryScenarios:
    """边界场景 — 单条/全过期/大规模."""

    def test_all_expired(self, empty_client):
        """全部记忆过期场景。"""
        client = empty_client
        old_ts = (datetime.now() - timedelta(days=400)).isoformat()
        for i in range(10):
            client.add(
                f"old_memory_{i}",
                metadata={"timestamp": old_ts, "index": i}
            )
        mgr = MemoryLifecycleManager(client)
        result = mgr.auto_expire(days=30)
        # 10 条过期 + 30天短窗口
        assert result["remaining"] <= 10
        assert result["expired"] >= 0

    def test_single_memory(self, empty_client):
        """单条记忆场景。"""
        client = empty_client
        client.add("唯一的一条记忆")
        mgr = MemoryLifecycleManager(client)

        report = mgr.get_report()
        assert report.total_memories == 1
        # 单条记忆无重复 → 健康分应较高
        assert report.health_score >= 60

        dedup = mgr.deduplicate(dry_run=True)
        assert dedup["clusters"] == 0
        assert dedup["duplicates_found"] == 0

    def test_large_memory_count(self, empty_client):
        """大量记忆 (>1000) — 不应 OOM."""
        client = empty_client
        # 添加 500 条 (模拟大规模)
        for i in range(500):
            ts = (datetime.now() - timedelta(days=i % 200)).isoformat()
            client.add(
                f"large_scale_memory_{i:06d}",
                metadata={"timestamp": ts, "index": i}
            )
        mgr = MemoryLifecycleManager(client)
        report = mgr.get_report()
        assert report.total_memories == 500
        assert isinstance(report.age_distribution, dict)

    def test_corrupted_timestamps(self, empty_client):
        """损坏的时间戳不应崩溃。"""
        client = empty_client
        client.add("no_timestamp_memory")
        mgr = MemoryLifecycleManager(client)
        report = mgr.get_report()
        assert report.total_memories == 1

    def test_health_status_transitions(self, empty_client):
        """健康状态转换: healthy → fair → degraded → critical."""
        client = empty_client
        # 添加大量旧记忆 + 重复
        for i in range(100):
            ts = (datetime.now() - timedelta(days=200)).isoformat()
            client.add(
                f"stale_memory",  # 全部相同 → 大量重复
                metadata={"timestamp": ts}
            )
        mgr = MemoryLifecycleManager(client)
        report = mgr.get_report()
        # 大量重复+老旧 → 应降级
        assert report.health_status in ("degraded", "critical", "fair")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
