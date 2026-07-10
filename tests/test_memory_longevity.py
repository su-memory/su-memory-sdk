"""超长准确记忆测试 — 频率加权衰减 + 语义归纳。"""
import os, sys, time, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
os.environ.setdefault("SU_MEMORY_SKIP_ENV_CHECK", "1")
os.environ.setdefault("SU_MEMORY_NO_LLM_ENERGY", "1")

from su_memory.sdk.lite_pro import SuMemoryLitePro
from su_memory.sdk.spacetime_multihop import SpacetimeMultihopEngine


@pytest.fixture
def client():
    with tempfile.TemporaryDirectory() as d:
        c = SuMemoryLitePro(storage_path=d, enable_vector=False, enable_graph=True)
        yield c


class TestFrequencyWeightedDecay:
    """访问频率高的记忆衰减更慢"""

    def test_access_count_starts_zero(self, client):
        mid = client.add("测试记忆")
        idx = client._memory_map[mid]
        assert client._memories[idx].access_count == 0

    def test_query_increments_access_count(self, client):
        mid = client.add("因为暴雨所以路滑")
        client.query("暴雨", top_k=5)
        idx = client._memory_map[mid]
        assert client._memories[idx].access_count >= 1
        assert client._memories[idx].last_accessed > 0

    def test_frequent_memory_decays_slower(self):
        """access_count=100 的记忆 30 天后衰减远小于 access_count=0"""
        engine = SpacetimeMultihopEngine(memory_nodes={})
        now = int(time.time())
        ts_30_days_ago = now - 30 * 86400

        decay_fresh = engine._calculate_time_decay(ts_30_days_ago, now, access_count=0)
        decay_frequent = engine._calculate_time_decay(ts_30_days_ago, now, access_count=100)

        assert decay_frequent > decay_fresh, (
            f"高频记忆应衰减更慢: frequent={decay_frequent} <= fresh={decay_fresh}"
        )
        # 100 次访问的记忆 30 天后保留率应 > 50%
        assert decay_frequent > 0.5

    def test_never_accessed_old_memory_decays(self):
        """从未访问的旧记忆正常衰减"""
        engine = SpacetimeMultihopEngine(memory_nodes={})
        now = int(time.time())
        ts_200_days_ago = now - 200 * 86400
        decay = engine._calculate_time_decay(ts_200_days_ago, now, access_count=0)
        # 200 天未访问 → 衰减到地板值
        assert decay <= 0.15


class TestConsolidation:
    """语义归纳: 同主题记忆合并为摘要"""

    def test_consolidate_merges_similar(self, client):
        """5 条高度相似的记忆 → 归纳成 1 条摘要"""
        for i in range(6):
            client.add(f"项目Q3季度营收增长数据报告第{i}部分")
        result = client.consolidate(similarity_threshold=0.4, min_cluster_size=5)
        assert result["summary_memories_created"] >= 1
        assert result["details_archived"] >= 5

    def test_consolidated_memories_marked(self, client):
        """被归纳的原始记忆标记 _consolidated=True"""
        for i in range(6):
            client.add(f"张总会议纪要第{i}段内容记录")
        client.consolidate(similarity_threshold=0.3, min_cluster_size=5)
        marked = [n for n in client._memories if n.metadata.get("_consolidated")]
        assert len(marked) >= 5

    def test_summary_has_member_ids(self, client):
        """摘要记忆的 metadata 含 _member_ids 列表"""
        for i in range(6):
            client.add(f"版本发布日志v2.{i}修复内容详情")
        client.consolidate(similarity_threshold=0.3, min_cluster_size=5)
        summaries = [n for n in client._memories if n.metadata.get("_consolidated_summary")]
        assert len(summaries) >= 1
        assert "_member_ids" in summaries[0].metadata
        assert len(summaries[0].metadata["_member_ids"]) >= 5

    def test_consolidate_skips_when_too_few(self, client):
        """记忆太少时不归纳"""
        client.add("孤立的记忆")
        result = client.consolidate(min_cluster_size=5)
        assert result["summary_memories_created"] == 0
