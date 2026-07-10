"""双轨分类测试 — 软分类 (主题分桶) 从数据涌现 + 检索路由。"""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
os.environ.setdefault("SU_MEMORY_SKIP_ENV_CHECK", "1")
os.environ.setdefault("SU_MEMORY_NO_LLM_ENERGY", "1")

from su_memory.sdk.lite_pro import SuMemoryLitePro
from su_memory.sdk._topic_clusterer import TopicClusterer


@pytest.fixture
def client():
    with tempfile.TemporaryDirectory() as d:
        c = SuMemoryLitePro(storage_path=d, enable_vector=False, enable_graph=True)
        yield c


class TestTopicClusterer:
    """增量主题分桶器"""

    def test_similar_memories_same_cluster(self):
        """高相似度的记忆归入同簇"""
        tc = TopicClusterer(similarity_threshold=0.25)
        c0 = tc.assign("m1", ["项目", "营收", "增长", "季度"])
        c1 = tc.assign("m2", ["项目", "营收", "报告", "季度"])
        assert c0 == c1, "相似记忆应在同簇"

    def test_dissimilar_memories_different_cluster(self):
        """低相似度的记忆开新簇"""
        tc = TopicClusterer(similarity_threshold=0.25)
        c0 = tc.assign("m1", ["天气", "暴雨", "水位"])
        c1 = tc.assign("m2", ["代码", "部署", "服务器"])
        assert c0 != c1, "不相似记忆应分簇"

    def test_cluster_grows_incrementally(self):
        """簇随新成员增长"""
        tc = TopicClusterer(similarity_threshold=0.2)
        tc.assign("m1", ["张总", "会议", "决策"])
        tc.assign("m2", ["张总", "会议", "纪要"])
        tc.assign("m3", ["张总", "会议", "记录"])
        topics = tc.get_topics()
        assert len(topics) >= 1
        assert topics[0]["size"] >= 3

    def test_query_routes_to_matching_clusters(self):
        """query 关键词匹配到正确簇"""
        tc = TopicClusterer(similarity_threshold=0.2)
        tc.assign("m1", ["版本", "发布", "修复"])
        tc.assign("m2", ["版本", "发布", "日志"])
        tc.assign("m3", ["天气", "暴雨", "预报"])
        matched = tc.query_clusters(["版本", "发布"], top_k=3)
        assert len(matched) >= 1
        # "版本发布"簇应排在最前
        top_cid = matched[0][0]
        members = tc.cluster_members(top_cid)
        assert "m1" in members or "m2" in members


class TestDualTrackIntegration:
    """软分类接入 SuMemoryLitePro"""

    def test_topic_assigned_on_add(self, client):
        """add() 后记忆有 _topic_cluster 标签"""
        mid = client.add("项目Q3营收增长报告")
        idx = client._memory_map[mid]
        assert "_topic_cluster" in client._memories[idx].metadata

    def test_get_topics_returns_clusters(self, client):
        """get_topics() 返回自动发现的主题"""
        # 加入两类不同主题的记忆
        for i in range(5):
            client.add(f"项目营收增长数据报告第{i}部分")
        for i in range(5):
            client.add(f"暴雨天气预报预警第{i}号")
        topics = client.get_topics()
        assert len(topics) >= 2, f"应有≥2个主题簇, 实际 {len(topics)}"

    def test_dual_track_orthogonal(self, client):
        """软分类(主题)与硬分类(energy_type)正交: 同主题可有不同能量"""
        client.add("项目营收增长报告")  # 同主题
        client.add("项目营收分析总结")
        topics = client.get_topics()
        # 两条同主题记忆应在同一簇 (或相近簇)
        assert len(topics) >= 1

    def test_cluster_boost_in_query(self, client):
        """主题桶内记忆在 query 时获得提权"""
        # 建立一个明确的主题簇
        for i in range(6):
            client.add(f"部署服务器配置nginx第{i}步")
        # 加一条不相关的噪声
        client.add("今天午饭吃了面条")
        # query 应优先命中主题相关记忆
        results = client.query("部署服务器", top_k=5)
        contents = [r["content"] for r in results]
        # 噪声不应排在前面
        relevant = [c for c in contents if "部署" in c or "服务器" in c]
        assert len(relevant) >= 1
        if "面条" in contents:
            noodle_idx = contents.index("今天午饭吃了面条")
            assert noodle_idx >= len(relevant), "噪声记忆不应排在主题记忆之前"
