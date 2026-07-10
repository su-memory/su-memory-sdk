"""关联边成色测试 — 验证四档 confidence + 路径剪枝。"""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
os.environ.setdefault("SU_MEMORY_SKIP_ENV_CHECK", "1")
os.environ.setdefault("SU_MEMORY_NO_LLM_ENERGY", "1")

from su_memory.sdk.lite_pro import SuMemoryLitePro, MemoryGraph, Edge, MemoryNode


@pytest.fixture
def client():
    with tempfile.TemporaryDirectory() as d:
        c = SuMemoryLitePro(storage_path=d, enable_vector=False, enable_graph=True)
        yield c


class TestEdgeTiers:
    """四档成色: explicit > causal > semantic > heuristic"""

    def test_explicit_edge_gets_high_confidence(self, client):
        """显式声明 parent_ids → evidence_type=explicit, confidence=0.95"""
        parent_id = client.add("原始决策记录")
        child_id = client.add("后续行动", parent_ids=[parent_id])
        edge = client._graph.get_edge(parent_id, child_id)
        assert edge is not None
        assert edge.evidence_type == "explicit"
        assert edge.confidence == 0.95

    def test_heuristic_edge_low_confidence(self):
        """关键词推断 → evidence_type=heuristic, confidence=0.20"""
        g = MemoryGraph()
        n1 = MemoryNode(id="a", content="因为暴雨", metadata={})
        n2 = MemoryNode(id="b", content="所以水位暴涨", metadata={})
        g.add_node(n1)
        g.add_node(n2)  # n2 加入时自动推断与 n1 的 heuristic 关联
        edge = g.get_edge("a", "b")
        assert edge is not None
        assert edge.evidence_type == "heuristic"
        assert edge.confidence == 0.20

    def test_no_spurious_edge_without_keywords(self):
        """无因果连接词的内容不产生 heuristic 边"""
        g = MemoryGraph()
        n1 = MemoryNode(id="a", content="今天天气不错", metadata={})
        n2 = MemoryNode(id="b", content="午饭吃了面条", metadata={})
        g.add_node(n1)
        g.add_node(n2)
        assert g.get_edge("a", "b") is None
        assert g.get_edge("b", "a") is None

    def test_edge_upgrade_not_downgrade(self):
        """已有高成色边时, 低成色 add_edge 不降级"""
        g = MemoryGraph()
        g.add_node(MemoryNode(id="a", content="x", metadata={}))
        g.add_node(MemoryNode(id="b", content="y", metadata={}))
        g.add_edge("a", "b", confidence=0.95, evidence_type="explicit")
        g.add_edge("a", "b", confidence=0.20, evidence_type="heuristic")
        edge = g.get_edge("a", "b")
        assert edge.confidence == 0.95  # 保留高成色


class TestPathConfidencePruning:
    """路径置信度剪枝: heuristic 2 跳后淘汰, explicit 3 跳仍存活"""

    def test_heuristic_chain_pruned_at_2_hops(self):
        """heuristic(0.20)² = 0.04 < 0.1 阈值 → 2 跳路径被剪"""
        g = MemoryGraph()
        for nid in ["a", "b", "c"]:
            g.add_node(MemoryNode(id=nid, content="占位", metadata={}))
        g.add_edge("a", "b", confidence=0.20, evidence_type="heuristic")
        g.add_edge("b", "c", confidence=0.20, evidence_type="heuristic")
        results = g.bfs_hops(["a"], max_hops=3, min_path_confidence=0.1)
        node_ids = [r[0] for r in results]
        assert "a" in node_ids  # 起点
        assert "b" in node_ids  # 1 跳: 0.20 > 0.1 ✓
        assert "c" not in node_ids  # 2 跳: 0.04 < 0.1 ✗ 被剪

    def test_explicit_chain_survives_3_hops(self):
        """explicit(0.95)³ = 0.857 > 0.1 → 3 跳路径存活"""
        g = MemoryGraph()
        for nid in ["a", "b", "c", "d"]:
            g.add_node(MemoryNode(id=nid, content="占位", metadata={}))
        g.add_edge("a", "b", confidence=0.95, evidence_type="explicit")
        g.add_edge("b", "c", confidence=0.95, evidence_type="explicit")
        g.add_edge("c", "d", confidence=0.95, evidence_type="explicit")
        results = g.bfs_hops(["a"], max_hops=3, min_path_confidence=0.1)
        node_ids = [r[0] for r in results]
        assert "d" in node_ids  # 3 跳存活

    def test_path_confidence_in_results(self):
        """bfs_hops 返回值含 path_confidence (第 5 元素)"""
        g = MemoryGraph()
        g.add_node(MemoryNode(id="a", content="x", metadata={}))
        g.add_node(MemoryNode(id="b", content="y", metadata={}))
        g.add_edge("a", "b", confidence=0.80, evidence_type="causal")
        results = g.bfs_hops(["a"], max_hops=1)
        assert len(results[0]) == 5  # (node, hops, path, causal_type, path_conf)
        # 找 b 的结果
        b_result = [r for r in results if r[0] == "b"][0]
        assert b_result[4] == 0.80  # path_confidence


class TestBackwardCompat:
    """parent_ids / child_ids 向后兼容"""

    def test_parent_child_ids_still_work(self, client):
        """显式 parent_ids 仍写入 parent_ids/child_ids 列表"""
        pid = client.add("父记忆")
        cid = client.add("子记忆", parent_ids=[pid])
        parents = client._graph.get_parents(cid)
        children = client._graph.get_children(pid)
        assert pid in parents
        assert cid in children
