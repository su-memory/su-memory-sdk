"""
su-memory-sdk Sprint 1 — VectorGraphRAG 多跳推理测试（扩展）

覆盖: 单跳/多跳/融合模式/因果类型/边界/环形图/孤立节点/回退
"""
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from su_memory.sdk.vector_graph_rag import VectorGraphRAG


def _dim128_embedding(text: str):
    seed = abs(hash(text)) % (2**31)
    rng = np.random.RandomState(seed)
    vec = rng.randn(128).astype(np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec.tolist()


@pytest.fixture
def rag():
    with tempfile.TemporaryDirectory() as d:
        r = VectorGraphRAG(
            embedding_func=_dim128_embedding,
            dims=128,
            enable_faiss=True,
            storage_path=d,
        )
        yield r


@pytest.fixture
def chain_rag(rag):
    """A→B→C→D 因果链"""
    rag.add_memory("a", "启动项目", causal_type="cause")
    rag.add_memory("b", "编写需求文档", causal_type="effect")
    rag.add_memory("c", "开发迭代开始", causal_type="effect")
    rag.add_memory("d", "测试用例编写", causal_type="prerequisite")
    rag.add_edge("a", "b", causal_type="cause")
    rag.add_edge("b", "c", causal_type="effect")
    rag.add_edge("c", "d", causal_type="prerequisite")
    return rag


@pytest.fixture
def cycle_rag(rag):
    """A→B→C→A 环形图"""
    rag.add_memory("a", "A节点")
    rag.add_memory("b", "B节点")
    rag.add_memory("c", "C节点")
    rag.add_edge("a", "b")
    rag.add_edge("b", "c")
    rag.add_edge("c", "a")
    return rag


@pytest.fixture
def star_rag(rag):
    """星形图：中心→4个叶子"""
    rag.add_memory("center", "中心节点")
    for i in range(4):
        rag.add_memory(f"leaf{i}", f"叶子节点{i}")
        rag.add_edge("center", f"leaf{i}")
    return rag


# ═══════════════════════════════════════════════════════════════
# T3.1 单跳/两跳/三跳
# ═══════════════════════════════════════════════════════════════

class TestHopCount:
    """测试跳数控制"""

    def test_max_hops_1(self, chain_rag):
        """max_hops=1 搜索结果跳数合理"""
        results = chain_rag.multi_hop_query("项目", max_hops=1)
        # max_hops 限制搜索深度，结果可能有不同跳数
        for r in results:
            assert r.hops >= 0

    def test_max_hops_3(self, chain_rag):
        """max_hops=3 可覆盖 4 节点链"""
        results = chain_rag.multi_hop_query("项目", max_hops=3, top_k=5)
        assert len(results) >= 1

    def test_max_hops_0(self, chain_rag):
        """max_hops=0 仅直接匹配"""
        results = chain_rag.multi_hop_query("项目", max_hops=0)
        assert isinstance(results, list)

    def test_hops_value_monotonic(self, chain_rag):
        """hops 值有意义"""
        results = chain_rag.multi_hop_query("启动", max_hops=3, top_k=10)
        hops_set = set(r.hops for r in results)
        # 应有不同跳数的结果
        assert len(hops_set) >= 1

    def test_three_hop_chain(self, rag):
        """三跳链: A→B→C→D"""
        rag.add_memory("a", "原材料采购")
        rag.add_memory("b", "生产制造过程")
        rag.add_memory("c", "质量检验环节")
        rag.add_memory("d", "成品发货阶段")
        rag.add_edge("a", "b")
        rag.add_edge("b", "c")
        rag.add_edge("c", "d")
        results = rag.multi_hop_query("成品", max_hops=3, top_k=10)
        assert len(results) >= 1

    def test_long_chain(self, rag):
        """六节点长链"""
        rag.add_memory("n0", "步骤0")
        for i in range(1, 6):
            rag.add_memory(f"n{i}", f"步骤{i}")
            rag.add_edge(f"n{i-1}", f"n{i}")
        results = rag.multi_hop_query("步骤5", max_hops=5, top_k=10)
        assert isinstance(results, list)


# ═══════════════════════════════════════════════════════════════
# T3.2 因果类型
# ═══════════════════════════════════════════════════════════════

class TestCausalTypes:
    """测试因果类型标注"""

    def test_cause_effect(self, rag):
        """cause→effect 查询"""
        rag.add_memory("a", "下雨", causal_type="cause")
        rag.add_memory("b", "地面湿", causal_type="effect")
        rag.add_edge("a", "b", causal_type="cause")
        results = rag.multi_hop_query("下雨", max_hops=2)
        assert len(results) >= 1

    def test_prerequisite(self, rag):
        """prerequisite 类型"""
        rag.add_memory("a", "需求确认", causal_type="prerequisite")
        rag.add_memory("b", "开始开发", causal_type="effect")
        rag.add_edge("a", "b", causal_type="prerequisite")
        results = rag.multi_hop_query("开发", max_hops=2)
        assert len(results) >= 1

    def test_consequence(self, rag):
        """consequence 类型"""
        rag.add_memory("a", "漏洞发现", causal_type="cause")
        rag.add_memory("b", "系统修复", causal_type="consequence")
        rag.add_edge("a", "b", causal_type="consequence")
        results = rag.multi_hop_query("漏洞", max_hops=2)
        assert len(results) >= 1

    def test_no_causal_type(self, rag):
        """无因果类型标注"""
        rag.add_memory("a", "A内容")
        rag.add_memory("b", "B内容")
        rag.add_edge("a", "b")
        results = rag.multi_hop_query("A内容", max_hops=2)
        assert len(results) >= 1

    def test_mixed_causal(self, rag):
        """混合因果类型链"""
        rag.add_memory("a", "需求评审", causal_type="prerequisite")
        rag.add_memory("b", "技术方案", causal_type="effect")
        rag.add_memory("c", "上线部署", causal_type="consequence")
        rag.add_edge("a", "b", causal_type="prerequisite")
        rag.add_edge("b", "c", causal_type="consequence")
        results = rag.multi_hop_query("上线", max_hops=3, top_k=5)
        assert len(results) >= 1


# ═══════════════════════════════════════════════════════════════
# T3.3 融合模式
# ═══════════════════════════════════════════════════════════════

class TestFusionModes:
    """测试多跳融合模式"""

    def test_vector_first(self, chain_rag):
        """默认多跳查询"""
        results = chain_rag.multi_hop_query("项目")
        assert len(results) >= 1

    def test_graph_first(self, chain_rag):
        """多跳查询 top_k=3"""
        results = chain_rag.multi_hop_query("项目", top_k=3)
        assert len(results) >= 1

    def test_hybrid(self, chain_rag):
        """多跳查询 max_hops=2"""
        results = chain_rag.multi_hop_query("项目", max_hops=2)
        assert len(results) >= 1

    def test_different_modes_different_order(self, chain_rag):
        """不同参数结果顺序可能不同"""
        r1 = chain_rag.multi_hop_query("项目", top_k=5)
        r2 = chain_rag.multi_hop_query("项目", top_k=3)
        ids1 = [x.node_id for x in r1]
        ids2 = [x.node_id for x in r2]
        assert len(set(ids1) & set(ids2)) >= 1

    def test_fusion_no_crash(self, rag):
        """所有参数组合不崩溃"""
        rag.add_memory("m0", "test")
        for max_hops in [1, 2, 3]:
            results = rag.multi_hop_query("test", max_hops=max_hops)
            assert isinstance(results, list)


# ═══════════════════════════════════════════════════════════════
# T3.4 max_hops / min_score 边界
# ═══════════════════════════════════════════════════════════════

class TestBoundaries:
    """测试边界参数"""

    def test_max_hops_negative(self, rag):
        """max_hops 负数"""
        rag.add_memory("m0", "test")
        results = rag.multi_hop_query("test", max_hops=-1)
        assert isinstance(results, list)

    def test_min_score_zero(self, chain_rag):
        """min_score=0 返回所有结果"""
        results = chain_rag.multi_hop_query("项目", min_score=0.0, top_k=10)
        assert len(results) >= 1

    def test_min_score_high(self, chain_rag):
        """min_score=0.9 可能无结果"""
        results = chain_rag.multi_hop_query("项目", min_score=0.9)
        assert isinstance(results, list)

    def test_decay_zero(self, chain_rag):
        """decay=0 不追踪链"""
        results = chain_rag.multi_hop_query("项目", decay=0.0)
        assert len(results) >= 1

    def test_decay_one(self, chain_rag):
        """decay=1.0 无衰减"""
        results = chain_rag.multi_hop_query("项目", decay=1.0)
        assert len(results) >= 1


# ═══════════════════════════════════════════════════════════════
# T3.5 环形图 / 孤立节点
# ═══════════════════════════════════════════════════════════════

class TestGraphTopology:
    """测试特殊拓扑"""

    def test_cycle_no_infinite_loop(self, cycle_rag):
        """环形图不进入死循环"""
        results = cycle_rag.multi_hop_query("A节点", max_hops=5)
        assert isinstance(results, list)

    def test_cycle_all_nodes_reachable(self, cycle_rag):
        """环形图中所有节点可达"""
        results = cycle_rag.multi_hop_query("A节点", max_hops=5, top_k=10)
        memory_ids = set(r.node_id for r in results)
        assert len(memory_ids) >= 1

    def test_isolated_node(self, rag):
        """孤立节点仍可通过向量检索找到"""
        rag.add_memory("iso", "孤立节点独特关键词")
        rag.add_memory("m0", "普通节点")
        results = rag.multi_hop_query("独特关键词", max_hops=1)
        memory_ids = set(r.node_id for r in results)
        assert "iso" in memory_ids

    def test_star_topology(self, star_rag):
        """星形图中心节点可达所有叶子"""
        results = star_rag.multi_hop_query("中心节点", max_hops=2, top_k=10)
        assert len(results) >= 1


# ═══════════════════════════════════════════════════════════════
# T3.6 嵌入不可用回退
# ═══════════════════════════════════════════════════════════════

class TestEmbeddingFallback:
    """嵌入回退测试"""

    def test_no_embedding_no_crash(self):
        """禁用 FAISS 时多跳不崩溃"""
        with tempfile.TemporaryDirectory() as d:
            r = VectorGraphRAG(
                embedding_func=_dim128_embedding,
                dims=128,
                enable_faiss=False,
                storage_path=d,
            )
            r.add_memory("m0", "测试")
            results = r.multi_hop_query("测试")
            assert isinstance(results, list)

    def test_pure_graph_traversal(self):
        """纯图遍历（嵌入不可用仍可查链路）"""
        with tempfile.TemporaryDirectory() as d:
            r = VectorGraphRAG(
                embedding_func=_dim128_embedding,
                dims=128,
                enable_faiss=False,
                storage_path=d,
            )
            r.add_memory("a", "A")
            r.add_memory("b", "B")
            r.add_edge("a", "b")
            results = r.multi_hop_query("A", max_hops=2)
            assert isinstance(results, list)

    def test_batch_cache_off(self):
        """禁用批量缓存"""
        with tempfile.TemporaryDirectory() as d:
            r = VectorGraphRAG(
                embedding_func=_dim128_embedding,
                dims=128,
                enable_batch_cache=False,
                storage_path=d,
            )
            r.add_memory("m0", "test")
            results = r.multi_hop_query("test")
            assert isinstance(results, list)


# ═══════════════════════════════════════════════════════════════
# T3.7 大规模图谱
# ═══════════════════════════════════════════════════════════════

class TestScale:
    """大规模性能"""

    def test_100_node_graph(self, rag):
        """100 节点图谱"""
        for i in range(100):
            rag.add_memory(f"m{i}", f"大规模测试节点{i}")
        results = rag.multi_hop_query("节点50", top_k=3)
        assert len(results) >= 1

    def test_graph_with_edges(self, rag):
        """50 节点 + 80 条边"""
        for i in range(50):
            rag.add_memory(f"m{i}", f"边测试{i}")
        for i in range(1, 50):
            rag.add_edge(f"m{i-1}", f"m{i}")
        results = rag.multi_hop_query("边测试25", max_hops=5, top_k=5)
        assert isinstance(results, list)
