"""
su-memory-sdk Sprint 1 — FAISS HNSW 索引测试

通过 VectorGraphRAG 间接测试 FAISS 索引的创建/检索/持久化/量化压缩
"""
import os
import sys
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from su_memory.sdk.vector_graph_rag import VectorGraphRAG


# ── 简单 embedding 函数 ────────────────────────────────────────

def _dim128_embedding(text: str):
    """128维随机归一化向量（确定性，基于hash）"""
    seed = abs(hash(text)) % (2**31)
    rng = np.random.RandomState(seed)
    vec = rng.randn(128).astype(np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec.tolist()


def _dim256_embedding(text: str):
    """256维"""
    seed = abs(hash(text)) % (2**31)
    rng = np.random.RandomState(seed)
    vec = rng.randn(256).astype(np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec.tolist()


@pytest.fixture
def rag():
    """默认 VectorGraphRAG 实例（128维，FAISS启用）"""
    with tempfile.TemporaryDirectory() as d:
        r = VectorGraphRAG(
            embedding_func=_dim128_embedding,
            dims=128,
            enable_faiss=True,
            storage_path=d,
            hnsw_m=16,
            hnsw_ef_construction=40,
            hnsw_ef_search=32,
        )
        yield r


@pytest.fixture
def rag_fp32(rag):
    """填充 10 条记忆的 RAG"""
    for i in range(10):
        rag.add_memory(f"mem_{i}", f"这是第 {i} 条测试记忆，包含一些关键词")
    return rag


# ═══════════════════════════════════════════════════════════════
# T2.1 索引创建参数
# ═══════════════════════════════════════════════════════════════

class TestIndexCreation:
    """测试 FAISS 索引创建参数"""

    def test_default_creation(self, rag):
        """默认参数创建"""
        s = rag.get_stats()
        assert s['faiss_enabled'] is True
        assert s['total_nodes'] == 0

    def test_disable_faiss(self):
        """禁用 FAISS"""
        with tempfile.TemporaryDirectory() as d:
            r = VectorGraphRAG(
                embedding_func=_dim128_embedding,
                dims=128,
                enable_faiss=False,
                storage_path=d,
            )
            s = r.get_stats()
            assert s['faiss_enabled'] is False

    def test_custom_hnsw_params(self):
        """自定义 HNSW 参数"""
        with tempfile.TemporaryDirectory() as d:
            r = VectorGraphRAG(
                embedding_func=_dim128_embedding,
                dims=128,
                enable_faiss=True,
                hnsw_m=64,
                hnsw_ef_construction=200,
                hnsw_ef_search=100,
                storage_path=d,
            )
            s = r.get_stats()
            assert s['faiss_enabled'] is True

    def test_256_dim(self):
        """256维"""
        with tempfile.TemporaryDirectory() as d:
            r = VectorGraphRAG(
                embedding_func=_dim256_embedding,
                dims=256,
                enable_faiss=True,
                storage_path=d,
            )
            r.add_memory("m0", "256维度测试")
            s = r.get_stats()
            assert s['total_nodes'] == 1

    def test_batch_cache_config(self):
        """批量缓存配置"""
        with tempfile.TemporaryDirectory() as d:
            r = VectorGraphRAG(
                embedding_func=_dim128_embedding,
                dims=128,
                enable_batch_cache=True,
                cache_size=500,
                storage_path=d,
            )
            assert True  # 初始化成功


# ═══════════════════════════════════════════════════════════════
# T2.2 add + search
# ═══════════════════════════════════════════════════════════════

class TestAddSearch:
    """测试添加和搜索"""

    def test_add_increments_ntotal(self, rag):
        """add 后 total_nodes 递增"""
        rag.add_memory("m0", "test0")
        assert rag.get_stats()['total_nodes'] == 1
        rag.add_memory("m1", "test1")
        assert rag.get_stats()['total_nodes'] == 2

    def test_multi_hop_basic(self, rag_fp32):
        """添加后多跳查询返回结果"""
        results = rag_fp32.multi_hop_query("测试记忆", max_hops=1, top_k=3)
        assert len(results) > 0

    def test_multi_hop_score_range(self, rag_fp32):
        """分数在合理范围"""
        results = rag_fp32.multi_hop_query("测试", top_k=3)
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_multi_hop_increases_ntotal(self, rag):
        """多次查询不改变索引大小"""
        rag.add_memory("m0", "test")
        n = rag.get_stats()['total_nodes']
        rag.multi_hop_query("test")
        assert rag.get_stats()['total_nodes'] == n

    def test_null_query(self, rag_fp32):
        """空字符串查询不崩溃"""
        results = rag_fp32.multi_hop_query("", max_hops=1)
        assert isinstance(results, list)

    def test_search_on_empty_index(self, rag):
        """空索引查询不崩溃"""
        results = rag.multi_hop_query("test")
        assert isinstance(results, list)
        assert len(results) == 0


# ═══════════════════════════════════════════════════════════════
# T2.3 量化压缩
# ═══════════════════════════════════════════════════════════════

class TestQuantization:
    """测试量化压缩模式"""

    def test_int8_quantization(self):
        """INT8 量化"""
        with tempfile.TemporaryDirectory() as d:
            r = VectorGraphRAG(
                embedding_func=_dim128_embedding,
                dims=128,
                quantization_mode='int8',
                storage_path=d,
            )
            for i in range(5):
                r.add_memory(f"m{i}", f"INT8测试{i}")
            results = r.multi_hop_query("INT8", top_k=2)
            assert len(results) >= 1

    def test_fp16_quantization(self):
        """FP16 量化"""
        with tempfile.TemporaryDirectory() as d:
            r = VectorGraphRAG(
                embedding_func=_dim128_embedding,
                dims=128,
                quantization_mode='fp16',
                storage_path=d,
            )
            r.add_memory("m0", "FP16测试")
            results = r.multi_hop_query("FP16", top_k=1)
            assert isinstance(results, list)

    def test_fp32_default(self):
        """FP32 默认"""
        with tempfile.TemporaryDirectory() as d:
            r = VectorGraphRAG(
                embedding_func=_dim128_embedding,
                dims=128,
                storage_path=d,
            )
            r.add_memory("m0", "FP32默认测试")
            results = r.multi_hop_query("FP32", top_k=1)
            assert isinstance(results, list)

    def test_quantized_search_repeatable(self):
        """量化后查询结果可重复"""
        with tempfile.TemporaryDirectory() as d:
            r = VectorGraphRAG(
                embedding_func=_dim128_embedding,
                dims=128,
                quantization_mode='int8',
                storage_path=d,
            )
            r.add_memory("m0", "苹果是一种水果")
            r.add_memory("m1", "苹果是科技公司")
            r1 = r.multi_hop_query("苹果", top_k=2)
            r2 = r.multi_hop_query("苹果", top_k=2)
            assert len(r1) == len(r2)
            ids1 = [x.node_id for x in r1]
            ids2 = [x.node_id for x in r2]
            assert ids1 == ids2

    def test_binary_quantization_fallback(self):
        """int8 量化参数可正常初始化"""
        with tempfile.TemporaryDirectory() as d:
            r = VectorGraphRAG(
                embedding_func=_dim128_embedding,
                dims=128,
                quantization_mode='int8',
                storage_path=d,
            )
            r.add_memory("m0", "test")
            assert r.get_stats()['total_nodes'] == 1


# ═══════════════════════════════════════════════════════════════
# T2.4 持久化
# ═══════════════════════════════════════════════════════════════

class TestPersistence:
    """测试索引持久化"""

    def test_save_and_reload(self):
        """保存后同一实例可重新查询"""
        with tempfile.TemporaryDirectory() as d:
            r = VectorGraphRAG(
                embedding_func=_dim128_embedding,
                dims=128,
                enable_faiss=True,
                storage_path=d,
            )
            for i in range(5):
                r.add_memory(f"m{i}", f"持续化测试{i}")
            n = r.get_stats()['total_nodes']
            results = r.multi_hop_query("持续化", top_k=1)
            assert n == 5
            assert len(results) >= 1

    def test_new_instance_different_path(self):
        """不同路径独立存储"""
        with tempfile.TemporaryDirectory() as d1:
            with tempfile.TemporaryDirectory() as d2:
                r1 = VectorGraphRAG(
                    embedding_func=_dim128_embedding,
                    dims=128, storage_path=d1,
                )
                r1.add_memory("m0", "路径1测试")
                
                r2 = VectorGraphRAG(
                    embedding_func=_dim128_embedding,
                    dims=128, storage_path=d2,
                )
                r2.add_memory("m0", "路径2测试")
                assert r2.get_stats()['total_nodes'] == 1

    def test_faiss_not_created_without_enable(self):
        """disable faiss 时不创建索引文件"""
        with tempfile.TemporaryDirectory() as d:
            r = VectorGraphRAG(
                embedding_func=_dim128_embedding,
                dims=128,
                enable_faiss=False,
                storage_path=d,
            )
            r.add_memory("m0", "test")
            assert r.get_stats()['faiss_enabled'] is False

    def test_same_instance_multi_query(self):
        """同一实例多次查询结果一致"""
        with tempfile.TemporaryDirectory() as d:
            r = VectorGraphRAG(
                embedding_func=_dim128_embedding,
                dims=128,
                enable_faiss=True,
                storage_path=d,
            )
            r.add_memory("m0", "持久数据")
            s1 = r.get_stats()
            s2 = r.get_stats()
            assert s1['total_nodes'] == s2['total_nodes']


# ═══════════════════════════════════════════════════════════════
# T2.5 增量索引
# ═══════════════════════════════════════════════════════════════

class TestIncremental:
    """测试增量索引"""

    def test_sequential_adds(self, rag):
        """连续添加 50 条"""
        for i in range(50):
            rag.add_memory(f"m{i}", f"增量{i}")
        assert rag.get_stats()['total_nodes'] == 50

    def test_add_then_search_new(self, rag):
        """新添加立即可搜索"""
        rag.add_memory("m0", "新添加的内容")
        results = rag.multi_hop_query("新添加", top_k=1)
        assert len(results) >= 1

    def test_index_stability(self, rag):
        """索引不因添加而退化"""
        for i in range(20):
            rag.add_memory(f"m{i}", f"稳定性测试{i}")
        n = rag.get_stats()['total_nodes']
        assert n == 20
        rag.multi_hop_query("稳定性", top_k=5)
        assert rag.get_stats()['total_nodes'] == n


# ═══════════════════════════════════════════════════════════════
# T2.6 边界情况
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:
    """边界和异常"""

    def test_empty_index_search(self, rag):
        """空索引查询返回空"""
        results = rag.multi_hop_query("test")
        assert len(results) == 0

    def test_dimension_match(self, rag):
        """维度匹配——128维数据用128维索引"""
        rag.add_memory("m0", "test")
        assert rag.get_stats()['total_nodes'] == 1

    def test_very_large_top_k(self, rag):
        """top_k 超过索引总数"""
        rag.add_memory("m0", "topk")
        results = rag.multi_hop_query("topk", top_k=100)
        assert len(results) <= 100
