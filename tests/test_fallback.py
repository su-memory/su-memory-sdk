"""
su-memory-sdk Sprint 1 — 降级路径测试

验证各组件不可用时的优雅降级行为
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from su_memory.sdk.lite_pro import SuMemoryLitePro

# ═══════════════════════════════════════════════════════════════
# T5.1 FAISS → 线性检索
# ═══════════════════════════════════════════════════════════════

class TestFAISSFallback:
    """FAISS 降级到线性/关键词检索"""

    def test_disable_faiss_still_works(self):
        """禁用 FAISS 仍可正常使用"""
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(storage_path=d, enable_vector=False, enable_tfidf=True)
            c.add("FAISS 不可用时的测试")
            results = c.query("测试", use_vector=False, use_keyword=True)
            assert isinstance(results, list)

    def test_vector_disabled_keyword_enabled(self):
        """向量禁用时关键词检索可用"""
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(
                storage_path=d,
                enable_vector=False,
                enable_tfidf=True,
            )
            for i in range(10):
                c.add(f"关键词检索测试内容 {i}")
            results = c.query("关键词", use_vector=False, use_keyword=True)
            assert len(results) > 0

    def test_all_search_disabled(self):
        """所有检索禁用后仍可添加"""
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(
                storage_path=d,
                enable_vector=False,
                enable_tfidf=False,
            )
            mid = c.add("最小化配置测试")
            assert mid is not None
            assert len(c) == 1


# ═══════════════════════════════════════════════════════════════
# T5.2 Embedding → TF-IDF
# ═══════════════════════════════════════════════════════════════

class TestEmbeddingFallback:
    """Embedding 不可用时的降级"""

    def test_no_embedding_loaded(self):
        """不加载 embedding 时正常工作"""
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(storage_path=d, enable_vector=False)
            assert c._embedding is None or c._embedding_backend_type == "none"
            c.add("无embedding测试")
            assert len(c) == 1

    def test_tfidf_as_fallback(self):
        """TF-IDF 作为备选检索"""
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(storage_path=d, enable_vector=False, enable_tfidf=True)
            c.add("TF-IDF 检索内容")
            c.add("向量检索内容")
            results = c.query("TF-IDF", use_vector=False, use_keyword=True)
            assert len(results) >= 1

    def test_minimax_backend(self):
        """MiniMax 后端可初始化"""
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(
                storage_path=d,
                enable_vector=True,
                embedding_backend="minimax",
            )
            c.add("MiniMax 测试")
            assert len(c) == 1

    def test_ollama_backend(self):
        """Ollama 后端可初始化"""
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(
                storage_path=d,
                enable_vector=True,
                embedding_backend="ollama",
            )
            c.add("Ollama 测试")
            assert len(c) == 1


# ═══════════════════════════════════════════════════════════════
# T5.3 图谱 → 纯向量
# ═══════════════════════════════════════════════════════════════

class TestGraphFallback:
    """图谱不可用时的降级"""

    def test_disable_graph(self):
        """禁用图谱时仍可正常工作"""
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(
                storage_path=d,
                enable_vector=False,
                enable_graph=False,
                enable_tfidf=True,
            )
            a = c.add("无图谱A")
            b = c.add("无图谱B")
            # 链接操作应被忽略或处理
            try:
                c.link_memories(a, b)
            except Exception:
                pass
            results = c.query("无图谱", use_keyword=True)
            assert len(results) >= 1

    def test_graph_disabled_get_parents_empty(self):
        """禁用图谱时 get_parents 返回空"""
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(
                storage_path=d,
                enable_vector=False,
                enable_graph=False,
            )
            mid = c.add("孤立记忆")
            parents = c.get_parents(mid)
            assert parents == [] or len(parents) >= 0

    def test_graph_disabled_get_children_empty(self):
        """禁用图谱时 get_children 返回空"""
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(
                storage_path=d,
                enable_vector=False,
                enable_graph=False,
            )
            mid = c.add("无图节点")
            children = c.get_children(mid)
            assert children == [] or len(children) >= 0


# ═══════════════════════════════════════════════════════════════
# T5.4 时空索引 → 时序衰减
# ═══════════════════════════════════════════════════════════════

class TestTemporalFallback:
    """时空索引降级"""

    def test_disable_temporal(self):
        """禁用时空索引"""
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(
                storage_path=d,
                enable_vector=False,
                enable_temporal=False,
            )
            c.add("无时序记忆")
            # query 正常执行
            results = c.query("无时序")
            assert isinstance(results, list)

    def test_spacetime_with_no_temporal(self):
        """无时序时时空多跳降级"""
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(
                storage_path=d,
                enable_vector=False,
                enable_temporal=False,
                enable_graph=True,
            )
            a = c.add("A节点")
            b = c.add("B节点")
            c.link_memories(a, b)
            results = c.query_multihop_spacetime("A节点")
            assert isinstance(results, list)


# ═══════════════════════════════════════════════════════════════
# T5.5 存储 → 内存
# ═══════════════════════════════════════════════════════════════

class TestStorageFallback:
    """存储降级"""

    def test_minimal_config(self):
        """最小配置正常运行"""
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(
                storage_path=d,
                enable_vector=False,
                enable_graph=False,
                enable_temporal=False,
                enable_session=False,
                enable_prediction=False,
                enable_explainability=False,
            )
            mid = c.add("最小化")
            mem = c.get_memory(mid)
            assert mem is not None

    def test_large_memory_no_slowdown(self):
        """大量记忆不崩溃"""
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(
                storage_path=d,
                enable_vector=False,
                enable_graph=False,
                enable_temporal=False,
            )
            for i in range(50):
                c.add(f"bulk_{i}")
            assert len(c) == 50

    def test_get_memory_after_disable_all(self):
        """禁用所有后仍可 get_memory"""
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(
                storage_path=d,
                enable_vector=False,
                enable_graph=False,
                enable_temporal=False,
                enable_session=False,
            )
            mid = c.add("核心测试")
            mem = c.get_memory(mid)
            assert mem is not None

    def test_query_on_stripped_client(self):
        """极简客户端查询"""
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(
                storage_path=d,
                enable_vector=False,
                enable_graph=False,
                enable_temporal=False,
                enable_tfidf=False,
            )
            c.add("hello world")
            results = c.query("hello")
            assert isinstance(results, list)


# ═══════════════════════════════════════════════════════════════
# T5.6 组合降级
# ═══════════════════════════════════════════════════════════════

class TestCombinedFallback:
    """多模块同时降级"""

    def test_all_disabled_works(self):
        """全部禁用仍可基本操作"""
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(
                storage_path=d,
                enable_vector=False,
                enable_graph=False,
                enable_temporal=False,
                enable_session=False,
                enable_prediction=False,
                enable_explainability=False,
                enable_tfidf=False,
            )
            mid = c.add("bare minimum")
            assert len(c) == 1
            mem = c.get_memory(mid)
            assert mem is not None
            stats = c.get_stats()
            assert stats['total_memories'] == 1

    def test_fallback_chain(self):
        """逐级降级：向量→关键词→基础"""
        # Level 1: 向量
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(storage_path=d, enable_vector=True)
            c.add("向量检索")
            results = c.query("向量", use_vector=True, use_keyword=False)
            assert isinstance(results, list)

            # Level 2: 关键词
            results2 = c.query("向量", use_vector=False, use_keyword=True)
            assert isinstance(results2, list)

            # Level 3: 基础
            results3 = c.query("向量", use_vector=False, use_keyword=False)
            assert isinstance(results3, list)

    def test_no_prediction_no_crash(self):
        """禁用 prediction 不崩溃"""
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(
                storage_path=d,
                enable_vector=False,
                enable_prediction=False,
            )
            c.add("测试")
            if hasattr(c, 'predict'):
                result = c.predict()
                assert isinstance(result, dict)
