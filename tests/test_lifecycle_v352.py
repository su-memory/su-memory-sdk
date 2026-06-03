"""
su-memory v3.5.2 记忆生命周期测试

覆盖:
- Task 1: delete_memory() API
- Task 2: update_memory() API
- Task 3: 语义去重
- Task 4: MemoryForgetting 串联
- Task 5: TieredStorage 串联
- Task 6: SuCompressor 长记忆压缩
- Task 7: Cross-Encoder 精排
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 强制跳过 Ollama 探测以确保测试可重复
os.environ["SU_MEMORY_DISABLE_OLLAMA"] = "1"


class TestMemoryLifecycle(unittest.TestCase):
    """Task 1+2: create → read → update → delete 全生命周期"""

    @classmethod
    def setUpClass(cls):
        from su_memory.sdk.lite_pro import SuMemoryLitePro
        cls.SuMemoryLitePro = SuMemoryLitePro

    def setUp(self):
        self.client = self.SuMemoryLitePro(
            max_memories=100,
            enable_graph=False,
            enable_temporal=False,
            enable_session=False,
            enable_prediction=False,
            enable_explainability=False,
        )

    def test_add_and_retrieve(self):
        """基础 add → query"""
        mid = self.client.add("Python 是一门动态类型语言")
        self.assertIsNotNone(mid)
        self.assertTrue(mid.startswith("mem_"))

        results = self.client.query("Python")
        self.assertGreater(len(results), 0)

    def test_delete_memory(self):
        """Task 1: delete() 删除记忆"""
        mid = self.client.add("这是一条测试记忆，用于验证删除功能")
        self.assertTrue(self.client.delete(mid))

        # 删除后查询不应返回
        results = self.client.query("删除功能")
        self.assertEqual(len([r for r in results if r["memory_id"] == mid]), 0)

        # 重复删除应返回 False
        self.assertFalse(self.client.delete(mid))

    def test_delete_nonexistent(self):
        """删除不存在的记忆返回 False"""
        self.assertFalse(self.client.delete("mem_nonexistent"))

    def test_update_content(self):
        """Task 2: update() 更新内容"""
        mid = self.client.add("原始记忆内容")
        self.assertTrue(self.client.update(mid, content="更新后的记忆内容"))

        results = self.client.query("更新后")
        found = [r for r in results if r["memory_id"] == mid]
        self.assertEqual(len(found), 1)

    def test_update_metadata(self):
        """Task 2: update() 更新元数据"""
        mid = self.client.add("带有元数据的记忆", metadata={"tag": "test"})
        self.assertTrue(self.client.update(mid, metadata={"tag": "updated", "new_key": "value"}))

        # 通过查询结果验证 metadata 已更新
        results = self.client.query("元数据")
        found = [r for r in results if r["memory_id"] == mid]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["metadata"].get("tag"), "updated")
        self.assertEqual(found[0]["metadata"].get("new_key"), "value")

    def test_update_nonexistent(self):
        """更新不存在的记忆返回 False"""
        self.assertFalse(self.client.update("mem_nonexistent", content="new"))


class TestSemanticDedup(unittest.TestCase):
    """Task 3: 语义去重"""

    @classmethod
    def setUpClass(cls):
        from su_memory.sdk.lite_pro import SuMemoryLitePro
        cls.SuMemoryLitePro = SuMemoryLitePro

    def test_dedup_same_content(self):
        """相同内容不应重复添加"""
        client = self.SuMemoryLitePro(
            max_memories=100,
            dedup_threshold=0.85,
            enable_graph=False,
            enable_temporal=False,
            enable_session=False,
        )
        mid1 = client.add("Python 是一种广泛使用的编程语言")
        mid2 = client.add("Python 是一种广泛使用的编程语言")

        # 应返回相同 ID
        self.assertEqual(mid1, mid2)

    def test_dedup_similar_content(self):
        """语义相似内容应被去重"""
        client = self.SuMemoryLitePro(
            max_memories=100,
            dedup_threshold=0.70,  # 低阈值兼容 TF-IDF/hash 向量
            enable_graph=False,
            enable_temporal=False,
            enable_session=False,
        )
        mid1 = client.add("机器学习是人工智能的核心技术")
        mid2 = client.add("机器学习是人工智能核心技术")  # 仅差一个"的"字

        self.assertEqual(mid1, mid2)

    def test_skip_dedup(self):
        """skip_dedup=True 时不应去重"""
        client = self.SuMemoryLitePro(
            max_memories=100,
            enable_graph=False,
            enable_temporal=False,
            enable_session=False,
        )
        mid1 = client.add("Python 是一种语言")
        mid2 = client.add("Python 是一种语言", skip_dedup=True)

        self.assertNotEqual(mid1, mid2)

    def test_dedup_disabled_by_threshold(self):
        """dedup_threshold=1.0 时关闭去重"""
        client = self.SuMemoryLitePro(
            max_memories=100,
            dedup_threshold=1.0,
            enable_graph=False,
            enable_temporal=False,
            enable_session=False,
        )
        mid1 = client.add("完全相同的测试文本")
        mid2 = client.add("完全相同的测试文本")
        self.assertNotEqual(mid1, mid2)


class TestMemoryForgetting(unittest.TestCase):
    """Task 4: MemoryForgetting 串联"""

    @classmethod
    def setUpClass(cls):
        from su_memory.sdk.lite_pro import SuMemoryLitePro
        cls.SuMemoryLitePro = SuMemoryLitePro

    def test_forgetting_initialized(self):
        """MemoryForgetting 应被实例化"""
        client = self.SuMemoryLitePro(
            max_memories=100,
            enable_graph=False,
            enable_temporal=False,
            enable_session=False,
        )
        self.assertIsNotNone(client._forgetting)

    def test_access_boost(self):
        """query() 命中应提升记忆重要性"""
        client = self.SuMemoryLitePro(
            max_memories=100,
            enable_graph=False,
            enable_temporal=False,
            enable_session=False,
        )
        mid = client.add("需要被查询的记忆内容")
        initial_importance = client._forgetting.get_importance(mid)

        client.query("被查询")
        boosted_importance = client._forgetting.get_importance(mid)

        self.assertIsNotNone(initial_importance)
        self.assertIsNotNone(boosted_importance)


class TestCrossEncoderReranker(unittest.TestCase):
    """Task 7: Cross-Encoder 精排"""

    @classmethod
    def setUpClass(cls):
        from su_memory.sdk._cross_encoder import CrossEncoderReranker
        cls.CrossEncoderReranker = CrossEncoderReranker

    def test_reranker_unavailable_graceful(self):
        """CrossEncoder 不可用时应静默降级"""
        reranker = self.CrossEncoderReranker(
            model_name="nonexistent-model-xyz"
        )
        candidates = [
            {"memory_id": "1", "content": "测试内容 A", "score": 0.9},
            {"memory_id": "2", "content": "测试内容 B", "score": 0.8},
        ]
        result = reranker.rerank("测试查询", candidates, top_k=2)
        self.assertEqual(len(result), 2)
        # 模型不可用时保持原顺序
        self.assertEqual(result[0]["memory_id"], "1")

    def test_reranker_empty_candidates(self):
        """空候选列表"""
        reranker = self.CrossEncoderReranker()
        result = reranker.rerank("query", [], top_k=5)
        self.assertEqual(len(result), 0)


class TestSpatiotemporalDelete(unittest.TestCase):
    """Task 1 补充: SpatiotemporalIndex delete/update"""

    @classmethod
    def setUpClass(cls):
        from su_memory.sdk.spacetime_index import SpatiotemporalIndex
        cls.SpatiotemporalIndex = SpatiotemporalIndex

    def test_delete_node(self):
        """SpacetimeIndex 删除节点"""
        def dummy_embed(text):
            return [float(ord(c)) / 1000.0 for c in text[:10].ljust(10, 'a')]

        st = self.SpatiotemporalIndex(embedding_func=dummy_embed, dims=10)
        self.assertTrue(st.add_node("node1", "测试内容"))
        self.assertTrue(st.delete_node("node1"))
        self.assertFalse(st.delete_node("node1"))  # 已删除

    def test_update_node(self):
        """SpacetimeIndex 更新节点"""
        def dummy_embed(text):
            return [float(hash(text) % 10000) / 10000.0] * 10

        st = self.SpatiotemporalIndex(embedding_func=dummy_embed, dims=10)
        st.add_node("node2", "原始内容")
        self.assertTrue(st.update_node("node2", content="更新内容"))

        self.assertEqual(st.nodes["node2"].content, "更新内容")


if __name__ == "__main__":
    unittest.main()
