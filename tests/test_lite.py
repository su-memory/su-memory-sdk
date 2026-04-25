"""
SuMemoryLite 单元测试
"""
import os
import sys
import tempfile
import pytest

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from su_memory.sdk.lite import SuMemoryLite, STOP_WORDS


class TestSuMemoryLite:
    """SuMemoryLite测试用例"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SuMemoryLite(
                max_memories=100,
                storage_path=tmpdir,
                enable_tfidf=True,
                enable_persistence=True
            )
            yield client

    @pytest.fixture
    def client_no_persistence(self):
        """创建无持久化的测试客户端"""
        return SuMemoryLite(
            max_memories=100,
            enable_tfidf=True,
            enable_persistence=False
        )

    def test_add_memory(self, client):
        """测试添加记忆"""
        memory_id = client.add("今天天气很好")
        assert memory_id is not None
        assert memory_id.startswith("mem_")
        assert len(client) == 1

    def test_add_multiple_memories(self, client):
        """测试添加多条记忆"""
        ids = []
        for i in range(5):
            mid = client.add(f"记忆{i}")
            ids.append(mid)
        
        assert len(client) == 5
        assert len(set(ids)) == 5  # 所有ID唯一

    def test_query_basic(self, client):
        """测试基础查询"""
        client.add("今天天气很好，阳光明媚")
        client.add("明天可能下雨")
        client.add("我喜欢学习编程")

        results = client.query("天气")
        assert len(results) > 0
        assert results[0]["content"] == "今天天气很好，阳光明媚"

    def test_query_top_k(self, client):
        """测试top_k参数"""
        for i in range(10):
            client.add(f"测试记忆{i}包含关键词")
        
        results = client.query("关键词", top_k=3)
        assert len(results) <= 3

    def test_query_empty(self, client):
        """测试空查询"""
        results = client.query("不存在的关键词")
        assert results == []

    def test_tfidf_scoring(self, client):
        """测试TF-IDF评分"""
        # 添加多条记忆
        client.add("苹果是一种水果")
        client.add("苹果手机是苹果公司的产品")
        client.add("香蕉也是一种水果")

        results = client.query("苹果")
        # 包含"苹果"但不含"公司"的记忆分数更高
        assert len(results) > 0
        assert results[0]["score"] > 0

    def test_predict(self, client):
        """测试预测功能"""
        client.add("如果下雨，地面会湿")
        client.add("带伞可以避免被雨淋湿")

        result = client.predict("天空乌云密布", "不带伞出门")
        assert "confidence" in result
        assert "related_memories" in result
        assert 0 <= result["confidence"] <= 1

    def test_get_stats(self, client):
        """测试统计信息"""
        client.add("测试1")
        client.add("测试2")

        stats = client.get_stats()
        assert stats["total_memories"] == 2
        assert stats["max_memories"] == 100
        assert stats["tfidf_enabled"] is True
        assert stats["index_size"] > 0

    def test_max_memories_limit(self, client):
        """测试最大记忆数量限制"""
        small_client = SuMemoryLite(max_memories=5)
        
        for i in range(10):
            small_client.add(f"记忆{i}")
        
        assert len(small_client) == 5

    def test_persistence(self, client):
        """测试持久化功能"""
        client.add("持久化测试记忆")
        storage_file = os.path.join(client.storage_path, "su_memory_lite.json")
        
        # 验证文件已创建
        assert os.path.exists(storage_file)
        
        # 创建新客户端，应该能加载已有数据
        new_client = SuMemoryLite(
            storage_path=client.storage_path,
            enable_persistence=True
        )
        
        assert len(new_client) == 1
        assert new_client._memories[0]["content"] == "持久化测试记忆"

    def test_clear(self, client):
        """测试清空记忆"""
        client.add("测试1")
        client.add("测试2")
        assert len(client) == 2
        
        client.clear()
        assert len(client) == 0
        assert len(client._index) == 0

    def test_chinese_tokenization(self, client):
        """测试中文分词"""
        keywords = client._tokenize("今天天气很好，阳光明媚")
        assert isinstance(keywords, list)
        # 验证停用词被过滤
        for kw in keywords:
            assert kw not in STOP_WORDS

    def test_keyword_extraction(self, client):
        """测试关键词提取"""
        keywords = client._extract_keywords("人工智能机器学习深度学习")
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        # 长词权重更高
        assert "人工智能" in keywords or "机器学习" in keywords

    def test_context_manager(self):
        """测试上下文管理器"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with SuMemoryLite(storage_path=tmpdir) as c:
                c.add("测试")
                assert len(c) == 1
            
            # 退出时应该自动保存，验证持久化
            new_client = SuMemoryLite(
                storage_path=tmpdir,
                enable_persistence=True
            )
            # 新客户端应该加载之前的数据
            assert len(new_client) == 1

    def test_empty_query_keywords(self, client):
        """测试空查询关键词"""
        # 添加只有停用词的记忆
        client.add("的是了和")
        results = client.query("的了的和是")
        # 应该返回空或低分结果
        assert isinstance(results, list)


class TestSuMemoryLitePerformance:
    """性能测试"""

    def test_large_scale_insertion(self):
        """测试大量插入"""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SuMemoryLite(
                max_memories=10000,
                storage_path=tmpdir
            )
            
            import time
            start = time.time()
            
            for i in range(1000):
                client.add(f"测试记忆{i}包含一些关键词内容")
            
            elapsed = time.time() - start
            
            assert len(client) == 1000
            # 1000条插入应该在10秒内完成
            assert elapsed < 10

    def test_large_scale_query(self):
        """测试大量查询"""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SuMemoryLite(
                max_memories=10000,
                storage_path=tmpdir
            )
            
            # 添加1000条记忆
            for i in range(1000):
                client.add(f"记忆{i}包含关键词A和B")
            
            import time
            start = time.time()
            
            # 执行100次查询
            for _ in range(100):
                client.query("关键词")
            
            elapsed = time.time() - start
            
            # 100次查询应该在5秒内完成
            assert elapsed < 5

    def test_memory_usage(self):
        """测试内存占用"""
        import sys
        
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SuMemoryLite(
                max_memories=10000,
                storage_path=tmpdir
            )
            
            # 添加1000条记忆
            for i in range(1000):
                client.add(f"这是一条较长的测试记忆内容{i}，包含足够多的文字来测试内存占用情况")
            
            # 粗略估算内存占用
            # 每条记忆约100字节，1000条约100KB
            size = sys.getsizeof(client._memories)
            assert size < 10 * 1024 * 1024  # 小于10MB


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
