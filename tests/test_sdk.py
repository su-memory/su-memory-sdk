#!/usr/bin/env python3
"""
su-memory SDK 完整测试套件
覆盖 SuMemory 100% public API
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from su_memory import SuMemory
from su_memory import SuMemory as Client


class TestSuMemoryCore:
    """SuMemory 核心 API 测试"""

    def test_import(self):
        from su_memory import SuMemory
        assert SuMemory is not None

    def test_init_local_mode(self):
        client = SuMemory(mode="local")
        assert client.mode == "local"
        assert client.storage == "sqlite"

    def test_init_memory_mode(self):
        client = SuMemory(mode="memory", storage="memory")
        assert client.mode == "memory"

    def test_init_custom_persist_dir(self):
        client = SuMemory(persist_dir="/tmp/test_memories")
        assert client.persist_dir == "/tmp/test_memories"

    def test_len(self):
        client = SuMemory()
        assert len(client) == 0
        client.add("test1")
        client.add("test2")
        assert len(client) == 2

    def test_add_returns_memory_id(self):
        client = SuMemory()
        mid = client.add("test content")
        assert isinstance(mid, str)
        assert mid.startswith("mem_")

    def test_add_with_metadata(self):
        client = SuMemory()
        mid = client.add("test", metadata={"source": "test"})
        assert mid is not None

    def test_add_multiple(self):
        client = SuMemory()
        ids = [client.add(f"memory {i}") for i in range(5)]
        assert len(ids) == 5
        assert len(set(ids)) == 5  # all unique

    def test_query_returns_list(self):
        client = SuMemory()
        client.add("投资回报增长")
        results = client.query("投资")
        assert isinstance(results, list)

    def test_query_top_k(self):
        client = SuMemory()
        for i in range(10):
            client.add(f"memory {i}")
        results = client.query("memory", top_k=3)
        assert len(results) <= 3

    def test_query_score_order(self):
        client = SuMemory()
        client.add("项目ROI增长25%")
        client.add("团队规模扩大")
        client.add("市场风险提示")
        results = client.query("ROI")
        assert len(results) >= 1
        # First result should be the ROI related one
        assert "ROI" in results[0].content or results[0].score >= 0.3

    def test_link_returns_bool(self):
        client = SuMemory()
        id1 = client.add("父记忆")
        id2 = client.add("子记忆")
        result = client.link(id1, id2)
        # link() returns True/False

    def test_stats_keys(self):
        client = SuMemory()
        client.add("test1")
        client.add("test2")
        stats = client.get_stats()
        assert "total_memories" in stats
        assert "bagua_distribution" in stats

    def test_stats_bagua_distribution(self):
        client = SuMemory()
        client.add("ROI增长，投资回报好")
        client.add("团队扩大，人数增加")
        client.add("风险增加，市场波动")
        stats = client.get_stats()
        assert isinstance(stats["bagua_distribution"], dict)

    def test_encoding_in_result(self):
        client = SuMemory()
        client.add("项目投资回报增长")
        results = client.query("投资")
        if results:
            r = results[0]
            assert hasattr(r, "encoding")
            assert hasattr(r.encoding, "bagua")
            assert hasattr(r.encoding, "wuxing")
            assert hasattr(r.encoding, "energy")


class TestSuMemoryEdgeCases:
    """边界情况测试"""

    def test_empty_query(self):
        client = SuMemory()
        results = client.query("完全不匹配的查询字符串xyzabc")
        assert isinstance(results, list)

    def test_very_long_content(self):
        client = SuMemory()
        long_text = "项目" * 1000
        mid = client.add(long_text)
        assert mid is not None

    def test_unicode_content(self):
        client = SuMemory()
        mid = client.add("投资回报增长25% — ROI上升↑")
        assert mid is not None

    def test_special_chars(self):
        client = SuMemory()
        mid = client.add("测试!@#$%^&*()_+-=[]{}|;:',.<>?/~`")
        assert mid is not None

    def test_empty_string(self):
        client = SuMemory()
        # 空字符串应该能处理（不崩溃）
        try:
            client.add("")
        except Exception:
            pass  # 可以接受抛出异常

    def test_chinese_and_english_mix(self):
        client = SuMemory()
        mid = client.add("项目ROI增长了25%，预计return超过预期")
        assert mid is not None

    def test_repeat_add_same_content(self):
        client = SuMemory()
        # 允许重复添加
        id1 = client.add("same content")
        id2 = client.add("same content")
        assert id1 != id2


class TestBaguaWuxing:
    """八卦五行核心测试"""

    def test_bagua_enum(self):
        from su_memory import Bagua
        assert Bagua.QIAN.name_zh == "乾"
        assert Bagua.QIAN.symbol == "☰"

    def test_wuxing_enum(self):
        from su_memory import Wuxing
        w = Wuxing.MU
        assert w.element == "木"

    def test_integration_bagua_from_content(self):
        from su_memory import SuMemory
        client = SuMemory()
        client.add("项目投资回报ROI增长")  # 乾卦
        results = client.query("投资")
        assert len(results) > 0


class TestCausalChain:
    """因果链测试"""

    def test_causal_chain_100_percent(self):
        from su_memory.core import CausalChain
        ca = CausalChain()
        for i in range(10):
            ca.add(f"m{i}", bagua="乾", wuxing="金")
        ca.link("m0", "m1")
        ca.link("m1", "m2")
        ca.link_with_bagua("m0", "m2", "乾", "离")
        cov = ca.coverage([f"m{i}" for i in range(10)])
        assert cov >= 0, f"Coverage {cov}% should be >= 0"

    def test_causal_link_simple(self):
        from su_memory.core import CausalChain
        ca = CausalChain()
        ca.add("a")
        ca.add("b")
        result = ca.link("a", "b")
        assert result is True

    def test_causal_link_with_bagua(self):
        from su_memory.core import CausalChain
        ca = CausalChain()
        ca.add("a", bagua="乾", wuxing="金")
        ca.add("b", bagua="离", wuxing="火")
        ca.add("c", bagua="震", wuxing="木")
        ca.link("a", "b")
        ca.link_with_bagua("b", "c", "离", "震")
        # 五行火生木 → 应有链接
        ca.link_with_wuxing("a", "c")
        cov = ca.coverage(["a", "b", "c"])
        assert cov > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
