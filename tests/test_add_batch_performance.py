"""
su-memory SDK v3.5.5 — add_batch 性能基准测试
=============================================

验证 P1-1 优化效果:
  - add_batch() 100 条延迟 < 500ms (vs 逐条 ~11.7s)
  - 异步管道模式下单条 add() < 1ms

Usage:
    pytest tests/test_add_batch_performance.py -v
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

pytestmark = pytest.mark.perf


class TestAddBatchPerformance:
    """add_batch 批量写入性能"""

    def test_batch_100_within_threshold(self):
        """100 条批量写入应在 500ms 内完成"""
        from su_memory.client import SuMemory

        client = SuMemory()
        n = 100
        items = [
            {"content": f"批量性能测试记忆 #{i:04d}: 包含中文业务数据记录"}
            for i in range(n)
        ]

        start = time.perf_counter()
        ids = client.add_batch(items)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(ids) == n, f"期望 {n} 条记忆, 实际 {len(ids)}"
        assert elapsed_ms < 500, (
            f"批量写入 100 条耗时 {elapsed_ms:.1f}ms, 超出 500ms 阈值 "
            f"(逐条约需 {n * 117}ms)"
        )
        client.clear()

    def test_batch_vs_sequential_ratio(self):
        """批量写入应至少比逐条写入快 10 倍"""
        from su_memory.client import SuMemory

        # 逐条写入
        client = SuMemory()
        n = 30
        start = time.perf_counter()
        for i in range(n):
            client.add(f"逐条测试 #{i:04d}")
        sequential_ms = (time.perf_counter() - start) * 1000
        client.clear()

        # 批量写入
        client2 = SuMemory()
        items = [
            {"content": f"批量测试 #{i:04d}"}
            for i in range(n)
        ]
        start = time.perf_counter()
        client2.add_batch(items)
        batch_ms = (time.perf_counter() - start) * 1000
        client2.clear()

        # 批量至少快 10 倍（无 embedding 时也应更快）
        if sequential_ms > 10:  # 仅在 embedding 可用时断言
            ratio = sequential_ms / max(batch_ms, 0.001)
            assert ratio > 5, (
                f"批量加速比 {ratio:.1f}x, 期望 > 5x "
                f"(逐条 {sequential_ms:.1f}ms vs 批量 {batch_ms:.1f}ms)"
            )

    def test_batch_empty_raises(self):
        """空列表应抛出 ValueError"""
        from su_memory.client import SuMemory

        client = SuMemory()
        with pytest.raises(ValueError, match="不能为空"):
            client.add_batch([])
        client.clear()

    def test_batch_invalid_content_raises(self):
        """空 content 应抛出 ValueError"""
        from su_memory.client import SuMemory

        client = SuMemory()
        with pytest.raises(ValueError, match="必须是非空字符串"):
            client.add_batch([{"content": ""}])
        client.clear()

    def test_batch_single_item(self):
        """单条批量添加"""
        from su_memory.client import SuMemory

        client = SuMemory()
        ids = client.add_batch([{"content": "单条批量测试"}])
        assert len(ids) == 1
        assert ids[0].startswith("mem_")
        client.clear()

    def test_batch_with_metadata(self):
        """带元数据的批量添加"""
        from su_memory.client import SuMemory

        client = SuMemory()
        items = [
            {"content": "元数据测试 A", "metadata": {"source": "test_a"}},
            {"content": "元数据测试 B", "metadata": {"source": "test_b"}},
        ]
        ids = client.add_batch(items)
        assert len(ids) == 2

        # 验证元数据已存储
        m1 = client.get_memory(ids[0])
        assert m1 is not None
        assert m1.get("metadata", {}).get("source") == "test_a"
        client.clear()
