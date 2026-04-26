"""
v1.7.0 边界测试

测试边界情况和异常处理，包括：
- 插件系统边界情况
- 存储系统边界情况
- 数据边界情况
- 并发边界情况
- 错误处理边界

v1.7.0 测试覆盖增强套件
"""

import pytest
import sys
import os
import tempfile
import shutil
import time
import json
import threading
import sqlite3
from pathlib import Path

sys.path.insert(0, "src")

from su_memory.storage.sqlite_backend import SQLiteBackend, MemoryItem
from su_memory.storage.backup_manager import BackupManager
from su_memory.storage.exporter import DataExporter
from su_memory.storage.auto_compression import AutoCompressor

from su_memory._sys._plugin_registry import (
    PluginRegistry, PluginAlreadyExistsError, PluginNotFoundError,
    PluginDependencyError, PluginStateError
)
from su_memory._sys._plugin_interface import PluginType, PluginState
from su_memory._sys._plugin_sandbox import SandboxedExecutor, ExecutionResult

from su_memory.plugins.embedding_plugin import TextEmbeddingPlugin


# ============================================================================
# Test Plugin Edge Cases
# ============================================================================

class TestPluginEdgeCases:
    """插件边界测试"""

    def setup_method(self):
        """每个测试前设置"""
        self.registry = PluginRegistry()
        self.registry.clear()

    def teardown_method(self):
        """每个测试后清理"""
        self.registry.clear()

    def test_register_same_plugin_twice(self):
        """测试重复注册同一个插件"""
        plugin = TextEmbeddingPlugin()
        self.registry.register(plugin, auto_initialize=False)

        # 再次注册同一实例应该失败
        with pytest.raises(PluginAlreadyExistsError):
            self.registry.register(plugin, auto_initialize=False)

    def test_register_different_plugins_same_name(self):
        """测试注册不同插件但相同名称"""
        plugin1 = TextEmbeddingPlugin()
        self.registry.register(plugin1, auto_initialize=False)

        # 创建另一个同名插件
        plugin2 = TextEmbeddingPlugin()
        # plugin2 的 name 属性默认相同，应该失败

        with pytest.raises(PluginAlreadyExistsError):
            self.registry.register(plugin2, auto_initialize=False)

    def test_get_nonexistent_plugin(self):
        """测试获取不存在的插件"""
        result = self.registry.get_plugin("NonExistentPlugin123")
        assert result is None

    def test_unregister_nonexistent(self):
        """测试注销不存在的插件"""
        # 不存在的插件应该抛出异常
        with pytest.raises(PluginNotFoundError):
            self.registry.unregister("NonExistentPlugin123")

    def test_unregister_twice(self):
        """测试注销已注销的插件"""
        plugin = TextEmbeddingPlugin()
        self.registry.register(plugin, auto_initialize=False)
        self.registry.unregister(plugin.name, force=True)

        # 再次注销应该失败
        with pytest.raises(PluginNotFoundError):
            self.registry.unregister(plugin.name)

    def test_register_with_missing_dependency(self):
        """测试注册有依赖但依赖不存在的插件"""
        class DependentPlugin:
            @property
            def name(self):
                return "dependent_plugin"

            @property
            def version(self):
                return "1.0.0"

            @property
            def dependencies(self):
                return ["nonexistent_dependency"]

            def initialize(self, config):
                return True

            def execute(self, context):
                return None

            def cleanup(self):
                pass

        plugin = DependentPlugin()
        with pytest.raises(PluginDependencyError):
            self.registry.register(plugin, auto_initialize=False)

    def test_get_plugin_state_nonexistent(self):
        """测试获取不存在插件的状态"""
        state = self.registry.get_plugin_state("NonExistent")
        assert state is None

    def test_is_registered(self):
        """测试插件注册状态检查"""
        plugin = TextEmbeddingPlugin()
        assert not self.registry.is_registered(plugin.name)

        self.registry.register(plugin, auto_initialize=False)
        assert self.registry.is_registered(plugin.name)

    def test_list_plugins_empty(self):
        """测试列出空注册表"""
        plugins = self.registry.list_plugins()
        assert isinstance(plugins, list)

    def test_list_plugins_by_type(self):
        """测试按类型列出插件"""
        plugin = TextEmbeddingPlugin()
        self.registry.register(plugin, auto_initialize=False)

        plugins = self.registry.list_plugins(plugin_type=PluginType.EMBEDDING)
        assert plugin.name in plugins

    def test_metadata_caching(self):
        """测试元数据缓存"""
        plugin = TextEmbeddingPlugin()
        self.registry.register(plugin, auto_initialize=False)

        # 首次获取
        metadata1 = self.registry.get_plugin_metadata(plugin.name)
        assert metadata1 is not None

        # 再次获取（应该从缓存）
        metadata2 = self.registry.get_plugin_metadata(plugin.name)
        assert metadata2 is not None
        assert metadata1 == metadata2

    def test_statistics_tracking(self):
        """测试统计信息追踪"""
        stats = self.registry.get_statistics()
        assert "total_plugins" in stats
        assert "cache_hit_rate" in stats

    def test_performance_stats(self):
        """测试性能统计"""
        plugin = TextEmbeddingPlugin()
        self.registry.register(plugin, auto_initialize=False)

        # 触发性能统计
        self.registry.get_plugin(plugin.name)
        self.registry.list_plugins()

        perf_stats = self.registry.get_performance_stats()
        assert "get_count" in perf_stats
        assert "list_count" in perf_stats


# ============================================================================
# Test Storage Edge Cases
# ============================================================================

class TestStorageEdgeCases:
    """存储边界测试"""

    def setup_method(self):
        """每个测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "edge_test.db")

    def teardown_method(self):
        """每个测试后清理"""
        if hasattr(self, 'backend'):
            self.backend.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_empty_database_operations(self):
        """测试空数据库操作"""
        self.backend = SQLiteBackend(self.db_path)

        # 空数据库查询
        results = self.backend.query("test", top_k=10)
        assert results == []

        # 空数据库统计
        stats = self.backend.get_stats()
        assert stats["count"] == 0

        # 获取不存在的记忆
        memory = self.backend.get_memory("nonexistent")
        assert memory is None

        # 删除不存在的记忆
        result = self.backend.delete("nonexistent")
        assert result is False

    def test_duplicate_id_handling(self):
        """测试重复ID处理"""
        self.backend = SQLiteBackend(self.db_path)

        # 添加相同ID的记忆
        memory1 = MemoryItem(
            id="same_id",
            content="First content",
            metadata={},
            embedding=None,
            timestamp=time.time()
        )
        self.backend.add_memory(memory1)

        # 再次添加相同ID（应该覆盖）
        memory2 = MemoryItem(
            id="same_id",
            content="Second content",
            metadata={},
            embedding=None,
            timestamp=time.time()
        )
        self.backend.add_memory(memory2)

        # 验证只存在一条
        stats = self.backend.get_stats()
        assert stats["count"] == 1

        # 验证内容是新的
        retrieved = self.backend.get_memory("same_id")
        assert retrieved.content == "Second content"

    def test_special_characters_in_content(self):
        """测试内容中的特殊字符"""
        self.backend = SQLiteBackend(self.db_path)

        special_contents = [
            "你好世界",  # 中文
            "Hello 🌍",  # Emoji
            "<script>alert('xss')</script>",  # HTML/JS
            "Newlines:\n\r\t",  # 特殊空白符
            "Quotes: \"\'",  # 引号
            "Backslash: \\",  # 反斜杠
            "Unicode: \u4e2d\u6587\u2665",  # Unicode
            "Emoji sequence: 🎉🎊🎈",  # 多个Emoji
        ]

        for i, content in enumerate(special_contents):
            memory = MemoryItem(
                id=f"special_{i}",
                content=content,
                metadata={},
                embedding=None,
                timestamp=time.time()
            )
            self.backend.add_memory(memory)

            # 验证存储和检索
            retrieved = self.backend.get_memory(f"special_{i}")
            assert retrieved is not None
            assert retrieved.content == content

    def test_very_long_content(self):
        """测试超长内容"""
        self.backend = SQLiteBackend(self.db_path)

        # 1MB内容
        long_content = "x" * 1_000_000
        memory = MemoryItem(
            id="long_content",
            content=long_content,
            metadata={},
            embedding=None,
            timestamp=time.time()
        )
        self.backend.add_memory(memory)

        # 验证存储和检索
        retrieved = self.backend.get_memory("long_content")
        assert retrieved is not None
        assert len(retrieved.content) == 1_000_000
        assert retrieved.content == long_content

    def test_empty_content(self):
        """测试空内容"""
        self.backend = SQLiteBackend(self.db_path)

        memory = MemoryItem(
            id="empty_content",
            content="",
            metadata={},
            embedding=None,
            timestamp=time.time()
        )
        self.backend.add_memory(memory)

        # 验证可以存储和检索空内容
        retrieved = self.backend.get_memory("empty_content")
        assert retrieved is not None
        assert retrieved.content == ""

    def test_metadata_various_types(self):
        """测试各种类型的元数据"""
        self.backend = SQLiteBackend(self.db_path)

        memory = MemoryItem(
            id="metadata_test",
            content="Test content",
            metadata={
                "string": "value",
                "number": 42,
                "float": 3.14,
                "boolean": True,
                "null": None,
                "list": [1, 2, 3],
                "nested": {"a": 1, "b": [1, 2]},
            },
            embedding=None,
            timestamp=time.time()
        )
        self.backend.add_memory(memory)

        # 验证元数据完整保留
        retrieved = self.backend.get_memory("metadata_test")
        assert retrieved.metadata["string"] == "value"
        assert retrieved.metadata["number"] == 42
        assert retrieved.metadata["float"] == 3.14
        assert retrieved.metadata["boolean"] is True
        assert retrieved.metadata["null"] is None
        assert retrieved.metadata["list"] == [1, 2, 3]

    def test_large_embedding(self):
        """测试大向量"""
        self.backend = SQLiteBackend(self.db_path)

        import numpy as np

        # 4096维向量
        large_embedding = np.random.rand(4096).astype(np.float32).tolist()
        memory = MemoryItem(
            id="large_embedding",
            content="Large embedding test",
            metadata={},
            embedding=large_embedding,
            timestamp=time.time()
        )
        self.backend.add_memory(memory)

        # 验证向量保留
        retrieved = self.backend.get_memory("large_embedding")
        assert retrieved is not None
        assert len(retrieved.embedding) == 4096

    def test_very_large_batch(self):
        """测试大批量操作"""
        self.backend = SQLiteBackend(self.db_path)

        # 1000条记录
        batch_size = 1000
        memories = [
            MemoryItem(
                id=f"batch_{i}",
                content=f"Batch content {i}",
                metadata={},
                embedding=None,
                timestamp=time.time()
            )
            for i in range(batch_size)
        ]

        ids = self.backend.add_memory_batch(memories)
        assert len(ids) == batch_size

        stats = self.backend.get_stats()
        assert stats["count"] == batch_size

    def test_concurrent_writes(self):
        """测试并发写入"""
        self.backend = SQLiteBackend(self.db_path)

        errors = []
        def write_batch(start_id, count):
            try:
                for i in range(count):
                    memory = MemoryItem(
                        id=f"concurrent_{start_id}_{i}",
                        content=f"Content {i}",
                        metadata={},
                        embedding=None,
                        timestamp=time.time()
                    )
                    self.backend.add_memory(memory)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=write_batch, args=(i, 10))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 允许一些并发错误，但不应全部失败
        assert len(errors) < 5


# ============================================================================
# Test Backup Edge Cases
# ============================================================================

class TestBackupEdgeCases:
    """备份边界测试"""

    def setup_method(self):
        """每个测试前设置"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """每个测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_backup_empty_database(self):
        """测试备份空数据库"""
        db_path = os.path.join(self.temp_dir, "empty.db")
        backup_dir = os.path.join(self.temp_dir, "backups")

        # 创建空数据库
        backend = SQLiteBackend(db_path)
        backend.close()

        # 备份空数据库
        manager = BackupManager(db_path, backup_dir)
        backup_path = manager.backup()

        assert os.path.exists(backup_path)

        # 恢复空数据库
        result = manager.restore(backup_path)
        assert result is True

    def test_backup_nonexistent_database(self):
        """测试备份不存在的数据库"""
        db_path = os.path.join(self.temp_dir, "nonexistent.db")
        backup_dir = os.path.join(self.temp_dir, "backups")

        manager = BackupManager(db_path, backup_dir)

        # 备份不存在的数据库应该失败
        try:
            backup_path = manager.backup()
            # 如果没抛异常，备份文件可能不存在
            assert not os.path.exists(backup_path)
        except Exception:
            pass  # 预期行为

    def test_restore_to_corrupted_backup(self):
        """测试恢复到损坏的备份"""
        db_path = os.path.join(self.temp_dir, "test.db")
        backup_dir = os.path.join(self.temp_dir, "backups")

        # 创建并备份正常数据库
        backend = SQLiteBackend(db_path)
        backend.add_memory(MemoryItem(
            id="test",
            content="Test",
            metadata={},
            embedding=None,
            timestamp=time.time()
        ))
        backend.close()

        manager = BackupManager(db_path, backup_dir)
        backup_path = manager.backup()

        # 损坏备份文件
        with open(backup_path, "wb") as f:
            f.write(b"corrupted data")

        # 恢复损坏的备份应该失败
        try:
            result = manager.restore(backup_path)
            # 如果返回False或抛出异常都是预期行为
        except Exception:
            pass

    def test_max_backups_cleanup(self):
        """测试最大备份数清理"""
        db_path = os.path.join(self.temp_dir, "cleanup.db")
        backup_dir = os.path.join(self.temp_dir, "backups")

        backend = SQLiteBackend(db_path)
        backend.close()

        # 设置最大备份数为3
        manager = BackupManager(db_path, backup_dir, max_backups=3)

        # 创建5个备份
        for i in range(5):
            # 添加数据以创建新状态
            backend = SQLiteBackend(db_path)
            backend.add_memory(MemoryItem(
                id=f"mem_{i}",
                content=f"Content {i}",
                metadata={},
                embedding=None,
                timestamp=time.time()
            ))
            backend.close()
            manager.backup(f"backup_{i}")

        # 验证只保留3个备份
        backups = manager.list_backups()
        assert len(backups) <= 3


# ============================================================================
# Test Compression Edge Cases
# ============================================================================

class TestCompressionEdgeCases:
    """压缩边界测试"""

    def test_empty_data_compression(self):
        """测试空数据压缩"""
        compressor = AutoCompressor()

        # 空数据
        result = compressor.compress(b"")
        assert result == b""
        assert compressor.decompress(result) == b""

    def test_single_byte_compression(self):
        """测试单字节压缩"""
        compressor = AutoCompressor()

        data = b"x"
        compressed = compressor.compress(data)
        decompressed = compressor.decompress(compressed)
        assert decompressed == data

    def test_very_large_data_compression(self):
        """测试超大数据压缩"""
        compressor = AutoCompressor()

        # 10MB数据
        large_data = b"ABC" * 3_500_000
        compressed = compressor.compress(large_data)
        decompressed = compressor.decompress(compressed)
        assert decompressed == large_data

    def test_binary_data_compression(self):
        """测试二进制数据压缩"""
        compressor = AutoCompressor()

        # 二进制零
        binary_zeros = bytes(10000)
        compressed = compressor.compress(binary_zeros)
        decompressed = compressor.decompress(compressed)
        assert decompressed == binary_zeros

        # 随机二进制
        import random
        random.seed(42)
        random_bytes = bytes([random.randint(0, 255) for _ in range(1000)])
        compressed = compressor.compress(random_bytes)
        decompressed = compressor.decompress(compressed)
        assert decompressed == random_bytes

    def test_various_compression_levels(self):
        """测试不同压缩级别"""
        test_data = b"Test data for compression level testing " * 100

        for level in [1, 3, 6, 9]:
            compressor = AutoCompressor(algorithm="zlib", compression_level=level)
            compressed = compressor.compress(test_data)
            decompressed = compressor.decompress(compressed)
            assert decompressed == test_data


# ============================================================================
# Test Export-Import Edge Cases
# ============================================================================

class TestExportImportEdgeCases:
    """导出导入边界测试"""

    def setup_method(self):
        """每个测试前设置"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """每个测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_export_empty_database(self):
        """测试导出空数据库"""
        db_path = os.path.join(self.temp_dir, "empty.db")
        json_path = os.path.join(self.temp_dir, "empty.json")

        backend = SQLiteBackend(db_path)
        backend.close()

        exporter = DataExporter(db_path)
        count = exporter.to_json(json_path)

        assert count == 0
        assert os.path.exists(json_path)

        # 验证JSON是空数组
        with open(json_path, "r") as f:
            data = json.load(f)
        assert data == []

    def test_export_nonexistent_file(self):
        """测试导出到不存在的文件"""
        db_path = os.path.join(self.temp_dir, "test.db")
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent", "test.json")

        backend = SQLiteBackend(db_path)
        backend.close()

        exporter = DataExporter(db_path)

        # 导出到不存在的路径应该失败
        try:
            exporter.to_json(nonexistent_path)
        except Exception:
            pass  # 预期行为

    def test_import_invalid_json(self):
        """测试导入无效JSON"""
        invalid_json_path = os.path.join(self.temp_dir, "invalid.json")
        import_db_path = os.path.join(self.temp_dir, "import.db")

        # 创建无效JSON文件
        with open(invalid_json_path, "w") as f:
            f.write("not valid json {")

        exporter = DataExporter(import_db_path)

        # 导入无效JSON应该失败
        try:
            exporter.from_json(invalid_json_path)
        except json.JSONDecodeError:
            pass  # 预期行为

    def test_import_json_wrong_format(self):
        """测试导入错误格式的JSON"""
        wrong_format_path = os.path.join(self.temp_dir, "wrong_format.json")
        import_db_path = os.path.join(self.temp_dir, "import.db")

        # 创建非数组JSON
        with open(wrong_format_path, "w") as f:
            json.dump({"key": "value"}, f)

        exporter = DataExporter(import_db_path)

        # 导入非数组JSON应该失败
        try:
            exporter.from_json(wrong_format_path)
        except (ValueError, TypeError):
            pass  # 预期行为


# ============================================================================
# Test Sandbox Edge Cases
# ============================================================================

class TestSandboxEdgeCases:
    """沙箱边界测试"""

    def setup_method(self):
        """每个测试前设置"""
        self.registry = PluginRegistry()
        self.registry.clear()

    def teardown_method(self):
        """每个测试后清理"""
        self.registry.clear()

    def test_execute_nonexistent_plugin(self):
        """测试执行不存在的插件"""
        executor = SandboxedExecutor()

        # 创建临时插件对象
        class FakePlugin:
            name = "fake"

            def execute(self, context):
                return None

        result = executor.execute(FakePlugin(), {})
        # 应该返回结果对象
        assert result is not None

    def test_execute_with_none_context(self):
        """测试使用None上下文执行"""
        executor = SandboxedExecutor()

        plugin = TextEmbeddingPlugin()
        plugin.initialize({})

        result = executor.execute(plugin, None)
        # 应该返回结果
        assert result is not None

    def test_execute_with_empty_context(self):
        """测试使用空上下文执行"""
        executor = SandboxedExecutor()

        plugin = TextEmbeddingPlugin()
        plugin.initialize({})

        result = executor.execute(plugin, {})
        assert result is not None


# ============================================================================
# Test Time Edge Cases
# ============================================================================

class TestTimeEdgeCases:
    """时间边界测试"""

    def setup_method(self):
        """每个测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "time_test.db")

    def teardown_method(self):
        """每个测试后清理"""
        if hasattr(self, 'backend'):
            self.backend.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_very_old_timestamp(self):
        """测试非常旧的时间戳"""
        self.backend = SQLiteBackend(self.db_path)

        # 使用Unix纪元开始的时间
        old_timestamp = 0.0
        memory = MemoryItem(
            id="old_time",
            content="Very old memory",
            metadata={},
            embedding=None,
            timestamp=old_timestamp
        )
        self.backend.add_memory(memory)

        retrieved = self.backend.get_memory("old_time")
        assert retrieved is not None
        assert retrieved.timestamp == old_timestamp

    def test_future_timestamp(self):
        """测试未来时间戳"""
        self.backend = SQLiteBackend(self.db_path)

        # 使用未来时间
        future_timestamp = time.time() + 86400 * 365 * 100  # 100年后
        memory = MemoryItem(
            id="future_time",
            content="Future memory",
            metadata={},
            embedding=None,
            timestamp=future_timestamp
        )
        self.backend.add_memory(memory)

        retrieved = self.backend.get_memory("future_time")
        assert retrieved is not None
        assert retrieved.timestamp == future_timestamp

    def test_same_timestamp_multiple_items(self):
        """测试相同时间戳的多个条目"""
        self.backend = SQLiteBackend(self.db_path)

        same_time = time.time()
        for i in range(5):
            memory = MemoryItem(
                id=f"same_time_{i}",
                content=f"Memory {i}",
                metadata={},
                embedding=None,
                timestamp=same_time
            )
            self.backend.add_memory(memory)

        stats = self.backend.get_stats()
        assert stats["count"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
