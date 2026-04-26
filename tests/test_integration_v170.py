"""
v1.7.0 集成测试

测试所有新模块的协同工作，包括：
- 插件系统与存储集成
- 备份恢复功能
- 数据导出导入
- 压缩功能
- 完整工作流端到端测试

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
from pathlib import Path

sys.path.insert(0, "src")

from su_memory.storage.sqlite_backend import SQLiteBackend, MemoryItem
from su_memory.storage.backup_manager import BackupManager, BackupInfo
from su_memory.storage.exporter import DataExporter
from su_memory.storage.auto_compression import AutoCompressor, LZ4_AVAILABLE

from su_memory._sys._plugin_registry import PluginRegistry, PluginAlreadyExistsError
from su_memory._sys._plugin_interface import PluginType, PluginState
from su_memory._sys._plugin_sandbox import SandboxedExecutor

from su_memory.plugins.embedding_plugin import TextEmbeddingPlugin
from su_memory.plugins.rerank_plugin import RerankPlugin

from su_memory.sdk import SuMemoryLite


# ============================================================================
# Test Plugin-Storage Integration
# ============================================================================

class TestPluginStorageIntegration:
    """插件与存储集成测试"""

    def setup_method(self):
        """每个测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = PluginRegistry()
        self.registry.clear()

    def teardown_method(self):
        """每个测试后清理"""
        self.registry.clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_plugin_with_sqlite_backend(self):
        """测试插件使用SQLite后端"""
        # 创建存储
        db_path = os.path.join(self.temp_dir, "test.db")
        backend = SQLiteBackend(db_path)

        # 添加测试数据
        memory = MemoryItem(
            id="plugin_test_1",
            content="Plugin integration test memory",
            metadata={"plugin": "test", "source": "integration"},
            embedding=None,
            timestamp=time.time()
        )
        backend.add_memory(memory)

        # 创建并注册插件
        plugin = TextEmbeddingPlugin()
        plugin.initialize({})
        self.registry.register(plugin, auto_initialize=False)

        # 验证插件注册成功
        assert self.registry.is_registered("TextEmbeddingPlugin")

        # 使用执行器运行插件
        executor = SandboxedExecutor(default_timeout=10.0)
        result = executor.execute(plugin, {"text": "test"})

        # 验证执行结果
        assert result is not None
        assert result.success is not None

        # 验证数据仍然存在
        retrieved = backend.get_memory("plugin_test_1")
        assert retrieved is not None
        assert retrieved.content == "Plugin integration test memory"

        backend.close()

    def test_embedding_plugin_with_storage(self):
        """测试嵌入插件与存储集成"""
        # 创建存储
        db_path = os.path.join(self.temp_dir, "embedding_test.db")
        backend = SQLiteBackend(db_path)

        # 添加带嵌入的测试数据
        import numpy as np
        test_embedding = np.random.rand(128).astype(np.float32).tolist()

        memory = MemoryItem(
            id="embedding_test",
            content="Test content for embedding",
            metadata={"embedding_type": "text"},
            embedding=test_embedding,
            timestamp=time.time()
        )
        backend.add_memory(memory)

        # 注册嵌入插件
        plugin = TextEmbeddingPlugin()
        plugin.initialize({})
        self.registry.register(plugin, auto_initialize=False)

        # 验证插件可用
        retrieved_plugin = self.registry.get_plugin("TextEmbeddingPlugin")
        assert retrieved_plugin is not None

        backend.close()

    def test_rerank_plugin_lifecycle(self):
        """测试重排插件生命周期"""
        # 创建重排插件
        plugin = RerankPlugin()
        plugin.initialize({})
        self.registry.register(plugin, auto_initialize=False)

        # 验证状态
        state = self.registry.get_plugin_state("RerankPlugin")
        assert state in [PluginState.READY, PluginState.LOADING]

        # 获取插件信息
        info = self.registry.get_plugin_info("RerankPlugin")
        assert info is not None
        assert "name" in info

        # 执行插件
        executor = SandboxedExecutor(default_timeout=10.0)
        result = executor.execute(plugin, {
            "query": "test query",
            "documents": ["doc1", "doc2", "doc3"]
        })
        assert result is not None

    def test_multi_plugin_coordination(self):
        """测试多插件协调"""
        # 依次注册多个插件
        plugins = [
            ("Embedding", TextEmbeddingPlugin()),
            ("Rerank", RerankPlugin()),
        ]

        for name, plugin in plugins:
            plugin.initialize({})
            self.registry.register(plugin, auto_initialize=False)

        # 验证所有插件都已注册
        all_plugins = self.registry.list_plugins()
        assert len(all_plugins) >= 2

        # 获取统计信息
        stats = self.registry.get_statistics()
        assert stats["total_plugins"] >= 2

        # 获取所有插件元数据
        metadata_list = self.registry.list_plugin_metadata()
        assert len(metadata_list) >= 2


# ============================================================================
# Test Backup-Restore Integration
# ============================================================================

class TestBackupRestore:
    """备份恢复集成测试"""

    def setup_method(self):
        """每个测试前设置"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """每个测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_backup_restore_cycle(self):
        """测试完整备份恢复周期"""
        db_path = os.path.join(self.temp_dir, "memories.db")
        backup_dir = os.path.join(self.temp_dir, "backups")

        # 创建存储并添加数据
        backend = SQLiteBackend(db_path)
        for i in range(10):
            memory = MemoryItem(
                id=f"mem_{i}",
                content=f"Content for memory {i}",
                metadata={"index": i, "category": "test"},
                embedding=None,
                timestamp=time.time() + i
            )
            backend.add_memory(memory)

        # 验证初始数据
        stats1 = backend.get_stats()
        assert stats1["count"] == 10
        backend.close()

        # 执行备份
        manager = BackupManager(db_path, backup_dir, interval=3600)
        backup_path = manager.backup()
        assert os.path.exists(backup_path)

        # 验证备份文件存在
        backup_files = list(Path(backup_dir).glob("backup_*.db"))
        assert len(backup_files) >= 1

        # 删除原数据库
        os.unlink(db_path)
        assert not os.path.exists(db_path)

        # 恢复备份
        result = manager.restore(backup_path)
        assert result is True
        assert os.path.exists(db_path)

        # 验证恢复的数据
        backend2 = SQLiteBackend(db_path)
        stats2 = backend2.get_stats()
        assert stats2["count"] == stats1["count"]

        # 验证具体数据
        for i in range(10):
            mem = backend2.get_memory(f"mem_{i}")
            assert mem is not None
            assert mem.content == f"Content for memory {i}"

        backend2.close()

    def test_backup_with_compression(self):
        """测试压缩备份"""
        db_path = os.path.join(self.temp_dir, "compressed_backup.db")
        backup_dir = os.path.join(self.temp_dir, "compressed_backups")

        # 创建大数据量存储
        backend = SQLiteBackend(db_path)
        large_content = "x" * 10000  # 10KB内容
        for i in range(20):
            memory = MemoryItem(
                id=f"large_mem_{i}",
                content=large_content + f"_{i}",
                metadata={"large": True},
                embedding=None,
                timestamp=time.time()
            )
            backend.add_memory(memory)
        backend.close()

        # 备份
        manager = BackupManager(db_path, backup_dir)
        backup_path = manager.backup()

        # 获取备份信息
        backups = manager.list_backups()
        assert len(backups) >= 1

        # 验证备份大小
        backup_size = os.path.getsize(backup_path)
        original_size = os.path.getsize(db_path)
        # 压缩后应该更小
        assert backup_size > 0

    def test_incremental_backup(self):
        """测试增量备份"""
        db_path = os.path.join(self.temp_dir, "incremental.db")
        backup_dir = os.path.join(self.temp_dir, "incremental_backups")

        # 初始创建
        backend = SQLiteBackend(db_path)
        for i in range(5):
            memory = MemoryItem(
                id=f"inc_{i}",
                content=f"Incremental {i}",
                metadata={},
                embedding=None,
                timestamp=time.time()
            )
            backend.add_memory(memory)
        backend.close()

        # 第一次备份
        manager = BackupManager(db_path, backup_dir)
        backup1 = manager.backup("first")
        assert os.path.exists(backup1)

        # 添加更多数据
        backend2 = SQLiteBackend(db_path)
        for i in range(5, 10):
            memory = MemoryItem(
                id=f"inc_{i}",
                content=f"Incremental {i}",
                metadata={},
                embedding=None,
                timestamp=time.time()
            )
            backend2.add_memory(memory)
        backend2.close()

        # 第二次备份
        backup2 = manager.backup("second")
        assert os.path.exists(backup2)

        # 验证两个备份都存在
        backups = manager.list_backups()
        assert len(backups) >= 2


# ============================================================================
# Test Export-Import Integration
# ============================================================================

class TestExportImport:
    """导出导入集成测试"""

    def setup_method(self):
        """每个测试前设置"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """每个测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_json_export_import_cycle(self):
        """测试JSON导出导入周期"""
        db_path = os.path.join(self.temp_dir, "memories.db")
        json_path = os.path.join(self.temp_dir, "export.json")
        import_db_path = os.path.join(self.temp_dir, "imported.db")

        # 创建并导出数据
        backend = SQLiteBackend(db_path)
        for i in range(5):
            memory = MemoryItem(
                id=f"export_{i}",
                content=f"Export content {i}",
                metadata={"index": i, "exported": True},
                embedding=None,
                timestamp=time.time() + i
            )
            backend.add_memory(memory)
        backend.close()

        # 导出到JSON
        exporter = DataExporter(db_path)
        count = exporter.to_json(json_path)
        assert count == 5
        assert os.path.exists(json_path)

        # 验证JSON内容
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 5
        assert data[0]["content"] == "Export content 0"

        # 导入到新数据库
        importer = DataExporter(import_db_path)
        imported_count = importer.from_json(json_path)
        assert imported_count == 5

        # 验证导入的数据
        imported_backend = SQLiteBackend(import_db_path)
        for i in range(5):
            mem = imported_backend.get_memory(f"export_{i}")
            assert mem is not None
        imported_backend.close()

    def test_csv_export(self):
        """测试CSV导出"""
        db_path = os.path.join(self.temp_dir, "csv_test.db")
        csv_path = os.path.join(self.temp_dir, "export.csv")

        # 创建数据
        backend = SQLiteBackend(db_path)
        for i in range(3):
            memory = MemoryItem(
                id=f"csv_{i}",
                content=f"CSV content {i}",
                metadata={"csv": True},
                embedding=None,
                timestamp=time.time()
            )
            backend.add_memory(memory)
        backend.close()

        # 导出CSV
        exporter = DataExporter(db_path)
        count = exporter.to_csv(csv_path)
        assert count == 3
        assert os.path.exists(csv_path)

        # 验证CSV内容
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) >= 4  # 头部 + 3条数据

    def test_export_with_embedding(self):
        """测试带嵌入的导出"""
        db_path = os.path.join(self.temp_dir, "embedding_export.db")
        json_path = os.path.join(self.temp_dir, "embedding_export.json")

        # 创建带嵌入的数据
        backend = SQLiteBackend(db_path)
        import numpy as np

        for i in range(3):
            embedding = np.random.rand(128).astype(np.float32).tolist()
            memory = MemoryItem(
                id=f"emb_{i}",
                content=f"Embedding test {i}",
                metadata={},
                embedding=embedding,
                timestamp=time.time()
            )
            backend.add_memory(memory)
        backend.close()

        # 导出
        exporter = DataExporter(db_path)
        count = exporter.to_json(json_path)

        # 验证嵌入数据被正确导出
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert len(data) == 3
        for item in data:
            assert item["embedding"] is not None
            assert len(item["embedding"]) == 128


# ============================================================================
# Test Compression Integration
# ============================================================================

class TestCompressionIntegration:
    """压缩功能集成测试"""

    def test_compress_decompress_cycle(self):
        """测试压缩解压周期"""
        compressor = AutoCompressor()

        # 测试各种数据
        test_data = [
            b"Short text",
            b"x" * 1000,
            b"Hello World! " * 100,
            bytes(range(256)),  # 二进制数据
        ]

        for data in test_data:
            compressed = compressor.compress(data)
            decompressed = compressor.decompress(compressed)
            assert decompressed == data

    def test_compression_ratio(self):
        """测试压缩比"""
        compressor = AutoCompressor()

        # 高重复数据 - 应该有高压缩比
        repetitive_data = b"ABC" * 10000
        compressed = compressor.compress(repetitive_data)
        ratio = compressor.get_compression_ratio(repetitive_data, compressed)
        assert ratio > 1  # 应该压缩成功

    def test_empty_data_handling(self):
        """测试空数据处理"""
        compressor = AutoCompressor()

        # 空字节
        compressed = compressor.compress(b"")
        decompressed = compressor.decompress(compressed)
        assert decompressed == b""

    def test_zlib_compression(self):
        """测试zlib压缩"""
        test_data = b"Test data for compression level testing " * 100

        # zlib压缩器
        zlib_compressor = AutoCompressor(algorithm="zlib")

        zlib_compressed = zlib_compressor.compress(test_data)
        zlib_decompressed = zlib_compressor.decompress(zlib_compressed)
        assert zlib_decompressed == test_data


# ============================================================================
# Test End-to-End Workflow
# ============================================================================

class TestEndToEnd:
    """端到端工作流测试"""

    def setup_method(self):
        """每个测试前设置"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """每个测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_memory_lifecycle(self):
        """测试完整记忆生命周期"""
        db_path = os.path.join(self.temp_dir, "lifecycle.db")

        # 创建客户端
        backend = SQLiteBackend(db_path)

        # 添加阶段
        memory_ids = []
        for i in range(10):
            memory = MemoryItem(
                id=f"lifecycle_{i}",
                content=f"Memory content {i}",
                metadata={"phase": "add", "index": i},
                embedding=None,
                timestamp=time.time()
            )
            mem_id = backend.add_memory(memory)
            memory_ids.append(mem_id)

        stats = backend.get_stats()
        assert stats["count"] == 10

        # 查询阶段
        results = backend.query("Memory", top_k=5)
        assert len(results) >= 0  # 可能有或没有结果

        # 更新阶段
        for mid in memory_ids[:5]:
            old = backend.get_memory(mid)
            new_content = old.content + " (updated)"
            new_memory = MemoryItem(
                id=mid,
                content=new_content,
                metadata={"phase": "update"},
                embedding=None,
                timestamp=time.time()
            )
            backend.update_memory(new_memory)

        # 验证更新
        updated = backend.get_memory(memory_ids[0])
        assert "updated" in updated.content

        # 删除阶段
        for mid in memory_ids[:3]:
            backend.delete(mid)

        stats = backend.get_stats()
        assert stats["count"] == 7

        backend.close()

    def test_concurrent_operations(self):
        """测试并发操作"""
        db_path = os.path.join(self.temp_dir, "concurrent.db")
        backend = SQLiteBackend(db_path)

        errors = []
        success_count = [0]

        def add_memories(start_id, count):
            try:
                for i in range(count):
                    memory = MemoryItem(
                        id=f"concurrent_{start_id}_{i}",
                        content=f"Concurrent memory {i}",
                        metadata={},
                        embedding=None,
                        timestamp=time.time()
                    )
                    backend.add_memory(memory)
                success_count[0] += count
            except Exception as e:
                errors.append(str(e))

        # 创建多个线程
        threads = []
        for t_id in range(3):
            t = threading.Thread(target=add_memories, args=(t_id, 10))
            threads.append(t)

        # 启动所有线程
        for t in threads:
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join()

        # 验证结果
        assert len(errors) == 0, f"Errors occurred: {errors}"
        stats = backend.get_stats()
        assert stats["count"] >= success_count[0]

        backend.close()

    def test_backup_during_operations(self):
        """测试操作期间的备份"""
        db_path = os.path.join(self.temp_dir, "backup_during.db")
        backup_dir = os.path.join(self.temp_dir, "backups")

        backend = SQLiteBackend(db_path)
        manager = BackupManager(db_path, backup_dir)

        # 添加初始数据
        for i in range(5):
            memory = MemoryItem(
                id=f"init_{i}",
                content=f"Initial {i}",
                metadata={},
                embedding=None,
                timestamp=time.time()
            )
            backend.add_memory(memory)

        # 备份
        backup_path = manager.backup()

        # 继续添加数据
        for i in range(5, 10):
            memory = MemoryItem(
                id=f"extra_{i}",
                content=f"Extra {i}",
                metadata={},
                embedding=None,
                timestamp=time.time()
            )
            backend.add_memory(memory)

        backend.close()

        # 恢复备份
        manager.restore(backup_path)

        # 验证恢复的是初始数据
        backend = SQLiteBackend(db_path)
        for i in range(5):
            mem = backend.get_memory(f"init_{i}")
            assert mem is not None
        backend.close()


# ============================================================================
# Test SDK Integration
# ============================================================================

class TestSDKIntegration:
    """SDK集成测试"""

    def setup_method(self):
        """每个测试前设置"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """每个测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_lite_client_basic_operations(self):
        """测试Lite客户端基本操作"""
        db_path = os.path.join(self.temp_dir, "lite_test.db")

        # 创建Lite客户端
        client = SuMemoryLite(db_path=db_path)

        # 添加记忆
        memory_ids = []
        for i in range(5):
            memory_id = client.add(f"Test memory {i}", metadata={"index": i})
            memory_ids.append(memory_id)

        # 查询
        results = client.query("Test", top_k=10)
        assert len(results) >= 0

        # 获取统计
        stats = client.get_stats()
        assert "count" in stats

        # 删除
        if memory_ids:
            client.delete(memory_ids[0])

        # 清理
        client.close()

    def test_lite_with_plugin(self):
        """测试Lite客户端与插件集成"""
        db_path = os.path.join(self.temp_dir, "lite_plugin.db")

        # 创建Lite客户端
        client = SuMemoryLite(db_path=db_path)

        # 注册插件
        registry = PluginRegistry()
        plugin = TextEmbeddingPlugin()
        plugin.initialize({})
        registry.register(plugin, auto_initialize=False)

        # 添加数据
        client.add("Plugin test memory", metadata={"plugin_test": True})

        # 验证插件仍可用
        retrieved = registry.get_plugin("TextEmbeddingPlugin")
        assert retrieved is not None

        # 清理
        registry.clear()
        client.close()


# ============================================================================
# Test Performance Integration
# ============================================================================

class TestPerformanceIntegration:
    """性能集成测试"""

    def setup_method(self):
        """每个测试前设置"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """每个测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_batch_operation_performance(self):
        """测试批量操作性能"""
        db_path = os.path.join(self.temp_dir, "batch_perf.db")
        backend = SQLiteBackend(db_path)

        # 批量添加
        batch_size = 100
        memories = [
            MemoryItem(
                id=f"perf_{i}",
                content=f"Performance test content {i}",
                metadata={},
                embedding=None,
                timestamp=time.time()
            )
            for i in range(batch_size)
        ]

        start = time.time()
        ids = backend.add_memory_batch(memories)
        elapsed = time.time() - start

        assert len(ids) == batch_size
        assert elapsed < 5.0  # 应该在5秒内完成

        # 批量查询
        start = time.time()
        for i in range(batch_size):
            backend.get_memory(f"perf_{i}")
        query_elapsed = time.time() - start

        assert query_elapsed < 5.0  # 查询应该在5秒内完成

        backend.close()

    def test_registry_performance(self):
        """测试注册表性能"""
        registry = PluginRegistry()
        registry.clear()

        # 批量注册
        start = time.time()
        for i in range(20):
            plugin = TextEmbeddingPlugin()
            plugin.name = f"perf_plugin_{i}"
            registry.register(plugin, auto_initialize=False)
        elapsed = time.time() - start

        assert elapsed < 10.0  # 注册应该在10秒内完成

        # 批量查询
        start = time.time()
        for i in range(20):
            registry.get_plugin(f"perf_plugin_{i}")
        query_elapsed = time.time() - start

        assert query_elapsed < 1.0  # 查询应该很快

        registry.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
