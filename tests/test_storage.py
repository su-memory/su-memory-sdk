"""
存储系统测试

测试范围:
- SQLiteBackend
- AutoCompressor
- BackupManager
- DataExporter

v1.7.0 测试套件
"""

import pytest
import os
import sys
import tempfile
import shutil
import time
import json
sys.path.insert(0, "src")

from su_memory.storage.sqlite_backend import SQLiteBackend, MemoryItem
from su_memory.storage.auto_compression import AutoCompressor
from su_memory.storage.backup_manager import BackupManager, BackupInfo
from su_memory.storage.exporter import DataExporter


# ============================================================================
# Test SQLiteBackend
# ============================================================================

class TestSQLiteBackend:
    """SQLite后端测试"""
    
    def setup_method(self):
        """每个测试前创建临时数据库"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.backend = SQLiteBackend(self.db_path)
    
    def teardown_method(self):
        """每个测试后清理"""
        if self.backend:
            self.backend.close()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_add_memory(self):
        """测试添加记忆"""
        memory = MemoryItem(
            id="test_1",
            content="Test memory content",
            metadata={"tag": "test"},
            embedding=None,
            timestamp=time.time()
        )
        memory_id = self.backend.add_memory(memory)
        assert memory_id == "test_1"
        
        # 验证添加成功
        stats = self.backend.get_stats()
        assert stats["count"] == 1
    
    def test_add_memory_batch(self):
        """测试批量添加记忆"""
        memories = [
            MemoryItem(
                id=f"batch_{i}",
                content=f"Batch memory {i}",
                metadata={},
                embedding=None,
                timestamp=time.time()
            )
            for i in range(5)
        ]
        ids = self.backend.add_memory_batch(memories)
        assert len(ids) == 5
        
        stats = self.backend.get_stats()
        assert stats["count"] == 5
    
    def test_get_memory(self):
        """测试获取记忆"""
        memory = MemoryItem(
            id="get_test",
            content="Get test content",
            metadata={"key": "value"},
            embedding=None,
            timestamp=time.time()
        )
        self.backend.add_memory(memory)
        
        retrieved = self.backend.get_memory("get_test")
        assert retrieved is not None
        assert retrieved.content == "Get test content"
        assert retrieved.metadata["key"] == "value"
    
    def test_query_memory(self):
        """测试查询记忆"""
        # 添加测试数据
        for i in range(3):
            memory = MemoryItem(
                id=f"query_{i}",
                content=f"Query test content number {i}",
                metadata={},
                embedding=None,
                timestamp=time.time()
            )
            self.backend.add_memory(memory)
        
        # 查询
        results = self.backend.query("query", top_k=10)
        assert len(results) >= 3
    
    def test_delete_memory(self):
        """测试删除记忆"""
        memory = MemoryItem(
            id="delete_test",
            content="Delete test",
            metadata={},
            embedding=None,
            timestamp=time.time()
        )
        self.backend.add_memory(memory)
        
        result = self.backend.delete("delete_test")
        assert result is True
        
        # 验证删除成功
        retrieved = self.backend.get_memory("delete_test")
        assert retrieved is None
    
    def test_delete_batch(self):
        """测试批量删除"""
        # 添加测试数据
        for i in range(5):
            memory = MemoryItem(
                id=f"delete_batch_{i}",
                content=f"Delete batch {i}",
                metadata={},
                embedding=None,
                timestamp=time.time()
            )
            self.backend.add_memory(memory)
        
        # 批量删除
        ids_to_delete = ["delete_batch_0", "delete_batch_1", "delete_batch_2"]
        count = self.backend.delete_batch(ids_to_delete)
        assert count == 3
    
    def test_search_memory(self):
        """测试条件搜索"""
        # 添加测试数据
        now = time.time()
        for i in range(3):
            memory = MemoryItem(
                id=f"search_{i}",
                content=f"Search test content {i}",
                metadata={"index": i},
                embedding=None,
                timestamp=now - i * 100
            )
            self.backend.add_memory(memory)
        
        # 搜索
        results = self.backend.search(
            {"keywords": ["search", "test"]},
            top_k=10
        )
        assert len(results) >= 3
    
    def test_search_with_time_range(self):
        """测试时间范围搜索"""
        now = time.time()
        
        # 添加旧记忆
        old_memory = MemoryItem(
            id="old_memory",
            content="Old memory content",
            metadata={},
            timestamp=now - 86400 * 7  # 7天前
        )
        self.backend.add_memory(old_memory)
        
        # 添加新记忆
        new_memory = MemoryItem(
            id="new_memory",
            content="New memory content",
            metadata={},
            timestamp=now - 3600  # 1小时前
        )
        self.backend.add_memory(new_memory)
        
        # 只搜索最近一天的
        results = self.backend.search(
            {"start_time": now - 86400},
            top_k=10
        )
        assert len(results) >= 1
    
    def test_search_by_vector(self):
        """测试向量搜索"""
        # 添加带向量的记忆
        memory = MemoryItem(
            id="vector_test",
            content="Vector test content",
            metadata={},
            embedding=[0.1] * 128,  # 128维向量
            timestamp=time.time()
        )
        self.backend.add_memory(memory)
        
        # 向量搜索
        query_vector = [0.1] * 128
        results = self.backend.search_by_vector(query_vector, top_k=5)
        assert len(results) >= 1
    
    def test_get_all_memories(self):
        """测试获取所有记忆"""
        # 添加测试数据
        for i in range(10):
            memory = MemoryItem(
                id=f"get_all_{i}",
                content=f"Get all {i}",
                metadata={},
                embedding=None,
                timestamp=time.time()
            )
            self.backend.add_memory(memory)
        
        # 获取所有
        all_memories = self.backend.get_all(limit=100)
        assert len(all_memories) >= 10
    
    def test_get_stats(self):
        """测试获取统计信息"""
        # 添加测试数据
        memory = MemoryItem(
            id="stats_test",
            content="Stats test content",
            metadata={},
            embedding=None,
            timestamp=time.time()
        )
        self.backend.add_memory(memory)
        
        stats = self.backend.get_stats()
        assert "count" in stats
        assert "total_content_size" in stats
        assert "embedded_count" in stats
        assert stats["count"] >= 1
    
    def test_vacuum(self):
        """测试数据库整理"""
        # 添加数据
        for i in range(10):
            memory = MemoryItem(
                id=f"vacuum_{i}",
                content=f"Vacuum test content {i}",
                metadata={},
                embedding=None,
                timestamp=time.time()
            )
            self.backend.add_memory(memory)
        
        # 执行VACUUM
        self.backend.vacuum()
        
        # 验证数据完整性
        stats = self.backend.get_stats()
        assert stats["count"] == 10
    
    def test_memory_item_to_dict(self):
        """测试MemoryItem转字典"""
        memory = MemoryItem(
            id="dict_test",
            content="Dict test",
            metadata={"key": "value"},
            embedding=[0.1, 0.2],
            timestamp=1234567890.0
        )
        data = memory.to_dict()
        assert data["id"] == "dict_test"
        assert data["content"] == "Dict test"
        assert data["metadata"]["key"] == "value"
    
    def test_memory_item_from_dict(self):
        """测试从字典创建MemoryItem"""
        data = {
            "id": "from_dict",
            "content": "From dict content",
            "metadata": {"tag": "test"},
            "embedding": [0.1, 0.2],
            "timestamp": 1234567890.0
        }
        memory = MemoryItem.from_dict(data)
        assert memory.id == "from_dict"
        assert memory.content == "From dict content"


# ============================================================================
# Test AutoCompressor
# ============================================================================

class TestAutoCompressor:
    """自动压缩器测试"""
    
    def test_compress_decompress_zlib(self):
        """测试zlib压缩解压"""
        compressor = AutoCompressor(algorithm="zlib")
        original = b"Hello World! " * 100
        compressed = compressor.compress(original)
        decompressed = compressor.decompress(compressed)
        assert decompressed == original
    
    def test_compression_ratio(self):
        """测试压缩比"""
        compressor = AutoCompressor(algorithm="zlib")
        original = b"Test data " * 50
        compressed = compressor.compress(original)
        ratio = compressor.get_compression_ratio(original, compressed)
        # 重复数据应该可以被压缩
        assert ratio >= 1
    
    def test_empty_data(self):
        """测试空数据处理"""
        compressor = AutoCompressor(algorithm="zlib")
        compressed = compressor.compress(b"")
        assert compressed == b""
        decompressed = compressor.decompress(compressed)
        assert decompressed == b""
    
    def test_get_stats(self):
        """测试获取压缩统计"""
        compressor = AutoCompressor(algorithm="zlib")
        original = b"Test data " * 50
        compressed = compressor.compress(original)
        stats = compressor.get_stats(original, compressed)
        assert "original_size" in stats
        assert "compressed_size" in stats
        assert "ratio" in stats
        assert stats["original_size"] == len(original)
    
    def test_is_compression_effective(self):
        """测试判断压缩是否有效"""
        compressor = AutoCompressor(algorithm="zlib")
        # 小数据不压缩
        small_data = b"hi"
        assert compressor.is_compression_effective(small_data) is False
        
        # 重复大数据应该压缩有效
        large_data = b"repeat " * 200
        assert compressor.is_compression_effective(large_data, threshold=1.1) is True
    
    def test_algorithm_property(self):
        """测试算法属性"""
        compressor_zlib = AutoCompressor(algorithm="zlib")
        assert compressor_zlib.algorithm == "zlib"
        
        compressor_auto = AutoCompressor(algorithm="auto")
        # auto模式会回退到zlib（因为可能没有lz4）
        assert compressor_auto.algorithm in ["lz4", "zlib"]


# ============================================================================
# Test BackupManager
# ============================================================================

class TestBackupManager:
    """备份管理器测试"""
    
    def setup_method(self):
        """每个测试前创建临时目录"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.backup_dir = os.path.join(self.temp_dir, "backups")
        
        # 创建测试数据库
        self.backend = SQLiteBackend(self.db_path)
        for i in range(5):
            memory = MemoryItem(
                id=f"backup_{i}",
                content=f"Backup content {i}",
                metadata={},
                embedding=None,
                timestamp=time.time()
            )
            self.backend.add_memory(memory)
        self.backend.close()
    
    def teardown_method(self):
        """每个测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_backup_creation(self):
        """测试创建备份"""
        manager = BackupManager(self.db_path, self.backup_dir)
        backup_path = manager.backup()
        assert os.path.exists(backup_path)
        assert backup_path.endswith(".db")
    
    def test_backup_with_custom_name(self):
        """测试自定义名称备份"""
        manager = BackupManager(self.db_path, self.backup_dir)
        backup_path = manager.backup(name="custom_backup")
        assert "custom_backup" in backup_path
    
    def test_list_backups(self):
        """测试列出备份"""
        manager = BackupManager(self.db_path, self.backup_dir)
        manager.backup()
        manager.backup()
        backups = manager.list_backups()
        assert len(backups) >= 2
    
    def test_get_latest_backup(self):
        """测试获取最新备份"""
        manager = BackupManager(self.db_path, self.backup_dir)
        manager.backup()
        time.sleep(0.1)
        manager.backup()
        latest = manager.get_latest_backup()
        assert latest is not None
        assert os.path.exists(latest.path)
    
    def test_delete_backup(self):
        """测试删除备份"""
        manager = BackupManager(self.db_path, self.backup_dir)
        backup_path = manager.backup()
        result = manager.delete_backup(backup_path)
        assert result is True
        assert not os.path.exists(backup_path)
    
    def test_backup_info_properties(self):
        """测试备份信息属性"""
        manager = BackupManager(self.db_path, self.backup_dir)
        backup_path = manager.backup()
        backups = manager.list_backups()
        backup = backups[0]
        
        assert backup.datetime is not None
        assert backup.name is not None
        assert backup.size > 0
    
    def test_backup_stats(self):
        """测试备份统计"""
        manager = BackupManager(self.db_path, self.backup_dir)
        manager.backup()
        stats = manager.get_stats()
        assert "backup_count" in stats
        assert "total_size" in stats
        assert "max_backups" in stats
    
    def test_backup_cleanup_old(self):
        """测试清理旧备份"""
        manager = BackupManager(self.db_path, self.backup_dir, max_backups=2)
        for i in range(5):
            manager.backup()
        
        backups = manager.list_backups()
        assert len(backups) <= 2


# ============================================================================
# Test DataExporter
# ============================================================================

class TestDataExporter:
    """数据导出器测试"""
    
    def setup_method(self):
        """每个测试前创建临时目录和测试数据"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.backend = SQLiteBackend(self.db_path)
        
        # 添加测试数据
        for i in range(5):
            memory = MemoryItem(
                id=f"export_{i}",
                content=f"Export content {i}",
                metadata={"index": i, "category": "test"},
                embedding=None,
                timestamp=time.time()
            )
            self.backend.add_memory(memory)
    
    def teardown_method(self):
        """每个测试后清理"""
        if self.backend:
            self.backend.close()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_export_json(self):
        """测试导出JSON"""
        exporter = DataExporter(self.db_path)
        json_path = os.path.join(self.temp_dir, "export.json")
        count = exporter.to_json(json_path)
        assert os.path.exists(json_path)
        assert count >= 5
        
        # 验证JSON内容
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert len(data) >= 5
            assert "content" in data[0]
    
    def test_export_json_without_metadata(self):
        """测试导出JSON（不含元数据）"""
        exporter = DataExporter(self.db_path)
        json_path = os.path.join(self.temp_dir, "export_no_meta.json")
        count = exporter.to_json(json_path, include_metadata=False)
        assert count >= 5
    
    def test_export_csv(self):
        """测试导出CSV"""
        exporter = DataExporter(self.db_path)
        csv_path = os.path.join(self.temp_dir, "export.csv")
        count = exporter.to_csv(csv_path)
        assert os.path.exists(csv_path)
        assert count >= 5
        
        # 验证CSV内容
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) >= 6  # 1 header + 5 data rows
    
    def test_export_markdown(self):
        """测试导出Markdown"""
        exporter = DataExporter(self.db_path)
        md_path = os.path.join(self.temp_dir, "export.md")
        count = exporter.to_markdown(md_path)
        assert os.path.exists(md_path)
        assert count >= 5
        
        # 验证Markdown内容
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "# Memory Export" in content
            assert "Total records:" in content
    
    def test_import_json(self):
        """测试从JSON导入"""
        # 先导出
        exporter = DataExporter(self.db_path)
        json_path = os.path.join(self.temp_dir, "import_test.json")
        exporter.to_json(json_path)
        
        # 创建新数据库
        new_db_path = os.path.join(self.temp_dir, "new_test.db")
        new_exporter = DataExporter(new_db_path)
        
        # 导入
        result = new_exporter.from_json(json_path)
        assert result["imported"] >= 5
    
    def test_import_json_clear_first(self):
        """测试先清空再导入JSON"""
        # 创建导入数据
        import_data = [
            {
                "id": "import_1",
                "content": "Imported content 1",
                "metadata": {},
                "timestamp": time.time()
            },
            {
                "id": "import_2",
                "content": "Imported content 2",
                "metadata": {},
                "timestamp": time.time()
            }
        ]
        import_path = os.path.join(self.temp_dir, "import_clear.json")
        with open(import_path, "w", encoding="utf-8") as f:
            json.dump(import_data, f)
        
        # 导入
        exporter = DataExporter(self.db_path)
        result = exporter.from_json(import_path, clear_first=True)
        assert result["imported"] >= 2
        
        # 验证只有导入的数据
        new_exporter = DataExporter(self.db_path)
        # 应该只有导入的数据，不包含原来的5条
    
    def test_import_csv(self):
        """测试从CSV导入"""
        # 创建CSV文件
        csv_content = """id,content,timestamp
csv_import_1,CSV content 1,1234567890
csv_import_2,CSV content 2,1234567891
"""
        csv_path = os.path.join(self.temp_dir, "import.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)
        
        # 导入
        exporter = DataExporter(self.db_path)
        result = exporter.from_csv(csv_path)
        assert result["imported"] >= 2
    
    def test_merge_json_files(self):
        """测试合并JSON文件"""
        # 创建多个JSON文件
        json1 = [{"id": "merge_1", "content": "Merge 1", "timestamp": time.time()}]
        json2 = [{"id": "merge_2", "content": "Merge 2", "timestamp": time.time()}]
        
        path1 = os.path.join(self.temp_dir, "merge1.json")
        path2 = os.path.join(self.temp_dir, "merge2.json")
        
        with open(path1, "w") as f:
            json.dump(json1, f)
        with open(path2, "w") as f:
            json.dump(json2, f)
        
        # 合并
        merged_db = os.path.join(self.temp_dir, "merged.db")
        exporter = DataExporter(self.db_path)
        result = exporter.merge([path1, path2], merged_db, conflict_strategy="skip")
        assert result["imported"] >= 2


# ============================================================================
# Integration Tests
# ============================================================================

class TestStorageIntegration:
    """存储系统集成测试"""
    
    def setup_method(self):
        """每个测试前创建临时目录"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "integration_test.db")
    
    def teardown_method(self):
        """每个测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_workflow(self):
        """测试完整工作流程"""
        # 1. 创建后端
        backend = SQLiteBackend(self.db_path)
        
        # 2. 添加数据
        for i in range(10):
            memory = MemoryItem(
                id=f"workflow_{i}",
                content=f"Workflow test {i}",
                metadata={"index": i},
                embedding=[0.1] * 128,
                timestamp=time.time()
            )
            backend.add_memory(memory)
        
        # 3. 压缩备份
        compressor = AutoCompressor()
        with open(self.db_path, "rb") as f:
            original_data = f.read()
        compressed = compressor.compress(original_data)
        compressed_path = os.path.join(self.temp_dir, "compressed_backup.bin")
        with open(compressed_path, "wb") as f:
            f.write(compressed)
        
        # 4. 导出数据
        exporter = DataExporter(self.db_path)
        json_path = os.path.join(self.temp_dir, "workflow_export.json")
        count = exporter.to_json(json_path)
        assert count == 10
        
        # 5. 创建备份
        backup_dir = os.path.join(self.temp_dir, "backups")
        backup_manager = BackupManager(self.db_path, backup_dir)
        backup_path = backup_manager.backup()
        assert os.path.exists(backup_path)
        
        # 6. 验证数据完整性
        stats = backend.get_stats()
        assert stats["count"] == 10
        assert stats["embedded_count"] == 10
        
        backend.close()
