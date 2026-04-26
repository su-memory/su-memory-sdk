"""
v1.7.0 Integration Tests

Tests all new modules working together
"""

import pytest
import sys
import tempfile
import os
import time

sys.path.insert(0, "src")

from su_memory._sys._plugin_registry import PluginRegistry
from su_memory._sys._plugin_sandbox import SandboxedExecutor
from su_memory.storage.sqlite_backend import SQLiteBackend, MemoryItem
from su_memory.storage.backup_manager import BackupManager
from su_memory.storage.exporter import DataExporter


class TestPluginStorageIntegration:
    """插件与存储集成测试"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.registry = PluginRegistry()
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.registry.clear()
    
    def test_plugin_with_storage(self):
        """测试插件使用存储"""
        db_path = os.path.join(self.temp_dir, "test.db")
        backend = SQLiteBackend(db_path)
        
        # 添加测试数据
        memory = MemoryItem(
            id="test_1",
            content="Test memory",
            metadata={"plugin": "test"},
            embedding=None,
            timestamp=time.time()
        )
        backend.add_memory(memory)
        
        # 注册插件
        from su_memory.plugins.monitor_plugin import MonitorPlugin
        plugin = MonitorPlugin()
        plugin.initialize({})
        self.registry.register(plugin)
        
        # 执行插件
        executor = SandboxedExecutor()
        result = executor.execute(plugin, {
            "operation": "monitor",
            "action": "start"
        })
        
        assert result.success is True
        backend.close()


class TestBackupRestore:
    """备份恢复集成测试"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_backup_and_restore(self):
        """测试备份和恢复"""
        db_path = os.path.join(self.temp_dir, "memories.db")
        backup_dir = os.path.join(self.temp_dir, "backups")
        
        # 创建数据
        backend = SQLiteBackend(db_path)
        for i in range(10):
            memory = MemoryItem(
                id=f"mem_{i}",
                content=f"Content {i}",
                metadata={},
                embedding=None,
                timestamp=time.time()
            )
            backend.add_memory(memory)
        
        stats1 = backend.get_stats()
        backend.close()
        
        # 备份
        manager = BackupManager(db_path, backup_dir)
        backup_path = manager.backup()
        assert os.path.exists(backup_path)
        
        # 删除原数据库
        os.unlink(db_path)
        
        # 恢复
        result = manager.restore(backup_path)
        assert result is True
        assert os.path.exists(db_path)
        
        # 验证恢复
        backend2 = SQLiteBackend(db_path)
        stats2 = backend2.get_stats()
        assert stats2["count"] == stats1["count"]
        backend2.close()


class TestExportImport:
    """导出导入集成测试"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_json_export(self):
        """测试JSON导出"""
        db_path = os.path.join(self.temp_dir, "memories.db")
        json_path = os.path.join(self.temp_dir, "export.json")
        
        # 创建数据
        backend = SQLiteBackend(db_path)
        for i in range(5):
            memory = MemoryItem(
                id=f"mem_{i}",
                content=f"Content {i}",
                metadata={"index": i},
                embedding=None,
                timestamp=time.time()
            )
            backend.add_memory(memory)
        backend.close()
        
        # 导出
        exporter = DataExporter(db_path)
        exporter.to_json(json_path)
        assert os.path.exists(json_path)
        
        # 验证JSON内容
        import json
        with open(json_path) as f:
            data = json.load(f)
        assert len(data) == 5


class TestEndToEnd:
    """端到端测试"""
    
    def test_full_workflow(self):
        """测试完整工作流"""
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        try:
            # 创建客户端
            from su_memory import SuMemoryLite
            client = SuMemoryLite()
            
            # 添加记忆
            memory_ids = []
            for i in range(5):
                memory_id = client.add(f"Memory {i}", metadata={"index": i})
                memory_ids.append(memory_id)
            
            # 查询
            results = client.query("Memory", top_k=3)
            assert len(results) > 0
            
            # 删除
            client.delete(memory_ids[0])
            
            # 获取统计
            stats = client.get_stats()
            assert stats["count"] == 4
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
