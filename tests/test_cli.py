"""
CLI工具测试

测试范围:
- 所有CLI命令
- Click命令行界面

v1.7.0 测试套件
"""

import pytest
import sys
import os
import tempfile
import shutil
import time
sys.path.insert(0, "src")

from click.testing import CliRunner

from su_memory.cli.main import cli
from su_memory.cli.commands import (
    cmd_init, cmd_add, cmd_query, cmd_search, cmd_delete,
    cmd_stats, cmd_export, cmd_import, cmd_backup, cmd_restore,
    cmd_list_backups, cmd_plugin_list, get_backend
)


# ============================================================================
# Test CLI Commands (Function Level)
# ============================================================================

class TestCLICommands:
    """CLI命令函数测试"""
    
    def setup_method(self):
        """每个测试前创建临时目录"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.runner = CliRunner()
    
    def teardown_method(self):
        """每个测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cmd_init_new_database(self):
        """测试初始化新数据库"""
        result = cmd_init(self.db_path)
        assert result is True
        assert os.path.exists(self.db_path)
    
    def test_cmd_init_existing_database(self):
        """测试初始化已存在的数据库"""
        # 先初始化
        cmd_init(self.db_path)
        # 再次初始化不应报错
        result = cmd_init(self.db_path)
        assert result is True
    
    def test_cmd_init_force_reinit(self):
        """测试强制重新初始化"""
        # 先初始化
        cmd_init(self.db_path)
        # 强制重新初始化
        result = cmd_init(self.db_path, force=True)
        assert result is True
    
    def test_cmd_add_memory(self):
        """测试添加记忆"""
        cmd_init(self.db_path)
        memory_id = cmd_add(
            content="Test memory content",
            db_path=self.db_path
        )
        assert memory_id is not None
    
    def test_cmd_add_with_metadata(self):
        """测试带元数据添加记忆"""
        cmd_init(self.db_path)
        import json
        memory_id = cmd_add(
            content="Test with metadata",
            db_path=self.db_path,
            metadata=json.dumps({"tag": "test", "category": "cli"})
        )
        assert memory_id is not None
    
    def test_cmd_add_with_custom_id(self):
        """测试带自定义ID添加记忆"""
        cmd_init(self.db_path)
        custom_id = "custom_id_12345"
        memory_id = cmd_add(
            content="Custom ID test",
            db_path=self.db_path,
            id=custom_id
        )
        assert memory_id == custom_id
    
    def test_cmd_query(self):
        """测试查询记忆"""
        cmd_init(self.db_path)
        cmd_add("Python programming language", db_path=self.db_path)
        
        results = cmd_query("Python", db_path=self.db_path, top_k=5)
        assert len(results) >= 1
    
    def test_cmd_query_json_format(self):
        """测试JSON格式查询"""
        cmd_init(self.db_path)
        cmd_add("JSON test content", db_path=self.db_path)
        
        result = self.runner.invoke(cli, [
            "query", "JSON",
            "--db", self.db_path,
            "--format", "json"
        ])
        assert result.exit_code == 0
    
    def test_cmd_search(self):
        """测试搜索记忆"""
        cmd_init(self.db_path)
        cmd_add("Searchable content one", db_path=self.db_path)
        cmd_add("Searchable content two", db_path=self.db_path)
        
        results = cmd_search(
            keywords=["Searchable", "content"],
            db_path=self.db_path,
            limit=10
        )
        assert len(results) >= 2
    
    def test_cmd_search_with_time_range(self):
        """测试时间范围搜索"""
        cmd_init(self.db_path)
        now = time.time()
        
        # 添加旧记忆
        from su_memory.storage import SQLiteBackend, MemoryItem
        backend = get_backend(self.db_path)
        memory = MemoryItem(
            id="old_search",
            content="Old memory",
            timestamp=now - 86400 * 30
        )
        backend.add_memory(memory)
        
        results = cmd_search(
            keywords=["Old"],
            db_path=self.db_path,
            start_time=now - 86400 * 365,
            end_time=now
        )
        assert len(results) >= 1
    
    def test_cmd_delete(self):
        """测试删除记忆"""
        cmd_init(self.db_path)
        memory_id = cmd_add("Delete me", db_path=self.db_path)
        
        result = cmd_delete(memory_id, db_path=self.db_path)
        assert result is True
    
    def test_cmd_delete_nonexistent(self):
        """测试删除不存在的记忆"""
        cmd_init(self.db_path)
        result = cmd_delete("nonexistent_id", db_path=self.db_path)
        assert result is False
    
    def test_cmd_stats(self):
        """测试统计信息"""
        cmd_init(self.db_path)
        cmd_add("Stats test 1", db_path=self.db_path)
        cmd_add("Stats test 2", db_path=self.db_path)
        
        stats = cmd_stats(db_path=self.db_path)
        assert "count" in stats
        assert stats["count"] >= 2
    
    def test_cmd_export_json(self):
        """测试导出JSON"""
        cmd_init(self.db_path)
        cmd_add("Export test", db_path=self.db_path)
        
        export_path = os.path.join(self.temp_dir, "export.json")
        result = cmd_export(export_path, db_path=self.db_path, format="json")
        assert result is True
        assert os.path.exists(export_path)
    
    def test_cmd_export_csv(self):
        """测试导出CSV"""
        cmd_init(self.db_path)
        cmd_add("CSV export test", db_path=self.db_path)
        
        export_path = os.path.join(self.temp_dir, "export.csv")
        result = cmd_export(export_path, db_path=self.db_path, format="csv")
        assert result is True
        assert os.path.exists(export_path)
    
    def test_cmd_export_markdown(self):
        """测试导出Markdown"""
        cmd_init(self.db_path)
        cmd_add("Markdown export test", db_path=self.db_path)
        
        export_path = os.path.join(self.temp_dir, "export.md")
        result = cmd_export(export_path, db_path=self.db_path, format="md")
        assert result is True
        assert os.path.exists(export_path)
    
    def test_cmd_import_json(self):
        """测试导入JSON"""
        cmd_init(self.db_path)
        
        # 创建导入数据
        import json
        import_data = [
            {"id": "import_1", "content": "Import 1", "timestamp": time.time()},
            {"id": "import_2", "content": "Import 2", "timestamp": time.time()}
        ]
        import_path = os.path.join(self.temp_dir, "import.json")
        with open(import_path, "w", encoding="utf-8") as f:
            json.dump(import_data, f)
        
        result = cmd_import(import_path, db_path=self.db_path)
        assert "imported" in result
        assert result.get("imported", 0) >= 2
    
    def test_cmd_import_csv(self):
        """测试导入CSV"""
        cmd_init(self.db_path)
        
        # 创建CSV数据
        csv_content = """id,content,timestamp
csv_1,CSV Import 1,1234567890
csv_2,CSV Import 2,1234567891
"""
        csv_path = os.path.join(self.temp_dir, "import.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)
        
        result = cmd_import(csv_path, db_path=self.db_path, format="csv")
        assert "imported" in result
    
    def test_cmd_backup(self):
        """测试创建备份"""
        cmd_init(self.db_path)
        cmd_add("Backup test", db_path=self.db_path)
        
        backup_path = cmd_backup(db_path=self.db_path)
        assert backup_path is not None
        assert os.path.exists(backup_path)
    
    def test_cmd_backup_with_name(self):
        """测试自定义名称备份"""
        cmd_init(self.db_path)
        
        backup_path = cmd_backup(db_path=self.db_path, name="my_backup")
        assert backup_path is not None
        assert "my_backup" in backup_path
    
    def test_cmd_list_backups(self):
        """测试列出备份"""
        cmd_init(self.db_path)
        cmd_backup(db_path=self.db_path)
        cmd_backup(db_path=self.db_path)
        
        backups = cmd_list_backups(db_path=self.db_path)
        assert len(backups) >= 2
    
    def test_cmd_plugin_list(self):
        """测试列出插件"""
        plugins = cmd_plugin_list()
        # 可能返回空列表（如果插件系统未初始化）
        assert isinstance(plugins, list)


# ============================================================================
# Test Click CLI Interface
# ============================================================================

class TestCLIClickInterface:
    """Click CLI界面测试"""
    
    def setup_method(self):
        """每个测试前创建临时目录"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.runner = CliRunner()
    
    def teardown_method(self):
        """每个测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cli_help(self):
        """测试CLI帮助"""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "su-memory" in result.output
    
    def test_cli_init_command(self):
        """测试init命令"""
        result = self.runner.invoke(cli, [
            "init", self.db_path
        ])
        assert result.exit_code == 0
    
    def test_cli_add_command(self):
        """测试add命令"""
        cmd_init(self.db_path)
        
        result = self.runner.invoke(cli, [
            "add", "CLI add test",
            "--db", self.db_path
        ])
        assert result.exit_code == 0
    
    def test_cli_add_with_metadata(self):
        """测试带元数据add命令"""
        cmd_init(self.db_path)
        
        import json
        metadata = json.dumps({"tag": "test"})
        
        result = self.runner.invoke(cli, [
            "add", "Metadata test",
            "--db", self.db_path,
            "--meta", metadata
        ])
        assert result.exit_code == 0
    
    def test_cli_query_command(self):
        """测试query命令"""
        cmd_init(self.db_path)
        cmd_add("Query test content", db_path=self.db_path)
        
        result = self.runner.invoke(cli, [
            "query", "Query",
            "--db", self.db_path,
            "--top-k", "5"
        ])
        assert result.exit_code == 0
    
    def test_cli_search_command(self):
        """测试search命令"""
        cmd_init(self.db_path)
        cmd_add("Search keyword test", db_path=self.db_path)
        
        result = self.runner.invoke(cli, [
            "search", "Search", "keyword",
            "--db", self.db_path
        ])
        assert result.exit_code == 0
    
    def test_cli_delete_command(self):
        """测试delete命令"""
        cmd_init(self.db_path)
        memory_id = cmd_add("Delete test", db_path=self.db_path)
        
        result = self.runner.invoke(cli, [
            "delete", memory_id,
            "--db", self.db_path
        ])
        assert result.exit_code == 0
    
    def test_cli_stats_command(self):
        """测试stats命令"""
        cmd_init(self.db_path)
        cmd_add("Stats test", db_path=self.db_path)
        
        result = self.runner.invoke(cli, [
            "stats",
            "--db", self.db_path
        ])
        assert result.exit_code == 0
        assert "Statistics" in result.output
    
    def test_cli_export_command(self):
        """测试export命令"""
        cmd_init(self.db_path)
        cmd_add("Export test", db_path=self.db_path)
        
        export_path = os.path.join(self.temp_dir, "cli_export.json")
        result = self.runner.invoke(cli, [
            "export", export_path,
            "--db", self.db_path
        ])
        assert result.exit_code == 0
        assert os.path.exists(export_path)
    
    def test_cli_backup_command(self):
        """测试backup命令"""
        cmd_init(self.db_path)
        
        result = self.runner.invoke(cli, [
            "backup",
            "--db", self.db_path
        ])
        assert result.exit_code == 0
    
    def test_cli_list_backups_command(self):
        """测试list-backups命令"""
        cmd_init(self.db_path)
        cmd_backup(db_path=self.db_path)
        
        result = self.runner.invoke(cli, [
            "list-backups",
            "--db", self.db_path
        ])
        assert result.exit_code == 0
    
    def test_cli_plugin_list_command(self):
        """测试plugin-list命令"""
        result = self.runner.invoke(cli, ["plugin-list"])
        assert result.exit_code == 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestCLIIntegration:
    """CLI集成测试"""
    
    def setup_method(self):
        """每个测试前创建临时目录"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "integration.db")
        self.runner = CliRunner()
    
    def teardown_method(self):
        """每个测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_workflow(self):
        """测试完整工作流程"""
        # 1. 初始化
        result = self.runner.invoke(cli, ["init", self.db_path])
        assert result.exit_code == 0
        
        # 2. 添加数据
        result = self.runner.invoke(cli, [
            "add", "First memory",
            "--db", self.db_path
        ])
        assert result.exit_code == 0
        
        result = self.runner.invoke(cli, [
            "add", "Second memory",
            "--db", self.db_path
        ])
        assert result.exit_code == 0
        
        # 3. 查询
        result = self.runner.invoke(cli, [
            "query", "memory",
            "--db", self.db_path
        ])
        assert result.exit_code == 0
        
        # 4. 统计
        result = self.runner.invoke(cli, [
            "stats",
            "--db", self.db_path
        ])
        assert result.exit_code == 0
        assert "count" in result.output.lower() or "Total records" in result.output
        
        # 5. 备份
        result = self.runner.invoke(cli, [
            "backup",
            "--db", self.db_path
        ])
        assert result.exit_code == 0
        
        # 6. 导出
        export_path = os.path.join(self.temp_dir, "workflow_export.json")
        result = self.runner.invoke(cli, [
            "export", export_path,
            "--db", self.db_path
        ])
        assert result.exit_code == 0
    
    def test_batch_operations(self):
        """测试批量操作"""
        cmd_init(self.db_path)
        
        # 批量添加
        for i in range(10):
            cmd_add(f"Batch memory {i}", db_path=self.db_path)
        
        # 验证数量
        stats = cmd_stats(db_path=self.db_path)
        assert stats["count"] >= 10
        
        # 搜索
        results = cmd_search(
            keywords=["Batch"],
            db_path=self.db_path,
            limit=20
        )
        assert len(results) >= 10
    
    def test_backup_restore_cycle(self):
        """测试备份恢复周期"""
        # 1. 初始化并添加数据
        cmd_init(self.db_path)
        memory_id = cmd_add("Original data", db_path=self.db_path)
        
        # 2. 创建备份
        backup_path = cmd_backup(db_path=self.db_path)
        
        # 3. 添加更多数据
        cmd_add("Additional data", db_path=self.db_path)
        
        # 4. 恢复备份
        result = cmd_restore(backup_path, db_path=self.db_path)
        assert result is True
        
        # 5. 验证数据已恢复
        results = cmd_query("Original", db_path=self.db_path)
        # 注意：恢复后只有原始数据
