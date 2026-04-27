"""
su-memory CLI命令实现

包含所有CLI子命令的实现。

Example:
    >>> from su_memory.cli.commands import cmd_add
    >>> cmd_add("test content")
"""

import sys
import json
import time
from pathlib import Path
from typing import Optional, List, Dict

from su_memory.storage import SQLiteBackend, BackupManager, DataExporter, MemoryItem

# 全局客户端实例
_backend: Optional[SQLiteBackend] = None
_backup_manager: Optional[BackupManager] = None


def get_backend(db_path: str = "su_memory.db") -> SQLiteBackend:
    """获取或创建后端实例"""
    global _backend
    if _backend is None:
        _backend = SQLiteBackend(db_path)
    return _backend


def get_backup_manager(db_path: str = "su_memory.db") -> BackupManager:
    """获取或创建备份管理器实例"""
    global _backup_manager
    if _backup_manager is None:
        _backup_manager = BackupManager(db_path, backup_dir="backups")
    return _backup_manager


def print_success(msg: str):
    """打印成功消息"""
    print(f"✓ {msg}")


def print_error(msg: str):
    """打印错误消息"""
    print(f"✗ {msg}", file=sys.stderr)


def print_info(msg: str):
    """打印信息消息"""
    print(f"ℹ {msg}")


# === 基础命令 ===

def cmd_init(db_path: str = "su_memory.db", force: bool = False) -> bool:
    """初始化su-memory

    Args:
        db_path: 数据库路径
        force: 是否强制重新初始化

    Returns:
        是否成功
    """
    db_file = Path(db_path)

    if db_file.exists() and not force:
        print_info(f"Database already exists at {db_path}")
        print_info("Use --force to reinitialize")
        return True

    try:
        backend = get_backend(db_path)
        stats = backend.get_stats()
        print_success(f"su-memory initialized at {db_path}")
        print_info(f"Records: {stats.get('count', 0)}")
        return True
    except Exception as e:
        print_error(f"Initialization failed: {e}")
        return False


def cmd_add(
    content: str,
    db_path: str = "su_memory.db",
    metadata: Optional[str] = None,
    id: Optional[str] = None,
) -> Optional[str]:
    """添加记忆

    Args:
        content: 记忆内容
        db_path: 数据库路径
        metadata: 元数据JSON字符串
        id: 自定义ID

    Returns:
        添加的记忆ID
    """
    try:
        backend = get_backend(db_path)

        meta_dict = {}
        if metadata:
            meta_dict = json.loads(metadata)

        memory = MemoryItem(
            id=id or f"mem_{int(time.time() * 1000)}",
            content=content,
            metadata=meta_dict,
            timestamp=time.time(),
        )

        memory_id = backend.add_memory(memory)
        print_success(f"Memory added: {memory_id}")
        return memory_id
    except Exception as e:
        print_error(f"Failed to add memory: {e}")
        return None


def cmd_query(
    query: str,
    db_path: str = "su_memory.db",
    top_k: int = 5,
    format: str = "text",
) -> List[Dict]:
    """查询记忆

    Args:
        query: 查询文本
        db_path: 数据库路径
        top_k: 返回结果数量
        format: 输出格式 (text|json)

    Returns:
        查询结果列表
    """
    try:
        backend = get_backend(db_path)
        results = backend.query(query, top_k=top_k)

        if not results:
            print_info("No results found")
            return []

        if format == "json":
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            for r in results:
                score = r.get("score", 0)
                content = r["content"]
                if len(content) > 80:
                    content = content[:80] + "..."
                print(f"[{score:.2f}] {content}")

        return results
    except Exception as e:
        print_error(f"Query failed: {e}")
        return []


def cmd_search(
    keywords: List[str],
    db_path: str = "su_memory.db",
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    limit: int = 100,
) -> List[MemoryItem]:
    """搜索记忆

    Args:
        keywords: 关键词列表
        db_path: 数据库路径
        start_time: 开始时间戳
        end_time: 结束时间戳
        limit: 返回数量限制

    Returns:
        匹配的MemoryItem列表
    """
    try:
        backend = get_backend(db_path)

        filters = {"keywords": keywords}
        if start_time:
            filters["start_time"] = start_time
        if end_time:
            filters["end_time"] = end_time

        results = backend.search(filters, top_k=limit)

        print_info(f"Found {len(results)} results")
        for i, memory in enumerate(results):
            content = memory.content
            if len(content) > 80:
                content = content[:80] + "..."
            print(f"  {i+1}. [{memory.id}] {content}")

        return results
    except Exception as e:
        print_error(f"Search failed: {e}")
        return []


def cmd_delete(memory_id: str, db_path: str = "su_memory.db") -> bool:
    """删除记忆

    Args:
        memory_id: 记忆ID
        db_path: 数据库路径

    Returns:
        是否成功删除
    """
    try:
        backend = get_backend(db_path)
        success = backend.delete(memory_id)

        if success:
            print_success(f"Memory deleted: {memory_id}")
        else:
            print_error(f"Memory not found: {memory_id}")

        return success
    except Exception as e:
        print_error(f"Delete failed: {e}")
        return False


def cmd_stats(db_path: str = "su_memory.db") -> Dict:
    """显示统计信息

    Args:
        db_path: 数据库路径

    Returns:
        统计信息字典
    """
    try:
        backend = get_backend(db_path)
        stats = backend.get_stats()

        print("=" * 40)
        print("su-memory Statistics")
        print("=" * 40)
        print(f"  Total records:     {stats.get('count', 0)}")
        print(f"  With embeddings:   {stats.get('embedded_count', 0)}")
        print(f"  Total content:     {stats.get('total_content_size', 0):,} bytes")
        print(f"  Database size:     {backend.get_db_size():,} bytes")
        print(f"  Embedding dim:     {stats.get('embedding_dim', 'N/A')}")
        print("  Oldest record:     ", end="")

        if stats.get("oldest_timestamp"):
            import datetime
            dt = datetime.datetime.fromtimestamp(stats["oldest_timestamp"])
            print(dt.strftime("%Y-%m-%d %H:%M"))
        else:
            print("N/A")

        print("  Newest record:     ", end="")
        if stats.get("newest_timestamp"):
            import datetime
            dt = datetime.datetime.fromtimestamp(stats["newest_timestamp"])
            print(dt.strftime("%Y-%m-%d %H:%M"))
        else:
            print("N/A")

        print("=" * 40)

        return stats
    except Exception as e:
        print_error(f"Stats failed: {e}")
        return {}


def cmd_export(
    path: str,
    db_path: str = "su_memory.db",
    format: str = "json",
) -> bool:
    """导出数据

    Args:
        path: 输出文件路径
        db_path: 数据库路径
        format: 导出格式 (json|csv|md)

    Returns:
        是否成功
    """
    try:
        exporter = DataExporter(db_path)

        if format == "json":
            count = exporter.to_json(path)
        elif format == "csv":
            count = exporter.to_csv(path)
        elif format == "md":
            count = exporter.to_markdown(path)
        else:
            print_error(f"Unsupported format: {format}")
            return False

        print_success(f"Exported {count} records to {path}")
        return True
    except Exception as e:
        print_error(f"Export failed: {e}")
        return False


def cmd_import(
    path: str,
    db_path: str = "su_memory.db",
    format: Optional[str] = None,
    clear: bool = False,
) -> Dict:
    """导入数据

    Args:
        path: 输入文件路径
        db_path: 数据库路径
        format: 导入格式 (json|csv)，自动检测
        clear: 是否先清空现有数据

    Returns:
        导入结果统计
    """
    try:
        # 自动检测格式
        if format is None:
            if path.endswith(".json"):
                format = "json"
            elif path.endswith(".csv"):
                format = "csv"
            else:
                print_error("Cannot detect format. Please specify --format")
                return {"errors": 1}

        exporter = DataExporter(db_path)

        if format == "json":
            result = exporter.from_json(path, clear_first=clear)
        elif format == "csv":
            result = exporter.from_csv(path)
        else:
            print_error(f"Unsupported format: {format}")
            return {"errors": 1}

        print_success(f"Imported: {result.get('imported', 0)}")
        if result.get("updated", 0) > 0:
            print_info(f"Updated: {result['updated']}")
        if result.get("skipped", 0) > 0:
            print_info(f"Skipped: {result['skipped']}")
        if result.get("errors", 0) > 0:
            print_error(f"Errors: {result['errors']}")

        return result
    except Exception as e:
        print_error(f"Import failed: {e}")
        return {"errors": 1}


def cmd_backup(
    db_path: str = "su_memory.db",
    name: Optional[str] = None,
) -> Optional[str]:
    """创建备份

    Args:
        db_path: 数据库路径
        name: 自定义备份名

    Returns:
        备份文件路径
    """
    try:
        manager = get_backup_manager(db_path)
        backup_path = manager.backup(name)
        print_success(f"Backup created: {backup_path}")
        return backup_path
    except Exception as e:
        print_error(f"Backup failed: {e}")
        return None


def cmd_restore(
    backup_path: str,
    db_path: str = "su_memory.db",
) -> bool:
    """恢复备份

    Args:
        backup_path: 备份文件路径
        db_path: 数据库路径

    Returns:
        是否成功恢复
    """
    try:
        manager = get_backup_manager(db_path)
        success = manager.restore(backup_path)

        if success:
            print_success(f"Restored from: {backup_path}")
        else:
            print_error(f"Failed to restore: {backup_path}")

        return success
    except Exception as e:
        print_error(f"Restore failed: {e}")
        return False


def cmd_list_backups(db_path: str = "su_memory.db") -> List:
    """列出所有备份

    Args:
        db_path: 数据库路径

    Returns:
        备份信息列表
    """
    try:
        manager = get_backup_manager(db_path)
        backups = manager.list_backups()

        if not backups:
            print_info("No backups found")
            return []

        print(f"Found {len(backups)} backup(s):")
        for i, backup in enumerate(backups):
            dt = backup.datetime.strftime("%Y-%m-%d %H:%M:%S")
            size_kb = backup.size / 1024
            print(f"  {i+1}. {backup.name}")
            print(f"      Time: {dt}")
            print(f"      Size: {size_kb:.1f} KB")
            print(f"      Records: {backup.db_records}")

        return backups
    except Exception as e:
        print_error(f"List backups failed: {e}")
        return []


# === 插件命令 ===

def cmd_plugin_list() -> List[str]:
    """列出已加载的插件

    Returns:
        插件名称列表
    """
    try:
        from su_memory._sys._plugin_registry import PluginRegistry

        registry = PluginRegistry.get_instance()
        plugins = registry.list_plugins()

        if not plugins:
            print_info("No plugins loaded")
            return []

        print(f"Loaded plugins ({len(plugins)}):")
        for name in plugins:
            print(f"  - {name}")

        return plugins
    except ImportError:
        print_error("Plugin system not available")
        return []
    except Exception as e:
        print_error(f"Plugin list failed: {e}")
        return []


def cmd_plugin_load(plugin_path: str) -> bool:
    """加载插件

    Args:
        plugin_path: 插件路径或模块名

    Returns:
        是否成功加载
    """
    try:
        from su_memory._sys._plugin_registry import PluginRegistry

        registry = PluginRegistry.get_instance()

        # 尝试作为模块加载
        if "." in plugin_path:
            import importlib
            module = importlib.import_module(plugin_path)
        else:
            # 尝试作为路径加载
            import importlib.util
            spec = importlib.util.spec_from_file_location("plugin", plugin_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:
                print_error(f"Cannot load plugin: {plugin_path}")
                return False

        # 注册插件
        if hasattr(module, "register"):
            module.register(registry)

        print_success(f"Plugin loaded: {plugin_path}")
        return True
    except Exception as e:
        print_error(f"Plugin load failed: {e}")
        return False


# === 向量搜索命令 ===

def cmd_vector_search(
    vector: List[float],
    db_path: str = "su_memory.db",
    top_k: int = 10,
) -> List[Dict]:
    """向量相似度搜索

    Args:
        vector: 查询向量
        db_path: 数据库路径
        top_k: 返回结果数量

    Returns:
        搜索结果列表
    """
    try:
        backend = get_backend(db_path)
        results = backend.search_by_vector(vector, top_k=top_k)

        if not results:
            print_info("No similar vectors found")
            return []

        print(f"Found {len(results)} similar results:")
        for r in results:
            content = r["content"]
            if len(content) > 80:
                content = content[:80] + "..."
            print(f"  [{r['score']:.4f}] {content}")

        return results
    except Exception as e:
        print_error(f"Vector search failed: {e}")
        return []


def cmd_batch_add(
    contents: List[str],
    db_path: str = "su_memory.db",
) -> List[str]:
    """批量添加记忆

    Args:
        contents: 内容列表
        db_path: 数据库路径

    Returns:
        添加的记忆ID列表
    """
    try:
        backend = get_backend(db_path)

        memories = []
        for i, content in enumerate(contents):
            memory = MemoryItem(
                id=f"mem_{int(time.time() * 1000)}_{i}",
                content=content,
                metadata={},
                timestamp=time.time(),
            )
            memories.append(memory)

        ids = backend.add_memory_batch(memories)
        print_success(f"Added {len(ids)} memories")
        return ids
    except Exception as e:
        print_error(f"Batch add failed: {e}")
        return []


# === CLI主程序 ===

def run_cli():
    """运行CLI交互模式"""

    print("su-memory CLI")
    print("Type 'help' for commands, 'exit' to quit")
    print()

    while True:
        try:
            cmd = input("su-memory> ").strip()

            if not cmd:
                continue

            if cmd in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            if cmd == "help":
                print_help()
                continue

            # 简单命令解析
            parts = cmd.split()
            if parts[0] == "add":
                if len(parts) > 1:
                    cmd_add(" ".join(parts[1:]))
                else:
                    print_error("Usage: add <content>")
            elif parts[0] == "query":
                if len(parts) > 1:
                    cmd_query(" ".join(parts[1:]))
                else:
                    print_error("Usage: query <text>")
            elif parts[0] == "stats":
                cmd_stats()
            elif parts[0] == "list":
                cmd_list_backups()
            else:
                print_error(f"Unknown command: {parts[0]}")
                print_info("Type 'help' for available commands")

        except KeyboardInterrupt:
            print("\nUse 'exit' to quit")
        except EOFError:
            print("\nGoodbye!")
            break


def print_help():
    """打印帮助信息"""
    print("""
Available commands:
  init              Initialize su-memory
  add <content>     Add a memory
  query <text>     Query memories
  search <keywords> Search memories
  delete <id>      Delete a memory
  stats            Show statistics
  export <path>     Export data
  import <path>    Import data
  backup           Create a backup
  restore <path>   Restore from backup
  list             List backups
  plugin list      List plugins
  help             Show this help
  exit             Exit CLI
""")


# 命令行接口包装器
def create_cli_commands():
    """创建Click命令行接口"""
    import click

    @click.group()
    def cli():
        """su-memory CLI - Semantic Memory Engine"""
        pass

    @cli.command()
    @click.argument("db_path", default="su_memory.db")
    @click.option("--force", is_flag=True, help="Force reinitialize")
    def init(db_path, force):
        """Initialize su-memory database"""
        cmd_init(db_path, force)

    @cli.command()
    @click.argument("content")
    @click.option("--db", "db_path", default="su_memory.db")
    @click.option("--meta", "metadata", default=None, help="JSON metadata")
    @click.option("--id", "memory_id", default=None)
    def add(content, db_path, metadata, memory_id):
        """Add a memory"""
        cmd_add(content, db_path, metadata, memory_id)

    @cli.command()
    @click.argument("query")
    @click.option("--db", "db_path", default="su_memory.db")
    @click.option("--top-k", default=5)
    @click.option("--format", "fmt", default="text")
    def query(query, db_path, top_k, fmt):
        """Query memories"""
        cmd_query(query, db_path, top_k, fmt)

    @cli.command()
    @click.argument("keywords", nargs=-1)
    @click.option("--db", "db_path", default="su_memory.db")
    @click.option("--start", "start_time", type=float, default=None)
    @click.option("--end", "end_time", type=float, default=None)
    def search(keywords, db_path, start_time, end_time):
        """Search memories"""
        cmd_search(list(keywords), db_path, start_time, end_time)

    @cli.command()
    @click.argument("memory_id")
    @click.option("--db", "db_path", default="su_memory.db")
    def delete(memory_id, db_path):
        """Delete a memory"""
        cmd_delete(memory_id, db_path)

    @cli.command()
    @click.option("--db", "db_path", default="su_memory.db")
    def stats(db_path):
        """Show statistics"""
        cmd_stats(db_path)

    @cli.command()
    @click.argument("path")
    @click.option("--db", "db_path", default="su_memory.db")
    @click.option("--format", "fmt", default="json")
    def export(path, db_path, fmt):
        """Export data"""
        cmd_export(path, db_path, fmt)

    @cli.command()
    @click.argument("path")
    @click.option("--db", "db_path", default="su_memory.db")
    @click.option("--format", "fmt", default=None)
    @click.option("--clear", is_flag=True)
    def import_data(path, db_path, fmt, clear):
        """Import data"""
        cmd_import(path, db_path, fmt, clear)

    @cli.command()
    @click.option("--db", "db_path", default="su_memory.db")
    @click.option("--name", default=None)
    def backup(db_path, name):
        """Create a backup"""
        cmd_backup(db_path, name)

    @cli.command()
    @click.argument("backup_path")
    @click.option("--db", "db_path", default="su_memory.db")
    def restore(backup_path, db_path):
        """Restore from backup"""
        cmd_restore(backup_path, db_path)

    @cli.command()
    @click.option("--db", "db_path", default="su_memory.db")
    def list_backups(db_path):
        """List all backups"""
        cmd_list_backups(db_path)

    @cli.command()
    def plugin_list():
        """List loaded plugins"""
        cmd_plugin_list()

    @cli.command()
    @click.argument("plugin_path")
    def plugin_load(plugin_path):
        """Load a plugin"""
        cmd_plugin_load(plugin_path)

    return cli
