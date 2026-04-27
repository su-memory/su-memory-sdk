"""
su_memory.cli — 命令行工具模块

提供su-memory SDK的CLI工具。

Example:
    >>> from su_memory.cli import cli
    >>> cli()  # 运行CLI
"""

from su_memory.cli.main import cli
from su_memory.cli.commands import (
    cmd_init,
    cmd_add,
    cmd_query,
    cmd_search,
    cmd_delete,
    cmd_stats,
    cmd_export,
    cmd_import,
    cmd_backup,
    cmd_restore,
    cmd_plugin_list,
    cmd_plugin_load,
)

__all__ = [
    "cli",
    "cmd_init",
    "cmd_add",
    "cmd_query",
    "cmd_search",
    "cmd_delete",
    "cmd_stats",
    "cmd_export",
    "cmd_import",
    "cmd_backup",
    "cmd_restore",
    "cmd_plugin_list",
    "cmd_plugin_load",
]
