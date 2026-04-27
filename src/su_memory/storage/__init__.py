"""
su_memory.storage — 本地数据存储模块

提供本地数据持久化和管理功能。

Example:
    >>> from su_memory.storage import SQLiteBackend, BackupManager, DataExporter
    >>> backend = SQLiteBackend("memories.db")
"""

from su_memory.storage.sqlite_backend import (
    SQLiteBackend,
    MemoryItem,
)
from su_memory.storage.auto_compression import AutoCompressor
from su_memory.storage.backup_manager import BackupManager
from su_memory.storage.exporter import DataExporter

__all__ = [
    "SQLiteBackend",
    "MemoryItem",
    "AutoCompressor",
    "BackupManager",
    "DataExporter",
]
