"""
su_memory.storage — 本地数据存储模块

提供本地数据持久化和管理功能，v2.7.0 新增异步存储抽象和分层存储。

Example:
    >>> from su_memory.storage import SQLiteBackend, BackupManager, DataExporter
    >>> backend = SQLiteBackend("memories.db")
    >>> # v2.7.0 新增
    >>> from su_memory.storage.base import StorageBackend, AsyncMemoryItem
    >>> from su_memory.storage.pgvector_backend import PgVectorBackend
    >>> from su_memory.storage.tiered import TieredStorage, TierConfig
"""

from su_memory.storage.sqlite_backend import (
    SQLiteBackend,
    MemoryItem,
)
from su_memory.storage.auto_compression import AutoCompressor
from su_memory.storage.backup_manager import BackupManager
from su_memory.storage.exporter import DataExporter

# v2.7.0 异步存储
from su_memory.storage.base import StorageBackend, AsyncMemoryItem
from su_memory.storage.pgvector_backend import PgVectorBackend
from su_memory.storage.tiered import TieredStorage, TierConfig

__all__ = [
    "SQLiteBackend",
    "MemoryItem",
    "AutoCompressor",
    "BackupManager",
    "DataExporter",
    # v2.7.0
    "StorageBackend",
    "AsyncMemoryItem",
    "PgVectorBackend",
    "TieredStorage",
    "TierConfig",
]
