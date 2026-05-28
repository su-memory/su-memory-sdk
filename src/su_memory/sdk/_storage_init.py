"""
sdk/_storage_init.py — 存储后端初始化工具 (共享模块)

v3.0.0: 从 SuMemoryLite / SuMemoryLitePro 提取公共初始化逻辑，
避免 ~55 行代码重复。
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Backend type string → BackendType enum mapping
_TYPE_MAP: dict = {}  # lazy filled to avoid circular import


def _get_type_map() -> dict:
    global _TYPE_MAP
    if not _TYPE_MAP:
        from su_memory._sys._storage_backend import BackendType
        _TYPE_MAP = {
            "sqlite": BackendType.SQLITE,
            "postgresql": BackendType.POSTGRESQL,
            "redis": BackendType.REDIS,
            "auto": BackendType.AUTO,
        }
    return _TYPE_MAP


def init_storage_backend(
    backend_type: str,
    storage_path: Optional[str],
    caller_name: str = "SDK",
) -> Optional[Any]:
    """
    初始化分布式存储后端 (同步包装)。

    从 SuMemoryLite / SuMemoryLitePro 的 __init__ 中调用，
    为 SDK 实例附加一个可选的分布式存储后端。

    Args:
        backend_type: 后端类型字符串 ("sqlite" / "postgresql" / "redis" / "auto")
        storage_path: 持久化目录路径
        caller_name: 调用方名称 (用于日志)

    Returns:
        StorageBackend 实例或 None
    """
    from su_memory._sys._storage_backend import StorageConfig, create_backend

    type_map = _get_type_map()
    bt = type_map.get(backend_type)
    if bt is None:
        logger.warning("%s: unknown backend type '%s', falling back to default", caller_name, backend_type)
        return None

    config = StorageConfig(
        sqlite_path=os.path.join(storage_path, "storage.db") if storage_path else None,
        backend_type=bt,
    )

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(
                create_backend(bt, config), loop
            )
            backend = future.result(timeout=10)
        else:
            backend = asyncio.run(create_backend(bt, config))
    except RuntimeError:
        backend = asyncio.run(create_backend(bt, config))
    except Exception as e:
        logger.warning("%s: storage backend '%s' init failed, using default: %s", caller_name, backend_type, e)
        return None

    if backend:
        logger.info("%s: storage backend '%s' initialized", caller_name, backend.backend_type.value)
    else:
        logger.info("%s: using default JSON persistence", caller_name)

    return backend
