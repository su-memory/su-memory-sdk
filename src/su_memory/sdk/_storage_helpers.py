"""
v3.0.0: 存储后端初始化辅助模块

提供 SuMemoryLite / SuMemoryLitePro 共享的 `_init_storage_backend` 逻辑。
消除 lite.py 和 lite_pro.py 之间的代码重复 (QC P1)。
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# 后端类型 → BackendType 映射 (lite/lite_pro 共用)
_TYPE_MAP: Dict[str, Any] = {}

def _get_type_map():
    """延迟加载 BackendType 避免循环导入"""
    if not _TYPE_MAP:
        from su_memory._sys._storage_backend import BackendType
        _TYPE_MAP.update({
            "sqlite": BackendType.SQLITE,
            "postgresql": BackendType.POSTGRESQL,
            "redis": BackendType.REDIS,
            "auto": BackendType.AUTO,
        })
    return _TYPE_MAP


def init_storage_backend(
    instance: Any,
    backend_type: str,
    storage_path: Optional[str],
    label: str = "SDK",
) -> None:
    """
    初始化分布式存储后端 (同步包装)。

    供 SuMemoryLite / SuMemoryLitePro 的 __init__ 调用。

    Args:
        instance: SDK 实例 (设置了 _storage_backend / _storage_backend_type 属性)
        backend_type: 后端类型 ("sqlite" / "postgresql" / "redis" / "auto")
        storage_path: 持久化路径 (用于 SQLite 路径推导)
        label: 日志标签 (如 "SuMemoryLite" / "SuMemoryLitePro")
    """
    from su_memory._sys._storage_backend import StorageConfig, create_backend

    type_map = _get_type_map()
    bt = type_map.get(backend_type, type_map["sqlite"])

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
            instance._storage_backend = future.result(timeout=10)
        else:
            instance._storage_backend = asyncio.run(create_backend(bt, config))
    except RuntimeError:
        instance._storage_backend = asyncio.run(create_backend(bt, config))
    except Exception as e:
        logger.warning(
            "Storage backend '%s' init failed for %s, using default: %s",
            backend_type, label, e,
        )
        instance._storage_backend = None

    if instance._storage_backend:
        logger.info(
            "%s: storage backend '%s' initialized",
            label, instance._storage_backend.backend_type.value,
        )
    else:
        logger.info("%s: using default JSON persistence", label)
