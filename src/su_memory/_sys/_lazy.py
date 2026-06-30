"""
su-memory 懒加载工具

实现模块级别的延迟加载，减少启动时间。
通过 __getattr__ 代理，首次访问时才实际加载模块。

使用方式:
    # 在 __init__.py 中:
    from su_memory._sys._lazy import LazyModule
    _lazy_storage = LazyModule("su_memory.storage", [
        "SQLiteBackend", "MemoryItem", "AutoCompressor",
        "BackupManager", "DataExporter",
    ])

    # 用户代码不受影响:
    from su_memory import SQLiteBackend  # 正常
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


class _LazyProxy:
    """懒加载代理 — 延迟加载模块中指定符号

    用户 `from su_memory import SQLiteBackend` 时，
    第一次访问才触发 `import su_memory.storage`。
    """

    __slots__ = ("_module_name", "_symbols", "_module")

    def __init__(self, module_name: str, symbols: list[str]):
        self._module_name = module_name
        self._symbols = symbols
        self._module: Any | None = None

    def _load(self):
        """实际加载模块（仅一次）"""
        if self._module is not None:
            return
        try:
            self._module = importlib.import_module(self._module_name)
            logger.debug(f"Lazy loaded: {self._module_name}")
        except ImportError as e:
            logger.debug(f"Lazy import skipped ({self._module_name}): {e}")
            self._module = False  # Sentinel: import failed

    def __getattr__(self, name: str) -> Any:
        self._load()
        if self._module is False:
            return None
        if name in self._symbols:
            return getattr(self._module, name, None)
        raise AttributeError(f"Module '{self._module_name}' has no attribute '{name}'")

    def __dir__(self) -> list[str]:
        return list(self._symbols)


class LazyModule:
    """懒加载模块管理器

    在 __init__.py 中使用此管理器注入懒加载符号到模块命名空间。

    Example:
        >>> # 在 __init__.py 中:
        >>> _lazy = LazyModule(__name__)
        >>> _lazy.register("su_memory.storage", [
        ...     "SQLiteBackend", "MemoryItem"
        ... ])
        >>> _lazy.install()
    """

    def __init__(self, target_module: str):
        self._target_module = target_module
        self._entries: list[tuple[str, _LazyProxy]] = []

    def register(self, module_name: str, symbols: list[str]) -> LazyModule:
        """注册一个懒加载模块

        Args:
            module_name: 完整模块路径 (e.g. "su_memory.storage")
            symbols: 要导出的符号列表
        """
        proxy = _LazyProxy(module_name, symbols)
        self._entries.append((module_name, proxy))
        return self

    def install(self):
        """安装懒加载到目标模块"""
        import sys

        mod = sys.modules.get(self._target_module)
        if mod is None:
            raise RuntimeError(f"Module '{self._target_module}' not loaded yet")

        # 保存所有懒加载符号名
        lazy_symbols = {}
        for module_name, proxy in self._entries:
            for sym in proxy._symbols:
                lazy_symbols[sym] = proxy

        # 安装 __getattr__ 代理
        saved_getattr = getattr(mod, "__getattr__", None)

        def _module_getattr(name: str) -> Any:
            if name in lazy_symbols:
                return getattr(lazy_symbols[name], name)
            if saved_getattr is not None:
                return saved_getattr(name)
            raise AttributeError(
                f"module '{self._target_module}' has no attribute '{name}'"
            )

        mod.__getattr__ = _module_getattr

    def get_proxies(self) -> dict[str, _LazyProxy]:
        """返回 {module_name: proxy} 映射"""
        return {module_name: proxy for module_name, proxy in self._entries}
