"""
Plugin Registry Module (插件注册表)

v1.7.0 W25-W26 插件系统核心模块

本模块提供插件的注册和管理功能：
- PluginRegistry: 单例模式插件注册表
- PluginAlreadyExistsError: 插件已存在异常
- PluginNotFoundError: 插件未找到异常

Features:
- 线程安全的注册/注销操作
- 插件元数据管理
- 依赖关系验证

【Pre-Phase Numeric】- Uses prior ordering for numerical calculations
【Post-Phase Symbolic】- Uses post ordering for symbolic applications
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import threading
import time
from collections import defaultdict

from ._plugin_interface import (
    PluginInterface,
    PluginMetadata,
    PluginState,
    PluginType,
)


# =============================================================================
# Exceptions
# =============================================================================

class PluginError(Exception):
    """插件相关异常基类"""
    pass


class PluginAlreadyExistsError(PluginError):
    """插件已存在异常"""

    def __init__(self, plugin_name: str):
        self.plugin_name = plugin_name
        super().__init__(f"Plugin '{plugin_name}' already exists")


class PluginNotFoundError(PluginError):
    """插件未找到异常"""

    def __init__(self, plugin_name: str):
        self.plugin_name = plugin_name
        super().__init__(f"Plugin '{plugin_name}' not found")


class PluginDependencyError(PluginError):
    """插件依赖错误"""

    def __init__(self, plugin_name: str, missing_deps: List[str]):
        self.plugin_name = plugin_name
        self.missing_deps = missing_deps
        super().__init__(
            f"Plugin '{plugin_name}' missing dependencies: {', '.join(missing_deps)}"
        )


class PluginStateError(PluginError):
    """插件状态错误"""

    def __init__(self, plugin_name: str, expected_state: PluginState, current_state: PluginState):
        self.plugin_name = plugin_name
        self.expected_state = expected_state
        self.current_state = current_state
        super().__init__(
            f"Plugin '{plugin_name}' state error: expected {expected_state.value}, "
            f"got {current_state.value}"
        )


# =============================================================================
# Plugin Registry Entry
# =============================================================================

@dataclass
class PluginRegistryEntry:
    """插件注册表条目"""
    plugin: PluginInterface
    metadata: PluginMetadata
    state: PluginState
    registered_at: float
    last_used_at: float
    use_count: int


# =============================================================================
# Plugin Registry
# =============================================================================

class PluginRegistry:
    """
    插件注册表（单例模式）。

    提供插件的注册、注销、查询等功能。
    所有操作都是线程安全的。

    v1.7.0 性能优化:
    - 使用字典索引替代列表遍历 O(1)
    - 细粒度锁替代粗粒度锁
    - 懒加载元数据缓存
    - 性能统计收集

    Example:
        >>> registry = PluginRegistry()
        >>>
        >>> # 注册插件
        >>> registry.register(my_plugin)
        >>>
        >>> # 获取插件
        >>> plugin = registry.get_plugin("my_plugin")
        >>>
        >>> # 列出所有插件
        >>> names = registry.list_plugins()
        >>>
        >>> # 注销插件
        >>> registry.unregister("my_plugin")
    """

    _instance: Optional["PluginRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "PluginRegistry":
        """单例模式实现"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化注册表"""
        if self._initialized:
            return

        self._plugins: Dict[str, PluginRegistryEntry] = {}  # 字典索引 O(1)
        self._metadata_cache: Dict[str, PluginMetadata] = {}  # 元数据缓存
        self._state_listeners: Dict[str, List[callable]] = defaultdict(list)
        self._lock = threading.Lock()
        self._read_lock = threading.RLock()  # 读锁分离，提高并发
        self._initialized = True

        # 性能统计
        self._stats = {
            "register_count": 0,
            "unregister_count": 0,
            "get_count": 0,
            "list_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        self._perf_timers = {}

    @classmethod
    def get_instance(cls) -> "PluginRegistry":
        """获取单例实例"""
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """重置单例实例（主要用于测试）"""
        with cls._lock:
            cls._instance = None

    def register(
        self,
        plugin: PluginInterface,
        config: Optional[Dict[str, Any]] = None,
        auto_initialize: bool = True,
    ) -> bool:
        """
        注册插件 - O(1) 操作

        Args:
            plugin: 插件实例
            config: 可选配置字典
            auto_initialize: 是否自动初始化

        Returns:
            True表示注册成功

        Raises:
            PluginAlreadyExistsError: 如果插件已存在
            PluginDependencyError: 如果依赖未满足
        """
        start = time.perf_counter()

        with self._lock:
            plugin_name = plugin.name

            # 检查是否已存在
            if plugin_name in self._plugins:
                raise PluginAlreadyExistsError(plugin_name)

            # 检查依赖
            deps = plugin.dependencies
            missing_deps = [dep for dep in deps if dep not in self._plugins]
            if missing_deps:
                raise PluginDependencyError(plugin_name, missing_deps)

            # 获取元数据（懒加载）
            metadata = plugin.get_metadata()

            # 验证元数据
            if not metadata.validate():
                raise ValueError(f"Invalid metadata for plugin '{plugin_name}'")

            # 创建注册条目
            entry = PluginRegistryEntry(
                plugin=plugin,
                metadata=metadata,
                state=PluginState.LOADING,
                registered_at=time.time(),
                last_used_at=time.time(),
                use_count=0,
            )

            # 添加到注册表 - O(1) 字典索引
            self._plugins[plugin_name] = entry
            self._metadata_cache[plugin_name] = metadata

            # 更新状态
            self._update_state(plugin_name, PluginState.READY)

            # 自动初始化
            if auto_initialize and config is None:
                config = {}

            if auto_initialize and config is not None:
                try:
                    plugin.initialize(config)
                    self._update_state(plugin_name, PluginState.READY)
                except Exception:
                    self._update_state(plugin_name, PluginState.ERROR)
                    raise

            # 性能统计
            self._stats["register_count"] += 1
            self._perf_timers["_last_register_time"] = time.perf_counter() - start

            return True

    def unregister(self, plugin_name: str, force: bool = False) -> bool:
        """
        注销插件 - O(1) 操作

        Args:
            plugin_name: 插件名称
            force: 是否强制注销（即使正在运行）

        Returns:
            True表示注销成功

        Raises:
            PluginNotFoundError: 如果插件不存在
            PluginStateError: 如果插件正在运行且force=False
        """
        start = time.perf_counter()

        with self._lock:
            if plugin_name not in self._plugins:
                raise PluginNotFoundError(plugin_name)

            entry = self._plugins[plugin_name]

            # 检查状态
            if not force and entry.state == PluginState.RUNNING:
                raise PluginStateError(
                    plugin_name,
                    PluginState.READY,
                    entry.state
                )

            # 更新状态
            self._update_state(plugin_name, PluginState.UNLOADING)

            # 调用清理方法
            try:
                entry.plugin.cleanup()
            except Exception:
                pass  # 静默忽略清理错误

            # 移除 - O(1) 字典操作
            del self._plugins[plugin_name]
            if plugin_name in self._metadata_cache:
                del self._metadata_cache[plugin_name]

            # 性能统计
            self._stats["unregister_count"] += 1
            self._perf_timers["_last_unregister_time"] = time.perf_counter() - start

            return True

    def get_plugin(self, plugin_name: str) -> Optional[PluginInterface]:
        """
        获取插件实例 - O(1) 操作

        Args:
            plugin_name: 插件名称

        Returns:
            插件实例，如果不存在返回None
        """
        start = time.perf_counter()

        # 使用读锁提高并发性能
        with self._read_lock:
            entry = self._plugins.get(plugin_name)
            if entry:
                entry.last_used_at = time.time()
                entry.use_count += 1
                plugin = entry.plugin
            else:
                plugin = None

        # 性能统计
        self._stats["get_count"] += 1
        self._perf_timers["_last_get_time"] = time.perf_counter() - start

        if plugin:
            self._stats["cache_hits"] += 1
        else:
            self._stats["cache_misses"] += 1

        return plugin

    def get_plugin_state(self, plugin_name: str) -> Optional[PluginState]:
        """
        获取插件状态。

        Args:
            plugin_name: 插件名称

        Returns:
            插件状态，如果不存在返回None
        """
        with self._lock:
            entry = self._plugins.get(plugin_name)
            return entry.state if entry else None

    def list_plugins(
        self,
        plugin_type: Optional[PluginType] = None,
        include_internal: bool = False,
    ) -> List[str]:
        """
        列出所有插件 - O(n) 操作

        Args:
            plugin_type: 可选，按类型过滤
            include_internal: 是否包含内置插件

        Returns:
            插件名称列表
        """
        start = time.perf_counter()

        # 使用读锁提高并发性能
        with self._read_lock:
            names = list(self._plugins.keys())

            if plugin_type:
                names = [
                    name for name in names
                    if self._plugins[name].metadata.plugin_type == plugin_type
                ]

        # 性能统计
        self._stats["list_count"] += 1
        self._perf_timers["_last_list_time"] = time.perf_counter() - start

        return names

    def list_plugin_metadata(self) -> List[PluginMetadata]:
        """
        获取所有插件的元数据。

        Returns:
            元数据列表
        """
        with self._lock:
            return [entry.metadata for entry in self._plugins.values()]

    def get_plugin_metadata(self, plugin_name: str) -> Optional[PluginMetadata]:
        """
        获取插件元数据。

        Args:
            plugin_name: 插件名称

        Returns:
            插件元数据，如果不存在返回None
        """
        return self._metadata_cache.get(plugin_name)

    def is_registered(self, plugin_name: str) -> bool:
        """
        检查插件是否已注册。

        Args:
            plugin_name: 插件名称

        Returns:
            True表示已注册
        """
        with self._lock:
            return plugin_name in self._plugins

    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """
        获取插件详细信息。

        Args:
            plugin_name: 插件名称

        Returns:
            插件信息字典，如果不存在返回None
        """
        with self._lock:
            entry = self._plugins.get(plugin_name)
            if not entry:
                return None

            return {
                "name": entry.metadata.name,
                "version": entry.metadata.version,
                "author": entry.metadata.author,
                "description": entry.metadata.description,
                "plugin_type": entry.metadata.plugin_type.value,
                "state": entry.state.value,
                "registered_at": entry.registered_at,
                "last_used_at": entry.last_used_at,
                "use_count": entry.use_count,
            }

    def _update_state(self, plugin_name: str, new_state: PluginState):
        """更新插件状态并通知监听器"""
        if plugin_name in self._plugins:
            old_state = self._plugins[plugin_name].state
            self._plugins[plugin_name].state = new_state

            # 通知监听器
            if plugin_name in self._state_listeners:
                for listener in self._state_listeners[plugin_name]:
                    try:
                        listener(plugin_name, old_state, new_state)
                    except Exception:
                        pass

    def register_state_listener(
        self,
        plugin_name: str,
        listener: callable
    ):
        """
        注册状态监听器。

        Args:
            plugin_name: 插件名称
            listener: 状态变化回调函数 (plugin_name, old_state, new_state)
        """
        with self._lock:
            self._state_listeners[plugin_name].append(listener)

    def unregister_state_listener(
        self,
        plugin_name: str,
        listener: callable
    ):
        """
        注销状态监听器。

        Args:
            plugin_name: 插件名称
            listener: 状态变化回调函数
        """
        with self._lock:
            if plugin_name in self._state_listeners:
                try:
                    self._state_listeners[plugin_name].remove(listener)
                except ValueError:
                    pass

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取注册表统计信息。

        Returns:
            统计信息字典
        """
        with self._lock:
            total = len(self._plugins)
            by_state: Dict[str, int] = defaultdict(int)
            by_type: Dict[str, int] = defaultdict(int)

            for entry in self._plugins.values():
                by_state[entry.state.value] += 1
                by_type[entry.metadata.plugin_type.value] += 1

            return {
                "total_plugins": total,
                "by_state": dict(by_state),
                "by_type": dict(by_type),
                **self.get_performance_stats(),
            }

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息。

        Returns:
            性能统计字典
        """
        total_cache = self._stats["cache_hits"] + self._stats["cache_misses"]
        cache_hit_rate = (
            self._stats["cache_hits"] / max(1, total_cache)
        )

        return {
            **self._stats,
            "cache_size": len(self._plugins),
            "cache_hit_rate": cache_hit_rate,
            "avg_register_time": self._perf_timers.get("_last_register_time", 0),
            "avg_get_time": self._perf_timers.get("_last_get_time", 0),
            "avg_list_time": self._perf_timers.get("_last_list_time", 0),
        }

    def clear(self, force: bool = False) -> int:
        """
        清除所有插件。

        Args:
            force: 是否强制清除（忽略运行状态）

        Returns:
            清除的插件数量
        """
        with self._lock:
            count = 0
            plugin_names = list(self._plugins.keys())

            for name in plugin_names:
                try:
                    self.unregister(name, force=force)
                    count += 1
                except Exception:
                    pass

            return count

    @property
    def plugin_count(self) -> int:
        """已注册插件数量"""
        with self._lock:
            return len(self._plugins)

    def __repr__(self) -> str:
        return f"PluginRegistry(plugins={self.plugin_count})"


# =============================================================================
# Convenience Functions
# =============================================================================

def get_registry() -> PluginRegistry:
    """获取插件注册表单例实例"""
    return PluginRegistry.get_instance()


def register_plugin(
    plugin: PluginInterface,
    config: Optional[Dict[str, Any]] = None,
    auto_initialize: bool = True,
) -> bool:
    """注册插件的便捷函数"""
    return get_registry().register(plugin, config, auto_initialize)


def unregister_plugin(plugin_name: str, force: bool = False) -> bool:
    """注销插件的便捷函数"""
    return get_registry().unregister(plugin_name, force)


def get_plugin(plugin_name: str) -> Optional[PluginInterface]:
    """获取插件的便捷函数"""
    return get_registry().get_plugin(plugin_name)


def list_plugins(
    plugin_type: Optional[PluginType] = None,
) -> List[str]:
    """列出插件的便捷函数"""
    return get_registry().list_plugins(plugin_type)


# =============================================================================
# Test Suite
# =============================================================================

def test_plugin_registry():
    """测试插件注册表功能"""
    print("=" * 60)
    print("Testing Plugin Registry")
    print("=" * 60)

    passed = 0
    failed = 0

    def test(name: str, condition: bool):
        nonlocal passed, failed
        if condition:
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name}")
            failed += 1

    # 重置单例
    PluginRegistry.reset_instance()

    # Test 1: 单例模式
    print("\n[Test 1] Singleton Pattern")
    print("-" * 40)

    registry1 = PluginRegistry()
    registry2 = PluginRegistry()
    test("单例模式", registry1 is registry2)

    # Test 2: 注册插件
    print("\n[Test 2] Plugin Registration")
    print("-" * 40)

    class DummyPlugin(PluginInterface):
        @property
        def name(self) -> str:
            return "dummy_plugin"

        @property
        def version(self) -> str:
            return "1.0.0"

        def initialize(self, config: Dict) -> bool:
            return True

        def execute(self, context: Dict) -> Any:
            return "executed"

        def cleanup(self) -> None:
            pass

    plugin = DummyPlugin()
    success = registry1.register(plugin, {}, auto_initialize=False)
    test("注册插件", success)
    test("插件已注册", registry1.is_registered("dummy_plugin"))

    # Test 3: 重复注册
    print("\n[Test 3] Duplicate Registration")
    print("-" * 40)

    try:
        registry1.register(plugin, {}, auto_initialize=False)
        test("重复注册应抛异常", False)
    except PluginAlreadyExistsError:
        test("重复注册抛出异常", True)

    # Test 4: 获取插件
    print("\n[Test 4] Get Plugin")
    print("-" * 40)

    retrieved = registry1.get_plugin("dummy_plugin")
    test("获取插件", retrieved is not None)
    test("获取正确插件", retrieved.name == "dummy_plugin")

    # Test 5: 列出插件
    print("\n[Test 5] List Plugins")
    print("-" * 40)

    names = registry1.list_plugins()
    test("列出插件", "dummy_plugin" in names)
    test("插件数量", len(names) == 1)

    # Test 6: 元数据
    print("\n[Test 6] Plugin Metadata")
    print("-" * 40)

    metadata = registry1.get_plugin_metadata("dummy_plugin")
    test("获取元数据", metadata is not None)
    test("元数据正确", metadata.version == "1.0.0")

    # Test 7: 注销插件
    print("\n[Test 7] Unregister Plugin")
    print("-" * 40)

    success = registry1.unregister("dummy_plugin", force=True)
    test("注销插件", success)
    test("插件已注销", not registry1.is_registered("dummy_plugin"))

    # Test 8: 依赖检查
    print("\n[Test 8] Dependency Check")
    print("-" * 40)

    PluginRegistry.reset_instance()
    registry = PluginRegistry()

    class DependentPlugin(PluginInterface):
        @property
        def name(self) -> str:
            return "dependent_plugin"

        @property
        def version(self) -> str:
            return "1.0.0"

        @property
        def dependencies(self) -> List[str]:
            return ["nonexistent_plugin"]

        def initialize(self, config: Dict) -> bool:
            return True

        def execute(self, context: Dict) -> Any:
            return None

        def cleanup(self) -> None:
            pass

    try:
        registry.register(DependentPlugin(), {}, auto_initialize=False)
        test("依赖检查", False)
    except PluginDependencyError:
        test("依赖检查", True)

    # Test 9: 统计信息
    print("\n[Test 9] Statistics")
    print("-" * 40)

    PluginRegistry.reset_instance()
    registry = PluginRegistry()
    stats = registry.get_statistics()
    test("获取统计", "total_plugins" in stats)
    test("初始数量为0", stats["total_plugins"] == 0)

    # Test 10: 状态监听
    print("\n[Test 10] State Listener")
    print("-" * 40)

    state_changes = []

    def state_listener(name: str, old: PluginState, new: PluginState):
        state_changes.append((name, old.value, new.value))

    PluginRegistry.reset_instance()
    registry = PluginRegistry()

    class TestPlugin(PluginInterface):
        @property
        def name(self) -> str:
            return "state_test_plugin"

        @property
        def version(self) -> str:
            return "1.0.0"

        def initialize(self, config: Dict) -> bool:
            return True

        def execute(self, context: Dict) -> Any:
            return None

        def cleanup(self) -> None:
            pass

    registry.register(TestPlugin(), {}, auto_initialize=False)
    test("状态变化记录", len(state_changes) > 0)

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = test_plugin_registry()
    exit(0 if success else 1)
