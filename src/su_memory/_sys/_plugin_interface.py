"""
Plugin Interface Module (插件接口)

v1.7.0 W25-W26 插件系统核心模块

本模块定义了插件系统的核心接口和元数据结构：
- PluginInterface: 插件抽象基类
- PluginMetadata: 插件元数据
- PluginState: 插件状态枚举

Architecture:
- 抽象接口定义
- 标准化生命周期管理
- 配置模式支持

【Pre-Phase Numeric】- Uses prior ordering for numerical calculations
【Post-Phase Symbolic】- Uses post ordering for symbolic applications
"""

from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


# =============================================================================
# Enums
# =============================================================================

class PluginState(Enum):
    """插件状态枚举"""
    UNLOADED = "unloaded"      # 未加载
    LOADING = "loading"       # 加载中
    READY = "ready"           # 就绪
    RUNNING = "running"       # 运行中
    ERROR = "error"           # 错误状态
    UNLOADING = "unloading"   # 卸载中


class PluginType(Enum):
    """插件类型枚举"""
    EMBEDDING = "embedding"       # 嵌入插件
    RERANK = "rerank"           # 重排序插件
    MONITOR = "monitor"         # 监控插件
    PROCESSOR = "processor"     # 处理器插件
    CUSTOM = "custom"           # 自定义插件


# =============================================================================
# Plugin Metadata
# =============================================================================

@dataclass
class PluginMetadata:
    """
    插件元数据信息。

    包含插件的基本信息和依赖声明。

    Attributes:
        name: 插件名称（唯一标识）
        version: 插件版本号
        author: 作者信息
        description: 插件描述
        dependencies: 依赖的其他插件列表
        config_schema: 配置参数模式定义
        plugin_type: 插件类型
        entry_point: 入口类或函数名
        tags: 标签列表，用于分类和搜索
        created_at: 创建时间
        updated_at: 更新时间
    """
    name: str
    version: str
    author: str = "unknown"
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    plugin_type: PluginType = PluginType.CUSTOM
    entry_point: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """初始化后处理"""
        if self.created_at is None:
            object.__setattr__(self, 'created_at', datetime.now())
        if self.updated_at is None:
            object.__setattr__(self, 'updated_at', datetime.now())

    def validate(self) -> bool:
        """
        验证元数据完整性。

        Returns:
            True表示元数据有效
        """
        if not self.name or not self.name.strip():
            return False
        if not self.version:
            return False
        # 版本格式检查（简单检查）
        parts = self.version.split(".")
        if len(parts) != 3:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "dependencies": self.dependencies,
            "config_schema": self.config_schema,
            "plugin_type": self.plugin_type.value,
            "entry_point": self.entry_point,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginMetadata":
        """从字典创建元数据"""
        plugin_type_str = data.get("plugin_type", "custom")
        try:
            plugin_type = PluginType(plugin_type_str)
        except ValueError:
            plugin_type = PluginType.CUSTOM

        return cls(
            name=data.get("name", ""),
            version=data.get("version", "0.0.0"),
            author=data.get("author", "unknown"),
            description=data.get("description", ""),
            dependencies=data.get("dependencies", []),
            config_schema=data.get("config_schema", {}),
            plugin_type=plugin_type,
            entry_point=data.get("entry_point", ""),
            tags=data.get("tags", []),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


# =============================================================================
# Plugin Interface (Abstract Base Class)
# =============================================================================

class PluginInterface(ABC):
    """
    插件接口抽象基类。

    所有插件必须实现此接口定义的方法。

    Example:
        >>> class MyPlugin(PluginInterface):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_plugin"
        ...
        ...     @property
        ...     def version(self) -> str:
        ...         return "1.0.0"
        ...
        ...     def initialize(self, config: Dict) -> bool:
        ...         # 初始化逻辑
        ...         return True
        ...
        ...     def execute(self, context: Dict) -> Any:
        ...         # 执行逻辑
        ...         return result
        ...
        ...     def cleanup(self) -> None:
        ...         # 清理逻辑
        ...         pass
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """插件名称（唯一标识）"""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """插件版本号"""
        pass

    @property
    def description(self) -> str:
        """插件描述信息（可重写）"""
        return ""

    @property
    def author(self) -> str:
        """插件作者（可重写）"""
        return "unknown"

    @property
    def plugin_type(self) -> PluginType:
        """插件类型（可重写）"""
        return PluginType.CUSTOM

    @property
    def dependencies(self) -> List[str]:
        """依赖的插件列表（可重写）"""
        return []

    @property
    def config_schema(self) -> Dict[str, Any]:
        """配置参数模式（可重写）"""
        return {}

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        初始化插件。

        Args:
            config: 插件配置字典

        Returns:
            True表示初始化成功，False表示失败
        """
        pass

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Any:
        """
        执行插件核心逻辑。

        Args:
            context: 执行上下文字典

        Returns:
            执行结果（类型不限）
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        清理插件资源。

        在插件卸载前调用，用于释放资源、关闭连接等。
        """
        pass

    def get_metadata(self) -> PluginMetadata:
        """
        获取插件元数据。

        Returns:
            PluginMetadata对象
        """
        return PluginMetadata(
            name=self.name,
            version=self.version,
            author=self.author,
            description=self.description,
            dependencies=self.dependencies,
            config_schema=self.config_schema,
            plugin_type=self.plugin_type,
        )

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        验证配置是否有效。

        Args:
            config: 待验证的配置

        Returns:
            True表示配置有效
        """
        if not self.config_schema:
            return True

        # 简单验证：检查必需字段
        required_fields = self.config_schema.get("required", [])
        for field_name in required_fields:
            if field_name not in config:
                return False

        return True


# =============================================================================
# Plugin Events
# =============================================================================

@dataclass
class PluginEvent:
    """插件事件"""
    event_type: str
    plugin_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)


class PluginEventHandler:
    """插件事件处理器"""

    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}

    def register(self, event_type: str, handler: Callable[[PluginEvent], None]):
        """注册事件处理器"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def unregister(self, event_type: str, handler: Callable[[PluginEvent], None]):
        """注销事件处理器"""
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
            except ValueError:
                pass

    def emit(self, event: PluginEvent):
        """触发事件"""
        if event.event_type in self._handlers:
            for handler in self._handlers[event.event_type]:
                try:
                    handler(event)
                except Exception:
                    pass  # 静默忽略处理器错误

    def clear(self):
        """清除所有处理器"""
        self._handlers.clear()


# =============================================================================
# Utility Functions
# =============================================================================

def create_plugin_metadata(
    name: str,
    version: str,
    author: str = "unknown",
    description: str = "",
    dependencies: Optional[List[str]] = None,
    config_schema: Optional[Dict[str, Any]] = None,
    plugin_type: PluginType = PluginType.CUSTOM,
    entry_point: str = "",
    tags: Optional[List[str]] = None,
) -> PluginMetadata:
    """
    创建插件元数据的便捷函数。

    Args:
        name: 插件名称
        version: 版本号
        author: 作者
        description: 描述
        dependencies: 依赖列表
        config_schema: 配置模式
        plugin_type: 插件类型
        entry_point: 入口点
        tags: 标签

    Returns:
        PluginMetadata对象
    """
    return PluginMetadata(
        name=name,
        version=version,
        author=author,
        description=description,
        dependencies=dependencies or [],
        config_schema=config_schema or {},
        plugin_type=plugin_type,
        entry_point=entry_point,
        tags=tags or [],
    )


def validate_plugin(plugin: PluginInterface) -> bool:
    """
    验证插件是否实现了所有必需方法。

    Args:
        plugin: 插件实例

    Returns:
        True表示插件有效
    """
    try:
        # 检查属性
        _ = plugin.name
        _ = plugin.version

        # 检查方法
        if not callable(plugin.initialize):
            return False
        if not callable(plugin.execute):
            return False
        if not callable(plugin.cleanup):
            return False

        return True
    except Exception:
        return False


# =============================================================================
# Test Suite
# =============================================================================

def test_plugin_interface():
    """测试插件接口功能"""
    print("=" * 60)
    print("Testing Plugin Interface")
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

    # Test 1: PluginMetadata
    print("\n[Test 1] PluginMetadata")
    print("-" * 40)

    metadata = PluginMetadata(
        name="test_plugin",
        version="1.0.0",
        author="Test Author",
        description="Test plugin description",
        dependencies=["base_plugin"],
        plugin_type=PluginType.CUSTOM,
    )

    test("创建元数据", metadata.name == "test_plugin")
    test("验证元数据", metadata.validate())
    test("元数据转字典", "name" in metadata.to_dict())

    # Test 2: 从字典创建元数据
    print("\n[Test 2] From Dict")
    print("-" * 40)

    data = {
        "name": "dict_plugin",
        "version": "2.0.0",
        "author": "Dict Author",
        "plugin_type": "embedding",
    }
    metadata2 = PluginMetadata.from_dict(data)
    test("从字典创建", metadata2.name == "dict_plugin")
    test("插件类型转换", metadata2.plugin_type == PluginType.EMBEDDING)

    # Test 3: PluginInterface 实现
    print("\n[Test 3] PluginInterface Implementation")
    print("-" * 40)

    class TestPlugin(PluginInterface):
        def __init__(self):
            self._initialized = False

        @property
        def name(self) -> str:
            return "test_impl_plugin"

        @property
        def version(self) -> str:
            return "1.0.0"

        @property
        def description(self) -> str:
            return "Test implementation"

        def initialize(self, config: Dict[str, Any]) -> bool:
            self._initialized = True
            return True

        def execute(self, context: Dict[str, Any]) -> Any:
            return {"result": "executed", "context": context}

        def cleanup(self) -> None:
            self._initialized = False

    plugin = TestPlugin()
    test("插件验证", validate_plugin(plugin))
    test("初始化", plugin.initialize({}))
    test("执行", plugin.execute({"input": "test"}) is not None)
    test("获取元数据", plugin.get_metadata().name == "test_impl_plugin")

    # Test 4: 事件处理器
    print("\n[Test 4] PluginEventHandler")
    print("-" * 40)

    handler = PluginEventHandler()
    event_received = {"flag": False}

    def test_handler(event: PluginEvent):
        event_received["flag"] = True

    handler.register("test_event", test_handler)
    handler.emit(PluginEvent(event_type="test_event", plugin_name="test"))
    test("事件触发", event_received["flag"])

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = test_plugin_interface()
    exit(0 if success else 1)
