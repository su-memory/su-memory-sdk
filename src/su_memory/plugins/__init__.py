"""
Official Plugins Package (官方插件包)

v1.7.0 W25-W26

提供官方维护的插件实现。

Available Plugins:
- TextEmbeddingPlugin: 文本嵌入插件
- RerankPlugin: 检索重排序插件
- MonitorPlugin: 性能监控插件
- PluginRegistry: 插件注册表

Available in su_memory._sys:
- PluginInterface: 插件抽象接口
- SandboxedExecutor: 沙箱执行器

Example:
    >>> from su_memory.plugins import TextEmbeddingPlugin
    >>> plugin = TextEmbeddingPlugin()
    >>> plugin.initialize({})

    >>> from su_memory.plugins import PluginRegistry
    >>> registry = PluginRegistry()
"""

from .embedding_plugin import (
    TextEmbeddingPlugin,
    HashVectorizer,
    create_text_embedding_plugin,
)

from .rerank_plugin import (
    RerankPlugin,
    RerankScorer,
    ScoreResult,
    create_rerank_plugin,
)

from .monitor_plugin import (
    MonitorPlugin,
    PerformanceMetrics,
    MonitorContext,
    create_monitor_plugin,
)

# 从 _sys 导入 PluginRegistry
from su_memory._sys._plugin_registry import PluginRegistry

__all__ = [
    # Embedding Plugin
    "TextEmbeddingPlugin",
    "HashVectorizer",
    "create_text_embedding_plugin",

    # Rerank Plugin
    "RerankPlugin",
    "RerankScorer",
    "ScoreResult",
    "create_rerank_plugin",

    # Monitor Plugin
    "MonitorPlugin",
    "PerformanceMetrics",
    "MonitorContext",
    "create_monitor_plugin",

    # Plugin Registry
    "PluginRegistry",
]
