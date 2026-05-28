"""
PluginManager — 插件统一启动器

v3.0.0: 统一管理 _sys/ 50+ 模块的插件化生命周期。

Features:
- auto_discover(): 扫描 _sys/plugins/ 自动注册所有插件
- get_core_plugins(): 返回核心引擎插件集合
- hot_reload(): 不重启替换单个插件
- health_report(): 所有插件状态汇总
"""

import importlib
import logging
import os
from typing import Any, Dict, List, Optional, Type

from su_memory._sys._plugin_interface import (
    PluginInterface,
    PluginMetadata,
    PluginState,
    PluginType,
    create_plugin_metadata,
)
from su_memory._sys._plugin_registry import PluginRegistry, get_registry

logger = logging.getLogger(__name__)


class ModulePluginAdapter(PluginInterface):
    """
    通用模块插件适配器。

    将任意 _sys/ 模块包装为 PluginInterface，避免为每个模块
    写单独的包装类。支持懒加载和生命周期管理。
    """

    def __init__(
        self,
        name: str,
        module_path: str,
        plugin_type: PluginType = PluginType.UTILITY,
        description: str = "",
        version: str = "3.0.0",
        dependencies: Optional[List[str]] = None,
    ):
        self._name = name
        self._module_path = module_path
        self._plugin_type = plugin_type
        self._description = description
        self._version = version
        self._dependencies = dependencies or []
        self._module = None
        self._config: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def description(self) -> str:
        return self._description

    @property
    def plugin_type(self) -> PluginType:
        return self._plugin_type

    @property
    def dependencies(self) -> List[str]:
        return self._dependencies

    def initialize(self, config: Dict[str, Any]) -> bool:
        """懒加载模块"""
        try:
            self._config = config
            self._module = importlib.import_module(self._module_path)
            return True
        except ImportError as e:
            logger.warning(f"Failed to load plugin {self._name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Plugin {self._name} initialization error: {e}")
            return False

    def execute(self, context: Dict[str, Any]) -> Any:
        """执行插件功能 — 返回模块引用供直接使用"""
        if self._module is None:
            raise RuntimeError(f"Plugin {self._name} not initialized")
        return self._module

    def cleanup(self) -> None:
        """清理"""
        self._module = None
        self._config = {}

    def get_module(self) -> Any:
        """获取底层模块引用"""
        if self._module is None:
            raise RuntimeError(f"Plugin {self._name} not initialized. Call initialize() first.")
        return self._module


# ============================================================
# Plugin Manifest — 所有 _sys/ 模块的注册清单
# ============================================================

PLUGIN_MANIFEST: List[Dict[str, Any]] = [
    # === Core Engines (8) ===
    {"name": "energy_bus", "module": "su_memory._sys._energy_bus", "type": PluginType.EMBEDDING,
     "desc": "Energy bus — central energy routing and dispatch"},
    {"name": "energy_core", "module": "su_memory._sys._energy_core", "type": PluginType.EMBEDDING,
     "desc": "Energy core — five-element energy calculation engine"},
    {"name": "causal_engine", "module": "su_memory._sys._causal_engine", "type": PluginType.EMBEDDING,
     "desc": "Causal engine — causality inference and chain analysis"},
    {"name": "temporal_core", "module": "su_memory._sys._temporal_core", "type": PluginType.EMBEDDING,
     "desc": "Temporal core — time encoding and branch analysis"},
    {"name": "category_core", "module": "su_memory._sys._category_core", "type": PluginType.EMBEDDING,
     "desc": "Category core — trigram and semantic category engine"},
    {"name": "spacetime_index", "module": "su_memory._sys._spacetime_index", "type": PluginType.EMBEDDING,
     "desc": "Spacetime index — spatial-temporal indexing"},
    {"name": "async_embedder", "module": "su_memory._sys._async_embedder", "type": PluginType.EMBEDDING,
     "desc": "Async embedder — asynchronous embedding computation"},
    {"name": "energy_relations", "module": "su_memory._sys._energy_relations", "type": PluginType.EMBEDDING,
     "desc": "Energy relations — energy type relationship analysis"},

    # === Processor/Pipeline (12) ===
    {"name": "pattern_inference", "module": "su_memory._sys._pattern_inference", "type": PluginType.PROCESSOR,
     "desc": "Pattern inference — cross-memory pattern mining"},
    {"name": "adaptive_engine", "module": "su_memory._sys._adaptive_engine", "type": PluginType.PROCESSOR,
     "desc": "Adaptive engine — self-adaptive parameter tuning"},
    {"name": "incremental_learning", "module": "su_memory._sys._incremental_learning", "type": PluginType.PROCESSOR,
     "desc": "Incremental learning — online learning engine"},
    {"name": "dimension_map", "module": "su_memory._sys._dimension_map", "type": PluginType.PROCESSOR,
     "desc": "Dimension map — multi-dimensional projection and mapping"},
    {"name": "parameter_adapters", "module": "su_memory._sys._parameter_adapters", "type": PluginType.PROCESSOR,
     "desc": "Parameter adapters — runtime parameter adaptation"},
    {"name": "stream", "module": "su_memory._sys._stream", "type": PluginType.PROCESSOR,
     "desc": "Stream — streaming data pipeline"},
    {"name": "faiss_tuner", "module": "su_memory._sys._faiss_tuner", "type": PluginType.PROCESSOR,
     "desc": "FAISS tuner — vector index auto-tuning"},
    {"name": "embedding_cache", "module": "su_memory._sys._embedding_cache", "type": PluginType.PROCESSOR,
     "desc": "Embedding cache — embedding computation cache"},
    {"name": "lazy", "module": "su_memory._sys._lazy", "type": PluginType.PROCESSOR,
     "desc": "Lazy loader — deferred module loading"},
    {"name": "local_models", "module": "su_memory._sys._local_models", "type": PluginType.PROCESSOR,
     "desc": "Local models — offline model management"},
    {"name": "enums", "module": "su_memory._sys._enums", "type": PluginType.PROCESSOR,
     "desc": "Enums — shared enumeration types"},
    {"name": "terms", "module": "su_memory._sys._terms", "type": PluginType.PROCESSOR,
     "desc": "Terms — terminology mapping configuration"},

    # === Reasoning/Analysis (8) ===
    {"name": "bayesian", "module": "su_memory._sys.bayesian", "type": PluginType.REASONING,
     "desc": "Bayesian — Bayesian inference engine"},
    {"name": "bayesian_network", "module": "su_memory._sys.bayesian_network", "type": PluginType.REASONING,
     "desc": "Bayesian network — probabilistic graphical model"},
    {"name": "bayesian_reasoning", "module": "su_memory._sys.bayesian_reasoning", "type": PluginType.REASONING,
     "desc": "Bayesian reasoning — advanced Bayesian reasoning"},
    {"name": "causal", "module": "su_memory._sys.causal", "type": PluginType.REASONING,
     "desc": "Causal — causality analysis module"},
    {"name": "evidence", "module": "su_memory._sys.evidence", "type": PluginType.REASONING,
     "desc": "Evidence — evidence chain verification"},
    {"name": "multi_hop", "module": "su_memory._sys.multi_hop", "type": PluginType.REASONING,
     "desc": "Multi-hop — multi-hop reasoning engine"},
    {"name": "intent_classifier", "module": "su_memory._sys.intent_classifier", "type": PluginType.REASONING,
     "desc": "Intent classifier — query intent classification"},
    {"name": "fusion", "module": "su_memory._sys.fusion", "type": PluginType.REASONING,
     "desc": "Fusion — multi-source fusion scoring"},

    # === Monitor (7) ===
    {"name": "recall_trigger", "module": "su_memory._sys.recall_trigger", "type": PluginType.MONITOR,
     "desc": "Recall trigger — proactive memory recall"},
    {"name": "priority_boost", "module": "su_memory._sys.priority_boost", "type": PluginType.MONITOR,
     "desc": "Priority boost — memory priority weighting"},
    {"name": "recency_feedback", "module": "su_memory._sys.recency_feedback", "type": PluginType.MONITOR,
     "desc": "Recency feedback — recency-based feedback loop"},
    {"name": "awareness", "module": "su_memory._sys.awareness", "type": PluginType.MONITOR,
     "desc": "Awareness — system awareness monitoring"},
    {"name": "session_bridge", "module": "su_memory._sys.session_bridge", "type": PluginType.MONITOR,
     "desc": "Session bridge — cross-session memory bridge"},
    {"name": "wiki_linker", "module": "su_memory._sys.wiki_linker", "type": PluginType.MONITOR,
     "desc": "Wiki linker — knowledge wiki integration"},
    {"name": "meta_cognition", "module": "su_memory._sys.meta_cognition", "type": PluginType.MONITOR,
     "desc": "Meta-cognition — self-reflective monitoring"},

    # === Utility (18) ===
    {"name": "migrator", "module": "su_memory._sys.migrator", "type": PluginType.UTILITY,
     "desc": "Migrator — database schema migration"},
    {"name": "fallback", "module": "su_memory._sys.fallback", "type": PluginType.UTILITY,
     "desc": "Fallback — degradation fallback matrix"},
    {"name": "error_hints", "module": "su_memory._sys.error_hints", "type": PluginType.UTILITY,
     "desc": "Error hints — diagnostic error messages"},
    {"name": "codec", "module": "su_memory._sys.codec", "type": PluginType.UTILITY,
     "desc": "Codec — content encoding/decoding"},
    {"name": "encoders", "module": "su_memory._sys.encoders", "type": PluginType.UTILITY,
     "desc": "Encoders — 64-pattern encoding system"},
    {"name": "states", "module": "su_memory._sys.states", "type": PluginType.UTILITY,
     "desc": "States — state machine management"},
    {"name": "chrono", "module": "su_memory._sys.chrono", "type": PluginType.UTILITY,
     "desc": "Chrono — chronological encoding"},
    {"name": "license", "module": "su_memory._sys.license", "type": PluginType.UTILITY,
     "desc": "License — license validation"},
    {"name": "embedder", "module": "su_memory._sys.embedder", "type": PluginType.UTILITY,
     "desc": "Embedder — embedding computation"},
    {"name": "progressive_disclosure", "module": "su_memory._sys.progressive_disclosure", "type": PluginType.UTILITY,
     "desc": "Progressive disclosure — layered information access"},
    {"name": "c1", "module": "su_memory._sys._c1", "type": PluginType.UTILITY,
     "desc": "C1 — core utilities (level 1)"},
    {"name": "c2", "module": "su_memory._sys._c2", "type": PluginType.UTILITY,
     "desc": "C2 — core utilities (level 2)"},
    {"name": "time_code", "module": "su_memory._sys._time_code", "type": PluginType.UTILITY,
     "desc": "Time code — time encoding utilities"},
    {"name": "unified_unit", "module": "su_memory._sys._unified_unit", "type": PluginType.UTILITY,
     "desc": "Unified unit — unified information unit"},
    {"name": "energy_relations_sys", "module": "su_memory._sys._energy_relations", "type": PluginType.UTILITY,
     "desc": "Energy relations — energy relation utilities"},
    {"name": "dimension_map_sys", "module": "su_memory._sys._dimension_map", "type": PluginType.UTILITY,
     "desc": "Dimension map — dimension projection (sys level)"},
    {"name": "category_core_sys", "module": "su_memory._sys._category_core", "type": PluginType.UTILITY,
     "desc": "Category core — trigram core (sys level)"},
    {"name": "temporal_core_sys", "module": "su_memory._sys._temporal_core", "type": PluginType.UTILITY,
     "desc": "Temporal core — temporal engine (sys level)"},
]


# ============================================================
# PluginManager
# ============================================================

class PluginManager:
    """
    插件统一管理器。

    负责所有 _sys/ 模块的插件注册、生命周期管理、
    热重载和健康监控。

    Example:
        >>> pm = PluginManager()
        >>> pm.auto_discover()
        >>> report = pm.health_report()
        >>> pm.get_core_plugins()
    """

    def __init__(self):
        self._registry = get_registry()
        self._adapters: Dict[str, ModulePluginAdapter] = {}
        self._initialized = False

    def auto_discover(self, config: Optional[Dict[str, Any]] = None) -> int:
        """
        自动扫描 PLUGIN_MANIFEST 并注册所有插件。

        Args:
            config: 全局配置，应用于所有插件

        Returns:
            成功注册的插件数量
        """
        if self._initialized:
            logger.info("PluginManager already initialized, skipping auto_discover")
            return len(self._adapters)

        count = 0
        base_config = config or {}

        for entry in PLUGIN_MANIFEST:
            try:
                adapter = ModulePluginAdapter(
                    name=entry["name"],
                    module_path=entry["module"],
                    plugin_type=entry["type"],
                    description=entry["desc"],
                )

                if self._registry.is_registered(entry["name"]):
                    logger.debug(f"Plugin {entry['name']} already registered, skipping")
                    continue

                self._registry.register(adapter, base_config, auto_initialize=False)
                self._adapters[entry["name"]] = adapter
                count += 1
            except Exception as e:
                logger.error(f"Failed to register plugin {entry['name']}: {e}")

        self._initialized = True
        logger.info(f"PluginManager auto_discover: {count} plugins registered")
        return count

    def initialize_all(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        """
        懒加载初始化所有插件。

        Args:
            config: 全局配置

        Returns:
            {plugin_name: success} 字典
        """
        results = {}
        base_config = config or {}

        for name, adapter in self._adapters.items():
            try:
                success = adapter.initialize(base_config)
                results[name] = success
                if not success:
                    logger.warning(f"Plugin {name} failed to initialize")
            except Exception as e:
                logger.error(f"Plugin {name} initialization error: {e}")
                results[name] = False

        return results

    def get_core_plugins(self) -> Dict[str, ModulePluginAdapter]:
        """返回核心引擎插件集合（EMBEDDING 类型）"""
        return {
            name: adapter
            for name, adapter in self._adapters.items()
            if adapter.plugin_type == PluginType.EMBEDDING
        }

    def get_plugins_by_type(self, plugin_type: PluginType) -> Dict[str, ModulePluginAdapter]:
        """按类型获取插件"""
        return {
            name: adapter
            for name, adapter in self._adapters.items()
            if adapter.plugin_type == plugin_type
        }

    def hot_reload(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        热重载单个插件 — 不重启系统替换插件。

        Args:
            plugin_name: 插件名称
            config: 新配置

        Returns:
            重载成功返回 True
        """
        if plugin_name not in self._adapters:
            logger.error(f"Plugin {plugin_name} not found")
            return False

        adapter = self._adapters[plugin_name]

        # 1. 卸载旧版本
        try:
            adapter.cleanup()
        except Exception as e:
            logger.warning(f"Plugin {plugin_name} cleanup error (ignored): {e}")

        # 2. 从注册表移除
        try:
            self._registry.unregister(plugin_name, force=True)
        except Exception:
            pass

        # 3. 重新注册
        entry = None
        for e in PLUGIN_MANIFEST:
            if e["name"] == plugin_name:
                entry = e
                break

        if not entry:
            logger.error(f"Plugin {plugin_name} not found in manifest")
            return False

        new_adapter = ModulePluginAdapter(
            name=entry["name"],
            module_path=entry["module"],
            plugin_type=entry["type"],
            description=entry["desc"],
        )

        try:
            self._registry.register(new_adapter, config or {}, auto_initialize=False)
            new_adapter.initialize(config or {})
            self._adapters[plugin_name] = new_adapter
            logger.info(f"Plugin {plugin_name} hot-reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Plugin {plugin_name} hot-reload failed: {e}")
            return False

    def health_report(self) -> Dict[str, Any]:
        """
        健康报告 — 所有插件状态汇总。

        Returns:
            {total, by_type, by_state, details} 字典
        """
        stats = self._registry.get_statistics()

        details = {}
        for name in self._registry.list_plugins():
            info = self._registry.get_plugin_info(name)
            if info:
                details[name] = {
                    "type": info["plugin_type"],
                    "state": info["state"],
                    "version": info["version"],
                }

        return {
            "total_plugins": stats["total_plugins"],
            "by_type": stats.get("by_type", {}),
            "by_state": stats.get("by_state", {}),
            "details": details,
            "initialized": self._initialized,
        }

    def get_module(self, plugin_name: str) -> Any:
        """
        获取已加载插件的底层模块引用。

        Args:
            plugin_name: 插件名称

        Returns:
            模块引用

        Raises:
            RuntimeError: 如果插件未加载
        """
        if plugin_name not in self._adapters:
            raise RuntimeError(f"Plugin {plugin_name} not registered")

        adapter = self._adapters[plugin_name]
        return adapter.get_module()

    def shutdown(self):
        """关闭所有插件"""
        for name in list(self._adapters.keys()):
            try:
                self._registry.unregister(name, force=True)
            except Exception:
                pass
        self._adapters.clear()
        self._initialized = False

    def __repr__(self) -> str:
        return f"PluginManager(plugins={len(self._adapters)}, initialized={self._initialized})"


# ============================================================
# 全局单例
# ============================================================

_global_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """获取全局 PluginManager 单例"""
    global _global_plugin_manager
    if _global_plugin_manager is None:
        _global_plugin_manager = PluginManager()
    return _global_plugin_manager


def reset_plugin_manager():
    """重置全局 PluginManager（主要用于测试）"""
    global _global_plugin_manager
    if _global_plugin_manager:
        _global_plugin_manager.shutdown()
    _global_plugin_manager = None
    PluginRegistry.reset_instance()
