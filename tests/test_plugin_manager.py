"""
PluginManager / ModulePluginAdapter 测试

覆盖: 模块适配器生命周期、清单自动发现、类型过滤、热重载、
健康报告、单例与重置。不依赖外部服务。
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from su_memory._sys._plugin_interface import PluginType
from su_memory._sys._plugin_registry import PluginRegistry
from su_memory.sdk.plugin_manager import (
    PLUGIN_MANIFEST,
    ModulePluginAdapter,
    PluginManager,
    get_plugin_manager,
    reset_plugin_manager,
)

REAL_MODULE = "su_memory._sys._terms"


@pytest.fixture(autouse=True)
def _clean_plugin_state():
    """每个用例前重置注册表与全局单例, 避免用例间串扰"""
    reset_plugin_manager()
    PluginRegistry.reset_instance()
    yield
    reset_plugin_manager()


class TestModulePluginAdapter:
    def test_defaults(self):
        a = ModulePluginAdapter("demo", REAL_MODULE)
        assert a.name == "demo"
        assert a.version == "3.0.0"
        assert a.description == ""
        assert a.plugin_type is PluginType.UTILITY
        assert a.dependencies == []

    def test_properties_passthrough(self):
        a = ModulePluginAdapter(
            "demo", REAL_MODULE,
            plugin_type=PluginType.REASONING,
            description="测试插件", version="9.9.9",
            dependencies=["x"],
        )
        assert a.plugin_type is PluginType.REASONING
        assert a.description == "测试插件"
        assert a.version == "9.9.9"
        assert a.dependencies == ["x"]

    def test_initialize_loads_module(self):
        a = ModulePluginAdapter("demo", REAL_MODULE)
        assert a.initialize({"k": 1}) is True
        assert a.execute({}) is importlib.import_module(REAL_MODULE)
        assert a.get_module() is importlib.import_module(REAL_MODULE)

    def test_execute_before_init_raises(self):
        a = ModulePluginAdapter("demo", REAL_MODULE)
        with pytest.raises(RuntimeError):
            a.execute({})
        with pytest.raises(RuntimeError):
            a.get_module()

    def test_initialize_missing_module_returns_false(self):
        a = ModulePluginAdapter("demo", "su_memory._sys._no_such_module_xyz")
        assert a.initialize({}) is False

    def test_initialize_generic_error_returns_false(self, monkeypatch):
        def boom(_path):
            raise ValueError("boom")
        monkeypatch.setattr(importlib, "import_module", boom)
        a = ModulePluginAdapter("demo", REAL_MODULE)
        assert a.initialize({}) is False

    def test_cleanup_resets_state(self):
        a = ModulePluginAdapter("demo", REAL_MODULE)
        a.initialize({})
        a.cleanup()
        assert a._module is None
        assert a._config == {}


class TestPluginManager:
    def test_auto_discover_registers_manifest(self):
        pm = PluginManager()
        n = pm.auto_discover()
        assert n == len(PLUGIN_MANIFEST) >= 50
        # 幂等: 已初始化后再次发现不重复注册
        assert pm.auto_discover() == len(PLUGIN_MANIFEST)
        report = pm.health_report()
        assert report["total_plugins"] == len(PLUGIN_MANIFEST)
        assert report["initialized"] is True
        assert len(report["details"]) == len(PLUGIN_MANIFEST)

    def test_get_core_plugins_only_embedding(self):
        pm = PluginManager()
        pm.auto_discover()
        core = pm.get_core_plugins()
        assert core
        assert all(p.plugin_type is PluginType.EMBEDDING for p in core.values())

    def test_get_plugins_by_type(self):
        pm = PluginManager()
        pm.auto_discover()
        util = pm.get_plugins_by_type(PluginType.UTILITY)
        assert util
        assert all(p.plugin_type is PluginType.UTILITY for p in util.values())
        assert pm.get_plugins_by_type(PluginType.REASONING)

    def test_hot_reload_unknown_plugin_false(self):
        pm = PluginManager()
        pm.auto_discover()
        assert pm.hot_reload("not_exist_plugin") is False

    def test_hot_reload_success(self):
        pm = PluginManager()
        pm.auto_discover()
        name = PLUGIN_MANIFEST[0]["name"]
        assert pm.hot_reload(name) is True
        # 重载后适配器被替换, 仍可初始化加载
        adapter = pm._adapters[name]
        assert adapter.initialize({}) is True

    def test_get_module_requires_initialization(self):
        pm = PluginManager()
        pm.auto_discover()
        with pytest.raises(RuntimeError):
            pm.get_module("migrator")
        # 手动初始化后即可取到模块引用
        pm._adapters["migrator"].initialize({})
        assert pm.get_module("migrator") is importlib.import_module(
            "su_memory._sys.migrator"
        )

    def test_shutdown_clears(self):
        pm = PluginManager()
        pm.auto_discover()
        pm.shutdown()
        assert pm._adapters == {}
        assert pm._initialized is False
        assert "plugins=0" in repr(pm)

    def test_repr(self):
        pm = PluginManager()
        assert repr(pm) == "PluginManager(plugins=0, initialized=False)"


class TestSingleton:
    def test_get_plugin_manager_singleton(self):
        a = get_plugin_manager()
        b = get_plugin_manager()
        assert a is b

    def test_reset_plugin_manager_creates_new(self):
        a = get_plugin_manager()
        reset_plugin_manager()
        b = get_plugin_manager()
        assert a is not b
