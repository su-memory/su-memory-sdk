"""Registry quick test"""
import sys

sys.path.insert(0, "src")

from su_memory._sys._plugin_registry import PluginRegistry
from su_memory.plugins.embedding_plugin import TextEmbeddingPlugin


class TestRegistryCheck:
    def setup_method(self):
        self.registry = PluginRegistry()
        self.registry.clear()

    def teardown_method(self):
        self.registry.clear()

    def test_register(self):
        print("Registering plugin...")
        plugin = TextEmbeddingPlugin()
        plugin.initialize({})
        self.registry.register(plugin, auto_initialize=False)
        print("Plugin registered!")

        # 插件 name 约定已从类名迁移到 snake_case（如 "text_embedding_plugin"），
        # 动态取实际 name 做断言，避免硬编码类名导致与约定脱节。
        assert self.registry.is_registered(plugin.name)
        print("Test passed!")
