"""Registry quick test"""
import pytest
import sys
import os
import tempfile
import time

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
        
        assert self.registry.is_registered("TextEmbeddingPlugin")
        print("Test passed!")
