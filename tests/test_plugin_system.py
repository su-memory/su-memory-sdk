"""
插件系统测试

测试范围:
- PluginInterface
- PluginRegistry
- SandboxedExecutor
- 官方插件示例

v1.7.0 测试套件
"""

import pytest
import sys
import time
sys.path.insert(0, "src")

from su_memory._sys._plugin_interface import (
    PluginInterface, PluginMetadata, PluginState, PluginType,
    PluginEvent, PluginEventHandler, create_plugin_metadata, validate_plugin
)
from su_memory._sys._plugin_registry import (
    PluginRegistry, PluginAlreadyExistsError, PluginNotFoundError,
    PluginDependencyError, PluginStateError
)
from su_memory._sys._plugin_sandbox import (
    SandboxedExecutor, ExecutionResult, ExecutionStatus,
    ResourceLimit, ExecutionContext
)
from su_memory.plugins.embedding_plugin import TextEmbeddingPlugin, HashVectorizer
from su_memory.plugins.rerank_plugin import RerankPlugin, RerankScorer
from su_memory.plugins.monitor_plugin import MonitorPlugin, MonitorContext, PerformanceMetrics


# ============================================================================
# Test PluginInterface
# ============================================================================

class TestPluginInterface:
    """插件接口测试"""
    
    def test_plugin_metadata_creation(self):
        """测试元数据创建"""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            author="test",
            description="Test plugin"
        )
        assert metadata.name == "test_plugin"
        assert metadata.version == "1.0.0"
        assert metadata.author == "test"
        assert metadata.validate() is True
    
    def test_plugin_metadata_validation(self):
        """测试元数据验证"""
        # 无效：版本格式错误
        metadata = PluginMetadata(name="test", version="1.0")
        assert metadata.validate() is False
        
        # 无效：名称为空
        metadata = PluginMetadata(name="", version="1.0.0")
        assert metadata.validate() is False
    
    def test_plugin_metadata_to_dict(self):
        """测试元数据转字典"""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            author="test",
            description="Test",
            plugin_type=PluginType.EMBEDDING
        )
        data = metadata.to_dict()
        assert data["name"] == "test_plugin"
        assert data["plugin_type"] == "embedding"
    
    def test_plugin_metadata_from_dict(self):
        """测试从字典创建元数据"""
        data = {
            "name": "dict_plugin",
            "version": "2.0.0",
            "author": "Dict Author",
            "plugin_type": "embedding"
        }
        metadata = PluginMetadata.from_dict(data)
        assert metadata.name == "dict_plugin"
        assert metadata.plugin_type == PluginType.EMBEDDING
    
    def test_plugin_state_enum(self):
        """测试插件状态枚举"""
        assert PluginState.UNLOADED.value == "unloaded"
        assert PluginState.READY.value == "ready"
        assert PluginState.RUNNING.value == "running"
        assert PluginState.ERROR.value == "error"
    
    def test_plugin_type_enum(self):
        """测试插件类型枚举"""
        assert PluginType.EMBEDDING.value == "embedding"
        assert PluginType.RERANK.value == "rerank"
        assert PluginType.MONITOR.value == "monitor"


class TestPluginEventHandler:
    """插件事件处理器测试"""
    
    def test_register_and_emit(self):
        """测试注册和触发事件"""
        handler = PluginEventHandler()
        event_received = {"flag": False}
        
        def test_handler(event: PluginEvent):
            event_received["flag"] = True
        
        handler.register("test_event", test_handler)
        handler.emit(PluginEvent(event_type="test_event", plugin_name="test"))
        assert event_received["flag"] is True
    
    def test_unregister(self):
        """测试注销事件处理器"""
        handler = PluginEventHandler()
        called = {"count": 0}
        
        def handler_func(event):
            called["count"] += 1
        
        handler.register("event", handler_func)
        handler.unregister("event", handler_func)
        handler.emit(PluginEvent(event_type="event", plugin_name="test"))
        assert called["count"] == 0
    
    def test_clear_handlers(self):
        """测试清除所有处理器"""
        handler = PluginEventHandler()
        handler.register("event1", lambda e: None)
        handler.register("event2", lambda e: None)
        handler.clear()
        # 清除后不再抛出异常
        handler.emit(PluginEvent(event_type="event1", plugin_name="test"))


# ============================================================================
# Test PluginRegistry
# ============================================================================

class TestPluginRegistry:
    """插件注册表测试"""
    
    def setup_method(self):
        """每个测试前重置单例"""
        PluginRegistry.reset_instance()
        self.registry = PluginRegistry()
    
    def test_singleton_pattern(self):
        """测试单例模式"""
        registry1 = PluginRegistry()
        registry2 = PluginRegistry()
        assert registry1 is registry2
    
    def test_register_plugin(self):
        """测试注册插件"""
        plugin = TextEmbeddingPlugin()
        result = self.registry.register(plugin, {}, auto_initialize=False)
        assert result is True
        assert self.registry.is_registered("text_embedding_plugin")
    
    def test_register_duplicate_raises_error(self):
        """测试重复注册抛出异常"""
        plugin = TextEmbeddingPlugin()
        self.registry.register(plugin, {}, auto_initialize=False)
        with pytest.raises(PluginAlreadyExistsError):
            self.registry.register(plugin, {}, auto_initialize=False)
    
    def test_unregister_plugin(self):
        """测试注销插件"""
        plugin = TextEmbeddingPlugin()
        self.registry.register(plugin, {}, auto_initialize=False)
        result = self.registry.unregister("text_embedding_plugin", force=True)
        assert result is True
        assert not self.registry.is_registered("text_embedding_plugin")
    
    def test_unregister_nonexistent_raises_error(self):
        """测试注销不存在的插件抛出异常"""
        with pytest.raises(PluginNotFoundError):
            self.registry.unregister("nonexistent_plugin", force=True)
    
    def test_get_plugin(self):
        """测试获取插件"""
        plugin = TextEmbeddingPlugin()
        self.registry.register(plugin, {}, auto_initialize=False)
        retrieved = self.registry.get_plugin("text_embedding_plugin")
        assert retrieved is not None
        assert retrieved.name == "text_embedding_plugin"
    
    def test_list_plugins(self):
        """测试列出插件"""
        self.registry.register(TextEmbeddingPlugin(), {}, auto_initialize=False)
        self.registry.register(RerankPlugin(), {}, auto_initialize=False)
        plugins = self.registry.list_plugins()
        assert len(plugins) >= 2
        assert "text_embedding_plugin" in plugins
        assert "rerank_plugin" in plugins
    
    def test_list_plugins_by_type(self):
        """测试按类型列出插件"""
        self.registry.register(TextEmbeddingPlugin(), {}, auto_initialize=False)
        self.registry.register(RerankPlugin(), {}, auto_initialize=False)
        embedding_plugins = self.registry.list_plugins(plugin_type=PluginType.EMBEDDING)
        assert "text_embedding_plugin" in embedding_plugins
    
    def test_get_plugin_metadata(self):
        """测试获取插件元数据"""
        plugin = TextEmbeddingPlugin()
        self.registry.register(plugin, {}, auto_initialize=False)
        metadata = self.registry.get_plugin_metadata("text_embedding_plugin")
        assert metadata is not None
        assert metadata.name == "text_embedding_plugin"
    
    def test_get_plugin_state(self):
        """测试获取插件状态"""
        plugin = TextEmbeddingPlugin()
        self.registry.register(plugin, {}, auto_initialize=False)
        state = self.registry.get_plugin_state("text_embedding_plugin")
        assert state == PluginState.READY
    
    def test_plugin_statistics(self):
        """测试获取统计信息"""
        self.registry.register(TextEmbeddingPlugin(), {}, auto_initialize=False)
        self.registry.register(RerankPlugin(), {}, auto_initialize=False)
        stats = self.registry.get_statistics()
        assert stats["total_plugins"] >= 2
        assert "by_state" in stats
        assert "by_type" in stats
    
    def test_dependency_check(self):
        """测试依赖检查"""
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
        
        with pytest.raises(PluginDependencyError):
            registry.register(DependentPlugin(), {}, auto_initialize=False)
    
    def test_auto_initialize(self):
        """测试自动初始化"""
        plugin = TextEmbeddingPlugin()
        plugin.initialize({"dimension": 64})
        self.registry.register(plugin, {}, auto_initialize=True)
        # 自动初始化成功
        assert self.registry.get_plugin_state("text_embedding_plugin") == PluginState.READY
    
    def test_clear_all_plugins(self):
        """测试清空所有插件"""
        self.registry.register(TextEmbeddingPlugin(), {}, auto_initialize=False)
        self.registry.register(RerankPlugin(), {}, auto_initialize=False)
        count = self.registry.clear(force=True)
        assert count == 2
        assert self.registry.plugin_count == 0


# ============================================================================
# Test SandboxedExecutor
# ============================================================================

class TestSandboxedExecutor:
    """沙箱执行器测试"""
    
    def setup_method(self):
        """每个测试前创建新的执行器"""
        self.executor = SandboxedExecutor(default_timeout=5.0)
    
    def test_execute_success(self):
        """测试成功执行"""
        class DummyPlugin:
            name = "test_plugin"
            
            def execute(self, context):
                return {"result": "success"}
        
        result = self.executor.execute(DummyPlugin(), {"input": "test"})
        assert result.success is True
        assert result.result["result"] == "success"
        assert result.execution_time > 0
    
    def test_execute_timeout(self):
        """测试超时控制"""
        class SlowPlugin:
            name = "slow_plugin"
            
            def execute(self, context):
                time.sleep(10)
                return "done"
        
        result = self.executor.execute(SlowPlugin(), {}, timeout=0.5)
        assert result.success is False
        assert result.is_timeout() is True
        assert result.error == "TIMEOUT"
    
    def test_execute_exception_isolation(self):
        """测试异常隔离"""
        class ErrorPlugin:
            name = "error_plugin"
            
            def execute(self, context):
                raise ValueError("Test error")
        
        result = self.executor.execute(ErrorPlugin(), {})
        assert result.success is False
        assert result.error is not None
        assert result.error_traceback is not None
    
    def test_execute_with_retry(self):
        """测试重试机制"""
        attempt_count = {"count": 0}
        
        class FlakyPlugin:
            name = "flaky_plugin"
            
            def execute(self, context):
                attempt_count["count"] += 1
                if attempt_count["count"] < 2:
                    raise RuntimeError("Temporary failure")
                return "finally succeeded"
        
        result = self.executor.execute_with_retry(FlakyPlugin(), {}, max_retries=3, timeout=1.0)
        assert result.success is True
        assert attempt_count["count"] == 2
    
    def test_resource_limit(self):
        """测试资源限制"""
        self.executor.set_resource_limit(cpu_percent=50, memory_mb=256, timeout_seconds=10)
        limit = self.executor.get_resource_limit()
        assert limit.cpu_percent == 50
        assert limit.memory_mb == 256
        assert limit.timeout_seconds == 10
    
    def test_execution_statistics(self):
        """测试执行统计"""
        class DummyPlugin:
            name = "stats_plugin"
            
            def execute(self, context):
                return "done"
        
        self.executor.execute(DummyPlugin(), {})
        stats = self.executor.get_statistics()
        assert "total_executions" in stats
        assert "success_count" in stats
        assert "success_rate" in stats
    
    def test_execution_history(self):
        """测试执行历史"""
        class DummyPlugin:
            name = "history_plugin"
            
            def execute(self, context):
                return "done"
        
        self.executor.execute(DummyPlugin(), {})
        history = self.executor.get_execution_history(limit=10)
        assert len(history) >= 1
    
    def test_execution_result_to_dict(self):
        """测试执行结果转字典"""
        class DummyPlugin:
            name = "dict_plugin"
            
            def execute(self, context):
                return "done"
        
        result = self.executor.execute(DummyPlugin(), {})
        result_dict = result.to_dict()
        assert "success" in result_dict
        assert "execution_time" in result_dict
        assert "plugin_name" in result_dict


# ============================================================================
# Test Official Plugins
# ============================================================================

class TestTextEmbeddingPlugin:
    """文本嵌入插件测试"""
    
    def test_initialization(self):
        """测试初始化"""
        plugin = TextEmbeddingPlugin()
        result = plugin.initialize({"dimension": 128})
        assert result is True
    
    def test_embed_single_text(self):
        """测试单个文本嵌入"""
        plugin = TextEmbeddingPlugin()
        plugin.initialize({"dimension": 64})
        result = plugin.execute({
            "operation": "embed",
            "text": "Hello world"
        })
        assert result["success"] is True
        assert len(result["vector"]) == 64
    
    def test_batch_embed(self):
        """测试批量嵌入"""
        plugin = TextEmbeddingPlugin()
        plugin.initialize({"dimension": 64})
        result = plugin.execute({
            "operation": "batch_embed",
            "texts": ["Hello", "World", "Test"]
        })
        assert result["success"] is True
        assert result["count"] == 3
        assert len(result["vectors"][0]) == 64
    
    def test_same_text_same_embedding(self):
        """测试相同文本产生相同嵌入"""
        plugin = TextEmbeddingPlugin()
        plugin.initialize({"dimension": 64})
        result1 = plugin.execute({"operation": "embed", "text": "hello world"})
        result2 = plugin.execute({"operation": "embed", "text": "hello world"})
        assert result1["vector"] == result2["vector"]
    
    def test_different_text_different_embedding(self):
        """测试不同文本产生不同嵌入"""
        plugin = TextEmbeddingPlugin()
        plugin.initialize({"dimension": 64})
        result1 = plugin.execute({"operation": "embed", "text": "hello"})
        result2 = plugin.execute({"operation": "embed", "text": "world"})
        assert result1["vector"] != result2["vector"]
    
    def test_cleanup(self):
        """测试清理"""
        plugin = TextEmbeddingPlugin()
        plugin.initialize({})
        plugin.cleanup()
        # 清理后应无异常


class TestHashVectorizer:
    """Hash向量化器测试"""
    
    def test_transform(self):
        """测试向量化转换"""
        vectorizer = HashVectorizer(dimension=64)
        vector = vectorizer.transform("Hello world")
        assert len(vector) == 64
    
    def test_batch_transform(self):
        """测试批量向量化"""
        vectorizer = HashVectorizer(dimension=64)
        vectors = vectorizer.transform_batch(["Hello", "World"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 64


class TestRerankPlugin:
    """重排序插件测试"""
    
    def test_initialization(self):
        """测试初始化"""
        plugin = RerankPlugin()
        result = plugin.initialize({})
        assert result is True
    
    def test_rerank_operation(self):
        """测试重排序操作"""
        plugin = RerankPlugin()
        plugin.initialize({})
        items = [
            {"id": "1", "text": "人工智能技术", "score": 0.8},
            {"id": "2", "text": "机器学习算法", "score": 0.9},
        ]
        result = plugin.execute({
            "query": "人工智能",
            "items": items
        })
        assert result["success"] is True
        assert result["count"] == 2
        assert "rank" in result["ranked_items"][0]
    
    def test_top_k_parameter(self):
        """测试top_k参数"""
        plugin = RerankPlugin()
        plugin.initialize({})
        items = [
            {"id": str(i), "text": f"Item {i}", "score": 0.9 - i * 0.1}
            for i in range(10)
        ]
        result = plugin.execute({
            "query": "test",
            "items": items,
            "top_k": 3
        })
        assert result["count"] == 3
    
    def test_rerank_scorer(self):
        """测试评分器"""
        scorer = RerankScorer()
        items = [
            {"id": "1", "text": "人工智能", "score": 0.9},
            {"id": "2", "text": "机器学习", "score": 0.8},
        ]
        scores = scorer.score_items("人工智能", items)
        ranked = scorer.rerank(scores)
        assert len(ranked) == 2
        assert ranked[0].rank == 1
    
    def test_context_passing(self):
        """测试上下文传递"""
        plugin = RerankPlugin()
        plugin.initialize({})
        result = plugin.execute({
            "query": "test",
            "items": [{"id": "1", "text": "test", "score": 0.9}],
            "context": {"user": "test_user"}
        })
        assert result["context_used"] is True


class TestMonitorPlugin:
    """监控插件测试"""
    
    def test_initialization(self):
        """测试初始化"""
        plugin = MonitorPlugin()
        result = plugin.initialize({})
        assert result is True
    
    def test_wrap_execution(self):
        """测试包装执行"""
        plugin = MonitorPlugin()
        plugin.initialize({})
        
        def dummy_func(x, y=1):
            return x * y
        
        result = plugin.execute({
            "operation": "wrap",
            "func": dummy_func,
            "args": (5,),
            "kwargs": {"y": 2}
        })
        assert result["success"] is True
        assert result["result"] == 10
        assert result["execution_time"] > 0
    
    def test_manual_record(self):
        """测试手动记录"""
        plugin = MonitorPlugin()
        plugin.initialize({})
        result = plugin.execute({
            "operation": "record",
            "execution_time": 0.5,
            "memory_delta": 1024,
            "success": True
        })
        assert result["success"] is True
    
    def test_get_metrics(self):
        """测试获取指标"""
        plugin = MonitorPlugin()
        plugin.initialize({})
        plugin.execute({
            "operation": "record",
            "execution_time": 0.1,
            "success": True
        })
        result = plugin.execute({"operation": "metrics"})
        metrics = result["metrics"]
        assert metrics["execution_count"] >= 1
        assert "success_rate" in metrics
    
    def test_generate_report(self):
        """测试生成报告"""
        plugin = MonitorPlugin()
        plugin.initialize({})
        result = plugin.execute({"operation": "report"})
        assert result["success"] is True
        assert "uptime_seconds" in result
        assert "metrics" in result
    
    def test_reset_statistics(self):
        """测试重置统计"""
        plugin = MonitorPlugin()
        plugin.initialize({})
        plugin.execute({
            "operation": "record",
            "execution_time": 0.1,
            "success": True
        })
        plugin.execute({"operation": "reset"})
        result = plugin.execute({"operation": "metrics"})
        assert result["metrics"]["execution_count"] == 0
    
    def test_error_handling_in_wrap(self):
        """测试错误处理"""
        plugin = MonitorPlugin()
        plugin.initialize({})
        
        def failing_func():
            raise RuntimeError("Test error")
        
        result = plugin.execute({
            "operation": "wrap",
            "func": failing_func,
            "args": (),
            "kwargs": {}
        })
        assert result["success"] is False
        assert result["error"] is not None
    
    def test_cleanup(self):
        """测试清理"""
        plugin = MonitorPlugin()
        plugin.initialize({})
        plugin.cleanup()
        # 清理后应无异常


class TestMonitorContext:
    """监控上下文管理器测试"""
    
    def test_context_manager(self):
        """测试上下文管理器"""
        plugin = MonitorPlugin()
        plugin.initialize({})
        
        def dummy_func():
            return 42
        
        with MonitorContext(plugin) as ctx:
            result = dummy_func()
        
        assert ctx.execution_time > 0
        assert ctx.success is True
    
    def test_context_manager_error(self):
        """测试上下文管理器错误处理"""
        plugin = MonitorPlugin()
        plugin.initialize({})
        
        with MonitorContext(plugin) as ctx:
            raise ValueError("Test error")
        
        assert ctx.success is False
        assert ctx.error is not None


# ============================================================================
# Helper classes for type hints
# ============================================================================

from typing import Dict, List, Any
