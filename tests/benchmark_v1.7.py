"""
su-memory SDK v1.7.0 性能基准测试

测试插件注册表、SQLite后端和沙箱执行器的性能。

Usage:
    python tests/benchmark_v1.7.py
"""

import time
import sys
import os
import tempfile

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from su_memory._sys._plugin_registry import PluginRegistry, PluginAlreadyExistsError
from su_memory._sys._plugin_interface import PluginInterface, PluginMetadata, PluginType, PluginState
from su_memory.storage.sqlite_backend import SQLiteBackend, MemoryItem
from su_memory._sys._plugin_sandbox import SandboxedExecutor


# =============================================================================
# Test Classes
# =============================================================================

class DummyPlugin(PluginInterface):
    """测试用插件"""
    _counter = 0
    
    def __init__(self, name: str = "DummyPlugin"):
        self._name = name
        self._state = PluginState.READY
        DummyPlugin._counter += 1
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Dummy plugin for testing"
    
    @property
    def dependencies(self) -> list:
        return []
    
    @property
    def state(self) -> PluginState:
        return self._state
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self._name,
            version="1.0.0",
            description="Dummy plugin for testing",
            author="Benchmark",
            plugin_type=PluginType.CUSTOM,
        )
    
    def initialize(self, config: dict) -> bool:
        self._state = PluginState.READY
        return True
    
    def execute(self, context: dict):
        return {"result": "ok", "input": context.get("data", "")}
    
    def cleanup(self):
        self._state = PluginState.UNLOADING


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_plugin_registry(n_plugins: int = 500, n_gets: int = 500):
    """插件注册表基准测试"""
    print(f"\n{'='*60}")
    print(f"Plugin Registry Benchmark (n={n_plugins})")
    print(f"{'='*60}")
    
    # 重置单例
    PluginRegistry.reset_instance()
    registry = PluginRegistry()
    
    # 注册测试
    print(f"\n[Test 1] Register {n_plugins} plugins...")
    start = time.perf_counter()
    for i in range(n_plugins):
        plugin = DummyPlugin(f"Plugin_{i}")
        try:
            registry.register(plugin, auto_initialize=False)
        except PluginAlreadyExistsError:
            pass
    register_time = time.perf_counter() - start
    print(f"  Time: {register_time*1000:.2f}ms ({register_time/n_plugins*1000:.3f}ms/plugin)")
    
    # 获取测试
    print(f"\n[Test 2] Get {n_gets} plugins (random access)...")
    start = time.perf_counter()
    for i in range(n_gets):
        idx = i % n_plugins
        registry.get_plugin(f"Plugin_{idx}")
    get_time = time.perf_counter() - start
    print(f"  Time: {get_time*1000:.2f}ms ({get_time/n_gets*1000:.3f}ms/get)")
    
    # 列表测试
    print(f"\n[Test 3] List plugins (100 iterations)...")
    start = time.perf_counter()
    for _ in range(100):
        registry.list_plugins()
    list_time = time.perf_counter() - start
    print(f"  Time: {list_time*1000:.2f}ms ({list_time/100*1000:.3f}ms/list)")
    
    # 性能统计
    stats = registry.get_performance_stats()
    print(f"\n[Performance Stats]")
    print(f"  Cache size (plugins): {stats['cache_size']}")
    print(f"  Register count: {stats['register_count']}")
    print(f"  Get count: {stats['get_count']}")
    print(f"  List count: {stats['list_count']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache misses: {stats['cache_misses']}")
    print(f"  Cache hit rate: {stats.get('cache_hit_rate', 0)*100:.1f}%")
    print(f"  Avg register time: {stats.get('avg_register_time', 0)*1000:.3f}ms")
    print(f"  Avg get time: {stats.get('avg_get_time', 0)*1000:.3f}ms")
    print(f"  Avg list time: {stats.get('avg_list_time', 0)*1000:.3f}ms")
    
    # 清理
    registry.clear(force=True)
    PluginRegistry.reset_instance()
    
    return {
        "register_time": register_time,
        "get_time": get_time,
        "list_time": list_time,
    }


def benchmark_sqlite_backend(n_items: int = 500, n_queries: int = 50):
    """SQLite后端基准测试"""
    print(f"\n{'='*60}")
    print(f"SQLite Backend Benchmark (n={n_items})")
    print(f"{'='*60}")
    
    # 创建临时数据库
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_file.close()
    
    try:
        print(f"Creating backend...")
        sys.stdout.flush()
        backend = SQLiteBackend(temp_file.name)
        print("Backend created")
        sys.stdout.flush()
        
        # 创建测试数据
        print(f"\n[Test 1] Insert {n_items} memories...")
        memories = [
            MemoryItem(
                id=f"mem_{i}",
                content=f"Content {i} - This is test memory content for benchmarking",
                metadata={"index": i, "type": "test"},
                timestamp=time.time() + i,
            )
            for i in range(n_items)
        ]
        
        # 单条插入测试
        start = time.perf_counter()
        for mem in memories[:50]:  # 只测50条
            backend.add_memory(mem)
        single_insert_time = time.perf_counter() - start
        print(f"  Single insert (50): {single_insert_time*1000:.2f}ms")
        
        # 批量插入测试
        print(f"  Batch inserting {n_items-50} memories...")
        sys.stdout.flush()
        start = time.perf_counter()
        backend.add_memory_batch(memories[50:])
        batch_insert_time = time.perf_counter() - start
        print(f"  Batch insert ({n_items-50}): {batch_insert_time*1000:.2f}ms")
        
        # 查询测试
        print(f"\n[Test 2] Query {n_queries} times...")
        
        # 首次查询
        start = time.perf_counter()
        backend.query("Content", top_k=10)
        first_query_time = time.perf_counter() - start
        print(f"  First query: {first_query_time*1000:.2f}ms")
        
        # 重复查询（有缓存）
        start = time.perf_counter()
        for i in range(n_queries - 1):
            backend.query("Content", top_k=10)
        cached_query_time = time.perf_counter() - start
        print(f"  Next {n_queries-1} queries (cached): {cached_query_time*1000:.2f}ms")
        
        # 统计信息
        stats = backend.get_performance_stats()
        print(f"\n[Performance Stats]")
        print(f"  Total queries: {stats['query_count']}")
        print(f"  Total inserts: {stats['insert_count']}")
        print(f"  Batch count: {stats['batch_count']}")
        print(f"  Cache hits: {stats['cache_hits']}")
        print(f"  Cache misses: {stats['cache_misses']}")
        print(f"  Cache hit rate: {stats.get('cache_hit_rate', 0)*100:.1f}%")
        
        # 关闭
        backend.close()
        
        return {
            "single_insert_time": single_insert_time,
            "batch_insert_time": batch_insert_time,
            "cached_query_time": cached_query_time,
        }
    
    finally:
        # 清理临时文件
        os.unlink(temp_file.name)


def benchmark_sandbox_executor(n_executions: int = 2000):
    """沙箱执行器基准测试"""
    print(f"\n{'='*60}")
    print(f"Sandboxed Executor Benchmark (n={n_executions})")
    print(f"{'='*60}")
    
    executor = SandboxedExecutor(default_timeout=5.0)
    plugin = DummyPlugin("BenchmarkPlugin")
    
    # 首次执行（无缓存）
    print(f"\n[Test 1] First execution (no cache)...")
    start = time.perf_counter()
    result = executor.execute(plugin, {"data": "test"}, use_cache=False)
    first_time = time.perf_counter() - start
    print(f"  Time: {first_time*1000:.2f}ms")
    print(f"  Success: {result.success}")
    
    # 缓存执行测试
    print(f"\n[Test 2] Execute {n_executions} times (with cache)...")
    start = time.perf_counter()
    for i in range(n_executions):
        executor.execute(plugin, {"data": "test"})
    cached_time = time.perf_counter() - start
    print(f"  Time: {cached_time*1000:.2f}ms")
    print(f"  Avg per execution: {cached_time/n_executions*1000:.3f}ms")
    
    # 统计信息
    stats = executor.get_statistics()
    print(f"\n[Performance Stats]")
    print(f"  Total executions: {stats['total_executions']}")
    print(f"  Success count: {stats['success_count']}")
    print(f"  Failure count: {stats['failure_count']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache misses: {stats['cache_misses']}")
    print(f"  Cache hit rate: {stats.get('cache_hit_rate', 0)*100:.1f}%")
    print(f"  Avg execution time: {stats.get('average_execution_time', 0)*1000:.3f}ms")
    
    # 清理
    executor.clear_cache()
    executor.clear_history()
    
    return {
        "first_time": first_time,
        "cached_time": cached_time,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*60)
    print("su-memory SDK v1.7.0 Performance Benchmark")
    print("="*60)
    
    results = {}
    
    # 1. 插件注册表测试
    print("\n[1] Running Plugin Registry Benchmark...")
    results["plugin_registry"] = benchmark_plugin_registry(n_plugins=500, n_gets=500)
    
    # 2. SQLite后端测试
    print("\n[2] Running SQLite Backend Benchmark...")
    sys.stdout.flush()
    results["sqlite_backend"] = benchmark_sqlite_backend(n_items=100, n_queries=20)
    
    # 3. 沙箱执行器测试
    print("\n[3] Running Sandboxed Executor Benchmark...")
    results["sandbox_executor"] = benchmark_sandbox_executor(n_executions=2000)
    
    # 汇总
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"\nPlugin Registry:")
    print(f"  Register 1000 plugins: {results['plugin_registry']['register_time']*1000:.2f}ms")
    print(f"  Get 1000 plugins: {results['plugin_registry']['get_time']*1000:.2f}ms")
    print(f"  List 100 times: {results['plugin_registry']['list_time']*1000:.2f}ms")
    
    print(f"\nSQLite Backend:")
    print(f"  Single insert (100): {results['sqlite_backend']['single_insert_time']*1000:.2f}ms")
    print(f"  Batch insert: {results['sqlite_backend']['batch_insert_time']*1000:.2f}ms")
    print(f"  Cached queries: {results['sqlite_backend']['cached_query_time']*1000:.2f}ms")
    
    print(f"\nSandboxed Executor:")
    print(f"  First execution: {results['sandbox_executor']['first_time']*1000:.2f}ms")
    print(f"  Cached executions: {results['sandbox_executor']['cached_time']*1000:.2f}ms")
    
    print(f"\n{'='*60}")
    print("Benchmark completed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
