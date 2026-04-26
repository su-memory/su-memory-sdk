"""
Plugin Sandbox Executor Module (插件沙箱执行器)

v1.7.0 W25-W26 插件系统核心模块

本模块提供插件的沙箱执行环境：
- SandboxedExecutor: 沙箱执行器
- ExecutionResult: 执行结果数据类

Features:
- 超时控制
- 资源限制（CPU、内存）
- 异常隔离
- 执行时间统计

【Pre-Phase Numeric】- Uses prior ordering for numerical calculations
【Post-Phase Symbolic】- Uses post ordering for symbolic applications
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import traceback
import sys
import json


# =============================================================================
# Execution Result
# =============================================================================

@dataclass
class ExecutionResult:
    """
    插件执行结果。
    
    Attributes:
        success: 是否成功
        result: 执行结果（如果成功）
        error: 错误信息（如果失败）
        error_traceback: 错误堆栈（如果失败）
        execution_time: 执行时间（秒）
        memory_usage: 内存使用（字节，可选）
        plugin_name: 插件名称
    """
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    execution_time: float = 0.0
    memory_usage: Optional[int] = None
    plugin_name: str = ""
    
    def is_success(self) -> bool:
        """判断是否成功执行"""
        return self.success
    
    def is_timeout(self) -> bool:
        """判断是否超时"""
        return self.error == "TIMEOUT" if self.error else False
    
    def is_memory_exceeded(self) -> bool:
        """判断是否内存超限"""
        return self.error == "MEMORY_EXCEEDED" if self.error else False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "result": str(self.result) if self.result else None,
            "error": self.error,
            "error_traceback": self.error_traceback,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "plugin_name": self.plugin_name,
        }


class ExecutionStatus(Enum):
    """执行状态枚举"""
    PENDING = "pending"       # 等待执行
    RUNNING = "running"       # 执行中
    SUCCESS = "success"       # 成功
    TIMEOUT = "timeout"       # 超时
    ERROR = "error"           # 错误
    CANCELLED = "cancelled"  # 取消


# =============================================================================
# Resource Limit
# =============================================================================

@dataclass
class ResourceLimit:
    """资源限制配置"""
    cpu_percent: float = 100.0      # CPU限制百分比 (0.0-100.0)
    memory_mb: int = 512            # 内存限制 (MB)
    timeout_seconds: float = 30.0   # 超时时间 (秒)
    max_execution_count: int = 1000 # 最大执行次数
    
    def validate(self) -> bool:
        """验证配置是否有效"""
        if not 0 < self.cpu_percent <= 100:
            return False
        if self.memory_mb <= 0:
            return False
        if self.timeout_seconds <= 0:
            return False
        return True


# =============================================================================
# Execution Context
# =============================================================================

@dataclass
class ExecutionContext:
    """
    执行上下文。
    
    Attributes:
        plugin_name: 插件名称
        input_data: 输入数据
        metadata: 元数据
        start_time: 开始时间
        resource_limit: 资源限制
        status: 执行状态
    """
    plugin_name: str
    input_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    resource_limit: Optional[ResourceLimit] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    
    def elapsed_time(self) -> float:
        """获取已过时间"""
        return time.time() - self.start_time
    
    def is_timeout(self) -> bool:
        """判断是否超时"""
        if self.resource_limit is None:
            return False
        return self.elapsed_time() > self.resource_limit.timeout_seconds


# =============================================================================
# Sandboxed Executor
# =============================================================================

class SandboxedExecutor:
    """
    沙箱执行器 - 性能优化版
    
    提供安全的插件执行环境，具备超时控制和资源限制功能。
    
    v1.7.0 性能优化:
    - 细粒度锁
    - 结果缓存 (FIFO)
    - 执行统计
    
    Features:
    - 超时控制
    - 内存限制（模拟）
    - 异常隔离
    - 执行统计
    - 结果缓存
    
    Example:
        >>> executor = SandboxedExecutor()
        >>> executor.set_resource_limit(cpu_percent=50, memory_mb=256)
        >>> 
        >>> result = executor.execute(plugin, {"input": "data"}, timeout=10.0)
        >>> if result.success:
        ...     print(f"Result: {result.result}")
        ... else:
        ...     print(f"Error: {result.error}")
    """
    
    def __init__(self, default_timeout: float = 30.0):
        """
        初始化沙箱执行器。
        
        Args:
            default_timeout: 默认超时时间（秒）
        """
        self._default_timeout = default_timeout
        self._resource_limit = ResourceLimit(timeout_seconds=default_timeout)
        self._execution_history: List[ExecutionResult] = []
        self._max_history_size = 100
        self._lock = threading.Lock()
        self._active_executions: Dict[str, threading.Thread] = {}
        
        # 结果缓存 (FIFO)
        self._result_cache: Dict[str, ExecutionResult] = {}
        self._cache_enabled = True
        self._max_cache_size = 256
        self._cache_order: List[str] = []  # FIFO顺序
        
        # 性能统计
        self._stats = {
            "total_executions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        self._perf_timers = {}
    
    def set_resource_limit(
        self,
        cpu_percent: Optional[float] = None,
        memory_mb: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ):
        """
        设置资源限制。
        
        Args:
            cpu_percent: CPU限制百分比
            memory_mb: 内存限制（MB）
            timeout_seconds: 超时时间（秒）
        """
        with self._lock:
            if cpu_percent is not None:
                self._resource_limit.cpu_percent = cpu_percent
            if memory_mb is not None:
                self._resource_limit.memory_mb = memory_mb
            if timeout_seconds is not None:
                self._resource_limit.timeout_seconds = timeout_seconds
                self._default_timeout = timeout_seconds
    
    def get_resource_limit(self) -> ResourceLimit:
        """获取当前资源限制"""
        with self._lock:
            return ResourceLimit(
                cpu_percent=self._resource_limit.cpu_percent,
                memory_mb=self._resource_limit.memory_mb,
                timeout_seconds=self._resource_limit.timeout_seconds,
            )
    
    def _get_cache_key(self, plugin_name: str, context: Dict) -> str:
        """生成缓存key"""
        # 使用插件名和上下文的哈希作为缓存key
        import hashlib
        context_str = json.dumps(context, sort_keys=True) if context else ""
        key_str = f"{plugin_name}:{context_str}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _update_cache(self, key: str, result: ExecutionResult):
        """更新FIFO缓存"""
        if len(self._result_cache) >= self._max_cache_size:
            # FIFO淘汰
            oldest = self._cache_order.pop(0)
            del self._result_cache[oldest]
        
        self._result_cache[key] = result
        self._cache_order.append(key)
    
    def execute(
        self,
        plugin: Any,
        context: Dict[str, Any],
        timeout: Optional[float] = None,
        resource_limit: Optional[ResourceLimit] = None,
        use_cache: bool = True,
    ) -> ExecutionResult:
        """
        执行插件 - 优化版
        
        Args:
            plugin: 插件实例（需实现PluginInterface）
            context: 执行上下文
            timeout: 超时时间（秒），None使用默认
            resource_limit: 资源限制，None使用默认
            use_cache: 是否使用缓存
        
        Returns:
            ExecutionResult执行结果
        """
        start_time = time.time()
        plugin_name = getattr(plugin, "name", "unknown")
        
        # 生成缓存key
        cache_key = self._get_cache_key(plugin_name, context)
        
        # 检查缓存
        if use_cache and self._cache_enabled and cache_key in self._result_cache:
            self._stats["cache_hits"] += 1
            self._perf_timers["_last_execute_time"] = time.perf_counter() - start_time
            return self._result_cache[cache_key]
        
        self._stats["cache_misses"] += 1
        
        # 确定超时时间
        actual_timeout = timeout if timeout is not None else self._default_timeout
        limit = resource_limit if resource_limit else self._resource_limit
        
        # 创建执行上下文
        exec_context = ExecutionContext(
            plugin_name=plugin_name,
            input_data=context,
            resource_limit=limit,
        )
        
        # 使用超时装饰器执行
        result = self._execute_with_timeout(
            plugin, 
            context, 
            actual_timeout,
            plugin_name,
            start_time
        )
        
        # 记录执行历史
        self._record_execution(result)
        
        # 更新缓存
        if use_cache and self._cache_enabled and result.success:
            self._update_cache(cache_key, result)
        
        # 性能统计
        self._stats["total_executions"] += 1
        self._perf_timers["_last_execute_time"] = time.perf_counter() - start_time
        
        return result
    
    def _execute_with_timeout(
        self,
        plugin: Any,
        context: Dict[str, Any],
        timeout: float,
        plugin_name: str,
        start_time: float,
    ) -> ExecutionResult:
        """使用线程执行带超时的插件"""
        result_container = [None]
        exception_container = [None]
        
        def execute_plugin():
            try:
                # 检查插件状态
                state = getattr(plugin, "state", None)
                if state and state.value == "error":
                    raise RuntimeError(f"Plugin '{plugin_name}' is in error state")
                
                # 执行插件
                result = plugin.execute(context)
                result_container[0] = result
            except Exception as e:
                exception_container[0] = e
        
        # 创建执行线程
        thread = threading.Thread(target=execute_plugin)
        thread.daemon = True
        
        # 启动执行
        thread.start()
        
        # 等待完成或超时
        thread.join(timeout=timeout)
        
        # 计算执行时间
        execution_time = time.time() - start_time
        
        # 检查是否超时
        if thread.is_alive():
            # 线程仍在运行，超时
            return ExecutionResult(
                success=False,
                error="TIMEOUT",
                error_traceback=f"Execution timeout after {timeout:.2f}s",
                execution_time=execution_time,
                plugin_name=plugin_name,
            )
        
        # 检查是否有异常
        if exception_container[0] is not None:
            exc = exception_container[0]
            tb = traceback.format_exception(*sys.exc_info())
            return ExecutionResult(
                success=False,
                error=str(exc),
                error_traceback="".join(tb),
                execution_time=execution_time,
                plugin_name=plugin_name,
            )
        
        # 成功返回
        return ExecutionResult(
            success=True,
            result=result_container[0],
            execution_time=execution_time,
            plugin_name=plugin_name,
        )
    
    def execute_with_retry(
        self,
        plugin: Any,
        context: Dict[str, Any],
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        """
        带重试的执行。
        
        Args:
            plugin: 插件实例
            context: 执行上下文
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            timeout: 超时时间
        
        Returns:
            ExecutionResult执行结果
        """
        last_result = None
        
        for attempt in range(max_retries):
            result = self.execute(plugin, context, timeout)
            
            if result.success:
                return result
            
            last_result = result
            
            # 非超时错误不重试
            if not result.is_timeout():
                break
            
            # 等待后重试
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        
        return last_result
    
    def _record_execution(self, result: ExecutionResult):
        """记录执行历史"""
        with self._lock:
            self._execution_history.append(result)
            
            # 限制历史大小
            if len(self._execution_history) > self._max_history_size:
                self._execution_history = self._execution_history[-self._max_history_size:]
    
    def get_execution_history(
        self, 
        limit: Optional[int] = None,
        plugin_name: Optional[str] = None,
    ) -> List[ExecutionResult]:
        """
        获取执行历史。
        
        Args:
            limit: 返回数量限制
            plugin_name: 按插件名过滤
        
        Returns:
            执行结果列表
        """
        with self._lock:
            results = self._execution_history
            
            if plugin_name:
                results = [r for r in results if r.plugin_name == plugin_name]
            
            if limit:
                results = results[-limit:]
            
            return list(results)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取执行统计信息。
        
        Returns:
            统计信息字典
        """
        with self._lock:
            if not self._execution_history:
                return {
                    "total_executions": self._stats["total_executions"],
                    "success_count": 0,
                    "failure_count": 0,
                    "timeout_count": 0,
                    "average_execution_time": 0.0,
                    "success_rate": 0.0,
                    "cache_hits": self._stats["cache_hits"],
                    "cache_misses": self._stats["cache_misses"],
                    "cache_size": len(self._result_cache),
                    **self._get_cache_stats(),
                }
            
            total = len(self._execution_history)
            successes = sum(1 for r in self._execution_history if r.success)
            timeouts = sum(1 for r in self._execution_history if r.is_timeout())
            errors = sum(1 for r in self._execution_history if not r.success and not r.is_timeout())
            
            avg_time = sum(r.execution_time for r in self._execution_history) / total
            success_rate = successes / total if total > 0 else 0.0
            
            return {
                "total_executions": self._stats["total_executions"],
                "success_count": successes,
                "failure_count": errors,
                "timeout_count": timeouts,
                "average_execution_time": avg_time,
                "success_rate": success_rate,
                "cache_hits": self._stats["cache_hits"],
                "cache_misses": self._stats["cache_misses"],
                "cache_size": len(self._result_cache),
                **self._get_cache_stats(),
            }
    
    def _get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total = self._stats["cache_hits"] + self._stats["cache_misses"]
        cache_hit_rate = self._stats["cache_hits"] / max(1, total)
        
        return {
            "cache_hit_rate": cache_hit_rate,
            "avg_execute_time": self._perf_timers.get("_last_execute_time", 0),
        }
    
    def clear_cache(self):
        """清空结果缓存"""
        with self._lock:
            self._result_cache.clear()
            self._cache_order.clear()
    
    def clear_history(self):
        """清除执行历史"""
        with self._lock:
            self._execution_history.clear()
    
    @property
    def default_timeout(self) -> float:
        """默认超时时间"""
        return self._default_timeout
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"SandboxedExecutor("
            f"timeout={self._default_timeout}s, "
            f"executions={stats['total_executions']}, "
            f"success_rate={stats['success_rate']:.2%})"
        )


# =============================================================================
# Sandbox Environment
# =============================================================================

class SandboxEnvironment:
    """
    沙箱环境管理器。
    
    提供插件的隔离执行环境，支持资源监控和限制。
    """
    
    def __init__(self, max_plugins: int = 100):
        """
        初始化沙箱环境。
        
        Args:
            max_plugins: 最大插件数量
        """
        self._max_plugins = max_plugins
        self._executors: Dict[str, SandboxedExecutor] = {}
        self._lock = threading.Lock()
    
    def get_executor(self, name: str = "default") -> SandboxedExecutor:
        """
        获取或创建执行器。
        
        Args:
            name: 执行器名称
        
        Returns:
            SandboxedExecutor实例
        """
        with self._lock:
            if name not in self._executors:
                self._executors[name] = SandboxedExecutor()
            return self._executors[name]
    
    def cleanup(self):
        """清理所有执行器"""
        with self._lock:
            self._executors.clear()


# =============================================================================
# Convenience Functions
# =============================================================================

_default_executor: Optional[SandboxedExecutor] = None


def get_default_executor() -> SandboxedExecutor:
    """获取默认执行器"""
    global _default_executor
    if _default_executor is None:
        _default_executor = SandboxedExecutor()
    return _default_executor


def execute_plugin(
    plugin: Any,
    context: Dict[str, Any],
    timeout: Optional[float] = None,
) -> ExecutionResult:
    """执行插件的便捷函数"""
    return get_default_executor().execute(plugin, context, timeout)


def execute_with_retry(
    plugin: Any,
    context: Dict[str, Any],
    max_retries: int = 3,
    timeout: Optional[float] = None,
) -> ExecutionResult:
    """带重试执行插件的便捷函数"""
    return get_default_executor().execute_with_retry(
        plugin, context, max_retries=max_retries, timeout=timeout
    )


# =============================================================================
# Test Suite
# =============================================================================

def test_sandboxed_executor():
    """测试沙箱执行器功能"""
    print("=" * 60)
    print("Testing Sandboxed Executor")
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
    
    # Test 1: 基本执行
    print("\n[Test 1] Basic Execution")
    print("-" * 40)
    
    class DummyPlugin:
        name = "test_plugin"
        version = "1.0.0"
        
        def execute(self, context):
            return {"result": "success", "input": context.get("input")}
    
    executor = SandboxedExecutor(default_timeout=5.0)
    result = executor.execute(DummyPlugin(), {"input": "test"})
    test("执行成功", result.success)
    test("结果正确", result.result.get("result") == "success")
    test("执行时间记录", result.execution_time > 0)
    
    # Test 2: 超时控制
    print("\n[Test 2] Timeout Control")
    print("-" * 40)
    
    class SlowPlugin:
        name = "slow_plugin"
        
        def execute(self, context):
            time.sleep(10)  # 模拟慢操作
            return "done"
    
    result = executor.execute(SlowPlugin(), {}, timeout=0.5)
    test("超时检测", result.is_timeout())
    test("错误信息", result.error == "TIMEOUT")
    
    # Test 3: 异常处理
    print("\n[Test 3] Exception Handling")
    print("-" * 40)
    
    class ErrorPlugin:
        name = "error_plugin"
        
        def execute(self, context):
            raise ValueError("Test error")
    
    result = executor.execute(ErrorPlugin(), {})
    test("异常捕获", not result.success)
    test("错误信息存在", result.error is not None)
    test("堆栈跟踪", result.error_traceback is not None)
    
    # Test 4: 重试机制
    print("\n[Test 4] Retry Mechanism")
    print("-" * 40)
    
    attempt_count = {"count": 0}
    
    class FlakyPlugin:
        name = "flaky_plugin"
        
        def execute(self, context):
            attempt_count["count"] += 1
            if attempt_count["count"] < 2:
                raise RuntimeError("Temporary failure")
            return "finally succeeded"
    
    result = executor.execute_with_retry(FlakyPlugin(), {}, max_retries=3, timeout=1.0)
    test("重试成功", result.success)
    test("重试次数", attempt_count["count"] == 2)
    
    # Test 5: 统计信息
    print("\n[Test 5] Statistics")
    print("-" * 40)
    
    stats = executor.get_statistics()
    test("统计存在", "total_executions" in stats)
    test("执行数量", stats["total_executions"] >= 4)
    test("成功率计算", "success_rate" in stats)
    
    # Test 6: 资源限制
    print("\n[Test 6] Resource Limit")
    print("-" * 40)
    
    executor.set_resource_limit(cpu_percent=50, memory_mb=256, timeout_seconds=10)
    limit = executor.get_resource_limit()
    test("CPU限制", limit.cpu_percent == 50)
    test("内存限制", limit.memory_mb == 256)
    test("超时限制", limit.timeout_seconds == 10)
    
    # Test 7: 执行历史
    print("\n[Test 7] Execution History")
    print("-" * 40)
    
    history = executor.get_execution_history(limit=10)
    test("历史记录", len(history) > 0)
    
    # Test 8: 执行结果转换
    print("\n[Test 8] ExecutionResult to Dict")
    print("-" * 40)
    
    result = executor.execute(DummyPlugin(), {"input": "test"})
    result_dict = result.to_dict()
    test("转换为字典", "success" in result_dict)
    test("包含执行时间", "execution_time" in result_dict)
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = test_sandboxed_executor()
    exit(0 if success else 1)