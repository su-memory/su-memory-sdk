"""
Monitor Plugin (性能监控插件)

v1.7.0 W25-W26 官方插件示例

记录执行时间、内存使用等性能指标，演示生命周期管理。

Features:
- 执行时间统计
- 内存使用监控
- 调用次数统计
- 性能报告生成

【Pre-Phase Numeric】- Uses prior ordering for numerical calculations
【Post-Phase Symbolic】- Uses post ordering for symbolic applications
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import time
import threading
import os

# 可选依赖 psutil
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    _PSUTIL_AVAILABLE = False

from .._sys._plugin_interface import (
    PluginInterface,
    PluginType,
)


# =============================================================================
# Performance Metrics
# =============================================================================

@dataclass
class PerformanceMetrics:
    """性能指标数据"""
    execution_count: int = 0
    total_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    last_execution_time: Optional[float] = None
    memory_start: Optional[int] = None
    memory_end: Optional[int] = None
    memory_peak: Optional[int] = None
    error_count: int = 0
    success_count: int = 0
    
    def record_execution(self, exec_time: float, memory_delta: int, success: bool):
        """记录一次执行"""
        self.execution_count += 1
        self.total_execution_time += exec_time
        self.min_execution_time = min(self.min_execution_time, exec_time)
        self.max_execution_time = max(self.max_execution_time, exec_time)
        self.avg_execution_time = self.total_execution_time / self.execution_count
        self.last_execution_time = exec_time
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        
        if self.memory_peak is None or memory_delta > self.memory_peak:
            self.memory_peak = memory_delta
    
    def get_success_rate(self) -> float:
        """获取成功率"""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "execution_count": self.execution_count,
            "total_execution_time": self.total_execution_time,
            "min_execution_time": self.min_execution_time if self.min_execution_time != float('inf') else 0,
            "max_execution_time": self.max_execution_time,
            "avg_execution_time": self.avg_execution_time,
            "last_execution_time": self.last_execution_time,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": self.get_success_rate(),
            "memory_peak": self.memory_peak,
        }


@dataclass
class ExecutionRecord:
    """执行记录"""
    timestamp: datetime
    execution_time: float
    memory_usage: int
    success: bool
    error_message: Optional[str] = None


# =============================================================================
# Monitor Plugin
# =============================================================================

class MonitorPlugin(PluginInterface):
    """
    性能监控插件。
    
    监控插件执行性能和资源使用情况。
    
    Example:
        >>> plugin = MonitorPlugin()
        >>> plugin.initialize({})
        >>> 
        >>> # 执行监控的操作
        >>> result = plugin.execute({
        ...     "operation": "wrap",
        ...     "func": some_function,
        ...     "args": (),
        ...     "kwargs": {},
        ... })
        >>> 
        >>> # 获取监控报告
        >>> report = plugin.execute({
        ...     "operation": "report"
        ... })
    """
    
    def __init__(self):
        """初始化插件"""
        self._initialized = False
        self._config: Dict[str, Any] = {}
        self._metrics = PerformanceMetrics()
        self._records: List[ExecutionRecord] = []
        self._max_records = 1000
        self._lock = threading.Lock()
        self._process = psutil.Process(os.getpid()) if _PSUTIL_AVAILABLE else None
        self._start_time: Optional[datetime] = None
    
    @property
    def name(self) -> str:
        return "monitor_plugin"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "性能监控插件，记录执行时间、内存使用等指标"
    
    @property
    def author(self) -> str:
        return "su-memory-sdk"
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.MONITOR
    
    @property
    def dependencies(self) -> List[str]:
        return []
    
    @property
    def config_schema(self) -> Dict[str, Any]:
        return {
            "required": [],
            "properties": {
                "max_records": {
                    "type": "integer",
                    "default": 1000,
                    "description": "最大记录数"
                },
                "track_memory": {
                    "type": "boolean",
                    "default": True,
                    "description": "是否跟踪内存使用"
                },
                "report_interval": {
                    "type": "integer",
                    "default": 60,
                    "description": "报告生成间隔（秒）"
                }
            }
        }
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        初始化插件。
        
        Args:
            config: 配置字典
        
        Returns:
            True表示初始化成功
        """
        try:
            self._config = config
            self._max_records = config.get("max_records", 1000)
            self._metrics = PerformanceMetrics()
            self._records = []
            self._start_time = datetime.now()
            self._initialized = True
            return True
            
        except Exception as e:
            self._initialized = False
            return False
    
    def execute(self, context: Dict[str, Any]) -> Any:
        """
        执行监控操作。
        
        Args:
            context: 执行上下文
                - operation: str, 操作类型
                    - "wrap": 包装执行并监控
                    - "record": 记录一次执行
                    - "report": 获取监控报告
                    - "reset": 重置统计数据
                    - "metrics": 获取当前指标
        
        Returns:
            操作结果
        """
        if not self._initialized:
            raise RuntimeError("Plugin not initialized")
        
        operation = context.get("operation", "report")
        
        if operation == "wrap":
            return self._wrap_execution(context)
        elif operation == "record":
            return self._record_manual(context)
        elif operation == "report":
            return self._generate_report()
        elif operation == "reset":
            return self._reset_stats()
        elif operation == "metrics":
            return self._get_metrics()
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _wrap_execution(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """包装执行并监控"""
        func = context.get("func")
        args = context.get("args", ())
        kwargs = context.get("kwargs", {})
        
        # 获取执行前内存
        memory_start = self._get_memory_usage()
        
        # 记录开始时间
        start_time = time.time()
        success = True
        error_msg = None
        result = None
        
        try:
            if callable(func):
                result = func(*args, **kwargs)
            else:
                raise ValueError("'func' is not callable")
        except Exception as e:
            success = False
            error_msg = str(e)
            result = None
        
        # 记录结束时间
        exec_time = time.time() - start_time
        memory_end = self._get_memory_usage()
        memory_delta = max(0, memory_end - memory_start)
        
        # 更新指标
        self._metrics.record_execution(exec_time, memory_delta, success)
        
        # 记录执行
        record = ExecutionRecord(
            timestamp=datetime.now(),
            execution_time=exec_time,
            memory_usage=memory_delta,
            success=success,
            error_message=error_msg,
        )
        self._add_record(record)
        
        return {
            "success": success,
            "result": result,
            "execution_time": exec_time,
            "memory_delta": memory_delta,
            "error": error_msg,
        }
    
    def _record_manual(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """手动记录执行"""
        exec_time = context.get("execution_time", 0.0)
        memory_delta = context.get("memory_delta", 0)
        success = context.get("success", True)
        error_msg = context.get("error")
        
        self._metrics.record_execution(exec_time, memory_delta, success)
        
        record = ExecutionRecord(
            timestamp=datetime.now(),
            execution_time=exec_time,
            memory_usage=memory_delta,
            success=success,
            error_message=error_msg,
        )
        self._add_record(record)
        
        return {"success": True}
    
    def _generate_report(self) -> Dict[str, Any]:
        """生成监控报告"""
        uptime = None
        if self._start_time:
            uptime = (datetime.now() - self._start_time).total_seconds()
        
        return {
            "success": True,
            "uptime_seconds": uptime,
            "metrics": self._metrics.to_dict(),
            "recent_records": self._get_recent_records(10),
            "timestamp": datetime.now().isoformat(),
        }
    
    def _get_metrics(self) -> Dict[str, Any]:
        """获取当前指标"""
        return {
            "success": True,
            "metrics": self._metrics.to_dict(),
        }
    
    def _reset_stats(self) -> Dict[str, Any]:
        """重置统计数据"""
        with self._lock:
            self._metrics = PerformanceMetrics()
            self._records = []
            self._start_time = datetime.now()
        
        return {"success": True, "message": "Statistics reset"}
    
    def _get_memory_usage(self) -> int:
        """获取当前内存使用（字节）"""
        if not _PSUTIL_AVAILABLE or self._process is None:
            return 0
        try:
            mem_info = self._process.memory_info()
            return mem_info.rss
        except Exception:
            return 0
    
    def _add_record(self, record: ExecutionRecord):
        """添加执行记录"""
        with self._lock:
            self._records.append(record)
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records:]
    
    def _get_recent_records(self, count: int) -> List[Dict[str, Any]]:
        """获取最近的记录"""
        with self._lock:
            recent = self._records[-count:]
            return [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "execution_time": r.execution_time,
                    "memory_usage": r.memory_usage,
                    "success": r.success,
                    "error": r.error_message,
                }
                for r in recent
            ]
    
    def cleanup(self) -> None:
        """清理资源"""
        with self._lock:
            self._records.clear()
            self._metrics = PerformanceMetrics()
        self._initialized = False
        self._config = {}


# =============================================================================
# Context Manager for Monitoring
# =============================================================================

class MonitorContext:
    """
    监控上下文管理器。
    
    用于包装代码块的执行监控。
    
    Example:
        >>> plugin = MonitorPlugin()
        >>> plugin.initialize({})
        >>> 
        >>> with MonitorContext(plugin) as ctx:
        ...     # 你的代码
        ...     result = expensive_operation()
        >>> 
        >>> print(f"Execution time: {ctx.execution_time}")
    """
    
    def __init__(self, plugin: MonitorPlugin):
        """初始化监控上下文"""
        self._plugin = plugin
        self._start_time: Optional[float] = None
        self._memory_start: int = 0
        self.execution_time: float = 0.0
        self.memory_delta: int = 0
        self.success: bool = False
        self.error: Optional[str] = None
    
    def __enter__(self):
        """进入上下文"""
        self._start_time = time.time()
        if _PSUTIL_AVAILABLE:
            try:
                process = psutil.Process(os.getpid())
                self._memory_start = process.memory_info().rss
            except Exception:
                self._memory_start = 0
        else:
            self._memory_start = 0
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        self.execution_time = time.time() - self._start_time
        
        if _PSUTIL_AVAILABLE:
            try:
                process = psutil.Process(os.getpid())
                memory_end = process.memory_info().rss
                self.memory_delta = max(0, memory_end - self._memory_start)
            except Exception:
                self.memory_delta = 0
        else:
            self.memory_delta = 0
        
        if exc_type is None:
            self.success = True
        else:
            self.success = False
            self.error = str(exc_val)
        
        # 记录到插件
        self._plugin.execute({
            "operation": "record",
            "execution_time": self.execution_time,
            "memory_delta": self.memory_delta,
            "success": self.success,
            "error": self.error,
        })
        
        return False  # 不抑制异常


# =============================================================================
# Plugin Factory
# =============================================================================

def create_monitor_plugin() -> MonitorPlugin:
    """创建监控插件实例"""
    return MonitorPlugin()


# =============================================================================
# Test Suite
# =============================================================================

def test_monitor_plugin():
    """测试监控插件"""
    print("=" * 60)
    print("Testing Monitor Plugin")
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
    
    def dummy_function(x, y=1):
        """模拟耗时操作"""
        time.sleep(0.01)
        return x * y
    
    # Test 1: 插件创建
    print("\n[Test 1] Plugin Creation")
    print("-" * 40)
    
    plugin = MonitorPlugin()
    test("插件创建", plugin is not None)
    test("插件名称", plugin.name == "monitor_plugin")
    test("插件类型", plugin.plugin_type == PluginType.MONITOR)
    
    # Test 2: 初始化
    print("\n[Test 2] Initialization")
    print("-" * 40)
    
    config = {"max_records": 100, "track_memory": True}
    success = plugin.initialize(config)
    test("初始化成功", success)
    
    # Test 3: 包装执行
    print("\n[Test 3] Wrap Execution")
    print("-" * 40)
    
    result = plugin.execute({
        "operation": "wrap",
        "func": dummy_function,
        "args": (5,),
        "kwargs": {"y": 2},
    })
    
    test("执行成功", result.get("success"))
    test("结果正确", result.get("result") == 10)
    test("有执行时间", result.get("execution_time", 0) > 0)
    
    # Test 4: 手动记录
    print("\n[Test 4] Manual Record")
    print("-" * 40)
    
    result = plugin.execute({
        "operation": "record",
        "execution_time": 0.5,
        "memory_delta": 1024,
        "success": True,
    })
    
    test("记录成功", result.get("success"))
    
    # Test 5: 获取指标
    print("\n[Test 5] Get Metrics")
    print("-" * 40)
    
    result = plugin.execute({"operation": "metrics"})
    metrics = result.get("metrics", {})
    
    test("指标存在", metrics is not None)
    test("执行次数", metrics.get("execution_count", 0) >= 1)
    test("成功率", metrics.get("success_rate", 0) > 0)
    
    # Test 6: 生成报告
    print("\n[Test 6] Generate Report")
    print("-" * 40)
    
    result = plugin.execute({"operation": "report"})
    
    test("报告成功", result.get("success"))
    test("有运行时间", result.get("uptime_seconds") is not None)
    test("有指标", "metrics" in result)
    test("有最近记录", "recent_records" in result)
    
    # Test 7: 重置统计
    print("\n[Test 7] Reset Statistics")
    print("-" * 40)
    
    result = plugin.execute({"operation": "reset"})
    test("重置成功", result.get("success"))
    
    result = plugin.execute({"operation": "metrics"})
    metrics = result.get("metrics", {})
    test("执行次数已清零", metrics.get("execution_count") == 0)
    
    # Test 8: 监控上下文管理器
    print("\n[Test 8] Monitor Context Manager")
    print("-" * 40)
    
    with MonitorContext(plugin) as ctx:
        _ = dummy_function(3, y=4)
    
    test("上下文执行时间", ctx.execution_time > 0)
    test("上下文成功", ctx.success)
    
    # Test 9: 错误处理
    print("\n[Test 9] Error Handling")
    print("-" * 40)
    
    def failing_function():
        raise RuntimeError("Test error")
    
    result = plugin.execute({
        "operation": "wrap",
        "func": failing_function,
        "args": (),
        "kwargs": {},
    })
    
    test("错误被捕获", not result.get("success"))
    test("错误信息存在", result.get("error") is not None)
    
    # Test 10: 清理
    print("\n[Test 10] Cleanup")
    print("-" * 40)
    
    plugin.cleanup()
    test("清理完成", True)
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = test_monitor_plugin()
    exit(0 if success else 1)