#!/usr/bin/env python
"""
su-memory SDK v1.7.1 性能基准测试
验证修复后的性能指标
"""
import time
import sys
import os
import tracemalloc
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from su_memory.sdk import SuMemoryLite, SuMemoryLitePro


def format_bytes(bytes_val: float) -> str:
    """格式化字节数"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} TB"


def benchmark_add(lite: SuMemoryLite, count: int) -> dict:
    """测试添加性能"""
    gc.collect()
    tracemalloc.start()
    
    start = time.perf_counter()
    for i in range(count):
        lite.add(f"测试记忆 {i} 包含关键词A和B和C")
    end = time.perf_counter()
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        "total_time": end - start,
        "avg_time": (end - start) / count * 1000,  # ms
        "count": count,
        "memory_peak": format_bytes(peak),
        "memory_current": format_bytes(current)
    }


def benchmark_query(lite: SuMemoryLite, query_count: int) -> dict:
    """测试查询性能"""
    queries = ["测试", "关键词", "记忆", "ABC", "测试关键词"]
    
    gc.collect()
    tracemalloc.start()
    
    times = []
    for i in range(query_count):
        query = queries[i % len(queries)]
        start = time.perf_counter()
        lite.query(query, top_k=5)
        end = time.perf_counter()
        times.append(end - start)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    times_ms = [t * 1000 for t in times]
    times_ms.sort()
    
    return {
        "query_count": query_count,
        "p50": times_ms[len(times_ms) // 2],
        "p95": times_ms[int(len(times_ms) * 0.95)],
        "p99": times_ms[int(len(times_ms) * 0.99)] if len(times_ms) > 100 else times_ms[-1],
        "avg": sum(times_ms) / len(times_ms),
        "memory_peak": format_bytes(peak)
    }


def benchmark_max_memories(lite: SuMemoryLite, max_limit: int) -> dict:
    """测试max_memories限制"""
    lite.clear()
    
    # 添加超过限制的数据
    for i in range(max_limit + 100):
        lite.add(f"记忆 {i}")
    
    actual_count = len(lite._memories)
    
    return {
        "max_limit": max_limit,
        "actual_count": actual_count,
        "limit_enforced": actual_count <= max_limit,
        "excess_prevented": actual_count <= max_limit + 1  # 允许1个误差
    }


def benchmark_persistence(lite: SuMemoryLite, count: int) -> dict:
    """测试持久化性能"""
    lite.clear()
    
    # 添加数据
    for i in range(count):
        lite.add(f"持久化测试 {i}")
    
    # 测试保存
    gc.collect()
    start = time.perf_counter()
    lite._save()
    save_time = time.perf_counter() - start
    save_size = os.path.getsize(lite._get_storage_file()) if lite._get_storage_file() else 0
    
    # 测试加载
    gc.collect()
    start = time.perf_counter()
    lite._load()
    load_time = time.perf_counter() - start
    
    return {
        "count": count,
        "save_time": save_time * 1000,  # ms
        "load_time": load_time * 1000,  # ms
        "file_size": format_bytes(save_size)
    }


def main():
    print("=" * 60)
    print("su-memory SDK v1.7.1 性能基准测试")
    print("=" * 60)
    
    # 测试配置
    SCALES = [
        ("小规模", 100),
        ("中规模", 1000),
        ("大规模", 5000),
    ]
    
    MAX_LIMITS = [10, 100, 1000]
    
    print("\n" + "=" * 60)
    print("1. 添加性能测试")
    print("=" * 60)
    
    for scale_name, count in SCALES:
        lite = SuMemoryLite(max_memories=count * 2, enable_persistence=False)
        result = benchmark_add(lite, count)
        print(f"\n【{scale_name}】({count}条记忆)")
        print(f"  总耗时: {result['total_time']*1000:.2f} ms")
        print(f"  平均耗时: {result['avg_time']:.3f} ms/条")
        print(f"  内存峰值: {result['memory_peak']}")
    
    print("\n" + "=" * 60)
    print("2. 查询性能测试")
    print("=" * 60)
    
    for scale_name, count in SCALES:
        lite = SuMemoryLite(max_memories=count * 2, enable_persistence=False)
        # 添加测试数据
        for i in range(count):
            lite.add(f"测试记忆 {i} 包含关键词A和B和C")
        
        result = benchmark_query(lite, 100)
        print(f"\n【{scale_name}】({count}条记忆, 100次查询)")
        print(f"  P50: {result['p50']:.3f} ms")
        print(f"  P95: {result['p95']:.3f} ms")
        print(f"  P99: {result['p99']:.3f} ms")
        print(f"  平均: {result['avg']:.3f} ms")
        print(f"  内存峰值: {result['memory_peak']}")
    
    print("\n" + "=" * 60)
    print("3. max_memories限制测试")
    print("=" * 60)
    
    for max_limit in MAX_LIMITS:
        lite = SuMemoryLite(max_memories=max_limit, enable_persistence=False)
        result = benchmark_max_memories(lite, max_limit)
        status = "✅" if result['limit_enforced'] else "❌"
        print(f"\n【max_memories={max_limit}】{status}")
        print(f"  限制生效: {result['limit_enforced']}")
        print(f"  实际数量: {result['actual_count']}")
    
    print("\n" + "=" * 60)
    print("4. 持久化性能测试")
    print("=" * 60)
    
    for scale_name, count in SCALES[:2]:  # 只测试小规模和中等规模
        lite = SuMemoryLite(max_memories=count * 2, enable_persistence=False)
        result = benchmark_persistence(lite, count)
        print(f"\n【{scale_name}】({count}条记忆)")
        print(f"  保存耗时: {result['save_time']:.2f} ms")
        print(f"  加载耗时: {result['load_time']:.2f} ms")
        print(f"  文件大小: {result['file_size']}")
    
    print("\n" + "=" * 60)
    print("5. SuMemoryLitePro 基础测试")
    print("=" * 60)
    
    try:
        lite_pro = SuMemoryLitePro()
        lite_pro.add("测试记忆1 包含关键词")
        lite_pro.add("测试记忆2 包含关键词")
        results = lite_pro.query("测试", top_k=5)
        print(f"\n✅ SuMemoryLitePro 基础功能正常")
        print(f"  添加记忆数: 2")
        print(f"  查询结果数: {len(results)}")
    except Exception as e:
        print(f"\n❌ SuMemoryLitePro 测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
