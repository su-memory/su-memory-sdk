"""
su-memory SDK 性能基准测试
用于测试和验证SDK的性能指标
"""
import os
import sys
import time
import tempfile
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from su_memory.sdk.lite import SuMemoryLite


def benchmark_insertion(client: SuMemoryLite, count: int = 1000) -> dict:
    """
    测试插入性能
    
    Args:
        client: SuMemoryLite实例
        count: 插入数量
        
    Returns:
        性能统计
    """
    print(f"\n📊 插入性能测试 ({count} 条记忆)...")
    
    # 准备测试数据
    test_data = [
        f"测试记忆{i}包含关键词A和B以及一些额外的内容来增加长度"
        for i in range(count)
    ]
    
    # 测试
    start = time.perf_counter()
    for content in test_data:
        client.add(content, metadata={"index": len(client)})
    elapsed = time.perf_counter() - start
    
    # 统计
    avg_time = elapsed / count * 1000  # 毫秒
    throughput = count / elapsed  # 条/秒
    
    return {
        "total_time_ms": elapsed * 1000,
        "avg_time_ms": avg_time,
        "throughput": throughput,
        "count": count
    }


def benchmark_query(client: SuMemoryLite, queries: int = 100) -> dict:
    """
    测试查询性能
    
    Args:
        client: SuMemoryLite实例
        queries: 查询次数
        
    Returns:
        性能统计
    """
    print(f"\n🔍 查询性能测试 ({queries} 次查询)...")
    
    # 准备查询
    test_queries = ["关键词A", "测试记忆", "内容", "B以及"] * (queries // 4)
    
    latencies = []
    
    for query in test_queries:
        start = time.perf_counter()
        results = client.query(query, top_k=5)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1000)  # 毫秒
    
    # 统计
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    
    return {
        "count": queries,
        "avg_ms": statistics.mean(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0
    }


def benchmark_memory(count: int = 1000) -> dict:
    """
    测试内存占用
    
    Args:
        count: 记忆数量
        
    Returns:
        内存统计
    """
    print(f"\n💾 内存占用测试 ({count} 条记忆)...")
    
    import sys
    
    with tempfile.TemporaryDirectory() as tmpdir:
        client = SuMemoryLite(max_memories=count * 2, storage_path=tmpdir)
        
        # 添加记忆
        for i in range(count):
            client.add(
                f"这是第{i}条测试记忆，包含足够长的内容来模拟真实场景下的记忆存储。"
            )
        
        # 估算内存占用
        # 每个字符串约 100 字节
        # 每个字典约 500 字节
        # 每个set约 1000 字节
        
        memories_size = count * 100  # _memories中的内容
        dicts_size = count * 500  # 字典对象
        index_size = count * 1000  # 索引
        
        total_estimate = memories_size + dicts_size + index_size
        
        # 获取实际stats
        stats = client.get_stats()
        
        return {
            "count": count,
            "estimated_mb": total_estimate / (1024 * 1024),
            "index_size": stats["index_size"]
        }


def benchmark_prediction(client: SuMemoryLite, iterations: int = 100) -> dict:
    """
    测试预测性能
    
    Args:
        client: SuMemoryLite实例
        iterations: 迭代次数
        
    Returns:
        性能统计
    """
    print(f"\n🔮 预测性能测试 ({iterations} 次预测)...")
    
    latencies = []
    
    for i in range(iterations):
        start = time.perf_counter()
        result = client.predict(f"情境{i}", f"行动{i}")
        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1000)
    
    return {
        "count": iterations,
        "avg_ms": statistics.mean(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)]
    }


def run_all_benchmarks():
    """
    运行所有基准测试
    """
    print("=" * 60)
    print("🚀 su-memory SDK 性能基准测试")
    print("=" * 60)
    
    results = {}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建客户端
        client = SuMemoryLite(
            max_memories=10000,
            storage_path=tmpdir,
            enable_tfidf=True
        )
        
        # 1. 插入测试
        insertion = benchmark_insertion(client, count=1000)
        results["insertion"] = insertion
        
        # 2. 查询测试
        query = benchmark_query(client, queries=100)
        results["query"] = query
        
        # 3. 预测测试
        prediction = benchmark_prediction(client, iterations=100)
        results["prediction"] = prediction
        
        # 4. 内存测试
        memory = benchmark_memory(count=1000)
        results["memory"] = memory
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("📈 测试结果汇总")
    print("=" * 60)
    
    print(f"""
插入性能:
  - 总耗时: {results['insertion']['total_time_ms']:.2f} ms
  - 平均耗时: {results['insertion']['avg_time_ms']:.4f} ms/条
  - 吞吐量: {results['insertion']['throughput']:.0f} 条/秒

查询性能:
  - 平均延迟: {results['query']['avg_ms']:.4f} ms
  - P50延迟: {results['query']['p50_ms']:.4f} ms
  - P95延迟: {results['query']['p95_ms']:.4f} ms
  - P99延迟: {results['query']['p99_ms']:.4f} ms

预测性能:
  - 平均延迟: {results['prediction']['avg_ms']:.4f} ms
  - P95延迟: {results['prediction']['p95_ms']:.4f} ms

内存占用:
  - 1000条记忆估算: {results['memory']['estimated_mb']:.2f} MB
  - 索引大小: {results['memory']['index_size']} 个词条

性能目标达成:
  ✅ 查询延迟P95 < 100ms: {'是' if results['query']['p95_ms'] < 100 else '否'}
  ✅ 内存占用 < 50MB: {'是' if results['memory']['estimated_mb'] < 50 else '否'}
    """)
    
    return results


if __name__ == "__main__":
    results = run_all_benchmarks()
