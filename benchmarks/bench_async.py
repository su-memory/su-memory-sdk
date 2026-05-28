"""
bench_async.py — 异步性能基准测试 (v2.7.0)

对比同步 vs 异步的性能差异，测量：
- aadd 吞吐：同步 vs 异步单条写入
- aquery 延迟：单条 vs 并发 10/50/100
- astream_query 首字节时间 vs 完整响应时间
- 异步并发写入：asyncio.gather(N tasks)

用法:
    python benchmarks/bench_async.py
    python benchmarks/bench_async.py --concurrency 100
    python benchmarks/bench_async.py --stream
"""

from __future__ import annotations

import sys
import os
import time
import json
import asyncio
import statistics
import tempfile
import shutil
from typing import List, Dict, Any
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# =============================================================================
# 工具
# =============================================================================

def percentiles(data: List[float]) -> Dict[str, float]:
    if not data:
        return {"p50": 0, "p95": 0, "p99": 0}
    s = sorted(data)
    n = len(s)
    return {
        "p50": s[int(n * 0.50)] if n >= 2 else s[0],
        "p95": s[int(n * 0.95)] if n >= 20 else s[-1],
        "p99": s[int(n * 0.99)] if n >= 100 else s[-1],
    }


# =============================================================================
# 基准测试
# =============================================================================

async def bench_async_vs_sync_add(n: int = 100) -> Dict:
    """对比同步 vs 异步 aadd"""
    from su_memory import SuMemory

    d = tempfile.mkdtemp()
    try:
        # 同步基准
        client = SuMemory(storage_path=d)
        contents = [f"sync_add_{i:05d}: benchmark content" for i in range(n)]

        t0 = time.perf_counter()
        for c in contents:
            client.add(c)
        sync_time = time.perf_counter() - t0
        sync_tput = n / sync_time

        # 异步基准
        from su_memory.async_client import AsyncSuMemory

        async_client = await AsyncSuMemory.create(storage_path=d + "_async")
        contents_async = [f"async_add_{i:05d}: benchmark content" for i in range(n)]

        t0 = time.perf_counter()
        tasks = [async_client.aadd(c) for c in contents_async]
        await asyncio.gather(*tasks)
        async_time = time.perf_counter() - t0
        async_tput = n / async_time

        await async_client.aclose()

        return {
            "sync": {
                "total_sec": round(sync_time, 3),
                "throughput_ops": round(sync_tput, 1),
                "avg_ms": round(sync_time / n * 1000, 2),
            },
            "async": {
                "total_sec": round(async_time, 3),
                "throughput_ops": round(async_tput, 1),
                "avg_ms": round(async_time / n * 1000, 2),
            },
            "speedup_x": round(sync_time / max(async_time, 0.001), 2),
        }
    finally:
        shutil.rmtree(d, ignore_errors=True)


async def bench_async_concurrent_query(
    n_memories: int = 1000, concurrency: int = 50
) -> Dict:
    """并发查询压测"""
    from su_memory import SuMemory
    import random

    d = tempfile.mkdtemp()
    try:
        # 准备数据
        client = SuMemory(storage_path=d)
        batch = 200
        for offset in range(0, n_memories, batch):
            bsize = min(batch, n_memories - offset)
            items = [f"bench_query_{offset + i:06d}" for i in range(bsize)]
            client.add_batch(items)

        # 异步并发查询
        latencies = []

        async def do_query(i):
            idx = random.randint(0, n_memories - 1)
            q = f"bench_query_{idx:06d}"
            t0 = time.perf_counter()
            await asyncio.to_thread(client.query, q, top_k=5)
            latencies.append((time.perf_counter() - t0) * 1000)

        tasks = [do_query(i) for i in range(concurrency)]
        t0 = time.perf_counter()
        await asyncio.gather(*tasks)
        total_time = time.perf_counter() - t0

        p = percentiles(latencies)
        return {
            "concurrency": concurrency,
            "total_sec": round(total_time, 3),
            "queries_per_sec": round(concurrency / max(total_time, 0.001), 1),
            "query_latency": {
                "p50_ms": round(p["p50"], 2),
                "p95_ms": round(p["p95"], 2),
                "p99_ms": round(p["p99"], 2),
            },
        }
    finally:
        shutil.rmtree(d, ignore_errors=True)


async def bench_stream_first_byte(n_memories: int = 500) -> Dict:
    """流式查询首字节时间 vs 完整响应时间"""
    from su_memory import SuMemory

    d = tempfile.mkdtemp()
    try:
        client = SuMemory(storage_path=d)
        items = [f"stream_bench_{i:06d}: streaming benchmark content number {i}"
                 for i in range(n_memories)]
        client.add_batch(items)

        # 测量流式查询
        async def measure_stream():
            t_start = time.perf_counter()
            first_byte = None
            chunk_count = 0

            async for chunk in client.astream_query("stream_bench_000123", top_k=5):
                if first_byte is None:
                    first_byte = time.perf_counter()
                chunk_count += 1

            complete_time = time.perf_counter()
            return {
                "first_byte_ms": round((first_byte - t_start) * 1000, 2) if first_byte else 0,
                "complete_ms": round((complete_time - t_start) * 1000, 2),
                "chunks_received": chunk_count,
            }

        result = await measure_stream()
        return result
    finally:
        shutil.rmtree(d, ignore_errors=True)


async def bench_async_batch_concurrent(
    n_items: int = 5000, n_tasks: int = 10
) -> Dict:
    """异步批量并发写入"""
    from su_memory.async_client import AsyncSuMemory

    d = tempfile.mkdtemp()
    try:
        async_client = await AsyncSuMemory.create(storage_path=d)
        per_task = n_items // n_tasks

        async def write_task(tid: int):
            items = [
                f"batch_async_{tid}_{i:06d}: concurrent batch write benchmark"
                for i in range(per_task)
            ]
            t0 = time.perf_counter()
            await async_client.aadd_batch(items)
            return time.perf_counter() - t0

        t0 = time.perf_counter()
        task_times = await asyncio.gather(*[write_task(i) for i in range(n_tasks)])
        total_time = time.perf_counter() - t0

        await async_client.aclose()

        return {
            "n_tasks": n_tasks,
            "total_items": n_items,
            "total_sec": round(total_time, 3),
            "throughput_ops": round(n_items / max(total_time, 0.001), 1),
            "task_times_p50_ms": round(
                statistics.median(task_times) * 1000, 2
            ),
        }
    finally:
        shutil.rmtree(d, ignore_errors=True)


# =============================================================================
# 主程序
# =============================================================================

async def main_async(args):
    results = {}

    # 1. 同步 vs 异步 aadd
    print("\n[1/4] 同步 vs 异步 add")
    r = await bench_async_vs_sync_add(n=args.n)
    results["sync_vs_async_add"] = r
    print(f"  sync:  {r['sync']['throughput_ops']:.0f} ops/s")
    print(f"  async: {r['async']['throughput_ops']:.0f} ops/s")
    print(f"  speedup: {r['speedup_x']}x")

    # 2. 并发查询
    print(f"\n[2/4] 并发查询 (c={args.concurrency})")
    r2 = await bench_async_concurrent_query(
        n_memories=args.n_memories, concurrency=args.concurrency
    )
    results["concurrent_query"] = r2
    print(f"  qps: {r2['queries_per_sec']:.0f} query/s")
    print(f"  p50: {r2['query_latency']['p50_ms']}ms")

    # 3. 流式查询
    if args.stream:
        print("\n[3/4] 流式查询首字节")
        r3 = await bench_stream_first_byte(n_memories=min(args.n_memories, 500))
        results["stream"] = r3
        print(f"  first_byte: {r3['first_byte_ms']}ms")
        print(f"  complete:   {r3['complete_ms']}ms")

    # 4. 异步批量并发
    print("\n[4/4] 异步批量并发写入")
    r4 = await bench_async_batch_concurrent(
        n_items=min(args.n_memories, 5000), n_tasks=10
    )
    results["async_batch_concurrent"] = r4
    print(f"  throughput: {r4['throughput_ops']:.0f} ops/s")

    print("\n" + "=" * 50)
    print("✅ Async benchmarks complete")
    return results


def main():
    parser = argparse.ArgumentParser(description="Async performance benchmarks")
    parser.add_argument("--n", type=int, default=100, help="Number of items for add")
    parser.add_argument("--n-memories", type=int, default=1000)
    parser.add_argument("--concurrency", "-c", type=int, default=50)
    parser.add_argument("--stream", action="store_true", help="Include stream benchmarks")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    print("=" * 50)
    print("su-memory-sdk v2.7.0 异步性能基准")
    print("=" * 50)

    results = asyncio.run(main_async(args))

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
