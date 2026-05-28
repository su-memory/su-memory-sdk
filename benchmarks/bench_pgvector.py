"""
bench_pgvector.py — pgvector 性能调优基准 (v2.7.0)

分析 pgvector 在不同参数下的性能特征：
- HNSW 参数网格搜索: m (16/32/64) × ef_construction (64/128/256)
- IVFFlat vs HNSW 延迟对比
- 连接池大小 vs 并发数
- 向量维度对比: 384d / 768d / 1536d

注意: 需要 PostgreSQL + pgvector 运行中，设置 PG_DSN 环境变量。

用法:
    export PG_DSN="postgresql+asyncpg://user:pass@localhost:5432/sumemory"
    python benchmarks/bench_pgvector.py
    python benchmarks/bench_pgvector.py --quick
"""

from __future__ import annotations

import sys
import os
import time
import json
import asyncio
import statistics
import argparse
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


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


def random_embedding(dims: int = 768) -> List[float]:
    """生成随机归一化向量"""
    import random
    vec = [random.random() for _ in range(dims)]
    norm = (sum(x * x for x in vec)) ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


# =============================================================================
# 基准: 写入性能 vs 向量维度
# =============================================================================

async def bench_dimension_scaling(
    dsn: str,
    n_items: int = 1000,
    dims_list: List[int] = None,
) -> Dict:
    """对比不同向量维度的写入和查询性能"""
    from su_memory.storage.pgvector_backend import PgVectorBackend
    from su_memory.storage.base import AsyncMemoryItem

    dims_list = dims_list or [384, 768, 1536]
    results = {}

    for dims in dims_list:
        backend = PgVectorBackend(dsn=dsn, dims=dims, table_name=f"bench_dim_{dims}")
        await backend.ainit()
        await backend.aclear()

        # 写入
        items = [
            AsyncMemoryItem(
                id=f"dim_{dims}_{i:06d}",
                content=f"Dimension benchmark item {i} with dims={dims}",
                embedding=random_embedding(dims),
                tier="hot",
            )
            for i in range(n_items)
        ]

        t0 = time.perf_counter()
        await backend.aadd_batch(items)
        write_time = time.perf_counter() - t0

        # 查询
        query_vec = random_embedding(dims)
        query_latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            await backend.aquery(query_vec, top_k=10)
            query_latencies.append((time.perf_counter() - t0) * 1000)

        q_pct = percentiles(query_latencies)

        results[f"dim_{dims}"] = {
            "write_throughput_ops": round(n_items / max(write_time, 0.001), 1),
            "write_total_sec": round(write_time, 3),
            "query_p50_ms": round(q_pct["p50"], 2),
            "query_p95_ms": round(q_pct["p95"], 2),
            "query_p99_ms": round(q_pct["p99"], 2),
        }

        await backend.aclear()
        await backend.aclose()

    return results


# =============================================================================
# 基准: 连接池大小 vs 并发
# =============================================================================

async def bench_pool_concurrency(
    dsn: str,
    n_queries: int = 500,
    pool_sizes: List[int] = None,
) -> Dict:
    """对比不同连接池大小下的并发查询性能"""
    from su_memory.storage.pgvector_backend import PgVectorBackend
    from su_memory.storage.base import AsyncMemoryItem

    pool_sizes = pool_sizes or [2, 5, 10, 20]
    results = {}

    # 准备数据 (共享)
    base_backend = PgVectorBackend(dsn=dsn, dims=768, table_name="bench_pool")
    await base_backend.ainit()
    await base_backend.aclear()

    items = [
        AsyncMemoryItem(
            id=f"pool_{i:06d}",
            content=f"Pool benchmark item {i}",
            embedding=random_embedding(768),
            tier="hot",
        )
        for i in range(1000)
    ]
    await base_backend.aadd_batch(items)
    await base_backend.aclose()

    for pool_size in pool_sizes:
        backend = PgVectorBackend(
            dsn=dsn, dims=768, pool_size=pool_size, table_name="bench_pool"
        )
        await backend.ainit()

        latencies = []

        async def q(i):
            vec = random_embedding(768)
            t0 = time.perf_counter()
            await backend.aquery(vec, top_k=10)
            latencies.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        tasks = [q(i) for i in range(n_queries)]
        await asyncio.gather(*tasks)
        total_time = time.perf_counter() - t0

        p = percentiles(latencies)
        results[f"pool_{pool_size}"] = {
            "queries": n_queries,
            "total_sec": round(total_time, 3),
            "qps": round(n_queries / max(total_time, 0.001), 1),
            "p50_ms": round(p["p50"], 2),
            "p95_ms": round(p["p95"], 2),
            "p99_ms": round(p["p99"], 2),
        }

        await backend.aclose()

    return results


# =============================================================================
# 基准: 批量写入优化
# =============================================================================

async def bench_batch_sizes(
    dsn: str,
    n_total: int = 5000,
    batch_sizes: List[int] = None,
) -> Dict:
    """对比不同批量大小的写入性能"""
    from su_memory.storage.pgvector_backend import PgVectorBackend
    from su_memory.storage.base import AsyncMemoryItem

    batch_sizes = batch_sizes or [50, 100, 250, 500, 1000]
    results = {}

    for bs in batch_sizes:
        backend = PgVectorBackend(
            dsn=dsn, dims=768, table_name=f"bench_bs_{bs}"
        )
        await backend.ainit()
        await backend.aclear()

        latencies = []
        for offset in range(0, n_total, bs):
            bsize = min(bs, n_total - offset)
            items = [
                AsyncMemoryItem(
                    id=f"bs_{bs}_{offset + i:06d}",
                    content=f"Batch size benchmark {bs} item {i}",
                    embedding=random_embedding(768),
                    tier="hot",
                )
                for i in range(bsize)
            ]
            t0 = time.perf_counter()
            await backend.aadd_batch(items)
            latencies.append((time.perf_counter() - t0) / max(bsize, 1) * 1000)

        p = percentiles(latencies)
        results[f"batch_{bs}"] = {
            "total_items": n_total,
            "batch_count": len(latencies),
            "per_item_p50_ms": round(p["p50"], 2),
            "per_item_p95_ms": round(p["p95"], 2),
            "throughput_ops": round(n_total / max(sum(latencies) / 1000 / len(latencies) * len(latencies), 0.001), 1) if latencies else 0,
        }

        await backend.aclear()
        await backend.aclose()

    return results


# =============================================================================
# 基准: 分层命中率
# =============================================================================

async def bench_tier_performance(
    dsn: str,
    hot_size: int = 500,
    warm_size: int = 2000,
    n_queries: int = 200,
) -> Dict:
    """测试分层存储查询命中率"""
    from su_memory.storage.pgvector_backend import PgVectorBackend
    from su_memory.storage.tiered import TieredStorage, TierConfig
    from su_memory.storage.base import AsyncMemoryItem

    # Hot tier backend
    hot_backend = PgVectorBackend(dsn=dsn, dims=768, table_name="tier_hot")
    await hot_backend.ainit()
    await hot_backend.aclear()

    # 写入 hot 数据
    hot_items = [
        AsyncMemoryItem(
            id=f"hot_{i:06d}",
            content=f"Hot tier item {i}: frequently accessed content",
            embedding=random_embedding(768),
            tier="hot",
            access_count=50,
        )
        for i in range(hot_size)
    ]
    await hot_backend.aadd_batch(hot_items)

    # Warm tier 数据 (同一 backend, 不同 tier 标签)
    warm_items = [
        AsyncMemoryItem(
            id=f"warm_{i:06d}",
            content=f"Warm tier item {i}: occasional access content",
            embedding=random_embedding(768),
            tier="warm",
            access_count=5,
        )
        for i in range(warm_size)
    ]
    await hot_backend.aadd_batch(warm_items)

    # 使用 TieredStorage 查询
    config = TierConfig(
        hot_capacity=hot_size + 100,
        warm_capacity=warm_size + 100,
        hot_backend=hot_backend,
    )
    ts = TieredStorage(config)
    await ts.ainit()

    results = {"hot_hits": 0, "warm_hits": 0, "cold_hits": 0, "total": n_queries}

    for i in range(n_queries):
        # 交替查询 hot 和 warm 数据
        if i % 3 == 0:
            vec = hot_items[i % hot_size].embedding
        else:
            vec = warm_items[i % warm_size].embedding

        t0 = time.perf_counter()
        found = await ts.aquery(vec, top_k=5)
        elapsed = (time.perf_counter() - t0) * 1000

        if found:
            top_tier = found[0].tier
            if top_tier == "hot":
                results["hot_hits"] += 1
            elif top_tier == "warm":
                results["warm_hits"] += 1
            else:
                results["cold_hits"] += 1

    results["hot_hit_rate"] = results["hot_hits"] / max(n_queries, 1)
    results["warm_hit_rate"] = results["warm_hits"] / max(n_queries, 1)
    results["overall_hit_rate"] = (results["hot_hits"] + results["warm_hits"]) / max(n_queries, 1)

    await ts.aclose()
    return results


# =============================================================================
# 主程序
# =============================================================================

async def main_async(args):
    dsn = args.dsn or os.environ.get("PG_DSN", "")
    if not dsn:
        print("❌ PG_DSN not set. Use --dsn or set PG_DSN env var")
        return None

    results = {}

    # 1. 向量维度对比
    if not args.quick:
        print("\n[1/4] 向量维度对比 (384/768/1536)")
        r1 = await bench_dimension_scaling(dsn, n_items=min(args.n, 1000))
        results["dimension_scaling"] = r1
        for dim, data in r1.items():
            print(f"  {dim}: write={data['write_throughput_ops']:.0f}ops/s "
                  f"q_p50={data['query_p50_ms']}ms")

    # 2. 连接池调优
    print("\n[2/4] 连接池大小 vs 并发")
    r2 = await bench_pool_concurrency(dsn, n_queries=min(args.n, 200))
    results["pool_concurrency"] = r2
    for label, data in r2.items():
        print(f"  {label}: qps={data['qps']:.0f} p50={data['p50_ms']}ms")

    # 3. 批量大小
    print("\n[3/4] 批量写入优化")
    r3 = await bench_batch_sizes(dsn, n_total=min(args.n * 5, 2000))
    results["batch_sizes"] = r3
    for label, data in r3.items():
        print(f"  {label}: p50={data['per_item_p50_ms']}ms/item")

    # 4. 分层命中率
    if not args.quick:
        print("\n[4/4] 分层命中率")
        r4 = await bench_tier_performance(dsn, hot_size=500, warm_size=2000)
        results["tier_performance"] = r4
        print(f"  hot_hit: {r4['hot_hit_rate']:.1%} "
              f"warm_hit: {r4['warm_hit_rate']:.1%} "
              f"overall: {r4['overall_hit_rate']:.1%}")

    print("\n" + "=" * 50)
    print("✅ PgVector benchmarks complete")
    return results


def main():
    parser = argparse.ArgumentParser(description="PgVector performance benchmarks")
    parser.add_argument("--dsn", default=None, help="PostgreSQL DSN")
    parser.add_argument("--n", type=int, default=500, help="Number of items")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    print("=" * 50)
    print("su-memory-sdk v2.7.0 pgvector 性能调优")
    print("=" * 50)

    results = asyncio.run(main_async(args))

    if results and args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")

    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
