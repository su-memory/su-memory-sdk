#!/usr/bin/env python3
"""
su-memory v3.2.0 — Capacity Scaling Benchmark

测试 SuMemoryLite 在 10K/25K/50K 规模下的性能曲线：
1. 插入吞吐 (items/s)
2. 查询延迟分布 (P50/P95/P99)
3. 内存密度 (MB/1K)
4. 混合存储检索 (热层 + 温层)
5. 降级曲线 (100 → 50K)

用法: python benchmarks/benchmark_scaling.py [--quick]
      --quick: 仅测试 10K (快速模式)
"""

from __future__ import annotations

import sys
import os
import time
import json
import tempfile
import statistics
from typing import List, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from su_memory.sdk import SuMemoryLite

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCALES = [
    ("100",   100),
    ("1K",    1000),
    ("5K",    5000),
    ("10K",   10000),
    ("25K",   25000),
    ("50K",   50000),
]

QUERY_SAMPLES = 50  # queries per scale for latency measurement


def fmt_ms(ms: float) -> str:
    """Format milliseconds."""
    if ms < 1:
        return f"{ms*1000:.1f}µs"
    return f"{ms:.2f}ms"


def fmt_bytes(b: float) -> str:
    """Format bytes."""
    if b < 1024:
        return f"{b:.0f}B"
    elif b < 1024 * 1024:
        return f"{b/1024:.1f}KB"
    else:
        return f"{b/1024/1024:.1f}MB"


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_scale(n: int, max_memories: int = None) -> Dict[str, Any]:
    """Run benchmark at given scale."""
    if max_memories is None:
        max_memories = n * 2  # Allow room

    with tempfile.TemporaryDirectory() as tmp:
        engine = SuMemoryLite(
            storage_path=tmp,
            max_memories=max_memories,
            enable_persistence=False,  # v3.2.0: 性能基准跳过持久化 I/O
        )

        # --- Insert ---
        probes = []  # Every 10th item is a probe
        t0 = time.perf_counter()
        for i in range(n):
            if i % 10 == 0:
                content = f"容量基准探针唯一标识第{i}项核心数据"
                engine.add(content)
                probes.append(i)
            else:
                engine.add(f"容量基准填充噪声第{i}条无关紧要的内容")
        insert_elapsed = time.perf_counter() - t0
        insert_items_per_sec = n / max(insert_elapsed, 0.001)

        # --- Query probes ---
        query_times: List[float] = []
        sample_probes = probes[:QUERY_SAMPLES] if len(probes) >= QUERY_SAMPLES else probes

        for idx in sample_probes:
            t0 = time.perf_counter()
            results = engine.query(f"容量基准探针唯一标识第{idx}项", top_k=3)
            qt = (time.perf_counter() - t0) * 1000  # ms
            query_times.append(qt)

        # Check recall
        hits = 0
        for idx in probes:
            results = engine.query(f"容量基准探针唯一标识第{idx}项", top_k=3)
            for r in results:
                if "容量基准探针" in r["content"]:
                    hits += 1
                    break
        recall = hits / len(probes) if probes else 0

        # --- Memory ---
        mem_bytes = 0
        try:
            for m in engine._memories:
                mem_bytes += sys.getsizeof(m.get("content", ""))
            stats = engine.get_stats()
            mem_bytes += stats.get("index_size", 0) * 100
        except Exception:
            mem_bytes = sys.getsizeof(engine._memories)

        mem_mb = mem_bytes / (1024 * 1024)
        mem_per_1k = mem_mb / max(n, 1) * 1000

        # --- Warm layer stats ---
        warm_count = 0
        if engine._tiered_storage is not None:
            warm_count = engine._tiered_storage.warm_count

        return {
            "scale": n,
            "insert_ms": round(insert_elapsed * 1000, 1),
            "insert_items_per_sec": round(insert_items_per_sec),
            "query_p50_ms": round(percentile(query_times, 50), 3),
            "query_p95_ms": round(percentile(query_times, 95), 3),
            "query_p99_ms": round(percentile(query_times, 99), 3),
            "recall": round(recall, 4),
            "memory_mb": round(mem_mb, 2),
            "memory_per_1k_mb": round(mem_per_1k, 3),
            "warm_count": warm_count,
            "hot_count": len(engine),
        }


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile (linear interpolation)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100.0
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_data):
        return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
    return sorted_data[f]


def print_table(results: List[Dict[str, Any]]) -> None:
    """Print results table."""
    print()
    print(f"{'Scale':>6} {'Insert':>10} {'Items/s':>10} {'QryP50':>10} {'QryP95':>10} {'QryP99':>10} {'Recall':>8} {'Mem':>8} {'Mem/1K':>10} {'Warm':>6}")
    print("-" * 95)
    for r in results:
        print(
            f"{r['scale']:>6} "
            f"{fmt_ms(r['insert_ms']):>10} "
            f"{r['insert_items_per_sec']:>10,} "
            f"{fmt_ms(r['query_p50_ms']):>10} "
            f"{fmt_ms(r['query_p95_ms']):>10} "
            f"{fmt_ms(r['query_p99_ms']):>10} "
            f"{r['recall']:>7.1%} "
            f"{r['memory_mb']:>7.1f}M "
            f"{r['memory_per_1k_mb']:>9.3f}M "
            f"{r['warm_count']:>5}"
        )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Capacity Scaling Benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick mode (up to 10K)")
    args = parser.parse_args()

    print("📊 su-memory v3.2.0 — Capacity Scaling Benchmark")
    print(f"   Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Mode: {'QUICK (≤10K)' if args.quick else 'FULL (≤50K)'}")

    scales_to_run = [s for s in SCALES if not args.quick or s[1] <= 10000]

    results = []
    for label, n in scales_to_run:
        print(f"\n  [{label}] Running {n} items...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            r = run_scale(n)
            elapsed = time.perf_counter() - t0
            print(f"done ({elapsed:.1f}s)")
            results.append(r)
        except Exception as e:
            print(f"FAILED: {e}")

    print_table(results)

    # Summary
    if results:
        largest = results[-1]
        print(f"\n{'='*60}")
        print(f"  Summary at {largest['scale']} items:")
        print(f"    Insert: {largest['insert_items_per_sec']:,} items/s")
        print(f"    Query P95: {fmt_ms(largest['query_p95_ms'])}")
        print(f"    Memory: {largest['memory_mb']:.1f}MB ({largest['memory_per_1k_mb']:.3f}MB/1K)")
        print(f"    Recall: {largest['recall']:.1%}")
        print(f"    Warm layer: {largest['warm_count']} items")
        print(f"{'='*60}")

    # Save
    output = {
        "version": "v3.2.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(
        out_dir,
        f"scaling_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Results: {out_file}")


if __name__ == "__main__":
    main()
