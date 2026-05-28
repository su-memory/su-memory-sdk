#!/usr/bin/env python3
"""
su-memory v3.0.0 — SOTA Benchmark Suite

Comprehensive performance benchmark covering:
  1. SuMemoryLite add/query/count/memory at scale (100/1K/10K)
  2. Storage backend comparison (default JSON vs SQLite)
  3. PluginManager auto_discover + health_report
  4. SuMemoryLitePro baseline
  5. Regression check vs v2.x targets

Output: benchmarks/benchmark_results.json + console report
"""

from __future__ import annotations

import gc
import json
import os
import statistics
import sys
import tempfile
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from su_memory.sdk import SuMemoryLite, SuMemoryLitePro, MemoryProtocol
from su_memory.sdk.plugin_manager import PluginManager


# =============================================================================
# Config
# =============================================================================

SCALES = [
    ("100", 100),
    ("1K", 1000),
    ("10K", 10000),
]

QUERY_COUNT = 100
BENCHMARK_OUTPUT = os.path.join(os.path.dirname(__file__), "benchmark_results.json")

# v2.x baseline targets (from BENCHMARK.md)
V2_TARGETS = {
    "insert_throughput_1k": 500,      # items/sec minimum
    "query_p95_ms": 100,               # P95 < 100ms
    "memory_per_1k_mb": 50,            # < 50MB per 1000 items
    "plugin_discover_ms": 5000,        # auto_discover < 5s
}


# =============================================================================
# Helpers
# =============================================================================

def format_bytes(b: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if b < 1024:
            return f"{b:.2f} {unit}"
        b /= 1024
    return f"{b:.2f} TB"


def format_time(ms: float) -> str:
    if ms < 1:
        return f"{ms*1000:.1f} µs"
    if ms < 1000:
        return f"{ms:.2f} ms"
    return f"{ms/1000:.2f} s"


def fmt_pct(v: float) -> str:
    return f"{v*100:.1f}%" if v <= 1 else f"{v:.1f}%"


def perc(latencies: List[float], p: float) -> float:
    """Percentile from sorted list."""
    if not latencies:
        return 0.0
    latencies.sort()
    idx = max(0, min(len(latencies) - 1, int(len(latencies) * p / 100)))
    return latencies[idx]


# =============================================================================
# Benchmark 1: SuMemoryLite CRUD
# =============================================================================

def bench_lite_add(lite: SuMemoryLite, count: int) -> Dict:
    """Insertion throughput."""
    gc.collect()
    t0 = time.perf_counter()
    for i in range(count):
        lite.add(f"bench记忆{i} 包含关键词Alpha和Beta以及Gamma")
    elapsed = time.perf_counter() - t0
    return {
        "count": count,
        "total_ms": elapsed * 1000,
        "avg_ms_per_item": elapsed / count * 1000,
        "throughput_per_sec": count / elapsed,
    }


def bench_lite_query(lite: SuMemoryLite, n_queries: int = 100) -> Dict:
    """Query latency percentiles."""
    queries = ["Alpha", "Beta", "Gamma", "记忆", "bench", "关键词", "Alpha Beta"]
    lats = []
    for i in range(n_queries):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        lite.query(q, top_k=5)
        lats.append((time.perf_counter() - t0) * 1000)
    lats.sort()
    return {
        "count": n_queries,
        "avg_ms": statistics.mean(lats),
        "min_ms": lats[0],
        "max_ms": lats[-1],
        "p50_ms": perc(lats, 50),
        "p95_ms": perc(lats, 95),
        "p99_ms": perc(lats, 99),
    }


def bench_lite_count(lite: SuMemoryLite, iterations: int = 50) -> Dict:
    """Count operation speed."""
    lats = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        lite.count()
        lats.append((time.perf_counter() - t0) * 1000)
    lats.sort()
    return {
        "iterations": iterations,
        "avg_ms": statistics.mean(lats),
        "p95_ms": perc(lats, 95),
        "p99_ms": perc(lats, 99),
    }


def bench_lite_memory(lite: SuMemoryLite) -> Dict:
    """Memory footprint estimation."""
    stats = lite.get_stats()
    mem_count = len(lite._memories)
    import sys as _sys
    # Rough estimate from object sizes
    content_size = sum(_sys.getsizeof(m.get("content", "")) for m in lite._memories)
    dict_size = sum(_sys.getsizeof(m) for m in lite._memories)
    index_size = _sys.getsizeof(lite._index) if hasattr(lite, "_index") else 0
    total_est = content_size + dict_size + index_size
    return {
        "memory_count": mem_count,
        "estimated_mb": total_est / (1024 * 1024),
        "per_item_bytes": total_est / max(mem_count, 1),
        "index_entries": stats.get("index_size", 0),
    }


# =============================================================================
# Benchmark 2: Storage Backend Comparison
# =============================================================================

def bench_storage_backend(label: str, storage_backend_type: str,
                          count: int, storage_path: str) -> Dict:
    """Benchmark a specific storage backend."""
    lite = SuMemoryLite(
        max_memories=count * 2,
        storage_path=storage_path,
        storage_backend=storage_backend_type,
    )
    mems = [f"storage-test-{i} content for backend comparison benchmark" for i in range(count)]

    # Add
    gc.collect()
    t0 = time.perf_counter()
    for m in mems:
        lite.add(m)
    add_elapsed = time.perf_counter() - t0

    # Query
    q_lats = []
    for i in range(min(count, 50)):
        t0 = time.perf_counter()
        lite.query(f"storage-test-{i}", top_k=5)
        q_lats.append((time.perf_counter() - t0) * 1000)

    backend = lite.get_storage_backend()
    backend_info = str(backend.backend_type.value) if backend else "none"

    return {
        "label": label,
        "backend": backend_info,
        "backend_type": storage_backend_type,
        "count": count,
        "add_total_ms": add_elapsed * 1000,
        "add_avg_ms": add_elapsed / count * 1000,
        "add_throughput": count / add_elapsed if add_elapsed > 0 else 0,
        "query_avg_ms": statistics.mean(q_lats) if q_lats else 0,
        "query_p95_ms": perc(q_lats, 95),
    }


# =============================================================================
# Benchmark 3: PluginManager
# =============================================================================

def bench_plugin_manager() -> Dict:
    """PluginManager auto_discover + health_report."""
    pm = PluginManager()

    # Auto-discover
    gc.collect()
    t0 = time.perf_counter()
    n_registered = pm.auto_discover()
    discover_ms = (time.perf_counter() - t0) * 1000

    # Initialize all
    t0 = time.perf_counter()
    init_status = pm.initialize_all()
    init_ms = (time.perf_counter() - t0) * 1000
    n_initialized = sum(1 for v in init_status.values() if v) if init_status else 0

    # Health report
    t0 = time.perf_counter()
    report = pm.health_report()
    health_ms = (time.perf_counter() - t0) * 1000

    return {
        "registered_plugins": n_registered,
        "discover_ms": discover_ms,
        "initialized_plugins": n_initialized,
        "init_ms": init_ms,
        "health_report_ms": health_ms,
        "health_summary": {
            k: v for k, v in report.items()
            if k in ("total_plugins", "healthy", "unhealthy", "error")
        } if report else {},
    }


# =============================================================================
# Benchmark 4: SuMemoryLitePro
# =============================================================================

def bench_lite_pro(count: int, storage_path: str) -> Dict:
    """SuMemoryLitePro baseline."""
    try:
        pro = SuMemoryLitePro(
            max_memories=count * 2,
            storage_path=storage_path,
            enable_tfidf=True,
        )

        # Add
        t0 = time.perf_counter()
        for i in range(count):
            pro.add(f"pro-memory-{i} 包含 Alpha Beta Gamma Delta")
        add_ms = (time.perf_counter() - t0) * 1000

        # Query
        q_lats = []
        queries = ["Alpha", "Beta", "Gamma", "Delta", "pro-memory"]
        for _ in range(50):
            for q in queries:
                t0 = time.perf_counter()
                pro.query(q, top_k=5)
                q_lats.append((time.perf_counter() - t0) * 1000)
        q_lats.sort()

        return {
            "count": count,
            "add_total_ms": add_ms,
            "add_avg_ms": add_ms / count,
            "add_throughput": count / (add_ms / 1000) if add_ms > 0 else 0,
            "query_avg_ms": statistics.mean(q_lats) if q_lats else 0,
            "query_p50_ms": perc(q_lats, 50),
            "query_p95_ms": perc(q_lats, 95),
            "count_operation": pro.count(),
            "memory_protocol": isinstance(pro, MemoryProtocol),
        }
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# Benchmark 5: v3.0.0 New Features
# =============================================================================

def bench_v3_features() -> Dict:
    """New v3.0.0-specific features."""
    results = {}

    # 5a: PgStorageBackend import check (no runtime PG needed)
    try:
        from su_memory._sys._pg_storage import PgStorageBackend
        from su_memory._sys._storage_backend import BackendType, StorageConfig
        cfg = StorageConfig(
            pg_host="localhost", pg_port=5432, pg_database="test",
            pg_user="test", pg_password="test",
        )
        pg = PgStorageBackend(cfg)
        results["pg_import"] = "OK"
        results["pg_health_available"] = "no local PG (import checks only)"
    except ImportError:
        results["pg_import"] = "asyncpg not installed"
    except Exception as e:
        results["pg_import"] = f"Error: {e}"

    # 5b: RedisStorageBackend import check
    try:
        from su_memory._sys._redis_storage import RedisStorageBackend
        results["redis_import"] = "OK"
    except ImportError:
        results["redis_import"] = "redis not installed"
    except Exception as e:
        results["redis_import"] = f"Error: {e}"

    # 5c: SqliteStorageBackend integration
    with tempfile.TemporaryDirectory() as tmp:
        try:
            from su_memory._sys._sqlite_storage import SqliteStorageBackend
            from su_memory._sys._storage_backend import StorageConfig, BackendType, create_backend
            import asyncio

            async def _test():
                cfg = StorageConfig(sqlite_path=os.path.join(tmp, "v3test.db"))
                be = SqliteStorageBackend(cfg)
                ok = await be.initialize()
                if ok:
                    await be.add("v3-001", "test content")
                    cnt = await be.count()
                    await be.close()
                    return cnt
                return -1

            cnt = asyncio.run(_test())
            results["sqlite_storage_backend"] = f"OK (count={cnt})"
        except Exception as e:
            results["sqlite_storage_backend"] = f"Error: {e}"

    # 5d: _storage_helpers shared module
    try:
        from su_memory.sdk._storage_helpers import init_storage_backend
        results["shared_helpers"] = "OK"
    except Exception as e:
        results["shared_helpers"] = f"Error: {e}"

    return results


# =============================================================================
# Regression Check
# =============================================================================

def check_regression(results: Dict) -> List[Dict]:
    """Compare vs v2.x targets."""
    checks = []

    # Insert throughput at 1K
    if "insertion_1k" in results:
        tp = results["insertion_1k"].get("throughput_per_sec", 0)
        target = V2_TARGETS["insert_throughput_1k"]
        checks.append({
            "metric": "Insert throughput (1K)",
            "value": f"{tp:.0f} items/s",
            "target": f">{target} items/s",
            "pass": tp >= target,
        })

    # Query P95 at 1K
    if "query_1k" in results:
        p95 = results["query_1k"].get("p95_ms", 999)
        target = V2_TARGETS["query_p95_ms"]
        checks.append({
            "metric": "Query P95 latency (1K)",
            "value": f"{p95:.2f} ms",
            "target": f"<{target} ms",
            "pass": p95 < target,
        })

    # Memory at 1K
    if "memory_1k" in results:
        mem = results["memory_1k"].get("estimated_mb", 999)
        target = V2_TARGETS["memory_per_1k_mb"]
        checks.append({
            "metric": "Memory (1K items)",
            "value": f"{mem:.2f} MB",
            "target": f"<{target} MB",
            "pass": mem < target,
        })

    # Plugin discover speed
    if "plugin_manager" in results:
        discover = results["plugin_manager"].get("discover_ms", 99999)
        target = V2_TARGETS["plugin_discover_ms"]
        checks.append({
            "metric": "Plugin auto_discover",
            "value": f"{discover:.0f} ms",
            "target": f"<{target} ms",
            "pass": discover < target,
        })

    return checks


# =============================================================================
# Report Generation
# =============================================================================

def print_report(results: Dict, checks: List[Dict]) -> None:
    """Pretty-print benchmark report."""
    W = 70
    print()
    print("=" * W)
    print("  su-memory v3.0.0 — SOTA Benchmark Report".center(W))
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(W))
    print("=" * W)

    # ── SDK Core Performance ──
    print(f"\n{'─' * W}")
    print("  1. SuMemoryLite CRUD Performance")
    print(f"{'─' * W}")
    header = f"{'Scale':>6} {'Add(ms)':>10} {'Avg/Item':>10} {'Items/s':>10} {'Qry P50':>10} {'Qry P95':>10} {'Qry P99':>10} {'Count(ms)':>10}"
    print(header)
    print("-" * len(header))
    for lbl in ["100", "1K", "10K"]:
        add_key = f"insertion_{lbl.lower()}"
        qry_key = f"query_{lbl.lower()}"
        cnt_key = f"count_{lbl.lower()}"
        add = results.get(add_key, {})
        qry = results.get(qry_key, {})
        cnt = results.get(cnt_key, {})
        print(
            f"{lbl:>6} "
            f"{add.get('total_ms', 0):>10.1f} "
            f"{add.get('avg_ms_per_item', 0):>10.3f} "
            f"{add.get('throughput_per_sec', 0):>10.0f} "
            f"{qry.get('p50_ms', 0):>10.2f} "
            f"{qry.get('p95_ms', 0):>10.2f} "
            f"{qry.get('p99_ms', 0):>10.2f} "
            f"{cnt.get('avg_ms', 0):>10.4f}"
        )

    # ── Memory ──
    print(f"\n{'─' * W}")
    print("  2. Memory Footprint")
    print(f"{'─' * W}")
    for lbl in ["100", "1K", "10K"]:
        mem_key = f"memory_{lbl.lower()}"
        mem = results.get(mem_key, {})
        print(
            f"  {lbl:>4} items: {mem.get('estimated_mb', 0):.2f} MB "
            f"({mem.get('per_item_bytes', 0):.0f} B/item) "
            f"│ index entries: {mem.get('index_entries', 0)}"
        )

    # ── Storage Backend ──
    print(f"\n{'─' * W}")
    print("  3. Storage Backend Comparison (1K items)")
    print(f"{'─' * W}")
    header2 = f"{'Backend':>12} {'Add(ms)':>10} {'Avg/item':>10} {'Items/s':>10} {'Qry avg':>10} {'Qry P95':>10}"
    print(header2)
    print("-" * len(header2))
    for key, label in [("storage_default_1k", "Default JSON"), ("storage_sqlite_1k", "SQLite")]:
        be = results.get(key, {})
        if be:
            print(
                f"{label:>12} "
                f"{be.get('add_total_ms', 0):>10.1f} "
                f"{be.get('add_avg_ms', 0):>10.3f} "
                f"{be.get('add_throughput', 0):>10.0f} "
                f"{be.get('query_avg_ms', 0):>10.2f} "
                f"{be.get('query_p95_ms', 0):>10.2f}"
            )

    # ── PluginManager ──
    print(f"\n{'─' * W}")
    print("  4. PluginManager")
    print(f"{'─' * W}")
    pm = results.get("plugin_manager", {})
    print(f"  Auto-discover: {pm.get('registered_plugins', 0)} plugins in {pm.get('discover_ms', 0):.0f} ms")
    print(f"  Initialize:    {pm.get('initialized_plugins', 0)}/{pm.get('registered_plugins', 0)} in {pm.get('init_ms', 0):.0f} ms")
    print(f"  Health report: {pm.get('health_report_ms', 0):.1f} ms")
    hs = pm.get("health_summary", {})
    if hs:
        print(f"  Health: total={hs.get('total_plugins', '?')} healthy={hs.get('healthy', '?')}")

    # ── LitePro ──
    print(f"\n{'─' * W}")
    print("  5. SuMemoryLitePro (10 items)")
    print(f"{'─' * W}")
    pro = results.get("lite_pro_10", {})
    if "error" in pro:
        print(f"  ❌ Error: {pro['error']}")
    else:
        print(f"  Add:        {pro.get('add_total_ms', 0):.1f} ms total ({pro.get('add_avg_ms', 0):.1f} ms/item)")
        print(f"  Throughput: {pro.get('add_throughput', 0):.1f} items/s")
        print(f"  Query P50:  {pro.get('query_p50_ms', 0):.2f} ms")
        print(f"  Query P95:  {pro.get('query_p95_ms', 0):.2f} ms")
        print(f"  Count:      {pro.get('count_operation', '?')}")
        print(f"  Protocol:   MemoryProtocol={'✅' if pro.get('memory_protocol') else '❌'}")

    # ── v3.0.0 Features ──
    print(f"\n{'─' * W}")
    print("  6. v3.0.0 New Features Check")
    print(f"{'─' * W}")
    v3 = results.get("v3_features", {})
    for feat, status in v3.items():
        icon = "✅" if status.startswith("OK") else "⚠️"
        print(f"  {icon} {feat}: {status}")

    # ── Regression Check ──
    print(f"\n{'─' * W}")
    print("  7. Regression Check (vs v2.x targets)")
    print(f"{'─' * W}")
    all_pass = True
    for c in checks:
        icon = "✅" if c["pass"] else "❌"
        if not c["pass"]:
            all_pass = False
        print(f"  {icon} {c['metric']}: {c['value']} (target: {c['target']})")

    # ── Summary ──
    print(f"\n{'=' * W}")
    score = sum(1 for c in checks if c["pass"])
    total = len(checks)
    grade = "A+" if score == total else ("A" if score >= total - 1 else ("B" if score >= total - 2 else "C"))
    print(f"  Regression Score: {score}/{total} ({grade})".center(W))
    print(f"  Overall Status: {'✅ ALL PASS' if all_pass else '⚠️  SOME FAILURES'}".center(W))
    print("=" * W)
    print()


# =============================================================================
# Main
# =============================================================================

def main():
    results: Dict[str, Any] = {
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat(),
        "python": sys.version,
    }

    with tempfile.TemporaryDirectory() as tmp:
        print("\n🔬 Running su-memory v3.0.0 SOTA Benchmark...\n")

        # ── 1. SuMemoryLite at scale ──
        for lbl, n in SCALES:
            print(f"  [{lbl}] SuMemoryLite add/query/count...", end=" ", flush=True)
            lite = SuMemoryLite(max_memories=n * 2, storage_path=tmp, enable_persistence=False)
            results[f"insertion_{lbl.lower()}"] = bench_lite_add(lite, n)
            results[f"query_{lbl.lower()}"] = bench_lite_query(lite, QUERY_COUNT)
            results[f"count_{lbl.lower()}"] = bench_lite_count(lite)
            results[f"memory_{lbl.lower()}"] = bench_lite_memory(lite)
            print("done")

        # ── 2. Storage backend comparison ──
        print(f"  [Storage] Backend comparison...", end=" ", flush=True)
        results["storage_default_1k"] = bench_storage_backend("Default", "default", 1000, tmp)
        results["storage_sqlite_1k"] = bench_storage_backend("SQLite", "sqlite", 1000, tmp)
        print("done")

        # ── 3. PluginManager ──
        print(f"  [Plugin] Manager benchmark...", end=" ", flush=True)
        results["plugin_manager"] = bench_plugin_manager()
        print("done")

        # ── 4. SuMemoryLitePro (fast check only) ──
        print(f"  [Pro] SuMemoryLitePro (10 items)...", end=" ", flush=True)
        results["lite_pro_10"] = bench_lite_pro(10, tmp)
        print("done")

        # ── 5. v3 features ──
        print(f"  [v3.0] New features...", end=" ", flush=True)
        results["v3_features"] = bench_v3_features()
        print("done")

    # ── Regression check ──
    checks = check_regression(results)
    results["regression_checks"] = [
        {"metric": c["metric"], "value": c["value"], "target": c["target"], "pass": c["pass"]}
        for c in checks
    ]
    score = sum(1 for c in checks if c["pass"])
    results["regression_score"] = f"{score}/{len(checks)}"

    # ── Output ──
    print_report(results, checks)

    # Save JSON
    os.makedirs(os.path.dirname(BENCHMARK_OUTPUT), exist_ok=True)
    with open(BENCHMARK_OUTPUT, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"📄 Results saved to: {BENCHMARK_OUTPUT}")

    return results


if __name__ == "__main__":
    main()
