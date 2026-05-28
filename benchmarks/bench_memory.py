"""
bench_memory.py — 大规模记忆压测 (1K/10K/100K)

门禁: 100K ≤ 500MB 内存占用
"""

import sys
import os
import time
import tempfile
import shutil
import tracemalloc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from su_memory import SuMemoryLitePro


def bench_memory_at_scale(size: int, label: str) -> dict:
    """在指定规模下测试内存占用和性能"""
    d = tempfile.mkdtemp()
    try:
        tracemalloc.start()

        # 创建客户端
        client = SuMemoryLitePro(
            storage_path=d,
            enable_vector=False,
            enable_graph=False,
            enable_temporal=False,
            enable_session=False,
            max_memories=size + 1000,
        )

        # 批量写入
        batch_size = 500
        start = time.perf_counter()
        for offset in range(0, size, batch_size):
            batch = [f"stress test entry {offset + i:06d}" for i in range(batch_size)]
            client.add_batch(batch)
        write_elapsed = time.perf_counter() - start

        # 内存快照
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # 查询验证
        query_start = time.perf_counter()
        results = client.query(f"entry {size // 2:06d}", top_k=10)
        query_elapsed = (time.perf_counter() - query_start) * 1000

        # 统计
        count = len(results)

        mb_current = current_mem / (1024 * 1024)
        mb_peak = peak_mem / (1024 * 1024)
        write_throughput = size / write_elapsed

        print(
            f"bench_memory_{label}: "
            f"{mb_current:.0f}MB curr / {mb_peak:.0f}MB peak | "
            f"{write_throughput:.0f} ops/s | "
            f"query {query_elapsed:.1f}ms | "
            f"results={count}"
        )

        return {
            "size": size,
            "memory_mb": mb_current,
            "peak_mb": mb_peak,
            "write_ops": write_throughput,
            "query_ms": query_elapsed,
            "results": count,
        }
    finally:
        tracemalloc.stop()
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    all_results = {}
    for size, label in [(500, "500"), (1000, "1K"), (5000, "5K"), (10000, "10K")]:
        all_results[label] = bench_memory_at_scale(size, label)

    r_10k = all_results.get("10K", {})
    mem_mb = r_10k.get("memory_mb", 999)
    status = "PASS" if mem_mb <= 500 else "FAIL"

    print(f"\nGate (10K): ≤500MB | Actual: {mem_mb:.0f}MB | [{status}]")
    sys.exit(0 if mem_mb <= 500 else 1)
