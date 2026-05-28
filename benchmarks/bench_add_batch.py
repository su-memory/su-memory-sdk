"""
bench_add_batch.py — 批量写入吞吐基准

门禁: ≥ 500 条/s (100条批次)
"""

import sys
import os
import time
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from su_memory import SuMemoryLitePro


def bench_batch(size: int, label: str):
    """批量写入基准"""
    d = tempfile.mkdtemp()
    try:
        client = SuMemoryLitePro(
            storage_path=d,
            enable_vector=False,
            enable_graph=False,
            enable_temporal=False,
            enable_session=False,
        )

        items = [f"batch item {i:06d}" for i in range(size)]
        start = time.perf_counter()
        results = client.add_batch(items)
        elapsed = time.perf_counter() - start

        throughput = size / elapsed
        latency_ms = (elapsed / size) * 1000
        print(f"bench_batch_{label}:  {throughput:.1f} ops/s ({size} items, {elapsed*1000:.0f}ms, avg {latency_ms:.1f}ms)")
        return throughput
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    results = {}
    for size, label in [(100, "100"), (500, "500"), (1000, "1000")]:
        results[label] = bench_batch(size, label)

    t_100 = results.get("100", 0)
    status = "PASS" if t_100 >= 500 else "FAIL"
    print(f"\nGate (100-batch): ≥500 ops/s | Actual: {t_100:.1f} | [{status}]")
    sys.exit(0 if t_100 >= 500 else 1)
