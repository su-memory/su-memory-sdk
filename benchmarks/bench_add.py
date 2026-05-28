"""
bench_add.py — 单条记忆写入吞吐基准

门禁: ≥ 80 条/s
"""

import sys
import os
import time
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from su_memory import SuMemoryLitePro


def bench_add_single():
    """测试单条写入吞吐"""
    d = tempfile.mkdtemp()
    try:
        client = SuMemoryLitePro(
            storage_path=d,
            enable_vector=False,
            enable_graph=False,
            enable_temporal=False,
            enable_session=False,
        )

        COUNT = 200
        start = time.perf_counter()
        for i in range(COUNT):
            client.add(f"benchmark memory entry number {i:05d}")
        elapsed = time.perf_counter() - start

        throughput = COUNT / elapsed
        latency_ms = (elapsed / COUNT) * 1000
        status = "PASS" if throughput >= 80 else "FAIL"

        print(f"bench_add_single: {throughput:.1f} ops/s | avg {latency_ms:.1f}ms | [{status}]")
        return throughput
    finally:
        shutil.rmtree(d, ignore_errors=True)


def bench_add_with_vector():
    """测试带向量的单条写入"""
    d = tempfile.mkdtemp()
    try:
        client = SuMemoryLitePro(
            storage_path=d,
            enable_vector=True,
            enable_graph=False,
            enable_temporal=False,
            enable_session=False,
        )

        COUNT = 50
        start = time.perf_counter()
        for i in range(COUNT):
            client.add(f"benchmark memory entry number {i:05d}")
        elapsed = time.perf_counter() - start

        throughput = COUNT / elapsed
        latency_ms = (elapsed / COUNT) * 1000
        print(f"bench_add_vector:  {throughput:.1f} ops/s | avg {latency_ms:.1f}ms (vector)")
        return throughput
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    t_single = bench_add_single()
    t_vector = bench_add_with_vector()
    sys.exit(0 if t_single >= 80 else 1)
