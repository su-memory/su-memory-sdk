"""
bench_query.py — 查询延迟基准 (P50/P95/P99)

门禁: P99 ≤ 50ms
"""

import sys
import os
import time
import tempfile
import shutil
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from su_memory import SuMemoryLitePro


def bench_query_latency():
    """测试查询延迟分布"""
    d = tempfile.mkdtemp()
    try:
        client = SuMemoryLitePro(
            storage_path=d,
            enable_vector=False,
            enable_graph=False,
            enable_temporal=False,
            enable_session=False,
        )

        # 预热：填充数据
        NUM_MEMORIES = 500
        for i in range(NUM_MEMORIES):
            client.add(f"query benchmark entry for latency test number {i:05d}")

        # 查询延迟采样
        SAMPLES = 200
        latencies = []
        for i in range(SAMPLES):
            query = f"entry {i % NUM_MEMORIES}"
            start = time.perf_counter()
            results = client.query(query, top_k=10)
            elapsed_ms = (time.perf_counter() - start) * 1000

            # 验证结果正确性
            assert len(results) > 0, f"Query returned empty: {query}"

            latencies.append(elapsed_ms)

        latencies.sort()
        p50 = statistics.median(latencies)
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        avg = sum(latencies) / len(latencies)
        mx = max(latencies)

        status = "PASS" if p99 <= 50 else "FAIL"
        print(f"bench_query: P50={p50:.1f}ms P95={p95:.1f}ms P99={p99:.1f}ms avg={avg:.1f}ms max={mx:.1f}ms [{status}]")
        print(f"  (N={NUM_MEMORIES}, samples={SAMPLES})")
        return p99
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    p99 = bench_query_latency()
    sys.exit(0 if p99 <= 50 else 1)
