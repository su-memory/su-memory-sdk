"""
bench_multihop.py — 多跳推理延迟基准

门禁: 3-hop ≤ 200ms
"""

import sys
import os
import time
import tempfile
import shutil
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from su_memory import SuMemoryLitePro


def bench_multihop(hops: int, label: str):
    """测试多跳推理延迟"""
    d = tempfile.mkdtemp()
    try:
        client = SuMemoryLitePro(
            storage_path=d,
            enable_vector=False,
            enable_graph=True,
            enable_temporal=False,
            enable_session=False,
        )

        # 构建链式关系: node0 → node1 → node2 → ... → node{N}
        for i in range(hops + 2):
            client.add(f"hop node {i}: information about step {i}")

        # 建立因果链
        for i in range(hops + 1):
            parent = client.add(f"parent node {i}")
            child = client.add(f"child node {i}")
            client.add_edge(str(parent), str(child), "causes")

        SAMPLES = 30
        latencies = []
        for _ in range(SAMPLES):
            start = time.perf_counter()
            results = client.query_multihop(
                "hop information", max_hops=hops, top_k=5
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        avg = sum(latencies) / len(latencies)
        print(f"bench_multihop_{label}: avg {avg:.1f}ms (max_hops={hops}, samples={SAMPLES})")
        return avg
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    results = {}
    for hops, label in [(1, "1hop"), (2, "2hop"), (3, "3hop")]:
        results[label] = bench_multihop(hops, label)

    t_3hop = results.get("3hop", 999)
    status = "PASS" if t_3hop <= 200 else "FAIL"
    print(f"\nGate (3-hop): ≤200ms | Actual: {t_3hop:.1f}ms | [{status}]")
    sys.exit(0 if t_3hop <= 200 else 1)
