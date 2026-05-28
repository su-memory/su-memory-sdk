"""
bench_faiss.py — FAISS 构建/搜索/持久化基准

门禁: search ≤ 10ms
"""

import sys
import os
import time
import tempfile
import shutil
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from su_memory import SuMemoryLitePro


def bench_faiss_build_search():
    """测试 FAISS 构建和搜索延迟"""
    d = tempfile.mkdtemp()
    try:
        client = SuMemoryLitePro(
            storage_path=d,
            enable_vector=True,
            enable_graph=False,
            enable_temporal=False,
            enable_session=False,
        )

        # 构建阶段
        NUM_ITEMS = 200
        build_start = time.perf_counter()
        for i in range(NUM_ITEMS):
            client.add(f"faiss benchmark item {i:05d} with enough text for embedding")
        build_time = (time.perf_counter() - build_start) * 1000
        print(f"bench_faiss_build:  {build_time:.0f}ms ({NUM_ITEMS} items, {NUM_ITEMS/(build_time/1000):.0f} ops/s)")

        # 搜索阶段
        SAMPLES = 100
        latencies = []
        for i in range(SAMPLES):
            query = f"search for item {i % NUM_ITEMS}"
            start = time.perf_counter()
            results = client.query(query, top_k=10)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        latencies.sort()
        avg = sum(latencies) / len(latencies)
        p50 = statistics.median(latencies)
        p99 = latencies[int(len(latencies) * 0.99)]

        status = "PASS" if avg <= 10 else "FAIL"
        print(f"bench_faiss_search: avg={avg:.1f}ms P50={p50:.1f}ms P99={p99:.1f}ms [{status}]")
        print(f"  (N={NUM_ITEMS}, samples={SAMPLES})")

        return avg
    finally:
        shutil.rmtree(d, ignore_errors=True)


def bench_faiss_persist():
    """测试 FAISS 持久化和恢复"""
    d = tempfile.mkdtemp()
    try:
        # 创建并持久化
        client = SuMemoryLitePro(
            storage_path=d,
            enable_vector=True,
            enable_graph=False,
            enable_temporal=False,
            enable_session=False,
        )

        for i in range(50):
            client.add(f"persist item {i:05d} with enough text for embedding")

        persist_start = time.perf_counter()
        # 触发持久化 (通过查询触发内部保存)
        client.query("persist", top_k=5)
        persist_time = (time.perf_counter() - persist_start) * 1000

        # 恢复
        recover_start = time.perf_counter()
        client2 = SuMemoryLitePro(
            storage_path=d,
            enable_vector=True,
            enable_graph=False,
            enable_temporal=False,
            enable_session=False,
        )
        recover_time = (time.perf_counter() - recover_start) * 1000

        count2 = len(client2)
        print(f"bench_faiss_persist: save {persist_time:.0f}ms | load {recover_time:.0f}ms | recovered {count2} items")
        return recover_time
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    avg_search = bench_faiss_build_search()
    bench_faiss_persist()
    sys.exit(0 if avg_search <= 10 else 1)
