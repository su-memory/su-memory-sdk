"""
bench_concurrency.py — 并发读写吞吐基准

门禁: 4线程线性扩展 > 2.5x
"""

import sys
import os
import time
import tempfile
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from su_memory import SuMemoryLitePro


def bench_concurrent_writes(threads: int, items_per_thread: int) -> float:
    """测试并发写入吞吐"""
    d = tempfile.mkdtemp()
    try:
        client = SuMemoryLitePro(
            storage_path=d,
            enable_vector=False,
            enable_graph=False,
            enable_temporal=False,
            enable_session=False,
        )

        def writer(thread_id):
            for i in range(items_per_thread):
                client.add(f"thread {thread_id} item {i:05d}")

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(writer, i) for i in range(threads)]
            for f in as_completed(futures):
                f.result()
        elapsed = time.perf_counter() - start

        total = threads * items_per_thread
        throughput = total / elapsed
        count = len(client)
        print(f"bench_concurrent_write ({threads}t): {throughput:.0f} ops/s | {count} items")
        return throughput
    finally:
        shutil.rmtree(d, ignore_errors=True)


def bench_concurrent_read_write(threads: int, items: int) -> float:
    """测试读写混合吞吐"""
    d = tempfile.mkdtemp()
    try:
        client = SuMemoryLitePro(
            storage_path=d,
            enable_vector=False,
            enable_graph=False,
            enable_temporal=False,
            enable_session=False,
        )

        # 预填数据
        for i in range(200):
            client.add(f"preload item {i:05d}")

        def mixed_worker(thread_id):
            for i in range(items):
                if i % 2 == 0:
                    client.add(f"mixed write thread {thread_id} item {i:05d}")
                else:
                    client.query(f"item {i % 200:05d}", top_k=5)

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(mixed_worker, i) for i in range(threads)]
            for f in as_completed(futures):
                f.result()
        elapsed = time.perf_counter() - start

        total = threads * items
        throughput = total / elapsed
        print(f"bench_concurrent_mixed ({threads}t): {throughput:.0f} ops/s")
        return throughput
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    # 单线程基准
    t1 = bench_concurrent_writes(1, 200)
    print(f"  Single thread baseline: {t1:.0f} ops/s")

    # 4线程
    t4 = bench_concurrent_writes(4, 50)
    scale = t4 / t1 if t1 > 0 else 0
    status = "PASS" if scale > 2.5 else "FAIL"
    print(f"\n  Scale factor (4t/1t): {scale:.1f}x | Gate: >2.5x | [{status}]")

    # 读写混合
    bench_concurrent_read_write(2, 100)
    bench_concurrent_read_write(4, 50)

    sys.exit(0 if scale > 2.5 else 1)
