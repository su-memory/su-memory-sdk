#!/usr/bin/env python3
"""v3.5.5 FAISS 查询性能验证脚本"""
import time, hashlib, struct, tempfile, shutil, gc, sys, os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
BENCH_DIR = os.path.join(PROJECT_ROOT, "benchmarks")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

from su_memory import SuMemory
from run_benchmark import generate_test_texts, generate_query_texts, percentile, get_memory_usage

class FastHashEmbedder:
    def __init__(self, dims=128):
        self.dims = dims
    def encode(self, text):
        vec = [0.0] * self.dims
        for i, ch in enumerate(text):
            h = hashlib.sha256(f"{i}:{ch}".encode()).digest()[:2]
            idx = struct.unpack("<H", h)[0] % self.dims
            vec[idx] += 1.0
        norm = (sum(v * v for v in vec)) ** 0.5
        return [v / norm for v in vec] if norm > 0 else vec

# --- v3.5.4 baseline (from P2 results) ---
BASELINE_QUERY_P50 = {100: 99.8, 1000: 102.2, 10000: 109.3, 50000: 128.1, 100000: 135.3}
BASELINE_QUERY_P95 = {100: 104.4, 1000: 105.1, 10000: 121.1, 50000: 165.0, 100000: 213.0}

print("=" * 70)
print("  v3.5.5 FAISS 查询扩展性验证 (Hash Embedder, dim=128)")
print("=" * 70)

sizes = [100, 1000, 10000, 50000, 100000]
query_samples = 200
results = {}

for size in sizes:
    label = f"{size // 1000}K" if size >= 1000 else str(size)
    tmpdir = tempfile.mkdtemp(prefix=f"su_faiss_s{size}_")
    try:
        t0 = time.perf_counter()
        client = SuMemory(mode="local", persist_dir=tmpdir, embedder=FastHashEmbedder(128))
        texts = generate_test_texts(size)

        t_add = time.perf_counter()
        for t in texts:
            client.add(t)
        add_elapsed = time.perf_counter() - t_add

        queries = generate_query_texts(query_samples)
        query_latencies = []
        for q in queries:
            tq = time.perf_counter()
            client.query(q, top_k=5)
            query_latencies.append((time.perf_counter() - tq) * 1000)

        qp50 = percentile(query_latencies, 50)
        qp95 = percentile(query_latencies, 95)
        qp99 = percentile(query_latencies, 99)

        stats = client.get_stats()
        mem = get_memory_usage()
        faiss_size = stats.get("faiss_index_size", 0)

        results[size] = {"qp50": qp50, "qp95": qp95, "qp99": qp99}

        print(f"  [{label:>5s}] size={size:>6d}  "
              f"add={add_elapsed:.2f}s  "
              f"query P50={qp50:.4f}ms P95={qp95:.4f}ms P99={qp99:.4f}ms  "
              f"FAISS={faiss_size}  RSS={mem.get('rss_mb', 0):.0f}MB")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        if size >= 50000:
            gc.collect()

print()
print("  查询延迟对比 (v3.5.4 → v3.5.5):")
print(f"  {'规模':<10s} {'v3.5.4 P50':>12s} {'v3.5.5 P50':>12s} {'提升':>10s} {'扩展比':>12s}")
print(f"  {'─' * 10} {'─' * 12} {'─' * 12} {'─' * 10} {'─' * 12}")

base_q100 = BASELINE_QUERY_P50.get(100, 100)
for size in sizes:
    old_p50 = BASELINE_QUERY_P50.get(size, 100)
    new_p50 = results[size]["qp50"]
    speedup = old_p50 / new_p50 if new_p50 > 0 else 0
    scale_ratio = new_p50 / results[100]["qp50"] if results[100]["qp50"] > 0 else 0
    print(f"  {str(size):<10s} {old_p50:>9.1f}ms {new_p50:>9.4f}ms {speedup:>8.0f}x {scale_ratio:>10.1f}x")

print()
print(f"  数据量 1000x 增长 → 查询延迟 {results[100000]['qp50'] / results[100]['qp50']:.1f}x 增长")
print(f"  (v3.5.4: 1.36x, 目标: <1000x 亚线性)")

# Check P1: stress query P95 target
print()
print("  P1/P2 达标验证:")
print(f"  100K 查询 P50={results[100000]['qp50']:.1f}ms  {'✅' if results[100000]['qp50'] < 300 else '❌'} (目标 <300ms)")
print(f"  100K 查询 P95={results[100000]['qp95']:.1f}ms  {'✅' if results[100000]['qp95'] < 500 else '❌'} (目标 <500ms)")
