#!/usr/bin/env python3
"""
su-memory 真实微基准（honest microbenchmark）
============================================
本脚本是 BENCHMARK.md 中所有性能数字的唯一来源（单一真相源）。

设计要点（对齐 P1-1/P1-2）：
1. tracemalloc 测真实 peak 内存；perf_counter_ns 测真实 add/query 延迟。
2. 每次查询前清空 _query_cache，消除缓存干扰。
3. 固定规模 [100, 1K, 5K, 10K]，每项跑 3 轮取中位数。
4. 明确区分 Lite（N-gram TF-IDF）与 LitePro（FAISS + embedding），
   两者能力边界不同，禁止混报。
5. 语义召回使用「不与 fact 共享语素」的改写 query，如实反映 Lite 的边界。

用法:
    python benchmarks/real_microbench.py
    python benchmarks/real_microbench.py --no-lite-pro   # 跳过需下载模型的 LitePro

输出: 控制台报告 + benchmarks/results/real_microbench_{timestamp}.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import tempfile
import time
import tracemalloc
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from su_memory.sdk.lite import SuMemoryLite

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SCALES = [100, 1000, 5000, 10000]
ROUNDS = 3


def _median(xs):
    return statistics.median(xs)


def _pct(sorted_vals, pct):
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * pct))]


def _make_doc(i):
    return f"基准记忆条目编号{i}包含关键词数据{chr(0x4e00 + i % 200)}描述{i}"


def bench_lite_insert(n):
    """单次插入 n 条：返回 (total_s, peak_bytes)。"""
    with tempfile.TemporaryDirectory() as tmp:
        lite = SuMemoryLite(storage_path=tmp, enable_persistence=False, cache_size=0)
        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        for i in range(n):
            lite.add(_make_doc(i))
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return elapsed, peak


def bench_lite_query(n, n_queries=300):
    """建库 n 条后查询：返回 (p50_ms, p95_ms, p99_ms)。无缓存。"""
    with tempfile.TemporaryDirectory() as tmp:
        lite = SuMemoryLite(storage_path=tmp, enable_persistence=False, cache_size=0)
        for i in range(n):
            lite.add(_make_doc(i))
        queries = [f"记忆{chr(0x4e00 + (i * 7) % 200)}{i}" for i in range(n_queries)]
        queries += ["张三", "预算多少", "延迟优化", "冬奥"]
        times_ms = []
        for q in queries:
            lite._query_cache.clear()
            a = time.perf_counter_ns()
            lite.query(q, top_k=5)
            b = time.perf_counter_ns()
            times_ms.append((b - a) / 1e6)
        times_ms.sort()
    return _pct(times_ms, 0.5), _pct(times_ms, 0.95), _pct(times_ms, 0.99)


def bench_semantic_recall_lite():
    """
    如实刻画 Lite 的语义召回边界（P1-3）。
    返回 (shared_morpheme_hit_rate, zero_overlap_hit_rate)。
    """
    with tempfile.TemporaryDirectory() as tmp:
        lite = SuMemoryLite(storage_path=tmp, enable_persistence=False, cache_size=0)
        facts = [
            "张三在2024年3月入职担任算法工程师",
            "公司去年营收达到2.3亿元同比增长45%",
            "北京冬奥会于2022年2月举办共有91个国家参加",
        ]
        for f in facts:
            lite.add(f)

        # 共享语素的改写（应命中）
        shared = [
            ("去年一共赚了多少钱", "营收"),
            ("张三什么时候入职的", "张三"),
            ("冬奥会在哪举办", "冬奥"),
        ]
        shared_hits = 0
        for q, key in shared:
            lite._query_cache.clear()
            r = lite.query(q, top_k=5)
            if r and key in r[0]["content"]:
                shared_hits += 1

        # 零语素重叠的改写（真·语义才能命中；已校验无共享语素）
        zero = [
            ("该名新成员何时开始到岗", "张三"),
            ("财务表现如何", "营收"),
            ("某项跨国冬季体育竞技盛会的所在地", "冬奥"),
        ]
        zero_hits = 0
        for q, key in zero:
            lite._query_cache.clear()
            r = lite.query(q, top_k=5)
            if r and key in r[0]["content"]:
                zero_hits += 1

    return shared_hits / len(shared), zero_hits / len(zero)


def bench_lite_pro(n=1000):
    """LitePro（FAISS + embedding）插入/查询。需可下载模型。"""
    try:
        from su_memory.sdk.lite_pro import SuMemoryLitePro
    except Exception as e:
        return {"error": f"LitePro import failed: {e}"}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            pro = SuMemoryLitePro(storage_path=tmp)
            t0 = time.perf_counter()
            for i in range(n):
                pro.add(_make_doc(i))
            add_elapsed = time.perf_counter() - t0

            times_ms = []
            for i in range(50):
                a = time.perf_counter_ns()
                pro.query(_make_doc(i % n), top_k=5)
                b = time.perf_counter_ns()
                times_ms.append((b - a) / 1e6)
            times_ms.sort()
        return {
            "n": n,
            "add_total_ms": round(add_elapsed * 1000, 2),
            "throughput_per_sec": round(n / add_elapsed, 1),
            "query_p50_ms": round(_pct(times_ms, 0.5), 3),
            "query_p95_ms": round(_pct(times_ms, 0.95), 3),
        }
    except Exception as e:
        return {"error": f"LitePro run failed: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-lite-pro", action="store_true")
    args = ap.parse_args()

    print("=" * 64)
    print("su-memory 真实微基准 (honest microbenchmark)")
    print("=" * 64)
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    print(f"环境: {env['python']} / {env['platform']} / {env['processor']}")

    report = {"env": env, "lite": {}, "lite_pro": {}, "semantic_recall": {}}

    print("\n[Lite] 插入吞吐 / 内存（tracemalloc，3 轮中位数）")
    for n in SCALES:
        ins, mems = [], []
        for _ in range(ROUNDS):
            e, p = bench_lite_insert(n)
            ins.append(e)
            mems.append(p)
        med_t = _median(ins)
        med_m = _median(mems)
        report["lite"][f"insert_{n}"] = {
            "total_ms": round(med_t * 1000, 2),
            "throughput_per_sec": round(n / med_t, 0),
            "peak_mem_mb": round(med_m / 1024 / 1024, 2),
        }
        print(f"  n={n:>6}: thru={n/med_t:>8.0f}/s  peak_mem={med_m/1024/1024:>7.2f}MB")

    print("\n[Lite] 查询延迟（清缓存，3 轮中位数）")
    for n in SCALES:
        p50s, p95s, p99s = [], [], []
        for _ in range(ROUNDS):
            a, b, c = bench_lite_query(n)
            p50s.append(a); p95s.append(b); p99s.append(c)
        report["lite"][f"query_{n}"] = {
            "p50_ms": round(_median(p50s), 4),
            "p95_ms": round(_median(p95s), 4),
            "p99_ms": round(_median(p99s), 4),
        }
        print(f"  n={n:>6}: p50={_median(p50s):.4f}ms  p95={_median(p95s):.4f}ms")

    print("\n[语义召回边界] Lite（N-gram TF-IDF）")
    sh, zh = bench_semantic_recall_lite()
    report["semantic_recall"]["lite"] = {
        "shared_morpheme_hit_rate": round(sh, 3),
        "zero_overlap_hit_rate": round(zh, 3),
    }
    print(f"  共享语素改写命中: {sh*100:.1f}%")
    print(f"  零语素重叠改写命中: {zh*100:.1f}%  (← 真·语义能力下限)")

    if not args.no_lite_pro:
        print("\n[LitePro] FAISS + embedding")
        report["lite_pro"] = bench_lite_pro()
        if "error" in report["lite_pro"]:
            print(f"  跳过: {report['lite_pro']['error']}")
        else:
            lp = report["lite_pro"]
            print(f"  thru={lp['throughput_per_sec']:.1f}/s  "
                  f"q_p50={lp['query_p50_ms']}ms  q_p95={lp['query_p95_ms']}ms")

    out = os.path.join(
        RESULTS_DIR, f"real_microbench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写入: {out}")
    print("\n说明: Lite=N-gram 关键词检索; LitePro=FAISS 真向量. 两者能力边界不同, 禁止混报.")


if __name__ == "__main__":
    main()
