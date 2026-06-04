#!/usr/bin/env python3
"""
su-memory SDK v3.5.4 P1 大规模性能基准测试

针对 10K 数据规模的核心测试：
1. 延迟测试 — P50/P95/P99 (add + query)
2. 吞吐量测试 — QPS / 错误率 (高并发)
3. 资源占用测试 — 内存/RSS
4. 扩展性测试 — 不同数据规模 (100/1K/10K)

使用方式:
    python benchmarks/run_benchmark_p1.py
"""

import json
import os
import sys
import time
import random
import tempfile
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from su_memory import SuMemory

# ============================================================
# 测试数据
# ============================================================

SAMPLE_TEXTS = [
    "项目ROI增长25%，投资回报超出预期",
    "团队协作效率持续提升，沟通成本下降",
    "市场风险增加，需要关注不确定性",
    "知识网络互联互通，信息传递高效",
    "技术架构升级以应对更高并发需求",
    "合作协议达成，双方满意共赢",
    "产品渗透率扩散至二三线城市",
    "项目进展顺利，关键节点完成",
    "客户满意度提升，正向反馈增加",
    "运营流程标准化，基础体系稳固",
    "数据安全策略更新，风险控制加强",
    "新功能开发启动，迭代周期缩短",
    "财务报表显示营收稳步增长",
    "用户体验优化方案通过评审",
    "供应链管理效率显著提高",
    "品牌影响力扩展到海外市场",
    "人才培养体系逐步完善",
    "成本控制措施有效执行",
    "创新驱动发展战略稳步推进",
    "数字化转型取得阶段性成果",
    "人工智能应用场景不断拓展",
    "云计算基础设施稳定运行",
    "微服务架构部署完成上线",
    "容器化方案降低运维成本",
    "自动化测试覆盖率提升至85%",
    "持续集成流水线优化完成",
    "日志监控体系全面升级",
    "网络安全防护等级提升",
    "数据中台建设初见成效",
    "业务智能分析平台上线",
]


def generate_test_texts(count: int) -> List[str]:
    texts = []
    for i in range(count):
        base = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
        texts.append(f"编号{i}: {base}，指标{random.randint(1, 999)}")
    return texts


def generate_query_texts(count: int) -> List[str]:
    queries = [
        "投资回报", "团队协作", "市场风险", "知识网络", "技术架构",
        "合作协议", "产品渗透", "项目进展", "用户满意", "运营流程",
        "数据安全", "新功能", "财务报表", "用户体验", "供应链",
        "品牌影响", "人才培养", "成本控制", "创新驱动", "数字化",
        "AI应用", "云计算", "微服务", "容器化", "自动化",
    ]
    return [queries[i % len(queries)] for i in range(count)]


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (p / 100.0) * (len(sorted_data) - 1)
    lower = int(idx)
    upper = min(lower + 1, len(sorted_data) - 1)
    frac = idx - lower
    return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac


def get_memory_usage() -> Dict[str, float]:
    if not PSUTIL_AVAILABLE:
        return {"rss_mb": 0, "vms_mb": 0, "available": False}
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    return {
        "rss_mb": round(mem.rss / 1024 / 1024, 2),
        "vms_mb": round(mem.vms / 1024 / 1024, 2),
        "available": True,
    }


# ============================================================
# P1 延迟测试
# ============================================================

def p1_latency_add(samples: int, data_count: int) -> Dict:
    print(f"\n  [P1延迟] SDK写入 10K预填 - {samples}次")
    tmpdir = tempfile.mkdtemp(prefix="su_p1_")
    try:
        client = SuMemory(mode="local", persist_dir=tmpdir)
        texts = generate_test_texts(data_count)
        for t in texts:
            client.add(t)

        test_texts = generate_test_texts(samples)
        latencies = []
        for text in test_texts:
            start = time.perf_counter()
            client.add(text)
            latencies.append(time.perf_counter() - start)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return {
        "operation": "SDK add (10K prefill)",
        "samples": samples,
        "prefill_count": data_count,
        "p50_ms": round(percentile(latencies, 50) * 1000, 3),
        "p95_ms": round(percentile(latencies, 95) * 1000, 3),
        "p99_ms": round(percentile(latencies, 99) * 1000, 3),
        "min_ms": round(min(latencies) * 1000, 3),
        "max_ms": round(max(latencies) * 1000, 3),
        "mean_ms": round(float(np.mean(latencies)) * 1000, 3),
        "targets": {"p50": 150, "p95": 300, "p99": 500},
    }


def p1_latency_query(samples: int, data_count: int) -> Dict:
    print(f"\n  [P1延迟] SDK检索 10K预填 - {samples}次")
    tmpdir = tempfile.mkdtemp(prefix="su_p1_")
    try:
        client = SuMemory(mode="local", persist_dir=tmpdir)
        texts = generate_test_texts(data_count)
        for t in texts:
            client.add(t)

        queries = generate_query_texts(samples)
        latencies = []
        for q in queries:
            start = time.perf_counter()
            client.query(q, top_k=5)
            latencies.append(time.perf_counter() - start)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return {
        "operation": "SDK query (10K prefill)",
        "samples": samples,
        "prefill_count": data_count,
        "p50_ms": round(percentile(latencies, 50) * 1000, 3),
        "p95_ms": round(percentile(latencies, 95) * 1000, 3),
        "p99_ms": round(percentile(latencies, 99) * 1000, 3),
        "min_ms": round(min(latencies) * 1000, 3),
        "max_ms": round(max(latencies) * 1000, 3),
        "mean_ms": round(float(np.mean(latencies)) * 1000, 3),
        "targets": {"p50": 100, "p95": 200, "p99": 400},
    }


# ============================================================
# P1 吞吐量测试 (高并发)
# ============================================================

def p1_throughput_add(workers: int, duration: float) -> Dict:
    print(f"\n  [P1吞吐] SDK写入 {workers}并发 {duration}s")
    texts = generate_test_texts(500)
    errors = 0
    total_ops = 0
    lock = threading.Lock()

    def worker(wid: int):
        nonlocal errors, total_ops
        tmpdir = tempfile.mkdtemp(prefix=f"su_p1_add_{wid}_")
        try:
            client = SuMemory(mode="local", persist_dir=tmpdir)
            local_ops = 0
            local_errors = 0
            deadline = time.time() + duration
            while time.time() < deadline:
                try:
                    client.add(texts[local_ops % len(texts)])
                    local_ops += 1
                except Exception:
                    local_errors += 1
                    local_ops += 1
            with lock:
                total_ops += local_ops
                errors += local_errors
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, i) for i in range(workers)]
        for f in as_completed(futures):
            f.result()
    elapsed = time.perf_counter() - start

    return {
        "operation": f"SDK add ({workers}w)",
        "workers": workers,
        "duration_sec": round(elapsed, 2),
        "total_ops": total_ops,
        "qps": round(total_ops / elapsed, 1),
        "errors": errors,
        "error_rate": round(errors / max(total_ops, 1), 4),
        "target_qps": 50,
        "target_error_rate": 0.001,
    }


def p1_throughput_query(workers: int, duration: float) -> Dict:
    print(f"\n  [P1吞吐] SDK检索 {workers}并发 {duration}s")
    queries = generate_query_texts(200)
    errors = 0
    total_ops = 0
    lock = threading.Lock()

    def worker(wid: int):
        nonlocal errors, total_ops
        tmpdir = tempfile.mkdtemp(prefix=f"su_p1_q_{wid}_")
        try:
            client = SuMemory(mode="local", persist_dir=tmpdir)
            fill_texts = generate_test_texts(100)
            for t in fill_texts:
                client.add(t)
            local_ops = 0
            local_errors = 0
            deadline = time.time() + duration
            while time.time() < deadline:
                try:
                    client.query(queries[local_ops % len(queries)], top_k=5)
                    local_ops += 1
                except Exception:
                    local_errors += 1
                    local_ops += 1
            with lock:
                total_ops += local_ops
                errors += local_errors
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, i) for i in range(workers)]
        for f in as_completed(futures):
            f.result()
    elapsed = time.perf_counter() - start

    return {
        "operation": f"SDK query ({workers}w)",
        "workers": workers,
        "duration_sec": round(elapsed, 2),
        "total_ops": total_ops,
        "qps": round(total_ops / elapsed, 1),
        "errors": errors,
        "error_rate": round(errors / max(total_ops, 1), 4),
        "target_qps": 50,
        "target_error_rate": 0.001,
    }


# ============================================================
# P1 资源稳定性测试 (长时间运行)
# ============================================================

def p1_resource_stability(data_count: int, duration_sec: float) -> Dict:
    print(f"\n  [P1资源] 长时间稳定性 {data_count}条, {duration_sec}s")
    tmpdir = tempfile.mkdtemp(prefix="su_p1_stab_")
    try:
        client = SuMemory(mode="local", persist_dir=tmpdir)
        texts = generate_test_texts(data_count)
        queries = generate_query_texts(30)

        # 初始填充
        for t in texts:
            client.add(t)

        mem_samples = []
        deadline = time.time() + duration_sec
        ops = 0
        while time.time() < deadline:
            q = queries[ops % len(queries)]
            client.query(q, top_k=5)
            ops += 1
            if ops % 100 == 0:
                mem_samples.append(get_memory_usage())

        mem_after = get_memory_usage()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    rss_vals = [s.get("rss_mb", 0) for s in mem_samples] if mem_samples else [mem_after.get("rss_mb", 0)]

    return {
        "operation": "long-run stability",
        "data_count": data_count,
        "duration_sec": duration_sec,
        "total_ops": ops,
        "qps": round(ops / duration_sec, 1),
        "rss_start_mb": rss_vals[0] if rss_vals else 0,
        "rss_end_mb": mem_after.get("rss_mb", 0),
        "rss_mean_mb": round(float(np.mean(rss_vals)), 1) if rss_vals else 0,
        "rss_max_mb": max(rss_vals) if rss_vals else 0,
        "rss_growth_mb": round(mem_after.get("rss_mb", 0) - (rss_vals[0] if rss_vals else 0), 2),
        "target_rss_growth_mb": 50,
    }


# ============================================================
# P1 扩展性测试 (优化版 - 仅采样)
# ============================================================

def p1_scalability(data_sizes: List[int], query_samples: int = 100) -> Dict:
    print(f"\n  [P1扩展] 数据规模: {data_sizes}")
    results = {}
    base_w = None
    base_q = None

    for size in data_sizes:
        print(f"    {size}条...", end=" ", flush=True)
        tmpdir = tempfile.mkdtemp(prefix=f"su_p1_s_{size}_")
        try:
            client = SuMemory(mode="local", persist_dir=tmpdir)
            texts = generate_test_texts(size)
            queries = generate_query_texts(query_samples)

            write_latencies = []
            sample_count = min(50, size)
            step = max(1, size // sample_count)
            for i in range(size):
                start = time.perf_counter()
                client.add(texts[i])
                elapsed = time.perf_counter() - start
                if i % step == 0:
                    write_latencies.append(elapsed)

            query_latencies = []
            for q in queries:
                start = time.perf_counter()
                client.query(q, top_k=5)
                query_latencies.append(time.perf_counter() - start)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        wp50 = percentile(write_latencies, 50) * 1000
        winc = ((wp50 - base_w) / base_w * 100) if base_w is not None else 0
        qp50 = percentile(query_latencies, 50) * 1000
        qinc = ((qp50 - base_q) / base_q * 100) if base_q is not None else 0

        if base_w is None:
            base_w = wp50
            base_q = qp50

        key = f"{size // 1000}K" if size >= 1000 else str(size)
        results[key] = {
            "data_size": size,
            "write_p50_ms": round(wp50, 3),
            "query_p50_ms": round(qp50, 3),
            "write_increase_pct": round(winc, 1),
            "query_increase_pct": round(qinc, 1),
        }
        print(f"写入P50={wp50:.3f}ms (增幅{winc:+.1f}%) 检索P50={qp50:.3f}ms (增幅{qinc:+.1f}%)")

    return results


# ============================================================
# 达标检查
# ============================================================

def check_targets(all_results: Dict) -> List[Dict]:
    checks = []
    # 延迟
    for key in ["p1_latency_add", "p1_latency_query"]:
        data = all_results.get(key, {})
        for pk, tv in data.get("targets", {}).items():
            av = data.get(f"{pk}_ms", float("inf"))
            checks.append({
                "test": f"{data.get('operation', key)} {pk.upper()}",
                "actual_ms": av, "target_ms": tv,
                "passed": av < tv,
            })
    # 吞吐量
    for key in ["p1_tp_add", "p1_tp_query"]:
        data = all_results.get(key, {})
        for metric, field, op in [("QPS", "qps", ">="), ("错误率", "error_rate", "<=")]:
            tv = data.get(f"target_{field}" if field == "qps" else "target_error_rate", 0)
            av = data.get(field, 0)
            passed = av >= tv if op == ">=" else av <= tv
            checks.append({
                "test": f"{data.get('operation', key)} {metric}",
                "actual": av if field == "qps" else f"{av:.4f}",
                "target": tv,
                "passed": passed,
            })
    # 资源
    stab = all_results.get("p1_stability", {})
    growth = stab.get("rss_growth_mb", 0)
    checks.append({
        "test": "长时间运行 RSS 增长",
        "actual_mb": growth,
        "target_mb": stab.get("target_rss_growth_mb", 50),
        "passed": growth < stab.get("target_rss_growth_mb", 50),
    })

    return checks


# ============================================================
# 汇总打印
# ============================================================

def print_report(all_results: Dict, checks: List[Dict]):
    print("\n" + "=" * 70)
    print("  su-memory SDK v3.5.4 P1 大规模基准测试报告")
    print("=" * 70)

    # 延迟
    print("\n--- P1 延迟 (10K预填) ---")
    print(f"{'测试':<28s} {'P50(ms)':>10s} {'P95(ms)':>10s} {'P99(ms)':>10s} {'达标':>6s}")
    print("-" * 68)
    for key in ["p1_latency_add", "p1_latency_query"]:
        d = all_results.get(key, {})
        targets = d.get("targets", {})
        all_ok = all(d.get(f"{k}_ms", float("inf")) < v for k, v in targets.items())
        print(f"{d.get('operation', key):<28s} {d.get('p50_ms',0):>10.3f} {d.get('p95_ms',0):>10.3f} {d.get('p99_ms',0):>10.3f} {'✅' if all_ok else '❌':>6s}")

    # 吞吐量
    print("\n--- P1 吞吐量 (高并发) ---")
    print(f"{'测试':<28s} {'QPS':>10s} {'错误率':>10s} {'目标QPS':>10s} {'达标':>6s}")
    print("-" * 68)
    for key in ["p1_tp_add", "p1_tp_query"]:
        d = all_results.get(key, {})
        qps_ok = d.get("qps", 0) >= d.get("target_qps", 0)
        err_ok = d.get("error_rate", 0) <= d.get("target_error_rate", 1)
        print(f"{d.get('operation', key):<28s} {d.get('qps',0):>10.1f} {d.get('error_rate',0):>10.4f} {d.get('target_qps',0):>10.1f} {'✅' if (qps_ok and err_ok) else '❌':>6s}")

    # 资源
    print("\n--- P1 资源稳定性 ---")
    stab = all_results.get("p1_stability", {})
    print(f"  持续运行: {stab.get('duration_sec',0)}s | 总操作: {stab.get('total_ops',0)}")
    print(f"  RSS 起始: {stab.get('rss_start_mb',0):.1f}MB → 结束: {stab.get('rss_end_mb',0):.1f}MB")
    print(f"  RSS 均值: {stab.get('rss_mean_mb',0):.1f}MB | 最大: {stab.get('rss_max_mb',0):.1f}MB | 增长: {stab.get('rss_growth_mb',0):.1f}MB")

    # 扩展性
    print("\n--- P1 扩展性 ---")
    print(f"{'数据量':<10s} {'写入P50':>10s} {'检索P50':>10s} {'写入增幅':>10s} {'检索增幅':>10s}")
    print("-" * 52)
    for key, d in all_results.get("p1_scalability", {}).items():
        print(f"{d.get('data_size',0):<10d} {d.get('write_p50_ms',0):>8.3f}ms {d.get('query_p50_ms',0):>8.3f}ms {d.get('write_increase_pct',0):>+8.1f}% {d.get('query_increase_pct',0):>+8.1f}%")

    # 达标
    print("\n--- 达标清单 ---")
    print(f"{'测试项':<45s} {'结果':>8s}")
    print("-" * 55)
    passed = sum(1 for c in checks if c["passed"])
    failed = len(checks) - passed
    for c in checks:
        print(f"{c['test']:<45s} {'✅ PASS' if c['passed'] else '❌ FAIL':>8s}")
    print(f"\n总计: {len(checks)}项 | 通过: {passed} | 未通过: {failed}")


# ============================================================
# 主入口
# ============================================================

def main():
    print("=" * 70)
    print("  su-memory SDK v3.5.4 P1 大规模基准测试")
    print(f"  Python: {sys.version.split()[0]}")
    print("=" * 70)

    results = {
        "meta": {
            "version": "3.5.4",
            "test_level": "P1",
            "python_version": sys.version,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    # P1 延迟 (10K 预填)
    print("\n" + "=" * 50)
    print("  [1/5] P1 大规模延迟测试")
    print("=" * 50)
    results["p1_latency_add"] = p1_latency_add(samples=500, data_count=10000)
    print(f"    → P50={results['p1_latency_add']['p50_ms']}ms P95={results['p1_latency_add']['p95_ms']}ms P99={results['p1_latency_add']['p99_ms']}ms")
    results["p1_latency_query"] = p1_latency_query(samples=500, data_count=10000)
    print(f"    → P50={results['p1_latency_query']['p50_ms']}ms P95={results['p1_latency_query']['p95_ms']}ms P99={results['p1_latency_query']['p99_ms']}ms")

    # P1 吞吐量 (高并发)
    print("\n" + "=" * 50)
    print("  [2/5] P1 高并发吞吐量测试")
    print("=" * 50)
    results["p1_tp_add"] = p1_throughput_add(workers=20, duration=10.0)
    print(f"    → QPS={results['p1_tp_add']['qps']} 错误率={results['p1_tp_add']['error_rate']}")
    results["p1_tp_query"] = p1_throughput_query(workers=20, duration=10.0)
    print(f"    → QPS={results['p1_tp_query']['qps']} 错误率={results['p1_tp_query']['error_rate']}")

    # P1 资源稳定性
    print("\n" + "=" * 50)
    print("  [3/5] P1 长时间资源稳定性")
    print("=" * 50)
    results["p1_stability"] = p1_resource_stability(data_count=5000, duration_sec=60.0)
    print(f"    → QPS={results['p1_stability']['qps']} RSS增长={results['p1_stability']['rss_growth_mb']}MB")

    # P1 扩展性
    print("\n" + "=" * 50)
    print("  [4/5] P1 扩展性测试")
    print("=" * 50)
    results["p1_scalability"] = p1_scalability([100, 1000, 10000, 50000], query_samples=100)

    # 达标检查
    print("\n" + "=" * 50)
    print("  [5/5] 达标检查")
    print("=" * 50)
    checks = check_targets(results)
    results["compliance"] = checks

    print_report(results, checks)

    # 保存
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results_v354_p1.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 报告已保存: {output_path}")

    return results


if __name__ == "__main__":
    main()
