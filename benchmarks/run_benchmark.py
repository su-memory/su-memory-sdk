#!/usr/bin/env python3
"""
su-memory SDK 性能基准测试（Benchmark）

测量三大类指标：
1. 延迟测试 — P50/P95/P99
2. 吞吐量测试 — QPS / 错误率
3. 资源占用测试 — 内存/RSS/写入耗时
4. 扩展性测试 — 不同数据规模下的性能变化

使用方式:
    python benchmarks/run_benchmark.py --scale small   # 100条数据
    python benchmarks/run_benchmark.py --scale medium  # 1K条数据（默认）
    python benchmarks/run_benchmark.py --scale large   # 10K条数据
"""

import argparse
import json
import os
import sys
import time
import random
import hashlib
import tempfile
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

# 确保可以导入项目模块
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
from su_core import SemanticEncoder, EncoderCore, SuCompressor, MultiViewRetriever


# ============================================================
# 测试数据生成
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
    "边缘计算节点部署完成",
    "5G应用场景验证成功",
    "区块链存证系统试运行",
    "物联网设备接入量突破百万",
    "深度学习模型精度提升5%",
    "自然语言处理引擎优化",
    "推荐算法点击率提高12%",
    "搜索引擎响应时间降低30%",
    "消息队列吞吐量提升至10万QPS",
    "缓存命中率优化至99.5%",
]


def generate_test_texts(count: int) -> List[str]:
    """生成指定数量的测试文本"""
    texts = []
    for i in range(count):
        base = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
        variation = f"编号{i}: {base}，指标{random.randint(1, 999)}"
        texts.append(variation)
    return texts


def generate_query_texts(count: int) -> List[str]:
    """生成查询文本"""
    queries = [
        "投资回报", "团队协作", "市场风险", "知识网络", "技术架构",
        "合作协议", "产品渗透", "项目进展", "用户满意", "运营流程",
        "数据安全", "新功能", "财务报表", "用户体验", "供应链",
        "品牌影响", "人才培养", "成本控制", "创新驱动", "数字化",
        "AI应用", "云计算", "微服务", "容器化", "自动化",
        "持续集成", "日志监控", "网络安全", "数据中台", "业务智能",
    ]
    return [queries[i % len(queries)] for i in range(count)]


# ============================================================
# 工具函数
# ============================================================

def percentile(data: List[float], p: float) -> float:
    """计算百分位数"""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (p / 100.0) * (len(sorted_data) - 1)
    lower = int(idx)
    upper = min(lower + 1, len(sorted_data) - 1)
    frac = idx - lower
    return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac


def get_memory_usage() -> Dict[str, float]:
    """获取当前进程内存使用"""
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
# 1. 延迟测试
# ============================================================

def benchmark_latency_encode(samples: int) -> Dict[str, Any]:
    """编码操作延迟测试（su_core SemanticEncoder.encode）"""
    print(f"\n  [延迟] 编码操作 (SemanticEncoder.encode) - {samples}次")
    encoder = SemanticEncoder()
    texts = generate_test_texts(samples)

    latencies = []
    for text in texts:
        start = time.perf_counter()
        encoder.encode(text, "fact")
        latencies.append(time.perf_counter() - start)

    result = {
        "operation": "encode (SemanticEncoder)",
        "samples": samples,
        "p50_ms": round(percentile(latencies, 50) * 1000, 3),
        "p95_ms": round(percentile(latencies, 95) * 1000, 3),
        "p99_ms": round(percentile(latencies, 99) * 1000, 3),
        "min_ms": round(min(latencies) * 1000, 3),
        "max_ms": round(max(latencies) * 1000, 3),
        "mean_ms": round(float(np.mean(latencies)) * 1000, 3),
        "targets": {"p50": 1, "p95": 5, "p99": 10},
    }
    print(f"    P50={result['p50_ms']:.3f}ms  P95={result['p95_ms']:.3f}ms  P99={result['p99_ms']:.3f}ms")
    return result


def benchmark_latency_holographic(samples: int) -> Dict[str, Any]:
    """全息检索延迟测试（su_core EncoderCore.retrieve_holographic）"""
    print(f"\n  [延迟] 全息检索 (EncoderCore.retrieve_holographic) - {samples}次")
    ec = EncoderCore()
    candidate_indices = list(range(64))

    latencies = []
    for i in range(samples):
        query_index = i % 64
        start = time.perf_counter()
        ec.retrieve_holographic(query_index, candidate_indices, top_k=8)
        latencies.append(time.perf_counter() - start)

    result = {
        "operation": "holographic retrieve (EncoderCore)",
        "samples": samples,
        "p50_ms": round(percentile(latencies, 50) * 1000, 3),
        "p95_ms": round(percentile(latencies, 95) * 1000, 3),
        "p99_ms": round(percentile(latencies, 99) * 1000, 3),
        "min_ms": round(min(latencies) * 1000, 3),
        "max_ms": round(max(latencies) * 1000, 3),
        "mean_ms": round(float(np.mean(latencies)) * 1000, 3),
        "targets": {"p50": 5, "p95": 10, "p99": 20},
    }
    print(f"    P50={result['p50_ms']:.3f}ms  P95={result['p95_ms']:.3f}ms  P99={result['p99_ms']:.3f}ms")
    return result


def benchmark_latency_compress(samples: int) -> Dict[str, Any]:
    """压缩操作延迟测试（SuCompressor.compress）"""
    print(f"\n  [延迟] 压缩操作 (SuCompressor.compress) - {samples}次")
    compressor = SuCompressor()
    texts = generate_test_texts(samples)

    latencies = []
    for text in texts:
        start = time.perf_counter()
        compressor.compress(text)
        latencies.append(time.perf_counter() - start)

    result = {
        "operation": "compress (SuCompressor)",
        "samples": samples,
        "p50_ms": round(percentile(latencies, 50) * 1000, 3),
        "p95_ms": round(percentile(latencies, 95) * 1000, 3),
        "p99_ms": round(percentile(latencies, 99) * 1000, 3),
        "min_ms": round(min(latencies) * 1000, 3),
        "max_ms": round(max(latencies) * 1000, 3),
        "mean_ms": round(float(np.mean(latencies)) * 1000, 3),
        "targets": {"p50": 10, "p95": 50, "p99": 100},
    }
    print(f"    P50={result['p50_ms']:.3f}ms  P95={result['p95_ms']:.3f}ms  P99={result['p99_ms']:.3f}ms")
    return result


def benchmark_latency_sdk_add(samples: int) -> Dict[str, Any]:
    """SDK 写入延迟测试（SuMemory.add 本地模式）"""
    print(f"\n  [延迟] SDK写入 (SuMemory.add 本地模式) - {samples}次")
    tmpdir = tempfile.mkdtemp(prefix="su_bench_")
    try:
        client = SuMemory(mode="local", persist_dir=tmpdir)
        texts = generate_test_texts(samples)

        latencies = []
        for text in texts:
            start = time.perf_counter()
            client.add(text)
            latencies.append(time.perf_counter() - start)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    result = {
        "operation": "SDK add (local mode)",
        "samples": samples,
        "p50_ms": round(percentile(latencies, 50) * 1000, 3),
        "p95_ms": round(percentile(latencies, 95) * 1000, 3),
        "p99_ms": round(percentile(latencies, 99) * 1000, 3),
        "min_ms": round(min(latencies) * 1000, 3),
        "max_ms": round(max(latencies) * 1000, 3),
        "mean_ms": round(float(np.mean(latencies)) * 1000, 3),
        "targets": {"p50": 150, "p95": 300, "p99": 500},
    }
    print(f"    P50={result['p50_ms']:.3f}ms  P95={result['p95_ms']:.3f}ms  P99={result['p99_ms']:.3f}ms")
    return result


def benchmark_latency_sdk_query(samples: int, prefill: int = 100) -> Dict[str, Any]:
    """SDK 检索延迟测试（SuMemory.query 本地模式）"""
    print(f"\n  [延迟] SDK检索 (SuMemory.query 本地模式) - {samples}次 (预填{prefill}条)")
    tmpdir = tempfile.mkdtemp(prefix="su_bench_")
    try:
        client = SuMemory(mode="local", persist_dir=tmpdir)
        fill_texts = generate_test_texts(prefill)
        for t in fill_texts:
            client.add(t)

        queries = generate_query_texts(samples)
        latencies = []
        for q in queries:
            start = time.perf_counter()
            client.query(q, top_k=5)
            latencies.append(time.perf_counter() - start)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    result = {
        "operation": "SDK query (local mode)",
        "samples": samples,
        "prefill_count": prefill,
        "p50_ms": round(percentile(latencies, 50) * 1000, 3),
        "p95_ms": round(percentile(latencies, 95) * 1000, 3),
        "p99_ms": round(percentile(latencies, 99) * 1000, 3),
        "min_ms": round(min(latencies) * 1000, 3),
        "max_ms": round(max(latencies) * 1000, 3),
        "mean_ms": round(float(np.mean(latencies)) * 1000, 3),
        "targets": {"p50": 100, "p95": 200, "p99": 400},
    }
    print(f"    P50={result['p50_ms']:.3f}ms  P95={result['p95_ms']:.3f}ms  P99={result['p99_ms']:.3f}ms")
    return result


def run_latency_tests(scale_config: Dict) -> Dict[str, Any]:
    """运行所有延迟测试"""
    print("\n" + "=" * 70)
    print("  延迟测试 (Latency Benchmarks)")
    print("=" * 70)

    results = {}
    results["encode"] = benchmark_latency_encode(scale_config["encode_samples"])
    results["holographic"] = benchmark_latency_holographic(scale_config["holographic_samples"])
    results["compress"] = benchmark_latency_compress(scale_config["compress_samples"])
    results["sdk_add"] = benchmark_latency_sdk_add(scale_config["sdk_samples"])
    results["sdk_query"] = benchmark_latency_sdk_query(
        scale_config["sdk_samples"], prefill=scale_config["data_count"]
    )
    return results


# ============================================================
# 2. 吞吐量测试
# ============================================================

def benchmark_throughput_encode(workers: int, duration_sec: float = 5.0) -> Dict[str, Any]:
    """纯编码吞吐量测试"""
    print(f"\n  [吞吐] 纯编码 (SemanticEncoder) - {workers}并发, {duration_sec}s")
    encoder = SemanticEncoder()
    texts = generate_test_texts(500)
    errors = 0
    total_ops = 0
    lock = threading.Lock()

    def worker():
        nonlocal errors, total_ops
        local_ops = 0
        local_errors = 0
        deadline = time.time() + duration_sec
        while time.time() < deadline:
            try:
                text = texts[local_ops % len(texts)]
                encoder.encode(text, "fact")
                local_ops += 1
            except Exception:
                local_errors += 1
                local_ops += 1
        with lock:
            total_ops += local_ops
            errors += local_errors

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker) for _ in range(workers)]
        for f in as_completed(futures):
            f.result()
    elapsed = time.perf_counter() - start

    qps = total_ops / elapsed
    error_rate = errors / max(total_ops, 1)

    result = {
        "operation": "encode (SemanticEncoder)",
        "workers": workers,
        "duration_sec": round(elapsed, 2),
        "total_ops": total_ops,
        "qps": round(qps, 1),
        "errors": errors,
        "error_rate": round(error_rate, 4),
        "target_qps": 1000,
        "target_error_rate": 0.001,
    }
    print(f"    QPS={result['qps']:.1f}  错误率={result['error_rate']:.4f}  总操作={total_ops}")
    return result


def benchmark_throughput_holographic(workers: int, duration_sec: float = 5.0) -> Dict[str, Any]:
    """纯检索吞吐量测试（全息检索）"""
    print(f"\n  [吞吐] 纯检索 (全息检索) - {workers}并发, {duration_sec}s")
    ec = EncoderCore()
    candidate_indices = list(range(64))
    errors = 0
    total_ops = 0
    lock = threading.Lock()

    def worker():
        nonlocal errors, total_ops
        local_ops = 0
        local_errors = 0
        deadline = time.time() + duration_sec
        idx = 0
        while time.time() < deadline:
            try:
                ec.retrieve_holographic(idx % 64, candidate_indices, top_k=8)
                local_ops += 1
                idx += 1
            except Exception:
                local_errors += 1
                local_ops += 1
                idx += 1
        with lock:
            total_ops += local_ops
            errors += local_errors

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker) for _ in range(workers)]
        for f in as_completed(futures):
            f.result()
    elapsed = time.perf_counter() - start

    qps = total_ops / elapsed
    error_rate = errors / max(total_ops, 1)

    result = {
        "operation": "holographic retrieve (EncoderCore)",
        "workers": workers,
        "duration_sec": round(elapsed, 2),
        "total_ops": total_ops,
        "qps": round(qps, 1),
        "errors": errors,
        "error_rate": round(error_rate, 4),
        "target_qps": 500,
        "target_error_rate": 0.001,
    }
    print(f"    QPS={result['qps']:.1f}  错误率={result['error_rate']:.4f}  总操作={total_ops}")
    return result


def benchmark_throughput_sdk_add(workers: int, duration_sec: float = 5.0) -> Dict[str, Any]:
    """SDK add 吞吐量测试"""
    print(f"\n  [吞吐] SDK写入 (SuMemory.add) - {workers}并发, {duration_sec}s")
    texts = generate_test_texts(500)
    errors = 0
    total_ops = 0
    lock = threading.Lock()

    def worker(worker_id: int):
        nonlocal errors, total_ops
        tmpdir = tempfile.mkdtemp(prefix=f"su_bench_add_{worker_id}_")
        try:
            client = SuMemory(mode="local", persist_dir=tmpdir)
            local_ops = 0
            local_errors = 0
            deadline = time.time() + duration_sec
            while time.time() < deadline:
                try:
                    text = texts[local_ops % len(texts)]
                    client.add(text)
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

    qps = total_ops / elapsed
    error_rate = errors / max(total_ops, 1)

    result = {
        "operation": "SDK add (local mode)",
        "workers": workers,
        "duration_sec": round(elapsed, 2),
        "total_ops": total_ops,
        "qps": round(qps, 1),
        "errors": errors,
        "error_rate": round(error_rate, 4),
        "target_qps": 50,
        "target_error_rate": 0.001,
    }
    print(f"    QPS={result['qps']:.1f}  错误率={result['error_rate']:.4f}  总操作={total_ops}")
    return result


def benchmark_throughput_sdk_query(workers: int, duration_sec: float = 5.0) -> Dict[str, Any]:
    """SDK query 吞吐量测试"""
    print(f"\n  [吞吐] SDK检索 (SuMemory.query) - {workers}并发, {duration_sec}s")
    queries = generate_query_texts(200)
    errors = 0
    total_ops = 0
    lock = threading.Lock()

    def worker(worker_id: int):
        nonlocal errors, total_ops
        tmpdir = tempfile.mkdtemp(prefix=f"su_bench_q_{worker_id}_")
        try:
            client = SuMemory(mode="local", persist_dir=tmpdir)
            fill_texts = generate_test_texts(100)
            for t in fill_texts:
                client.add(t)

            local_ops = 0
            local_errors = 0
            deadline = time.time() + duration_sec
            while time.time() < deadline:
                try:
                    q = queries[local_ops % len(queries)]
                    client.query(q, top_k=5)
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

    qps = total_ops / elapsed
    error_rate = errors / max(total_ops, 1)

    result = {
        "operation": "SDK query (local mode)",
        "workers": workers,
        "duration_sec": round(elapsed, 2),
        "total_ops": total_ops,
        "qps": round(qps, 1),
        "errors": errors,
        "error_rate": round(error_rate, 4),
        "target_qps": 50,
        "target_error_rate": 0.001,
    }
    print(f"    QPS={result['qps']:.1f}  错误率={result['error_rate']:.4f}  总操作={total_ops}")
    return result


def benchmark_throughput_mixed(workers: int, write_ratio: float = 0.7, duration_sec: float = 5.0) -> Dict[str, Any]:
    """混合读写吞吐量测试（7:3）"""
    print(f"\n  [吞吐] 混合读写 (写{write_ratio*100:.0f}%/读{(1-write_ratio)*100:.0f}%) - {workers}并发, {duration_sec}s")
    texts = generate_test_texts(500)
    queries = generate_query_texts(200)
    errors = 0
    total_ops = 0
    write_ops = 0
    read_ops = 0
    lock = threading.Lock()

    def worker(worker_id: int):
        nonlocal errors, total_ops, write_ops, read_ops
        tmpdir = tempfile.mkdtemp(prefix=f"su_bench_mix_{worker_id}_")
        try:
            client = SuMemory(mode="local", persist_dir=tmpdir)
            fill_texts = generate_test_texts(50)
            for t in fill_texts:
                client.add(t)

            local_ops = 0
            local_errors = 0
            local_writes = 0
            local_reads = 0
            deadline = time.time() + duration_sec
            while time.time() < deadline:
                try:
                    if random.random() < write_ratio:
                        text = texts[local_ops % len(texts)]
                        client.add(text)
                        local_writes += 1
                    else:
                        q = queries[local_ops % len(queries)]
                        client.query(q, top_k=5)
                        local_reads += 1
                    local_ops += 1
                except Exception:
                    local_errors += 1
                    local_ops += 1
            with lock:
                total_ops += local_ops
                errors += local_errors
                write_ops += local_writes
                read_ops += local_reads
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, i) for i in range(workers)]
        for f in as_completed(futures):
            f.result()
    elapsed = time.perf_counter() - start

    qps = total_ops / elapsed
    error_rate = errors / max(total_ops, 1)

    result = {
        "operation": "mixed read/write (7:3)",
        "workers": workers,
        "duration_sec": round(elapsed, 2),
        "total_ops": total_ops,
        "write_ops": write_ops,
        "read_ops": read_ops,
        "qps": round(qps, 1),
        "errors": errors,
        "error_rate": round(error_rate, 4),
        "target_qps": 40,
        "target_error_rate": 0.005,
    }
    print(f"    QPS={result['qps']:.1f}  错误率={result['error_rate']:.4f}  写={write_ops} 读={read_ops}")
    return result


def run_throughput_tests(scale_config: Dict) -> Dict[str, Any]:
    """运行所有吞吐量测试"""
    print("\n" + "=" * 70)
    print("  吞吐量测试 (Throughput Benchmarks)")
    print("=" * 70)

    duration = scale_config.get("throughput_duration", 5.0)
    results = {}
    results["encode"] = benchmark_throughput_encode(10, duration)
    results["holographic"] = benchmark_throughput_holographic(10, duration)
    results["sdk_add"] = benchmark_throughput_sdk_add(10, duration)
    results["sdk_query"] = benchmark_throughput_sdk_query(10, duration)
    results["mixed"] = benchmark_throughput_mixed(20, 0.7, duration)
    return results


# ============================================================
# 3. 资源占用测试
# ============================================================

def benchmark_resource_usage(data_count: int) -> Dict[str, Any]:
    """资源占用测试"""
    print(f"\n  [资源] {data_count}条记忆的内存占用")

    mem_before = get_memory_usage()

    tmpdir = tempfile.mkdtemp(prefix="su_bench_res_")
    try:
        client = SuMemory(mode="local", persist_dir=tmpdir)
        texts = generate_test_texts(data_count)

        start = time.perf_counter()
        for text in texts:
            client.add(text)
        write_elapsed = time.perf_counter() - start

        mem_after = get_memory_usage()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    rss_delta = mem_after.get("rss_mb", 0) - mem_before.get("rss_mb", 0)
    target_mem = 500 if data_count <= 1000 else 1500

    result = {
        "data_count": data_count,
        "write_elapsed_sec": round(write_elapsed, 3),
        "write_qps": round(data_count / write_elapsed, 1) if write_elapsed > 0 else 0,
        "rss_before_mb": mem_before.get("rss_mb", 0),
        "rss_after_mb": mem_after.get("rss_mb", 0),
        "rss_delta_mb": round(rss_delta, 2),
        "vms_before_mb": mem_before.get("vms_mb", 0),
        "vms_after_mb": mem_after.get("vms_mb", 0),
        "psutil_available": PSUTIL_AVAILABLE,
        "target_rss_mb": target_mem,
    }

    print(f"    写入耗时={result['write_elapsed_sec']:.3f}s  写入QPS={result['write_qps']:.1f}")
    print(f"    RSS: {mem_before.get('rss_mb', 0):.1f}MB -> {mem_after.get('rss_mb', 0):.1f}MB  (增量={rss_delta:.1f}MB)")
    return result


def run_resource_tests(scale_config: Dict) -> Dict[str, Any]:
    """运行资源占用测试"""
    print("\n" + "=" * 70)
    print("  资源占用测试 (Resource Benchmarks)")
    print("=" * 70)

    results = {}
    for count in scale_config["resource_counts"]:
        key = f"{count // 1000}K" if count >= 1000 else str(count)
        results[key] = benchmark_resource_usage(count)
    return results


# ============================================================
# 4. 扩展性测试
# ============================================================

def benchmark_scalability(data_sizes: List[int], query_samples: int = 200) -> Dict[str, Any]:
    """扩展性测试"""
    print("\n" + "=" * 70)
    print("  扩展性测试 (Scalability Benchmarks)")
    print("=" * 70)

    results = {}
    base_write_latency = None
    base_query_latency = None

    for size in data_sizes:
        print(f"\n  [扩展] 数据规模: {size}条")
        tmpdir = tempfile.mkdtemp(prefix=f"su_bench_scale_{size}_")
        try:
            client = SuMemory(mode="local", persist_dir=tmpdir)
            texts = generate_test_texts(size)
            queries = generate_query_texts(query_samples)

            # 写入延迟（采样）
            write_latencies = []
            sample_count = min(100, size)
            step = max(1, size // sample_count)
            for i in range(size):
                start = time.perf_counter()
                client.add(texts[i])
                elapsed = time.perf_counter() - start
                if i % step == 0:
                    write_latencies.append(elapsed)

            # 检索延迟
            query_latencies = []
            for q in queries:
                start = time.perf_counter()
                client.query(q, top_k=5)
                query_latencies.append(time.perf_counter() - start)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        write_p50 = percentile(write_latencies, 50) * 1000
        write_p95 = percentile(write_latencies, 95) * 1000
        query_p50 = percentile(query_latencies, 50) * 1000
        query_p95 = percentile(query_latencies, 95) * 1000

        if base_write_latency is None:
            base_write_latency = write_p50
            base_query_latency = query_p50

        write_increase = ((write_p50 - base_write_latency) / base_write_latency * 100
                          if base_write_latency > 0 else 0)
        query_increase = ((query_p50 - base_query_latency) / base_query_latency * 100
                          if base_query_latency > 0 else 0)

        result = {
            "data_size": size,
            "write_p50_ms": round(write_p50, 3),
            "write_p95_ms": round(write_p95, 3),
            "query_p50_ms": round(query_p50, 3),
            "query_p95_ms": round(query_p95, 3),
            "write_increase_pct": round(write_increase, 1),
            "query_increase_pct": round(query_increase, 1),
        }
        key = f"{size // 1000}K" if size >= 1000 else str(size)
        results[key] = result

        print(f"    写入 P50={write_p50:.3f}ms (增幅{write_increase:+.1f}%)  "
              f"检索 P50={query_p50:.3f}ms (增幅{query_increase:+.1f}%)")

    return results


# ============================================================
# 达标检查
# ============================================================

def check_latency_targets(latency_results: Dict) -> List[Dict]:
    """检查延迟是否达标"""
    checks = []
    for key, data in latency_results.items():
        targets = data.get("targets", {})
        for p_key, target_ms in targets.items():
            actual_key = f"{p_key}_ms"
            actual_ms = data.get(actual_key, float("inf"))
            passed = actual_ms < target_ms
            checks.append({
                "test": f"{data['operation']} {p_key.upper()}",
                "actual_ms": actual_ms,
                "target_ms": target_ms,
                "passed": passed,
            })
    return checks


def check_throughput_targets(throughput_results: Dict) -> List[Dict]:
    """检查吞吐量是否达标"""
    checks = []
    for key, data in throughput_results.items():
        if "target_qps" in data:
            passed = data["qps"] >= data["target_qps"]
            checks.append({
                "test": f"{data['operation']} QPS",
                "actual": data["qps"],
                "target": data["target_qps"],
                "passed": passed,
            })
        if "target_error_rate" in data:
            passed = data["error_rate"] <= data["target_error_rate"]
            checks.append({
                "test": f"{data['operation']} 错误率",
                "actual": data["error_rate"],
                "target": data["target_error_rate"],
                "passed": passed,
            })
    return checks


def check_resource_targets(resource_results: Dict) -> List[Dict]:
    """检查资源占用是否达标"""
    checks = []
    for key, data in resource_results.items():
        if "target_rss_mb" in data and data.get("psutil_available"):
            rss_after = data.get("rss_after_mb", 0)
            target = data["target_rss_mb"]
            passed = rss_after < target
            checks.append({
                "test": f"资源占用 {data['data_count']}条 RSS",
                "actual_mb": rss_after,
                "target_mb": target,
                "passed": passed,
            })
    return checks


# ============================================================
# 汇总报告
# ============================================================

def print_summary(all_results: Dict):
    """打印可读的汇总表格"""
    print("\n" + "=" * 70)
    print("  基准测试汇总报告")
    print("=" * 70)

    # 延迟汇总
    print("\n+------------------------------------------------------------------+")
    print("|  延迟测试汇总                                                    |")
    print("+--------------------------+----------+----------+----------+------+")
    print("| 操作                     | P50(ms)  | P95(ms)  | P99(ms)  | 达标 |")
    print("+--------------------------+----------+----------+----------+------+")

    latency = all_results.get("latency", {})
    for key, data in latency.items():
        op = data.get("operation", key)[:24]
        p50 = data.get("p50_ms", 0)
        p95 = data.get("p95_ms", 0)
        p99 = data.get("p99_ms", 0)
        targets = data.get("targets", {})
        all_pass = all(
            data.get(f"{k}_ms", float("inf")) < v
            for k, v in targets.items()
        )
        status = "PASS" if all_pass else "FAIL"
        print(f"| {op:<24s} | {p50:>8.3f} | {p95:>8.3f} | {p99:>8.3f} | {status:>4s} |")

    print("+--------------------------+----------+----------+----------+------+")

    # 吞吐量汇总
    print("\n+------------------------------------------------------------------+")
    print("|  吞吐量测试汇总                                                  |")
    print("+--------------------------+----------+----------+----------+------+")
    print("| 操作                     | QPS      | 错误率   | 目标QPS  | 达标 |")
    print("+--------------------------+----------+----------+----------+------+")

    throughput = all_results.get("throughput", {})
    for key, data in throughput.items():
        op = data.get("operation", key)[:24]
        qps = data.get("qps", 0)
        err = data.get("error_rate", 0)
        target_qps = data.get("target_qps", 0)
        qps_pass = qps >= target_qps
        err_pass = err <= data.get("target_error_rate", 1.0)
        status = "PASS" if (qps_pass and err_pass) else "FAIL"
        print(f"| {op:<24s} | {qps:>8.1f} | {err:>8.4f} | {target_qps:>8.1f} | {status:>4s} |")

    print("+--------------------------+----------+----------+----------+------+")

    # 资源占用汇总
    print("\n+------------------------------------------------------------------+")
    print("|  资源占用汇总                                                    |")
    print("+----------------+----------+----------+----------+----------------+")
    print("| 数据量         | 写入耗时 | 写入QPS  | RSS(MB)  | 增量RSS(MB)    |")
    print("+----------------+----------+----------+----------+----------------+")

    resource = all_results.get("resource", {})
    for key, data in resource.items():
        count = data.get("data_count", 0)
        elapsed = data.get("write_elapsed_sec", 0)
        wqps = data.get("write_qps", 0)
        rss = data.get("rss_after_mb", 0)
        delta = data.get("rss_delta_mb", 0)
        print(f"| {count:<14d} | {elapsed:>7.3f}s | {wqps:>8.1f} | {rss:>8.1f} | {delta:>14.1f} |")

    print("+----------------+----------+----------+----------+----------------+")

    # 扩展性汇总
    print("\n+------------------------------------------------------------------+")
    print("|  扩展性分析                                                      |")
    print("+----------------+----------+----------+----------+----------------+")
    print("| 数据量         | 写入P50  | 检索P50  | 写入增幅 | 检索增幅       |")
    print("+----------------+----------+----------+----------+----------------+")

    scalability = all_results.get("scalability", {})
    for key, data in scalability.items():
        size = data.get("data_size", 0)
        wp50 = data.get("write_p50_ms", 0)
        qp50 = data.get("query_p50_ms", 0)
        winc = data.get("write_increase_pct", 0)
        qinc = data.get("query_increase_pct", 0)
        print(f"| {size:<14d} | {wp50:>7.3f}ms| {qp50:>7.3f}ms| {winc:>+7.1f}% | {qinc:>+13.1f}% |")

    print("+----------------+----------+----------+----------+----------------+")

    # 达标/不达标清单
    print("\n+------------------------------------------------------------------+")
    print("|  达标/不达标清单                                                  |")
    print("+--------------------------------------------------------+--------+")
    print("| 测试项                                                 | 结果   |")
    print("+--------------------------------------------------------+--------+")

    all_checks = []
    if latency:
        all_checks.extend(check_latency_targets(latency))
    if throughput:
        all_checks.extend(check_throughput_targets(throughput))
    if resource:
        all_checks.extend(check_resource_targets(resource))

    pass_count = 0
    fail_count = 0
    for check in all_checks:
        status = "PASS" if check["passed"] else "FAIL"
        if check["passed"]:
            pass_count += 1
        else:
            fail_count += 1
        test_name = check["test"][:54]
        print(f"| {test_name:<54s} | {status:>6s} |")

    print("+--------------------------------------------------------+--------+")
    total = pass_count + fail_count
    print(f"| 总计: {total}项  通过: {pass_count}  未通过: {fail_count}{' ' * (40 - len(str(total)) - len(str(pass_count)) - len(str(fail_count)))} |")
    print("+--------------------------------------------------------+--------+")


# ============================================================
# 主入口
# ============================================================

SCALE_CONFIGS = {
    "small": {
        "data_count": 100,
        "encode_samples": 1000,
        "holographic_samples": 500,
        "compress_samples": 500,
        "sdk_samples": 100,
        "query_samples": 100,
        "resource_counts": [100, 1000],
        "scalability_sizes": [100, 1000],
        "throughput_duration": 3.0,
    },
    "medium": {
        "data_count": 1000,
        "encode_samples": 5000,
        "holographic_samples": 1000,
        "compress_samples": 1000,
        "sdk_samples": 1000,
        "query_samples": 1000,
        "resource_counts": [1000, 10000],
        "scalability_sizes": [100, 1000, 10000],
        "throughput_duration": 5.0,
    },
    "large": {
        "data_count": 10000,
        "encode_samples": 5000,
        "holographic_samples": 2000,
        "compress_samples": 2000,
        "sdk_samples": 2000,
        "query_samples": 5000,
        "resource_counts": [1000, 10000],
        "scalability_sizes": [100, 1000, 10000],
        "throughput_duration": 10.0,
    },
}


def main():
    parser = argparse.ArgumentParser(description="su-memory SDK 性能基准测试")
    parser.add_argument(
        "--scale",
        choices=["small", "medium", "large"],
        default="medium",
        help="测试规模: small(100), medium(1K), large(10K)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON报告输出路径 (默认: benchmarks/benchmark_results.json)"
    )
    args = parser.parse_args()

    config = SCALE_CONFIGS[args.scale]
    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "benchmark_results.json"
    )

    print("=" * 70)
    print(f"  su-memory SDK 性能基准测试")
    print(f"  规模: {args.scale}  数据量: {config['data_count']}")
    print(f"  Python: {sys.version.split()[0]}")
    print("=" * 70)

    all_results = {
        "meta": {
            "scale": args.scale,
            "config": config,
            "python_version": sys.version,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    # 运行所有测试
    all_results["latency"] = run_latency_tests(config)
    all_results["throughput"] = run_throughput_tests(config)
    all_results["resource"] = run_resource_tests(config)
    all_results["scalability"] = benchmark_scalability(
        config["scalability_sizes"],
        query_samples=config["query_samples"]
    )

    # 添加达标检查
    all_results["compliance"] = {
        "latency": check_latency_targets(all_results["latency"]),
        "throughput": check_throughput_targets(all_results["throughput"]),
        "resource": check_resource_targets(all_results["resource"]),
    }

    # 打印汇总
    print_summary(all_results)

    # 保存JSON报告
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 报告已保存: {output_path}")

    return all_results


if __name__ == "__main__":
    main()
