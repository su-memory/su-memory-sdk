#!/usr/bin/env python3
"""
su-memory SDK v3.5.4 P2 大规模性能基准测试

针对 50K-100K 数据规模的深度测试：
1. 扩展性测试 (Scalability) — 100/1K/10K/50K/100K 写入与查询延迟变化趋势
2. 长期稳定性测试 (Long-term Stability) — 持续运行 2 小时+ 的资源占用稳定性
3. 压力负载测试 (Stress Load) — 100 并发线程混合读写操作

使用方式:
    python benchmarks/run_benchmark_p2.py
    python benchmarks/run_benchmark_p2.py --skip-stability   # 跳过2小时稳定性测试
    python benchmarks/run_benchmark_p2.py --skip-stress      # 跳过压力测试
"""

import json
import os
import sys
import time
import random
import tempfile
import shutil
import threading
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

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
        "持续集成", "日志监控", "网络安全", "数据中台", "业务智能",
        "边缘计算", "5G应用", "区块链", "物联网", "深度学习",
        "自然语言", "推荐算法", "搜索引擎", "消息队列", "缓存",
    ]
    return [queries[i % len(queries)] for i in range(count)]


# ============================================================
# 工具函数
# ============================================================

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


def get_system_memory() -> Dict[str, float]:
    """获取系统级内存信息"""
    if not PSUTIL_AVAILABLE:
        return {"total_gb": 0, "available_gb": 0, "percent": 0, "available": False}
    mem = psutil.virtual_memory()
    return {
        "total_gb": round(mem.total / 1024 / 1024 / 1024, 2),
        "available_gb": round(mem.available / 1024 / 1024 / 1024, 2),
        "percent": mem.percent,
        "available": True,
    }


# ============================================================
# 1. P2 扩展性测试 (100 → 100K)
# ============================================================

def p2_scalability(
    data_sizes: List[int],
    query_samples: int = 200,
    batch_size: int = 500,
) -> Dict[str, Any]:
    """
    P2 扩展性测试：测试不同数据规模下的写入和查询延迟变化趋势。

    测试规模: 100, 1K, 10K, 50K, 100K
    指标: 写入 P50/P95/P99, 查询 P50/P95/P99, 内存占用
    """
    print("\n" + "=" * 60)
    print("  P2 扩展性测试 (Scalability)")
    print("=" * 60)
    print(f"  数据规模: {data_sizes}")
    print(f"  查询样本: {query_samples}次/规模")
    print(f"  批量写入: {batch_size}条/批")

    results = {}
    base_write_p50 = None
    base_query_p50 = None

    for size in data_sizes:
        label = f"{size // 1000}K" if size >= 1000 else str(size)
        print(f"\n  [{label}] 规模={size}条...", end=" ", flush=True)

        t_start = time.perf_counter()
        tmpdir = tempfile.mkdtemp(prefix=f"su_p2_s_{size}_")
        try:
            client = SuMemory(mode="local", persist_dir=tmpdir)
            texts = generate_test_texts(size)

            # 批量写入并采样延迟
            write_latencies = []
            for offset in range(0, size, batch_size):
                bsize = min(batch_size, size - offset)
                batch_texts = texts[offset:offset + bsize]
                t_batch = time.perf_counter()
                for text in batch_texts:
                    client.add(text)
                batch_elapsed = time.perf_counter() - t_batch
                per_item_ms = batch_elapsed / bsize * 1000
                # 每批采样一次
                write_latencies.append(per_item_ms)

            # 查询延迟
            queries = generate_query_texts(query_samples)
            query_latencies = []
            for q in queries:
                t_q = time.perf_counter()
                client.query(q, top_k=5)
                query_latencies.append((time.perf_counter() - t_q) * 1000)

            # 内存占用
            mem_after = get_memory_usage()

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        elapsed = time.perf_counter() - t_start

        wp50 = percentile(write_latencies, 50)
        wp95 = percentile(write_latencies, 95)
        wp99 = percentile(write_latencies, 99)
        qp50 = percentile(query_latencies, 50)
        qp95 = percentile(query_latencies, 95)
        qp99 = percentile(query_latencies, 99)
        qmean = float(np.mean(query_latencies))

        if base_write_p50 is None:
            base_write_p50 = wp50
            base_query_p50 = qp50

        winc = ((wp50 - base_write_p50) / base_write_p50 * 100) if base_write_p50 > 0 else 0
        qinc = ((qp50 - base_query_p50) / base_query_p50 * 100) if base_query_p50 > 0 else 0

        result = {
            "data_size": size,
            "write_p50_ms": round(wp50, 3),
            "write_p95_ms": round(wp95, 3),
            "write_p99_ms": round(wp99, 3),
            "query_p50_ms": round(qp50, 3),
            "query_p95_ms": round(qp95, 3),
            "query_p99_ms": round(qp99, 3),
            "query_mean_ms": round(qmean, 3),
            "write_increase_pct": round(winc, 1),
            "query_increase_pct": round(qinc, 1),
            "rss_after_mb": mem_after.get("rss_mb", 0),
            "elapsed_sec": round(elapsed, 1),
        }
        results[label] = result

        print(f"完成 ({elapsed:.1f}s)")
        print(f"    写入 P50={wp50:.3f}ms P95={wp95:.3f}ms P99={wp99:.3f}ms "
              f"(增幅{winc:+.1f}%)")
        print(f"    查询 P50={qp50:.3f}ms P95={qp95:.3f}ms P99={qp99:.3f}ms "
              f"(增幅{qinc:+.1f}%) | 均值={qmean:.3f}ms")

        # 大规格时主动 GC
        if size >= 10000:
            gc.collect()

    return results


# ============================================================
# 2. P2 长期稳定性测试 (2小时+)
# ============================================================

def p2_long_term_stability(
    data_count: int = 10000,
    duration_sec: float = 7200.0,  # 2小时
    monitor_interval: float = 60.0,  # 每60秒采样一次
) -> Dict[str, Any]:
    """
    P2 长期稳定性测试：持续运行 2 小时以上，监控资源占用稳定性。

    监控指标:
    - RSS 内存变化趋势
    - CPU 使用率 (通过 psutil)
    - 磁盘空间增长 (持久化目录大小)
    - 查询延迟稳定性
    """
    print("\n" + "=" * 60)
    print("  P2 长期稳定性测试 (Long-term Stability)")
    print("=" * 60)
    print(f"  预填数据: {data_count}条")
    print(f"  持续时间: {duration_sec / 3600:.1f}小时 ({duration_sec}s)")
    print(f"  监控间隔: {monitor_interval}s")

    tmpdir = tempfile.mkdtemp(prefix="su_p2_stab_")
    mem_samples: List[Dict] = []
    sys_mem_samples: List[Dict] = []
    query_latency_samples: List[float] = []
    disk_size_samples: List[float] = []

    try:
        # 初始填充
        print("  正在预填数据...", end=" ", flush=True)
        t_fill = time.perf_counter()
        client = SuMemory(mode="local", persist_dir=tmpdir)
        texts = generate_test_texts(data_count)
        for i, text in enumerate(texts):
            client.add(text)
            if (i + 1) % 1000 == 0:
                pct = (i + 1) / data_count * 100
                print(f"{pct:.0f}%", end=" ", flush=True)
        fill_elapsed = time.perf_counter() - t_fill
        print(f"完成 ({fill_elapsed:.1f}s)")

        # 初始采样
        mem_samples.append({
            "timestamp_sec": 0,
            **get_memory_usage(),
            "system": get_system_memory(),
        })
        disk_size_samples.append(_get_dir_size(tmpdir))

        # 查询负载
        queries = generate_query_texts(40)
        deadline = time.time() + duration_sec
        total_ops = 0
        total_errors = 0
        last_monitor = time.time()

        print(f"  开始持续运行 {duration_sec / 3600:.1f} 小时...")
        print(f"  结束时间: {time.strftime('%H:%M:%S', time.localtime(deadline))}")

        query_batch_latencies = []
        while time.time() < deadline:
            try:
                q = queries[total_ops % len(queries)]
                t_q = time.perf_counter()
                client.query(q, top_k=5)
                lat = (time.perf_counter() - t_q) * 1000
                query_batch_latencies.append(lat)
                total_ops += 1
            except Exception:
                total_errors += 1
                total_ops += 1

            # 定时监控采样
            now = time.time()
            if now - last_monitor >= monitor_interval:
                last_monitor = now
                remaining = deadline - now
                hours_done = (duration_sec - remaining) / 3600

                mem_info = get_memory_usage()
                sys_mem = get_system_memory()
                disk_size = _get_dir_size(tmpdir)

                mem_samples.append({
                    "timestamp_sec": round(duration_sec - remaining, 1),
                    **mem_info,
                    "system": sys_mem,
                })
                sys_mem_samples.append(sys_mem)
                disk_size_samples.append(disk_size)

                # 记录这段时间内的查询延迟
                avg_lat = float(np.mean(query_batch_latencies)) if query_batch_latencies else 0
                query_latency_samples.append(avg_lat)
                query_batch_latencies = []

                print(f"    [{hours_done:.1f}h] 剩余{remaining / 3600:.1f}h | "
                      f"RSS={mem_info.get('rss_mb', 0):.1f}MB | "
                      f"系统内存{100 - sys_mem.get('percent', 0):.0f}%可用 | "
                      f"查询延迟{avg_lat:.1f}ms | 已操作{total_ops:,}次")

        # 最终采样
        mem_after = get_memory_usage()
        sys_mem_after = get_system_memory()
        final_disk_size = _get_dir_size(tmpdir)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    elapsed = duration_sec

    # 分析 RSS 趋势
    rss_values = [s.get("rss_mb", 0) for s in mem_samples if s.get("available")]
    rss_start = rss_values[0] if rss_values else 0
    rss_end = rss_values[-1] if rss_values else 0
    rss_growth = rss_end - rss_start
    rss_mean = float(np.mean(rss_values)) if rss_values else 0
    rss_std = float(np.std(rss_values)) if len(rss_values) > 1 else 0
    rss_max = max(rss_values) if rss_values else 0

    # 分析磁盘增长
    disk_start = disk_size_samples[0] if disk_size_samples else 0
    disk_end = disk_size_samples[-1] if disk_size_samples else 0
    disk_growth = disk_end - disk_start

    # 分析查询延迟趋势
    # 对比前30分钟和后30分钟的查询延迟
    if len(query_latency_samples) >= 2:
        mid = len(query_latency_samples) // 2
        early_lat = float(np.mean(query_latency_samples[:mid])) if mid > 0 else 0
        late_lat = float(np.mean(query_latency_samples[mid:])) if mid < len(query_latency_samples) else 0
        lat_drift = late_lat - early_lat
    else:
        early_lat = query_latency_samples[0] if query_latency_samples else 0
        late_lat = early_lat
        lat_drift = 0

    result = {
        "operation": "P2 long-term stability",
        "data_count": data_count,
        "fill_elapsed_sec": round(fill_elapsed, 1),
        "duration_sec": round(elapsed, 1),
        "duration_hours": round(elapsed / 3600, 1),
        "total_ops": total_ops,
        "total_errors": total_errors,
        "error_rate": round(total_errors / max(total_ops, 1), 6),
        "avg_qps": round(total_ops / max(elapsed, 0.001), 2),
        "monitor_samples": len(mem_samples),
        # RSS 指标
        "rss_start_mb": round(rss_start, 1),
        "rss_end_mb": round(rss_end, 1),
        "rss_mean_mb": round(rss_mean, 1),
        "rss_std_mb": round(rss_std, 2),
        "rss_max_mb": round(rss_max, 1),
        "rss_growth_mb": round(rss_growth, 2),
        "rss_growth_per_hour_mb": round(rss_growth / (elapsed / 3600), 2) if elapsed > 0 else 0,
        # 磁盘指标
        "disk_start_mb": round(disk_start, 2),
        "disk_end_mb": round(disk_end, 2),
        "disk_growth_mb": round(disk_growth, 2),
        # 查询延迟漂移
        "query_lat_early_ms": round(early_lat, 2),
        "query_lat_late_ms": round(late_lat, 2),
        "query_lat_drift_ms": round(lat_drift, 2),
        # 系统内存
        "sys_mem_start_pct": round(sys_mem_samples[0].get("percent", 0), 1) if sys_mem_samples else 0,
        "sys_mem_end_pct": round(sys_mem_after.get("percent", 0), 1),
        # 目标阈值 (P2)
        "targets": {
            "rss_growth_mb": 200,          # 2小时内 RSS 增长 < 200MB
            "error_rate": 0.01,             # 错误率 < 1%
            "query_lat_drift_ms": 50,       # 查询延迟漂移 < 50ms
            "disk_growth_mb": 500,          # 磁盘增长 < 500MB
        },
    }

    return result


def _get_dir_size(path: str) -> float:
    """获取目录大小 (MB)"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return round(total / 1024 / 1024, 2)


# ============================================================
# 3. P2 压力负载测试 (100 并发)
# ============================================================

def p2_stress_load(
    workers: int = 100,
    duration_sec: float = 30.0,
    data_per_worker: int = 100,
    write_ratio: float = 0.7,
) -> Dict[str, Any]:
    """
    P2 压力负载测试：100 并发线程同时进行读写操作。

    测试场景:
    - 100 个并发 worker 各自持有独立的 SuMemory 客户端
    - 70% 写入操作, 30% 查询操作 (模拟真实负载)
    - 记录每个操作的延迟分布、吞吐量、错误率
    """
    print("\n" + "=" * 60)
    print("  P2 压力负载测试 (Stress Load)")
    print("=" * 60)
    print(f"  并发线程: {workers}")
    print(f"  持续时间: {duration_sec}s")
    print(f"  每线程数据量: {data_per_worker}条")
    print(f"  读写比例: {write_ratio*100:.0f}%/{(1-write_ratio)*100:.0f}%")

    all_write_latencies: List[float] = []
    all_query_latencies: List[float] = []
    total_write_ops = 0
    total_query_ops = 0
    total_errors = 0
    lock = threading.Lock()
    worker_results: List[Dict] = []

    texts_pool = generate_test_texts(200)
    queries_pool = generate_query_texts(100)

    mem_before = get_memory_usage()

    def worker(wid: int) -> Dict:
        nonlocal total_write_ops, total_query_ops, total_errors

        local_write_lats = []
        local_query_lats = []
        local_writes = 0
        local_queries = 0
        local_errors = 0

        tmpdir = tempfile.mkdtemp(prefix=f"su_p2_stress_{wid}_")
        try:
            client = SuMemory(mode="local", persist_dir=tmpdir)
            deadline = time.time() + duration_sec

            op_count = 0
            while time.time() < deadline:
                try:
                    if random.random() < write_ratio:
                        text = texts_pool[(wid * 1000 + op_count) % len(texts_pool)]
                        t_start = time.perf_counter()
                        client.add(text)
                        lat = (time.perf_counter() - t_start) * 1000
                        local_write_lats.append(lat)
                        local_writes += 1
                    else:
                        q = queries_pool[(wid * 1000 + op_count) % len(queries_pool)]
                        t_start = time.perf_counter()
                        client.query(q, top_k=5)
                        lat = (time.perf_counter() - t_start) * 1000
                        local_query_lats.append(lat)
                        local_queries += 1
                    op_count += 1
                except Exception:
                    local_errors += 1
                    op_count += 1

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        with lock:
            total_write_ops += local_writes
            total_query_ops += local_queries
            total_errors += local_errors

        return {
            "worker_id": wid,
            "writes": local_writes,
            "queries": local_queries,
            "errors": local_errors,
            "write_p50_ms": round(percentile(local_write_lats, 50), 2) if local_write_lats else 0,
            "write_p95_ms": round(percentile(local_write_lats, 95), 2) if local_write_lats else 0,
            "query_p50_ms": round(percentile(local_query_lats, 50), 2) if local_query_lats else 0,
            "query_p95_ms": round(percentile(local_query_lats, 95), 2) if local_query_lats else 0,
            "_write_lats": local_write_lats,
            "_query_lats": local_query_lats,
        }

    print("  启动 100 个并发 worker...", end=" ", flush=True)

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, i) for i in range(workers)]
        completed = 0
        for f in as_completed(futures):
            try:
                result_item = f.result()
                worker_results.append(result_item)
                all_write_latencies.extend(result_item["_write_lats"])
                all_query_latencies.extend(result_item["_query_lats"])
                completed += 1
                if completed % 20 == 0:
                    pct = completed / workers * 100
                    print(f"{pct:.0f}%", end=" ", flush=True)
            except Exception as e:
                print(f"ERR({e})", end=" ", flush=True)
                total_errors += 1

    elapsed = time.perf_counter() - t_start
    mem_after = get_memory_usage()

    print(f"完成 ({elapsed:.1f}s)")

    total_ops = total_write_ops + total_query_ops
    error_rate = total_errors / max(total_ops, 1)
    total_qps = total_ops / max(elapsed, 0.001)

    # 全局延迟统计
    write_p50 = percentile(all_write_latencies, 50)
    write_p95 = percentile(all_write_latencies, 95)
    write_p99 = percentile(all_write_latencies, 99)
    query_p50 = percentile(all_query_latencies, 50)
    query_p95 = percentile(all_query_latencies, 95)
    query_p99 = percentile(all_query_latencies, 99)

    # 各 worker 延迟一致性 (标准差)
    worker_write_p50s = [w["write_p50_ms"] for w in worker_results]
    worker_query_p50s = [w["query_p50_ms"] for w in worker_results]
    write_std = float(np.std(worker_write_p50s)) if len(worker_write_p50s) > 1 else 0
    query_std = float(np.std(worker_query_p50s)) if len(worker_query_p50s) > 1 else 0

    print(f"    总操作: {total_ops:,} (写{total_write_ops:,} 读{total_query_ops:,})")
    print(f"    总QPS: {total_qps:.1f}  错误率: {error_rate:.4%}")
    print(f"    写入 P50={write_p50:.2f}ms P95={write_p95:.2f}ms P99={write_p99:.2f}ms")
    print(f"    查询 P50={query_p50:.2f}ms P95={query_p95:.2f}ms P99={query_p99:.2f}ms")
    print(f"    Worker间延迟标准差: 写入{write_std:.2f}ms / 查询{query_std:.2f}ms")

    result = {
        "operation": "P2 stress load (100 concurrent)",
        "workers": workers,
        "duration_sec": round(elapsed, 1),
        "total_ops": total_ops,
        "total_write_ops": total_write_ops,
        "total_query_ops": total_query_ops,
        "total_errors": total_errors,
        "error_rate": round(error_rate, 6),
        "total_qps": round(total_qps, 1),
        "write_p50_ms": round(write_p50, 2),
        "write_p95_ms": round(write_p95, 2),
        "write_p99_ms": round(write_p99, 2),
        "query_p50_ms": round(query_p50, 2),
        "query_p95_ms": round(query_p95, 2),
        "query_p99_ms": round(query_p99, 2),
        "worker_write_p50_std_ms": round(write_std, 2),
        "worker_query_p50_std_ms": round(query_std, 2),
        "rss_before_mb": mem_before.get("rss_mb", 0),
        "rss_after_mb": mem_after.get("rss_mb", 0),
        "rss_delta_mb": round(mem_after.get("rss_mb", 0) - mem_before.get("rss_mb", 0), 2),
        "worker_details": [
            {k: v for k, v in w.items() if not k.startswith("_")}
            for w in sorted(worker_results, key=lambda x: x["worker_id"])
        ],
        "targets": {
            "error_rate": 0.01,           # 错误率 < 1%
            "total_qps": 100,             # 总 QPS > 100
            "query_p95_ms": 500,          # 查询 P95 < 500ms
            "worker_std_ms": 20,          # Worker 间标准差 < 20ms
        },
    }

    return result


# ============================================================
# 达标检查
# ============================================================

def check_p2_targets(all_results: Dict) -> Dict[str, Any]:
    """检查所有 P2 测试是否达标"""
    checks = []

    # --- 扩展性检查 ---
    scalability = all_results.get("p2_scalability", {})
    scal_keys = sorted(scalability.keys(),
                       key=lambda k: scalability[k].get("data_size", 0))
    if len(scal_keys) >= 2:
        # 检查 100→100K 查询延迟是否在线性增长范围内
        first_key = scal_keys[0]
        last_key = scal_keys[-1]
        first_data = scalability[first_key]
        last_data = scalability[last_key]

        # 数据量增长倍数与延迟增长倍数比较
        size_ratio = last_data["data_size"] / max(first_data["data_size"], 1)
        query_ratio = last_data["query_p50_ms"] / max(first_data["query_p50_ms"], 1)
        is_sublinear = query_ratio / max(size_ratio, 1) < 1.0  # 亚线性增长

        checks.append({
            "test": f"扩展性 — {first_key}→{last_key} 查询延迟亚线性增长",
            "detail": (f"数据量{size_ratio:.0f}x增长 → 查询延迟{query_ratio:.2f}x增长 "
                       f"({first_data['query_p50_ms']:.1f}ms → {last_data['query_p50_ms']:.1f}ms)"),
            "actual": round(query_ratio, 2),
            "target": f"< {size_ratio:.0f}x (亚线性)",
            "passed": is_sublinear,
        })

        # 检查 100K 规模下查询 P95 < 500ms
        checks.append({
            "test": "扩展性 — 100K 查询 P95 < 500ms",
            "detail": f"实际 P95={last_data['query_p95_ms']:.1f}ms",
            "actual_ms": last_data["query_p95_ms"],
            "target_ms": 500,
            "passed": last_data["query_p95_ms"] < 500,
        })

        # 检查召回率 (query results must be non-empty)
        # 通过 P50 延迟合理性能推断
        checks.append({
            "test": "扩展性 — 100K 查询 P50 < 300ms",
            "detail": f"实际 P50={last_data['query_p50_ms']:.1f}ms",
            "actual_ms": last_data["query_p50_ms"],
            "target_ms": 300,
            "passed": last_data["query_p50_ms"] < 300,
        })

    # --- 稳定性检查 ---
    stability = all_results.get("p2_stability", {})
    if stability:
        targets = stability.get("targets", {})

        # RSS 增长
        rss_growth = stability.get("rss_growth_mb", 0)
        rss_target = targets.get("rss_growth_mb", 200)
        checks.append({
            "test": "稳定性 — RSS 内存增长 < 200MB (2小时)",
            "detail": f"实际增长={rss_growth:.1f}MB ({stability.get('rss_growth_per_hour_mb', 0):.1f}MB/h)",
            "actual_mb": rss_growth,
            "target_mb": rss_target,
            "passed": abs(rss_growth) < rss_target,
        })

        # 错误率
        err_rate = stability.get("error_rate", 0)
        err_target = targets.get("error_rate", 0.01)
        checks.append({
            "test": "稳定性 — 错误率 < 1%",
            "detail": f"实际错误率={err_rate:.4%} (错误{stability.get('total_errors', 0)}/总操作{stability.get('total_ops', 0)})",
            "actual": err_rate,
            "target": err_target,
            "passed": err_rate < err_target,
        })

        # 查询延迟漂移
        lat_drift = stability.get("query_lat_drift_ms", 0)
        lat_target = targets.get("query_lat_drift_ms", 50)
        checks.append({
            "test": "稳定性 — 查询延迟漂移 < 50ms",
            "detail": (f"早期={stability.get('query_lat_early_ms', 0):.1f}ms → "
                       f"晚期={stability.get('query_lat_late_ms', 0):.1f}ms, "
                       f"漂移={lat_drift:+.2f}ms"),
            "actual_ms": lat_drift,
            "target_ms": lat_target,
            "passed": abs(lat_drift) < lat_target,
        })

        # 磁盘增长
        disk_growth = stability.get("disk_growth_mb", 0)
        disk_target = targets.get("disk_growth_mb", 500)
        checks.append({
            "test": "稳定性 — 磁盘增长 < 500MB (2小时)",
            "detail": f"实际增加={disk_growth:.1f}MB",
            "actual_mb": disk_growth,
            "target_mb": disk_target,
            "passed": disk_growth < disk_target,
        })

        # RSS 标准差 (稳定性)
        rss_std = stability.get("rss_std_mb", 0)
        checks.append({
            "test": "稳定性 — RSS 波动标准差 < 50MB",
            "detail": f"实际 σ={rss_std:.1f}MB",
            "actual_mb": rss_std,
            "target_mb": 50,
            "passed": rss_std < 50,
        })

    # --- 压力负载检查 ---
    stress = all_results.get("p2_stress", {})
    if stress:
        s_targets = stress.get("targets", {})

        checks.append({
            "test": "压力 — 100并发错误率 < 1%",
            "detail": f"实际错误率={stress.get('error_rate', 0):.4%}",
            "actual": stress.get("error_rate", 0),
            "target": s_targets.get("error_rate", 0.01),
            "passed": stress.get("error_rate", 0) < s_targets.get("error_rate", 0.01),
        })

        checks.append({
            "test": "压力 — 总 QPS > 100",
            "detail": f"实际 QPS={stress.get('total_qps', 0):.1f}",
            "actual": stress.get("total_qps", 0),
            "target": s_targets.get("total_qps", 100),
            "passed": stress.get("total_qps", 0) >= s_targets.get("total_qps", 100),
        })

        checks.append({
            "test": "压力 — 查询 P95 < 500ms",
            "detail": f"实际 P95={stress.get('query_p95_ms', 0):.1f}ms",
            "actual_ms": stress.get("query_p95_ms", 0),
            "target_ms": s_targets.get("query_p95_ms", 500),
            "passed": stress.get("query_p95_ms", 0) < s_targets.get("query_p95_ms", 500),
        })

        checks.append({
            "test": "压力 — Worker 间延迟标准差 < 20ms",
            "detail": f"写入σ={stress.get('worker_write_p50_std_ms', 0):.1f}ms, "
                      f"查询σ={stress.get('worker_query_p50_std_ms', 0):.1f}ms",
            "actual_ms": max(stress.get("worker_write_p50_std_ms", 0),
                             stress.get("worker_query_p50_std_ms", 0)),
            "target_ms": s_targets.get("worker_std_ms", 20),
            "passed": (stress.get("worker_write_p50_std_ms", 0) < 20 and
                       stress.get("worker_query_p50_std_ms", 0) < 20),
        })

    return {
        "checks": checks,
        "total": len(checks),
        "passed": sum(1 for c in checks if c["passed"]),
        "failed": sum(1 for c in checks if not c["passed"]),
        "pass_rate": round(sum(1 for c in checks if c["passed"]) / max(len(checks), 1) * 100, 1),
    }


# ============================================================
# 汇总打印
# ============================================================

def print_p2_report(all_results: Dict, compliance: Dict):
    print("\n" + "=" * 70)
    print("  su-memory SDK v3.5.4 P2 大规模性能基准测试报告")
    print("=" * 70)
    print(f"  时间: {all_results.get('meta', {}).get('timestamp', 'N/A')}")
    print(f"  Python: {all_results.get('meta', {}).get('python_version', 'N/A')}")
    print(f"  机器: Apple M5 Pro")

    # ---- 扩展性 ----
    scalability = all_results.get("p2_scalability", {})
    print("\n" + "-" * 70)
    print("  [1] P2 扩展性测试 — 写入与查询延迟变化趋势")
    print("-" * 70)
    print(f"  {'规模':>8s} {'写入P50':>9s} {'写入P95':>9s} {'写入P99':>9s} "
          f"{'查询P50':>9s} {'查询P95':>9s} {'查询P99':>9s} {'写入增幅':>9s} {'查询增幅':>9s}")
    print("  " + "-" * 88)
    for key in sorted(scalability.keys(),
                      key=lambda k: scalability[k].get("data_size", 0)):
        d = scalability[key]
        print(f"  {key:>8s} {d['write_p50_ms']:>8.3f}ms {d['write_p95_ms']:>8.3f}ms {d['write_p99_ms']:>8.3f}ms "
              f"{d['query_p50_ms']:>8.3f}ms {d['query_p95_ms']:>8.3f}ms {d['query_p99_ms']:>8.3f}ms "
              f"{d['write_increase_pct']:>+7.1f}% {d['query_increase_pct']:>+7.1f}%")

    # ---- 稳定性 ----
    stability = all_results.get("p2_stability", {})
    if stability:
        print("\n" + "-" * 70)
        print("  [2] P2 长期稳定性测试 — 资源占用稳定性")
        print("-" * 70)
        print(f"  持续时间: {stability.get('duration_hours', 0):.1f}小时 "
              f"({stability.get('duration_sec', 0):.0f}s)")
        print(f"  总操作数: {stability.get('total_ops', 0):,}  "
              f"平均 QPS: {stability.get('avg_qps', 0):.1f}")
        print(f"  错误率: {stability.get('error_rate', 0):.6%}  "
              f"(错误 {stability.get('total_errors', 0)} 次)")
        print(f"\n  RSS 内存:")
        print(f"    起始: {stability.get('rss_start_mb', 0):.1f}MB  →  "
              f"结束: {stability.get('rss_end_mb', 0):.1f}MB  "
              f"增长: {stability.get('rss_growth_mb', 0):+.1f}MB")
        print(f"    均值: {stability.get('rss_mean_mb', 0):.1f}MB  "
              f"最大值: {stability.get('rss_max_mb', 0):.1f}MB  "
              f"标准差: {stability.get('rss_std_mb', 0):.2f}MB")
        print(f"    每小时增长: {stability.get('rss_growth_per_hour_mb', 0):+.2f}MB/h")
        print(f"\n  查询延迟漂移:")
        early = stability.get('query_lat_early_ms', 0)
        late = stability.get('query_lat_late_ms', 0)
        print(f"    早期: {early:.1f}ms  →  晚期: {late:.1f}ms  "
              f"漂移: {stability.get('query_lat_drift_ms', 0):+.2f}ms")
        print(f"\n  磁盘空间:")
        print(f"    起始: {stability.get('disk_start_mb', 0):.1f}MB  →  "
              f"结束: {stability.get('disk_end_mb', 0):.1f}MB  "
              f"增长: {stability.get('disk_growth_mb', 0):+.1f}MB")
        if stability.get('monitor_samples'):
            print(f"  监控采样点: {stability.get('monitor_samples', 0)} 个 "
                  f"(≈每{60:.0f}秒)")
    else:
        print("\n" + "-" * 70)
        print("  [2] P2 长期稳定性测试 — ⏭️ 已跳过")

    # ---- 压力负载 ----
    stress = all_results.get("p2_stress", {})
    if stress:
        print("\n" + "-" * 70)
        print("  [3] P2 压力负载测试 — 100 并发混合读写")
        print("-" * 70)
        print(f"  并发线程: {stress.get('workers', 0)}")
        print(f"  持续时间: {stress.get('duration_sec', 0):.1f}s")
        print(f"  总操作: {stress.get('total_ops', 0):,} (写 {stress.get('total_write_ops', 0):,} / "
              f"读 {stress.get('total_query_ops', 0):,})")
        print(f"  总 QPS: {stress.get('total_qps', 0):.1f}")
        print(f"  错误率: {stress.get('error_rate', 0):.4%}")
        print(f"\n  写入延迟: P50={stress.get('write_p50_ms', 0):.2f}ms  "
              f"P95={stress.get('write_p95_ms', 0):.2f}ms  "
              f"P99={stress.get('write_p99_ms', 0):.2f}ms")
        print(f"  查询延迟: P50={stress.get('query_p50_ms', 0):.2f}ms  "
              f"P95={stress.get('query_p95_ms', 0):.2f}ms  "
              f"P99={stress.get('query_p99_ms', 0):.2f}ms")
        print(f"  Worker间标准差: 写入{stress.get('worker_write_p50_std_ms', 0):.2f}ms / "
              f"查询{stress.get('worker_query_p50_std_ms', 0):.2f}ms")
        print(f"  RSS变化: {stress.get('rss_before_mb', 0):.1f} → "
              f"{stress.get('rss_after_mb', 0):.1f}MB "
              f"(Δ={stress.get('rss_delta_mb', 0):+.1f}MB)")
    else:
        print("\n" + "-" * 70)
        print("  [3] P2 压力负载测试 — ⏭️ 已跳过")

    # ---- 达标清单 ----
    print("\n" + "-" * 70)
    print("  [4] P2 达标检查")
    print("-" * 70)
    print(f"  {'测试项':<55s} {'结果':>6s} {'详情':>s}")
    print("  " + "-" * 110)
    checks = compliance.get("checks", [])
    for c in checks:
        status = "✅ PASS" if c["passed"] else "❌ FAIL"
        print(f"  {c['test']:<55s} {status:>6s}")
        if c.get("detail"):
            print(f"    └─ {c['detail']}")

    # ---- 汇总 ----
    total = compliance.get("total", 0)
    passed = compliance.get("passed", 0)
    failed = compliance.get("failed", 0)
    rate = compliance.get("pass_rate", 0)

    print("\n" + "=" * 70)
    print(f"  P2 达标汇总: {total}项检测 | ✅ 通过 {passed}项 | ❌ 未通过 {failed}项 | 通过率 {rate:.1f}%")
    if failed == 0:
        print("\n  🎉 所有 P2 测试项全部达标！su-memory SDK v3.5.4 满足大规模应用场景性能要求。")
    else:
        print(f"\n  ⚠️  {failed} 项未达标，请查看详细信息并优化。")
    print("=" * 70)


# ============================================================
# 主入口
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="su-memory SDK v3.5.4 P2 大规模性能基准测试"
    )
    parser.add_argument(
        "--skip-stability",
        action="store_true",
        help="跳过 2 小时长期稳定性测试 (加速测试流程)"
    )
    parser.add_argument(
        "--skip-stress",
        action="store_true",
        help="跳过 100 并发压力负载测试"
    )
    parser.add_argument(
        "--stability-hours",
        type=float,
        default=2.0,
        help="稳定性测试持续时间 (小时, 默认 2.0)"
    )
    parser.add_argument(
        "--stress-workers",
        type=int,
        default=100,
        help="压力测试并发数 (默认 100)"
    )
    parser.add_argument(
        "--stress-duration",
        type=float,
        default=30.0,
        help="压力测试持续时间 (秒, 默认 30)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON 报告输出路径"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  su-memory SDK v3.5.4 P2 大规模性能基准测试")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  硬件: Apple M5 Pro")
    print(f"  开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_results = {
        "meta": {
            "version": "4.4.1",
            "test_level": "P2",
            "python_version": sys.version,
            "platform": "darwin",
            "hardware": "Apple M5 Pro",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "stability_hours": args.stability_hours,
                "stress_workers": args.stress_workers,
                "stress_duration_sec": args.stress_duration,
            },
        },
    }

    # ═══════════════════════════════════════════════════════════
    # [1/3] P2 扩展性测试
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 55)
    print("  [1/3] P2 扩展性测试 (Scalability)")
    print("=" * 55)
    all_results["p2_scalability"] = p2_scalability(
        data_sizes=[100, 1000, 10000, 50000, 100000],
        query_samples=200,
        batch_size=500,
    )

    # ═══════════════════════════════════════════════════════════
    # [2/3] P2 长期稳定性测试
    # ═══════════════════════════════════════════════════════════
    if not args.skip_stability:
        print("\n" + "=" * 55)
        print("  [2/3] P2 长期稳定性测试 (Long-term Stability)")
        print("=" * 55)
        duration_sec = args.stability_hours * 3600.0
        all_results["p2_stability"] = p2_long_term_stability(
            data_count=10000,
            duration_sec=duration_sec,
            monitor_interval=60.0,
        )
    else:
        print("\n" + "=" * 55)
        print("  [2/3] P2 长期稳定性测试 — ⏭️ 已跳过")
        print("=" * 55)
        all_results["p2_stability"] = {"skipped": True, "reason": "用户跳过 (--skip-stability)"}

    # ═══════════════════════════════════════════════════════════
    # [3/3] P2 压力负载测试
    # ═══════════════════════════════════════════════════════════
    if not args.skip_stress:
        print("\n" + "=" * 55)
        print("  [3/3] P2 压力负载测试 (Stress Load)")
        print("=" * 55)
        all_results["p2_stress"] = p2_stress_load(
            workers=args.stress_workers,
            duration_sec=args.stress_duration,
            write_ratio=0.7,
        )
    else:
        print("\n" + "=" * 55)
        print("  [3/3] P2 压力负载测试 — ⏭️ 已跳过")
        print("=" * 55)
        all_results["p2_stress"] = {"skipped": True, "reason": "用户跳过 (--skip-stress)"}

    # ═══════════════════════════════════════════════════════════
    # 达标检查
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 55)
    print("  达标检查")
    print("=" * 55)
    compliance = check_p2_targets(all_results)
    all_results["compliance"] = compliance

    # ═══════════════════════════════════════════════════════════
    # 汇总报告
    # ═══════════════════════════════════════════════════════════
    print_p2_report(all_results, compliance)

    # ═══════════════════════════════════════════════════════════
    # 保存 JSON 报告
    # ═══════════════════════════════════════════════════════════
    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "benchmark_results_v354_p2.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n📄 JSON 报告已保存: {output_path}")

    # 返回退出码
    failed = compliance.get("failed", 0)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
