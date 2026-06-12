#!/usr/bin/env python3
"""
su-memory v3.5.5 — 持续性能监控 (Performance Monitor)
======================================================

对核心操作 (add/query/forget) 进行微基准测试，记录延迟/吞吐历史，
支持与上一次运行对比，自动检测性能回归 (退化 >20% 标记 WARNING)。

使用:
    # 运行基准并保存结果
    python benchmarks/perf_monitor.py

    # 与上一次运行对比
    python benchmarks/perf_monitor.py --compare

    # 指定输出目录
    python benchmarks/perf_monitor.py --output results/

集成到 CI:
    python benchmarks/perf_monitor.py --compare --ci
    # --ci 模式: 回归时 exit(1), 输出 GitHub Actions 格式

门禁:
    - add 吞吐: >= 80 ops/s
    - query P99: <= 50ms
    - forget: <= 5ms
    - 回归阈值: 任一项退化 >20% → WARNING
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
import time
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ═══════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════


@dataclass
class PerfMetric:
    """单个性能指标。"""
    name: str
    value: float
    unit: str
    gate: float | None = None          # 门禁阈值 (PASS/FAIL)
    gate_direction: str = "gte"        # "gte"=大于等于, "lte"=小于等于
    baseline_value: float | None = None
    regression_pct: float | None = None

    @property
    def gate_status(self) -> str:
        if self.gate is None:
            return "N/A"
        if self.gate_direction == "gte":
            return "PASS" if self.value >= self.gate else "FAIL"
        return "PASS" if self.value <= self.gate else "FAIL"

    @property
    def regression_status(self) -> str:
        if self.regression_pct is None:
            return "N/A"
        if self.regression_pct > 20:
            return "⚠️  WARN"
        if self.regression_pct > 10:
            return "🟡 MARGINAL"
        return "✅ OK"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "gate": self.gate,
            "gate_status": self.gate_status,
            "baseline": self.baseline_value,
            "regression_pct": self.regression_pct,
            "regression_status": self.regression_status,
        }


@dataclass
class PerfSnapshot:
    """一次性能快照。"""
    timestamp: str
    python_version: str
    platform: str
    metrics: list[PerfMetric] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "python_version": self.python_version,
            "platform": self.platform,
            "metrics": [m.to_dict() for m in self.metrics],
        }

    @classmethod
    def from_dict(cls, d: dict) -> PerfSnapshot:
        metrics = [
            PerfMetric(
                name=m["name"], value=m["value"], unit=m["unit"],
                gate=m.get("gate"), gate_direction=m.get("gate_direction", "gte"),
                baseline_value=m.get("baseline"), regression_pct=m.get("regression_pct"),
            )
            for m in d.get("metrics", [])
        ]
        return cls(
            timestamp=d["timestamp"],
            python_version=d["python_version"],
            platform=d["platform"],
            metrics=metrics,
        )

    def compare(self, baseline: PerfSnapshot) -> list[PerfMetric]:
        """与基线对比，计算回归百分比。"""
        baseline_map = {m.name: m.value for m in baseline.metrics}
        compared = []
        for m in self.metrics:
            base_val = baseline_map.get(m.name)
            if base_val is not None and base_val > 0:
                # 吞吐量指标: 值下降 = 退化 (正回归%)
                # 延迟指标: 值上升 = 退化 (正回归%)
                if m.unit in ("ops/s",):
                    regression = (base_val - m.value) / base_val * 100
                else:
                    regression = (m.value - base_val) / base_val * 100
                m.baseline_value = round(base_val, 2)
                m.regression_pct = round(regression, 2)
            else:
                m.baseline_value = None
                m.regression_pct = None
            compared.append(m)
        return compared


# ═══════════════════════════════════════════════════════════════
# Micro-Benchmarks
# ═══════════════════════════════════════════════════════════════


def _create_client(storage_dir: str):
    """创建 SuMemoryLitePro 客户端 (无向量模式, 极速)。"""
    from su_memory import SuMemoryLitePro
    return SuMemoryLitePro(
        storage_path=storage_dir,
        enable_vector=False,
        enable_graph=False,
        enable_temporal=False,
        enable_session=False,
        enable_prediction=False,
        enable_explainability=False,
    )


def bench_add_throughput(storage_dir: str) -> float:
    """测量 add 吞吐量 (ops/s)。"""
    client = _create_client(storage_dir)
    COUNT = 500
    start = time.perf_counter()
    for i in range(COUNT):
        client.add(f"perf monitor test memory entry number {i:06d}")
    elapsed = time.perf_counter() - start
    return COUNT / elapsed


def bench_query_latency(storage_dir: str) -> dict[str, float]:
    """测量 query 延迟分布 (P50/P95/P99, ms)。"""
    client = _create_client(storage_dir)
    # 预热数据 (如果还没有)
    if client.count() == 0:
        for i in range(500):
            client.add(f"query latency test memory {i:06d}")

    SAMPLES = 200
    latencies = []
    for i in range(SAMPLES):
        query = f"memory {i % 500}"
        t0 = time.perf_counter()
        results = client.query(query, top_k=10)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if len(results) == 0:
            continue  # 跳过空结果
        latencies.append(elapsed_ms)

    latencies.sort()
    return {
        "p50_ms": statistics.median(latencies) if latencies else 0,
        "p95_ms": latencies[int(len(latencies) * 0.95)] if latencies else 0,
        "p99_ms": latencies[int(len(latencies) * 0.99)] if latencies else 0,
        "avg_ms": sum(latencies) / len(latencies) if latencies else 0,
        "max_ms": max(latencies) if latencies else 0,
    }


def bench_forget_latency(storage_dir: str) -> dict[str, float]:
    """测量 forget 延迟。"""
    client = _create_client(storage_dir)
    # 确保有数据可删
    if client.count() < 10:
        for i in range(50):
            client.add(f"forget test memory {i:06d}")

    # 获取一些 memory ID
    memories = client.get_all_memories() if hasattr(client, 'get_all_memories') else []
    if not memories:
        # Fallback: 使用内部 _memories
        memories = [{"id": m.id} for m in getattr(client, '_memories', [])]

    ids_to_forget = [m["id"] for m in memories[:20]] if memories else []

    latencies = []
    for mid in ids_to_forget:
        t0 = time.perf_counter()
        client.forget(mid)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

    if not latencies:
        return {"avg_ms": 0, "max_ms": 0}

    return {
        "avg_ms": sum(latencies) / len(latencies),
        "max_ms": max(latencies),
    }


def bench_memory_usage(storage_dir: str) -> dict[str, float]:
    """测量内存占用 (1K memories)。"""
    import sys as _sys
    client = _create_client(storage_dir)

    # 清除并重新填充 1K 条
    client.clear()
    for i in range(1000):
        client.add(f"memory usage test entry {i:06d} with some extra words for realism")

    # 估算内存 (粗略)
    import gc
    gc.collect()
    # 使用 sys.getsizeof 无法准确测量，改用 RSS 估算
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS: bytes, Linux: KB
        rss_mb = rss / (1024 * 1024) if _sys.platform == "darwin" else rss / 1024
    except ImportError:
        rss_mb = 0

    return {"rss_mb": round(rss_mb, 2), "n_memories": client.count()}


# ═══════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════


def run_all_benchmarks(storage_dir: str) -> PerfSnapshot:
    """运行全部微基准并返回 PerfSnapshot。"""
    import platform

    metrics = []

    # 1. Add throughput
    try:
        tps = bench_add_throughput(storage_dir)
        metrics.append(PerfMetric(
            name="add_throughput", value=round(tps, 2), unit="ops/s", gate=80, gate_direction="gte",
        ))
    except Exception as e:
        metrics.append(PerfMetric(
            name="add_throughput", value=0, unit="ops/s", gate=80, gate_direction="gte",
        ))
        print(f"  ⚠️  add_throughput error: {e}")

    # 2. Query latency
    try:
        qlat = bench_query_latency(storage_dir)
        for key, gate_val in [("p50_ms", None), ("p95_ms", None), ("p99_ms", 50), ("avg_ms", None)]:
            metrics.append(PerfMetric(
                name=f"query_{key}", value=round(qlat.get(key, 0), 3),
                unit="ms", gate=gate_val, gate_direction="lte",
            ))
    except Exception as e:
        metrics.append(PerfMetric(
            name="query_p99_ms", value=999, unit="ms", gate=50, gate_direction="lte",
        ))
        print(f"  ⚠️  query_latency error: {e}")

    # 3. Forget latency
    try:
        flat = bench_forget_latency(storage_dir)
        metrics.append(PerfMetric(
            name="forget_avg_ms", value=round(flat.get("avg_ms", 0), 3),
            unit="ms", gate=5, gate_direction="lte",
        ))
    except Exception as e:
        metrics.append(PerfMetric(
            name="forget_avg_ms", value=999, unit="ms", gate=5, gate_direction="lte",
        ))
        print(f"  ⚠️  forget_latency error: {e}")

    # 4. Memory usage
    try:
        musage = bench_memory_usage(storage_dir)
        metrics.append(PerfMetric(
            name="rss_mb_1k", value=round(musage.get("rss_mb", 0), 2),
            unit="MB", gate=5, gate_direction="lte",
        ))
    except Exception as e:
        metrics.append(PerfMetric(
            name="rss_mb_1k", value=999, unit="MB", gate=5, gate_direction="lte",
        ))
        print(f"  ⚠️  memory_usage error: {e}")

    return PerfSnapshot(
        timestamp=datetime.now().isoformat(),
        python_version=platform.python_version(),
        platform=platform.platform(),
        metrics=metrics,
    )


# ═══════════════════════════════════════════════════════════════
# Persistence
# ═══════════════════════════════════════════════════════════════


def get_history_dir(output_dir: str | None = None) -> Path:
    """获取性能历史目录。"""
    base = Path(output_dir) if output_dir else Path(__file__).parent / "results" / "perf_history"
    base.mkdir(parents=True, exist_ok=True)
    return base


def load_baseline(history_dir: Path) -> PerfSnapshot | None:
    """加载最近一次性能快照作为基线。"""
    files = sorted(history_dir.glob("perf_*.json"), reverse=True)
    for f in files:
        try:
            with open(f) as fp:
                return PerfSnapshot.from_dict(json.load(fp))
        except Exception:
            continue
    return None


def save_snapshot(snapshot: PerfSnapshot, history_dir: Path) -> str:
    """保存性能快照。"""
    filename = f"perf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = history_dir / filename
    with open(filepath, "w") as f:
        json.dump(snapshot.to_dict(), f, indent=2, ensure_ascii=False)
    return str(filepath)


# ═══════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════


def print_report(snapshot: PerfSnapshot, baseline: PerfSnapshot | None = None, ci_mode: bool = False) -> int:
    """
    打印性能报告。

    Returns:
        exit_code: 0=OK, 1=回归检测到
    """
    W = 85
    has_regression = False
    has_gate_failure = False

    print("\n" + "=" * W)
    print("  su-memory Performance Monitor Report")
    print("=" * W)
    print(f"  Timestamp: {snapshot.timestamp}")
    print(f"  Python:    {snapshot.python_version}")
    print(f"  Platform:  {snapshot.platform}")

    if baseline:
        print(f"  Baseline:  {baseline.timestamp}")

    print("-" * W)

    if baseline:
        print(f"  {'Metric':<22} {'Value':>10} {'Unit':>6} {'Gate':>7} "
              f"{'Baseline':>10} {'Δ%':>8} {'Status'}")
    else:
        print(f"  {'Metric':<22} {'Value':>10} {'Unit':>6} {'Gate':>7} {'Status'}")

    print("-" * W)

    for m in snapshot.metrics:
        gate_str = f"{m.gate_status}"
        if baseline and m.baseline_value is not None:
            delta_str = f"{m.regression_pct:+.1f}%" if m.regression_pct is not None else "—"
            status = m.regression_status if m.regression_pct is not None else "NEW"
            print(f"  {m.name:<22} {m.value:>10.2f} {m.unit:>6} {gate_str:>7} "
                  f"{m.baseline_value:>10.2f} {delta_str:>8} {status}")
            if m.regression_pct is not None and m.regression_pct > 20:
                has_regression = True
        else:
            print(f"  {m.name:<22} {m.value:>10.2f} {m.unit:>6} {gate_str:>7} {'NEW'}")

        if m.gate_status == "FAIL":
            has_gate_failure = True

    print("=" * W)

    # Summary
    if has_regression:
        print("  ⚠️  PERFORMANCE REGRESSION DETECTED (>20% degradation)")
    if has_gate_failure:
        print("  ❌ GATE FAILURE — 性能指标未达门禁标准")

    if not has_regression and not has_gate_failure:
        print("  ✅ All metrics within acceptable range")

    print("=" * W + "\n")

    # CI mode annotations
    if ci_mode:
        if has_regression:
            print("::warning::Performance regression detected — check perf_monitor output")
        if has_gate_failure:
            print("::error::Gate failure — performance below threshold")

    return 1 if (has_regression or has_gate_failure) else 0


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════


def main():
    import argparse
    parser = argparse.ArgumentParser(description="su-memory 持续性能监控")
    parser.add_argument("--compare", action="store_true", help="与上一次运行对比")
    parser.add_argument("--ci", action="store_true", help="CI 模式 (回归时 exit(1), GitHub 注释)")
    parser.add_argument("--output", type=str, default=None, help="性能历史目录")
    args = parser.parse_args()

    d = tempfile.mkdtemp(prefix="su_memory_perf_")
    try:
        print(f"\n  🔬 运行性能微基准... (temp: {d})")
        snapshot = run_all_benchmarks(d)

        # 始终保存
        history_dir = get_history_dir(args.output)
        filepath = save_snapshot(snapshot, history_dir)
        print(f"  📁 快照已保存: {filepath}")

        # 加载基线对比
        baseline = None
        if args.compare:
            baseline = load_baseline(get_history_dir(args.output))
            if baseline and baseline.timestamp != snapshot.timestamp:
                snapshot.metrics = snapshot.compare(baseline)
            else:
                print("  ℹ️  无基线可对比 (首次运行)")

        exit_code = print_report(snapshot, baseline, ci_mode=args.ci)
        sys.exit(exit_code)

    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
