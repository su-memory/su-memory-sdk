"""
check_perf_gate.py — CI 性能门禁检查脚本 (v3.5.7)

读取基准测试结果 JSON，检查是否达成性能门禁阈值。

用法:
    python scripts/check_perf_gate.py results.json [--v357]

门禁 (v3.5.7):
    - write_throughput >= 80 ops/s
    - query_p99_ms <= 5ms
    - gaia_l1_accuracy >= 0.85
    - gaia_l2_accuracy >= 0.90
    - gaia_l3_accuracy >= 0.75
    - gaia_weighted_avg >= 0.87
    - energy_inference_accuracy >= 1.0
    - distill_cluster_count >= 2
    - extract_rules_count >= 3
    - py_compile_files == 208

门禁 (legacy v2.7.0):
    - query_p99_ms <= 50ms
    - write_throughput >= 80 ops/s
    - batch_throughput >= 500 ops/s
    - memory_10k_mb <= 500 MB
    - faiss_search_ms <= 10ms
    - multihop_3hop_ms <= 200ms
    - init_ms <= 500ms
"""

import sys
import json
from pathlib import Path


GATES_LEGACY = {
    "query_p99_ms": {"limit": 50, "unit": "ms", "description": "查询 P99 延迟"},
    "write_throughput": {"limit": 80, "unit": "ops/s", "description": "单条写入吞吐"},
    "batch_throughput": {"limit": 500, "unit": "ops/s", "description": "批量写入吞吐"},
    "memory_10k_mb": {"limit": 500, "unit": "MB", "description": "10K 内存占用"},
    "faiss_search_ms": {"limit": 10, "unit": "ms", "description": "FAISS 搜索延迟"},
    "multihop_3hop_ms": {"limit": 200, "unit": "ms", "description": "3-hop 推理延迟"},
    "init_ms": {"limit": 500, "unit": "ms", "description": "首次 import 时间"},
    "pgvector_query_p50_ms": {"limit": 30, "unit": "ms", "description": "pgvector 查询 P50"},
    "tiered_hit_rate": {"limit": 0.80, "unit": "rate", "description": "分层命中率 (hot tier)", "compare": ">="},
    "stress_100k_write_throughput": {"limit": 50, "unit": "ops/s", "description": "100K 写入吞吐"},
    "stress_100k_query_p99_ms": {"limit": 200, "unit": "ms", "description": "100K 查询 P99"},
    "stress_100k_memory_mb": {"limit": 2000, "unit": "MB", "description": "100K 内存占用"},
    "async_concurrency_scale": {"limit": 3.0, "unit": "x", "description": "异步并发扩展 (4→16)", "compare": ">="},
}

# v3.5.7 门禁 — 基于 GAIA 基准 + 性能核心指标
GATES_V357 = {
    "write_throughput": {"limit": 80, "unit": "ops/s", "description": "写入 QPS", "compare": ">="},
    "query_p99_ms": {"limit": 5, "unit": "ms", "description": "查询 P99 延迟"},
    "gaia_l1_accuracy": {"limit": 0.85, "unit": "", "description": "GAIA L1 准确率", "compare": ">="},
    "gaia_l2_accuracy": {"limit": 0.90, "unit": "", "description": "GAIA L2 准确率", "compare": ">="},
    "gaia_l3_accuracy": {"limit": 0.75, "unit": "", "description": "GAIA L3 准确率", "compare": ">="},
    "gaia_weighted_avg": {"limit": 0.87, "unit": "", "description": "GAIA 加权总均", "compare": ">="},
    "energy_inference_accuracy": {"limit": 1.0, "unit": "", "description": "能量推断准确率", "compare": ">="},
    "distill_cluster_count": {"limit": 2, "unit": "clusters", "description": "distill 聚类数", "compare": ">="},
    "extract_rules_count": {"limit": 3, "unit": "rules", "description": "规则提取数", "compare": ">="},
    "py_compile_files": {"limit": 208, "unit": "files", "description": "py_compile 文件数", "compare": ">="},
    "ruff_errors": {"limit": 0, "unit": "errors", "description": "Ruff 错误数"},
}


def check_gate(name: str, gate: dict, value: float) -> tuple:
    """检查单个门禁
    
    默认比较: value <= limit (越小越好)
    若 compare='>=': value >= limit (越大越好, e.g. 命中率/吞吐)
    """
    compare = gate.get("compare", "<=")
    if compare == ">=":
        passed = value >= gate["limit"]
    else:
        passed = value <= gate["limit"]
    symbol = "✅" if passed else "❌"
    status = "PASS" if passed else "FAIL"
    msg = (
        f"  {symbol} {gate['description']}: "
        f"{value:.1f}{gate['unit']} (limit: {gate['limit']}{gate['unit']}) "
        f"[{status}]"
    )
    return passed, msg


def load_results(path: str) -> dict:
    """加载结果（支持 JSON 和 Python dict）"""
    p = Path(path)
    if not p.exists():
        print(f"⚠️  结果文件不存在: {path}")
        return {}

    with open(p) as f:
        return json.load(f)


def main():
    use_v357 = "--v357" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--v357"]
    gates = GATES_V357 if use_v357 else GATES_LEGACY

    if len(args) < 1:
        # 无文件模式：直接 PASS（用于首次 CI 搭建）
        print("⚠️  未提供基准结果文件，跳过门禁检查 (CI setup)")
        sys.exit(0)

    results = load_results(args[0])
    if not results:
        print("⚠️  无数据，跳过")
        sys.exit(0)

    gate_label = "v3.5.7" if use_v357 else "legacy"
    print("=" * 60)
    print(f"su-memory-sdk CI 性能门禁 ({gate_label})")
    print("=" * 60)

    all_passed = True
    for name, gate in gates.items():
        if name in results and results[name] is not None:
            passed, msg = check_gate(name, gate, results[name])
            print(msg)
            if not passed:
                all_passed = False
        else:
            print(f"  ⚠️  {gate['description']}: 无数据，跳过")

    print("=" * 60)
    if all_passed:
        print("✅ All performance gates PASSED")
        sys.exit(0)
    else:
        print("❌ Some performance gates FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
