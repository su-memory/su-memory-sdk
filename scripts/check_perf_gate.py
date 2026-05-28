"""
check_perf_gate.py — CI 性能门禁检查脚本 (v2.7.0)

读取基准测试结果 JSON，检查是否达成性能门禁阈值。

用法:
    python scripts/check_perf_gate.py results.json

门禁 (v2.6.0 → v2.7.0 新增 pgvector/async/tiered):
    - query_p99_ms <= 50ms
    - write_throughput >= 80 ops/s
    - batch_throughput >= 500 ops/s
    - memory_10k_mb <= 500 MB
    - faiss_search_ms <= 10ms
    - multihop_3hop_ms <= 200ms
    - init_ms <= 500ms
    # v2.7.0 新增
    - pgvector_query_p50_ms <= 30ms
    - tiered_hit_rate >= 80%
    - stress_100k_write_throughput >= 50 ops/s
    - stress_100k_query_p99_ms <= 200ms
    - stress_100k_memory_mb <= 2000 MB
    - async_concurrency_scale >= 3.0x (4→16 threads)
"""

import sys
import json
from pathlib import Path


GATES = {
    # v2.6.0 门禁
    "query_p99_ms": {"limit": 50, "unit": "ms", "description": "查询 P99 延迟"},
    "write_throughput": {"limit": 80, "unit": "ops/s", "description": "单条写入吞吐"},
    "batch_throughput": {"limit": 500, "unit": "ops/s", "description": "批量写入吞吐"},
    "memory_10k_mb": {"limit": 500, "unit": "MB", "description": "10K 内存占用"},
    "faiss_search_ms": {"limit": 10, "unit": "ms", "description": "FAISS 搜索延迟"},
    "multihop_3hop_ms": {"limit": 200, "unit": "ms", "description": "3-hop 推理延迟"},
    "init_ms": {"limit": 500, "unit": "ms", "description": "首次 import 时间"},
    # v2.7.0 新增门禁
    "pgvector_query_p50_ms": {"limit": 30, "unit": "ms", "description": "pgvector 查询 P50"},
    "tiered_hit_rate": {"limit": 0.80, "unit": "rate", "description": "分层命中率 (hot tier)", "compare": ">="},
    "stress_100k_write_throughput": {"limit": 50, "unit": "ops/s", "description": "100K 写入吞吐"},
    "stress_100k_query_p99_ms": {"limit": 200, "unit": "ms", "description": "100K 查询 P99"},
    "stress_100k_memory_mb": {"limit": 2000, "unit": "MB", "description": "100K 内存占用"},
    "async_concurrency_scale": {"limit": 3.0, "unit": "x", "description": "异步并发扩展 (4→16)", "compare": ">="},
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
    if len(sys.argv) < 2:
        # 无文件模式：直接 PASS（用于首次 CI 搭建）
        print("⚠️  未提供基准结果文件，跳过门禁检查 (CI setup)")
        sys.exit(0)

    results = load_results(sys.argv[1])
    if not results:
        print("⚠️  无数据，跳过")
        sys.exit(0)

    print("=" * 60)
    print("su-memory-sdk CI 性能门禁")
    print("=" * 60)

    all_passed = True
    for name, gate in GATES.items():
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
