"""
MemScore 复合指标模块 (v3.5.5)
===============================
对标 SuperMemory MemoryBench 的 MemScore 三元组：
    accuracy% / latencyMs / contextTokens

提供 Memory Provider 的质量/延迟/成本三维度对比能力。

使用:
    >>> from benchmarks.memscore import MemScore
    >>> ms = MemScore(accuracy_pct=86.0, latency_ms=145, context_tokens=1823)
    >>> print(ms)  # 86% / 145ms / 1823tok
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemScore:
    """MemScore 复合指标 — 对标 SuperMemory MemoryBench

    三元组语义：在 latency_ms 延迟和 context_tokens 上下文的条件下，
              达到 accuracy_pct 准确率。
    """

    accuracy_pct: float       # 准确率百分比 (0-100)
    latency_ms: float         # 平均检索延迟 (ms)
    context_tokens: int       # 平均上下文 Token 数

    # 可选扩展维度
    recall_at_5: float = 0.0  # Recall@5
    mrr: float = 0.0          # Mean Reciprocal Rank
    throughput_qps: float = 0.0  # 写入吞吐量
    mem_usage_mb: float = 0.0    # 内存占用

    def __str__(self) -> str:
        return f"{self.accuracy_pct:.0f}% / {self.latency_ms:.0f}ms / {self.context_tokens}tok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy_pct": self.accuracy_pct,
            "latency_ms": self.latency_ms,
            "context_tokens": self.context_tokens,
            "recall_at_5": self.recall_at_5,
            "mrr": self.mrr,
            "throughput_qps": self.throughput_qps,
            "mem_usage_mb": self.mem_usage_mb,
            "display": str(self),
        }

    def compare(self, other: MemScore) -> MemScoreComparison:
        """与另一个 MemScore 对比"""
        return MemScoreComparison(
            self_score=self,
            other_score=other,
            accuracy_delta=self.accuracy_pct - other.accuracy_pct,
            latency_ratio=self.latency_ms / max(other.latency_ms, 1),
            token_ratio=self.context_tokens / max(other.context_tokens, 1),
        )

    @classmethod
    def from_benchmark_result(cls, result: dict[str, Any]) -> MemScore:
        """从 benchmark 结果字典构建 MemScore"""
        accuracy = result.get("accuracy", 0) * 100
        latency = result.get("avg_query_time_ms", 0)
        tokens = result.get("total_context_tokens", 0)
        return cls(
            accuracy_pct=round(accuracy, 1),
            latency_ms=round(latency, 1),
            context_tokens=int(tokens),
            recall_at_5=round(result.get("recall_at_5", 0) * 100, 1),
            mrr=round(result.get("mrr", 0), 4),
            throughput_qps=round(result.get("write_qps", 0), 1),
        )

    @classmethod
    def from_perf_benchmark(cls, perf: dict[str, Any]) -> MemScore:
        """从性能基准结果构建 MemScore"""
        # 性能基准关注延迟和吞吐量
        return cls(
            accuracy_pct=100.0,  # 性能测试默认 100% 正确
            latency_ms=round(perf.get("p50_ms", 0), 1),
            context_tokens=0,
            throughput_qps=round(perf.get("qps", 0), 1),
            mem_usage_mb=round(perf.get("rss_mb", 0), 1),
        )


@dataclass
class MemScoreComparison:
    """MemScore 对比结果"""

    self_score: MemScore
    other_score: MemScore
    accuracy_delta: float      # 正值 = self 更优
    latency_ratio: float       # <1 = self 更快
    token_ratio: float         # <1 = self 更省 Token

    @property
    def verdict(self) -> str:
        """综合判定"""
        advantages = 0
        if self.accuracy_delta > 1:
            advantages += 1
        if self.latency_ratio < 0.95:
            advantages += 1
        if self.token_ratio < 0.95:
            advantages += 1

        if advantages >= 2:
            return "🏆 WIN"
        elif advantages == 1:
            return "✅ COMPETITIVE"
        elif self.accuracy_delta > -2:
            return "≈ DRAW"
        else:
            return "❌ BEHIND"

    def __str__(self) -> str:
        return (
            f"{self.verdict} | "
            f"ΔACC={self.accuracy_delta:+.1f}% | "
            f"LAT={self.latency_ratio:.2f}x | "
            f"TOK={self.token_ratio:.2f}x"
        )


# ============================================================
# SOTA Competitor MemScores (静态基线)
# ============================================================

COMPETITOR_MEMSCORES: dict[str, MemScore] = {
    "Hindsight v5 (LongMemEval)": MemScore(
        accuracy_pct=91.4, latency_ms=300, context_tokens=2500,
    ),
    "Mem0 (LongMemEval)": MemScore(
        accuracy_pct=82.0, latency_ms=180, context_tokens=1800,
    ),
    "Zep (LongMemEval)": MemScore(
        accuracy_pct=79.0, latency_ms=200, context_tokens=2000,
    ),
    "GPT-4 Turbo (full ctx)": MemScore(
        accuracy_pct=72.0, latency_ms=450, context_tokens=8000,
    ),
}


def compare_to_competitors(our_score: MemScore) -> list[MemScoreComparison]:
    """将 su-memory 的 MemScore 与所有竞品对比"""
    comparisons = []
    for name, competitor in COMPETITOR_MEMSCORES.items():
        comp = our_score.compare(competitor)
        comp.other_name = name  # type: ignore
        comparisons.append(comp)
    return comparisons


def print_leaderboard(our_score: MemScore) -> None:
    """打印 MemScore 排行榜"""
    W = 70
    print("=" * W)
    print("  MemScore Leaderboard (对标 SuperMemory MemoryBench)")
    print("=" * W)
    print(f"  {'Provider':<30} {'MemScore':>25} {'vs su-memory':>12}")
    print("-" * W)

    # su-memory 自己
    print(f"  {'** su-memory v3.5.5 **':<30} {str(our_score):>25} {'—':>12}")

    # 竞品
    for name, competitor in COMPETITOR_MEMSCORES.items():
        comp = our_score.compare(competitor)
        print(f"  {name:<30} {str(competitor):>25} {comp.verdict:>12}")

    print("=" * W)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="su-memory MemScore 计算器")
    parser.add_argument("--accuracy", type=float, default=86.0, help="准确率 %")
    parser.add_argument("--latency", type=float, default=145.0, help="延迟 ms")
    parser.add_argument("--tokens", type=int, default=1823, help="上下文 Token 数")
    parser.add_argument("--compare", action="store_true", help="与竞品对比")
    args = parser.parse_args()

    score = MemScore(
        accuracy_pct=args.accuracy,
        latency_ms=args.latency,
        context_tokens=args.tokens,
    )
    print(f"\nsu-memory MemScore: {score}\n")

    if args.compare:
        print_leaderboard(score)
