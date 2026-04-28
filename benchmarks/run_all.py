#!/usr/bin/env python3
"""
su-memory Unified Benchmark Runner
===================================
Runs all benchmarks: LongMemEval, HotpotQA, BEIR
Generates consolidated report with competitor comparison.
"""
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from longmem_eval import LongMemEvalRunner
from hotpotqa import HotpotQARunner
from beir import BEIRRunner

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BENCHMARK_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_all_benchmarks():
    """Run complete benchmark suite."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 70)
    print("  su-memory v2.0.0 — Complete Benchmark Suite")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    results = {}
    
    # 1. LongMemEval
    print("\n" + "🔬 LongMemEval — Long-term Memory Evaluation")
    print("-" * 50)
    try:
        runner = LongMemEvalRunner()
        lm_result = runner.run(verbose=True)
        results["longmemeval"] = lm_result.to_dict()
        report = runner.format_report(lm_result)
        print(report)
        with open(os.path.join(RESULTS_DIR, f"longmemeval_{timestamp}.txt"), 'w') as f:
            f.write(report)
        with open(os.path.join(RESULTS_DIR, f"longmemeval_{timestamp}.json"), 'w') as f:
            json.dump(lm_result.to_dict(), f, indent=2)
    except Exception as e:
        print(f"  ❌ LongMemEval failed: {e}")
        results["longmemeval"] = {"error": str(e)}
    
    # 2. HotpotQA
    print("\n" + "🔗 HotpotQA — Multi-hop Reasoning")
    print("-" * 50)
    try:
        runner = HotpotQARunner()
        hq_result = runner.run(verbose=True)
        results["hotpotqa"] = hq_result.to_dict()
        report = runner.format_report(hq_result)
        print(report)
        with open(os.path.join(RESULTS_DIR, f"hotpotqa_{timestamp}.txt"), 'w') as f:
            f.write(report)
        with open(os.path.join(RESULTS_DIR, f"hotpotqa_{timestamp}.json"), 'w') as f:
            json.dump(hq_result.to_dict(), f, indent=2)
    except Exception as e:
        print(f"  ❌ HotpotQA failed: {e}")
        results["hotpotqa"] = {"error": str(e)}
    
    # 3. BEIR
    print("\n" + "📚 BEIR — Zero-shot Retrieval")
    print("-" * 50)
    try:
        runner = BEIRRunner()
        beir_results = runner.run(verbose=True)
        results["beir"] = {k: v.to_dict() for k, v in beir_results.items()}
        report = runner.format_report(beir_results)
        print(report)
        with open(os.path.join(RESULTS_DIR, f"beir_{timestamp}.txt"), 'w') as f:
            f.write(report)
        with open(os.path.join(RESULTS_DIR, f"beir_{timestamp}.json"), 'w') as f:
            json.dump(results["beir"], f, indent=2)
    except Exception as e:
        print(f"  ❌ BEIR failed: {e}")
        results["beir"] = {"error": str(e)}
    
    # Generate consolidated report
    consolidated = generate_consolidated_report(results, timestamp)
    print("\n" + consolidated)
    
    with open(os.path.join(RESULTS_DIR, f"consolidated_{timestamp}.txt"), 'w') as f:
        f.write(consolidated)
    with open(os.path.join(RESULTS_DIR, f"consolidated_{timestamp}.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Symlink latest
    for f in os.listdir(RESULTS_DIR):
        if f.startswith("consolidated_") and f.endswith(".txt"):
            latest = os.path.join(RESULTS_DIR, "LATEST.txt")
            if os.path.exists(latest):
                os.remove(latest)
            os.symlink(f, latest)
    
    print(f"\nResults saved to: {RESULTS_DIR}/")
    return results


def generate_consolidated_report(results: dict, timestamp: str) -> str:
    """Generate aggregated report with overall ranking."""
    lines = [
        "=" * 70,
        "  su-memory v2.0.0 — CONSOLIDATED BENCHMARK REPORT",
        f"  Date: {timestamp}",
        "=" * 70,
        "",
        "# 1. LongMemEval (Long-term Memory)",
    ]
    
    lm = results.get("longmemeval", {})
    if "error" not in lm:
        lines.extend([
            f"  Accuracy:  {lm.get('accuracy', 0):.2%}",
            f"  Recall@1:  {lm.get('recall_at_1', 0):.2%}",
            f"  Early Recall:  {lm.get('early_recall', 0):.2%}",
            f"  Late Recall:   {lm.get('late_recall', 0):.2%}",
        ])
    
    lines.extend(["", "# 2. HotpotQA (Multi-hop Reasoning)"])
    hq = results.get("hotpotqa", {})
    if "error" not in hq:
        lines.extend([
            f"  EM:    {hq.get('exact_match', 0):.2%}",
            f"  F1:    {hq.get('f1_score', 0):.2%}",
        ])
    
    lines.extend(["", "# 3. BEIR (Zero-shot Retrieval)"])
    beir = results.get("beir", {})
    if "error" not in beir:
        avg_ndcg = sum(r.get("ndcg_at_10", 0) for r in beir.values()) / len(beir) if beir else 0
        lines.extend([
            f"  Avg NDCG@10: {avg_ndcg:.3f}",
            "",
            "  Per-dataset:",
        ])
        for ds, r in beir.items():
            lines.append(f"    {ds:<15} NDCG@10={r.get('ndcg_at_10', 0):.3f} MAP={r.get('map_score', 0):.3f}")
    
    lines.extend([
        "",
        "# 4. Leaderboard Position",
        "",
        "  | Benchmark    | su-memory | Best Competitor | Rank |",
        "  |-------------|-----------|-----------------|------|",
    ])
    
    if "error" not in lm:
        su_acc = lm.get("accuracy", 0)
        comp_acc = 0.523  # Hindsight
        rank = "🥇 #1" if su_acc > comp_acc else f"#{2 if su_acc > 0.481 else 3}"
        lines.append(f"  | LongMemEval | {su_acc:.1%}    | {comp_acc:.1%} (Hindsight)  | {rank}  |")
    
    if "error" not in hq:
        su_em = hq.get("exact_match", 0)
        comp_em = 0.501  # Hindsight
        rank = "🥇 #1" if su_em > comp_em else f"#{2 if su_em > 0.482 else 3}"
        lines.append(f"  | HotpotQA    | {su_em:.1%}    | {comp_em:.1%} (Hindsight)  | {rank}  |")
    
    if "error" not in beir and beir:
        su_beir = avg_ndcg
        comp_beir = 0.499  # SPLADE++
        rank = "🥈 #2" if su_beir > 0.466 else "🥉 #3"
        lines.append(f"  | BEIR        | {su_beir:.3f}   | {comp_beir:.3f} (SPLADE++)  | {rank}  |")
    
    lines.extend([
        "",
        "=" * 70,
        "  Report generated by su-memory benchmark suite v2.0.0",
        f"  https://github.com/su-memory/su-memory-sdk",
        "=" * 70,
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    run_all_benchmarks()
