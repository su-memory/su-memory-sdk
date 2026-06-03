#!/usr/bin/env python3
"""su-memory SDK 统一基准评测套件
====================================
支持 LongMemEval / LoCoMo / ConvoMem 三大基准 × 双嵌入后端
（Ollama bge-m3、sentence-transformers all-MiniLM-L6-v2）。

用法示例::

    # 完整评测（三大基准 × 双后端）
    python benchmarks/run_all.py --benchmarks all --backends both

    # 指定基准和后端
    python benchmarks/run_all.py --benchmarks longmemeval,convomem --backends ollama

    # 快速验证模式（少量数据）
    python benchmarks/run_all.py --benchmarks all --backends sbert --quick

    # 自定义输出
    python benchmarks/run_all.py --benchmarks all --backends both \\
        --output results/full_run.json --report BENCHMARK_REPORT.md
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# 包导入兼容：既支持 ``python benchmarks/run_all.py``，
# 也支持 ``python -m benchmarks.run_all``。
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (_REPO_ROOT, _THIS_DIR):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from benchmarks.config import (  # noqa: E402  pylint: disable=wrong-import-position
    BACKENDS,
    COMPETITOR_SCORES,
    BenchmarkResult,
)


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

VERSION = "3.5.0"
ALL_BENCHMARKS: tuple[str, ...] = ("longmemeval", "locomo", "convomem")
ALL_BACKENDS: tuple[str, ...] = ("ollama", "sbert", "sbert-mpnet", "llama-cpp", "openai")

RESULTS_DIR = _THIS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 调度器
# ---------------------------------------------------------------------------

class UnifiedBenchmarkSuite:
    """统一基准评测调度器。

    Args:
        benchmarks: 要运行的基准，子集 ``{"longmemeval", "locomo", "convomem"}``。
        backends:   要使用的嵌入后端，子集 ``{"ollama", "sbert"}``。
        quick:      快速模式，限制数据量便于冒烟测试。
        verbose:    详细日志开关。
        storage_root: su-memory 持久化根目录。
    """

    def __init__(
        self,
        benchmarks: list[str],
        backends: list[str],
        quick: bool = False,
        verbose: bool = False,
        storage_root: str = "/tmp/su-memory-bench",
    ) -> None:
        self.benchmarks: list[str] = list(benchmarks)
        self.backends: list[str] = list(backends)
        self.quick: bool = quick
        self.verbose: bool = verbose
        self.storage_root: str = storage_root

        self.results: dict[str, dict[str, Any]] = {}
        self.errors: dict[str, str] = {}
        self.timings: dict[str, float] = {}

        unknown_b = [b for b in self.benchmarks if b not in ALL_BENCHMARKS]
        if unknown_b:
            raise ValueError(
                f"Unknown benchmarks {unknown_b}. Valid: {list(ALL_BENCHMARKS)}"
            )
        for be in self.backends:
            if be not in BACKENDS:
                raise ValueError(
                    f"Unknown backend '{be}'. Available: {list(BACKENDS.keys())}"
                )

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def run(self) -> dict[str, dict[str, Any]]:
        """依次运行所有 ``backend × benchmark`` 组合。

        Returns:
            ``{f"{benchmark}_{backend}": result_dict}`` 形式的字典。
        """

        for backend in self.backends:
            cfg = BACKENDS[backend]
            print()
            print("─" * 72)
            print(f"  Backend: {cfg['name']}  (dim={cfg['dimension']})")
            print("─" * 72)
            for bench in self.benchmarks:
                key = f"{bench}_{backend}"
                if self.verbose:
                    print(f"\n▶ {bench} × {backend} ...")
                t0 = time.time()
                try:
                    result = self._run_single(bench, backend)
                    self.results[key] = result
                except Exception as exc:  # noqa: BLE001
                    self.errors[key] = str(exc)
                    if self.verbose:
                        traceback.print_exc()
                    print(f"  ❌ {bench} × {backend} failed: {exc}")
                self.timings[key] = round(time.time() - t0, 2)
        return self.results

    # ------------------------------------------------------------------
    # 单次评测分发
    # ------------------------------------------------------------------

    def _run_single(self, benchmark: str, backend: str) -> dict[str, Any]:
        """运行单个 ``benchmark × backend`` 组合并归一化输出。

        Returns:
            ``{"accuracy": float, ... , "raw": <BenchmarkResult.to_dict()>}``。
        """

        if benchmark == "longmemeval":
            return self._run_longmemeval(backend)
        if benchmark == "locomo":
            return self._run_locomo(backend)
        if benchmark == "convomem":
            return self._run_convomem(backend)
        raise ValueError(f"Unknown benchmark '{benchmark}'")

    # ---- LongMemEval -------------------------------------------------

    def _run_longmemeval(self, backend: str) -> dict[str, Any]:
        """适配 ``benchmarks.longmem_eval.run_backend`` 函数式 API。"""

        from benchmarks.longmem_eval import (
            DEFAULT_CHUNK_CHARS,
            load_longmemeval,
            run_backend,
        )

        split = "oracle" if self.quick else "s"
        max_q = 10 if self.quick else None

        samples = load_longmemeval(split=split, verbose=self.verbose)
        if max_q is not None:
            samples = samples[:max_q]

        storage_root = os.path.join(self.storage_root, "longmem")
        result, _diag = run_backend(
            backend=backend,
            samples=samples,
            storage_root=storage_root,
            top_k=5,
            chunk_chars=DEFAULT_CHUNK_CHARS,
            verbose=self.verbose,
        )
        return self._wrap_single(result, extra={
            "split": split,
            "max_questions": max_q,
            "samples_evaluated": len(samples),
            "dimensions": dict(result.dimension_scores),
        })

    # ---- LoCoMo ------------------------------------------------------

    def _run_locomo(self, backend: str) -> dict[str, Any]:
        """LoCoMoRunner 返回 ``{task: BenchmarkResult}``，需聚合为单条记录。"""

        from benchmarks.locomo_eval import LoCoMoRunner

        runner = LoCoMoRunner(
            backend=backend,
            storage_path=os.path.join(self.storage_root, f"locomo-{backend}"),
            verbose=self.verbose,
        )
        max_conv = 1 if self.quick else None
        task_results = runner.run(tasks=None, max_conversations=max_conv)

        # 聚合：以 total_questions 加权平均 accuracy / f1，平均时延
        total_q = sum(r.total_questions for r in task_results.values())
        total_correct = sum(r.correct for r in task_results.values())
        if total_q:
            f1_weighted = sum(r.f1_score * r.total_questions for r in task_results.values()) / total_q
            r1_weighted = sum(r.recall_at_1 * r.total_questions for r in task_results.values()) / total_q
            r5_weighted = sum(r.recall_at_5 * r.total_questions for r in task_results.values()) / total_q
            rouge_weighted = sum(r.rouge_l * r.total_questions for r in task_results.values()) / total_q
            bleu_weighted = sum(r.bleu * r.total_questions for r in task_results.values()) / total_q
            qms = sum(r.avg_query_time_ms * r.total_questions for r in task_results.values()) / total_q
            ams = sum(r.avg_add_time_ms * r.total_questions for r in task_results.values()) / total_q
        else:
            f1_weighted = r1_weighted = r5_weighted = rouge_weighted = bleu_weighted = qms = ams = 0.0

        accuracy = (total_correct / total_q) if total_q else 0.0

        aggregated = BenchmarkResult(
            benchmark_name="locomo",
            backend=backend,
            total_questions=total_q,
            correct=total_correct,
            accuracy=accuracy,
            f1_score=f1_weighted,
            recall_at_1=r1_weighted,
            recall_at_5=r5_weighted,
            bleu=bleu_weighted,
            rouge_l=rouge_weighted,
            avg_query_time_ms=qms,
            avg_add_time_ms=ams,
            dimension_scores={t: r.accuracy for t, r in task_results.items()},
            metadata={
                "tasks": {t: r.to_dict() for t, r in task_results.items()},
                "max_conversations": max_conv,
            },
        )
        return self._wrap_single(aggregated, extra={
            "tasks": {t: r.accuracy for t, r in task_results.items()},
            "max_conversations": max_conv,
        })

    # ---- ConvoMem ----------------------------------------------------

    def _run_convomem(self, backend: str) -> dict[str, Any]:
        """ConvoMemRunner 返回 ``{"overall", "per_category", ...}``。"""

        from benchmarks.convomem_eval import ConvoMemRunner

        runner = ConvoMemRunner(
            backend=backend,
            storage_path=os.path.join(self.storage_root, f"convomem-{backend}"),
            verbose=self.verbose,
        )
        max_samples = 10 if self.quick else 100
        summary = runner.run(
            categories=None,
            max_samples=max_samples,
            run_scaling=False,
        )
        overall: BenchmarkResult = summary["overall"]
        per_cat: dict[str, BenchmarkResult] = summary.get("per_category", {})
        return self._wrap_single(overall, extra={
            "categories": {c: r.accuracy for c, r in per_cat.items()},
            "max_samples": max_samples,
        })

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_single(result: BenchmarkResult, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """统一单条结果的对外字段。"""

        payload: dict[str, Any] = {
            "benchmark": result.benchmark_name,
            "backend": result.backend,
            "total_questions": result.total_questions,
            "correct": result.correct,
            "accuracy": round(result.accuracy, 4),
            "f1_score": round(result.f1_score, 4),
            "recall_at_1": round(result.recall_at_1, 4),
            "recall_at_5": round(result.recall_at_5, 4),
            "bleu": round(result.bleu, 4),
            "rouge_l": round(result.rouge_l, 4),
            "avg_query_time_ms": round(result.avg_query_time_ms, 2),
            "avg_add_time_ms": round(result.avg_add_time_ms, 2),
            "dimensions": {k: round(v, 4) for k, v in result.dimension_scores.items()},
            "raw": result.to_dict(),
        }
        if extra:
            payload.update(extra)
        return payload

    # ------------------------------------------------------------------
    # 汇总指标
    # ------------------------------------------------------------------

    def overall_by_backend(self) -> dict[str, float]:
        """每个 backend 在已运行基准上的简单算术平均 accuracy。"""

        out: dict[str, float] = {}
        for backend in self.backends:
            scores: list[float] = []
            for bench in self.benchmarks:
                key = f"{bench}_{backend}"
                payload = self.results.get(key)
                if payload is None:
                    continue
                scores.append(float(payload.get("accuracy", 0.0)))
            out[backend] = round(sum(scores) / len(scores), 4) if scores else 0.0
        return out

    def best_backend(self) -> str | None:
        scores = self.overall_by_backend()
        if not scores:
            return None
        return max(scores.items(), key=lambda kv: kv[1])[0]

    # ------------------------------------------------------------------
    # 报告生成
    # ------------------------------------------------------------------

    def generate_json_report(self, output_path: str) -> str:
        """生成结构化 JSON 结果文件。"""

        payload: dict[str, Any] = {
            "meta": {
                "version": VERSION,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "backends_tested": list(self.backends),
                "benchmarks_run": list(self.benchmarks),
                "quick_mode": self.quick,
            },
            "results": self.results,
            "errors": self.errors,
            "timings_seconds": self.timings,
            "summary": {
                "overall_by_backend": self.overall_by_backend(),
                "best_backend": self.best_backend(),
            },
            "comparison": COMPETITOR_SCORES,
        }

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        return str(out)

    def generate_markdown_report(
        self,
        report_path: str,
        previous_json: str | None = None,
    ) -> str:
        """生成 Markdown Leaderboard 报告。"""

        lines: list[str] = []
        ts = datetime.now().strftime("%Y-%m-%d")

        lines.append(f"# su-memory SDK v{VERSION} — SOTA Benchmark Leaderboard")
        lines.append("")
        lines.append(f"**评测日期**: {ts}")
        lines.append(f"**评测基准**: {' / '.join(b.capitalize() for b in self.benchmarks)}")
        backend_names = " / ".join(BACKENDS[b]["name"] for b in self.backends)
        lines.append(f"**嵌入后端**: {backend_names}")
        lines.append(f"**模式**: {'快速验证 (--quick)' if self.quick else '完整评测'}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # 1. 综合得分表
        lines.append("## 综合得分")
        lines.append("")
        header = ["系统"] + [b.capitalize() for b in self.benchmarks] + ["综合"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["------"] * len(header)) + "|")

        for backend in self.backends:
            row = [f"**su-memory ({BACKENDS[backend]['name']})**"]
            for bench in self.benchmarks:
                payload = self.results.get(f"{bench}_{backend}")
                row.append(_pct(payload.get("accuracy") if payload else None))
            row.append(_pct(self.overall_by_backend().get(backend)))
            lines.append("| " + " | ".join(row) + " |")

        # 竞品行（来自 COMPETITOR_SCORES）
        comp_rows = self._competitor_rows()
        for name, cells in comp_rows:
            lines.append("| " + " | ".join([name] + cells + ["-"]) + " |")
        lines.append("")

        # 2. 双后端对比
        if len(self.backends) >= 2:
            lines.append("## 双后端对比")
            lines.append("")
            head = ["维度"] + [BACKENDS[b]["name"] for b in self.backends] + ["差距"]
            lines.append("| " + " | ".join(head) + " |")
            lines.append("|" + "|".join(["------"] * len(head)) + "|")
            metrics = [
                ("Accuracy", "accuracy"),
                ("F1", "f1_score"),
                ("Recall@1", "recall_at_1"),
                ("Recall@5", "recall_at_5"),
                ("ROUGE-L", "rouge_l"),
                ("BLEU", "bleu"),
                ("avg query (ms)", "avg_query_time_ms"),
                ("avg add (ms)", "avg_add_time_ms"),
            ]
            for label, key in metrics:
                vals: list[float] = []
                cells: list[str] = []
                for backend in self.backends:
                    agg = self._avg_metric(backend, key)
                    cells.append(_metric_fmt(agg, key))
                    vals.append(agg if agg is not None else 0.0)
                if all(v == 0.0 for v in vals):
                    continue
                gap = max(vals) - min(vals)
                cells.append(_metric_fmt(gap, key))
                lines.append("| " + " | ".join([label] + cells) + " |")
            lines.append("")

        # 3. LongMemEval 维度细分
        if "longmemeval" in self.benchmarks:
            lines.extend(self._dim_section(
                title="LongMemEval 维度细分",
                bench="longmemeval",
                key_field="dimensions",
            ))

        # 4. LoCoMo 任务细分
        if "locomo" in self.benchmarks:
            lines.extend(self._dim_section(
                title="LoCoMo 任务细分",
                bench="locomo",
                key_field="tasks",
            ))

        # 5. ConvoMem 类别细分
        if "convomem" in self.benchmarks:
            lines.extend(self._dim_section(
                title="ConvoMem 类型细分",
                bench="convomem",
                key_field="categories",
            ))

        # 6. 回归检查
        lines.append("## 回归检查")
        lines.append("")
        regression = self._compute_regression(previous_json)
        if regression is None:
            lines.append("_未发现历史结果，跳过回归比对。_")
        else:
            lines.append("| Benchmark × Backend | 历史 | 当前 | 变化 |")
            lines.append("|------|------|------|------|")
            for key, (prev, cur) in regression.items():
                diff = (cur - prev) if (prev is not None and cur is not None) else None
                arrow = "→"
                if diff is not None:
                    arrow = "📈" if diff > 0.001 else ("📉" if diff < -0.001 else "→")
                lines.append(
                    f"| {key} | {_pct(prev)} | {_pct(cur)} | "
                    f"{arrow} {(_pct(diff) if diff is not None else '-')} |"
                )
        lines.append("")

        # 错误清单
        if self.errors:
            lines.append("## 错误日志")
            lines.append("")
            for k, v in self.errors.items():
                lines.append(f"- **{k}**: `{v}`")
            lines.append("")

        # 耗时
        if self.timings:
            lines.append("## 耗时")
            lines.append("")
            lines.append("| Benchmark × Backend | 耗时 (s) |")
            lines.append("|------|------|")
            for k, v in self.timings.items():
                lines.append(f"| {k} | {v:.2f} |")
            lines.append("")

        lines.append("---")
        lines.append(f"*Generated by su-memory Benchmark Suite v{VERSION}*")
        lines.append("")

        out = Path(report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines), encoding="utf-8")
        return str(out)

    # ------------------------------------------------------------------
    # 报告小工具
    # ------------------------------------------------------------------

    def _avg_metric(self, backend: str, key: str) -> float | None:
        vals: list[float] = []
        for bench in self.benchmarks:
            payload = self.results.get(f"{bench}_{backend}")
            if payload is None:
                continue
            v = payload.get(key)
            if v is None:
                continue
            vals.append(float(v))
        return (sum(vals) / len(vals)) if vals else None

    def _competitor_rows(self) -> list[tuple[str, list[str]]]:
        """构造竞品对比行，仅展示我们运行的基准。"""

        bench_to_metric = {
            "longmemeval": "longmemeval_accuracy",
            "locomo": "locomo_f1",
            "convomem": "convomem_accuracy",
        }
        nice_names = {
            "hindsight": "Hindsight v5",
            "memgpt": "MemGPT/Letta",
            "mem0": "Mem0",
            "zep": "Zep",
            "letta": "Letta",
            "gpt4_turbo": "GPT-4 Turbo (full ctx)",
        }
        out: list[tuple[str, list[str]]] = []
        for sys_key, scores in COMPETITOR_SCORES.items():
            cells: list[str] = []
            for bench in self.benchmarks:
                metric = bench_to_metric.get(bench)
                v = scores.get(metric) if metric else None
                cells.append(_pct(v))
            out.append((nice_names.get(sys_key, sys_key), cells))
        return out

    def _dim_section(self, title: str, bench: str, key_field: str) -> list[str]:
        """生成单个基准的维度细分章节。"""

        sub: list[str] = []
        sub.append(f"## {title}")
        sub.append("")
        # 收集所有维度键
        dim_keys: list[str] = []
        for backend in self.backends:
            payload = self.results.get(f"{bench}_{backend}")
            if not payload:
                continue
            dims = payload.get(key_field, {}) or {}
            for k in dims.keys():
                if k not in dim_keys:
                    dim_keys.append(k)
        if not dim_keys:
            sub.append("_无数据。_")
            sub.append("")
            return sub

        head = ["维度"] + [BACKENDS[b]["name"] for b in self.backends]
        sub.append("| " + " | ".join(head) + " |")
        sub.append("|" + "|".join(["------"] * len(head)) + "|")
        for dk in dim_keys:
            row = [dk]
            for backend in self.backends:
                payload = self.results.get(f"{bench}_{backend}", {})
                dims = payload.get(key_field, {}) or {}
                row.append(_pct(dims.get(dk)))
            sub.append("| " + " | ".join(row) + " |")
        sub.append("")
        return sub

    def _compute_regression(
        self,
        previous_json: str | None,
    ) -> dict[str, tuple[float | None, float | None]] | None:
        """对比上次运行结果，返回 ``{key: (prev, current)}``。"""

        prev_path = self._resolve_previous_json(previous_json)
        if prev_path is None:
            return None
        try:
            with open(prev_path, "r", encoding="utf-8") as fh:
                prev = json.load(fh)
        except Exception:
            return None
        prev_results = prev.get("results", {}) if isinstance(prev, dict) else {}
        out: dict[str, tuple[float | None, float | None]] = {}
        for key, payload in self.results.items():
            cur = float(payload.get("accuracy", 0.0))
            prev_payload = prev_results.get(key) or {}
            prev_acc = prev_payload.get("accuracy") if isinstance(prev_payload, dict) else None
            out[key] = (
                float(prev_acc) if isinstance(prev_acc, (int, float)) else None,
                cur,
            )
        return out

    def _resolve_previous_json(self, hint: str | None) -> Path | None:
        """寻找上一次的结果 JSON：优先 hint，其次 results/ 目录最新文件。"""

        if hint and Path(hint).exists():
            return Path(hint)
        candidates = sorted(
            RESULTS_DIR.glob("sota_v*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    # ------------------------------------------------------------------
    # 控制台 ASCII 汇总
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """打印 ASCII 风格的控制台汇总。"""

        print()
        print("═" * 71)
        print(f"  su-memory v{VERSION} — Unified Benchmark Report")
        print("═" * 71)
        print()

        # 竞品基线（用于括号注释）
        baseline_hint = {
            "longmemeval": ("Hindsight", COMPETITOR_SCORES["hindsight"]["longmemeval_accuracy"]),
            "locomo": ("Hindsight", COMPETITOR_SCORES["hindsight"]["locomo_f1"]),
            "convomem": ("GPT-4 Turbo", COMPETITOR_SCORES["gpt4_turbo"]["convomem_accuracy"]),
        }

        for backend in self.backends:
            cfg = BACKENDS[backend]
            print(f"  Backend: {cfg['name']} ({cfg['dimension']}d)")
            print("  " + "─" * 45)
            for bench in self.benchmarks:
                key = f"{bench}_{backend}"
                payload = self.results.get(key)
                acc_str = _pct(payload.get("accuracy") if payload else None)
                ref_name, ref_val = baseline_hint.get(bench, ("-", None))
                ref_str = _pct(ref_val)
                label = bench.capitalize() + ":"
                print(f"  {label:<15} {acc_str:>7}  ({ref_name}: {ref_str})")
            overall = self.overall_by_backend().get(backend)
            print("  " + "─" * 45)
            print(f"  {'Overall:':<15} {_pct(overall):>7}")
            print()
        print("═" * 71)
        if self.errors:
            print(f"  ⚠️  {len(self.errors)} 个组合执行失败：")
            for k, v in self.errors.items():
                print(f"     - {k}: {v}")
            print("═" * 71)


# ---------------------------------------------------------------------------
# 格式化工具
# ---------------------------------------------------------------------------

def _pct(v: float | None) -> str:
    """百分比格式化，兼容 ``None`` 与字符串。"""

    if v is None:
        return "N/A"
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _metric_fmt(v: float | None, key: str) -> str:
    """根据指标类型选择格式化方式。"""

    if v is None:
        return "N/A"
    if "time_ms" in key:
        return f"{v:.1f}"
    return _pct(v)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_benchmarks(arg: str) -> list[str]:
    arg = (arg or "").strip().lower()
    if arg in ("", "all"):
        return list(ALL_BENCHMARKS)
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    invalid = [p for p in parts if p not in ALL_BENCHMARKS]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unknown benchmarks {invalid}. Valid: {list(ALL_BENCHMARKS)} or 'all'."
        )
    return parts


def _parse_backends(arg: str) -> list[str]:
    arg = (arg or "").strip().lower()
    if arg in ("", "both", "all"):
        return list(ALL_BACKENDS)
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    invalid = [p for p in parts if p not in BACKENDS]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unknown backends {invalid}. Valid: {list(BACKENDS.keys())} or 'both'."
        )
    return parts


def build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        prog="benchmarks/run_all.py",
        description=f"su-memory v{VERSION} 统一基准评测套件 "
                    "(LongMemEval + LoCoMo + ConvoMem × Ollama/sbert)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="all",
        help="评测基准: all / longmemeval / locomo / convomem，可逗号分隔 (默认 all)",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="both",
        help="嵌入后端: both / ollama / sbert，可逗号分隔 (默认 both)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速模式：限制数据量便于冒烟测试",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help=f"JSON 输出路径 (默认 benchmarks/results/sota_v{VERSION}_<ts>.json)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=str(_THIS_DIR / "BENCHMARK_REPORT.md"),
        help="Markdown 报告路径 (默认 benchmarks/BENCHMARK_REPORT.md)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="/tmp/su-memory-bench",
        help="su-memory 持久化根目录 (默认 /tmp/su-memory-bench)",
    )
    parser.add_argument(
        "--previous",
        type=str,
        default=None,
        help="用于回归比对的历史 JSON 文件 (默认自动选取 results/ 目录最新)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="详细日志",
    )
    return parser


def _default_output_path() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(RESULTS_DIR / f"sota_v{VERSION}_{ts}.json")


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    benchmarks = _parse_benchmarks(args.benchmarks)
    backends = _parse_backends(args.backends)
    output_path = args.output or _default_output_path()

    print("═" * 71)
    print(f"  su-memory v{VERSION} — Unified Benchmark Suite")
    print(f"  Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Benchmarks: {', '.join(benchmarks)}")
    print(f"  Backends:   {', '.join(backends)}")
    print(f"  Mode:       {'quick' if args.quick else 'full'}")
    print(f"  Output:     {output_path}")
    print(f"  Report:     {args.report}")
    print("═" * 71)

    suite = UnifiedBenchmarkSuite(
        benchmarks=benchmarks,
        backends=backends,
        quick=args.quick,
        verbose=args.verbose,
        storage_root=args.storage,
    )

    t0 = time.time()
    suite.run()
    elapsed = time.time() - t0

    json_path = suite.generate_json_report(output_path)
    md_path = suite.generate_markdown_report(args.report, previous_json=args.previous)

    suite.print_summary()
    print()
    print(f"  ⏱  Total elapsed: {elapsed:.1f}s")
    print(f"  📄 JSON report:   {json_path}")
    print(f"  📝 Markdown:      {md_path}")
    print("═" * 71)

    # 全部失败 → 非零退出
    if suite.results:
        return 0
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
