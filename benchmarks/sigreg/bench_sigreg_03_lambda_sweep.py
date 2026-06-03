#!/usr/bin/env python3
"""
bench_sigreg_03_lambda_sweep.py
================================
待测量 2: 检索期 λ-sweep

问题: SIGReg 代码里 __init__ 默认 λ=0.01, apply_sigreg_to_index 默认 λ=0.02;
而 LeJEPA 训练期论文里 λ ∈ [0.1, 1.0] —— **两个数量级的差距**到底是检索期的真实最优,
还是只是 SDK 保守工程默认值?

8 点扫描 λ ∈ {0.000, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200, 0.500}
每点跑 HotpotQA dev 500 queries, 记录 Recall@5。

依赖: 脚本 1 已跑完, ./cache/ 里有 passage_embs.npy + query_embs.npy

用法:
  python bench_sigreg_03_lambda_sweep.py
  python bench_sigreg_03_lambda_sweep.py --cache-dir ./cache --top-k 5
  python bench_sigreg_03_lambda_sweep.py --lambdas "0.0,0.01,0.05,0.1,0.5"
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

# ============================================================
# P1-R5: 锁定线程, 消除多线程不确定性
# ============================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import faiss  # noqa: E402

faiss.omp_set_num_threads(1)  # FAISS HNSW 建索引锁单线程

SU_MEMORY_SRC = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SU_MEMORY_SRC))

from su_memory.sdk._sigreg import SIGReg  # noqa: E402

# ============================================================
# 全局常量
# ============================================================
REPRODUCIBILITY_SEED = 42  # P0-R2: passage/query 共用同一 sketch 矩阵


# ============================================================
# 工具 (与脚本 2 相同)
# ============================================================

@dataclass
class LambdaPointResult:
    lambda_reg: float
    isotropy_after: float
    recall_at_k: float
    build_time_s: float
    query_time_s: float


def build_hnsw(embs: np.ndarray, m: int = 32, efc: int = 64, efs: int = 64):
    n, d = embs.shape
    index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = efc
    index.hnsw.efSearch = efs
    t0 = time.perf_counter()
    index.add(np.ascontiguousarray(embs.astype(np.float32)))
    return index, time.perf_counter() - t0


def recall_at_k(index, query_embs: np.ndarray, gold_lists: list[list[int]], k: int):
    n = query_embs.shape[0]
    t0 = time.perf_counter()
    _, I = index.search(np.ascontiguousarray(query_embs.astype(np.float32)), k)
    qt = time.perf_counter() - t0
    hits = sum(
        1 for i, golds in enumerate(gold_lists)
        if any(g in set(I[i].tolist()) for g in golds)
    )
    return hits / n if n else 0.0, qt


def make_gold_lists(queries, pid_to_idx):
    return [[pid_to_idx[pid] for pid in q["gold_passage_ids"] if pid in pid_to_idx]
            for q in queries]


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="SIGReg λ-sweep on retrieval Recall@5")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(__file__).resolve().parent / "cache")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--lambdas", type=str,
                        default="0.000,0.005,0.010,0.020,0.050,0.100,0.200,0.500",
                        help="逗号分隔的 λ 值列表")
    parser.add_argument("--sketch-dim", type=int, default=64)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-efc", type=int, default=64)
    parser.add_argument("--hnsw-efs", type=int, default=64)
    args = parser.parse_args()

    cache = args.cache_dir.resolve()
    if not (cache / "passage_embs.npy").exists():
        print(f"[ERROR] {cache}/passage_embs.npy 不存在, 请先跑 bench_sigreg_01_prepare.py")
        sys.exit(1)

    print(f"[LOAD] 读取 {cache} ...")
    passage_embs = np.load(cache / "passage_embs.npy")
    query_embs = np.load(cache / "query_embs.npy")
    queries = [json.loads(l) for l in (cache / "hotpotqa_queries.jsonl").open()]
    corpus = [json.loads(l) for l in (cache / "hotpotqa_corpus.jsonl").open()]
    pid_to_idx = {c["pid"]: i for i, c in enumerate(corpus)}
    gold_lists = make_gold_lists(queries, pid_to_idx)
    print(f"       passages={passage_embs.shape[0]}  queries={query_embs.shape[0]}  top-k={args.top_k}")

    # baseline (λ=0): 直接 L2 归一化的 raw bge 输出 (作为对照, 不用 SIGReg)
    raw_norm = passage_embs / np.maximum(np.linalg.norm(passage_embs, axis=1, keepdims=True), 1e-10)
    raw_norm = raw_norm.astype(np.float32)
    q_norm = query_embs / np.maximum(np.linalg.norm(query_embs, axis=1, keepdims=True), 1e-10)
    q_norm = q_norm.astype(np.float32)
    np.random.seed(REPRODUCIBILITY_SEED)
    sigreg_baseline = SIGReg(lambda_reg=0.0, sketch_dim=args.sketch_dim)
    iso_raw = sigreg_baseline.compute_isotropy_score(passage_embs)
    print(f"       isotropy(raw bge embeddings) = {iso_raw:.3e}")

    # 预热: baseline
    print(f"\n--- baseline (λ=0, no SIGReg) ---")
    idx_raw, build_raw = build_hnsw(raw_norm, args.hnsw_m, args.hnsw_efc, args.hnsw_efs)
    recall_raw, query_raw = recall_at_k(idx_raw, q_norm, gold_lists, args.top_k)
    print(f"  Recall@{args.top_k} = {recall_raw:.4f}   build={build_raw:.2f}s  query={query_raw:.2f}s")

    # 跑 sweep
    lambda_values = [float(x) for x in args.lambdas.split(",")]
    print(f"\n[SWEEP] {len(lambda_values)} 个 λ 点: {lambda_values}")

    # P2-R11: 断点续跑 —— 检查已有结果
    partial_path = cache / "results_lambda_sweep_partial.json"
    completed_lambdas: dict[float, dict] = {}
    if partial_path.exists():
        try:
            partial = json.loads(partial_path.read_text())
            for r in partial.get("sweep", []):
                completed_lambdas[r["lambda_reg"]] = r
            print(f"  ✔ 发现断点文件, 已完成 {len(completed_lambdas)} 个 λ 点, 跳过重跑")
        except Exception:
            pass

    results: list[LambdaPointResult] = []
    sweep_start = time.time()
    for idx, lam in enumerate(lambda_values, 1):
        # 断点恢复: 跳过已完成的 λ
        if lam in completed_lambdas:
            r_dict = completed_lambdas[lam]
            results.append(LambdaPointResult(**r_dict))
            print(f"\n--- [{idx}/{len(lambda_values)}] λ = {lam} (cached) Recall@{args.top_k}={r_dict['recall_at_k']:.4f} ---")
            continue

        print(f"\n--- [{idx}/{len(lambda_values)}] λ = {lam} ---")
        np.random.seed(REPRODUCIBILITY_SEED)  # P0-R2: 每个 λ 用同一 sketch
        sigreg = SIGReg(lambda_reg=lam, sketch_dim=args.sketch_dim)
        reg_p = sigreg.regularize(passage_embs)
        np.random.seed(REPRODUCIBILITY_SEED)  # P0-R2: query 用同一 sketch
        reg_q = sigreg.regularize(query_embs)
        iso = sigreg.compute_isotropy_score(reg_p)
        print(f"  shape={reg_p.shape}  isotropy={iso:.3e}")

        idx_faiss, build_t = build_hnsw(reg_p, args.hnsw_m, args.hnsw_efc, args.hnsw_efs)
        recall, query_t = recall_at_k(idx_faiss, reg_q, gold_lists, args.top_k)
        print(f"  build={build_t:.2f}s  query={query_t:.2f}s  Recall@{args.top_k}={recall:.4f}")

        point = LambdaPointResult(
            lambda_reg=lam,
            isotropy_after=iso,
            recall_at_k=recall,
            build_time_s=build_t,
            query_time_s=query_t,
        )
        results.append(point)

        # P2-R11: 每个 λ 点完即写盘
        _save_partial(partial_path, args, results, iso_raw, recall_raw, build_raw, query_raw)

        # P2-R10: 释放索引内存
        del idx_faiss, reg_p, reg_q
        gc.collect()

        # 进度提示
        elapsed = time.time() - sweep_start
        remaining = len(lambda_values) - idx
        eta = (elapsed / idx) * remaining if idx > 0 else 0
        print(f"  → 进度 {idx}/{len(lambda_values)},  elapsed={elapsed:.0f}s,  ETA≈{eta:.0f}s")

    # 输出 JSON
    out_path = cache / "results_lambda_sweep.json"
    out_path.write_text(json.dumps(
        {
            "config": {
                "top_k": args.top_k,
                "sketch_dim": args.sketch_dim,
                "hnsw_m": args.hnsw_m,
                "hnsw_efc": args.hnsw_efc,
                "hnsw_efs": args.hnsw_efs,
                "n_passages": int(passage_embs.shape[0]),
                "n_queries": int(query_embs.shape[0]),
                "reproducibility_seed": REPRODUCIBILITY_SEED,
                "single_threaded": True,
                "granularity": "paragraph-level",
            },
            "baseline": {
                "lambda_reg": 0.0,
                "isotropy": iso_raw,
                "recall_at_k": recall_raw,
                "build_time_s": build_raw,
                "query_time_s": query_raw,
            },
            "sweep": [asdict(r) for r in results],
        },
        indent=2,
        ensure_ascii=False,
    ))

    # 报告
    print("\n" + "=" * 78)
    print(f"  SIGReg λ-Sweep — Recall@{args.top_k} on HotpotQA dev (baseline λ=0 = raw bge)")
    print("=" * 78)
    print(f"  {'λ':>8}  {'iso':>11}  {'Recall@5':>10}  {'Δ vs base':>11}  {'build(s)':>10}")
    print("  " + "-" * 64)
    print(f"  {'0.000':>8}  {iso_raw:>11.3e}  {recall_raw:>10.4f}  {'—':>11}  {build_raw:>10.2f}")
    for r in results:
        delta = r.recall_at_k - recall_raw
        print(f"  {r.lambda_reg:>8.3f}  {r.isotropy_after:>11.3e}  {r.recall_at_k:>10.4f}  "
              f"{delta:>+11.4f}  {r.build_time_s:>10.2f}")
    print("  " + "-" * 64)

    # 找最佳
    best = max([r for r in results], key=lambda r: r.recall_at_k) if results else None
    if best is not None:
        print(f"\n  🏆 最佳 λ = {best.lambda_reg},  Recall@{args.top_k} = {best.recall_at_k:.4f}")
        print(f"     相对 baseline 提升 Δ = {best.recall_at_k - recall_raw:+.4f}")
    print(f"\n  结果已保存: {out_path}")

    # P1-R8: 写 env_info
    _write_env_info(cache)

    # 清理断点文件 (已完成全部 sweep)
    if partial_path.exists():
        partial_path.unlink()
        print(f"  ✔ 断点文件已清理: {partial_path.name}")


def _save_partial(path: Path, args, results, iso_raw, recall_raw, build_raw, query_raw):
    """每个 λ 点完成后立即写盘, 支持断点续跑。"""
    path.write_text(json.dumps(
        {
            "config": {
                "top_k": args.top_k,
                "sketch_dim": args.sketch_dim,
                "hnsw_m": args.hnsw_m,
                "hnsw_efc": args.hnsw_efc,
                "hnsw_efs": args.hnsw_efs,
                "reproducibility_seed": REPRODUCIBILITY_SEED,
                "single_threaded": True,
            },
            "baseline": {
                "lambda_reg": 0.0,
                "isotropy": iso_raw,
                "recall_at_k": recall_raw,
                "build_time_s": build_raw,
                "query_time_s": query_raw,
            },
            "sweep": [asdict(r) for r in results],
        },
        indent=2,
        ensure_ascii=False,
    ))


def _write_env_info(cache: Path):
    """写入环境元数据, 方便复现。"""
    import su_memory
    info = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "faiss": faiss.__version__,
        "su_memory": su_memory.__version__,
        "single_threaded": True,
        "reproducibility_seed": REPRODUCIBILITY_SEED,
        "script": Path(__file__).name,
    }
    (cache / "env_info_03.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"  环境信息已保存: {cache / 'env_info_03.json'}")


if __name__ == "__main__":
    main()
