#!/usr/bin/env python3
"""
bench_sigreg_02_back_projection.py
===================================
待测量 1: back-projection ablation

问题: SIGReg 的 sketched whitening 末尾有一行
    z_reg = z_whitened @ sketch.T
把 64 维 sketched 子空间的白化结果**回投影**到原始 d=512 空间。
这一步到底有没有用?

3 个变体:
  A. Raw baseline            d=512, 不做任何处理
  B. Sketched-only           d=64,  把 z_whitened 直接拿去建 HNSW 索引
  C. Back-projected (现状)    d=512, 回投影后建 HNSW 索引

指标: Recall@5 (HotpotQA dev, 500 queries)

依赖: 脚本 1 已跑完, ./cache/ 里有 passage_embs.npy + query_embs.npy

用法:
  python bench_sigreg_02_back_projection.py
  python bench_sigreg_02_back_projection.py --cache-dir ./cache --top-k 5 --lambda 0.02
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

# 让 su_memory 可导入
SU_MEMORY_SRC = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SU_MEMORY_SRC))

from su_memory.sdk._sigreg import SIGReg  # noqa: E402

# ============================================================
# 全局常量
# ============================================================
REPRODUCIBILITY_SEED = 42  # P0-R2/R3: passage/query 共用同一 sketch 矩阵


# ============================================================
# 工具
# ============================================================

@dataclass
class VariantResult:
    name: str
    dim: int
    isotropy_before: float
    isotropy_after: float
    recall_at_k: float
    build_time_s: float
    query_time_s: float
    extra: dict


def build_hnsw(embs: np.ndarray, m: int = 32, ef_construction: int = 64, ef_search: int = 64):
    """建 FAISS IndexHNSWFlat, 返回索引 + 建索引时间(s)。"""
    n, d = embs.shape
    index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    t0 = time.perf_counter()
    index.add(np.ascontiguousarray(embs.astype(np.float32)))
    build_time = time.perf_counter() - t0
    return index, build_time


def recall_at_k(index, query_embs: np.ndarray, gold_lists: list[list[int]], k: int = 5):
    """
    标准 Recall@k: 第 i 条 query 在 top-k 中命中 gold_lists[i] 任一即算 1, 取平均。

    gold_lists 是 list[list[int]], 元素是 gold passage 的 index。
    granularity: paragraph-level (P1-R7)
    """
    n, d = query_embs.shape
    t0 = time.perf_counter()
    _, I = index.search(np.ascontiguousarray(query_embs.astype(np.float32)), k)
    query_time = time.perf_counter() - t0

    hits = 0
    for i, golds in enumerate(gold_lists):
        retrieved = set(I[i].tolist())
        if any(g in retrieved for g in golds):
            hits += 1
    return hits / n if n else 0.0, query_time


def make_gold_lists(queries: list[dict], pid_to_idx: dict[str, int]) -> list[list[int]]:
    return [[pid_to_idx[pid] for pid in q["gold_passage_ids"] if pid in pid_to_idx]
            for q in queries]


# ============================================================
# 变体实现
# ============================================================

def variant_raw(passage_embs: np.ndarray, sigreg: SIGReg):
    """A. 不做任何处理。"""
    return passage_embs.copy(), {
        "isotropy_score": sigreg.compute_isotropy_score(passage_embs),
    }


def variant_sketched_only(passage_embs: np.ndarray, sigreg: SIGReg):
    """
    B. 仅在 sketched 子空间做白化, 不回投影, 直接以 d=64 建索引。

    这是 v1 LeJEPA 论文里的做法。
    """
    np.random.seed(REPRODUCIBILITY_SEED)  # P0-R3: 与 query 共用同一 sketch
    z = passage_embs.astype(np.float64)
    z = z - z.mean(axis=0, keepdims=True)
    sketch = np.random.randn(z.shape[1], sigreg.sketch_dim) / np.sqrt(sigreg.sketch_dim)
    z_sketch = z @ sketch  # (n, sketch_dim)
    cov_sketch = z_sketch.T @ z_sketch / (z.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov_sketch)
    eigvals = np.maximum(eigvals, 1e-6)
    whitening = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    z_whitened = z_sketch @ whitening  # (n, sketch_dim=64)

    iso = sigreg.compute_isotropy_score(z_whitened)
    return z_whitened.astype(np.float32), {"isotropy_score": iso}


def variant_back_projected(passage_embs: np.ndarray, sigreg: SIGReg):
    """
    C. 当前 SIGReg 实现: sketched 白化后**回投影**到原始 d 空间, 再 L2 归一化。

    即 su_memory.sdk._sigreg.SIGReg.regularize() 的语义。
    """
    np.random.seed(REPRODUCIBILITY_SEED)  # P0-R2: 锁 sketch 矩阵
    regularized = sigreg.regularize(passage_embs)
    iso = sigreg.compute_isotropy_score(regularized)
    return regularized, {"isotropy_score": iso}


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="SIGReg back-projection ablation")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(__file__).resolve().parent / "cache")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--lambda", dest="lambda_reg", type=float, default=0.02,
                        help="SIGReg 正则强度 (默认 0.02 与 apply_sigreg_to_index 一致)")
    parser.add_argument("--sketch-dim", type=int, default=64)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-efc", type=int, default=64)
    parser.add_argument("--hnsw-efs", type=int, default=64)
    args = parser.parse_args()

    cache = args.cache_dir.resolve()
    if not (cache / "passage_embs.npy").exists():
        print(f"[ERROR] {cache}/passage_embs.npy 不存在, 请先跑 bench_sigreg_01_prepare.py")
        sys.exit(1)

    # 加载
    print(f"[LOAD] 读取 {cache} ...")
    passage_embs = np.load(cache / "passage_embs.npy")
    query_embs = np.load(cache / "query_embs.npy")
    queries = [json.loads(l) for l in (cache / "hotpotqa_queries.jsonl").open()]
    corpus = [json.loads(l) for l in (cache / "hotpotqa_corpus.jsonl").open()]
    pid_to_idx = {c["pid"]: i for i, c in enumerate(corpus)}
    gold_lists = make_gold_lists(queries, pid_to_idx)
    print(f"       passage_embs={passage_embs.shape}  query_embs={query_embs.shape}")
    print(f"       {len(queries)} queries,  {len(corpus)} passages,  top-k={args.top_k}")

    sigreg = SIGReg(lambda_reg=args.lambda_reg, sketch_dim=args.sketch_dim)
    sigreg_iso_baseline = sigreg.compute_isotropy_score(passage_embs)
    print(f"       SIGReg 超参: lambda_reg={args.lambda_reg}, sketch_dim={args.sketch_dim}")
    print(f"       isotropy(baseline) = {sigreg_iso_baseline:.3e}")

    # 跑三个变体
    variants = [
        ("A_raw_baseline",       variant_raw,             {"description": "no transformation"}),
        ("B_sketched_only",      variant_sketched_only,   {"description": "whitening in 64-dim sketch, no back-projection"}),
        ("C_back_projected",     variant_back_projected,  {"description": "sketched whitening + back-project to d=512 (current SIGReg)"}),
    ]

    results: list[VariantResult] = []
    for name, fn, extra in variants:
        print(f"\n--- {name} ---")
        embs, info = fn(passage_embs, sigreg)
        d = embs.shape[1]
        print(f"  shape={embs.shape}, isotropy={info['isotropy_score']:.3e}")

        # 同步处理 query (注意: query 走 raw baseline 还是过 SIGReg?)
        # 这里采用"训练时/索引时用 SIGReg, 查询时也用"的对称方案
        # 对 raw 变体, query 保持原样; 对其他变体, query 也用对应变换
        if name == "A_raw_baseline":
            q_embs = query_embs.copy()
        elif name == "B_sketched_only":
            # P0-R3: query 用与 passage 完全相同的 sketch 矩阵 (同一 seed)
            np.random.seed(REPRODUCIBILITY_SEED)
            z = query_embs.astype(np.float64)
            z = z - z.mean(axis=0, keepdims=True)
            sketch = np.random.randn(z.shape[1], sigreg.sketch_dim) / np.sqrt(sigreg.sketch_dim)
            z_sketch = z @ sketch
            cov_sketch = z_sketch.T @ z_sketch / (z.shape[0] - 1)
            eigvals, eigvecs = np.linalg.eigh(cov_sketch)
            eigvals = np.maximum(eigvals, 1e-6)
            whitening = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            q_embs = (z_sketch @ whitening).astype(np.float32)
        else:  # C: SIGReg on queries
            np.random.seed(REPRODUCIBILITY_SEED)  # P0-R2: 同步 sketch
            q_embs = sigreg.regularize(query_embs)

        # 维度必须一致才能共享索引
        assert q_embs.shape[1] == embs.shape[1], (
            f"query/passage dim mismatch: q={q_embs.shape[1]}, p={embs.shape[1]}"
        )

        # 对 raw 也 L2 归一化(FAISS IP 检索需要), 公平起见
        if name == "A_raw_baseline":
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            embs = (embs / np.maximum(norms, 1e-10)).astype(np.float32)
            qn = np.linalg.norm(q_embs, axis=1, keepdims=True)
            q_embs = (q_embs / np.maximum(qn, 1e-10)).astype(np.float32)

        index, build_t = build_hnsw(embs, args.hnsw_m, args.hnsw_efc, args.hnsw_efs)
        recall, query_t = recall_at_k(index, q_embs, gold_lists, args.top_k)
        print(f"  build={build_t:.2f}s  query_total={query_t:.2f}s  Recall@{args.top_k}={recall:.4f}")

        results.append(VariantResult(
            name=name,
            dim=d,
            isotropy_before=sigreg_iso_baseline,
            isotropy_after=info["isotropy_score"],
            recall_at_k=recall,
            build_time_s=build_t,
            query_time_s=query_t,
            extra=extra,
        ))

    # 输出
    out_path = cache / "results_back_projection.json"
    out_path.write_text(json.dumps(
        {
            "config": {
                "top_k": args.top_k,
                "lambda_reg": args.lambda_reg,
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
            "results": [asdict(r) for r in results],
        },
        indent=2,
        ensure_ascii=False,
    ))

    # 报告
    print("\n" + "=" * 70)
    print(f"  Back-Projection Ablation — Recall@{args.top_k} on HotpotQA dev")
    print("=" * 70)
    print(f"  {'Variant':<22} {'dim':>5} {'iso':>11} {'Recall@5':>10} {'build(s)':>10}")
    print("  " + "-" * 60)
    for r in results:
        print(f"  {r.name:<22} {r.dim:>5} {r.isotropy_after:>11.3e} {r.recall_at_k:>10.4f} {r.build_time_s:>10.2f}")
    print("  " + "-" * 60)

    # 关键结论
    raw = next(r for r in results if r.name == "A_raw_baseline")
    sketch = next(r for r in results if r.name == "B_sketched_only")
    bp = next(r for r in results if r.name == "C_back_projected")
    delta_bp_raw = bp.recall_at_k - raw.recall_at_k
    delta_bp_sketch = bp.recall_at_k - sketch.recall_at_k
    print(f"\n  ΔRecall (back-projected − raw baseline) = {delta_bp_raw:+.4f}")
    print(f"  ΔRecall (back-projected − sketched-only) = {delta_bp_sketch:+.4f}")

    if delta_bp_raw > 0 and delta_bp_sketch > 0:
        print("  → Back-projection 在两个对照上都帮上忙, 是有效设计")
    elif delta_bp_raw < 0 < delta_bp_sketch:
        print("  → Back-projection 输给 raw, 但赢过 sketched-only, 复杂局面")
    else:
        print("  → Back-projection 优势不显著, 需结合 λ-sweep 进一步判断")
    print(f"\n  结果已保存: {out_path}")

    # P1-R8: 写 env_info
    _write_env_info(cache, time.time())


def _write_env_info(cache: Path, end_ts: float):
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
    (cache / "env_info_02.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"  环境信息已保存: {cache / 'env_info_02.json'}")


if __name__ == "__main__":
    main()
