#!/usr/bin/env python3
"""
M4 超参搜索脚本
================

自动搜索最优 (learning_rate, hidden_dim, key_dim) 组合。

搜索空间:
- learning_rate:  {0.005, 0.01, 0.02, 0.05}
- hidden_dim:     {8, 16, 32}
- key_dim:        {8, 16, 32}

每个组合跑 3 次取平均，报告 top-3。

用法:
    python scripts/hp_search_m4.py [--quick]
    --quick: 快速模式，每组合 1 次，减少 epoch
"""

import sys
import os
import time
import json
from itertools import product

# 确保项目根目录及 src/ 在 sys.path 中，支持包导入
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.train_m3_e2e import train_and_evaluate, generate_causal_timeline, compute_energy_conservation
from su_memory.sdk._world_model import MCIWorldModel
from su_memory.sdk._jepa_dataset import JEPADataset
from su_memory.sdk._jepa_trainer import JEPATrainer
from su_memory.sdk._jepa_gat_encoder import preprocess_memories_to_features, features_to_state
from su_memory.sdk._jepa_gnn import align_adjacency, GNNPredictor


# =============================================================================
# 评估: Edge F1 (类似 eval_m3_quality.py，但轻量)
# =============================================================================

def evaluate_edge_f1(wm, dataset) -> float:
    """计算训练后模型的 Edge F1。"""
    import numpy as np
    from su_memory.sdk._world_model import CausalWorldModelState

    f1_list = []
    has_states = hasattr(dataset, 'state_pairs') and dataset.state_pairs

    for idx, (mem_t, _) in enumerate(dataset.memory_pairs):
        s_t1 = dataset.state_pairs[idx][1] if has_states and idx < len(dataset.state_pairs) else None
        if s_t1 is None:
            continue
        s_t_cache = dataset.state_pairs[idx][0] if has_states and idx < len(dataset.state_pairs) else None

        X, node_index, _ = preprocess_memories_to_features(wm, mem_t, state=s_t_cache)
        if X.shape[0] == 0:
            continue

        A_enc = wm._jepa_encoder.gat_encoder.forward(X)
        s_enc = features_to_state(A_enc, node_index, CausalWorldModelState())
        wm._jepa_predictor.training_predict(s_enc)
        A_pred = wm._jepa_predictor.get_predicted_adj()
        pred_ni = wm._jepa_predictor.get_node_index()
        if A_pred is None or pred_ni is None:
            continue

        A_target = np.abs(align_adjacency(s_t1, pred_ni))
        n = A_pred.shape[0]
        if A_target.shape[0] != n:
            padded = np.zeros((n, n), dtype=A_target.dtype)
            m = min(A_target.shape[0], n)
            padded[:m, :m] = A_target[:m, :m]
            A_target = padded

        # Edge F1 with abs threshold
        thresh = 0.1
        pred_bin = (np.abs(A_pred) >= thresh).astype(np.int32)
        targ_bin = (np.abs(A_target) >= thresh).astype(np.int32)
        np.fill_diagonal(pred_bin, 0)
        np.fill_diagonal(targ_bin, 0)

        tp = int((pred_bin & targ_bin).sum())
        fp = int((pred_bin & ~targ_bin).sum())
        fn = int((~pred_bin & targ_bin).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        f1_list.append(f1)

    return float(np.mean(f1_list)) if f1_list else 0.0


# =============================================================================
# 单次评估
# =============================================================================

def run_trial(
    lr: float,
    hidden_dim: int,
    key_dim: int,
    n_epochs: int = 15,
    n_timesteps: int = 30,
    n_memories_per_step: int = 10,
    seed: int = 42,
) -> dict:
    """运行单次训练 + 评估。"""
    import numpy as np
    np.random.seed(seed)

    memories = generate_causal_timeline(
        n_timesteps=n_timesteps,
        n_memories_per_step=n_memories_per_step,
        seed=seed,
    )

    wm = MCIWorldModel()
    wm.initialize()

    # 创建 dataset（使用原始 discover，不需要 M3）
    dataset = JEPADataset.from_memories(wm, memories, window_size=10)

    if dataset.n_pairs < 3:
        return {"error": "insufficient_pairs", "final_loss": 999.0, "edge_f1": 0.0}

    # 启用 M3
    wm.enable_m3(encoder_key_dim=key_dim, predictor_hidden_dim=hidden_dim)

    # 训练
    t0 = time.time()
    trainer = JEPATrainer(
        encoder=wm._jepa_encoder,
        predictor=wm._jepa_predictor,
        dataset=dataset,
        alpha_energy=0.1,
        beta_cons=0.05,
    )
    stats = trainer.train(n_epochs=n_epochs, learning_rate=lr)
    elapsed = time.time() - t0

    loss_history = stats.loss_history
    final_loss = loss_history[-1] if loss_history else 999.0
    initial_loss = loss_history[0] if loss_history else 999.0

    # 收敛率: 损失下降百分比
    convergence = (initial_loss - final_loss) / max(initial_loss, 1e-10) * 100

    # Edge F1
    edge_f1 = evaluate_edge_f1(wm, dataset)

    # 能量守恒
    energy = compute_energy_conservation(wm, dataset)
    cons_score = energy.get("conservation_score", 0.0)

    return {
        "lr": lr,
        "hidden_dim": hidden_dim,
        "key_dim": key_dim,
        "initial_loss": round(initial_loss, 6),
        "final_loss": round(final_loss, 6),
        "convergence_pct": round(convergence, 1),
        "edge_f1": round(edge_f1, 4),
        "conservation_score": round(cons_score, 4),
        "elapsed_sec": round(elapsed, 2),
        "error": None,
    }


# =============================================================================
# 网格搜索
# =============================================================================

def _combo_done(lr: float, hd: int, kd: int, resume_data: list) -> bool:
    """检查组合是否已在断点文件中完成。"""
    for r in resume_data:
        if r["lr"] == lr and r["hidden_dim"] == hd and r["key_dim"] == kd:
            return r.get("n_valid_trials", 0) > 0
    return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="M4 Hyperparameter Search")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 1 trial, fewer epochs")
    parser.add_argument("--n-trials", type=int, default=3, help="Trials per combo")
    parser.add_argument("--n-epochs", type=int, default=15, help="Epochs per trial")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save structured results to JSON (default: ../hp_search_results.json)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume from a previous results JSON, skipping completed combos")
    args = parser.parse_args()

    # ── 断点续跑：加载已完成结果 ──
    resume_data = []
    if args.resume_from:
        if os.path.exists(args.resume_from):
            with open(args.resume_from, "r") as f:
                resume_data = json.load(f)
            print(f"📂 Loaded {len(resume_data)} completed combos from {args.resume_from}")
        else:
            print(f"⚠️  Resume file not found: {args.resume_from}, starting fresh")

    if args.quick:
        n_trials = 1
        n_epochs = 8
        lr_grid = [0.01, 0.02]
        hidden_grid = [8, 16]
        key_grid = [8, 16]
    else:
        n_trials = args.n_trials
        n_epochs = args.n_epochs
        lr_grid = [0.005, 0.01, 0.02, 0.05]
        hidden_grid = [8, 16, 32]
        key_grid = [8, 16, 32]

    combos = list(product(lr_grid, hidden_grid, key_grid))
    n_combos = len(combos)
    n_total = n_combos * n_trials

    # ── 从断点数据继续 ──
    all_results = list(resume_data)
    skipped = 0

    print("=" * 70)
    print("M4 Hyperparameter Search")
    print("=" * 70)
    print(f"  Search space: lr={lr_grid}, hidden_dim={hidden_grid}, key_dim={key_grid}")
    print(f"  Combos: {n_combos}, Trials per combo: {n_trials}, Total runs: {n_total}")
    print(f"  Epochs per trial: {n_epochs}")
    if resume_data:
        print(f"  Resume: {len(resume_data)} combos already completed")
    print()

    for combo_idx, (lr, hd, kd) in enumerate(combos):
        # 跳过已完成的组合
        if _combo_done(lr, hd, kd, resume_data):
            skipped += 1
            print(f"  [{combo_idx+1}/{n_combos}] lr={lr:.3f}, hd={hd}, kd={kd} ⏭️  skipped (resume)")
            continue

        combo_trials = []
        for trial in range(n_trials):
            seed = 42 + trial * 100 + combo_idx
            run_id = f"[{combo_idx+1}/{n_combos}][{trial+1}/{n_trials}]"
            print(f"  {run_id} lr={lr:.3f}, hd={hd}, kd={kd} ... ", end="", flush=True)

            result = run_trial(
                lr=lr, hidden_dim=hd, key_dim=kd,
                n_epochs=n_epochs,
                seed=seed,
            )

            if result.get("error"):
                print(f"❌ {result['error']}")
                result["final_loss"] = 999.0
                result["edge_f1"] = 0.0
                result["convergence_pct"] = 0.0
            else:
                print(f"loss={result['final_loss']:.4f}, f1={result['edge_f1']:.3f}, "
                      f"conv={result['convergence_pct']:.1f}%, {result['elapsed_sec']:.1f}s")

            combo_trials.append(result)

        # 聚合
        valid = [t for t in combo_trials if t.get("final_loss", 999) < 999]
        if valid:
            avg_loss = sum(t["final_loss"] for t in valid) / len(valid)
            avg_f1 = sum(t["edge_f1"] for t in valid) / len(valid)
            avg_conv = sum(t["convergence_pct"] for t in valid) / len(valid)
            avg_time = sum(t["elapsed_sec"] for t in valid) / len(valid)
            all_results.append({
                "lr": lr, "hidden_dim": hd, "key_dim": kd,
                "avg_loss": round(avg_loss, 6),
                "avg_f1": round(avg_f1, 4),
                "avg_convergence": round(avg_conv, 1),
                "avg_time_sec": round(avg_time, 2),
                "n_valid_trials": len(valid),
                "trials": combo_trials,
            })

    # ── 排序 (Edge F1 降序, 然后 loss 升序) ──
    all_results.sort(key=lambda r: (-r["avg_f1"], r["avg_loss"]))

    print("\n" + "=" * 70)
    print("Top Results (sorted by Edge F1 ↓, then loss ↓)")
    print("=" * 70)
    print(f"{'Rank':<5} {'lr':>6} {'hd':>4} {'kd':>4} {'AvgLoss':>10} {'AvgF1':>8} {'Conv%':>7} {'Time':>7}")
    print("-" * 60)

    for i, r in enumerate(all_results[:10]):
        rank = "🏆" if i == 0 else f"#{i+1}"
        print(f"{rank:<5} {r['lr']:>6.3f} {r['hidden_dim']:>4} {r['key_dim']:>4} "
              f"{r['avg_loss']:>10.6f} {r['avg_f1']:>8.4f} {r['avg_convergence']:>6.1f}% "
              f"{r['avg_time_sec']:>6.1f}s")

    # ── 最佳组合 ──
    if all_results:
        best = all_results[0]
        print(f"\n✅ Best: lr={best['lr']:.3f}, hidden_dim={best['hidden_dim']}, "
              f"key_dim={best['key_dim']}")
        print(f"   Avg Loss: {best['avg_loss']:.6f}, Avg F1: {best['avg_f1']:.4f}, "
              f"Convergence: {best['avg_convergence']:.1f}%")

    # ── 保存结果 ──
    if args.output_json:
        out_path = args.output_json
    else:
        out_path = os.path.join(os.path.dirname(__file__), "..", "hp_search_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"   Results saved to: {out_path}")
    if skipped:
        print(f"   ({skipped} combos skipped via resume)")


if __name__ == "__main__":
    main()
