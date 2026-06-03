#!/usr/bin/env python3
"""
M4 真实记忆训练脚本
====================

从 disk 文件或 su-memory SDK 内建存储加载真实记忆序列，
执行 M3 端到端训练并评估。

数据格式 (JSON lines):
    {"content": "memory text here", "timestamp": "2024-01-01T00:00:00"}
    {"content": "another memory", "timestamp": "2024-01-01T01:00:00"}

用法:
    # 从文件加载
    python scripts/train_m3_real.py --input memories.jsonl

    # 使用默认合成数据 + 最优参数
    python scripts/train_m3_real.py --synthetic

    # 从 SDK lite_pro 存储加载
    python scripts/train_m3_real.py --lite-pro
"""

import sys
import os
import time
import json
import argparse

# 确保项目根目录及 src/ 在 sys.path 中，支持包导入
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# =============================================================================
# 记忆加载
# =============================================================================

def load_memories_from_file(path: str) -> list[dict]:
    """从 JSONL 文件加载记忆。"""
    memories = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                mem = json.loads(line)
                if "content" in mem:
                    memories.append(mem)
            except json.JSONDecodeError:
                # 普通文本行 → 当作 content
                memories.append({"content": line, "timestamp": ""})
    return memories


def load_memories_from_dir(path: str) -> list[dict]:
    """从目录加载所有 .json 文件中的记忆。"""
    memories = []
    for fname in sorted(os.listdir(path)):
        if fname.endswith(".json") or fname.endswith(".jsonl"):
            filepath = os.path.join(path, fname)
            memories.extend(load_memories_from_file(filepath))
    return memories


def load_memories_from_lite_pro() -> list[dict]:
    """从 SDK 内建的 lite_pro 加载记忆（需要已初始化的 MCIWorldModel）。"""
    from su_memory.sdk._world_model import MCIWorldModel

    wm = MCIWorldModel()
    wm.initialize()

    memories = wm._get_memories_from_lite_pro()
    if memories:
        return memories

    # 尝试 lite_pro 直接访问
    try:
        if hasattr(wm, "_lite_pro") and wm._lite_pro is not None:
            lite = wm._lite_pro
            if hasattr(lite, "query"):
                results = lite.query("*", top_k=200)
                return [
                    {"id": r.get("id", str(i)), "content": r.get("content", "")}
                    for i, r in enumerate(results)
                ]
    except Exception:
        pass

    return []


# =============================================================================
# 训练 + 评估
# =============================================================================

def train_on_real_data(
    memories: list[dict],
    learning_rate: float = 0.02,
    hidden_dim: int = 8,
    key_dim: int = 16,
    n_epochs: int = 20,
    window_size: int = 10,
    checkpoint_dir: str | None = None,
) -> dict:
    """在真实记忆上训练 M3 模型。"""
    import numpy as np
    from su_memory.sdk._world_model import MCIWorldModel
    from su_memory.sdk._jepa_dataset import JEPADataset
    from su_memory.sdk._jepa_trainer import JEPATrainer

    print("=" * 60)
    print("M4 真实记忆 M3 训练")
    print("=" * 60)
    print(f"  Memories: {len(memories)}")
    print(f"  Params: lr={learning_rate}, hidden_dim={hidden_dim}, key_dim={key_dim}")
    print(f"  Epochs: {n_epochs}, Window: {window_size}")

    # ── 1. World Model + Dataset ──
    wm = MCIWorldModel()
    wm.initialize()
    dataset = JEPADataset.from_memories(wm, memories, window_size=window_size)
    print(f"  Dataset: {dataset.n_states} states, {dataset.n_pairs} pairs, "
          f"{dataset.n_memory_pairs} memory pairs")

    if dataset.n_pairs < 3:
        print("  ⚠️  Not enough pairs for training")
        return {"error": "insufficient_pairs"}

    # ── 2. Enable M3 ──
    wm.enable_m3(encoder_key_dim=key_dim, predictor_hidden_dim=hidden_dim)

    # ── 3. Train ──
    print(f"\n  Training {n_epochs} epochs...")
    t0 = time.time()
    trainer = JEPATrainer(
        encoder=wm._jepa_encoder,
        predictor=wm._jepa_predictor,
        dataset=dataset,
        alpha_energy=0.1,
        beta_cons=0.05,
    )
    stats = trainer.train(n_epochs=n_epochs, learning_rate=learning_rate)
    elapsed = time.time() - t0

    loss_history = stats.loss_history
    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Loss: {loss_history[0]:.6f} → {loss_history[-1]:.6f} "
          f"({(1-loss_history[-1]/max(loss_history[0],1e-10))*100:.1f}% decrease)")

    # ── 4. Evaluate edge quality ──
    from su_memory.sdk._jepa_gat_encoder import preprocess_memories_to_features, features_to_state
    from su_memory.sdk._jepa_gnn import align_adjacency
    from su_memory.sdk._world_model import CausalWorldModelState

    f1_scores = []
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
        f1_scores.append(f1)

    avg_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    print(f"  Edge F1 (trained): {avg_f1:.4f} ({len(f1_scores)} valid pairs)")

    # ── 5. Energy conservation ──
    from scripts.train_m3_e2e import compute_energy_conservation
    energy = compute_energy_conservation(wm, dataset)
    cons = energy.get("conservation_score", 0)
    print(f"  Energy conservation: {cons:.4f}")

    result = {
        "n_memories": len(memories),
        "n_pairs": dataset.n_pairs,
        "n_epochs": n_epochs,
        "lr": learning_rate,
        "hidden_dim": hidden_dim,
        "key_dim": key_dim,
        "initial_loss": round(loss_history[0], 6) if loss_history else 0,
        "final_loss": round(loss_history[-1], 6) if loss_history else 0,
        "edge_f1": round(avg_f1, 4),
        "conservation_score": round(cons, 4),
        "elapsed_sec": round(elapsed, 1),
    }

    # ── 6. Save checkpoint ──
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_checkpoint(wm, checkpoint_dir)
        print(f"  Checkpoint saved to: {checkpoint_dir}")

    return result


# =============================================================================
# Save/Load
# =============================================================================

def save_checkpoint(wm, checkpoint_dir: str):
    """Save trained GAT + GNN weights to disk."""
    import numpy as np

    os.makedirs(checkpoint_dir, exist_ok=True)

    # GAT weights
    if wm._jepa_encoder and wm._jepa_encoder.gat_encoder:
        gat = wm._jepa_encoder.gat_encoder
        np.savez(
            os.path.join(checkpoint_dir, "gat_weights.npz"),
            W_q=gat.W_q,
            W_k=gat.W_k,
            key_dim=gat._key_dim,
            input_dim=gat._input_dim,
        )

    # GNN weights
    if wm._jepa_predictor:
        gnn = wm._jepa_predictor
        hidden_dim = gnn._hidden_dim
        np.savez(
            os.path.join(checkpoint_dir, "gnn_weights.npz"),
            W1=gnn.W1,
            W2=gnn.W2,
            W3=gnn.W3,
            hidden_dim=hidden_dim,
        )
        with open(os.path.join(checkpoint_dir, "gnn_config.json"), "w") as f:
            json.dump({"hidden_dim": hidden_dim, "train_steps": gnn._train_steps}, f)


def load_checkpoint(wm, checkpoint_dir: str):
    """Load trained weights into world model."""
    import numpy as np

    # GAT
    gat_path = os.path.join(checkpoint_dir, "gat_weights.npz")
    if os.path.exists(gat_path) and wm._jepa_encoder and wm._jepa_encoder.gat_encoder:
        data = np.load(gat_path)
        wm._jepa_encoder.gat_encoder.W_q = data["W_q"]
        wm._jepa_encoder.gat_encoder.W_k = data["W_k"]
        print(f"  Loaded GAT weights (key_dim={data['key_dim']})")

    # GNN
    gnn_path = os.path.join(checkpoint_dir, "gnn_weights.npz")
    if os.path.exists(gnn_path) and wm._jepa_predictor:
        data = np.load(gnn_path)
        wm._jepa_predictor.W1 = data["W1"]
        wm._jepa_predictor.W2 = data["W2"]
        wm._jepa_predictor.W3 = data["W3"]
        print(f"  Loaded GNN weights (hidden_dim={data['hidden_dim']})")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="M4 Real Memory Training")
    parser.add_argument("--input", "-i", type=str, help="JSONL file or directory of memories")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--lite-pro", action="store_true", help="Load from SDK lite_pro")
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--key-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/m4-real")
    args = parser.parse_args()

    # ── Load memories ──
    if args.input:
        if os.path.isdir(args.input):
            memories = load_memories_from_dir(args.input)
        else:
            memories = load_memories_from_file(args.input)
        print(f"Loaded {len(memories)} memories from {args.input}")
    elif args.lite_pro:
        memories = load_memories_from_lite_pro()
        print(f"Loaded {len(memories)} memories from lite_pro")
    elif args.synthetic:
        from scripts.train_m3_e2e import generate_causal_timeline
        memories = generate_causal_timeline(n_timesteps=50, n_memories_per_step=10)
        print(f"Generated {len(memories)} synthetic memories")
    else:
        print("Error: need --input, --synthetic, or --lite-pro")
        return

    if len(memories) < 10:
        print(f"⚠️  Not enough memories ({len(memories)} < 10)")
        return

    result = train_on_real_data(
        memories=memories,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        key_dim=args.key_dim,
        n_epochs=args.epochs,
        checkpoint_dir=args.checkpoint,
    )

    print("\n" + "=" * 60)
    if result.get("error"):
        print(f"❌ Failed: {result['error']}")
    else:
        print(f"✅ Training complete: F1={result['edge_f1']:.4f}, "
              f"Loss={result['final_loss']:.6f}")


if __name__ == "__main__":
    main()
