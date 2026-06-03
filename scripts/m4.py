#!/usr/bin/env python3
"""
M4 统一 CLI 入口
=================

train:    训练 M3 模型 (GAT+GNN)
search:   超参网格搜索
evaluate: 评估已训练模型的 Edge F1 / 能量守恒

用法:
    python scripts/m4.py train --input memories.jsonl --lr 0.02
    python scripts/m4.py search --quick
    python scripts/m4.py search --output-json results.json --resume-from results.json
    python scripts/m4.py evaluate --checkpoint ./checkpoints/m4-inference-test/
"""

from __future__ import annotations

import sys
import os

# 确保项目根目录及 src/ 在 sys.path 中，支持包导入
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def cmd_train(args):
    """训练 M3 模型。"""
    from scripts.train_m3_real import main as train_main
    # 构造 argparse namespace 以兼容 train_m3_real.main()
    import argparse as _argparse
    ns = _argparse.Namespace(
        input=args.input,
        synthetic=args.synthetic,
        lite_pro=args.lite_pro,
        output=args.output,
        lr=args.lr,
        epochs=args.epochs,
    )
    # 直接调用训练函数（train_m3_real 内部也是 argparse，需要桥接）
    from scripts.train_m3_real import load_memories_from_file, load_memories_from_lite_pro
    from scripts.train_m3_e2e import generate_causal_timeline
    from su_memory.sdk._world_model import MCIWorldModel
    from su_memory.sdk._jepa_dataset import JEPADataset
    from su_memory.sdk._jepa_trainer import JEPATrainer
    from scripts.train_m3_real import save_checkpoint

    print("=" * 60)
    print(f"M4 Train — lr={args.lr}, epochs={args.epochs}")
    print("=" * 60)

    if args.input:
        memories = load_memories_from_file(args.input)
        print(f"Loaded {len(memories)} memories from {args.input}")
    elif args.lite_pro:
        memories = load_memories_from_lite_pro()
        print(f"Loaded {len(memories)} memories from lite_pro")
    elif args.synthetic:
        memories = generate_causal_timeline(n_timesteps=50, n_memories_per_step=10)
        print(f"Generated {len(memories)} synthetic memories")
    else:
        memories = generate_causal_timeline(n_timesteps=50, n_memories_per_step=10)
        print(f"Generated {len(memories)} synthetic memories (default)")

    wm = MCIWorldModel()
    wm.initialize()
    dataset = JEPADataset.from_memories(wm, memories, window_size=10)
    wm.enable_m3(encoder_key_dim=16, predictor_hidden_dim=8)

    trainer = JEPATrainer(
        encoder=wm._jepa_encoder,
        predictor=wm._jepa_predictor,
        dataset=dataset,
        alpha_energy=0.1,
        beta_cons=0.05,
    )
    stats = trainer.train(n_epochs=args.epochs, learning_rate=args.lr)
    print(f"\nLoss: {stats.loss_history[0]:.4f} → {stats.loss_history[-1]:.4f}")

    if args.output:
        save_checkpoint(wm, args.output)
        print(f"Checkpoint saved to {args.output}/")

    print("✅ Training complete")


def cmd_search(args):
    """超参网格搜索。"""
    import subprocess
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "hp_search_m4.py")]
    if args.quick:
        cmd.append("--quick")
    if args.output_json:
        cmd.extend(["--output-json", args.output_json])
    if args.resume_from:
        cmd.extend(["--resume-from", args.resume_from])
    if args.n_trials:
        cmd.extend(["--n-trials", str(args.n_trials)])
    if args.n_epochs:
        cmd.extend(["--n-epochs", str(args.n_epochs)])
    subprocess.run(cmd, check=True)


def cmd_evaluate(args):
    """评估已训练模型。"""
    from scripts.eval_m3_quality import main as eval_main
    import subprocess
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "eval_m3_quality.py")]
    subprocess.run(cmd, check=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="M4 Unified CLI — Train, Search, Evaluate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/m4.py train --synthetic --lr 0.02 --epochs 15
  python scripts/m4.py train --input memories.jsonl --output ./ckpt/
  python scripts/m4.py search --quick
  python scripts/m4.py search --output-json results.json --resume-from results.json
  python scripts/m4.py evaluate
        """,
    )
    sub = parser.add_subparsers(dest="command", help="Subcommand")

    # train
    p_train = sub.add_parser("train", help="Train M3 model (GAT+GNN)")
    p_train.add_argument("--input", type=str, default=None, help="JSONL file with memories")
    p_train.add_argument("--synthetic", action="store_true", help="Use synthetic causal timeline")
    p_train.add_argument("--lite-pro", action="store_true", help="Load from lite_pro backend")
    p_train.add_argument("--output", type=str, default=None, help="Checkpoint output directory")
    p_train.add_argument("--lr", type=float, default=0.02, help="Learning rate (default: 0.02)")
    p_train.add_argument("--epochs", type=int, default=15, help="Number of epochs (default: 15)")

    # search
    p_search = sub.add_parser("search", help="Hyperparameter grid search")
    p_search.add_argument("--quick", action="store_true", help="Quick mode: fewer combos/epochs")
    p_search.add_argument("--n-trials", type=int, default=None, help="Trials per combo")
    p_search.add_argument("--n-epochs", type=int, default=None, help="Epochs per trial")
    p_search.add_argument("--output-json", type=str, default=None, help="Save results to JSON")
    p_search.add_argument("--resume-from", type=str, default=None, help="Resume from previous JSON")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate trained model quality")
    p_eval.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint directory (uses eval_m3_quality defaults)")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
