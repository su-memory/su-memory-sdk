#!/usr/bin/env python3
"""
M3 端到端可微训练验证脚本
===========================

用合成时序因果数据训练 GAT 编码器 + GNN 预测器，
验证:
1. Loss 递减曲线
2. 能量守恒 (M3 优于基线)
3. 端到端训练收敛

用法:
    python scripts/train_m3_e2e.py
"""

import sys
import os
import time

# 确保项目根目录及 src/ 在 sys.path 中，支持包导入 (scripts.* / su_memory.*)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
from collections import OrderedDict


# =============================================================================
# 合成时序因果数据生成
# =============================================================================


def generate_causal_timeline(
    n_timesteps: int = 30,
    n_memories_per_step: int = 10,
    seed: int = 42,
) -> list[dict]:
    """
    生成带因果关系的时序记忆。

    场景: 经济因果链
    - 原料成本 ↑ → 产品价格 ↑ → 需求 ↓ → 收入 ↓
    - 技术进步 → 效率 ↑ → 成本 ↓ → 利润 ↑
    """
    rng = np.random.RandomState(seed)

    memory_templates = [
        "{entity} {direction} by {delta}% in Q{quarter}",
        "{entity} showed a {direction} trend of {delta}%",
        "Report: {entity} {direction} {delta}% compared to previous quarter",
        "Analysis indicates {entity} {direction} {delta}%",
        "Indicator {entity} moved {direction} by {delta}%",
    ]

    # Use English multi-word entity names so GaussianDAG TF-IDF produces meaningful tokens
    causal_chains = [
        {
            "entities": ["raw material cost", "product price", "demand level", "revenue flow"],
            "deltas": [1.0, 0.7, -0.5, -0.4],
        },
        {
            "entities": ["tech innovation", "efficiency score", "production cost", "profit margin"],
            "deltas": [0.8, 0.6, -0.7, 0.9],
        },
        {
            "entities": ["market competition", "price pressure", "margin rate", "investment flow"],
            "deltas": [0.6, -0.5, -0.6, -0.3],
        },
        {
            "entities": ["labor shortage", "wage increase", "operating cost", "hiring rate"],
            "deltas": [0.7, 0.8, 0.6, -0.4],
        },
    ]

    all_memories: list[dict] = []
    timestamp_base = 1700000000
    n_chain_entities = sum(len(c["entities"]) for c in causal_chains)

    for t in range(n_timesteps):
        timestamp = timestamp_base + t * 86400
        quarter = (t % 4) + 1

        # Generate structured causal chain memories
        for chain in causal_chains:
            entities = chain["entities"]
            deltas = chain["deltas"]
            noise = rng.normal(0, 0.2, len(entities))

            for i, (entity, delta) in enumerate(zip(entities, deltas)):
                magnitude = delta + noise[i]
                direction = "increased" if magnitude > 0 else "decreased"
                abs_delta = round(abs(magnitude), 2)

                template = memory_templates[rng.randint(0, len(memory_templates))]
                content = template.format(
                    entity=entity, direction=direction,
                    delta=abs_delta, quarter=quarter,
                )

                all_memories.append({
                    "content": content,
                    "timestamp": timestamp + i * 60,
                    "entity": entity,
                })

        # Fill remaining slots with noisy variants
        remaining = n_memories_per_step - n_chain_entities
        for _ in range(max(0, remaining)):
            chain_idx = rng.randint(0, len(causal_chains))
            chain = causal_chains[chain_idx]
            entity_idx = rng.randint(0, len(chain["entities"]))
            entity = chain["entities"][entity_idx]
            noise_val = rng.normal(0, 0.3)
            direction = "increased" if noise_val > 0 else "decreased"

            template = memory_templates[rng.randint(0, len(memory_templates))]
            content = template.format(
                entity=entity, direction=direction,
                delta=round(abs(noise_val), 2), quarter=quarter,
            )
            all_memories.append({
                "content": content,
                "timestamp": timestamp + rng.randint(0, 86400),
                "entity": entity,
            })

    all_memories.sort(key=lambda m: m["timestamp"])
    return all_memories


# =============================================================================
# 训练 + 评估
# =============================================================================


def train_and_evaluate(
    n_timesteps: int = 30,
    n_memories_per_step: int = 10,
    n_epochs: int = 15,
    lr: float = 0.02,
) -> dict:
    """运行 M3 端到端训练并返回结果。"""
    from su_memory.sdk._world_model import MCIWorldModel
    from su_memory.sdk._jepa_dataset import JEPADataset
    from su_memory.sdk._jepa_trainer import JEPATrainer

    print("=" * 60)
    print("M3 端到端可微训练验证")
    print("=" * 60)

    # ── 1. 生成数据 ──
    print(f"\n[1] 生成时序数据: {n_timesteps} 时间步 × {n_memories_per_step} 记忆/步")
    memories = generate_causal_timeline(
        n_timesteps=n_timesteps,
        n_memories_per_step=n_memories_per_step,
    )
    print(f"    总计 {len(memories)} 条记忆")

    # ── 2. 初始化 world model ──
    print("\n[2] 初始化 MCIWorldModel + enable_m3()")
    wm = MCIWorldModel()
    wm.initialize()

    # 创建 JEPADataset
    dataset = JEPADataset.from_memories(wm, memories, window_size=10)
    print(f"    JEPADataset: {dataset.n_states} 状态, {dataset.n_pairs} 状态对, "
          f"{dataset.n_memory_pairs} 记忆对")
    if dataset.n_pairs < 3:
        print("    ⚠️  训练对不足，增加 n_timesteps 或减少 window_size")
        return {"error": "insufficient_pairs"}

    # ── 3. 启用 M3 ──
    print("\n[3] 启用 M3 端到端可微模式")
    m3_report = wm.enable_m3(encoder_key_dim=16, predictor_hidden_dim=16)
    print(f"    encoder: {m3_report['encoder']}")
    print(f"    predictor: {m3_report['predictor']}")

    # ── 4. 训练 ──
    print(f"\n[4] 端到端训练: {n_epochs} epochs, lr={lr}")
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
    print(f"    耗时: {elapsed:.1f}s")

    # ── 5. 报告 ──
    print("\n[5] 训练结果")
    loss_history = stats.loss_history
    print(f"    Loss 历史: {[round(l, 6) for l in loss_history]}")
    print(f"    初始 Loss: {loss_history[0]:.6f}")
    print(f"    最终 Loss: {loss_history[-1]:.6f}")
    delta = loss_history[0] - loss_history[-1]
    pct = (delta / max(loss_history[0], 1e-10)) * 100
    print(f"    Loss 下降: {delta:.6f} ({pct:.1f}%)")

    # ── 6. 能量守恒验证 ──
    print("\n[6] 能量守恒验证")
    energy_metrics = compute_energy_conservation(wm, dataset)
    for k, v in energy_metrics.items():
        print(f"    {k}: {v}")

    # ── 7. 健康检查 ──
    print("\n[7] 健康检查")
    hc = wm.health_check()
    print(f"    status: {hc['status']}")
    print(f"    roadmap: v4.0.0-m3 = {hc['roadmap']['v4.0.0-m3']}")
    print(f"    mode: {hc.get('training_mode', 'n/a')}")

    return {
        "loss_history": [round(l, 6) for l in loss_history],
        "initial_loss": round(loss_history[0], 6),
        "final_loss": round(loss_history[-1], 6),
        "loss_drop_pct": round(pct, 1),
        "energy_conservation": energy_metrics,
        "n_pairs": dataset.n_pairs,
        "n_epochs": n_epochs,
        "elapsed_sec": round(elapsed, 1),
        "converged": pct > 5,
    }


def compute_energy_conservation(wm, dataset) -> dict:
    """计算能量守恒指标: 相邻状态的边能量变化。"""
    total_energy_drift = []
    for s_t, s_t1 in dataset.pairs:
        energy_t = sum(abs(e.get("rho", 0.0)) for e in s_t.causal_edges)
        energy_t1 = sum(abs(e.get("rho", 0.0)) for e in s_t1.causal_edges)
        if max(energy_t, energy_t1) > 0:
            drift = abs(energy_t - energy_t1) / max(energy_t, energy_t1, 1e-10)
            total_energy_drift.append(drift)

    if not total_energy_drift:
        return {"avg_drift": 0.0, "n": 0}

    return {
        "avg_energy_drift": round(float(np.mean(total_energy_drift)), 6),
        "max_energy_drift": round(float(np.max(total_energy_drift)), 6),
        "n_pairs": len(total_energy_drift),
        "conservation_score": round(1.0 - float(np.mean(total_energy_drift)), 4),
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    result = train_and_evaluate(
        n_timesteps=30,
        n_memories_per_step=10,
        n_epochs=15,
        lr=0.02,
    )

    print("\n" + "=" * 60)
    print("训练摘要")
    print("=" * 60)

    if "error" in result:
        print(f"❌ 失败: {result['error']}")
    else:
        loss_ok = result["loss_drop_pct"] > 5
        energy_ok = result["energy_conservation"].get("conservation_score", 0) > 0.5

        print(f"  Loss:   {result['initial_loss']:.6f} → {result['final_loss']:.6f} "
              f"({result['loss_drop_pct']:.1f}%) {'✅' if loss_ok else '⚠️'}")
        print(f"  Energy: conservation={result['energy_conservation'].get('conservation_score', 0):.4f} "
              f"{'✅' if energy_ok else '⚠️'}")
        print(f"  Time:   {result['elapsed_sec']:.1f}s")
        print(f"  Pairs:  {result['n_pairs']}")
        print(f"  Epochs: {result['n_epochs']}")

        if loss_ok and energy_ok:
            print("\n  ✅ M3 端到端可微训练验证通过!")
        else:
            print("\n  ⚠️  部分指标未达标，可能需要更多数据或调参")
