#!/usr/bin/env python3
"""
su-memory SDK v2.0 快速体验脚本
测试：写入记忆 → 语义检索 → 因果关联 → 多跳推理
"""

import sys
sys.path.insert(0, "src")

from su_memory import SuMemory

# ── 1. 初始化 ──────────────────────────────────────────────
print("=" * 60)
print("🚀 初始化 SuMemory 客户端...")
client = SuMemory(persist_dir="./demo_data")
print("✅ 初始化完成\n")

# ── 2. 写入记忆 ──────────────────────────────────────────────
print("=" * 60)
print("📝 写入测试记忆（Nutri-Brain 项目相关）...")

memories = [
    ("Nutri-Brain 目标成为三甲医院营养科首选 AI 供应商", {"source": "strategy"}),
    ("首轮融资目标 500 万元，用于产品研发和医院对接", {"source": "finance"}),
    ("核心算法基于代谢组学 + 大语言模型", {"source": "tech"}),
    ("Q3 计划对接 3 家三甲医院营养科", {"source": "milestone"}),
    ("当前团队 5 人，CTO 来自医疗 AI 背景", {"source": "team"}),
    ("投资人关注：市场规模、差异化、合规路径", {"source": "investor"}),
    ("竞品定价过高，医院采购决策周期长", {"source": "market"}),
    ("营养科主任反馈：最关注患者依从性数据", {"source": "feedback"}),
    ("种子轮估值建议不超过 2000 万", {"source": "finance"}),
    ("需要补齐 II 类医疗器械证", {"source": "compliance"}),
]

ids = []
for content, meta in memories:
    mid = client.add(content, metadata=meta)
    ids.append(mid)
    print(f"  ✅ [{mid}] {content[:40]}...")

print(f"\n共写入 {len(ids)} 条记忆\n")

# ── 3. 语义检索 ──────────────────────────────────────────────
print("=" * 60)
print("🔍 语义检索测试")

queries = [
    "融资和估值",
    "医院合作",
    "技术产品",
]

for q in queries:
    results = client.query(q, top_k=3)
    print(f"\n  Query:「{q}」")
    for r in results:
        enc = r.encoding
        extra = f" | category:{enc.category} / energy:{enc.energy_type}" if hasattr(enc, 'category') else ""
        print(f"    → [{r.score:.3f}] {r.content[:40]}...{extra}")

# ── 4. 因果关联 ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("🔗 建立因果关联")

links = [
    (ids[0], ids[1], "目标→需要资金支撑"),
    (ids[1], ids[8], "融资→影响估值"),
    (ids[2], ids[3], "技术→支撑落地目标"),
    (ids[3], ids[7], "医院对接→获得真实反馈"),
    (ids[7], ids[0], "反馈→反哺产品定位"),
]

for src, dst, label in links:
    client.link(src, dst)
    print(f"  ✅ {src} → {dst}（{label}）")

# ── 5. 多跳推理 ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("🧠 多跳推理测试")

questions = [
    "技术能力如何影响融资？",
    "医院反馈如何影响产品定位？",
]

for q in questions:
    chain = client.query_multihop(q, top_k=3)
    print(f"\n  Query:「{q}」")
    if chain:
        for i, step in enumerate(chain):
            if hasattr(step, 'content'):
                print(f"    Step {i+1}: {step.content[:50]}...")
            else:
                print(f"    Step {i+1}: {str(step)[:50]}")
    else:
        print("    （未找到推理路径）")

# ── 6. 统计信息 ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("📊 统计信息")

stats = client.get_stats()
for k, v in stats.items():
    print(f"  {k}: {v}")

print("\n" + "=" * 60)
print("🎉 体验完成！")
print("如需清理数据，删除 demo_data 目录即可")
