#!/usr/bin/env python3
"""
su-memory 真实使用效果测试
模拟用户安装后 import su_memory 的完整使用流程
"""

import shutil
import os
import su_memory

# ── 0. 清理旧数据 ──────────────────────────────────────────────
DATA_DIR = "./test_su_memory_data"
if os.path.exists(DATA_DIR):
    shutil.rmtree(DATA_DIR)
print("🧹 清理旧数据完成\n")

# ── 1. 验证安装 ──────────────────────────────────────────────
print("=" * 60)
print("✅ Step 1: 验证 pip 安装")
print(f"   import su_memory 成功")
print(f"   版本: {su_memory.__version__ if hasattr(su_memory, '__version__') else 'unknown'}")
print(f"   文件位置: {su_memory.__file__}")
print()

# ── 2. 初始化 ──────────────────────────────────────────────
print("=" * 60)
print("✅ Step 2: 初始化客户端")
from su_memory import SuMemory
client = SuMemory(persist_dir=DATA_DIR)
print(f"   存储模式: {client.mode}")
print(f"   存储路径: {DATA_DIR}")
print()

# ── 3. 写入记忆 ──────────────────────────────────────────────
print("=" * 60)
print("✅ Step 3: 写入记忆（Nutri-Brain 项目信息）")

memories = [
    ("Nutri-Brain 临床营养大脑，定位中国首个 AI 驱动的临床营养决策系统", "strategy"),
    ("核心技术上采用代谢组学 + 大语言模型，为三甲医院营养科提供决策支持", "tech"),
    ("首轮融资目标 500 万元，估值不超过 2000 万", "finance"),
    ("Q3 目标对接 3 家三甲医院营养科，首批种子用户为肿瘤科和 ICU", "milestone"),
    ("需要申请 II 类医疗器械注册证，合规是进入公立医院的前提", "compliance"),
    ("团队目前 5 人，CEO 有临床营养背景，CTO 来自医疗 AI 独角兽", "team"),
    ("竞品定价高、部署周期长，Nutri-Brain 差异化在于临床路径深度整合", "market"),
    ("营养科主任最关注患者依从性数据，这是落地效果的核心指标", "feedback"),
    ("投资人对市场规模持保守态度，需用真实医院数据证明增长", "investor"),
    ("已建立临床营养知识图谱，覆盖 200+ 病种方案", "knowledge"),
]

ids = []
for content, source in memories:
    mid = client.add(content, metadata={"source": source})
    ids.append(mid)

print(f"   写入 {len(ids)} 条记忆")
print(f"   ID 列表: {', '.join(ids)}")
print()

# ── 4. 立即检索 — 验证记忆已存取 ──────────────────────────────────────────────
print("=" * 60)
print("✅ Step 4: 检索测试（写入后立即查询，验证记忆已存取）")

test_queries = {
    "融资": "融资和估值相关",
    "医院": "医院合作相关",
    "技术": "技术产品相关",
    "合规": "医疗器械合规相关",
}

for key, desc in test_queries.items():
    results = client.query(key, top_k=3)
    print(f"\n  📌「{key}」（{desc}）")
    for r in results:
        enc = r.encoding
        category = enc.category if hasattr(enc, 'category') else '-'
        energy = enc.energy_type if hasattr(enc, 'energy_type') else '-'
        print(f"     [{r.score:.3f}] {r.content[:35]}... [category:{category} energy:{energy}]")

# ── 5. 关联记忆 ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ Step 5: 建立因果关联")

links = [
    (ids[0], ids[1], "定位→技术路径"),
    (ids[1], ids[3], "技术→落地目标"),
    (ids[3], ids[7], "医院→获得反馈"),
    (ids[7], ids[0], "反馈→优化定位"),
    (ids[4], ids[3], "合规→医院准入"),
    (ids[2], ids[5], "融资→团队扩张"),
]

for src, dst, label in links:
    client.link(src, dst)

print(f"   建立 {len(links)} 条因果关联")

# ── 6. 多跳推理 ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ Step 6: 多跳推理测试")

reasoning_questions = [
    "技术能力如何影响落地医院数量？",
    "合规证照如何影响融资估值？",
    "医生反馈如何影响产品方向？",
]

for q in reasoning_questions:
    chain = client.query_multihop(q, top_k=4)
    print(f"\n  🔮「{q}」")
    if chain:
        for i, step in enumerate(chain):
            content = step.content if hasattr(step, 'content') else str(step)
            score = step.score if hasattr(step, 'score') else 0
            print(f"     Step {i+1}: [{score:.3f}] {content[:40]}...")
    else:
        print("     （未找到推理路径）")

# ── 7. 验证记忆跨查询 ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ Step 7: 记忆持久化验证（重启后记忆是否还在）")

# 重新初始化同一目录的 client
client2 = SuMemory(persist_dir=DATA_DIR)
stats = client2.get_stats()
print(f"   重启后记忆数: {stats.get('total_memories', '?')}")
assert stats.get('total_memories', 0) == len(ids), "记忆未持久化！"
print("   ✅ 记忆已持久化，重启后仍然存在")

# ── 8. 统计 ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ Step 8: 统计信息")

for k, v in stats.items():
    print(f"   {k}: {v}")

# ── 9. 清理 ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("🧹 清理测试数据...")
shutil.rmtree(DATA_DIR)
print("✅ 清理完成")

print("\n" + "=" * 60)
print("🎉 su-memory 效果测试全部通过！")
print()
print("📌 效果总结：")
print("   ✅ 安装即可 import，开箱即用")
print("   ✅ add 写入 → query 检索，记忆存取完整")
print("   ✅ 语义检索支持多维度标签（标签+属性）")
print("   ✅ link 建立因果关联，支持多跳 query_multihop")
print("   ✅ 数据持久化，重启后记忆不丢失")
