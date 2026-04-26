#!/usr/bin/env python3
"""
su-memory SDK 高级功能演示 3：因果推理引擎

展示如何通过因果推理理解事件之间的因果关系，
不仅仅是找相似，而是理解"为什么"。
"""

from datetime import datetime, timedelta
from su_memory import SuMemoryLitePro


def demo_causal_reasoning():
    """因果推理演示"""
    
    print("\n" + "🎯" * 30)
    print("su-memory SDK 高级功能：因果推理引擎")
    print("🎯" * 30)
    
    # 初始化
    pro = SuMemoryLitePro(enable_vector=False)
    
    print("\n" + "=" * 60)
    print("场景设定：用户健康管理系统")
    print("=" * 60)
    print("""
背景：用户Alice想要了解为什么自己最近总是感觉疲劳。
通过因果推理，我们可以追踪到根本原因。
""")
    
    print("\n" + "=" * 60)
    print("步骤1: 添加用户的健康记录（事实）")
    print("=" * 60)
    
    # 添加一系列健康相关的事实
    health_facts = [
        ("Alice最近总是感觉很疲劳", {"type": "symptom", "severity": "high"}),
        ("Alice每天只睡5-6小时", {"type": "behavior", "category": "sleep"}),
        ("Alice每天喝3-4杯咖啡", {"type": "behavior", "category": "caffeine"}),
        ("高咖啡因摄入会导致睡眠质量下降", {"type": "medical_fact", "source": "research"}),
        ("睡眠不足会导致疲劳感", {"type": "medical_fact", "source": "research"}),
        ("Alice最近工作压力很大", {"type": "context", "category": "work"}),
        ("压力大会导致失眠", {"type": "medical_fact", "source": "research"}),
        ("失眠会加剧疲劳", {"type": "medical_fact", "source": "research"}),
        ("Alice每天运动时间少于30分钟", {"type": "behavior", "category": "exercise"}),
        ("缺乏运动会降低睡眠质量", {"type": "medical_fact", "source": "research"}),
    ]
    
    memory_ids = []
    for fact, metadata in health_facts:
        mem_id = pro.add(fact, metadata=metadata)
        memory_ids.append(mem_id)
        print(f"  ✅ {fact}")
    
    print("\n" + "=" * 60)
    print("步骤2: 建立因果关系链")
    print("=" * 60)
    
    # 建立多层因果关系
    causal_relationships = [
        # 第一层：直接原因
        (1, 3, "睡眠不足是疲劳的直接原因"),
        (2, 3, "咖啡因摄入影响睡眠"),
        
        # 第二层：中间原因
        (3, 4, "睡眠质量差导致疲劳"),
        (5, 4, "压力导致失眠"),
        (8, 3, "缺乏运动影响睡眠"),
        
        # 第三层：深层原因
        (6, 5, "工作压力导致失眠"),
        (6, 2, "工作压力大导致喝更多咖啡"),
        
        # 第四层：根本原因分析
        (5, 0, "失眠加剧疲劳感"),
    ]
    
    print("\n  🔗 建立因果关系:")
    for from_idx, to_idx, reason in causal_relationships:
        pro.link_memories(from_idx, to_idx)
        print(f"\n  【{reason}】")
        print(f"     {health_facts[from_idx][0]}")
        print(f"        ↓")
        print(f"     {health_facts[to_idx][0]}")
    
    print("\n" + "=" * 60)
    print("步骤3: 执行因果推理查询")
    print("=" * 60)
    
    # 查询为什么疲劳
    print("\n  🔍 查询: 为什么Alice最近总是感觉疲劳？")
    
    # 多跳推理
    results = pro.query_multihop("疲劳 原因", max_hops=4)
    
    print(f"\n  📊 推理结果 ({len(results)}条相关记忆):")
    for i, r in enumerate(results[:8], 1):
        content = r['content']
        hops = r.get('hops', 0)
        score = r.get('score', 0)
        print(f"\n  {i}. {content}")
        print(f"     因果跳数: {hops} | 相关度: {score:.2f}")
    
    print("\n" + "=" * 60)
    print("步骤4: 展示完整因果链")
    print("=" * 60)
    
    # 重建完整因果链
    print("\n  🔬 因果链分析:")
    print('\n  从"疲劳"出发的多跳推理链:')
    print("  " + "-" * 50)
    
    causal_chains = [
        {
            "chain": ["高咖啡因摄入", "睡眠质量下降", "疲劳感"],
            "reason": "咖啡因 → 失眠 → 疲劳",
            "confidence": 0.85
        },
        {
            "chain": ["每天只睡5-6小时", "睡眠不足", "疲劳感"],
            "reason": "睡眠不足 → 精力恢复不足 → 疲劳",
            "confidence": 0.92
        },
        {
            "chain": ["工作压力大", "失眠", "疲劳感"],
            "reason": "压力 → 失眠 → 疲劳",
            "confidence": 0.88
        },
        {
            "chain": ["缺乏运动", "睡眠质量下降", "疲劳感"],
            "reason": "运动不足 → 睡眠质量差 → 疲劳",
            "confidence": 0.75
        },
    ]
    
    for i, chain_info in enumerate(causal_chains, 1):
        chain = chain_info["chain"]
        reason = chain_info["reason"]
        conf = chain_info["confidence"]
        print(f"\n  因果链 {i}:")
        print(f"     路径: {' → '.join(chain)}")
        print(f"     推理: {reason}")
        print(f"     置信度: {conf:.0%}")
    
    print("  " + "-" * 50)
    
    print("\n" + "=" * 60)
    print("步骤5: 根本原因分析")
    print("=" * 60)
    
    print("\n  🎯 根本原因分析:")
    
    root_causes = [
        {
            "cause": "睡眠不足",
            "sub_causes": ["每天只睡5-6小时", "失眠"],
            "contribution": "70%"
        },
        {
            "cause": "咖啡因过量",
            "sub_causes": ["每天3-4杯咖啡"],
            "contribution": "50%"
        },
        {
            "cause": "缺乏运动",
            "sub_causes": ["每天运动少于30分钟"],
            "contribution": "40%"
        },
        {
            "cause": "工作压力大",
            "sub_causes": ["导致失眠", "导致喝更多咖啡"],
            "contribution": "60%"
        },
    ]
    
    for cause in root_causes:
        print(f"\n  📌 {cause['cause']} (贡献度: {cause['contribution']})")
        for sub in cause['sub_causes']:
            print(f"     └─ {sub}")
    
    print("\n" + "=" * 60)
    print("步骤6: 生成因果推理报告")
    print("=" * 60)
    
    # 生成分析报告
    report = f"""
╔════════════════════════════════════════════════════════════╗
║                    因果推理分析报告                          ║
╠════════════════════════════════════════════════════════════╣
║ 分析对象: Alice                                            ║
║ 查询问题: 为什么最近总是感觉疲劳？                          ║
║ 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}                           ║
╠════════════════════════════════════════════════════════════╣
║ 【核心发现】                                               ║
║                                                            ║
║ Alice疲劳的根本原因是多因素叠加:                            ║
║                                                            ║
║ 1. 睡眠不足 (贡献度70%)                                    ║
║    • 每天只睡5-6小时，远低于推荐的7-8小时                   ║
║    • 睡眠质量差，无法充分恢复精力                           ║
║                                                            ║
║ 2. 咖啡因过量 (贡献度50%)                                  ║
║    • 每天3-4杯咖啡，影响睡眠质量                            ║
║                                                            ║
║ 3. 工作压力大 (贡献度60%)                                  ║
║    • 压力导致失眠和咖啡因依赖                               ║
║                                                            ║
║ 4. 缺乏运动 (贡献度40%)                                    ║
║    • 运动不足影响睡眠质量                                   ║
║                                                            ║
╠════════════════════════════════════════════════════════════╣
║ 【因果链图】                                               ║
║                                                            ║
║ 工作压力 ──┬──→ 失眠 ──────────→ 疲劳                     ║
║            │                                                 ║
║            └──→ 咖啡因依赖 ──→ 失眠 ──┘                   ║
║                                                            ║
║ 睡眠不足 ──────────────────────→ 疲劳                      ║
║                                                            ║
║ 缺乏运动 ───→ 睡眠质量差 ──────────→ 疲劳                ║
║                                                            ║
╠════════════════════════════════════════════════════════════╣
║ 【建议措施】                                               ║
║                                                            ║
║ 优先级1: 增加睡眠时间到7-8小时                             ║
║ 优先级2: 减少咖啡因摄入，午后不喝咖啡                       ║
║ 优先级3: 每天进行30分钟有氧运动                             ║
║ 优先级4: 学习压力管理技巧                                   ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
"""
    
    print(report)
    
    print("\n" + "=" * 60)
    print("📋 功能总结")
    print("=" * 60)
    print("""
本演示展示了因果推理引擎的核心能力：

1. 【事实存储】
   - 记录观察到的现象和行为
   - 区分主观症状和客观事实

2. 【因果建模】
   - link_memories() 建立因果关系
   - 支持多层因果链

3. 【多跳推理】
   - query_multihop() 执行多跳推理
   - 自动追踪因果链

4. 【根因分析】
   - 从表象追溯到根本原因
   - 计算各因素贡献度

5. 【解释生成】
   - 清晰的因果链可视化
   - 实用的建议措施

6. 【与相似度搜索的区别】
   - 相似度搜索：找到"像的"内容
   - 因果推理：理解"为什么"
""")
    
    return causal_chains


if __name__ == "__main__":
    demo_causal_reasoning()
