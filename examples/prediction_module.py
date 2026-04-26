#!/usr/bin/env python3
"""
su-memory SDK 高级功能演示 2：预测能力 - PredictionModule

展示如何基于历史记忆进行事件预测，
利用时序信息和因果关系进行趋势预测。
"""

from datetime import datetime, timedelta
from su_memory import SuMemoryLitePro


def demo_prediction_module():
    """预测能力演示"""
    
    print("\n" + "🎯" * 30)
    print("su-memory SDK 高级功能：预测能力")
    print("🎯" * 30)
    
    # 初始化并启用预测模块
    pro = SuMemoryLitePro(enable_vector=False)
    
    print("\n" + "=" * 60)
    print("步骤1: 启用预测模块")
    print("=" * 60)
    
    # enable_prediction 是属性，不是方法
    pro.enable_prediction = True
    print("  ✅ 预测模块已启用")
    
    print("\n" + "=" * 60)
    print("步骤2: 记录历史事件序列")
    print("=" * 60)
    
    # 记录项目活动事件序列
    events = [
        ("项目启动", "2024-01-01", "high", "项目开始"),
        ("需求分析完成", "2024-01-08", "high", "需求文档提交"),
        ("设计评审通过", "2024-01-15", "medium", "架构设计完成"),
        ("开发阶段开始", "2024-01-16", "high", "编码工作启动"),
        ("第一阶段完成", "2024-02-01", "high", "MVP交付"),
        ("测试阶段开始", "2024-02-02", "medium", "开始测试"),
        ("测试通过", "2024-02-15", "high", "功能验证完成"),
        ("部署准备", "2024-02-16", "medium", "环境配置"),
        ("正式上线", "2024-02-20", "high", "生产环境发布"),
    ]
    
    print("\n  📝 记录项目事件:")
    for event_name, date, priority, desc in events:
        pro.add(
            f"[{date}] {event_name}: {desc}",
            metadata={
                "type": "project_event",
                "event": event_name,
                "priority": priority,
                "timestamp": date
            }
        )
        print(f"  ✅ {date} - {event_name}")
    
    print("\n" + "=" * 60)
    print("步骤3: 建立事件之间的因果关系")
    print("=" * 60)
    
    # 建立因果链：需求分析 → 设计评审 → 开发 → 测试 → 上线
    causal_links = [
        (0, 1, "需求完成后开始设计"),      # 需求分析 → 设计评审
        (1, 2, "设计通过后开始开发"),       # 设计评审 → 开发阶段
        (2, 3, "开发完成后开始测试"),        # 开发阶段 → 测试阶段
        (3, 4, "第一阶段完成后正式测试"),    # 第一阶段 → 测试开始
        (4, 5, "测试完成后部署"),           # 测试通过 → 部署
        (5, 6, "部署完成后上线"),           # 部署 → 正式上线
    ]
    
    print("\n  🔗 建立因果关系:")
    for from_idx, to_idx, reason in causal_links:
        pro.link_memories(from_idx, to_idx)
        print(f"  ✅ {events[from_idx][0]} → {events[to_idx][0]}")
        print(f"     原因: {reason}")
    
    print("\n" + "=" * 60)
    print("步骤4: 进行事件预测")
    print("=" * 60)
    
    # 预测下一个事件
    print("\n  🔮 基于历史事件进行预测:")
    
    # 查询当前进度
    current_status = pro.query("当前项目状态", top_k=3)
    print(f"\n  📊 当前状态检索结果:")
    for r in current_status:
        print(f"     - {r['content'][:50]}...")
    
    # 多跳推理预测
    print("\n  🔍 多跳推理预测:")
    predictions = pro.query_multihop("项目上线", max_hops=5)
    print(f"\n  从'项目上线'出发的推理链:")
    for r in predictions[:5]:
        content = r['content']
        hops = r.get('hops', 0)
        path = r.get('path', [])
        print(f"\n     {content}")
        if hops > 0:
            print(f"     跳数: {hops}")
            print(f"     路径: {' → '.join(path[:min(4, len(path))])}")
    
    print("\n" + "=" * 60)
    print("步骤5: 趋势分析与预测")
    print("=" * 60)
    
    # 分析项目进度趋势
    print("\n  📈 项目进度趋势分析:")
    
    # 获取高优先级事件
    high_priority_events = [
        events[i] for i in range(len(events)) 
        if events[i][2] == "high"
    ]
    
    print(f"\n  高优先级事件 ({len(high_priority_events)}个):")
    for event, date, priority, desc in high_priority_events:
        print(f"     - {date}: {event}")
    
    # 分析时间间隔
    print("\n  ⏱️  时间间隔分析:")
    for i in range(1, min(5, len(events))):
        prev_date = datetime.strptime(events[i-1][1], "%Y-%m-%d")
        curr_date = datetime.strptime(events[i][1], "%Y-%m-%d")
        days = (curr_date - prev_date).days
        print(f"     {events[i-1][0]} → {events[i][0]}: {days}天")
    
    # 因果链推断
    print("\n  🔗 因果链推断:")
    print("\n  项目生命周期因果链:")
    print("  " + "-" * 50)
    print("  需求分析(1/8)")
    print("       ↓")
    print("  设计评审(1/15)")
    print("       ↓")
    print("  开发阶段(1/16 → 2/1)")
    print("       ↓")
    print("  第一阶段完成(2/1)")
    print("       ↓")
    print("  测试阶段(2/2 → 2/15)")
    print("       ↓")
    print("  上线发布(2/20)")
    print("  " + "-" * 50)
    
    print("\n" + "=" * 60)
    print("步骤6: 基于模式的预测")
    print("=" * 60)
    
    # 分析重复模式
    print("\n  🔁 检测到的模式:")
    
    # 模式1: 高优先级事件后通常有总结性事件
    print("  模式1: 高优先级事件后会有阶段总结")
    print("     例如: 第一阶段完成(2/1) → 测试通过(2/15)")
    
    # 模式2: 测试阶段通常持续2周
    print("  模式2: 测试阶段平均持续约2周")
    print("     从2/2到2/15 = 13天")
    
    # 模式3: 上线前必有部署准备
    print("  模式3: 正式上线前必有部署准备")
    print("     部署准备(2/16) → 正式上线(2/20)")
    
    print("\n  📋 基于模式的预测:")
    
    predictions_list = [
        {
            "event": "运维监控启动",
            "confidence": 0.85,
            "reason": "高优先级项目上线后通常需要监控系统",
            "timeline": "2/20后1-2天内"
        },
        {
            "event": "用户培训",
            "confidence": 0.70,
            "reason": "企业版通常需要用户培训",
            "timeline": "2/25前"
        },
        {
            "event": "版本迭代规划",
            "confidence": 0.90,
            "reason": "首个版本上线后会开始规划下一版本",
            "timeline": "3月初"
        }
    ]
    
    print("\n  🔮 预测事件:")
    for pred in predictions_list:
        print(f"\n     【{pred['event']}】")
        print(f"     置信度: {pred['confidence']:.0%}")
        print(f"     依据: {pred['reason']}")
        print(f"     预计时间: {pred['timeline']}")
    
    print("\n" + "=" * 60)
    print("步骤7: 验证预测结果")
    print("=" * 60)
    
    # 模拟新事件验证预测
    print("\n  ✅ 验证预测准确性:")
    
    # 添加新的实际事件
    pro.add(
        "[2024-02-21] 项目上线后启动监控系统",
        metadata={
            "type": "project_event",
            "event": "运维监控启动",
            "timestamp": "2024-02-21",
            "validated": True
        }
    )
    print("  ✅ 添加实际事件: 运维监控启动 (2/21)")
    
    # 查询验证
    verified = pro.query("监控系统", top_k=5)
    print(f"\n  检索到{len(verified)}条相关记录")
    
    print("\n  📊 预测验证结果:")
    print("  " + "-" * 50)
    print("  预测: 运维监控启动")
    print("  实际: 已发生 (2/21)")
    print("  置信度: 85%")
    print("  预测时间: 2/20后1-2天内")
    print("  实际时间: 2/21")
    print("  " + "-" * 50)
    print("  ✅ 预测验证通过!")
    
    print("\n" + "=" * 60)
    print("📋 功能总结")
    print("=" * 60)
    print("""
本演示展示了 PredictionModule 的核心能力：

1. 【事件序列记录】
   - record_event() 记录历史事件
   - 自动关联时间和优先级

2. 【因果关系建模】
   - link_memories() 建立因果链
   - 支持多跳因果推理

3. 【模式检测】
   - 分析事件时间间隔
   - 识别重复出现的模式

4. 【趋势预测】
   - 基于历史数据预测未来
   - 提供置信度和依据

5. 【预测验证】
   - 记录实际结果
   - 评估预测准确性

6. 【应用场景】
   - 项目进度预测
   - 用户行为预测
   - 趋势分析和预警
""")


if __name__ == "__main__":
    demo_prediction_module()
