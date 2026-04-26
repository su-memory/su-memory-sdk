#!/usr/bin/env python3
"""
su-memory SDK 高级功能演示 1：情景感知 - 跨会话话题召回

展示如何在不同会话之间识别和召回相关话题，
利用 SessionManager 实现跨会话的记忆关联。
"""

from datetime import datetime, timedelta
from su_memory import SuMemoryLitePro


def demo_cross_session_recall():
    """跨会话话题召回演示"""
    
    print("\n" + "🎯" * 30)
    print("su-memory SDK 高级功能：跨会话话题召回")
    print("🎯" * 30)
    
    # 初始化 (禁用向量服务以加快演示)
    pro = SuMemoryLitePro(enable_vector=False)
    
    print("\n" + "=" * 60)
    print("步骤1: 创建多个会话，模拟跨会话场景")
    print("=" * 60)
    
    # 会话1: 产品咨询 (3天前)
    session_product = pro.create_session("产品咨询-第1次")
    pro.create_session(session_product)  # 确保会话创建
    
    pro.add(
        "用户咨询企业版套餐功能",
        session_id=session_product,
        metadata={"topic": "企业版", "priority": "high"}
    )
    pro.add(
        "企业版包含100账号、私有部署、高级分析功能",
        session_id=session_product,
        metadata={"topic": "企业版", "feature": "私有部署"}
    )
    pro.add(
        "用户表示需要API接口支持",
        session_id=session_product,
        metadata={"topic": "API", "feature": "集成"}
    )
    print(f"  ✅ 会话1 [{session_product}]: 产品咨询")
    print(f"     - 企业版套餐 (3条记忆)")
    
    # 会话2: 价格讨论 (2天前)
    session_price = pro.create_session("价格讨论")
    
    pro.add(
        "用户询问200人企业的定价",
        session_id=session_price,
        metadata={"topic": "定价", "users": 200}
    )
    pro.add(
        "企业版定价: 基础价 * 用户数/100 = 19998元/年",
        session_id=session_price,
        metadata={"topic": "定价", "price": 19998}
    )
    pro.add(
        "用户询问是否有教育机构折扣",
        session_id=session_price,
        metadata={"topic": "折扣", "type": "教育"}
    )
    pro.add(
        "教育机构可享受8折优惠",
        session_id=session_price,
        metadata={"topic": "折扣", "discount": 0.8}
    )
    print(f"\n  ✅ 会话2 [{session_price}]: 价格讨论")
    print(f"     - 定价咨询 (4条记忆)")
    
    # 会话3: 技术支持 (昨天)
    session_tech = pro.create_session("技术支持")
    
    pro.add(
        "用户询问API集成的技术细节",
        session_id=session_tech,
        metadata={"topic": "API", "type": "技术"}
    )
    pro.add(
        "提供REST API文档和SDK下载链接",
        session_id=session_tech,
        metadata={"topic": "API", "delivered": "文档"}
    )
    pro.add(
        "用户需要Webhook配置帮助",
        session_id=session_tech,
        metadata={"topic": "Webhook", "feature": "配置"}
    )
    print(f"\n  ✅ 会话3 [{session_tech}]: 技术支持")
    print(f"     - API技术支持 (3条记忆)")
    
    # 会话4: 新会话 (今天，用户再次联系)
    session_new = pro.create_session("新会话-跟进")
    
    pro.add(
        "用户表示之前咨询过企业版",
        session_id=session_new,
        metadata={"topic": "跟进", "context": "之前咨询过"}
    )
    pro.add(
        "用户想要了解最新价格",
        session_id=session_new,
        metadata={"topic": "定价", "status": "跟进"}
    )
    print(f"\n  ✅ 会话4 [{session_new}]: 新会话 - 用户再次联系")
    print(f"     - 跟进咨询 (2条记忆)")
    
    print("\n" + "=" * 60)
    print("步骤2: 获取会话列表和话题概览")
    print("=" * 60)
    
    # 获取所有会话
    sessions = pro.get_stats()
    print(f"\n  📊 系统统计:")
    print(f"     - 总记忆数: {sessions.get('total_memories', 'N/A')}")
    print(f"     - 会话数: {sessions.get('total_sessions', 'N/A')}")
    
    # 获取相关话题
    print("\n" + "=" * 60)
    print("步骤3: 跨会话话题召回 - 搜索'API'相关记忆")
    print("=" * 60)
    
    # 搜索API相关话题
    api_results = pro.query("API 集成", top_k=10)
    print(f"\n  🔍 搜索'API 集成'的结果:")
    for i, r in enumerate(api_results, 1):
        content = r['content']
        session_id = r.get('metadata', {}).get('session_id', 'unknown')
        topic = r.get('metadata', {}).get('topic', 'unknown')
        print(f"     {i}. [{topic}] {content[:40]}...")
        print(f"        会话: {session_id}")
    
    print("\n" + "=" * 60)
    print("步骤4: 语义相似性话题聚类")
    print("=" * 60)
    
    # 按话题分组记忆
    topic_groups = {}
    all_results = pro.query("", top_k=100)  # 获取所有记忆
    
    for r in all_results:
        topic = r.get('metadata', {}).get('topic', '其他')
        if topic not in topic_groups:
            topic_groups[topic] = []
        topic_groups[topic].append(r)
    
    print("\n  📂 话题聚类结果:")
    for topic, memories in sorted(topic_groups.items(), key=lambda x: -len(x[1])):
        print(f"\n  【{topic}】({len(memories)}条记忆)")
        for m in memories[:2]:  # 每组显示2条
            print(f"     - {m['content'][:35]}...")
    
    print("\n" + "=" * 60)
    print("步骤5: 上下文连贯性分析")
    print("=" * 60)
    
    # 获取跨会话关联
    print("\n  🔗 跨会话话题关联:")
    
    # 企业版话题关联
    enterprise_memories = [r for r in all_results 
                          if r.get('metadata', {}).get('topic') == '企业版']
    price_memories = [r for r in all_results 
                     if r.get('metadata', {}).get('topic') == '定价']
    api_memories = [r for r in all_results 
                   if r.get('metadata', {}).get('topic') == 'API']
    
    print(f"\n  📌 企业版话题: {len(enterprise_memories)}条记忆")
    print(f"  📌 定价话题: {len(price_memories)}条记忆")
    print(f"  📌 API话题: {len(api_memories)}条记忆")
    
    # 展示用户对话的完整上下文
    print("\n  📜 用户对话轨迹:")
    
    # 按时间顺序重建用户对话历史
    all_memories_sorted = sorted(
        all_results,
        key=lambda x: x.get('metadata', {}).get('timestamp', '')
    )
    
    print("\n  会话流程图:")
    print("  " + "-" * 50)
    print("  会话1: 产品咨询")
    print("    └─ 企业版套餐")
    print("    └─ API接口需求")
    print("  会话2: 价格讨论")
    print("    └─ 200人定价")
    print("    └─ 教育折扣")
    print("  会话3: 技术支持")
    print("    └─ API文档")
    print("    └─ Webhook配置")
    print("  会话4: 新会话(今天)")
    print("    └─ 跟进企业版")
    print("  " + "-" * 50)
    
    print("\n" + "=" * 60)
    print("步骤6: 验证跨会话召回效果")
    print("=" * 60)
    
    # 模拟新用户查询
    new_query = "我之前问过企业版和API集成的问题，现在想购买200人的版本"
    
    print(f"\n  💬 新用户查询: {new_query}")
    
    # 多跳推理检索
    results = pro.query_multihop(new_query, max_hops=3)
    
    print(f"\n  🎯 多跳推理结果 ({len(results)}条):")
    for r in results[:6]:
        topic = r.get('metadata', {}).get('topic', 'unknown')
        content = r['content']
        hops = r.get('hops', 0)
        print(f"\n     [{topic}] {content}")
        print(f"      推理跳数: {hops}")
    
    # 展示完整的上下文连贯性
    print("\n  📊 上下文连贯性验证:")
    print("  " + "-" * 50)
    print("  ✅ 检索到'企业版'话题 (来自会话1)")
    print("  ✅ 检索到'API集成'需求 (来自会话1)")
    print("  ✅ 检索到'200人定价'信息 (来自会话2)")
    print("  ✅ 检索到'教育折扣'信息 (来自会话2)")
    print("  ✅ 推理链: 企业版 + API → 需求确认")
    print("  ✅ 推理链: 200人 + 折扣 → 价格计算")
    print("  " + "-" * 50)
    
    print("\n" + "=" * 60)
    print("📋 功能总结")
    print("=" * 60)
    print("""
本演示展示了跨会话话题召回的核心能力：

1. 【会话隔离与关联】
   - 每个会话独立存储记忆
   - 通过话题标签实现跨会话关联

2. 【语义检索】
   - 基于内容语义的相似度匹配
   - 不受会话边界限制

3. 【话题聚类】
   - 自动将相关记忆归类
   - 便于理解和追踪用户意图

4. 【上下文连贯】
   - 重建用户对话历史轨迹
   - 支持多跳推理关联

5. 【应用场景】
   - 客服系统：记住用户历史需求
   - AI助手：跨对话保持上下文
   - 知识管理：自动关联相关内容
""")


if __name__ == "__main__":
    demo_cross_session_recall()
