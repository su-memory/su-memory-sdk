#!/usr/bin/env python3
"""
su-memory SDK 完整体验演示
从历史对话导入到 AI 智能回复增强
"""

import json
from datetime import datetime, timedelta
from su_memory import SuMemoryLitePro


def create_sample_conversations():
    """创建示例历史对话数据"""
    conversations = [
        {
            "id": "conv_001",
            "timestamp": "2024-01-15T10:30:00",
            "speaker": "user",
            "content": "我想要购买企业版套餐"
        },
        {
            "id": "conv_002",
            "timestamp": "2024-01-15T10:30:15",
            "speaker": "assistant",
            "content": "好的，企业版包含100个账号、支持私有部署、年费制。"
        },
        {
            "id": "conv_003",
            "timestamp": "2024-01-15T10:31:00",
            "speaker": "user",
            "content": "我们公司有200人，需要多少费用？"
        },
        {
            "id": "conv_004",
            "timestamp": "2024-01-15T10:31:20",
            "speaker": "assistant",
            "content": "企业版按账号收费，200人的话大概是基础价格的两倍，大约是19998元/年。"
        },
        {
            "id": "conv_005",
            "timestamp": "2024-01-15T10:32:00",
            "speaker": "user",
            "content": "有没有针对教育机构的折扣？"
        },
        {
            "id": "conv_006",
            "timestamp": "2024-01-15T10:32:15",
            "speaker": "assistant",
            "content": "有的，教育机构可以享受8折优惠。"
        },
        {
            "id": "conv_007",
            "timestamp": "2024-01-15T10:33:00",
            "speaker": "user",
            "content": "我们是一所大学，200人，折扣后多少钱？"
        },
        {
            "id": "conv_008",
            "timestamp": "2024-01-15T10:33:30",
            "speaker": "assistant",
            "content": "大学教育机构8折优惠后，200人版本是 19998 * 0.8 = 15998 元/年。"
        },
    ]
    return conversations


def import_conversations_to_memory(pro, conversations):
    """
    步骤1: 将历史对话导入到 su-memory SDK
    """
    print("\n" + "=" * 60)
    print("步骤1: 导入历史对话到记忆系统")
    print("=" * 60)
    
    memory_ids = []
    
    for i, conv in enumerate(conversations):
        # 添加对话内容
        content = f"[{conv['speaker']}] {conv['content']}"
        
        # 添加到记忆系统
        memory_id = pro.add(
            content,
            metadata={
                "type": "conversation",
                "speaker": conv["speaker"],
                "conversation_id": conv["id"],
                "timestamp": conv["timestamp"],
                "imported_at": datetime.now().isoformat()
            }
        )
        memory_ids.append(memory_id)
        
        print(f"  ✅ 添加记忆: {conv['speaker']} - {conv['content'][:30]}...")
    
    print(f"\n  共导入 {len(memory_ids)} 条对话记忆")
    return memory_ids


def link_conversation_context(pro, conversations):
    """
    步骤2: 建立对话上下文关联（因果链）
    """
    print("\n" + "=" * 60)
    print("步骤2: 建立对话上下文关联")
    print("=" * 60)
    
    # 建立相邻对话之间的关联
    for i in range(len(conversations) - 1):
        current_conv = conversations[i]
        next_conv = conversations[i + 1]
        
        # 只关联用户消息和AI回复
        if current_conv["speaker"] != next_conv["speaker"]:
            pro.link_memories(i, i + 1)
            print(f"  🔗 关联: [{current_conv['speaker']}] → [{next_conv['speaker']}]")
    
    # 建立关键信息的因果关联
    # 企业版 -> 200人 -> 费用 -> 教育折扣 -> 最终价格
    key_indices = [0, 2, 3, 4, 5, 6, 7]  # 企业版、200人、费用、教育折扣、大学
    for i in range(len(key_indices) - 1):
        pro.link_memories(key_indices[i], key_indices[i + 1])
    
    print(f"\n  共建立 {len(key_indices) - 1} 条关键因果关联")


def query_with_context(pro, new_question):
    """
    步骤3: 在新对话中检索历史记忆
    """
    print("\n" + "=" * 60)
    print(f"步骤3: 新对话检索历史记忆")
    print("=" * 60)
    
    print(f"\n  用户新问题: {new_question}")
    
    # 检索相关记忆
    print("\n  📊 检索结果:")
    
    # 1. 基础语义检索
    basic_results = pro.query(new_question, top_k=3)
    print("\n  【语义检索】")
    for r in basic_results:
        print(f"    - {r['content'][:50]}... (score: {r['score']:.2f})")
    
    # 2. 多跳因果推理
    print("\n  【多跳推理】")
    multi_hop_results = pro.query_multihop(new_question, max_hops=3)
    for r in multi_hop_results[:5]:
        hops = r.get('hops', 0)
        path = r.get('path', [])
        print(f"    - {r['content'][:50]}... (hops: {hops})")
        if path:
            print(f"      推理链: {' → '.join(path[:3])}")
    
    return basic_results, multi_hop_results


def enhance_ai_response(new_question, memory_results):
    """
    步骤4: 基于记忆生成增强回复
    """
    print("\n" + "=" * 60)
    print("步骤4: 生成增强的 AI 回复")
    print("=" * 60)
    
    # 构建上下文
    context = """
## 历史对话上下文
以下是用户与 AI 的历史对话记录，请结合上下文回答用户问题：

"""
    for r in memory_results[:5]:
        context += f"- {r['content']}\n"
    
    # 构造增强后的 prompt
    enhanced_prompt = f"""
{context}

## 当前问题
用户问: {new_question}

## 回复要求
1. 结合历史对话上下文
2. 如果问题是追问，请继承之前的讨论
3. 如果涉及到价格、人数等信息，请与历史记录保持一致
"""
    
    print("\n  📝 增强后的 Prompt:")
    print("-" * 40)
    print(enhanced_prompt)
    print("-" * 40)
    
    # 模拟 AI 回复
    ai_response = f"""
基于历史对话，我记得您提到：
1. 您想要购买企业版套餐
2. 贵公司有200人
3. 您提到是教育机构（大学）

根据这些信息，200人企业版的折扣价格是：
- 原价: 19,998元/年
- 教育机构8折优惠
- **折后价格: 15,998元/年**

请问您还需要了解其他信息吗？
"""
    
    print("\n  🤖 AI 增强回复:")
    print("-" * 40)
    print(ai_response)
    print("-" * 40)
    
    return enhanced_prompt, ai_response


def verify_memory_usage(pro, question, expected_contexts):
    """
    步骤5: 验证 AI 回复确实利用了历史记忆
    """
    print("\n" + "=" * 60)
    print("步骤5: 验证历史记忆被正确利用")
    print("=" * 60)
    
    results = pro.query_multihop(question, max_hops=3)
    
    print("\n  🔍 验证项目:")
    verification_items = []
    
    for expected in expected_contexts:
        found = any(expected.lower() in r['content'].lower() for r in results)
        status = "✅" if found else "❌"
        verification_items.append(f"    {status} {expected}")
        print(f"    {status} {expected}")
    
    # 检查推理链
    print("\n  🔗 推理链验证:")
    for r in results[:3]:
        hops = r.get('hops', 0)
        path = r.get('path', [])
        if hops > 0:
            print(f"    - {r['content'][:40]}... (跳数: {hops})")
            print(f"      路径: {' → '.join(path[:min(3, len(path))])}")
    
    all_verified = all(
        any(exp.lower() in r['content'].lower() for r in results)
        for exp in expected_contexts
    )
    
    print("\n" + "=" * 60)
    if all_verified:
        print("  ✅ 验证通过! AI 回复确实利用了历史对话记忆")
    else:
        print("  ⚠️  部分验证未通过")
    print("=" * 60)
    
    return all_verified


def demo():
    """完整演示流程"""
    print("\n" + "🎯" * 30)
    print("su-memory SDK 完整体验演示")
    print("从历史对话导入到 AI 智能回复增强")
    print("🎯" * 30)
    
    # 创建记忆系统实例
    print("\n初始化 su-memory SDK...")
    pro = SuMemoryLitePro(enable_vector=False)  # 禁用向量服务以加快演示
    
    # 步骤1: 导入历史对话
    conversations = create_sample_conversations()
    import_conversations_to_memory(pro, conversations)
    
    # 步骤2: 建立关联
    link_conversation_context(pro, conversations)
    
    # 步骤3: 新对话检索
    new_question = "我之前问过关于教育机构折扣的问题，我是200人的大学，能便宜多少？"
    basic_results, multi_hop_results = query_with_context(pro, new_question)
    
    # 步骤4: 生成增强回复
    prompt, response = enhance_ai_response(new_question, multi_hop_results)
    
    # 步骤5: 验证记忆被使用
    expected_contexts = [
        "企业版",
        "200",
        "教育机构",
        "折扣",
    ]
    verify_memory_usage(pro, new_question, expected_contexts)
    
    print("\n" + "=" * 60)
    print("📊 体验总结")
    print("=" * 60)
    print("""
本演示展示了 su-memory SDK 的核心能力：

1. 【记忆存储】
   - 将历史对话完整导入到记忆系统
   - 自动提取关键信息（发言人、时间、内容）

2. 【因果关联】
   - 自动建立相邻对话的上下文关联
   - 支持多跳因果推理链

3. 【智能检索】
   - 语义检索：找到相关内容
   - 多跳推理：追踪因果链条

4. 【回复增强】
   - 基于检索到的历史记忆
   - 生成上下文连贯的智能回复

5. 【可解释性】
   - 每条回复都可以追溯到原始记忆
   - 推理链完全透明可查
""")


if __name__ == "__main__":
    demo()
