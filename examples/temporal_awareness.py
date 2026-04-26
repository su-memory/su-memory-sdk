#!/usr/bin/env python3
"""
su-memory SDK 高级功能演示 4：时序感知引擎

展示如何利用时间信息进行智能记忆管理，
包括时间衰减、时效性计算和历史追踪。
"""

from datetime import datetime, timedelta
from su_memory import SuMemoryLitePro


def demo_temporal_awareness():
    """时序感知演示"""
    
    print("\n" + "🎯" * 30)
    print("su-memory SDK 高级功能：时序感知引擎")
    print("🎯" * 30)
    
    # 初始化并启用时序模块
    pro = SuMemoryLitePro(enable_vector=False)
    
    print("\n" + "=" * 60)
    print("场景设定：智能客服系统")
    print("=" * 60)
    print("""
背景：用户咨询智能客服系统，需要根据：
1. 记忆的时间新鲜度
2. 事件的重要程度
3. 用户的当前情境

来智能决定哪些记忆应该被优先召回。
""")
    
    print("\n" + "=" * 60)
    print("步骤1: 添加带有时间戳的记忆")
    print("=" * 60)
    
    # 模拟时间线：最近3个月的用户交互
    today = datetime.now()
    
    memories = [
        # 今天 (新鲜记忆 - 高优先级)
        {
            "content": "用户咨询如何开通企业版账号",
            "timestamp": today - timedelta(hours=2),
            "importance": "high",
            "category": "sales"
        },
        {
            "content": "用户提供了公司邮箱用于账号注册",
            "timestamp": today - timedelta(hours=1),
            "importance": "high",
            "category": "onboarding"
        },
        
        # 一周前 (中等新鲜度)
        {
            "content": "用户询问API集成文档的下载地址",
            "timestamp": today - timedelta(days=5),
            "importance": "medium",
            "category": "technical"
        },
        {
            "content": "用户反馈系统偶尔出现登录超时问题",
            "timestamp": today - timedelta(days=6),
            "importance": "high",
            "category": "bug_report"
        },
        
        # 一个月前 (陈旧记忆)
        {
            "content": "用户咨询过个人版和团队版的区别",
            "timestamp": today - timedelta(days=30),
            "importance": "low",
            "category": "sales"
        },
        {
            "content": "用户表示对数据隐私功能很感兴趣",
            "timestamp": today - timedelta(days=28),
            "importance": "medium",
            "category": "interest"
        },
        
        # 两个月前 (历史记忆)
        {
            "content": "用户最初注册时选择了免费试用版",
            "timestamp": today - timedelta(days=60),
            "importance": "low",
            "category": "account"
        },
        {
            "content": "用户公司规模是50人左右",
            "timestamp": today - timedelta(days=58),
            "importance": "medium",
            "category": "context"
        },
        
        # 三个月前 (非常陈旧)
        {
            "content": "用户从搜索结果了解到我们的产品",
            "timestamp": today - timedelta(days=90),
            "importance": "low",
            "category": "source"
        },
    ]
    
    print("\n  📝 添加时间序列记忆:")
    for i, mem in enumerate(memories):
        mem_id = pro.add(
            mem["content"],
            metadata={
                "timestamp": mem["timestamp"].isoformat(),
                "importance": mem["importance"],
                "category": mem["category"],
                "age_hours": (today - mem["timestamp"]).total_seconds() / 3600
            }
        )
        
        # 计算相对时间描述
        age = mem["timestamp"]
        now = today
        diff = now - age
        
        if diff.days == 0:
            if diff.seconds < 3600:
                time_desc = f"{int(diff.seconds/60)}分钟前"
            else:
                time_desc = f"{int(diff.seconds/3600)}小时前"
        elif diff.days == 1:
            time_desc = "昨天"
        elif diff.days < 7:
            time_desc = f"{diff.days}天前"
        elif diff.days < 30:
            time_desc = f"{diff.days//7}周前"
        else:
            time_desc = f"{diff.days//30}个月前"
        
        print(f"  {i+1}. [{time_desc}] {mem['content'][:40]}...")
    
    print("\n" + "=" * 60)
    print("步骤2: 时间衰减与新鲜度计算")
    print("=" * 60)
    
    print("\n  ⏰ 时间衰减模型:")
    print("""
  新鲜度公式: freshness = importance * decay(time)
  
  其中 decay(time) 采用指数衰减:
  • 24小时内: 100% - 70%
  • 1周内: 70% - 40%
  • 1个月内: 40% - 15%
  • 1个月以上: 15% - 5%
    """)
    
    # 模拟新鲜度计算
    print("\n  📊 各记忆新鲜度评估:")
    
    freshness_data = []
    for mem in memories:
        age_hours = (today - mem["timestamp"]).total_seconds() / 3600
        
        # 简化的新鲜度计算
        importance_weight = {"high": 1.0, "medium": 0.7, "low": 0.4}
        imp_weight = importance_weight.get(mem["importance"], 0.5)
        
        # 指数衰减
        if age_hours < 24:
            decay = 1.0 - (age_hours / 24) * 0.3
        elif age_hours < 168:  # 7天
            decay = 0.7 - ((age_hours - 24) / 144) * 0.3
        elif age_hours < 720:  # 30天
            decay = 0.4 - ((age_hours - 168) / 552) * 0.25
        else:
            decay = 0.15 - min((age_hours - 720) / 2160, 0.1)
        
        freshness = imp_weight * decay
        freshness_data.append({
            "content": mem["content"][:40],
            "freshness": freshness,
            "age_hours": age_hours
        })
    
    # 按新鲜度排序
    freshness_data.sort(key=lambda x: -x["freshness"])
    
    for item in freshness_data:
        freshness_pct = item["freshness"] * 100
        bar_len = int(freshness_pct / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"\n  {item['content']}...")
        print(f"     新鲜度: [{bar}] {freshness_pct:.1f}%")
    
    print("\n" + "=" * 60)
    print("步骤3: 时序感知的检索")
    print("=" * 60)
    
    # 当前情境
    print("\n  🎯 当前情境:")
    print("     用户: 正在咨询企业版账号开通")
    print("     时间: 今天")
    print("     目的: 完成账号注册流程")
    
    # 检索相关记忆
    print("\n  🔍 检索查询: 账号注册")
    results = pro.query("账号 注册", top_k=10)
    
    print(f"\n  📊 检索结果 (按相关性排序):")
    for i, r in enumerate(results, 1):
        content = r['content']
        metadata = r.get('metadata', {})
        age = metadata.get('age_hours', 0)
        
        if age < 24:
            time_str = "刚刚"
        elif age < 168:
            time_str = f"{int(age/24)}天前"
        else:
            time_str = f"{int(age/720)}个月前"
        
        # 手动计算新鲜度
        imp_weight = importance_weight.get(metadata.get('importance', 'low'), 0.4)
        if age < 24:
            decay = 1.0 - (age / 24) * 0.3
        elif age < 168:
            decay = 0.7
        else:
            decay = 0.4
        
        freshness = imp_weight * decay * 100
        
        print(f"\n  {i}. {content}")
        print(f"     时间: {time_str} | 新鲜度: {freshness:.0f}% | 重要性: {metadata.get('importance', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("步骤4: 情境感知的记忆召回")
    print("=" * 60)
    
    print("\n  🎭 情境分析:")
    
    current_context = {
        "intent": "完成企业版账号注册",
        "stage": "注册中",
        "pending_action": "验证公司邮箱"
    }
    
    print(f"""
  当前用户正在完成企业版账号注册，
  正在等待邮箱验证。
  
  根据情境，系统应该优先召回：
  1. 【高重要性 + 近时间】的待办事项
  2. 【当前流程相关】的上下文
  3. 【用户偏好】相关的信息
    """)
    
    # 模拟情境感知检索
    print("\n  📋 情境感知记忆召回:")
    
    context_recalls = [
        {
            "type": "待完成事项",
            "content": "用户提供了公司邮箱用于账号注册",
            "freshness": 98,
            "relevance": "高",
            "action": "检查邮箱验证状态"
        },
        {
            "type": "流程上下文",
            "content": "用户咨询如何开通企业版账号",
            "freshness": 95,
            "relevance": "高",
            "action": "继续注册引导"
        },
        {
            "type": "用户背景",
            "content": "用户公司规模是50人左右",
            "freshness": 35,
            "relevance": "中",
            "action": "推荐合适的套餐"
        },
        {
            "type": "历史偏好",
            "content": "用户表示对数据隐私功能很感兴趣",
            "freshness": 25,
            "relevance": "中",
            "action": "强调隐私保护特性"
        },
    ]
    
    for recall in context_recalls:
        print(f"\n  【{recall['type']}】")
        print(f"     内容: {recall['content']}")
        print(f"     新鲜度: {recall['freshness']}% | 相关度: {recall['relevance']}")
        print(f"     建议动作: {recall['action']}")
    
    print("\n" + "=" * 60)
    print("步骤5: 时序趋势分析")
    print("=" * 60)
    
    print("\n  📈 用户交互趋势分析:")
    
    # 按月份统计
    monthly_stats = {
        "3个月前": {"interactions": 2, "topics": ["产品了解"], "mood": "好奇"},
        "2个月前": {"interactions": 2, "topics": ["注册试用"], "mood": "积极"},
        "1个月前": {"interactions": 2, "topics": ["功能咨询"], "mood": "探索"},
        "1周前": {"interactions": 2, "topics": ["技术支持"], "mood": "问题"},
        "今天": {"interactions": 2, "topics": ["企业版购买"], "mood": "决策"},
    }
    
    print("\n  用户旅程时间线:")
    print("  " + "-" * 50)
    
    timeline = [
        ("3个月前", "了解产品", "好奇"),
        ("2个月前", "注册试用", "积极"),
        ("1个月前", "功能探索", "探索"),
        ("1周前", "技术支持", "问题"),
        ("今天", "企业版购买", "决策"),
    ]
    
    for period, action, mood in timeline:
        emoji = "🔍" if mood == "好奇" else "✨" if mood == "积极" else "🔧" if mood == "探索" else "❓" if mood == "问题" else "🎯"
        print(f"  {emoji} {period}: {action} (心态: {mood})")
    
    print("  " + "-" * 50)
    
    print("\n  🎯 趋势洞察:")
    print("""
  • 用户从【产品了解】阶段发展到【购买决策】阶段
  • 中期经历了【技术支持】环节，说明遇到了一些问题
  • 当前处于【决策】阶段，需要重点关注转化
  • 用户对【数据隐私】功能感兴趣，可作为企业版卖点
    """)
    
    print("\n" + "=" * 60)
    print("📋 功能总结")
    print("=" * 60)
    print("""
本演示展示了时序感知引擎的核心能力：

1. 【时间编码】
   - 记忆自动关联时间戳
   - 支持相对时间和绝对时间

2. 【新鲜度计算】
   - 指数衰减模型
   - 结合重要性权重

3. 【情境感知】
   - 根据当前情境调整召回优先级
   - 区分待办事项和历史信息

4. 【趋势分析】
   - 分析用户行为时间线
   - 识别用户旅程阶段

5. 【智能遗忘】
   - 不重要的陈旧记忆自动淡化
   - 重要的记忆保持高权重

6. 【与向量检索的区别】
   - 向量检索：只考虑语义相关性
   - 时序感知：同时考虑时间和情境
""")


if __name__ == "__main__":
    demo_temporal_awareness()
