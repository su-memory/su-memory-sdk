"""
su-memory SDK 快速开始示例

这个脚本演示了如何使用su-memory SDK的核心功能。
"""

# ==================== 1. 基本使用 ====================

def basic_usage():
    """基本使用示例"""
    from su_memory.sdk import SuMemoryClient

    # 创建客户端
    client = SuMemoryClient(mode="local")

    # 添加记忆
    mid1 = client.add("今天学习了Python编程")
    mid2 = client.add("项目ROI增长了25%")
    mid3 = client.add("团队会议上讨论了Q3计划")

    print(f"添加了3条记忆: {mid1}, {mid2}, {mid3}")

    # 查询记忆
    results = client.query("学习内容", top_k=2)
    print(f"查询结果: {results}")

    # 获取统计
    stats = client.get_stats()
    print(f"记忆统计: {stats}")


# ==================== 2. 轻量级版本 ====================

def lite_usage():
    """轻量级版本示例"""
    from su_memory.sdk import SuMemoryLite

    # 创建轻量级客户端
    client = SuMemoryLite(max_memories=1000)

    # 添加记忆
    for i in range(10):
        client.add(f"这是第{i}条记忆内容")

    # 查询
    results = client.query("第5条", top_k=3)
    print(f"查询结果: {results}")

    # 预测
    prediction = client.predict("当前情况", "采取行动")
    print(f"预测结果: {prediction}")


# ==================== 3. 带元数据 ====================

def with_metadata():
    """带元数据的示例"""
    from su_memory.sdk import SuMemoryClient

    client = SuMemoryClient(mode="local")

    # 添加带元数据的记忆
    client.add(
        "完成了用户管理模块的开发",
        metadata={
            "type": "task",
            "priority": "high",
            "tags": ["backend", "user"]
        }
    )

    client.add(
        "修复了登录页面的bug",
        metadata={
            "type": "bugfix",
            "priority": "urgent",
            "tags": ["frontend", "auth"]
        }
    )

    # 查询
    results = client.query("用户相关")
    print(f"用户相关内容: {results}")


# ==================== 4. 因果链接 ====================

def causal_link():
    """因果链接示例"""
    from su_memory.sdk import SuMemoryClient

    client = SuMemoryClient(mode="local")

    # 添加记忆
    mid1 = client.add("用户点击了注册按钮")
    mid2 = client.add("系统显示了注册表单")
    mid3 = client.add("用户提交了注册信息")
    mid4 = client.add("系统创建了用户账户")

    # 建立因果链接
    client.link(mid1, mid2)  # 点击 -> 显示表单
    client.link(mid2, mid3)  # 显示表单 -> 提交信息
    client.link(mid3, mid4)  # 提交信息 -> 创建账户

    print("因果链接已建立")


# ==================== 5. LangChain集成 ====================

def langchain_integration():
    """LangChain集成示例"""
    try:
        from langchain.agents import Agent
        from su_memory.adapters import SuMemoryMemory
        from su_memory.sdk import SuMemoryClient

        # 创建记忆客户端
        memory_client = SuMemoryClient(mode="local")

        # 创建LangChain记忆适配器
        memory = SuMemoryMemory(client=memory_client)

        print("LangChain集成准备完成")
        print("可以在LangChain Agent中使用此memory对象")

    except ImportError:
        print("LangChain未安装，跳过此示例")


# ==================== 6. 云端模式 ====================

def cloud_usage():
    """云端模式示例"""
    from su_memory.sdk import SuMemoryClient, SDKConfig

    # 创建云端配置
    config = SDKConfig(
        mode="cloud",
        api_url="https://api.sumemory.io",
        api_key="your-api-key"
    )

    # 创建云端客户端
    client = SuMemoryClient(config=config)

    # 添加记忆（发送到云端）
    mid = client.add("云端记忆示例")

    # 查询
    results = client.query("云端")

    print(f"云端模式结果: {results}")


# ==================== 运行所有示例 ====================

if __name__ == "__main__":
    print("=" * 50)
    print("su-memory SDK 快速开始示例")
    print("=" * 50)

    print("\n1. 基本使用:")
    basic_usage()

    print("\n2. 轻量级版本:")
    lite_usage()

    print("\n3. 带元数据:")
    with_metadata()

    print("\n4. 因果链接:")
    causal_link()

    print("\n5. LangChain集成:")
    langchain_integration()

    print("\n6. 云端模式:")
    print("(需要有效的API密钥)")

    print("\n" + "=" * 50)
    print("示例完成!")
    print("=" * 50)
