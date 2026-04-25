"""
su-memory SDK Pro vs Lite 性能基准测试
验证增强版在各维度的提升
"""
import os
import sys
import time
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from su_memory.sdk.lite import SuMemoryLite
from su_memory.sdk.lite_pro import SuMemoryLitePro


def benchmark_query_lite(client, queries, name):
    """测试Lite版查询性能"""
    print(f"\n{name} - 查询性能测试 ({len(queries)}次)...")
    
    latencies = []
    for q in queries:
        start = time.perf_counter()
        client.query(q, top_k=5)
        latencies.append((time.perf_counter() - start) * 1000)
    
    latencies.sort()
    return {
        "avg_ms": sum(latencies) / len(latencies),
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
        "p99_ms": latencies[int(len(latencies) * 0.99)]
    }


def benchmark_multihop(client, queries, name):
    """测试多跳推理性能"""
    if not hasattr(client, 'query_multihop'):
        return None
    
    print(f"\n{name} - 多跳推理测试 ({len(queries)}次)...")
    
    latencies = []
    for q in queries:
        start = time.perf_counter()
        client.query_multihop(q, max_hops=3)
        latencies.append((time.perf_counter() - start) * 1000)
    
    if not latencies:
        return None
    
    latencies.sort()
    return {
        "avg_ms": sum(latencies) / len(latencies),
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
    }


def test_single_hop_retrieval():
    """单跳检索测试"""
    print("\n" + "="*60)
    print("测试1: 单跳检索能力")
    print("="*60)
    
    test_data = [
        ("苹果是一种水果", "水果"),
        ("苹果手机是苹果公司的产品", "手机"),
        ("香蕉也是一种水果", "水果"),
        ("Python是一种编程语言", "编程"),
        ("Java是另一种编程语言", "编程"),
        ("深度学习是机器学习的分支", "学习"),
        ("机器学习是AI的子领域", "AI"),
        ("人工智能改变世界", "智能"),
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Lite版本
        lite = SuMemoryLite(storage_path=tmpdir, enable_persistence=False)
        for content, _ in test_data:
            lite.add(content)
        
        # Pro版本
        pro = SuMemoryLitePro(storage_path=tmpdir, enable_vector=False, enable_persistence=False)
        for content, _ in test_data:
            pro.add(content)
        
        # 测试查询
        queries = ["水果", "编程", "学习"]
        
        # Lite查询
        lite_scores = []
        for q in queries:
            results = lite.query(q, top_k=3)
            if results:
                lite_scores.append(results[0]["score"])
        
        # Pro查询
        pro_scores = []
        for q in queries:
            results = pro.query(q, top_k=3)
            if results:
                pro_scores.append(results[0]["score"])
        
        print(f"\nLite查询相关性分数: {lite_scores}")
        print(f"Pro查询相关性分数: {pro_scores}")
        
        # 多跳推理测试（Pro独有）
        print("\n多跳推理测试:")
        pro.add("如果下雨，地面会湿", parent_ids=[])
        pro.add("地面湿了走路会滑", parent_ids=[pro._memories[0].id] if pro._memories else [])
        
        # 创建因果链
        if len(pro._memories) >= 2:
            pro.link_memories(pro._memories[-2].id, pro._memories[-1].id)
        
        multihop_results = pro.query_multihop("下雨", max_hops=3)
        print(f"多跳查询结果数量: {len(multihop_results)}")
        for r in multihop_results:
            print(f"  - {r['content']} (hops: {r['hops']})")
        
        assert len(multihop_results) > 0, "多跳推理应该返回结果"


def test_multi_hop_reasoning():
    """多跳推理测试"""
    print("\n" + "="*60)
    print("测试2: 多跳因果推理")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pro = SuMemoryLitePro(
            storage_path=tmpdir,
            enable_vector=False,
            enable_graph=True,
            enable_persistence=False
        )
        
        # 建立因果链
        m1 = pro.add("如果努力学习，成绩会提高")
        m2 = pro.add("成绩提高了会获得奖学金")
        m3 = pro.add("获得奖学金可以减轻家庭负担")
        
        pro.link_memories(m1, m2)
        pro.link_memories(m2, m3)
        
        print(f"\n因果链: {m1} -> {m2} -> {m3}")
        
        # 多跳查询
        results = pro.query_multihop("努力学习", max_hops=3)
        
        print(f"多跳查询'努力学习'结果:")
        
        for r in results[:5]:  # 最多显示5个
            print(f"  - {r['content'][:30]}... (hops: {r['hops']})")
        
        # 验证至少有结果
        has_results = len(results) > 0
        print(f"\n多跳推理有效: {has_results}")
        
        assert has_results, "多跳推理应该返回结果"


def test_temporal_understanding():
    """时序理解测试"""
    print("\n" + "="*60)
    print("测试3: 时序理解能力")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pro = SuMemoryLitePro(
            storage_path=tmpdir,
            enable_vector=False,
            enable_temporal=True,
            enable_persistence=False
        )
        
        # 获取当前time_code
        time_code = pro._temporal.get_time_code()
        print(f"当前time_code: {time_code}")
        
        # 测试时效性计算
        now = int(time.time())
        
        # 不同时间的记忆
        recent = pro.add("最近的会议结论", metadata={"type": "meeting"})
        old = pro.add("很久以前的会议结论", metadata={"type": "meeting"})
        
        # 模拟时间差
        recent_node = pro._memories[pro._memory_map[recent]]
        old_node = pro._memories[pro._memory_map[old]]
        
        # 旧记忆时间戳往前推
        old_node.timestamp = now - 86400 * 30  # 30天前
        
        # 计算时效分
        recency_recent = pro._temporal.calculate_recency_score(recent_node.timestamp, recent_node.energy_type)
        recency_old = pro._temporal.calculate_recency_score(old_node.timestamp, old_node.energy_type)
        
        print(f"\n新记忆时效分: {recency_recent:.4f}")
        print(f"旧记忆时效分: {recency_old:.4f}")
        print(f"时效性正确: {recency_recent > recency_old}")
        
        assert recency_recent > recency_old, "新记忆的时效分应该高于旧记忆"


def test_session_management():
    """多会话管理测试"""
    print("\n" + "="*60)
    print("测试4: 多会话管理")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pro = SuMemoryLitePro(
            storage_path=tmpdir,
            enable_vector=False,
            enable_session=True,
            enable_persistence=False
        )
        
        # 创建多个会话
        s1 = pro._sessions.create_session("项目A")
        s2 = pro._sessions.create_session("项目B")
        
        pro._sessions.set_current_session(s1)
        pro.add("项目A的第一个任务完成", topic="任务")
        pro.add("项目A的第二个任务开始", topic="任务")
        
        pro._sessions.set_current_session(s2)
        pro.add("项目B的任务规划完成", topic="规划")
        
        # 验证会话隔离
        s1_mems = pro._sessions.get_session_memories(s1)
        s2_mems = pro._sessions.get_session_memories(s2)
        
        print(f"\n项目A会话记忆数: {len(s1_mems)}")
        print(f"项目B会话记忆数: {len(s2_mems)}")
        
        # 跨会话话题召回
        task_mems = pro._sessions.get_topic_memories("任务")
        print(f"任务话题跨会话记忆: {task_mems}")
        
        session_correct = len(s1_mems) == 2 and len(s2_mems) == 1
        print(f"会话管理正确: {session_correct}")
        
        assert session_correct, "会话管理应该正确隔离记忆"


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("🚀 su-memory SDK Pro 能力验证测试")
    print("="*60)
    
    results = {}
    
    # 测试1: 单跳检索
    try:
        results["单跳检索"] = test_single_hop_retrieval()
    except Exception as e:
        print(f"单跳检索测试失败: {e}")
        results["单跳检索"] = False
    
    # 测试2: 多跳推理
    try:
        results["多跳推理"] = test_multi_hop_reasoning()
    except Exception as e:
        print(f"多跳推理测试失败: {e}")
        results["多跳推理"] = False
    
    # 测试3: 时序理解
    try:
        results["时序理解"] = test_temporal_understanding()
    except Exception as e:
        print(f"时序理解测试失败: {e}")
        results["时序理解"] = False
    
    # 测试4: 多会话
    try:
        results["多会话"] = test_session_management()
    except Exception as e:
        print(f"多会话测试失败: {e}")
        results["多会话"] = False
    
    # 打印汇总
    print("\n" + "="*60)
    print("📊 测试结果汇总")
    print("="*60)
    
    for name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
    
    passed_count = sum(1 for v in results.values() if v)
    print(f"\n通过: {passed_count}/{len(results)}")
    
    return all(results.values())


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
