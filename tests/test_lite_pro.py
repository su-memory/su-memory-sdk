"""
SuMemoryLitePro 功能测试
验证向量检索、多跳推理、时序系统、会话管理
"""
import os
import sys
import time
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from su_memory.sdk.lite_pro import SuMemoryLitePro


def test_basic_operations():
    """测试基础操作"""
    print("\n" + "="*60)
    print("测试1: 基础操作")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        client = SuMemoryLitePro(
            storage_path=tmpdir,
            enable_vector=False,  # 先用hash fallback测试
            enable_graph=True,
            enable_temporal=True,
            enable_session=True
        )
        
        # 添加记忆
        mid1 = client.add("今天天气很好，阳光明媚")
        mid2 = client.add("明天可能下雨，记得带伞")
        mid3 = client.add("我喜欢学习编程，特别是Python")
        mid4 = client.add("周末想去公园散步")
        
        print(f"添加了4条记忆: {mid1}, {mid2}, {mid3}, {mid4}")
        print(f"当前记忆数量: {len(client)}")
        
        # 查询
        results = client.query("天气", top_k=3)
        print(f"\n查询'天气'结果:")
        for r in results:
            print(f"  - {r['content']} (score: {r['score']:.4f})")
        
        stats = client.get_stats()
        print(f"\n统计: {stats}")
        
        assert len(client) == 4
        assert len(results) > 0
        print("✅ 基础操作测试通过")


def test_graph_operations():
    """测试图谱操作"""
    print("\n" + "="*60)
    print("测试2: 图谱与多跳推理")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        client = SuMemoryLitePro(
            storage_path=tmpdir,
            enable_vector=False,
            enable_graph=True,
            enable_temporal=True
        )
        
        # 添加记忆并建立因果链
        mid1 = client.add("如果下雨，地面会湿")
        mid2 = client.add("地面湿了，走路会滑")
        mid3 = client.add("走路滑了可能摔倒")
        
        # 建立链接
        client.link_memories(mid1, mid2)
        client.link_memories(mid2, mid3)
        
        print(f"建立了因果链: {mid1} -> {mid2} -> {mid3}")
        
        # 多跳查询
        results = client.query_multihop("下雨", max_hops=3)
        print(f"\n多跳查询'下雨'结果:")
        for r in results:
            print(f"  - {r['content']} (hops: {r['hops']}, score: {r['score']:.4f})")
        
        # 获取子节点
        children = client.get_children(mid1)
        print(f"\n{mid1}的子节点: {[c['memory_id'] for c in children]}")
        
        assert len(results) > 0
        print("✅ 图谱操作测试通过")


def test_temporal_system():
    """测试时序系统"""
    print("\n" + "="*60)
    print("测试3: 时序系统")
    print("="*60)
    
    from su_memory.sdk.lite_pro import TemporalSystem
    
    temporal = TemporalSystem()
    
    # 获取当前time_code
    time_code = temporal.get_time_code()
    print(f"当前time_code: {time_code}")
    
    # 计算时效性
    old_timestamp = int(time.time()) - 86400 * 7  # 7天前
    recent_timestamp = int(time.time()) - 3600  # 1小时前
    
    old_score = temporal.calculate_recency_score(old_timestamp, "wood")
    recent_score = temporal.calculate_recency_score(recent_timestamp, "wood")
    
    print(f"7天前的记忆时效分: {old_score:.4f}")
    print(f"1小时前的记忆时效分: {recent_score:.4f}")
    
    assert recent_score > old_score
    print("✅ 时序系统测试通过")


def test_session_management():
    """测试会话管理"""
    print("\n" + "="*60)
    print("测试4: 会话管理")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        client = SuMemoryLitePro(
            storage_path=tmpdir,
            enable_vector=False,
            enable_session=True
        )
        
        # 创建两个会话
        session1 = client._sessions.create_session("会议讨论")
        session2 = client._sessions.create_session("项目规划")
        
        print(f"创建会话: {session1}, {session2}")
        
        # 在会话1中添加记忆
        client._sessions.set_current_session(session1)
        mid1 = client.add("会议决定采用新技术方案", topic="技术")
        mid2 = client.add("会议确定了下周发布日期", topic="进度")
        
        # 在会话2中添加记忆
        client._sessions.set_current_session(session2)
        mid3 = client.add("项目第一阶段完成", topic="进度")
        
        # 获取会话1的记忆
        session1_mems = client._sessions.get_session_memories(session1)
        print(f"会话1的记忆: {session1_mems}")
        
        # 按话题查询
        tech_mems = client._sessions.get_topic_memories("技术")
        progress_mems = client._sessions.get_topic_memories("进度")
        print(f"技术话题记忆: {tech_mems}")
        print(f"进度话题记忆: {progress_mems}")
        
        stats = client.get_stats()
        print(f"统计: sessions={stats['sessions']}")
        
        print("✅ 会话管理测试通过")


def test_hybrid_retrieval():
    """测试混合检索"""
    print("\n" + "="*60)
    print("测试5: 混合检索")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        client = SuMemoryLitePro(
            storage_path=tmpdir,
            enable_vector=False,  # 使用hash fallback
            enable_tfidf=True,
            enable_temporal=True
        )
        
        # 添加记忆
        client.add("苹果是一种水果，富含维生素C")
        client.add("苹果手机是苹果公司的产品")
        client.add("香蕉也是一种水果")
        client.add("橙子是柑橘类水果")
        
        # 关键词检索
        kw_results = client.query("水果", use_vector=False, use_keyword=True)
        print(f"关键词检索'水果': {[r['content'] for r in kw_results]}")
        
        # 语义检索（使用embedding）
        vec_results = client.query("苹果", use_vector=True, use_keyword=False)
        print(f"向量检索'苹果': {[r['content'] for r in vec_results]}")
        
        # 混合检索（RRF融合）
        hybrid_results = client.query("水果", use_vector=True, use_keyword=True)
        print(f"混合检索'水果': {[r['content'] for r in hybrid_results]}")
        
        print("✅ 混合检索测试通过")


def test_memory_operations():
    """测试记忆CRUD操作"""
    print("\n" + "="*60)
    print("测试6: 记忆CRUD")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        client = SuMemoryLitePro(
            storage_path=tmpdir,
            enable_vector=False,
            enable_graph=True
        )
        
        # 添加
        mid = client.add("测试记忆", metadata={"type": "test"})
        print(f"添加记忆: {mid}")
        
        # 获取
        mem = client.get_memory(mid)
        print(f"获取记忆: {mem['content']}")
        
        # 链接
        mid2 = client.add("关联记忆")
        client.link_memories(mid, mid2)
        
        # 获取父子
        parents = client.get_parents(mid2)
        children = client.get_children(mid)
        print(f"记忆1的子节点: {[c['memory_id'] for c in children]}")
        print(f"记忆2的父节点: {[p['memory_id'] for p in parents]}")
        
        # 统计
        stats = client.get_stats()
        print(f"统计: {stats}")
        
        print("✅ 记忆CRUD测试通过")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("🚀 su-memory SDK Pro 功能测试")
    print("="*60)
    
    try:
        test_basic_operations()
        test_graph_operations()
        test_temporal_system()
        test_session_management()
        test_hybrid_retrieval()
        test_memory_operations()
        
        print("\n" + "="*60)
        print("🎉 所有测试通过!")
        print("="*60)
        
        return True
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
