"""Task 16 验证：SDK优化与性能提升"""
import sys
sys.path.insert(0, "/Users/mac/.openclaw/workspace/su-memory/src")
sys.path.insert(0, "/Users/mac/.openclaw/workspace/su-memory")

import time

def test_basic_add_query():
    from su_memory import SuMemory
    client = SuMemory()
    
    mid1 = client.add("我喜欢吃苹果和橘子", metadata={"type": "preference"})
    mid2 = client.add("明天下午3点有个重要会议", metadata={"type": "event"})
    mid3 = client.add("Python是一门优秀的编程语言", metadata={"type": "knowledge"})
    mid4 = client.add("我最近学会了做红烧肉", metadata={"type": "event"})
    mid5 = client.add("苹果富含维生素C，对健康有益", metadata={"type": "fact"})
    
    print(f"已添加 {len(client)} 条记忆")
    
    results = client.query("水果和健康")
    print(f"查询 '水果和健康' 结果 ({len(results)} 条):")
    for r in results:
        print(f"  [{r.score:.4f}] {r.content[:30]}... bagua={r.encoding.bagua}")
    
    assert len(results) > 0, "应返回至少一条结果"
    # 验证苹果/维生素相关的在结果中
    contents = [r.content for r in results]
    assert any("苹果" in c or "维生素" in c for c in contents), "应包含水果相关结果"
    print("PASSED: basic add/query")

def test_semantic_quality():
    from su_memory import SuMemory
    client = SuMemory()
    
    client.add("我非常喜欢跑步和游泳", metadata={"type": "preference"})
    client.add("股票市场今天大涨了5%", metadata={"type": "event"})
    client.add("我喜欢各种户外运动", metadata={"type": "preference"})
    client.add("明天天气预报说会下雨", metadata={"type": "fact"})
    
    results = client.query("运动爱好")
    print(f"查询 '运动爱好' 结果:")
    for r in results:
        print(f"  [{r.score:.4f}] {r.content}")
    
    # 运动相关的应该在 top-3 中（考虑英文模型对中文语义的有限支持）
    top3_contents = " ".join(r.content for r in results[:3])
    has_sport = "运动" in top3_contents or "跑步" in top3_contents or "游泳" in top3_contents
    assert has_sport, f"Top-3 应包含运动相关内容，得到: {top3_contents}"
    print("PASSED: semantic quality")

def test_performance():
    from su_memory import SuMemory
    client = SuMemory()
    
    topics = ["喜欢水果", "重要会议", "编程学习", "健康饮食", "旅行计划",
              "工作安排", "家庭关系", "投资理财", "运动健身", "读书笔记"]
    
    for i in range(100):
        topic = topics[i % len(topics)]
        client.add(f"{topic}第{i+1}条记忆内容：这是一段测试文本，包含{topic}相关信息。")
    
    start = time.time()
    for _ in range(10):
        client.query("投资回报和理财", top_k=5)
    elapsed = (time.time() - start) / 10
    
    print(f"100条记忆，平均查询时间: {elapsed*1000:.1f}ms")
    assert elapsed < 2.0, f"查询时间应小于2秒，得到 {elapsed:.3f}秒"
    print("PASSED: performance (100 records)")

def test_index_structure():
    from su_memory import SuMemory
    client = SuMemory()
    
    client.add("确定性规则", metadata={"type": "fact"})
    client.add("快乐偏好", metadata={"type": "preference"})
    client.add("突然事件", metadata={"type": "event"})
    
    assert hasattr(client, '_bagua_index'), "应有八卦索引"
    assert hasattr(client, '_wuxing_index'), "应有五行索引"
    assert hasattr(client, '_vectors'), "应有向量存储"
    
    total_indexed = sum(len(v) for v in client._bagua_index.values())
    assert total_indexed == 3, f"索引应包含3条记忆，得到 {total_indexed}"
    
    assert len(client._vectors) == 3, "向量列表应有3条"
    assert all(v is not None for v in client._vectors), "所有向量不应为None（模型已加载）"
    
    print(f"八卦索引: {dict(client._bagua_index)}")
    print(f"五行索引: {dict(client._wuxing_index)}")
    print("PASSED: index structure")

def test_encoding_info():
    from su_memory import SuMemory
    client = SuMemory()
    
    client.add("我喜欢编程", metadata={"type": "preference"})
    results = client.query("编程")
    
    assert len(results) > 0, "应有结果"
    enc = results[0].encoding
    print(f"编码信息: bagua={enc.bagua}, wuxing={enc.wuxing}, hexagram={enc.hexagram}")
    print(f"  hexagram_name={enc.hexagram_name}, causal_depth={enc.causal_depth}")
    print(f"  bagua_probs keys: {list(enc.bagua_probs.keys()) if enc.bagua_probs else None}")
    print(f"  wuxing_scores keys: {list(enc.wuxing_scores.keys()) if enc.wuxing_scores else None}")
    assert enc.bagua != "", "八卦不应为空"
    assert enc.wuxing != "", "五行不应为空"
    assert enc.bagua_probs is not None, "应有八卦概率分布"
    assert enc.wuxing_scores is not None, "应有五行得分"
    print("PASSED: encoding info")

def test_multi_hop():
    from su_memory import SuMemory
    client = SuMemory()
    
    client.add("火能融化金属", metadata={"type": "fact"})
    client.add("木材可以生火", metadata={"type": "fact"})
    client.add("水可以浇灭火焰", metadata={"type": "fact"})
    client.add("金属可以砍伐树木", metadata={"type": "fact"})
    
    results = client.query_multi_hop("木材的用途", top_k=5)
    print(f"多跳检索结果 ({len(results)} 条):")
    for r in results:
        print(f"  [{r.score:.4f}] {r.content} (depth={r.encoding.causal_depth})")
    assert len(results) > 0, "多跳应返回结果"
    print("PASSED: multi-hop query")

def test_backward_compat():
    """测试向后兼容：add/query 接口签名不变"""
    from su_memory import SuMemory
    client = SuMemory()
    
    # 无 metadata 调用
    mid = client.add("简单测试")
    assert mid.startswith("mem_"), f"ID 格式应为 mem_xxx，得到: {mid}"
    
    results = client.query("测试")
    assert isinstance(results, list), "query 应返回列表"
    
    stats = client.get_stats()
    assert "total_memories" in stats, "stats 应含 total_memories"
    assert stats["total_memories"] == 1
    
    print(f"backward compat: id={mid}, stats={stats}")
    print("PASSED: backward compatibility")

if __name__ == "__main__":
    test_basic_add_query()
    print("---")
    test_semantic_quality()
    print("---")
    test_performance()
    print("---")
    test_index_structure()
    print("---")
    test_encoding_info()
    print("---")
    test_multi_hop()
    print("---")
    test_backward_compat()
    print("\n✅ All Task 16 tests passed!")
