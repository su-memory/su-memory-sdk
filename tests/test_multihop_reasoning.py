"""
su-memory SDK 多跳推理测试
基于 VectorGraphRAG 技术

测试场景：
1. 单跳语义检索：验证基础向量搜索
2. 多跳因果推理：验证关系向量搜索
3. 同义词扩展：验证语义理解能力
4. 性能基准：验证大规模数据下的表现

使用方法:
    python test_multihop_reasoning.py
"""

import sys
import time
sys.path.insert(0, 'src')

from su_memory import SuMemoryLitePro


def test_semantic_recall():
    """测试1：单跳语义检索"""
    print("\n" + "=" * 60)
    print("测试1：单跳语义检索")
    print("=" * 60)
    
    pro = SuMemoryLitePro()
    
    # 添加测试记忆
    test_data = [
        "机器学习是人工智能的核心技术",
        "深度学习是机器学习的一个重要分支",
        "神经网络是深度学习的基础架构",
        "卷积神经网络在图像识别中表现优异",
        "自然语言处理是AI的重要应用领域",
        "Transformer模型革新了自然语言处理",
        "大语言模型基于Transformer架构",
        "GPT是OpenAI开发的大语言模型",
    ]
    
    ids = []
    for content in test_data:
        mid = pro.add(content)
        ids.append(mid)
        print(f"  + {mid[:12]}...: {content[:30]}...")
    
    print(f"\n共添加 {len(test_data)} 条记忆")
    
    # 测试查询
    queries = [
        ("深度学习", ["深度学习是机器学习的一个重要分支"]),
        ("神经网络", ["神经网络是深度学习的基础架构"]),
        ("Transformer", ["Transformer模型革新了自然语言处理"]),
        ("GPT", ["GPT是OpenAI开发的大语言模型"]),
    ]
    
    total = 0
    correct = 0
    
    for query, expected in queries:
        print(f"\n查询: \"{query}\"")
        results = pro.query(query, top_k=3)
        
        if results:
            top_result = results[0]["content"]
            print(f"  Top结果: {top_result[:40]}...")
            
            # 检查是否包含期望关键词
            for exp in expected:
                if any(word in top_result for word in exp.split() if len(word) > 2):
                    correct += 1
                    print(f"  ✓ 命中: {exp}")
                else:
                    print(f"  ✗ 漏检: {exp}")
            total += 1
        
        time.sleep(0.1)  # 避免API过载
    
    recall = (correct / total * 100) if total > 0 else 0
    print(f"\n单跳检索召回率: {recall:.1f}% ({correct}/{total})")
    
    return recall >= 80


def test_multihop_reasoning():
    """测试2：多跳因果推理"""
    print("\n" + "=" * 60)
    print("测试2：多跳因果推理")
    print("=" * 60)
    
    pro = SuMemoryLitePro()
    
    # 构建因果链：机器学习 → 深度学习 → 神经网络 → CNN
    m0 = pro.add("机器学习是人工智能的核心技术")
    m1 = pro.add("深度学习是机器学习的一个重要分支", parent_ids=[m0])
    m2 = pro.add("深度学习在图像识别中表现优异", parent_ids=[m1])
    m3 = pro.add("神经网络是深度学习的基础架构", parent_ids=[m1])
    m4 = pro.add("卷积神经网络是神经网络的一种", parent_ids=[m3])
    m5 = pro.add("ResNet是经典的卷积神经网络架构", parent_ids=[m4])
    
    print(f"构建因果链: m0 → m1 → m2/m3 → m4 → m5")
    print(f"节点数: {len(pro._vector_graph.nodes)}, 边数: {len(pro._vector_graph.edges)}")
    
    # 测试多跳查询
    test_cases = [
        # (查询, 期望路径跳数, 期望包含关键词)
        ("深度学习的影响", 2, ["机器学习", "图像识别", "神经网络"]),
        ("神经网络的作用", 2, ["深度学习", "卷积神经网络"]),
        ("从机器学习到CNN", 4, ["深度学习", "神经网络", "卷积神经网络"]),
    ]
    
    total = 0
    correct = 0
    
    for query, min_hops, expected_keywords in test_cases:
        print(f"\n查询: \"{query}\"")
        print(f"期望: 至少{min_hops}跳，包含 {expected_keywords}")
        
        results = pro.query_multihop(query, max_hops=4, top_k=10)
        
        if results:
            # 检查是否有足够跳数的结果
            max_actual_hops = max(r.get("hops", 0) for r in results)
            
            # 检查结果中是否包含期望关键词
            all_content = " ".join(r["content"] for r in results[:5])
            hits = [kw for kw in expected_keywords if kw in all_content]
            
            print(f"  实际最大跳数: {max_actual_hops}")
            print(f"  关键词命中: {hits}")
            
            if max_actual_hops >= min_hops and len(hits) >= len(expected_keywords) // 2:
                correct += 1
                print(f"  ✓ 通过")
            else:
                print(f"  ✗ 未达标")
            total += 1
        else:
            print("  (无结果)")
            total += 1
        
        time.sleep(0.1)
    
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\n多跳推理通过率: {accuracy:.1f}% ({correct}/{total})")
    
    return accuracy >= 60


def test_synonym_expansion():
    """测试3：同义词扩展"""
    print("\n" + "=" * 60)
    print("测试3：同义词扩展（语义理解）")
    print("=" * 60)
    
    pro = SuMemoryLitePro()
    
    # 添加包含特定词的记忆
    test_data = [
        "会议讨论了项目进度",
        "项目组召开了周例会",
        "深度学习在计算机视觉中的应用",
        "人工智能技术在医疗领域的突破",
    ]
    
    for content in test_data:
        pro.add(content)
    
    # 用同义词查询
    test_cases = [
        ("开会", ["会议", "召开"]),
        ("机器学习", ["深度学习", "人工智能"]),
        ("CV", ["计算机视觉", "视觉"]),
    ]
    
    total = 0
    correct = 0
    
    for query, expected_words in test_cases:
        print(f"\n查询: \"{query}\"")
        print(f"期望包含: {expected_words}")
        
        results = pro.query(query, top_k=3)
        
        if results:
            all_content = " ".join(r["content"] for r in results)
            hits = [w for w in expected_words if w in all_content]
            
            print(f"  Top结果: {results[0]['content'][:40]}...")
            print(f"  命中: {hits}")
            
            if hits:
                correct += 1
                print(f"  ✓ 语义理解生效")
            else:
                print(f"  ✗ 仅关键词匹配")
            total += 1
        
        time.sleep(0.1)
    
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\n同义词扩展通过率: {accuracy:.1f}% ({correct}/{total})")
    
    return accuracy >= 50


def test_performance():
    """测试4：性能基准"""
    print("\n" + "=" * 60)
    print("测试4：性能基准")
    print("=" * 60)
    
    pro = SuMemoryLitePro()
    
    # 添加大量记忆
    n_memories = 100
    print(f"添加 {n_memories} 条记忆...")
    
    start = time.time()
    for i in range(n_memories):
        pro.add(f"测试记忆 {i}：这是一条用于性能测试的记录，包含一些常见的中文词汇和内容。")
    add_time = time.time() - start
    print(f"添加耗时: {add_time*1000:.1f}ms ({add_time*1000/n_memories:.1f}ms/条)")
    
    # 测试查询延迟
    queries = ["测试记忆 50", "记录 30", "中文词汇"]
    
    print(f"\n查询性能测试...")
    total_latency = 0
    
    for query in queries:
        start = time.time()
        results = pro.query(query, top_k=5)
        latency = (time.time() - start) * 1000
        total_latency += latency
        
        print(f"  \"{query}\": {latency:.1f}ms ({len(results)} 结果)")
        
        time.sleep(0.05)
    
    avg_latency = total_latency / len(queries)
    print(f"\n平均查询延迟: {avg_latency:.1f}ms")
    
    return avg_latency < 500  # 100条记忆下延迟应小于500ms


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("su-memory SDK 多跳推理测试套件")
    print("基于 VectorGraphRAG 技术")
    print("=" * 70)
    
    results = {}
    
    # 测试1：单跳语义检索
    try:
        results["语义检索"] = test_semantic_recall()
    except Exception as e:
        print(f"\n测试1 异常: {e}")
        results["语义检索"] = False
    
    # 测试2：多跳因果推理
    try:
        results["多跳推理"] = test_multihop_reasoning()
    except Exception as e:
        print(f"\n测试2 异常: {e}")
        results["多跳推理"] = False
    
    # 测试3：同义词扩展
    try:
        results["同义词扩展"] = test_synonym_expansion()
    except Exception as e:
        print(f"\n测试3 异常: {e}")
        results["同义词扩展"] = False
    
    # 测试4：性能基准
    try:
        results["性能基准"] = test_performance()
    except Exception as e:
        print(f"\n测试4 异常: {e}")
        results["性能基准"] = False
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
    
    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    print(f"\n总计: {passed_count}/{total_count} 通过")
    
    score = (passed_count / total_count) * 5
    print(f"综合评分: {score:.1f}/5.0")
    
    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
