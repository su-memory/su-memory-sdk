"""
SuMemoryLitePro + Ollama bge-m3 向量检索测试
验证Ollama embedding集成效果
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from su_memory.sdk.lite_pro import SuMemoryLitePro


def test_ollama_embedding():
    """测试Ollama embedding"""
    print("Testing Ollama bge-m3 embedding...")
    
    pro = SuMemoryLitePro(
        storage_path=None,
        embedding_backend='ollama',
        enable_vector=True,
        enable_graph=True,
        enable_temporal=True,
        enable_session=True
    )
    
    assert pro.enable_vector == True
    print(f"Vector enabled: {pro.enable_vector}")
    
    # 添加测试数据
    memories = [
        '苹果是一种水果，富含维生素C',
        '苹果手机是苹果公司的产品',
        '香蕉也是一种水果',
        'Python是一种编程语言',
        'Java是另一种编程语言',
        '深度学习是机器学习的分支',
        '机器学习是AI的子领域',
        '人工智能改变世界',
    ]
    
    for mem in memories:
        pro.add(mem)
    
    assert len(pro) == 8
    print(f"Added {len(pro)} memories")


def test_semantic_retrieval():
    """测试语义检索"""
    print("\nTesting semantic retrieval...")
    
    pro = SuMemoryLitePro(
        storage_path=None,
        embedding_backend='ollama',
        enable_vector=True,
        enable_graph=True,
        enable_temporal=True,
        enable_session=True
    )
    
    # 添加测试数据
    memories = [
        ('苹果是一种水果', '水果'),
        ('香蕉也是一种水果', '水果'),
        ('橙子是柑橘类水果', '水果'),
        ('苹果手机是苹果公司的产品', '手机'),
        ('华为手机是中国品牌', '手机'),
    ]
    
    for content, _ in memories:
        pro.add(content)
    
    # 测试语义检索
    results = pro.query('水果', use_vector=True, use_keyword=False, top_k=3)
    print(f"Query '水果' results:")
    for r in results:
        print(f"  - {r['content']} (score: {r['score']:.4f})")
    
    # 验证水果相关内容排在前面
    assert len(results) > 0
    assert '水果' in results[0]['content']
    print("Semantic retrieval test passed!")


def test_cross_domain_retrieval():
    """测试跨域检索"""
    print("\nTesting cross-domain retrieval...")
    
    pro = SuMemoryLitePro(
        storage_path=None,
        embedding_backend='ollama',
        enable_vector=True,
        enable_graph=True,
        enable_temporal=True,
        enable_session=True
    )
    
    # 添加测试数据
    memories = [
        '苹果是一种水果，富含维生素C',
        '苹果手机是苹果公司的产品',
        'Python是一种编程语言',
        '深度学习是机器学习的分支',
    ]
    
    for mem in memories:
        pro.add(mem)
    
    # 测试跨域查询 - "编程"不应该返回"水果"
    results = pro.query('编程语言', use_vector=True, use_keyword=False, top_k=3)
    print(f"Query '编程语言' results:")
    for r in results:
        print(f"  - {r['content']} (score: {r['score']:.4f})")
    
    # 验证编程相关内容排在前面
    found_programming = False
    for r in results[:2]:
        if '编程' in r['content']:
            found_programming = True
            break
    
    assert found_programming, "Should find programming content"
    print("Cross-domain retrieval test passed!")


def test_multihop_with_ollama():
    """测试多跳推理"""
    print("\nTesting multihop reasoning...")
    
    pro = SuMemoryLitePro(
        storage_path=None,
        embedding_backend='ollama',
        enable_vector=True,
        enable_graph=True,
        enable_temporal=True,
        enable_session=True
    )
    
    # 添加因果链
    m1 = pro.add('如果努力学习，成绩会提高')
    m2 = pro.add('成绩提高了会获得奖学金')
    m3 = pro.add('获得奖学金可以减轻家庭负担')
    
    pro.link_memories(m1, m2)
    pro.link_memories(m2, m3)
    
    # 多跳查询
    results = pro.query_multihop('努力学习', max_hops=3)
    print(f"Multihop query '努力学习' results:")
    for r in results[:5]:
        print(f"  - {r['content'][:30]}... (hops: {r['hops']})")
    
    # 验证因果链被正确检索
    assert len(results) >= 3, "Should find at least 3 related memories"
    print("Multihop reasoning test passed!")


def run_all_tests():
    """运行所有测试"""
    print("="*60)
    print("SuMemoryLitePro + Ollama bge-m3 向量检索测试")
    print("="*60)
    
    tests = [
        test_ollama_embedding,
        test_semantic_retrieval,
        test_cross_domain_retrieval,
        test_multihop_with_ollama,
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✅ {test.__name__}")
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"测试结果: {passed}/{len(tests)} 通过")
    print("="*60)
    
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
