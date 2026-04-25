"""Task 14 验证：推理能力释放"""
import sys
sys.path.insert(0, "/Users/mac/.openclaw/workspace/su-memory")
sys.path.insert(0, "/Users/mac/.openclaw/workspace/su-memory/src")

def test_causal_inference():
    from su_core._sys.causal import CausalInference
    ci = CausalInference()
    
    # 测试直接推理: 乾(金) → 离(火)
    # 乾 generates 离 in BAGUA_CAUSALITY
    result = ci.infer_relation("乾", "金", "离", "火")
    print(f"乾金 → 离火: {result}")
    assert result["relation"] in ("generates", "same"), f"乾应生离，得到 {result['relation']}"
    assert result["score"] > 0.5
    
    # 测试相克: 乾(金) → 巽(木), 乾 contradicts 巽
    # But 金生水 not 木, and 金克木 in wuxing
    result2 = ci.infer_relation("乾", "金", "巽", "木")
    print(f"乾金 → 巽木: {result2}")
    assert result2["relation"] == "contradicts"
    
    print("PASSED: causal inference")

def test_multi_hop():
    from su_core._sys.causal import CausalInference
    ci = CausalInference()
    
    memories = [
        {"id": "m1", "bagua_name": "离", "wuxing": "火"},
        {"id": "m2", "bagua_name": "震", "wuxing": "木"},
        {"id": "m3", "bagua_name": "坎", "wuxing": "水"},
        {"id": "m4", "bagua_name": "坤", "wuxing": "土"},
    ]
    
    results = ci.multi_hop_inference("乾", "金", memories, max_hops=2)
    print(f"多跳推理结果: {len(results)} 条")
    for r in results:
        print(f"  {r['id']}: score={r.get('hop_score', 0):.3f}, hops={r.get('hop_count', 0)}")
    print("PASSED: multi-hop inference")

def test_yijing_inference():
    from su_core._sys.yijing import YiJingInference, compute_shi_ying, HexagramType
    
    # 测试世爻/应爻
    shi, ying = compute_shi_ying(HexagramType.QIAN, HexagramType.QIAN)
    print(f"乾卦 世爻={shi}, 应爻={ying}")
    assert shi == 6 and ying == 3, "乾（八纯卦）世爻应为6，应爻应为3"
    
    shi2, ying2 = compute_shi_ying(HexagramType.QIAN, HexagramType.KUN)
    print(f"天地否 世爻={shi2}, 应爻={ying2}")
    
    # 测试三层推理
    yi = YiJingInference()
    results = yi.three_layer_retrieve(
        query_index=0,  # 乾卦
        candidate_indices=list(range(64)),
        top_k=5
    )
    print(f"三层推理检索 top-5:")
    for r in results:
        print(f"  index={r['index']}, score={r['score']:.3f}, trend={r.get('trend', '')}")
    print("PASSED: yijing inference")

def test_memory_yijing():
    from su_core._sys.yijing import YiJingInference
    yi = YiJingInference()
    
    my = yi.create_memory_yijing(0, "这是一条非常重要的确定性信息")
    print(f"MemoryYiJing: {my.ben_gua.name} → {my.hu_gua.name if my.hu_gua else 'None'} → {my.bian_gua.name if my.bian_gua else 'None'}")
    print(f"世爻={my.shi_yao}, 应爻={my.ying_yao}, 动爻={my.dong_yao}")
    
    trend = yi.get_trend_analysis(my)
    print(f"趋势分析: {trend}")
    print("PASSED: memory yijing creation")

if __name__ == "__main__":
    test_causal_inference()
    print("---")
    test_multi_hop()
    print("---")
    test_yijing_inference()
    print("---")
    test_memory_yijing()
    print("\n✅ All Task 14 tests passed!")
