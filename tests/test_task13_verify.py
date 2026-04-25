"""Task 13 验证：多维融合检索"""
import sys
sys.path.insert(0, "/Users/mac/.openclaw/workspace/su-memory")
sys.path.insert(0, "/Users/mac/.openclaw/workspace/su-memory/src")

def test_fusion_weights():
    from su_core._sys.fusion import MultiViewRetriever
    r = MultiViewRetriever()
    assert "semantic" in r._weights, "应包含 semantic 权重"
    assert "causal" in r._weights, "应包含 causal 权重"
    assert abs(sum(r._weights.values()) - 1.0) < 0.01, "权重之和应为 1.0"
    print(f"权重配置: {r._weights}")
    print("PASSED: fusion weights")

def test_wuxing_five_states():
    from su_core._sys._c2 import Wuxing, get_wuxing_state
    state, intensity = get_wuxing_state(Wuxing.MU, Wuxing.MU)
    assert state == "旺", f"木在春季应为旺，得到 {state}"
    assert intensity == 2.0

    state2, intensity2 = get_wuxing_state(Wuxing.HUO, Wuxing.MU)
    assert state2 == "相", f"火在春季应为相，得到 {state2}"
    assert intensity2 == 1.3
    print("PASSED: wuxing five states")

def test_wuxing_cheng_wu():
    from su_core._sys._c2 import Wuxing, check_cheng_wu
    # 金克木，金太强 → 乘
    result = check_cheng_wu(Wuxing.JIN, Wuxing.MU, 3.0, 1.0)
    assert result == "cheng", f"应为乘，得到 {result}"

    # 金克木，木太强 → 侮
    result2 = check_cheng_wu(Wuxing.JIN, Wuxing.MU, 1.0, 3.0)
    assert result2 == "wu", f"应为侮，得到 {result2}"
    print("PASSED: wuxing cheng/wu")

def test_wuxing_similarity():
    from su_core._sys._c2 import Wuxing, wuxing_similarity
    assert wuxing_similarity(Wuxing.MU, Wuxing.MU) == 1.0
    assert wuxing_similarity(Wuxing.MU, Wuxing.HUO) == 0.7  # 木生火
    assert wuxing_similarity(Wuxing.MU, Wuxing.TU) == 0.1   # 木克土
    print("PASSED: wuxing similarity")

def test_fusion_retrieve():
    from su_core._sys.fusion import MultiViewRetriever
    from su_core._sys.encoders import SemanticEncoder, EncodingInfo

    encoder = SemanticEncoder()
    retriever = MultiViewRetriever()

    query_info = encoder.encode("我喜欢吃苹果", "preference")

    # 模拟候选
    candidates = []
    for i, text in enumerate(["我喜欢吃橘子", "明天有会议", "苹果很好吃"]):
        info = encoder.encode(text, "fact")
        candidates.append({
            "id": f"mem_{i}",
            "content": text,
            "hexagram_index": info.index,
            "vector_score": 0.8 - i * 0.2,
            "bagua_probs": info.bagua_probs,
            "wuxing_scores": info.wuxing_scores,
            "payload": {"content": text, "wuxing": info.wuxing, "hexagram_name": info.name},
        })

    results = retriever.retrieve(
        query_content="我喜欢吃苹果",
        query_hexagram=query_info,
        candidates=candidates,
        top_k=3
    )

    print(f"融合检索结果 ({len(results)} 条):")
    for r in results:
        print(f"  {r['content']}: holographic={r.get('holographic_score', 0):.4f}")
        if 'fusion_detail' in r:
            print(f"    detail: {r['fusion_detail']}")
    print("PASSED: fusion retrieve")

if __name__ == "__main__":
    test_fusion_weights()
    print("---")
    test_wuxing_five_states()
    print("---")
    test_wuxing_cheng_wu()
    print("---")
    test_wuxing_similarity()
    print("---")
    test_fusion_retrieve()
    print("\n✅ All Task 13 tests passed!")
