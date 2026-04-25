"""
Phase 1 集成测试
测试su_core全部模块的功能正确性
"""

import sys
sys.path.insert(0, '.')

import time


def test_all_modules():
    """Phase 1 全部模块功能测试"""
    
    print("=" * 60)
    print("su-memory Phase 1 集成测试")
    print("=" * 60)
    
    from su_core import (
        SemanticEncoder, EncoderCore, EncodingInfo,
        MultiViewRetriever, SuCompressor,
        TemporalSystem, TemporalInfo, DynamicPriority,
        BeliefTracker, BeliefState, BeliefStage,
        MetaCognition, CognitiveGap, KnowledgeAging
    )
    
    results = []
    
    # ==================== 编码器测试 ====================
    print("\n[1/8] SemanticEncoder 测试...")
    try:
        encoder = SemanticEncoder()
        
        # 测试编码
        info1 = encoder.encode("用户有高血压病史", "fact")
        assert info1.name in ["乾", "坤", "屯", "蒙", "需", "讼", "师", "比",
                               "小畜", "履", "泰", "否", "同人", "大有", "谦", "豫",
                               "随", "蛊", "临", "观", "噬嗑", "贲", "剥", "复",
                               "无妄", "大畜", "颐", "大过", "坎", "离", "咸", "恒",
                               "遁", "大壮", "晋", "明夷", "家人", "睽", "蹇", "解",
                               "损", "益", "夬", "姤", "萃", "升", "困", "井",
                               "革", "鼎", "震", "艮", "渐", "归妹", "丰", "旅",
                               "巽", "兑", "涣", "节", "中孚", "小过", "既济", "未济"], f"无效卦名: {info1.name}"
        assert info1.wuxing in ["金", "木", "水", "火", "土"], f"无效五行: {info1.wuxing}"
        
        # 测试批量编码
        batch = encoder.batch_encode([
            {"content": "测试1", "type": "fact"},
            {"content": "测试2", "type": "preference"}
        ])
        assert len(batch) == 2
        
        print(f"  ✓ 编码器: 内容→{info1.name}卦({info1.wuxing})")
        results.append(("SemanticEncoder", True, ""))
    except Exception as e:
        results.append(("SemanticEncoder", False, str(e)))
        print(f"  ✗ 失败: {e}")
    
    # ==================== 编码核心测试 ====================
    print("\n[2/8] EncoderCore 测试...")
    try:
        core = EncoderCore()
        
        # 测试全息视图
        views = core.get_holographic_views(0)
        assert "本卦" in views and "互卦" in views and "综卦" in views and "错卦" in views
        assert views["本卦"] == 0
        
        # 测试全息检索
        candidates = [0, 1, 2, 3, 4, 5]
        scored = core.retrieve_holographic(0, candidates, top_k=3)
        assert len(scored) <= 3
        assert scored[0][0] == 0  # 本卦应该排第一
        
        print(f"  ✓ 编码核心: 4视图检索, top-3相关性{len(scored)}条")
        results.append(("EncoderCore", True, ""))
    except Exception as e:
        results.append(("EncoderCore", False, str(e)))
        print(f"  ✗ 失败: {e}")
    
    # ==================== 压缩器测试 ====================
    print("\n[3/8] SuCompressor 测试...")
    try:
        compressor = SuCompressor()
        
        # 无损压缩
        lossless = compressor.compress("这是一个测试内容", mode="lossless")
        assert lossless["ratio"] > 1.0
        
        # 平衡压缩
        balanced = compressor.compress("这是一个测试内容用于验证压缩功能是否正常工作", mode="balanced")
        assert "compressed" in balanced
        
        # 语义压缩
        semantic = compressor.compress("用户孩子5岁有扁平足需要预约骨科检查", mode="semantic")
        assert "compressed" in semantic
        assert "entities" in semantic
        
        print(f"  ✓ 压缩器: 无损{lossless['ratio']}x, 语义{len(semantic.get('entities',[]))}个实体")
        results.append(("SuCompressor", True, ""))
    except Exception as e:
        results.append(("SuCompressor", False, str(e)))
        print(f"  ✗ 失败: {e}")
    
    # ==================== 时序系统测试 ====================
    print("\n[4/8] TemporalSystem 测试...")
    try:
        ts = TemporalSystem()
        
        # 当前时空
        now = ts.get_current_ganzhi()
        assert now.ganzhi
        assert now.season in ["春", "夏", "秋", "冬", "四季"]
        assert now.wuxing in ["金", "木", "水", "火", "土"]
        
        # 动态优先级
        dp = ts.calculate_priority(5, now, "木")
        assert 0 <= dp.final_priority <= 1
        
        print(f"  ✓ 时序系统: {now.ganzhi}({now.season}季), 优先级调整{dp.final_priority}")
        results.append(("TemporalSystem", True, ""))
    except Exception as e:
        results.append(("TemporalSystem", False, str(e)))
        print(f"  ✗ 失败: {e}")
    
    # ==================== 信念追踪测试 ====================
    print("\n[5/8] BeliefTracker 测试...")
    try:
        tracker = BeliefTracker()
        
        # 初始化
        state = tracker.initialize("test_mem_001")
        assert state.stage == "认知"
        assert state.confidence == 0.5
        
        # 强化
        for _ in range(3):
            tracker.reinforce("test_mem_001")
        
        state_after = tracker.get_state("test_mem_001")
        assert state_after.confidence > 0.5
        
        # 动摇
        tracker.shake("test_mem_001")
        state_shaken = tracker.get_state("test_mem_001")
        assert state_shaken.confidence < state_after.confidence
        
        # 阶段分布
        dist = tracker.get_stage_distribution()
        assert "强化" in dist
        
        print(f"  ✓ 信念追踪: 认知→{state_after.stage}, 置信度{state_after.confidence}")
        results.append(("BeliefTracker", True, ""))
    except Exception as e:
        results.append(("BeliefTracker", False, str(e)))
        print(f"  ✗ 失败: {e}")
    
    # ==================== 元认知测试 ====================
    print("\n[6/8] MetaCognition 测试...")
    try:
        mc = MetaCognition()
        
        # 认知空洞发现
        memory_types = {"fact": 5, "preference": 1, "event": 3}
        user_domains = ["健康", "医疗"]
        memory_list = [
            {"id": "1", "type": "fact", "timestamp": time.time() - 86400 * 100},
            {"id": "2", "type": "fact", "timestamp": time.time() - 86400 * 5},
        ]
        
        gaps = mc.discover_gaps(memory_types, user_domains, memory_list)
        assert isinstance(gaps, list)
        
        # 知识老化预警
        aging = mc.get_aging_warnings(memory_list)
        assert isinstance(aging, list)
        
        print(f"  ✓ 元认知: 发现{len(gaps)}个空洞, {len(aging)}个老化预警")
        results.append(("MetaCognition", True, ""))
    except Exception as e:
        results.append(("MetaCognition", False, str(e)))
        print(f"  ✗ 失败: {e}")
    
    # ==================== 多视图检索测试 ====================
    print("\n[7/8] MultiViewRetriever 测试...")
    try:
        from su_core import MultiViewRetriever, SemanticEncoder, EncoderCore
        
        encoder = SemanticEncoder()
        retriever = MultiViewRetriever()
        
        query_hexagram = encoder.encode("用户有高血压", "fact")
        
        candidates = [
            {"id": "1", "content": "高血压患者", "hexagram_index": query_hexagram.index, "vector_score": 0.8},
            {"id": "2", "content": "糖尿病史", "hexagram_index": 10, "vector_score": 0.6},
            {"id": "3", "content": "用户喜欢苹果", "hexagram_index": 20, "vector_score": 0.5},
        ]
        
        results_ret = retriever.retrieve("用户健康问题", query_hexagram, candidates, top_k=3)
        assert len(results_ret) <= 3
        assert "holographic_score" in results_ret[0]
        
        print(f"  ✓ 多视图检索: 返回{len(results_ret)}条, top相关度{results_ret[0].get('holographic_score',0):.3f}")
        results.append(("MultiViewRetriever", True, ""))
    except Exception as e:
        results.append(("MultiViewRetriever", False, str(e)))
        print(f"  ✗ 失败: {e}")
    
    # ==================== 端到端流程测试 ====================
    print("\n[8/8] 端到端记忆流程测试...")
    try:
        # 1. 编码
        encoder = SemanticEncoder()
        memory_type = "fact"
        info = encoder.encode("用户有高血压病史10年", memory_type)
        
        # 2. 压缩
        compressor = SuCompressor()
        compressed = compressor.compress("用户有高血压病史10年", mode="semantic")
        
        # 3. 时序优先级
        ts = TemporalSystem()
        temporal_info = ts.get_current_ganzhi()
        dp = ts.calculate_priority(7, temporal_info, info.wuxing)
        
        # 4. 信念初始化
        tracker = BeliefTracker()
        belief_state = tracker.initialize("e2e_test_001")
        
        print(f"  ✓ 端到端: 编码→{info.name}, 压缩→{compressed['ratio']}x, 优先级→{dp.final_priority}")
        results.append(("端到端流程", True, ""))
    except Exception as e:
        results.append(("端到端流程", False, str(e)))
        print(f"  ✗ 失败: {e}")
    
    # ==================== 测试结果汇总 ====================
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    
    for name, ok, err in results:
        status = "✅ PASS" if ok else f"❌ FAIL: {err}"
        print(f"  {name:20s} {status}")
    
    print(f"\n通过: {passed}/{len(results)}")
    
    if failed == 0:
        print("\n🎉 Phase 1 全部测试通过！")
    else:
        print(f"\n⚠️  {failed}项测试失败")
    
    return failed == 0


if __name__ == "__main__":
    success = test_all_modules()
    sys.exit(0 if success else 1)
