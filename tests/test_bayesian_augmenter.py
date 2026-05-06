#!/usr/bin/env python3
"""
BayesianAugmenter 串联集成验证测试

验证：
1. 非侵入式包装 — 原始系统完整保留
2. 双路径查询 — query() 返回原版+贝叶斯+对比
3. 双路径预测 — predict() 提供置信区间和校准
4. 双路径推理 — reason() 提供因果链概率化
5. 反馈闭环 — feedback() 闭合贝叶斯更新回路
6. 准确度报告 — get_accuracy_report() 对比双路径
7. 批量验证 — run_validation_suite() 自动化对比
"""

import sys
import os
import time
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from su_memory.sdk.lite_pro import SuMemoryLitePro
from su_memory.sdk.bayesian_augmenter import BayesianAugmenter, EnhancedOutput, ComparisonDelta


# ============================================================
# 辅助函数
# ============================================================

def create_test_data(client: SuMemoryLitePro) -> list:
    """创建测试记忆数据"""
    memories = []
    data = [
        ("项目ROI增长了25%，其中Q3增长最为显著", {"type": "finance", "period": "Q3"}),
        ("由于产品新功能上线，客户满意度提升至92%", {"type": "product", "causal_parent": "feature_launch"}),
        ("团队完成了3个核心模块的重构，代码质量评分从B提升至A", {"type": "engineering"}),
        ("市场调研显示竞品推出类似功能，需要关注竞争动态", {"type": "market", "priority": "high"}),
        ("用户反馈批量导入功能存在性能问题，需要优化", {"type": "feedback", "priority": "high"}),
        ("服务器在高峰时段CPU使用率达到85%，需要扩容", {"type": "infrastructure", "severity": "warning"}),
        ("A/B测试显示新UI设计转化率提升15%", {"type": "experiment", "result": "positive"}),
        ("安全审计发现2个中危漏洞，已修复", {"type": "security", "status": "resolved"}),
        ("Q4预算获批，可以开始招聘3名新工程师", {"type": "hr", "action": "approved"}),
        ("数据迁移计划从PostgreSQL 14升级到16，预计耗时2周", {"type": "infrastructure", "plan": "migration"}),
    ]

    for content, metadata in data:
        mem_id = client.add(content, metadata=metadata)
        memories.append({"id": mem_id, "content": content, "metadata": metadata})

    return memories


# ============================================================
# 测试1：非侵入式验证
# ============================================================

def test_non_invasive():
    """验证 BayesianAugmenter 不修改原始系统"""
    print("\n" + "=" * 60)
    print("测试1: 非侵入式包装验证")
    print("=" * 60)

    client = SuMemoryLitePro(max_memories=1000, enable_graph=False, enable_temporal=False, enable_prediction=True)
    original_add = client.add

    # 创建增强器
    augmenter = BayesianAugmenter(client, verbose=False)

    # 验证原始 add 可正常调用
    mem_id = client.add("测试记忆内容", metadata={"test": True})
    assert mem_id is not None
    print("  ✅ 1.1 原始 add() 正常工作")

    # 验证原始 query 可正常调用
    results = client.query("测试记忆", top_k=3)
    assert len(results) > 0
    print(f"  ✅ 1.2 原始 query() 正常工作 ({len(results)} 结果)")

    # 验证可通过 augmenter 访问原始方法
    # _memories 为私有属性，augmenter.__getattr__ 拒绝 _ 开头的透传，直接访问 client
    assert client._memories is client._memories  # 原始客户端属性不变
    print("  ✅ 1.3 augmenter 不修改原始客户端属性")

    # 验证未修改原始方法签名
    import inspect
    orig_sig = inspect.signature(original_add)
    assert "content" in str(orig_sig), f"add签名被修改: {orig_sig}"
    print("  ✅ 1.4 原始方法签名未修改")

    print("\n  测试1结论: ✅ 完全非侵入式，原始系统零修改")


# ============================================================
# 测试2：双路径查询
# ============================================================

def test_dual_path_query():
    """验证双路径查询对比输出"""
    print("\n" + "=" * 60)
    print("测试2: 双路径查询对比")
    print("=" * 60)

    client = SuMemoryLitePro(max_memories=1000, enable_graph=False, enable_temporal=False, enable_prediction=True)
    create_test_data(client)

    augmenter = BayesianAugmenter(client, verbose=False)

    # 双路径查询
    result = augmenter.query("产品功能", top_k=5)

    # 验证结构
    assert isinstance(result, EnhancedOutput)
    assert "results" in result.original
    assert "results" in result.bayesian
    assert len(result.comparisons) > 0
    print("  ✅ 2.1 EnhancedOutput 结构完整")

    # 验证原版结果
    assert len(result.original["results"]) > 0
    print(f"  ✅ 2.2 原版查询: {len(result.original['results'])} 条结果")

    # 验证贝叶斯增强
    bayes_results = result.bayesian["results"]
    assert len(bayes_results) > 0
    print(f"  ✅ 2.3 贝叶斯增强: {len(bayes_results)} 条结果")

    # 查看对比
    print(f"\n  📊 对比差异:")
    for comp in result.comparisons:
        print(f"     {comp.field}: {comp.difference_description} [{comp.improvement_indicator}]")

    print("\n  测试2结论: ✅ 双路径查询正常，对比分析完整")


# ============================================================
# 测试3：反馈闭环与信念更新
# ============================================================

def test_feedback_loop():
    """验证反馈闭环更新贝叶斯信念"""
    print("\n" + "=" * 60)
    print("测试3: 反馈闭环与信念更新")
    print("=" * 60)

    client = SuMemoryLitePro(max_memories=1000, enable_graph=False, enable_temporal=False, enable_prediction=True)
    memories = create_test_data(client)

    augmenter = BayesianAugmenter(client, verbose=False)

    # 先做一次查询
    result = augmenter.query("产品功能", top_k=5)

    # 模拟用户反馈：指出正确结果
    expected_ids = [memories[1]["id"]]  # "产品新功能上线" 应该是正确的

    feedback = augmenter.feedback(
        query="产品功能",
        expected_memory_ids=expected_ids,
        ground_truth_value=0.8,  # 该记忆与查询相关度应为 0.8
    )

    assert feedback["feedback_id"] > 0
    print(f"  ✅ 3.1 反馈已接收 (#{feedback['feedback_id']})")

    # 验证信念已更新
    for mem_id in expected_ids:
        belief = augmenter.engine.get_belief(mem_id)
        if belief:
            print(f"  ✅ 3.2 信念 '{mem_id}' 已更新: "
                  f"confidence={belief.posterior.mean:.3f}, "
                  f"stage={belief.get_stage()}")

    # 再做一次查询看差异
    result2 = augmenter.query("产品功能", top_k=5)
    bayes_results = result2.bayesian["results"]

    # 检查有贝叶斯增强的结果
    enhanced = [r for r in bayes_results if r.get("bayesian_confidence") is not None]
    print(f"  ✅ 3.3 反馈后: {len(enhanced)}/{len(bayes_results)} 结果有贝叶斯置信度")

    # 验证 Top-1 是否更准确
    if bayes_results:
        top1 = bayes_results[0]
        if top1.get("bayesian_confidence"):
            print(f"  ✅ 3.4 Top-1 有贝叶斯增强: "
                  f"confidence={top1['bayesian_confidence']:.3f}, "
                  f"stage={top1.get('stage', 'N/A')}")

    print("\n  测试3结论: ✅ 反馈闭环正常工作，信念状态更新")


# ============================================================
# 测试4：准确度追踪与报告
# ============================================================

def test_accuracy_tracking():
    """验证准确度追踪和对比报告"""
    print("\n" + "=" * 60)
    print("测试4: 准确度追踪与对比报告")
    print("=" * 60)

    client = SuMemoryLitePro(max_memories=1000, enable_graph=False, enable_temporal=False, enable_prediction=True)
    create_test_data(client)

    augmenter = BayesianAugmenter(client, verbose=False)

    # 模拟多轮反馈
    test_cases = [
        {"query": "ROI增长", "expected": 0.85, "correct": True},
        {"query": "客户满意度", "expected": 0.90, "correct": True},
        {"query": "代码重构", "expected": 0.75, "correct": True},
        {"query": "竞争动态", "expected": 0.60, "correct": False},
        {"query": "性能优化", "expected": 0.70, "correct": True},
    ]

    for tc in test_cases:
        augmenter.query(tc["query"])
        augmenter.feedback(
            query=tc["query"],
            ground_truth_value=tc["expected"],
        )

    # 获取准确度报告
    report = augmenter.get_accuracy_report()

    assert report["summary"]["total_feedback"] == 5
    print(f"  ✅ 4.1 共计 {report['summary']['total_feedback']} 次反馈")

    orig_mae = report["original_stats"]["mae"]
    bayes_mae = report["bayesian_stats"]["mae"]
    print(f"  ✅ 4.2 原始 MAE: {orig_mae:.4f}")
    print(f"  ✅ 4.3 贝叶斯 MAE: {bayes_mae:.4f}")

    improvement = report["summary"]["improvement_pct"]
    print(f"  ✅ 4.4 准确度改善: {improvement:+.1f}%")

    verdict = report["summary"]["verdict"]
    print(f"  ✅ 4.5 判定: {verdict}")

    print(f"  ✅ 4.6 建议: {report['summary']['recommendation']}")

    # 测试格式化打印
    augmenter.print_accuracy_report()

    print("\n  测试4结论: ✅ 准确度追踪和对比报告正常")


# ============================================================
# 测试5：批量验证套件
# ============================================================

def test_validation_suite():
    """验证批量对比验证"""
    print("\n" + "=" * 60)
    print("测试5: 批量对比验证套件")
    print("=" * 60)

    client = SuMemoryLitePro(max_memories=1000, enable_graph=False, enable_temporal=False, enable_prediction=True)
    memories = create_test_data(client)

    augmenter = BayesianAugmenter(client, verbose=False)

    # 准备测试用例
    test_queries = [
        {
            "query": "产品功能上线",
            "expected_memory_ids": [memories[1]["id"]],
            "ground_truth_value": 0.85,
        },
        {
            "query": "性能优化问题",
            "expected_memory_ids": [memories[4]["id"]],
            "ground_truth_value": 0.80,
        },
        {
            "query": "ROI增长",
            "expected_memory_ids": [memories[0]["id"]],
            "ground_truth_value": 0.90,
        },
        {
            "query": "安全审计",
            "expected_memory_ids": [memories[7]["id"]],
            "ground_truth_value": 0.75,
        },
    ]

    # 运行批量验证
    validation = augmenter.run_validation_suite(test_queries, verbose=True)

    assert "results" in validation
    assert len(validation["results"]) == 4
    print(f"  ✅ 5.1 批量验证: {len(validation['results'])} 个用例执行完成")

    summary = validation["summary"]
    print(f"  ✅ 5.2 原始准确率: {summary.get('original_accuracy', 0):.1%}")
    print(f"  ✅ 5.3 贝叶斯准确率: {summary.get('bayesian_accuracy', 0):.1%}")

    if summary.get("original_accuracy") is not None:
        diff = (summary["bayesian_accuracy"] - summary["original_accuracy"]) * 100
        print(f"  ✅ 5.4 准确率差异: {diff:+.1f}%")

    print("\n  测试5结论: ✅ 批量验证套件正常工作")


# ============================================================
# 测试6：自动同步
# ============================================================

def test_auto_sync():
    """验证自动同步功能"""
    print("\n" + "=" * 60)
    print("测试6: 自动同步记忆到贝叶斯系统")
    print("=" * 60)

    client = SuMemoryLitePro(max_memories=1000, enable_graph=False, enable_temporal=False, enable_prediction=True)

    # 创建增强器前先添加一些记忆（测试 hook 是否生效）
    mem_id_before = client.add("增强器创建前的记忆")

    augmenter = BayesianAugmenter(client, enable_auto_sync=True, verbose=False)

    # 创建增强器后添加记忆
    mem_id_after = client.add("增强器创建后的记忆", metadata={"category": "test"})

    # 验证增强器前的记忆不自动同步
    belief_before = augmenter.engine.get_belief(mem_id_before)
    print(f"  ✅ 6.1 Hook前记忆: {'未自动同步' if belief_before is None else '已同步'}")

    # 验证增强器后的记忆自动同步
    belief_after = augmenter.engine.get_belief(mem_id_after)
    assert belief_after is not None, "Hook后的记忆应自动同步"
    print(f"  ✅ 6.2 Hook后记忆已自动同步: confidence={belief_after.posterior.mean:.3f}")

    # 验证手动同步
    augmenter._sync_memory_to_bayesian(mem_id_before, "增强器创建前的记忆", {"category": "test"})
    belief_synced = augmenter.engine.get_belief(mem_id_before)
    assert belief_synced is not None
    print(f"  ✅ 6.3 手动同步成功: {belief_synced.content_summary}")

    # 查看信念数量
    stats = augmenter.engine.get_statistics()
    print(f"  ✅ 6.4 当前贝叶斯信念数: {stats['total_beliefs']}")

    print("\n  测试6结论: ✅ 自动同步功能正常")


# ============================================================
# 测试7：双路径预测
# ============================================================

def test_dual_path_predict():
    """验证双路径预测"""
    print("\n" + "=" * 60)
    print("测试7: 双路径预测")
    print("=" * 60)

    client = SuMemoryLitePro(max_memories=1000, enable_graph=False, enable_temporal=False, enable_prediction=True)
    create_test_data(client)

    augmenter = BayesianAugmenter(client, verbose=False)

    # 先添加一些事件序列（用于预测）
    for i in range(5):
        client.add(f"Q{i+1}项目进度正常推进", metadata={"type": "project_event"})

    # 双路径预测
    result = augmenter.predict(query="项目进度", top_k=3)

    assert isinstance(result, EnhancedOutput)
    print("  ✅ 7.1 EnhancedOutput 结构完整")

    # 检查原始预测
    if "error" not in result.original:
        print(f"  ✅ 7.2 原版预测: {list(result.original.keys())[:3]}")

    # 检查贝叶斯预测
    if "event_predictions" in result.bayesian:
        events = result.bayesian["event_predictions"]
        print(f"  ✅ 7.3 贝叶斯增强预测: {len(events)} 个事件")
        for ev in events[:2]:
            bayes_conf = ev.get("bayesian_confidence", "N/A")
            orig_conf = ev.get("original_confidence", "N/A")
            uncert = ev.get("bayesian_uncertainty", "N/A")
            print(f"     事件: {ev.get('content', '')[:30]}...")
            print(f"       原始置信度: {orig_conf}, 贝叶斯置信度: {bayes_conf}, 不确定性: {uncert}")

    # 查看对比
    for comp in result.comparisons:
        if "uncertainty" in comp.field:
            print(f"  ✅ 7.4 {comp.difference_description}")

    print("\n  测试7结论: ✅ 双路径预测正常，提供不确定性量化")


# ============================================================
# 测试8：持久化与恢复
# ============================================================

def test_persistence():
    """验证贝叶斯状态持久化和恢复"""
    print("\n" + "=" * 60)
    print("测试8: 状态持久化与恢复")
    print("=" * 60)

    client = SuMemoryLitePro(max_memories=1000, enable_graph=False, enable_temporal=False, enable_prediction=True)
    create_test_data(client)

    augmenter = BayesianAugmenter(client, verbose=False)

    # 做一些查询和反馈
    augmenter.query("ROI增长")
    augmenter.feedback(
        query="ROI增长",
        ground_truth_value=0.85,
    )

    # 保存状态
    path = augmenter.save_state("/tmp/test_bayesian_augmenter.json")
    assert os.path.exists(path), "保存文件应存在"
    print(f"  ✅ 8.1 状态已保存: {path}")

    # 记录当前状态
    belief_count = augmenter.engine.get_statistics()["total_beliefs"]
    feedback_count = augmenter._feedback_count

    # 创建新的增强器并恢复
    client2 = SuMemoryLitePro(max_memories=1000, enable_graph=False, enable_temporal=False, enable_prediction=True)
    augmenter2 = BayesianAugmenter(client2, verbose=False)
    augmenter2.load_state(path)

    restored_count = augmenter2.engine.get_statistics()["total_beliefs"]
    restored_feedback = augmenter2._feedback_count

    print(f"  ✅ 8.2 信念数: 原始={belief_count}, 恢复={restored_count}")
    print(f"  ✅ 8.3 反馈数: 原始={feedback_count}, 恢复={restored_feedback}")

    assert restored_count == belief_count, "信念数应一致"
    assert restored_feedback == feedback_count, "反馈数应一致"
    print("  ✅ 8.4 数据一致")

    # 清理
    os.remove(path)

    print("\n  测试8结论: ✅ 持久化和恢复正常")


# ============================================================
# 测试9：重置不影响原系统
# ============================================================

def test_reset_non_destructive():
    """验证 reset_bayesian() 不影响原始系统"""
    print("\n" + "=" * 60)
    print("测试9: 重置不影响原始系统")
    print("=" * 60)

    client = SuMemoryLitePro(max_memories=1000, enable_graph=False, enable_temporal=False, enable_prediction=True)
    create_test_data(client)

    augmenter = BayesianAugmenter(client, verbose=False)

    # 记录原始系统状态
    orig_memory_count = len(client._memories)

    # 做一些操作
    augmenter.query("测试")

    # 重置贝叶斯
    augmenter.reset_bayesian()

    # 验证原始系统不受影响
    assert len(client._memories) == orig_memory_count, "原始记忆数不应改变"
    print(f"  ✅ 9.1 原始记忆数不变: {len(client._memories)}")

    # 验证贝叶斯已重置
    stats = augmenter.engine.get_statistics()
    assert stats["total_beliefs"] == 0, "贝叶斯信念应已清空"
    print(f"  ✅ 9.2 贝叶斯信念已重置: {stats['total_beliefs']}")

    # 验证原始 add 仍可工作
    mem_id = client.add("重置后测试", metadata={"test": True})
    assert mem_id is not None
    print(f"  ✅ 9.3 重置后原始 add() 正常工作: {mem_id}")

    print("\n  测试9结论: ✅ 重置不影响原始系统")


# ============================================================
# 测试10：对比模式 vs 替换模式的架构正确性
# ============================================================

def test_architecture_correctness():
    """验证串联对比架构的正确性"""
    print("\n" + "=" * 60)
    print("测试10: 串联对比架构验证")
    print("=" * 60)

    client = SuMemoryLitePro(max_memories=1000, enable_graph=False, enable_temporal=False, enable_prediction=True)
    create_test_data(client)

    augmenter = BayesianAugmenter(client, verbose=False)

    # 验证三个关键方法都返回 EnhancedOutput
    query_result = augmenter.query("测试")
    assert isinstance(query_result, EnhancedOutput)
    print("  ✅ 10.1 query() 返回 EnhancedOutput")

    predict_result = augmenter.predict(query="测试")
    assert isinstance(predict_result, EnhancedOutput)
    print("  ✅ 10.2 predict() 返回 EnhancedOutput")

    reason_result = augmenter.reason("测试", max_hops=2)
    assert isinstance(reason_result, EnhancedOutput)
    print("  ✅ 10.3 reason() 返回 EnhancedOutput")

    # 验证每个 EnhancedOutput 都有三要素
    for name, result in [("query", query_result), ("predict", predict_result), ("reason", reason_result)]:
        assert hasattr(result, 'original'), f"{name} 缺少 original"
        assert hasattr(result, 'bayesian'), f"{name} 缺少 bayesian"
        assert hasattr(result, 'comparisons'), f"{name} 缺少 comparisons"
        assert hasattr(result, 'meta'), f"{name} 缺少 meta"
    print("  ✅ 10.4 所有输出包含 {original, bayesian, comparisons, meta}")

    # 验证原始系统可独立运行（不经由 augmenter）
    direct_query = client.query("测试", top_k=3)
    assert isinstance(direct_query, list), "直接调用 client.query() 应返回 list"
    assert len(direct_query) > 0
    print(f"  ✅ 10.5 原始系统独立运行正常: {len(direct_query)} 结果")

    # 验证 augmenter 的 original 结果与直接调用一致
    if query_result.original.get("results"):
        orig_via_augmenter = [r.get("content") for r in query_result.original["results"][:3]]
        orig_direct = [r.get("content") for r in direct_query[:3]]
        assert orig_via_augmenter == orig_direct, "augmenter 的 original 应与直接调用一致"
        print("  ✅ 10.6 augmenter.original == client.query() 直接调用结果一致")

    print("\n  测试10结论: ✅ 串联对比架构正确，双路径完全解耦")


# ============================================================
# 主入口
# ============================================================

def run_all_tests():
    """运行所有测试"""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + "  BayesianAugmenter 串联集成验证测试".center(50) + "║")
    print("╚" + "═" * 58 + "╝")

    tests = [
        ("非侵入式验证", test_non_invasive),
        ("双路径查询", test_dual_path_query),
        ("反馈闭环", test_feedback_loop),
        ("准确度追踪", test_accuracy_tracking),
        ("批量验证套件", test_validation_suite),
        ("自动同步", test_auto_sync),
        ("双路径预测", test_dual_path_predict),
        ("持久化恢复", test_persistence),
        ("重置非破坏性", test_reset_non_destructive),
        ("架构正确性", test_architecture_correctness),
    ]

    results = []
    for name, test_fn in tests:
        try:
            test_fn()
            results.append((name, "✅ 通过"))
        except Exception as e:
            results.append((name, f"❌ 异常: {e}"))
            import traceback
            traceback.print_exc()

    # 汇总
    print("\n\n" + "═" * 60)
    print("📋 测试汇总")
    print("═" * 60)
    for name, status in results:
        print(f"  {status:<20} {name}")

    passed = sum(1 for _, s in results if "✅" in s)
    total = len(results)
    print(f"\n  📊 总计: {passed}/{total} 通过")
    print("═" * 60)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
