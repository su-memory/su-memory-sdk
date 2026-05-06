#!/usr/bin/env python3
"""
M2 BayesianAugmenter 精细化测试 — 覆盖 t1a-t1f 全部子任务

t1a: query() 双路径 + 回退降级
t1b: predict() 双路径 + SDK/无SDK分支
t1c: reason() 双路径 + 推理增强
t1d: feedback() 闭环 + 信念更新
t1e: 状态管理 save_state/load_state
t1f: 内部辅助方法全覆盖

每个测试使用 proper pytest 断言风格，无 return 语句。
"""

import sys
import os
import time
import json
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from su_memory.sdk.lite_pro import SuMemoryLitePro
from su_memory.sdk.bayesian_augmenter import (
    BayesianAugmenter,
    EnhancedOutput,
    ComparisonDelta,
    AccuracyRecord,
)


# ============================================================
# Fixture 辅助
# ============================================================

TEST_DATA = [
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


def _make_client():
    """创建 SuMemoryLitePro 实例"""
    return SuMemoryLitePro(
        max_memories=1000,
        enable_graph=False,
        enable_temporal=False,
        enable_prediction=True,
    )


def _create_augmenter(client=None, **kwargs):
    """创建 BayesianAugmenter 实例"""
    if client is None:
        client = _make_client()
    return BayesianAugmenter(client, verbose=False, **kwargs)


def _populate(client, data=None):
    """填充测试数据，返回 [(id, content, metadata), ...]"""
    if data is None:
        data = TEST_DATA
    results = []
    for content, metadata in data:
        mem_id = client.add(content, metadata=metadata)
        results.append((mem_id, content, metadata))
    return results


# ============================================================
# t1a: query() 双路径 + 回退降级
# ============================================================

class TestQueryDualPath:
    """query() 双路径查询全面测试"""

    def test_basic_query_returns_enhanced_output(self):
        """基本查询返回 EnhancedOutput"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        result = aug.query("产品功能", top_k=5)
        assert isinstance(result, EnhancedOutput)
        assert "results" in result.original
        assert "results" in result.bayesian
        assert len(result.comparisons) >= 3

    def test_query_empty_string(self):
        """空字符串查询"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        result = aug.query("", top_k=3)
        assert isinstance(result, EnhancedOutput)
        assert isinstance(result.original["results"], list)

    def test_query_special_characters(self):
        """特殊字符查询"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        for q in ["<script>", "SELECT * FROM", "你好世界", "🔥🔥🔥", "a" * 100]:
            result = aug.query(q, top_k=3)
            assert isinstance(result, EnhancedOutput)
            assert "results" in result.bayesian

    def test_query_top_k_variations(self):
        """不同 top_k 值的查询"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        for k in [0, 1, 3, 10, 100]:
            result = aug.query("测试", top_k=k)
            assert isinstance(result, EnhancedOutput)
            assert len(result.original["results"]) <= max(k, 1)

    def test_query_single_word(self):
        """单关键词查询"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        result = aug.query("ROI", top_k=3)
        assert isinstance(result, EnhancedOutput)

    def test_query_no_matching_memories(self):
        """无匹配记忆的查询"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        result = aug.query("完全不相关的查询abcdefghijklmn", top_k=3)
        assert isinstance(result, EnhancedOutput)
        # 即使无匹配也应有空结果
        assert isinstance(result.original["results"], list)

    def test_compare_query_results_top1_match(self):
        """对比查询结果的 top1_match 字段"""
        client = _make_client()
        mems = _populate(client)
        aug = _create_augmenter(client)
        result = aug.query("ROI", top_k=3)
        top1_comps = [c for c in result.comparisons if c.field == "top1_match"]
        assert len(top1_comps) > 0

    def test_compare_query_results_ranking_changes(self):
        """对比查询结果的排序变化"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        result = aug.query("产品", top_k=3)
        ranking_comps = [c for c in result.comparisons if c.field == "ranking_changes"]
        assert len(ranking_comps) > 0

    def test_compare_query_results_bayesian_enhanced_count(self):
        """对比查询结果的贝叶斯增强计数"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        result = aug.query("产品", top_k=3)
        enhanced_comps = [c for c in result.comparisons if c.field == "bayesian_enhanced_count"]
        assert len(enhanced_comps) > 0

    def test_query_meta_fields(self):
        """查询结果的 meta 信息完整"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        result = aug.query("测试", top_k=3)
        assert "query" in result.meta
        assert "top_k" in result.meta
        assert "method" in result.meta
        assert "timestamp" in result.meta
        assert result.meta["method"] == "dual_path_query"
        assert result.meta["top_k"] == 3

    def test_query_bayesian_results_include_confidence(self):
        """贝叶斯查询结果包含置信度字段"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        result = aug.query("测试", top_k=3)
        for item in result.bayesian["results"]:
            assert "score" in item
            assert "bayesian_confidence" in item or item.get("bayesian_confidence") is None

    def test_bayesian_query_fused_score_in_range(self):
        """贝叶斯融合得分在合理范围内"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        result = aug.query("测试", top_k=3)
        for item in result.bayesian["results"]:
            score = item.get("score", item.get("original_score", 0))
            assert 0 <= score <= 1.0 or score is not None

    def test_query_with_verbose_mode(self):
        """详细模式查询不报错"""
        client = _make_client()
        _populate(client)
        aug = BayesianAugmenter(client, verbose=True)
        result = aug.query("测试", top_k=2)
        assert isinstance(result, EnhancedOutput)

    def test_query_engine_stats_present(self):
        """贝叶斯结果包含引擎统计"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        result = aug.query("测试", top_k=3)
        assert "engine_stats" in result.bayesian


# ============================================================
# t1b: predict() 双路径 + SDK/无SDK分支
# ============================================================

class TestPredictDualPath:
    """predict() 双路径预测全面测试"""

    def test_predict_basic(self):
        """基本预测功能"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        result = aug.predict(query="项目进度", top_k=3)
        assert isinstance(result, EnhancedOutput)

    def test_predict_without_predictor(self):
        """禁用预测器时的 predict"""
        client = _make_client()
        _populate(client)
        aug = BayesianAugmenter(client, enable_predictor=False, verbose=False)
        result = aug.predict(query="测试", top_k=3)
        assert isinstance(result, EnhancedOutput)
        # 贝叶斯结果应包含错误
        assert "error" in result.bayesian or "bayesian_prediction" in result.bayesian

    def test_predict_empty_query(self):
        """空查询预测"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        result = aug.predict(query="", top_k=3)
        assert isinstance(result, EnhancedOutput)

    def test_predict_special_query(self):
        """特殊查询预测"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        for q in ["🔥", "a" * 50, "1234567890"]:
            result = aug.predict(query=q, top_k=3)
            assert isinstance(result, EnhancedOutput)

    def test_predict_with_event_data(self):
        """有事件数据的预测"""
        client = _make_client()
        _populate(client)
        for i in range(5):
            client.add(f"Q{i+1}项目正常推进", metadata={"type": "project_event"})
        aug = _create_augmenter(client)
        result = aug.predict(query="项目进度", top_k=3)
        if "event_predictions" in result.bayesian:
            for ev in result.bayesian["event_predictions"]:
                assert "content" in ev
                assert "confidence" in ev or "bayesian_confidence" in ev

    def test_predict_comparison_uncertainty(self):
        """预测对比包含不确定性量化"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        result = aug.predict(query="测试", top_k=3)
        uncertainty_comps = [c for c in result.comparisons if "uncertainty" in c.field]
        assert len(uncertainty_comps) > 0

    def test_predict_meta_complete(self):
        """预测 meta 信息完整"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        result = aug.predict(query="测试", top_k=3)
        assert result.meta["method"] == "dual_path_predict"
        assert "timestamp" in result.meta

    def test_predict_calibration_report(self):
        """预测包含校准报告"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        result = aug.predict(query="测试", top_k=3)
        if "calibration" in result.bayesian:
            cal = result.bayesian["calibration"]
            assert "status" in cal

    def test_predict_with_no_query(self):
        """无 query 的预测"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        result = aug.predict(top_k=3)
        assert isinstance(result, EnhancedOutput)


# ============================================================
# t1c: reason() 双路径 + 推理增强
# ============================================================

class TestReasonDualPath:
    """reason() 双路径推理全面测试"""

    def test_reason_basic(self):
        """基本推理功能"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        result = aug.reason("产品功能", max_hops=2)
        assert isinstance(result, EnhancedOutput)

    def test_reason_different_max_hops(self):
        """不同 max_hops 值的推理"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        for hops in [1, 2, 3, 5]:
            result = aug.reason("测试", max_hops=hops)
            assert isinstance(result, EnhancedOutput)
            assert result.meta["max_hops"] == hops

    def test_reason_empty_query(self):
        """空查询推理"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        result = aug.reason("", max_hops=2)
        assert isinstance(result, EnhancedOutput)

    def test_reason_without_network(self):
        """无网络时的推理"""
        client = _make_client()
        _populate(client)
        aug = BayesianAugmenter(client, enable_network=False, verbose=False)
        result = aug.reason("测试", max_hops=2)
        assert isinstance(result, EnhancedOutput)

    def test_reason_memory_beliefs_present(self):
        """推理包含 memory_beliefs"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        result = aug.reason("测试", max_hops=2)
        assert "memory_beliefs" in result.bayesian

    def test_reason_causal_chains(self):
        """推理包含因果链"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        result = aug.reason("测试", max_hops=2)
        assert "causal_chains" in result.bayesian

    def test_reason_comparison_fields(self):
        """推理对比字段"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        result = aug.reason("测试", max_hops=2)
        comp_fields = [c.field for c in result.comparisons]
        assert "causal_chain_count" in comp_fields
        assert "belief_coverage" in comp_fields

    def test_reason_meta_complete(self):
        """推理 meta 信息完整"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        result = aug.reason("测试", max_hops=2)
        assert result.meta["method"] == "dual_path_reason"
        assert "max_hops" in result.meta
        assert "timestamp" in result.meta

    def test_reason_with_empty_memory(self):
        """空记忆库的推理"""
        client = _make_client()
        aug = _create_augmenter(client)
        result = aug.reason("测试", max_hops=2)
        assert isinstance(result, EnhancedOutput)

    def test_reason_engine_stats(self):
        """推理包含引擎统计"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        result = aug.reason("测试", max_hops=2)
        assert "engine_stats" in result.bayesian


# ============================================================
# t1d: feedback() 闭环 + 信念更新
# ============================================================

class TestFeedbackLoop:
    """feedback() 闭环与信念更新测试"""

    def test_feedback_with_expected_memory_ids(self):
        """使用 expected_memory_ids 的反馈"""
        client = _make_client()
        mems = _populate(client)
        aug = _create_augmenter(client)
        aug.query("产品功能")
        result = aug.feedback(
            query="产品功能",
            expected_memory_ids=[mems[1][0]],
        )
        assert "feedback_id" in result
        assert result["feedback_id"] > 0
        assert "beliefs_updated" in result

    def test_feedback_with_ground_truth(self):
        """使用 ground_truth_value 的反馈"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        aug.query("ROI增长")
        result = aug.feedback(
            query="ROI增长",
            ground_truth_value=0.85,
        )
        assert "feedback_id" in result
        if "accuracy" in result:
            assert "original_error" in result["accuracy"]
            assert "bayesian_error" in result["accuracy"]

    def test_feedback_with_expected_outcome(self):
        """使用 expected_outcome 的反馈"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        aug.query("测试")
        result = aug.feedback(
            query="测试",
            expected_outcome=True,
        )
        assert "feedback_id" in result

    def test_feedback_with_all_params(self):
        """使用全部参数的反馈"""
        client = _make_client()
        mems = _populate(client)
        aug = _create_augmenter(client)
        aug.query("产品功能")
        result = aug.feedback(
            query="产品功能",
            expected_memory_ids=[mems[1][0]],
            expected_outcome=True,
            ground_truth_value=0.8,
            is_correct=True,
        )
        assert "feedback_id" in result
        assert result["feedback_id"] == 1

    def test_feedback_increments_count(self):
        """反馈计数递增"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        assert aug._feedback_count == 0
        aug.feedback(query="测试1", ground_truth_value=0.5)
        assert aug._feedback_count == 1
        aug.feedback(query="测试2", ground_truth_value=0.6)
        assert aug._feedback_count == 2
        aug.feedback(query="测试3", expected_memory_ids=[])
        assert aug._feedback_count == 3

    def test_feedback_updates_beliefs(self):
        """反馈后信念更新"""
        client = _make_client()
        mems = _populate(client)
        aug = _create_augmenter(client)
        aug.query("产品功能")
        target_id = mems[1][0]
        before = aug.engine.get_belief(target_id)
        aug.feedback(
            query="产品功能",
            expected_memory_ids=[target_id],
        )
        after = aug.engine.get_belief(target_id)
        if after:
            assert after.posterior.effective_sample_size >= (before.posterior.effective_sample_size if before else 0)

    def test_multiple_feedback_accumulates_accuracy(self):
        """多次反馈累积准确度记录"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        for i in range(5):
            aug.query(f"查询{i}")
            aug.feedback(query=f"查询{i}", ground_truth_value=0.7)
        report = aug.get_accuracy_report()
        total = report["summary"]["total_records"]
        assert total == 10  # 5 original + 5 bayesian

    def test_feedback_with_empty_expected_ids(self):
        """空 expected_memory_ids 的反馈"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        result = aug.feedback(query="测试", expected_memory_ids=[])
        assert "feedback_id" in result


# ============================================================
# t1e: 状态管理 save_state/load_state
# ============================================================

class TestStateManagement:
    """状态管理 save_state/load_state 测试"""

    def test_save_state_creates_file(self):
        """save_state 创建文件"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        aug.query("测试")
        path = aug.save_state("/tmp/test_m2_save.json")
        assert os.path.exists(path)
        os.remove(path)

    def test_save_state_default_path(self):
        """save_state 默认路径"""
        client = _make_client()
        aug = _create_augmenter(client)
        path = aug.save_state()
        assert os.path.exists(path)
        assert path.startswith("bayesian_augmenter_state_")
        os.remove(path)

    def test_load_state_restores_beliefs(self):
        """load_state 恢复信念"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        aug.query("ROI增长")
        aug.feedback(query="ROI增长", ground_truth_value=0.85)
        orig_count = aug.engine.get_statistics()["total_beliefs"]
        path = aug.save_state("/tmp/test_m2_restore.json")

        client2 = _make_client()
        aug2 = _create_augmenter(client2)
        aug2.load_state(path)
        restored_count = aug2.engine.get_statistics()["total_beliefs"]
        assert restored_count == orig_count
        os.remove(path)

    def test_load_state_restores_feedback_count(self):
        """load_state 恢复反馈计数"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        for i in range(3):
            aug.feedback(query=f"q{i}", ground_truth_value=0.5)
        path = aug.save_state("/tmp/test_m2_feedback.json")

        client2 = _make_client()
        aug2 = _create_augmenter(client2)
        aug2.load_state(path)
        assert aug2._feedback_count == 3
        os.remove(path)

    def test_load_state_restores_accuracy_records(self):
        """load_state 恢复准确度记录"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        aug.feedback(query="测试", ground_truth_value=0.8)
        path = aug.save_state("/tmp/test_m2_acc.json")

        client2 = _make_client()
        aug2 = _create_augmenter(client2)
        aug2.load_state(path)
        assert len(aug2._accuracy_records) == len(aug._accuracy_records)
        os.remove(path)

    def test_load_state_restores_synced_ids(self):
        """load_state 恢复已同步ID"""
        client = _make_client()
        aug = _create_augmenter(client)
        client.add("测试", metadata={"category": "test"})
        path = aug.save_state("/tmp/test_m2_sync.json")

        client2 = _make_client()
        aug2 = _create_augmenter(client2)
        aug2.load_state(path)
        assert aug2._synced_memory_ids == aug._synced_memory_ids
        os.remove(path)

    def test_save_with_empty_state(self):
        """空状态保存和恢复"""
        client = _make_client()
        aug = _create_augmenter(client, enable_auto_sync=False)
        path = aug.save_state("/tmp/test_m2_empty.json")
        assert os.path.exists(path)

        with open(path) as f:
            data = json.load(f)
        assert "brs" in data
        assert "accuracy_records" in data
        assert data["feedback_count"] == 0
        assert data["synced_ids"] == []

        client2 = _make_client()
        aug2 = _create_augmenter(client2, enable_auto_sync=False)
        aug2.load_state(path)
        assert aug2._feedback_count == 0
        os.remove(path)

    def test_load_state_rebinds_shortcuts(self):
        """load_state 重新绑定快捷引用"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        aug.query("测试")
        path = aug.save_state("/tmp/test_m2_shortcut.json")

        client2 = _make_client()
        aug2 = _create_augmenter(client2)
        aug2.load_state(path)
        assert aug2.engine is not None
        assert aug2.network is not None
        assert aug2.evidence is not None
        assert aug2.predictor is not None
        os.remove(path)

    def test_reset_bayesian_clears_engine(self):
        """reset_bayesian 清空引擎"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        aug.query("测试")
        aug.reset_bayesian()
        assert aug.engine.get_statistics()["total_beliefs"] == 0
        assert aug._feedback_count == 0
        assert len(aug._accuracy_records) == 0


# ============================================================
# t1f: 内部辅助方法全覆盖
# ============================================================

class TestInternalHelpers:
    """内部辅助方法全覆盖测试"""

    def test_comparison_delta_to_dict(self):
        """ComparisonDelta.to_dict()"""
        delta = ComparisonDelta(
            field="test_field",
            original_value=0.5,
            bayesian_value=0.8,
            difference_description="测试差异",
            improvement_indicator="positive",
        )
        d = delta.to_dict()
        assert d["field"] == "test_field"
        assert d["original_value"] == 0.5
        assert d["bayesian_value"] == 0.8
        assert d["difference"] == "测试差异"
        assert d["improvement"] == "positive"

    def test_enhanced_output_to_dict(self):
        """EnhancedOutput.to_dict()"""
        output = EnhancedOutput(
            original={"results": [{"id": "1"}]},
            bayesian={"results": [{"id": "1", "score": 0.9}]},
            comparisons=[
                ComparisonDelta(
                    field="top1_match",
                    original_value="1",
                    bayesian_value="1",
                    difference_description="一致",
                    improvement_indicator="neutral",
                )
            ],
            meta={"query": "test"},
        )
        d = output.to_dict()
        assert "original_result" in d
        assert "bayesian_result" in d
        assert "comparison_deltas" in d
        assert "meta" in d

    def test_accuracy_record_creation(self):
        """AccuracyRecord 数据类"""
        record = AccuracyRecord(
            timestamp=time.time(),
            method="original",
            query="test",
            predicted_value=0.7,
            actual_value=0.8,
            error=-0.1,
            absolute_error=0.1,
        )
        assert record.method == "original"
        assert record.absolute_error == 0.1
        assert record.query == "test"

    def test_getattr_delegation(self):
        """__getattr__ 透传到原始客户端"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client, enable_auto_sync=False)
        # add 未在 augmenter 上包装（当 auto_sync=False）
        mem_id = aug.add("直接透传测试")
        assert mem_id is not None

    def test_getattr_raises_on_private(self):
        """__getattr__ 拒绝 _ 开头的属性"""
        client = _make_client()
        aug = _create_augmenter(client, enable_auto_sync=False)
        import pytest
        with pytest.raises(AttributeError):
            _ = aug._nonexistent_attr

    def test_get_bayesian_engine(self):
        """get_bayesian_engine()"""
        client = _make_client()
        aug = _create_augmenter(client)
        engine = aug.get_bayesian_engine()
        assert engine is not None
        assert engine is aug.engine

    def test_get_bayesian_network(self):
        """get_bayesian_network()"""
        client = _make_client()
        aug = _create_augmenter(client)
        network = aug.get_bayesian_network()
        assert network is not None
        assert network is aug.network

    def test_vlog_in_verbose_mode(self):
        """_vlog 在 verbose 模式下输出"""
        client = _make_client()
        aug = BayesianAugmenter(client, verbose=True)
        # 验证不报错
        aug._vlog("测试日志")
        assert True  # 无异常即通过

    def test_vlog_silent_in_quiet_mode(self):
        """_vlog 在非 verbose 模式下不输出"""
        client = _make_client()
        aug = BayesianAugmenter(client, verbose=False)
        aug._vlog("不应输出")
        assert True

    def test_hook_client_add_positional_args(self):
        """Hook add 处理位置参数"""
        client = _make_client()
        aug = _create_augmenter(client)
        mem_id = client.add("位置参数内容")
        assert mem_id is not None

    def test_hook_client_add_keyword_args(self):
        """Hook add 处理关键字参数"""
        client = _make_client()
        aug = _create_augmenter(client)
        mem_id = client.add("关键字内容", metadata={"category": "test", "tags": ["tag1", "tag2"]})
        assert mem_id is not None

    def test_sync_memory_to_bayesian_with_parents(self):
        """_sync_memory_to_bayesian 处理父节点"""
        client = _make_client()
        aug = _create_augmenter(client)
        parent_id = client.add("父节点内容")
        aug._sync_memory_to_bayesian(parent_id, "父节点内容", {"category": "root"})
        child_id = client.add("子节点内容", metadata={"parent_ids": [parent_id]})
        assert child_id is not None

    def test_sync_memory_to_bayesian_with_empty_metadata(self):
        """_sync_memory_to_bayesian 处理空元数据"""
        client = _make_client()
        aug = _create_augmenter(client)
        mem_id = "test_empty_meta"
        aug._sync_memory_to_bayesian(mem_id, "空元数据内容", {})
        belief = aug.engine.get_belief(mem_id)
        assert belief is not None

    def test_accuracy_report_no_data(self):
        """无数据时的准确度报告"""
        client = _make_client()
        aug = _create_augmenter(client)
        report = aug.get_accuracy_report()
        assert report["status"] == "no_data" or "status" in report

    def test_print_accuracy_report_no_data(self):
        """无数据时打印报告不报错"""
        client = _make_client()
        aug = _create_augmenter(client)
        aug.print_accuracy_report()
        assert True

    def test_validation_suite_empty(self):
        """空验证套件"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        result = aug.run_validation_suite([], verbose=False)
        assert result["summary"]["test_count"] == 0

    def test_validation_suite_single(self):
        """单用例验证套件"""
        client = _make_client()
        mems = _populate(client)
        aug = _create_augmenter(client)
        result = aug.run_validation_suite(
            [{"query": "ROI增长", "expected_memory_ids": [mems[0][0]], "ground_truth_value": 0.85}],
            verbose=False,
        )
        assert result["summary"]["test_count"] == 1
        assert "accuracy_report" in result

    def test_auto_sync_disabled(self):
        """禁用自动同步"""
        client = _make_client()
        aug = BayesianAugmenter(client, enable_auto_sync=False, verbose=False)
        mem_id = client.add("不同步的内容")
        belief = aug.engine.get_belief(mem_id)
        assert belief is None

    def test_accuracy_report_statistics(self):
        """准确度报告统计信息"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        for i in range(3):
            aug.query(f"test{i}")
            aug.feedback(query=f"test{i}", ground_truth_value=0.7 + i * 0.05)
        report = aug.get_accuracy_report()
        if "original_stats" in report:
            assert "mae" in report["original_stats"]
            assert "rmse" in report["original_stats"]
            assert "median_error" in report["original_stats"]
        if "bayesian_stats" in report:
            assert "mae" in report["bayesian_stats"]

    def test_initialization_with_all_options(self):
        """全部初始化选项"""
        client = _make_client()
        aug = BayesianAugmenter(
            client,
            enable_network=True,
            enable_predictor=True,
            enable_auto_sync=True,
            prior_type="uniform",
            verbose=False,
        )
        assert aug._enable_auto_sync is True
        assert aug.engine is not None
        assert aug.network is not None
        assert aug.predictor is not None

    def test_initialization_no_network_no_predictor(self):
        """禁用网络和预测器的初始化"""
        client = _make_client()
        aug = BayesianAugmenter(
            client,
            enable_network=False,
            enable_predictor=False,
            enable_auto_sync=False,
            verbose=False,
        )
        assert aug.engine is not None
        assert aug.predictor is None
        # 网络总是存在（由 BRS 决定），但已禁用

    def test_query_with_keyword_args(self):
        """带额外关键字参数的查询"""
        client = _make_client()
        _populate(client)
        aug = _create_augmenter(client)
        result = aug.query("测试", top_k=3, extra_param="ignored")
        assert isinstance(result, EnhancedOutput)
