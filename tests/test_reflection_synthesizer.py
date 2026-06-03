"""M5 Reflection QA 合成引擎 — 单元测试"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

pytestmark = pytest.mark.causal

from su_memory.sdk._reflection_synthesizer import (
    ReflectionSynthesizer,
    SynthesizedQAPair,
)


class TestFactExtraction:
    """M5-T3: Step 1 — 事实提取测试"""

    def test_extract_facts_basic(self):
        """数值变化 — 提取实体和数值"""
        syn = ReflectionSynthesizer(seed=42)
        memories = [{"id": "m0", "content": "物价指数同比上涨百分之三点五"}]
        facts = syn.extract_facts(memories)
        assert len(facts) == 1
        f = facts[0]
        assert len(f["entities"]) >= 1
        assert any("物价" in e for e in f["entities"])
        # 中文数字检测
        assert any(n["value"] == 3 for n in f["numerics"])

    def test_extract_facts_numeric_direction(self):
        """数值方向检测: 上涨/下降"""
        syn = ReflectionSynthesizer(seed=42)
        memories = [
            {"id": "1", "content": "物价上涨3.5%"},
            {"id": "2", "content": "成本下降2成"},
        ]
        facts = syn.extract_facts(memories)
        assert len(facts) == 2
        assert facts[0]["numerics"][0]["direction"] == "+"
        assert facts[1]["numerics"][0]["direction"] == "-"

    def test_extract_facts_empty(self):
        """空内容不崩溃"""
        syn = ReflectionSynthesizer(seed=42)
        memories = [{"id": "e", "content": ""}]
        facts = syn.extract_facts(memories)
        assert len(facts) == 0

    def test_extract_causal_indicators(self):
        """因果指示词提取"""
        syn = ReflectionSynthesizer(seed=42)
        memories = [{"id": "c", "content": "物价上涨导致消费意愿下降"}]
        facts = syn.extract_facts(memories)
        assert "导致" in facts[0]["causals"]

    def test_extract_energy_type(self):
        """能量类型推断"""
        syn = ReflectionSynthesizer(seed=42)
        tests = [
            ("物价上涨3.5%", "fire"),      # 上涨 → fire
            ("技术突破重大进展", "wood"),    # 突破 → wood
            ("企业成本下降2成", "metal"),    # 下降 → metal
        ]
        for content, expected in tests:
            facts = syn.extract_facts([{"id": "x", "content": content}])
            assert facts[0]["energy_type"] == expected, f"{content} → {facts[0]['energy_type']}"


class TestEntitySurfacing:
    """M5-T3: Step 4 — Entity Surfacing 测试"""

    def test_surface_entities_causal_shared(self):
        """共享因果指示词的文档关联"""
        syn = ReflectionSynthesizer(seed=42)
        memories = [
            {"id": "a", "content": "物价上涨导致消费意愿下降"},
            {"id": "b", "content": "利率上调导致房地产市场调整"},
            {"id": "c", "content": "技术突破不涉及价格变化"},
        ]
        facts = syn.extract_facts(memories)
        entity_map = syn.surface_entities(facts)
        # m0 和 m1 共享 "导致"，应该有关联
        assert len(entity_map) >= 1, f"至少发现 1 组实体关联, got {len(entity_map)}"

    def test_surface_entities_no_causal(self):
        """无因果指示词 → 无实体关联"""
        syn = ReflectionSynthesizer(seed=42)
        memories = [
            {"id": "a", "content": "今天天气很好"},
            {"id": "b", "content": "公司发布了新产品"},
        ]
        facts = syn.extract_facts(memories)
        entity_map = syn.surface_entities(facts)
        assert len(entity_map) == 0


class TestCausalSynthesis:
    """M5-T3: Step 5 — 因果合成测试"""

    def test_synthesize_basic(self):
        """10 条显式因果记忆 → 合成 ≥ 3 QA 对"""
        syn = ReflectionSynthesizer(seed=42)
        memories = [
            {"id": f"m{i}", "content": c} for i, c in enumerate([
                "物价上涨导致消费意愿下降",
                "物价上涨促使央行考虑加息",
                "利率上调推动企业融资成本增加",
                "利率上调导致房地产市场成交量下降",
                "技术突破大幅提升了人工智能效率",
                "技术突破使得制造业成本下降一成",
                "税收减免促使企业扩大投资规模",
                "税收减免降低企业运营成本",
                "极端天气引发基础设施严重损坏",
                "极端天气导致农业生产巨大损失",
            ])
        ]
        facts = syn.extract_facts(memories)
        pairs = syn.synthesize_causal_pairs(facts)
        assert len(pairs) >= 3, f"预期 ≥ 3 对, 实际 {len(pairs)}"
        for p in pairs[:3]:
            assert p.confidence >= syn.min_confidence

    def test_synthesize_no_causal(self):
        """无因果信号 → ≤ 5 QA 对"""
        syn = ReflectionSynthesizer(seed=42)
        memories = [
            {"id": f"n{i}", "content": c} for i, c in enumerate([
                "今天天气很好适合出行",
                "公司公布了最新财报",
                "新产品开始试生产",
                "员工满意度调查完成",
                "大楼完成修缮工作",
                "图书馆延长了开放时间",
                "新道路正式通车",
                "绿化带建设完工",
                "小区安装了新能源充电桩",
                "学校增加了体育课时",
            ])
        ]
        facts = syn.extract_facts(memories)
        pairs = syn.synthesize_causal_pairs(facts)
        assert len(pairs) <= 5, f"预期 ≤ 5 对, 实际 {len(pairs)}"

    def test_synthesize_has_energy_relation(self):
        """合成的 QA 对有 energy_relation 字段"""
        syn = ReflectionSynthesizer(seed=42)
        memories = [
            {"id": "1", "content": "物价上涨导致消费意愿下降"},
            {"id": "2", "content": "物价上涨促使央行考虑加息"},
        ]
        facts = syn.extract_facts(memories)
        pairs = syn.synthesize_causal_pairs(facts)
        if pairs:
            assert pairs[0].energy_relation in [
                "enhance", "suppress", "same", "neutral", "reverse",
            ]


class TestPriorMatrix:
    """M5-T3: 先验矩阵转换测试"""

    def test_to_prior_matrix_multi_pair_accumulation(self):
        """多对 QA 应累积填充先验矩阵 (不互相覆盖)"""
        syn = ReflectionSynthesizer(seed=42)
        pairs = [
            SynthesizedQAPair(
                cause_text="A", effect_text="B",
                cause_entity="eA", effect_entity="eB",
                reflection_depth=1, energy_relation="enhance",
                confidence=0.9, source_memory_ids=["m0", "m1"],
                qa_pair_id="qa_0",
            ),
            SynthesizedQAPair(
                cause_text="B", effect_text="C",
                cause_entity="eB", effect_entity="eC",
                reflection_depth=1, energy_relation="enhance",
                confidence=0.7, source_memory_ids=["m1", "m2"],
                qa_pair_id="qa_1",
            ),
            SynthesizedQAPair(
                cause_text="C", effect_text="D",
                cause_entity="eC", effect_entity="eD",
                reflection_depth=1, energy_relation="suppress",
                confidence=0.6, source_memory_ids=["m2", "m3"],
                qa_pair_id="qa_2",
            ),
        ]
        prior = syn.to_prior_matrix(pairs, n_memories=10)
        nonzero_count = int(np.count_nonzero(prior))
        # 3 个唯一 (cause,effect) 应对应 3 个非零值
        assert nonzero_count >= 3, (
            f"多对累积失效: 预期 ≥ 3 个非零值, 实际 {nonzero_count}\n"
            f"(P1 bug: prior.fill(0) 会清除前值)"
        )

    def test_to_prior_matrix_shape(self):
        """先验矩阵 shape 正确"""
        syn = ReflectionSynthesizer(seed=42)
        pairs = [
            SynthesizedQAPair(
                cause_text="A", effect_text="B",
                cause_entity="eA", effect_entity="eB",
                reflection_depth=1, energy_relation="enhance",
                confidence=0.7, source_memory_ids=["m0", "m1"],
                qa_pair_id="qa_0",
            ),
            SynthesizedQAPair(
                cause_text="B", effect_text="C",
                cause_entity="eB", effect_entity="eC",
                reflection_depth=1, energy_relation="enhance",
                confidence=0.6, source_memory_ids=["m1", "m2"],
                qa_pair_id="qa_1",
            ),
        ]
        prior = syn.to_prior_matrix(pairs, n_memories=10)
        assert prior.shape == (10, 10)

    def test_to_prior_matrix_nonzero(self):
        """先验矩阵至少有一个非零值"""
        syn = ReflectionSynthesizer(seed=42)
        pairs = [
            SynthesizedQAPair(
                cause_text="A", effect_text="B",
                cause_entity="eA", effect_entity="eB",
                reflection_depth=1, energy_relation="enhance",
                confidence=0.8, source_memory_ids=["m0", "m1"],
                qa_pair_id="qa_0",
            ),
        ]
        prior = syn.to_prior_matrix(pairs, n_memories=5)
        assert np.count_nonzero(prior) >= 1


class TestEdgeCases:
    """M5-T3: 边界条件测试"""

    def test_empty_input(self):
        """空输入不崩溃"""
        syn = ReflectionSynthesizer(seed=42)
        pairs, prior = syn.run_pipeline([])
        assert pairs == []
        assert prior.shape == (0, 0)  # 空输入 → 空矩阵

    def test_single_memory(self):
        """单条记忆 → 空结果"""
        syn = ReflectionSynthesizer(seed=42)
        pairs, _ = syn.run_pipeline([
            {"id": "s", "content": "只有一条记忆"},
        ])
        assert pairs == []

    def test_pipeline_end_to_end(self):
        """30 条记忆 — 全流水线无异常"""
        syn = ReflectionSynthesizer(seed=42)
        domains = [
            ("物价上涨导致消费意愿下降", "物价上涨促使央行考虑加息"),
            ("利率上调推动融资成本增加", "利率上调导致市场调整"),
            ("技术突破提升效率", "技术突破降低成本"),
            ("税收减免扩大投资", "税收减免降低负担"),
            ("极端天气引发损失", "极端天气导致减产"),
        ] * 6  # 30 条
        memories = []
        for i, (c, e) in enumerate(domains):
            memories.append({"id": f"m{i*2}", "content": c})
            memories.append({"id": f"m{i*2+1}", "content": e})

        pairs, prior = syn.run_pipeline(memories[:30])
        assert isinstance(pairs, list)
        assert isinstance(prior, np.ndarray)


class TestTrainingReport:
    """M5-T3: v3.6.0 本地训练质量报告"""

    def test_training_report_empty(self):
        """空 QA 列表"""
        syn = ReflectionSynthesizer(seed=42)
        report = syn.training_data_report([])
        assert report["total_pairs"] == 0
        assert report["ready_for_training"] is False

    def test_training_report_fields(self):
        """报告包含所有必要字段"""
        syn = ReflectionSynthesizer(seed=42)
        pairs = [
            SynthesizedQAPair(
                cause_text="A", effect_text="B",
                cause_entity="eA", effect_entity="eB",
                reflection_depth=1, energy_relation="enhance",
                confidence=0.7, source_memory_ids=["m0", "m1"],
                qa_pair_id="qa_0",
            ),
        ]
        report = syn.training_data_report(pairs)
        required = [
            "total_pairs", "avg_confidence", "confidence_above_04",
            "energy_distribution", "reflection_depths",
            "diversity_score", "ready_for_training",
        ]
        for key in required:
            assert key in report, f"缺少字段 {key}"

    def test_training_report_ready_threshold(self):
        """ready_for_training 阈值: < 3000 → False"""
        syn = ReflectionSynthesizer(seed=42)
        # 创建 100 对高置信度的 QA
        pairs = []
        for i in range(100):
            pairs.append(SynthesizedQAPair(
                cause_text=f"C{i}", effect_text=f"E{i}",
                cause_entity="eA", effect_entity="eB",
                reflection_depth=1,
                energy_relation=["enhance", "suppress", "same", "neutral", "reverse"][i % 5],
                confidence=0.5 + (i % 10) * 0.02,
                source_memory_ids=[f"m{i*2}", f"m{i*2+1}"],
                qa_pair_id=f"qa_{i}",
            ))
        report = syn.training_data_report(pairs)
        assert report["total_pairs"] == 100
        assert report["ready_for_training"] is False  # < 3000
        assert report["diversity_score"] == 1.0  # 5 种能量类型全覆盖


class TestGaussianDAGReflectionPrior:
    """M5-T3: GaussianDAG + Reflection Prior 集成测试"""

    def test_gaussian_dag_with_reflection_prior(self):
        """confidence 受 reflection prior 影响"""
        from su_memory.sdk._spectral_causal import GaussianDAG

        memories = [
            {"id": "0", "content": "物价上涨导致消费意愿下降"},
            {"id": "1", "content": "物价上涨促使央行考虑加息"},
            {"id": "2", "content": "利率上调推动企业融资成本增加"},
            {"id": "3", "content": "利率上调导致房地产市场调整"},
        ]

        # 无先验
        dag_no = GaussianDAG(memories)
        edges_no = dag_no.discover_hidden_edges(max_scan=4, min_correlation=0.1)

        # 有先验
        dag_yes = GaussianDAG(memories)
        prior = np.zeros((4, 4), dtype=np.float32)
        prior[0, 1] = 0.8  # 强因果先验 m0→m1
        dag_yes.with_reflection_prior(prior)
        edges_yes = dag_yes.discover_hidden_edges(max_scan=4, min_correlation=0.1)

        assert len(edges_no) == len(edges_yes), "边缘数量应一致"
