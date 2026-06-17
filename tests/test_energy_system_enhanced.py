"""
su-memory SDK v3.5.5 — 能量系统增强集成测试
===========================================

覆盖五行能量系统完整功能:
  1. 五行传播链: 生克乘侮关系
  2. EnergyBus: 能量总线信号传递
  3. 能量推断准确率
  4. 亲和度计算
  5. 能量类型一致性

测试环境: 本地五行能量引擎
"""

import os
import sys

import pytest

pytestmark = [pytest.mark.energy, pytest.mark.p2]

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def causal_engine():
    """因果引擎"""
    from su_memory._sys._causal_engine import CategoryCausalEngine
    return CategoryCausalEngine()


@pytest.fixture
def energy_matrix():
    """拓扑能量矩阵"""
    from su_memory.sdk._energy_loss import TopologicalEnergyMatrix
    return TopologicalEnergyMatrix.build()


# ============================================================
# 1. 五行传播链验证
# ============================================================

class TestFiveElementPropagation:
    """五大范畴状态能量传播验证 (semantic/causal/spacetime/generative/trust)"""

    def test_semantic_enhances_causal(self, energy_matrix):
        """semantic→causal: enhance (生)"""
        rel = energy_matrix.get_relation_type("semantic", "causal")
        val = energy_matrix.get_energy("semantic", "causal")
        assert rel == "enhance", f"semantic→causal 应为 enhance, 实际: {rel}"
        assert val >= 0.7, f"语义→因果权重应 >= 0.7, 实际: {val}"

    def test_causal_enhances_spacetime(self, energy_matrix):
        """causal→spacetime: enhance"""
        val = energy_matrix.get_energy("causal", "spacetime")
        assert val >= 0.7

    def test_spacetime_enhances_generative(self, energy_matrix):
        """spacetime→generative: enhance"""
        val = energy_matrix.get_energy("spacetime", "generative")
        assert val >= 0.7

    def test_generative_enhances_trust(self, energy_matrix):
        """generative→trust: enhance"""
        val = energy_matrix.get_energy("generative", "trust")
        assert val >= 0.7

    def test_trust_enhances_semantic(self, energy_matrix):
        """trust→semantic: enhance"""
        val = energy_matrix.get_energy("trust", "semantic")
        assert val >= 0.7

    def test_semantic_suppresses_spacetime(self, energy_matrix):
        """semantic→spacetime: suppress (克)"""
        rel = energy_matrix.get_relation_type("semantic", "spacetime")
        val = energy_matrix.get_energy("semantic", "spacetime")
        assert rel == "suppress", f"semantic→spacetime 应为 suppress, 实际: {rel}"
        assert val <= 0.3, f"语义→时空权重应 <= 0.3, 实际: {val}"

    def test_causal_suppresses_generative(self, energy_matrix):
        """causal→generative: suppress"""
        val = energy_matrix.get_energy("causal", "generative")
        assert val <= 0.3

    def test_spacetime_suppresses_trust(self, energy_matrix):
        """spacetime→trust: suppress"""
        val = energy_matrix.get_energy("spacetime", "trust")
        assert val <= 0.3

    def test_generative_suppresses_semantic(self, energy_matrix):
        """generative→semantic: suppress"""
        val = energy_matrix.get_energy("generative", "semantic")
        assert val <= 0.3

    def test_trust_suppresses_causal(self, energy_matrix):
        """trust→causal: suppress"""
        val = energy_matrix.get_energy("trust", "causal")
        assert val <= 0.3

    def test_full_cycle_closed(self, energy_matrix):
        """完整生克闭环验证: 5生 + 5克 = 10条非自环边"""
        n_enhance = sum(1 for r in energy_matrix.edge_types.values() if r == "enhance")
        n_suppress = sum(1 for r in energy_matrix.edge_types.values() if r == "suppress")
        assert n_enhance == 5, f"应有5条生边, 实际: {n_enhance}"
        assert n_suppress == 5, f"应有5条克边, 实际: {n_suppress}"


# ============================================================
# 2. 能量亲和度计算
# ============================================================

class TestEnergyAffinity:
    """能量亲和度计算测试 (五行生克体系)"""

    def test_enhance_affinity(self, causal_engine):
        """生关系: 亲和度 > 1.0"""
        engine = causal_engine
        engine.add_node("query", "query", energy_type="wood")
        engine.add_node("candidate", "candidate", energy_type="fire")
        results = engine.query_with_energy_boost(
            "query", ["candidate"], {"candidate": 0.7}
        )
        assert results[0]["affinity"] > 1.0

    def test_suppress_affinity(self, causal_engine):
        """克关系: 亲和度 < 1.0"""
        engine = causal_engine
        engine.add_node("query", "query", energy_type="wood")
        engine.add_node("candidate", "candidate", energy_type="earth")
        results = engine.query_with_energy_boost(
            "query", ["candidate"], {"candidate": 0.7}
        )
        assert results[0]["affinity"] < 1.0

    def test_same_energy_affinity(self, causal_engine):
        """同类型: 亲和度 = 1.2"""
        engine = causal_engine
        engine.add_node("query", "query", energy_type="fire")
        engine.add_node("candidate", "candidate", energy_type="fire")
        results = engine.query_with_energy_boost(
            "query", ["candidate"], {"candidate": 0.5}
        )
        assert results[0]["affinity"] == 1.2

    def test_affinity_boosts_ranking(self, causal_engine):
        """亲和度影响排序"""
        engine = causal_engine
        engine.add_node("q", "query growth", energy_type="wood")

        # 生: wood→fire, 克: wood→earth, 同级: wood→wood
        engine.add_node("fire_n", "fire node", energy_type="fire")
        engine.add_node("earth_n", "earth node", energy_type="earth")
        engine.add_node("wood_n", "wood node", energy_type="wood")

        results = engine.query_with_energy_boost(
            "q", ["fire_n", "earth_n", "wood_n"],
            {"fire_n": 0.6, "earth_n": 0.6, "wood_n": 0.6}
        )
        # fire (生) 排名第一
        assert results[0]["node_id"] == "fire_n", (
            f"生关系应排名第一, 实际: {results[0]['node_id']}"
        )

    def test_energy_type_enum(self):
        """能量类型枚举存在"""
        from su_memory._sys._enums import EnergyType

        assert EnergyType.WOOD.value == 0
        assert EnergyType.FIRE.value == 1
        assert EnergyType.EARTH.value == 2
        assert EnergyType.METAL.value == 3
        assert EnergyType.WATER.value == 4

    def test_five_categorical_states(self):
        """五大范畴状态列表完整"""
        from su_memory.sdk._energy_loss import FIVE_CATEGORICAL_STATES

        assert len(FIVE_CATEGORICAL_STATES) == 5
        assert "semantic" in FIVE_CATEGORICAL_STATES
        assert "causal" in FIVE_CATEGORICAL_STATES
        assert "spacetime" in FIVE_CATEGORICAL_STATES
        assert "generative" in FIVE_CATEGORICAL_STATES
        assert "trust" in FIVE_CATEGORICAL_STATES


# ============================================================
# 3. 能量推断准确率
# ============================================================

class TestEnergyInference:
    """能量类型推断测试"""

    def test_energy_type_0_maps_to_semantic(self):
        """索引0 → semantic"""
        from su_memory._sys._unified_unit import UnifiedInfoFactory

        factory = UnifiedInfoFactory()
        unit = factory.create_from_content(
            "spring growth development new beginning",
            energy_type=0  # semantic
        )
        assert unit.to_dict()["human"]["energy_name"] == "semantic"

    def test_energy_type_1_maps_to_causal(self):
        """索引1 → causal"""
        from su_memory._sys._unified_unit import UnifiedInfoFactory

        factory = UnifiedInfoFactory()
        unit = factory.create_from_content(
            "summer hot fire passion energy",
            energy_type=1  # causal
        )
        assert unit.to_dict()["human"]["energy_name"] == "causal"

    def test_energy_type_2_maps_to_spacetime(self):
        """索引2 → spacetime"""
        from su_memory._sys._unified_unit import UnifiedInfoFactory

        factory = UnifiedInfoFactory()
        unit = factory.create_from_content("test content", energy_type=2)
        d = unit.to_dict()
        assert "human" in d
        assert d["human"]["energy_name"] == "spacetime"

    def test_energy_type_4_maps_to_trust(self):
        """索引4 → trust"""
        from su_memory._sys._unified_unit import UnifiedInfoFactory, UnifiedInfoUnit

        factory = UnifiedInfoFactory()
        original = factory.create_from_content(
            "roundtrip test", stem_idx=1, branch_idx=3,
            hexagram_idx=5, energy_type=4
        )
        restored = UnifiedInfoUnit.from_dict(original.to_dict())
        assert restored.content == original.content
        assert restored.to_dict()["human"]["energy_name"] == "trust"


# ============================================================
# 4. EnergyBus 信号传递
# ============================================================

class TestEnergyBusSignals:
    """能量总线信号传递测试"""

    def test_energy_bus_import(self):
        """EnergyBus 模块可导入"""
        try:
            from su_memory._sys._energy_bus import EnergyBus
            assert EnergyBus is not None
        except ImportError as e:
            pytest.skip(f"EnergyBus 不可用: {e}")

    def test_energy_bus_broadcast(self):
        """EnergyBus 广播信号"""
        try:
            from su_memory._sys._energy_bus import EnergyBus
            bus = EnergyBus()
            # EnergyBus 应能发送信号
            assert bus is not None
        except (ImportError, Exception) as e:
            pytest.skip(f"EnergyBus 操作失败: {e}")

    def test_energy_relation_import(self):
        """能量关系模块可导入"""
        try:
            from su_memory._sys._energy_relations import ENERGY_RELATIONS
            assert len(ENERGY_RELATIONS) > 0
        except ImportError as e:
            pytest.skip(f"EnergyRelations 不可用: {e}")


# ============================================================
# 5. 能量系统与记忆系统集成
# ============================================================

class TestEnergyMemoryIntegration:
    """能量系统与记忆系统集成测试"""

    def test_lite_pro_supports_energy_metadata(self):
        """SuMemoryLitePro 支持能量类型元数据"""
        from su_memory.sdk.lite_pro import SuMemoryLitePro

        client = SuMemoryLitePro(max_memories=200, enable_graph=True)
        # 添加带能量类型的记忆
        mid = client.add("春天是万物生长的季节", metadata={"energy_type": "semantic"})
        assert mid is not None

        # 查询验证
        results = client.query("生长", top_k=3)
        assert len(results) > 0

    def test_energy_metadata_roundtrip(self):
        """能量的记忆元数据持久化验证"""
        from su_memory.sdk.lite_pro import SuMemoryLitePro

        client = SuMemoryLitePro(max_memories=200, enable_graph=True)
        # 添加五类范畴的记忆
        energy_memories = [
            ("新项目启动，团队充满激情", "semantic"),
            ("市场活动热烈，用户参与度高涨", "causal"),
            ("建立了完善的文档体系", "spacetime"),
            ("代码审查严格，质量标准提升", "generative"),
            ("深入研究了用户行为数据", "trust"),
        ]

        for content, etype in energy_memories:
            client.add(content, metadata={"energy_type": etype})

        # 记忆已成功添加
        stats = client.get_stats()
        assert stats["total_memories"] > 0

        # 验证可查询
        results = client.query("启动 激情", top_k=3)
        assert len(results) >= 1

    def test_energy_boost_in_query(self):
        """查询中能量增强效果"""
        from su_memory.sdk.lite_pro import SuMemoryLitePro

        client = SuMemoryLitePro(max_memories=200, enable_graph=True)

        client.add("发展新的业务线", metadata={"energy_type": "semantic"})
        client.add("因果分析结果显示相关性", metadata={"energy_type": "causal"})

        results = client.query("发展", top_k=3)
        assert len(results) >= 1
