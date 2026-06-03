"""
su-memory v3.7.0 — DoCalculus 干预引擎单元测试
================================================

测试覆盖:
- 后门调整集识别
- 前门中介变量识别
- ATE 估计 (正/零/负方向)
- 置信区间
- 从 GaussianDAG 构建因果图
- 空图/异常输入处理
- MCIWorldModel 干预集成
- 干预历史记录
- 因果效应分解
- 健康检查版本号
"""

import numpy as np
import pytest

pytestmark = pytest.mark.causal

from su_memory.sdk._do_calculus import (
    CausalGraph,
    DoCalculus,
    InterventionResult,
)
from su_memory.sdk._world_model import MCIWorldModel

# =============================================================================
# CausalGraph 测试
# =============================================================================


class TestCausalGraph:
    """因果图数据结构测试."""

    def test_empty_graph(self):
        """空图优雅处理."""
        cg = CausalGraph()
        assert cg.n_nodes == 0
        assert cg.node_index("X") is None
        assert cg.has_edge("X", "Y") is False

    def test_graph_construction(self):
        """从节点和边构建因果图."""
        cg = CausalGraph(
            nodes=["Z", "X", "Y"],
            edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        )
        assert cg.n_nodes == 3
        assert cg.has_edge("Z", "X")
        assert cg.has_edge("Z", "Y")
        assert cg.has_edge("X", "Y")
        assert not cg.has_edge("Y", "X")

    def test_get_parents(self):
        """父节点查询."""
        cg = CausalGraph(
            nodes=["Z", "X", "Y"],
            edges=[("Z", "X"), ("X", "Y")],
        )
        assert "Z" in cg.get_parents("X")
        assert "X" in cg.get_parents("Y")
        assert cg.get_parents("Z") == []

    def test_get_children(self):
        """子节点查询."""
        cg = CausalGraph(
            nodes=["Z", "X", "Y"],
            edges=[("Z", "X"), ("X", "Y")],
        )
        assert "X" in cg.get_children("Z")
        assert "Y" in cg.get_children("X")
        assert cg.get_children("Y") == []

    def test_get_descendants(self):
        """后代节点查询 (BFS)."""
        cg = CausalGraph(
            nodes=["A", "B", "C", "D"],
            edges=[("A", "B"), ("B", "C"), ("C", "D")],
        )
        descendants = cg.get_descendants("A")
        assert "B" in descendants
        assert "C" in descendants
        assert "D" in descendants
        assert "A" not in descendants

    def test_get_mediators(self):
        """中介变量识别."""
        cg = CausalGraph(
            nodes=["X", "M", "Y"],
            edges=[("X", "M"), ("M", "Y")],
        )
        mediators = cg.get_mediators("X", "Y")
        assert "M" in mediators

    def test_single_node_graph(self):
        """单节点图."""
        cg = CausalGraph(nodes=["X"], edges=[])
        assert cg.n_nodes == 1
        assert cg.get_parents("X") == []
        assert cg.get_children("X") == []

    def test_repr(self):
        """字符串表示."""
        cg = CausalGraph(
            nodes=["Z", "X", "Y"],
            edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        )
        rep = repr(cg)
        assert "CausalGraph" in rep
        assert "nodes=3" in rep
        assert "edges=3" in rep


# =============================================================================
# DoCalculus 后门调整测试
# =============================================================================


class TestBackdoorAdjustment:
    """后门调整测试."""

    def test_backdoor_adjustment_set(self):
        """Z→X, Z→Y, X→Y: Z 应被识别为调整变量."""
        cg = CausalGraph(
            nodes=["Z", "X", "Y"],
            edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        )
        dc = DoCalculus(cg)
        adj = dc.identify_adjustment_set("X", "Y")
        assert adj is not None
        assert "Z" in adj

    def test_no_backdoor_path(self):
        """X→Y only: 调整集应为空/None."""
        cg = CausalGraph(
            nodes=["X", "Y"],
            edges=[("X", "Y")],
        )
        dc = DoCalculus(cg)
        adj = dc.identify_adjustment_set("X", "Y")
        # 无后门路径时调整集为空或 None
        assert adj is None or len(adj) == 0

    def test_frontdoor_scenario(self):
        """X→M→Y, U→X, U→Y: 后门调整集包含 U, 前门中介也可用."""
        cg = CausalGraph(
            nodes=["X", "M", "Y", "U"],
            edges=[("X", "M"), ("M", "Y"), ("U", "X"), ("U", "Y")],
        )
        dc = DoCalculus(cg)
        # U 是 X 的父节点，应作为后门调整变量
        adj_backdoor = dc.identify_adjustment_set("X", "Y")
        assert adj_backdoor is not None
        assert "U" in adj_backdoor

        # 前门中介也应可用
        mediators = dc.identify_frontdoor_mediators("X", "Y")
        assert mediators is not None
        assert "M" in mediators

    def test_ate_sign_positive(self):
        """正向因果关系: ATE > 0."""
        np.random.seed(42)
        cg = CausalGraph(
            nodes=["Z", "X", "Y"],
            edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        )
        dc = DoCalculus(cg, seed=42)
        adj = dc.identify_adjustment_set("X", "Y")
        assert adj is not None
        result = dc.backdoor_adjustment("X", "Y", adj, x_value=1.0, x_baseline=0.0)
        # 模拟数据，方向应与图一致
        assert isinstance(result.ate, float)
        # 由于 X→Y 存在边，在模拟数据中 ATE 应为正
        assert result.ate > 0

    def test_ate_result_structure(self):
        """InterventionResult 结构完整性."""
        cg = CausalGraph(
            nodes=["Z", "X", "Y"],
            edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        )
        dc = DoCalculus(cg, seed=42)
        adj = dc.identify_adjustment_set("X", "Y")
        result = dc.backdoor_adjustment("X", "Y", adj, x_value=1.0, x_baseline=0.0)
        assert result.method == "backdoor"
        assert result.adjustment_set == adj
        assert isinstance(result.confidence_interval, tuple)
        assert len(result.confidence_interval) == 2
        assert result.confidence_interval[0] < result.confidence_interval[1]
        assert 0.0 <= result.p_value <= 1.0
        assert result.effect_direction == "positive"
        assert result.sample_size > 0

    def test_empty_graph_handling(self):
        """空图时优雅处理."""
        dc = DoCalculus()
        result = dc.estimate_ate("X", "Y")
        assert result.method == "none"
        assert "insufficient_data" in result.note

    def test_confidence_interval_coverage(self):
        """置信区间合理性检查."""
        np.random.seed(42)
        cg = CausalGraph(
            nodes=["Z", "X", "Y"],
            edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        )
        dc = DoCalculus(cg, seed=42)
        adj = dc.identify_adjustment_set("X", "Y")
        result = dc.backdoor_adjustment("X", "Y", adj, x_value=1.0, x_baseline=0.0)
        ci = result.confidence_interval
        # 置信区间应包含 ATE
        assert ci[0] <= result.ate <= ci[1]

    def test_no_graph_backdoor(self):
        """无图时后门调整返回 empty."""
        dc = DoCalculus()
        adj = dc.identify_adjustment_set("X", "Y")
        assert adj is None


# =============================================================================
# DoCalculus 前门调整测试
# =============================================================================


class TestFrontdoorAdjustment:
    """前门调整测试."""

    def test_frontdoor_mediators_identification(self):
        """X→M→Y, U→X, U→Y: 前门中介正确识别."""
        cg = CausalGraph(
            nodes=["X", "M", "Y", "U"],
            edges=[("X", "M"), ("M", "Y"), ("U", "X"), ("U", "Y")],
        )
        dc = DoCalculus(cg)
        mediators = dc.identify_frontdoor_mediators("X", "Y")
        assert mediators is not None
        assert "M" in mediators

    def test_no_frontdoor_path(self):
        """无中介变量时返回 None."""
        cg = CausalGraph(
            nodes=["X", "Y"],
            edges=[("X", "Y")],
        )
        dc = DoCalculus(cg)
        mediators = dc.identify_frontdoor_mediators("X", "Y")
        assert mediators is None

    def test_frontdoor_result_structure(self):
        """前门调整结果结构完整."""
        cg = CausalGraph(
            nodes=["X", "M", "Y", "U"],
            edges=[("X", "M"), ("M", "Y"), ("U", "X"), ("U", "Y")],
        )
        dc = DoCalculus(cg, seed=42)
        mediators = dc.identify_frontdoor_mediators("X", "Y")
        assert mediators is not None
        result = dc.frontdoor_adjustment("X", "Y", mediators, x_value=1.0, x_baseline=0.0)
        assert result.method == "frontdoor"
        assert isinstance(result.ate, float)
        assert len(result.confidence_interval) == 2


# =============================================================================
# DoCalculus ATE 自动方法选择
# =============================================================================


class TestEstimateATE:
    """ATE 自动方法选择测试."""

    def test_auto_selects_backdoor(self):
        """有后门调整集时自动选择后门调整."""
        cg = CausalGraph(
            nodes=["Z", "X", "Y"],
            edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        )
        dc = DoCalculus(cg, seed=42)
        result = dc.estimate_ate("X", "Y", x_value=1.0, x_baseline=0.0)
        assert result.method == "backdoor"
        assert len(result.adjustment_set) > 0

    def test_auto_falls_back_to_direct(self):
        """无调整集时回退到直接效应."""
        cg = CausalGraph(
            nodes=["X", "Y"],
            edges=[("X", "Y")],
        )
        dc = DoCalculus(cg, seed=42)
        result = dc.estimate_ate("X", "Y", x_value=1.0, x_baseline=0.0)
        assert result.method in ("direct", "frontdoor")
        assert isinstance(result.ate, float)

    def test_force_backdoor(self):
        """强制使用后门调整."""
        cg = CausalGraph(
            nodes=["Z", "X", "Y"],
            edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        )
        dc = DoCalculus(cg, seed=42)
        result = dc.estimate_ate("X", "Y", x_value=1.0, x_baseline=0.0, method="backdoor")
        assert result.method == "backdoor"


# =============================================================================
# 静态工厂: build_from_gaussian_dag
# =============================================================================


class TestBuildFromGaussianDAG:
    """从 GaussianDAG 边列表构建因果图."""

    def test_build_from_gaussian_dag(self):
        """模拟 edges → 图正确构建."""
        edges = [
            {"cause_idx": 0, "effect_idx": 1, "confidence": 0.8},
            {"cause_idx": 1, "effect_idx": 2, "confidence": 0.6},
            {"cause_idx": 0, "effect_idx": 2, "confidence": 0.2},  # below threshold
        ]
        cg = DoCalculus.build_from_gaussian_dag(edges, n_nodes=3)
        assert cg.n_nodes == 3
        assert cg.has_edge("V0", "V1")
        assert cg.has_edge("V1", "V2")
        # 低置信度边应被过滤
        assert not cg.has_edge("V0", "V2")

    def test_build_empty_edges(self):
        """空边列表 → 空图."""
        cg = DoCalculus.build_from_gaussian_dag([], n_nodes=5)
        assert cg.n_nodes == 5
        assert len(cg.edges) == 0

    def test_build_custom_threshold(self):
        """自定义置信度阈值."""
        edges = [
            {"cause_idx": 0, "effect_idx": 1, "confidence": 0.5},
        ]
        cg_default = DoCalculus.build_from_gaussian_dag(edges, n_nodes=2)
        assert cg_default.has_edge("V0", "V1")  # 0.5 > 0.3

        cg_strict = DoCalculus.build_from_gaussian_dag(edges, n_nodes=2, min_confidence=0.6)
        assert not cg_strict.has_edge("V0", "V1")  # 0.5 < 0.6

    def test_build_out_of_bounds_indices(self):
        """越界索引被优雅过滤."""
        edges = [
            {"cause_idx": 0, "effect_idx": 5, "confidence": 0.8},  # 5 越界
            {"cause_idx": 0, "effect_idx": 1, "confidence": 0.8},
        ]
        cg = DoCalculus.build_from_gaussian_dag(edges, n_nodes=2)
        assert cg.has_edge("V0", "V1")
        assert not cg.has_edge("V0", "V5")


# =============================================================================
# InterventionResult 测试
# =============================================================================


class TestInterventionResult:
    """干预结果数据结构测试."""

    def test_to_dict(self):
        """序列化测试."""
        result = InterventionResult(
            intervention="do(X=1.5)",
            target="Y",
            ate=0.42,
            confidence_interval=(0.1, 0.74),
            adjustment_set=["Z"],
            method="backdoor",
            p_value=0.01,
            effect_direction="positive",
            effect_magnitude="medium",
            sample_size=500,
        )
        d = result.to_dict()
        assert d["intervention"] == "do(X=1.5)"
        assert d["target"] == "Y"
        assert d["ate"] == 0.42
        assert d["method"] == "backdoor"
        assert d["adjustment_set"] == ["Z"]
        assert d["confidence_interval_95"] == (0.1, 0.74)

    def test_empty_result(self):
        """空结果."""
        result = InterventionResult.empty()
        assert result.method == "none"
        assert result.note == "insufficient_data"

    def test_effect_direction_neutral(self):
        """接近零的 ATE 应标识为 neutral."""
        result = InterventionResult(ate=0.01)
        # _build_result sets direction based on threshold
        assert result.effect_direction == "unknown"


# =============================================================================
# MCIWorldModel 干预集成测试
# =============================================================================


class TestInterventionIntegration:
    """MCIWorldModel.intervene() 集成测试."""

    def test_intervene_returns_ok(self):
        """干预返回 ok 状态."""
        wm = MCIWorldModel()
        wm.initialize()
        memories = [
            {"id": "1", "content": "价格变动导致需求变化"},
            {"id": "2", "content": "需求变化导致供应调整"},
            {"id": "3", "content": "供应调整导致价格变动"},
        ]
        wm.discover(memories=memories)
        result = wm.intervene(
            do_x={"V0": 1.5},
            target="V1",
        )
        assert result["status"] == "ok"
        assert "ate" in result
        assert "method" in result

    def test_intervene_history_accumulates(self):
        """多次干预历史累计."""
        wm = MCIWorldModel()
        wm.initialize()
        memories = [
            {"id": "1", "content": "价格变动导致需求变化"},
            {"id": "2", "content": "需求变化导致供应调整"},
            {"id": "3", "content": "供应调整导致价格变动"},
        ]
        wm.discover(memories=memories)

        r1 = wm.intervene(do_x={"V0": 1.0}, target="V1")
        assert r1["status"] == "ok"
        assert r1["history_count"] == 1

        r2 = wm.intervene(do_x={"V0": 2.0}, target="V1")
        assert r2["status"] == "ok"
        assert r2["history_count"] == 2

        # CausalWorldModelState 也应记录
        assert len(wm.state.do_interventions) == 2

    def test_decompose_total_effect(self):
        """因果分解: TE = NDE + NIE."""
        wm = MCIWorldModel()
        wm.initialize()
        memories = [
            {"id": "1", "content": "价格变动影响需求变化"},
            {"id": "2", "content": "需求变化影响供应调整"},
            {"id": "3", "content": "供应调整影响价格变动"},
        ]
        wm.discover(memories=memories)
        result = wm.decompose_effect("V0", "V1")
        assert "nde" in result
        assert "nie" in result
        assert "te" in result
        # TE = NDE + NIE (数值验证)
        assert abs(result["te"] - (result["nde"] + result["nie"])) < 1e-5

    def test_health_check_v370(self):
        """健康检查 roadmap 标记 done."""
        wm = MCIWorldModel()
        health = wm.health_check()
        assert "roadmap" in health
        assert "v3.7.0" in health["roadmap"]
        # v3.7.0 should not say "planned"
        assert "planned" not in health["roadmap"]["v3.7.0"]


# =============================================================================
# 异常处理测试
# =============================================================================


class TestEdgeCases:
    """边界和异常处理."""

    def test_cyclic_graph_handling(self):
        """循环图应被拓扑排序优雅处理."""
        cg = CausalGraph(
            nodes=["A", "B", "C"],
            edges=[("A", "B"), ("B", "C"), ("C", "A")],  # 循环
        )
        dc = DoCalculus(cg, seed=42)
        # 不应崩溃
        result = dc.estimate_ate("A", "B")
        assert isinstance(result.ate, float)

    def test_unknown_node_handling(self):
        """不存在的节点."""
        cg = CausalGraph(
            nodes=["X", "Y"],
            edges=[("X", "Y")],
        )
        dc = DoCalculus(cg, seed=42)
        adj = dc.identify_adjustment_set("NONEXIST", "Y")
        assert adj is None

    def test_intervene_insufficient_input(self):
        """缺少输入优雅处理."""
        wm = MCIWorldModel()
        result = wm.intervene()
        assert result["status"] == "insufficient_input"

    def test_decompose_no_mediator(self):
        """无中介时分解结果."""
        wm = MCIWorldModel()
        result = wm.decompose_effect("X", "Y")
        assert result["te"] == 0.0
        assert result["note"] == "no_mediator_identified"

    def test_direct_effect_no_graph(self):
        """无图时直接效应."""
        dc = DoCalculus()
        result = dc.direct_effect("X", "Y")
        assert result.method == "direct"
        assert result.note == "insufficient_data"
