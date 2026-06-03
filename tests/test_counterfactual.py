"""
su-memory v3.8.0 — Counterfactual 反事实引擎测试套件
======================================================

覆盖:
- StructuralEquationModel: 构造, 模拟, 溯因, 干预
- Abduction: 精确噪声恢复, 链式/混杂溯因, 未观测噪声
- CounterfactualQuery: 三步算法, 确定性, 置信区间
- PN/PS/PNS: 必然性/充分性概率计算
- Integration: CausalGraph → CounterfactualEngine, MCIWorldModel
- Edge Cases: 空图, 未知节点, 循环图, 单节点
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.causal

from su_memory.sdk._counterfactual import (
    CounterfactualEngine,
    CounterfactualResult,
    StructuralEquationModel,
)
from su_memory.sdk._do_calculus import CausalGraph

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def chain_graph() -> CausalGraph:
    """链式图: X → M → Y."""
    return CausalGraph(
        nodes=["X", "M", "Y"],
        edges=[("X", "M"), ("M", "Y")],
    )


@pytest.fixture
def confounded_graph() -> CausalGraph:
    """混杂图: Z → X, Z → Y, X → Y."""
    return CausalGraph(
        nodes=["Z", "X", "Y"],
        edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
    )


@pytest.fixture
def sem_chain(chain_graph) -> StructuralEquationModel:
    """链式 SEM."""
    return StructuralEquationModel(
        coefficients=np.array(chain_graph.adjacency, dtype=np.float64),
        node_names=list(chain_graph.nodes),
        noise_std=0.3,
        seed=42,
    )


@pytest.fixture
def sem_confounded(confounded_graph) -> StructuralEquationModel:
    """混杂 SEM."""
    return StructuralEquationModel(
        coefficients=np.array(confounded_graph.adjacency, dtype=np.float64),
        node_names=list(confounded_graph.nodes),
        noise_std=0.3,
        seed=42,
    )


# =============================================================================
# TestStructuralEquationModel
# =============================================================================


class TestStructuralEquationModel:
    """SEM 引擎基础功能。"""

    def test_empty_sem(self):
        """空 SEM 优雅处理。"""
        sem = StructuralEquationModel(
            coefficients=np.zeros((0, 0)),
            node_names=[],
        )
        assert sem.n_nodes == 0
        assert sem.node_index("X") is None

    def test_simulation_output_shape(self, sem_chain):
        """模拟数据形状正确。"""
        data = sem_chain.simulate(n_samples=100)
        assert data.shape == (100, 3)
        assert not np.any(np.isnan(data))

    def test_simulation_reproducible(self, sem_chain):
        """相同种子产生相同结果。"""
        data1 = sem_chain.simulate(n_samples=50)
        # 重建相同 SEM
        sem2 = StructuralEquationModel(
            coefficients=sem_chain.coefficients.copy(),
            node_names=list(sem_chain.node_names),
            noise_std=0.3,
            seed=42,
        )
        data2 = sem2.simulate(n_samples=50)
        assert np.allclose(data1, data2)

    def test_topological_sort(self, sem_chain):
        """拓扑排序正确。"""
        order = sem_chain._topological_sort()
        assert order is not None
        assert len(order) == 3
        # X 应该在 M 之前, M 应该在 Y 之前
        x_idx = sem_chain.node_index("X")
        m_idx = sem_chain.node_index("M")
        y_idx = sem_chain.node_index("Y")
        assert order.index(x_idx) < order.index(m_idx)
        assert order.index(m_idx) < order.index(y_idx)

    def test_abduce_full_evidence(self, sem_chain):
        """全证据溯因 — 精确恢复噪声。"""
        # 先生成一批数据
        data = sem_chain.simulate(n_samples=1)
        # 用第一行作为证据
        evidence = {
            "X": float(data[0, 0]),
            "M": float(data[0, 1]),
            "Y": float(data[0, 2]),
        }
        noise = sem_chain.abduce(evidence, n_samples=1)
        assert noise.shape == (1, 3)
        # 已知证据 → 噪声应精确恢复 (无随机性)
        # 验证: 给定噪声和父节点值，SEM 应精确生成观测值
        topo = sem_chain._topological_sort()
        assert topo is not None
        reconstructed = np.zeros(3)
        for node_i in topo:
            psum = 0.0
            for p_idx in range(3):
                if sem_chain.coefficients[p_idx, node_i] != 0:
                    psum += sem_chain.coefficients[p_idx, node_i] * reconstructed[p_idx]
            reconstructed[node_i] = psum + noise[0, node_i]
        for name in evidence:
            idx = sem_chain.node_index(name)
            assert abs(reconstructed[idx] - evidence[name]) < 1e-10

    def test_abduce_partial_evidence(self, sem_chain):
        """部分证据溯因 — 未观测节点噪声随机。"""
        evidence = {"X": 1.0}  # 只观测 X
        noise = sem_chain.abduce(evidence, n_samples=5)
        assert noise.shape == (5, 3)
        # X 的噪声应确定为 1.0 - 0 = 1.0 (根节点)
        x_idx = sem_chain.node_index("X")
        assert np.allclose(noise[:, x_idx], 1.0)
        # M 和 Y 的噪声应是随机的
        m_idx = sem_chain.node_index("M")
        assert np.std(noise[:, m_idx]) > 0

    def test_intervene_cuts_edges(self, sem_confounded):
        """干预切断了指向 X 的所有入边。"""
        mutilated = sem_confounded.intervene({"X": 5.0})
        z_idx = sem_confounded.node_index("Z")
        x_idx = sem_confounded.node_index("X")
        assert z_idx is not None and x_idx is not None
        # 原始: Z → X 边存在
        assert sem_confounded.coefficients[z_idx, x_idx] != 0
        # 干预后: Z → X 边被切断
        assert mutilated.coefficients[z_idx, x_idx] == 0.0
        # Z → Y 边仍保留
        y_idx = sem_confounded.node_index("Y")
        assert mutilated.coefficients[z_idx, y_idx] != 0

    def test_node_index(self, sem_chain):
        """节点索引查询。"""
        assert sem_chain.node_index("X") == 0
        assert sem_chain.node_index("Y") == 2
        assert sem_chain.node_index("Unknown") is None

    def test_repr(self, sem_chain):
        """字符串表示。"""
        rep = repr(sem_chain)
        assert "StructuralEquationModel" in rep
        assert "nodes=3" in rep


# =============================================================================
# TestAbduction
# =============================================================================


class TestAbduction:
    """溯因推断专项测试。"""

    def test_chain_abduction(self, sem_chain):
        """链式结构溯因: 给定 X, Y 推断噪声。"""
        # 生成确定性的模拟数据
        original_data = sem_chain.simulate(n_samples=1)
        x_val = float(original_data[0, 0])
        y_val = float(original_data[0, 2])

        # 只用 X, Y 溯因 (M 未观测)
        evidence = {"X": x_val, "Y": y_val}
        noise = sem_chain.abduce(evidence, n_samples=1)

        # X 噪声 = X (根节点)
        x_idx = sem_chain.node_index("X")
        assert abs(noise[0, x_idx] - x_val) < 1e-10

        # Y 噪声 = Y - β_{MY}·M (但 M 未观测 → M 由随机噪声决定)
        # 验证 M 的噪声是随机非零的
        m_idx = sem_chain.node_index("M")
        assert abs(noise[0, m_idx]) > 0  # 应有值 (随机)

    def test_confounded_abduction(self, sem_confounded):
        """混杂结构溯因: Z → X, Z → Y, X → Y."""
        data = sem_confounded.simulate(n_samples=1)
        evidence = {
            "Z": float(data[0, 0]),
            "X": float(data[0, 1]),
            "Y": float(data[0, 2]),
        }
        noise = sem_confounded.abduce(evidence, n_samples=1)
        # 所有证据已知 → 所有噪声精确推断
        for name in evidence:
            idx = sem_confounded.node_index(name)
            assert not np.isnan(noise[0, idx])

    def test_unobserved_node_noise_is_random(self, sem_chain):
        """未观测节点的噪声在不同采样间不同。"""
        evidence = {"X": 1.0}
        noise1 = sem_chain.abduce(evidence, n_samples=10)
        # M 和 Y 的噪声应在不同采样间有变化
        m_idx = sem_chain.node_index("M")
        assert np.std(noise1[:, m_idx]) > 0.01  # 有明显变化


# =============================================================================
# TestCounterfactualQuery
# =============================================================================


class TestCounterfactualQuery:
    """反事实查询 — Pearl 三步算法。"""

    def test_simple_counterfactual(self, chain_graph):
        """简单反事实: X→Y。如果 X=0，Y 会是多少？"""
        engine = CounterfactualEngine.from_causal_graph(chain_graph, noise_std=0.3)
        assert engine is not None
        result = engine.query(
            evidence={"X": 1.0, "Y": 3.0},
            do_x={"X": 0.0},
            target="Y",
        )
        assert result.status == "ok"
        assert result.factual_value != 0.0
        # 反事实值应在合理范围内
        assert abs(result.counterfactual_value) < 100.0
        # 个体效应应有值
        assert isinstance(result.individual_effect, float)

    def test_counterfactual_deterministic_given_full_evidence(self, chain_graph):
        """全证据下反事实结果可复现。"""
        engine = CounterfactualEngine.from_causal_graph(chain_graph, noise_std=0.3, seed=42)
        assert engine is not None

        r1 = engine.query({"X": 1.0, "Y": 3.0}, {"X": 0.0}, "Y", compute_pns=False)
        r2 = engine.query({"X": 1.0, "Y": 3.0}, {"X": 0.0}, "Y", compute_pns=False)

        assert abs(r1.counterfactual_value - r2.counterfactual_value) < 1e-6
        assert abs(r1.individual_effect - r2.individual_effect) < 1e-6

    def test_no_effect_when_x_not_parent_of_y(self):
        """X 不是 Y 的父节点时效应为 0。"""
        cg = CausalGraph(
            nodes=["X", "Z", "Y"],
            edges=[("Z", "Y")],  # X 孤立
        )
        engine = CounterfactualEngine.from_causal_graph(cg, noise_std=0.3)
        assert engine is not None
        result = engine.query(
            evidence={"X": 1.0, "Z": 0.0, "Y": 2.0},
            do_x={"X": 0.0},
            target="Y",
            compute_pns=False,
        )
        assert result.status == "ok"
        # X 不是 Y 的父节点, do(X) 不应改变 Y
        assert abs(result.individual_effect) < 1e-6

    def test_ci_95_is_valid_interval(self, chain_graph):
        """置信区间格式正确。"""
        engine = CounterfactualEngine.from_causal_graph(chain_graph, noise_std=0.5)
        assert engine is not None
        result = engine.query(
            evidence={"X": 1.0},
            do_x={"X": 0.0},
            target="Y",
        )
        lo, hi = result.ci_95
        assert lo <= result.counterfactual_value <= hi
        assert lo < hi

    def test_counterfactual_direction(self, chain_graph):
        """干预方向正确: 增大 X 应增大 Y (正边权重)。"""
        engine = CounterfactualEngine.from_causal_graph(chain_graph, noise_std=0.01, seed=42)
        assert engine is not None

        # do(X=0) 的 Y 应 < do(X=2) 的 Y
        r_low = engine.query({"X": 0.0}, {"X": 0.0}, "Y", compute_pns=False)
        r_high = engine.query({"X": 0.0}, {"X": 2.0}, "Y", compute_pns=False)

        assert r_low.status == "ok"
        assert r_high.status == "ok"
        assert r_low.counterfactual_value < r_high.counterfactual_value


# =============================================================================
# TestPNPSPNS
# =============================================================================


class TestPNPSPNS:
    """必然性/充分性概率测试。"""

    def test_pn_computation(self, chain_graph):
        """PN 计算: 给定 X=1,Y=大值, do(X=0) 下 Y 变化概率。"""
        engine = CounterfactualEngine.from_causal_graph(chain_graph, noise_std=0.3, seed=42)
        assert engine is not None
        result = engine.query(
            evidence={"X": 1.0, "Y": 5.0},
            do_x={"X": 0.0},
            target="Y",
            compute_pns=True,
            n_mc=100,
        )
        assert result.pn >= 0.0
        assert result.ps >= 0.0
        assert result.pns >= 0.0

    def test_pn_in_range(self, chain_graph):
        """PN/PS/PNS 在 [0,1] 范围内。"""
        engine = CounterfactualEngine.from_causal_graph(chain_graph, noise_std=0.3, seed=42)
        assert engine is not None
        result = engine.query(
            evidence={"X": 1.0, "Y": 3.0},
            do_x={"X": 0.0},
            target="Y",
            compute_pns=True,
            n_mc=100,
        )
        assert 0.0 <= result.pn <= 1.0
        assert 0.0 <= result.ps <= 1.0
        assert 0.0 <= result.pns <= 1.0

    def test_pns_consistency(self, chain_graph):
        """PNS ≤ min(PN, PS) 始终成立。"""
        engine = CounterfactualEngine.from_causal_graph(chain_graph, noise_std=0.3, seed=42)
        assert engine is not None
        result = engine.query(
            evidence={"X": 1.0, "Y": 3.0},
            do_x={"X": 0.0},
            target="Y",
            compute_pns=True,
            n_mc=150,
        )
        assert result.pns <= min(result.pn, result.ps) + 0.05  # Monte Carlo 误差


# =============================================================================
# TestIntegration
# =============================================================================


class TestIntegration:
    """集成测试。"""

    def test_from_causal_graph(self, confounded_graph):
        """从 CausalGraph 构建反事实引擎。"""
        engine = CounterfactualEngine.from_causal_graph(confounded_graph, noise_std=0.5)
        assert engine is not None
        assert engine.node_names == ["Z", "X", "Y"]

    def test_from_empty_graph(self):
        """空图返回 None。"""
        cg = CausalGraph()
        engine = CounterfactualEngine.from_causal_graph(cg)
        assert engine is None

    def test_to_dict(self, chain_graph):
        """CounterfactualResult.to_dict() 结构正确。"""
        engine = CounterfactualEngine.from_causal_graph(chain_graph)
        assert engine is not None
        result = engine.query({"X": 1.0, "Y": 3.0}, {"X": 0.0}, "Y")
        d = result.to_dict()
        assert "evidence" in d
        assert "do_intervention" in d
        assert "target" in d
        assert "counterfactual_value" in d
        assert "individual_effect" in d
        assert "status" in d
        assert d["status"] == "ok"

    def test_world_model_counterfactual(self):
        """MCIWorldModel.query_counterfactual() 可用。"""
        from su_memory.sdk._world_model import MCIWorldModel

        wm = MCIWorldModel()
        result = wm.query_counterfactual(
            evidence={"X": 1.0, "Y": 3.0},
            do_x={"X": 0.0},
            target="Y",
            compute_pns=False,
        )
        assert "status" in result
        # 无因果图时也能给出合理结果
        assert result["status"] in ("ok", "error")

    def test_batch_query(self, chain_graph):
        """批量反事实查询。"""
        engine = CounterfactualEngine.from_causal_graph(chain_graph)
        assert engine is not None
        scenarios = [
            {"evidence": {"X": 1.0, "Y": 3.0}, "do_x": {"X": 0.0}, "target": "Y"},
            {"evidence": {"X": 2.0, "Y": 5.0}, "do_x": {"X": 1.0}, "target": "Y"},
        ]
        results = engine.batch_query(scenarios)
        assert len(results) == 2
        for r in results:
            assert r.status == "ok"


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """边界条件测试。"""

    def test_unknown_target_node(self, chain_graph):
        """未知目标节点返回错误。"""
        engine = CounterfactualEngine.from_causal_graph(chain_graph)
        assert engine is not None
        result = engine.query(
            evidence={"X": 1.0},
            do_x={"X": 0.0},
            target="Unknown",
        )
        assert result.status == "error"
        assert "not in graph" in result.note

    def test_unknown_evidence_node(self, chain_graph):
        """未知证据节点返回错误。"""
        engine = CounterfactualEngine.from_causal_graph(chain_graph)
        assert engine is not None
        result = engine.query(
            evidence={"W": 1.0},  # W 不在图中
            do_x={"X": 0.0},
            target="Y",
        )
        assert result.status == "error"

    def test_single_node_graph(self):
        """单节点图无因果效应。"""
        cg = CausalGraph(nodes=["X"], edges=[])
        engine = CounterfactualEngine.from_causal_graph(cg)
        assert engine is not None
        result = engine.query(
            evidence={"X": 1.0},
            do_x={"X": 0.0},
            target="X",
            compute_pns=False,
        )
        assert result.status == "ok"
        # do(X=0) 后 X 就是 0
        assert abs(result.counterfactual_value - 0.0) < 1e-10

    def test_counterfactual_result_empty(self):
        """empty() 工厂方法。"""
        empty = CounterfactualResult.empty(
            evidence={"X": 1.0},
            do_intervention={"X": 0.0},
            target="Y",
            note="test",
        )
        assert empty.status == "error"
        assert empty.note == "test"
        d = empty.to_dict()
        assert d["status"] == "error"

    def test_large_coefficient_stability(self):
        """大系数图数值稳定。"""
        adj = np.array([[0, 10.0], [0, 0]], dtype=np.float64)
        sem = StructuralEquationModel(
            coefficients=adj,
            node_names=["X", "Y"],
            noise_std=0.1,
            seed=42,
        )
        data = sem.simulate(n_samples=100)
        assert not np.any(np.isnan(data))
        assert not np.any(np.isinf(data))
