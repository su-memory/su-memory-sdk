"""v4.0.0 JEPA: 三个不可参基线预测器测试."""

from __future__ import annotations

import pytest

from su_memory.sdk._world_model import CausalWorldModelState
from su_memory.sdk._jepa_predictor import (
    IdentityPredictor,
    EnergyPropagationPredictor,
    BeliefPropagationPredictor,
)

pytestmark = pytest.mark.jepa


def _make_state(edges: list[dict] | None = None, timestamp: str = "") -> CausalWorldModelState:
    return CausalWorldModelState(
        causal_edges=edges or [],
        n_memories=10,
        timestamp=timestamp or "2026-01-01T00:00:00",
    )


class TestIdentityPredictor:
    """恒等预测器测试。"""

    def test_output_is_deep_equal(self):
        s = _make_state([
            {"cause": "price", "effect": "demand", "rho": 0.72, "confidence": 0.85},
            {"cause": "supply", "effect": "price", "rho": 0.58, "confidence": 0.70},
        ])
        p = IdentityPredictor()
        s_pred = p.predict(s)
        assert s_pred.state_distance(s) == 0.0

    def test_output_is_not_same_object(self):
        s = _make_state([{"cause": "a", "effect": "b", "rho": 0.5}])
        p = IdentityPredictor()
        s_pred = p.predict(s)
        assert s_pred is not s
        assert s_pred.causal_edges == s.causal_edges
        # 但 list 是不同对象
        assert s_pred.causal_edges is not s.causal_edges

    def test_preserves_metadata(self):
        s = _make_state([{"cause": "a", "effect": "b", "rho": 0.5}])
        s.n_confirmed = 3
        s.active_states = {"semantic", "causal"}
        p = IdentityPredictor()
        s_pred = p.predict(s)
        assert s_pred.n_confirmed == 3
        assert s_pred.active_states == {"semantic", "causal"}


class TestEnergyPropagationPredictor:
    """能量传播预测器测试。"""

    def test_enhance_edge_strengthens(self):
        s = _make_state([
            {"cause": "semantic", "effect": "causal", "rho": 0.70, "confidence": 0.80, "energy_relation": "enhance"},
        ])
        p = EnergyPropagationPredictor(propagation_alpha=0.2)
        s_pred = p.predict(s)
        new_rho = s_pred.causal_edges[0]["rho"]
        assert new_rho > 0.70  # 增强边强度增加

    def test_suppress_edge_weakens(self):
        s = _make_state([
            {"cause": "semantic", "effect": "spacetime", "rho": 0.70, "confidence": 0.80, "energy_relation": "suppress"},
        ])
        p = EnergyPropagationPredictor(propagation_alpha=0.2)
        s_pred = p.predict(s)
        new_rho = s_pred.causal_edges[0]["rho"]
        assert new_rho < 0.70  # 抑制边强度减弱

    def test_neutral_edge_dampens(self):
        s = _make_state([
            {"cause": "a", "effect": "b", "rho": 0.70, "confidence": 0.80, "energy_relation": "neutral"},
        ])
        p = EnergyPropagationPredictor(propagation_alpha=0.2)
        s_pred = p.predict(s)
        new_rho = s_pred.causal_edges[0]["rho"]
        assert new_rho <= 0.70  # 中性边轻微衰减

    def test_empty_input(self):
        s = _make_state()
        p = EnergyPropagationPredictor()
        s_pred = p.predict(s)
        assert len(s_pred.causal_edges) == 0


class TestBeliefPropagationPredictor:
    """贝叶斯信念传播预测器测试。"""

    def test_output_has_same_edge_count(self):
        s = _make_state([
            {"cause": "price", "effect": "demand", "rho": 0.72, "confidence": 0.85},
            {"cause": "supply", "effect": "price", "rho": 0.58, "confidence": 0.70},
        ])
        p = BeliefPropagationPredictor()
        s_pred = p.predict(s)
        assert len(s_pred.causal_edges) == 2

    def test_weak_edges_suppressed(self):
        s = _make_state([
            {"cause": "x", "effect": "y", "rho": 0.01, "confidence": 0.10},
        ])
        p = BeliefPropagationPredictor()
        s_pred = p.predict(s)
        # 弱边标记为 suppressed
        assert s_pred.causal_edges[0].get("verdict") == "suppressed"

    def test_empty_input(self):
        s = _make_state()
        p = BeliefPropagationPredictor()
        s_pred = p.predict(s)
        assert len(s_pred.causal_edges) == 0


class TestBaselineRanking:
    """基线精度排序验证。"""

    def test_identity_worse_than_energy(self):
        s = _make_state([
            {"cause": "price", "effect": "demand", "rho": 0.72, "energy_relation": "enhance"},
        ])
        s_actual = _make_state([
            {"cause": "price", "effect": "demand", "rho": 0.80},
        ])

        id_pred = IdentityPredictor().predict(s)
        ep_pred = EnergyPropagationPredictor(propagation_alpha=0.1).predict(s)

        id_err = id_pred.state_distance(s_actual)
        ep_err = ep_pred.state_distance(s_actual)
        # Energy 至少不比 Identity 差
        assert ep_err <= id_err + 0.01  # 允许浮点误差

    def test_evaluate_method(self):
        p = IdentityPredictor()
        s1 = _make_state([{"cause": "a", "effect": "b", "rho": 0.5}])
        s2 = _make_state([{"cause": "a", "effect": "b", "rho": 0.5}])
        result = p.evaluate([(s1, s2)])
        assert "avg_distance" in result
        assert result["n"] == 1

    def test_empty_evaluate(self):
        p = IdentityPredictor()
        result = p.evaluate([])
        assert result["n"] == 0
        assert result["avg_distance"] == 1.0
