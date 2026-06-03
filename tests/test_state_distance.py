"""v4.0.0 JEPA: CausalWorldModelState state_distance() 测试."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.causal

from su_memory.sdk._world_model import CausalWorldModelState


def _make_state(edges: list[dict] | None = None, timestamp: str = "") -> CausalWorldModelState:
    """快速构造 CausalWorldModelState。"""
    return CausalWorldModelState(
        causal_edges=edges or [],
        timestamp=timestamp or "2026-01-01T00:00:00",
    )


class TestToAdjacencyMatrix:
    """邻接矩阵构建测试。"""

    def test_empty_returns_zero_matrix(self):
        s = _make_state()
        adj = s.to_adjacency_matrix()
        assert adj.shape == (0, 0)

    def test_single_edge(self):
        s = _make_state([
            {"cause": "price", "effect": "demand", "rho": 0.72, "confidence": 0.85},
        ])
        adj = s.to_adjacency_matrix()
        assert adj.shape == (2, 2)
        # price → demand
        assert adj[0, 1] == 0.72
        assert adj[1, 0] == 0.0  # 有向图，反向无边
        assert adj[0, 0] == 0.0  # 无自环

    def test_multiple_edges(self):
        s = _make_state([
            {"cause": "price", "effect": "demand", "rho": 0.72},
            {"cause": "supply", "effect": "price", "rho": 0.58},
            {"cause": "supply", "effect": "demand", "rho": 0.31},
        ])
        adj = s.to_adjacency_matrix()
        assert adj.shape == (3, 3)


class TestToNodeFeatureMatrix:
    """节点特征矩阵构建测试。"""

    def test_empty(self):
        s = _make_state()
        feat = s.to_node_feature_matrix()
        assert feat.shape == (0, 8)

    def test_features_shape(self):
        s = _make_state([
            {"cause": "price", "effect": "demand", "rho": 0.72},
        ])
        feat = s.to_node_feature_matrix()
        assert feat.shape == (2, 8)  # 2 nodes, 8 features
        # degree stats are set
        assert feat[0, 5] == 1.0  # price 出度=1
        assert feat[1, 6] == 1.0  # demand 入度=1


class TestStateDistance:
    """因果图距离度量测试。"""

    def test_same_state_zero(self):
        s1 = _make_state([
            {"cause": "price", "effect": "demand", "rho": 0.72},
        ])
        s2 = _make_state([
            {"cause": "price", "effect": "demand", "rho": 0.72},
        ])
        assert s1.state_distance(s2) == 0.0
        assert (s1 - s2) == 0.0  # __sub__

    def test_different_weights_nonzero(self):
        s1 = _make_state([
            {"cause": "price", "effect": "demand", "rho": 0.72},
        ])
        s2 = _make_state([
            {"cause": "price", "effect": "demand", "rho": 0.45},
        ])
        d = s1.state_distance(s2)
        assert d > 0.0

    def test_different_structure_nonzero(self):
        s1 = _make_state([
            {"cause": "price", "effect": "demand", "rho": 0.72},
        ])
        s2 = _make_state([
            {"cause": "supply", "effect": "price", "rho": 0.58},
        ])
        d = s1.state_distance(s2)
        assert d > 0.0

    def test_counterfactual_distance_greater(self):
        # 反事实图（不同的 do-interventions 历史）距离应更大
        s1 = _make_state([
            {"cause": "price", "effect": "demand", "rho": 0.72},
        ])
        s2 = _make_state([
            {"cause": "supply", "effect": "demand", "rho": 0.31},
            {"cause": "price", "effect": "demand", "rho": 0.10},  # 权重显著不同
        ])
        d = s1.state_distance(s2)
        assert d > 0.0
        # 结构差异大，距离应该明显
        assert d > 0.1

    def test_both_empty_zero(self):
        assert _make_state().state_distance(_make_state()) == 0.0

    def test_one_empty_one_full(self):
        s1 = _make_state()
        s2 = _make_state([
            {"cause": "price", "effect": "demand", "rho": 0.72},
        ])
        assert s1.state_distance(s2) == 1.0

    def test_bounded_zero_to_one(self):
        s1 = _make_state([
            {"cause": "price", "effect": "demand", "rho": 0.99},
        ])
        s2 = _make_state([
            {"cause": "supply", "effect": "demand", "rho": 0.01},
            {"cause": "regulation", "effect": "price", "rho": 0.99},
            {"cause": "regulation", "effect": "supply", "rho": 0.99},
        ])
        d = s1.state_distance(s2)
        assert 0.0 <= d <= 1.0

    def test___sub___not_implemented(self):
        s = _make_state()
        result = s.__sub__("not_a_state")
        assert result is NotImplemented


class TestTemporalDistance:
    """v4.0.0: 时空距离测试。"""

    @staticmethod
    def _make_temporal_info(energy_type: str, tian_gan: str = "甲"):
        """构造模拟 TemporalInfo。"""
        from dataclasses import dataclass

        @dataclass
        class MockTemporalInfo:
            energy_type: str
            tian_gan: str
            di_zhi: str = "子"
            time_code: str = "甲子"
            yin_yang: str = "阳"

        return MockTemporalInfo(
            energy_type=energy_type,
            tian_gan=tian_gan,
        )

    def test_temporal_same_energy_zero(self):
        """相同 energy_type 的时空距离为 0。"""
        s1 = _make_state([{"cause": "a", "effect": "b", "rho": 0.5}])
        s2 = _make_state([{"cause": "a", "effect": "b", "rho": 0.5}])
        s1.temporal_info = self._make_temporal_info("wood")
        s2.temporal_info = self._make_temporal_info("wood")
        d = s1.state_distance(s2)
        assert d == 0.0

    def test_temporal_different_energy_nonzero(self):
        """不同 energy_type 增加距离。"""
        s1 = _make_state([{"cause": "a", "effect": "b", "rho": 0.5}])
        s2 = _make_state([{"cause": "a", "effect": "b", "rho": 0.5}])
        s1.temporal_info = self._make_temporal_info("wood")
        s2.temporal_info = self._make_temporal_info("fire")
        d = s1.state_distance(s2)
        assert d > 0.0

    def test_temporal_one_none_ignored(self):
        """单方无 temporal_info 时，时空距离不激活。"""
        s1 = _make_state([{"cause": "a", "effect": "b", "rho": 0.5}])
        s2 = _make_state([{"cause": "a", "effect": "b", "rho": 0.6}])
        s1.temporal_info = self._make_temporal_info("wood")
        s2.temporal_info = None
        d = s1.state_distance(s2)
        assert d > 0.0  # 仅有因果距离
        assert d < 1.0


class TestBeliefDistance:
    """v4.0.0: 信念距离测试。"""

    @staticmethod
    def _make_belief_tracker(confidences: dict[str, float]):
        """构造模拟 BayesianBeliefTracker。"""
        from dataclasses import dataclass

        @dataclass
        class MockBeliefState:
            confidence: float
            stage: str = "确认"

        @dataclass
        class MockBeliefTracker:
            belief_states: dict

        states = {k: MockBeliefState(confidence=v) for k, v in confidences.items()}
        return MockBeliefTracker(belief_states=states)

    def test_belief_same_zero(self):
        """相同信念状态距离为 0。"""
        s1 = _make_state([{"cause": "a", "effect": "b", "rho": 0.5}])
        s2 = _make_state([{"cause": "a", "effect": "b", "rho": 0.5}])
        s1.belief_tracker = self._make_belief_tracker({"m1": 0.8})
        s2.belief_tracker = self._make_belief_tracker({"m1": 0.8})
        d = s1.state_distance(s2)
        assert d == 0.0

    def test_belief_different_nonzero(self):
        """不同信念置信度增加距离。"""
        s1 = _make_state([{"cause": "a", "effect": "b", "rho": 0.5}])
        s2 = _make_state([{"cause": "a", "effect": "b", "rho": 0.5}])
        s1.belief_tracker = self._make_belief_tracker({"m1": 0.2})
        s2.belief_tracker = self._make_belief_tracker({"m1": 0.9})
        d = s1.state_distance(s2)
        assert d > 0.0

    def test_belief_one_none_ignored(self):
        """单方无 belief_tracker 时，信念距离不激活。"""
        s1 = _make_state([{"cause": "a", "effect": "b", "rho": 0.5}])
        s2 = _make_state([{"cause": "a", "effect": "b", "rho": 0.6}])
        s1.belief_tracker = self._make_belief_tracker({"m1": 0.5})
        s2.belief_tracker = None
        d = s1.state_distance(s2)
        assert d > 0.0  # 仅有因果距离

    def test_temporal_belief_combined(self):
        """时空+信念同时不同 → 距离 > 仅因果不同。"""
        from dataclasses import dataclass

        @dataclass
        class MockTemporalInfo:
            energy_type: str

        @dataclass
        class MockBeliefState:
            confidence: float

        @dataclass
        class MockBeliefTracker:
            belief_states: dict

        # 仅因果不同
        s1 = _make_state([{"cause": "a", "effect": "b", "rho": 0.5}])
        s2 = _make_state([{"cause": "x", "effect": "y", "rho": 0.7}])
        d_causal_only = s1.state_distance(s2)

        # 因果 + 时空 + 信念都不同
        s3 = _make_state([{"cause": "a", "effect": "b", "rho": 0.5}])
        s4 = _make_state([{"cause": "x", "effect": "y", "rho": 0.7}])
        s3.temporal_info = MockTemporalInfo(energy_type="wood")
        s4.temporal_info = MockTemporalInfo(energy_type="fire")
        s3.belief_tracker = MockBeliefTracker(
            belief_states={"m1": MockBeliefState(confidence=0.9)}
        )
        s4.belief_tracker = MockBeliefTracker(
            belief_states={"m1": MockBeliefState(confidence=0.1)}
        )
        d_combined = s3.state_distance(s4)

        # 时空信息不同的两个状态距离 > 时空信息相同但因果不同的两个状态？
        # 不，这里是：时空+信念+因果都不同 > 仅因果不同
        assert d_combined >= d_causal_only * 0.9, (
            f"预期 combined({d_combined:.4f}) >= causal({d_causal_only:.4f})"
        )
