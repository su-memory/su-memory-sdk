"""v4.0.0 JEPA: JEPADataset 测试."""

from __future__ import annotations

import pytest

from su_memory.sdk._world_model import CausalWorldModelState

pytestmark = pytest.mark.jepa


def _make_state(
    edges: list[dict] | None = None,
    n_memories: int = 10,
    timestamp: str = "",
) -> CausalWorldModelState:
    """快速构造 CausalWorldModelState。"""
    return CausalWorldModelState(
        causal_edges=edges or [],
        n_memories=n_memories,
        timestamp=timestamp or "2026-01-01T00:00:00",
    )


class TestJEPADatasetFromStates:
    """from_states() 测试。"""

    def test_empty_input(self):
        from su_memory.sdk._jepa_dataset import JEPADataset

        ds = JEPADataset.from_states([])
        assert ds.n_states == 0
        assert ds.n_pairs == 0
        assert len(ds) == 0

    def test_single_state_no_pairs(self):
        from su_memory.sdk._jepa_dataset import JEPADataset

        s1 = _make_state([
            {"cause": "price", "effect": "demand", "rho": 0.72},
        ])
        ds = JEPADataset.from_states([s1])
        assert ds.n_states == 1
        assert ds.n_pairs == 0

    def test_two_states_one_pair(self):
        from su_memory.sdk._jepa_dataset import JEPADataset

        s1 = _make_state([
            {"cause": "price", "effect": "demand", "rho": 0.72},
        ], timestamp="2026-01-01T00:00:00")
        s2 = _make_state([
            {"cause": "price", "effect": "demand", "rho": 0.65},
            {"cause": "supply", "effect": "price", "rho": 0.58},
        ], timestamp="2026-01-02T00:00:00")

        ds = JEPADataset.from_states([s1, s2])
        assert ds.n_states == 2
        assert ds.n_pairs == 1
        assert len(ds.pairs) == 1
        s_t, s_t1 = ds.pairs[0]
        assert s_t is s1
        assert s_t1 is s2

    def test_multiple_states_correct_pair_count(self):
        from su_memory.sdk._jepa_dataset import JEPADataset

        states = []
        for i in range(5):
            s = _make_state([
                {"cause": f"x{i}", "effect": f"y{i}", "rho": 0.5 + i * 0.1},
            ], timestamp=f"2026-01-0{i+1}T00:00:00")
            states.append(s)

        ds = JEPADataset.from_states(states)
        assert ds.n_states == 5
        assert ds.n_pairs == 4  # 5 states → 4 pairs

    def test_filters_below_min_memories(self):
        from su_memory.sdk._jepa_dataset import JEPADataset

        s1 = _make_state([
            {"cause": "price", "effect": "demand", "rho": 0.72},
        ], n_memories=10)
        s2 = _make_state([
            {"cause": "supply", "effect": "price", "rho": 0.58},
        ], n_memories=2)  # < min_memories_per_window (3)

        ds = JEPADataset.from_states([s1, s2], min_memories_per_window=3)
        # s2 的 n_memories=2 < 3，不产生 pair
        assert ds.n_pairs == 0

    def test_distance_stats(self):
        from su_memory.sdk._jepa_dataset import JEPADataset

        s1 = _make_state([
            {"cause": "price", "effect": "demand", "rho": 0.72},
        ], timestamp="2026-01-01T00:00:00")
        s2 = _make_state([
            {"cause": "price", "effect": "demand", "rho": 0.72},
        ], timestamp="2026-01-02T00:00:00")
        s3 = _make_state([
            {"cause": "supply", "effect": "demand", "rho": 0.31},
        ], timestamp="2026-01-03T00:00:00")

        ds = JEPADataset.from_states([s1, s2, s3])
        assert ds.n_pairs == 2
        assert ds.avg_distance >= 0.0
        assert ds.min_distance >= 0.0
        assert ds.max_distance <= 1.0

    def test_to_dict(self):
        from su_memory.sdk._jepa_dataset import JEPADataset

        s1 = _make_state([{"cause": "a", "effect": "b", "rho": 0.5}])
        s2 = _make_state([{"cause": "a", "effect": "b", "rho": 0.5}])
        ds = JEPADataset.from_states([s1, s2])
        d = ds.to_dict()
        assert d["n_states"] == 2
        assert d["n_pairs"] == 1
        assert "avg_distance" in d

    def test_iterate_pairs(self):
        from su_memory.sdk._jepa_dataset import JEPADataset

        states = [
            _make_state([{"cause": f"x{i}", "effect": "y", "rho": 0.5}])
            for i in range(3)
        ]
        ds = JEPADataset.from_states(states)
        pairs = list(ds)
        assert len(pairs) == 2
