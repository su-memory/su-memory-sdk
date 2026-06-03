"""v4.0.0 JEPA: JEPAEncoder 测试."""

from __future__ import annotations

import numpy as np
import pytest

from su_memory.sdk._jepa_encoder import JEPAEncoder
from su_memory.sdk._world_model import CausalWorldModelState, MCIWorldModel

pytestmark = pytest.mark.jepa


class TestJEPAEncoder:
    """Encoder 核心功能测试。"""

    def test_encode_invalid_world_model(self):
        encoder = JEPAEncoder(world_model=None)
        with pytest.raises(RuntimeError):
            encoder.encode([])

    def test_encode_empty_memories(self):
        wm = MCIWorldModel()
        encoder = JEPAEncoder(wm)
        state = encoder.encode([])
        assert isinstance(state, CausalWorldModelState)
        assert len(state.causal_edges) == 0

    def test_encode_count_increments(self):
        wm = MCIWorldModel()
        encoder = JEPAEncoder(wm)
        encoder.encode([])
        encoder.encode([])
        assert encoder.encode_count == 2

    def test_differentiable_defaults_false(self):
        encoder = JEPAEncoder(world_model=None)
        assert not encoder.is_differentiable

    def test_to_graph_tensors_empty(self):
        encoder = JEPAEncoder(world_model=None)
        state = CausalWorldModelState.empty()
        adj, node_feat, edge_feat = encoder.to_graph_tensors(state)
        assert adj.shape == (0, 0)
        assert node_feat.shape == (0, 8)
        assert edge_feat is None

    def test_to_graph_tensors_with_edges(self):
        encoder = JEPAEncoder(world_model=None)
        state = CausalWorldModelState(
            causal_edges=[
                {"cause": "price", "effect": "demand", "rho": 0.72, "confidence": 0.85, "bayes_factor": 3.2},
                {"cause": "supply", "effect": "price", "rho": 0.58, "confidence": 0.70},
            ],
        )
        adj, node_feat, edge_feat = encoder.to_graph_tensors(state)
        assert adj.shape == (3, 3)
        assert node_feat.shape == (3, 8)
        assert edge_feat is not None
        assert edge_feat.shape == (2, 3)
        assert edge_feat[0, 0] == 0.72
        assert edge_feat[0, 1] == 0.85
        assert edge_feat[0, 2] == 3.2

    def test_from_graph_tensors_roundtrip(self):
        encoder = JEPAEncoder(world_model=None)
        state = CausalWorldModelState(
            causal_edges=[
                {"cause": "price", "effect": "demand", "rho": 0.72, "confidence": 0.85},
                {"cause": "supply", "effect": "price", "rho": 0.58, "confidence": 0.70},
            ],
        )
        adj, node_feat, edge_feat = encoder.to_graph_tensors(state)
        # 重建
        reconstructed = encoder.from_graph_tensors(
            adj, ["price", "demand", "supply"], edge_feat,
        )
        assert len(reconstructed.causal_edges) == 2
        # 验证边内容
        edges_by_cause = {
            e["cause"]: e for e in reconstructed.causal_edges
        }
        assert "price" in edges_by_cause
        assert "supply" in edges_by_cause

    def test_from_graph_tensors_no_edges_below_threshold(self):
        encoder = JEPAEncoder(world_model=None)
        adj = np.array([[0.0, 0.005], [0.008, 0.0]], dtype=np.float32)  # 低于 0.01 阈值
        state = encoder.from_graph_tensors(adj, ["a", "b"])
        assert len(state.causal_edges) == 0  # 过滤掉弱边
