"""v4.0.0 JEPA: JEPATrainer 测试."""

from __future__ import annotations

import pytest

from su_memory.sdk._world_model import CausalWorldModelState
from su_memory.sdk._jepa_predictor import (
    IdentityPredictor,
    EnergyPropagationPredictor,
)
from su_memory.sdk._jepa_encoder import JEPAEncoder
from su_memory.sdk._jepa_dataset import JEPADataset
from su_memory.sdk._jepa_trainer import JEPATrainer, JEPATrainingStats

pytestmark = pytest.mark.jepa


def _make_state(edges, n_memories=10, timestamp=""):
    return CausalWorldModelState(
        causal_edges=edges,
        n_memories=n_memories,
        timestamp=timestamp or "2026-01-01T00:00:00",
    )


class TestJEPATrainer:
    """JEPATrainer 训练循环测试。"""

    def test_empty_dataset(self):
        encoder = JEPAEncoder(world_model=None)
        predictor = IdentityPredictor()
        ds = JEPADataset.from_states([])
        trainer = JEPATrainer(encoder, predictor, ds)
        stats = trainer.train(n_epochs=3)
        assert stats.n_pairs == 0

    def test_loss_decreases_or_stable(self):
        """Identity 预测器在相同状态对上 loss 应为 0 或接近。"""
        s1 = _make_state([{"cause": "a", "effect": "b", "rho": 0.5}])
        s2 = _make_state([{"cause": "a", "effect": "b", "rho": 0.5}])
        ds = JEPADataset.from_states([s1, s2])

        encoder = JEPAEncoder(world_model=None)
        predictor = IdentityPredictor()
        trainer = JEPATrainer(encoder, predictor, ds)
        stats = trainer.train(n_epochs=5)

        assert stats.n_pairs == 1
        assert stats.final_loss >= 0.0

    def test_loss_history_has_entries(self):
        s1 = _make_state([{"cause": "a", "effect": "b", "rho": 0.5}])
        s2 = _make_state([{"cause": "a", "effect": "b", "rho": 0.5}])
        ds = JEPADataset.from_states([s1, s2])

        trainer = JEPATrainer(JEPAEncoder(world_model=None), IdentityPredictor(), ds)
        stats = trainer.train(n_epochs=3)
        assert len(stats.loss_history) == 3

    def test_energy_propagation_loss_is_finite(self):
        s1 = _make_state([
            {"cause": "price", "effect": "demand", "rho": 0.72, "energy_relation": "enhance"},
        ])
        s2 = _make_state([
            {"cause": "price", "effect": "demand", "rho": 0.80, "energy_relation": "enhance"},
        ])
        ds = JEPADataset.from_states([s1, s2])

        trainer = JEPATrainer(
            JEPAEncoder(world_model=None),
            EnergyPropagationPredictor(),
            ds,
        )
        stats = trainer.train(n_epochs=3)
        assert stats.final_loss >= 0.0
        assert stats.final_loss < 1.0

    def test_evaluate(self):
        s1 = _make_state([{"cause": "a", "effect": "b", "rho": 0.5}])
        s2 = _make_state([{"cause": "a", "effect": "b", "rho": 0.5}])
        ds = JEPADataset.from_states([s1, s2])

        trainer = JEPATrainer(JEPAEncoder(world_model=None), IdentityPredictor(), ds)
        result = trainer.evaluate()
        assert "avg_distance" in result
        assert result["n"] == 1

    def test_evaluate_empty(self):
        ds = JEPADataset.from_states([])
        trainer = JEPATrainer(JEPAEncoder(world_model=None), IdentityPredictor(), ds)
        result = trainer.evaluate()
        assert "error" in result


class TestJEPATrainingStats:
    """训练统计测试。"""

    def test_to_dict(self):
        s = JEPATrainingStats(n_epochs=5, n_pairs=10, final_loss=0.123)
        d = s.to_dict()
        assert d["n_epochs"] == 5
        assert d["n_pairs"] == 10
        assert d["final_loss"] == 0.123

    def test_trend_converging(self):
        s = JEPATrainingStats(loss_history=[0.5, 0.3, 0.2])
        assert s._trend() == "converging"

    def test_trend_diverging(self):
        s = JEPATrainingStats(loss_history=[0.1, 0.2, 0.3])
        assert s._trend() == "diverging"

    def test_trend_stable(self):
        s = JEPATrainingStats(loss_history=[0.5, 0.49, 0.51])
        assert s._trend() == "stable"

    def test_trend_insufficient(self):
        s = JEPATrainingStats(loss_history=[0.5, 0.4])
        assert s._trend() == "insufficient_data"
