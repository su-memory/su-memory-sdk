"""
P1: Energy Integration tests

Tests:
1. CategoryCausalEngine.query_with_energy_boost wired into query()
2. UnifiedInfoFactory wired into add() producing enriched metadata
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from su_memory._sys._causal_engine import CategoryCausalEngine
from su_memory._sys._unified_unit import UnifiedInfoFactory


class TestCausalEngineIntegration:
    """Verify CategoryCausalEngine can be used to re-rank by energy affinity."""

    def test_causal_engine_boost_favors_enhanced_energy(self):
        """Energy-enhanced results should rank higher."""
        engine = CategoryCausalEngine()
        engine.add_node("q", "query about growth", energy_type="wood")
        engine.add_node("fire_n", "fire node - boosted", energy_type="fire")
        engine.add_node("metal_n", "metal node - suppressed", energy_type="metal")
        engine.add_node("earth_n", "earth node - neutral", energy_type="earth")

        results = engine.query_with_energy_boost(
            query_node_id="q",
            candidates=["fire_n", "metal_n", "earth_n"],
            base_scores={"fire_n": 0.7, "metal_n": 0.7, "earth_n": 0.7}
        )

        assert len(results) == 3
        assert results[0]["node_id"] == "fire_n", (
            f"Fire (enhanced by wood) should rank #1, got {results[0]['node_id']}"
        )
        assert results[0]["affinity"] > 1.0
        # Metal is REVERSE-suppressed by wood → affinity 0.3
        assert results[2]["affinity"] < 1.0

    def test_causal_engine_same_energy_boosts(self):
        """Same energy type gets affinity 1.2."""
        engine = CategoryCausalEngine()
        engine.add_node("q", "query", energy_type="earth")
        engine.add_node("n", "earth node", energy_type="earth")

        results = engine.query_with_energy_boost(
            query_node_id="q", candidates=["n"],
            base_scores={"n": 0.5}
        )
        assert results[0]["affinity"] == 1.2

    def test_causal_engine_empty_candidates(self):
        """Empty candidate list should not crash."""
        engine = CategoryCausalEngine()
        engine.add_node("q", "query", energy_type="fire")
        results = engine.query_with_energy_boost("q", [], {})
        assert results == []


class TestUnifiedInfoIntegration:
    """Verify UnifiedInfoFactory produces correct energy labels."""

    def test_create_from_content_produces_valid_unit(self):
        """Factory creates a unit with all three layers populated."""
        factory = UnifiedInfoFactory()
        unit = factory.create_from_content(
            "spring growth development",
            stem_idx=0, branch_idx=2,
            hexagram_idx=1, energy_type=0  # wood
        )

        d = unit.to_dict()
        assert d["human"]["energy_name"] == "wood"
        assert len(d["human"]["attributes"]["direction"]) > 0
        assert len(d["human"]["attributes"]["colors"]) > 0

    def test_create_from_content_string_energy_mapping(self):
        """Verify string-to-int energy mapping for SDK integration."""
        energy_map = {"wood": 0, "fire": 1, "earth": 2, "metal": 3, "water": 4}
        for s, i in energy_map.items():
            factory = UnifiedInfoFactory()
            unit = factory.create_from_content(f"test {s}", energy_type=i)
            assert unit.to_dict()["human"]["energy_name"] == s

    def test_unit_to_dict_roundtrip(self):
        """to_dict → from_dict preserves key fields."""
        from su_memory._sys._unified_unit import UnifiedInfoUnit

        factory = UnifiedInfoFactory()
        original = factory.create_from_content("test", energy_type=2)
        restored = UnifiedInfoUnit.from_dict(original.to_dict())
        assert restored.content == original.content
        assert restored.to_dict()["human"]["energy_name"] == "earth"
