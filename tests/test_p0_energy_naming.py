"""
P0-1: Test energy boost naming compatibility

This test proves the English/Chinese naming mismatch bug:
- lite_pro._infer_energy() returns English values ("earth", "wood", ...)
- SpacetimeMultihopEngine._calculate_energy_boost() uses Chinese keys
- Result: ALL energy_boost values are 1.0 (no-op)

Fix: Unify all energy type strings to English in SDK layer.
"""
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from su_memory.sdk.spacetime_multihop import SpacetimeMultihopEngine


class TestEnergyNamingCompatibility:
    """Verify energy boost actually works with English energy types."""

    def test_energy_boost_wood_enhances_fire(self):
        """
        Wood (wood) generates Fire (fire) in the enhancement cycle.
        A memory with energy_type="fire" should receive boost > 1.0
        when current energy is "wood".
        """
        engine = SpacetimeMultihopEngine()

        # Simulate what lite_pro._infer_energy() returns
        boost = engine._calculate_energy_boost(
            memory_energy="fire",   # English - from lite_pro._infer_energy
            current_energy="wood"   # English - from lite_pro._infer_energy
        )
        # Wood enhances Fire -> expected boost > 1.0
        assert boost > 1.0, (
            f"FAIL: Wood should enhance Fire, but boost={boost} (expected >1.0). "
            f"This confirms the English/Chinese naming bug."
        )

    def test_energy_boost_water_suppresses_fire(self):
        """Water suppresses Fire -> boost should be < 1.0"""
        engine = SpacetimeMultihopEngine()
        boost = engine._calculate_energy_boost(
            memory_energy="fire",
            current_energy="water"
        )
        assert boost < 1.0, (
            f"FAIL: Water should suppress Fire, but boost={boost} (expected <1.0). "
            f"English/Chinese naming bug confirmed."
        )

    def test_energy_boost_same_type(self):
        """Same energy type -> boost should be > 1.0 (slight affinity)"""
        engine = SpacetimeMultihopEngine()
        boost = engine._calculate_energy_boost(
            memory_energy="earth",
            current_energy="earth"
        )
        assert boost > 1.0, (
            f"FAIL: Same type should boost slightly, but boost={boost} (expected >1.0)"
        )

    def test_energy_boost_neutral(self):
        """Unrelated types -> boost should be 1.0 (neutral).
        
        Note: In the five-category system, all 10 unique pairs have either
        enhancement or suppression relations. Truly neutral only occurs
        when one energy type is not in the mapping (e.g., unknown types).
        """
        engine = SpacetimeMultihopEngine()
        # Unknown/unsupported energy type → neutral
        boost = engine._calculate_energy_boost(
            memory_energy="unknown_type",
            current_energy="earth"
        )
        assert boost == 1.0, (
            f"Unknown type should be neutral, but boost={boost}"
        )

    def test_all_enhance_pairs(self):
        """Verify all five enhancement pairs work."""
        engine = SpacetimeMultihopEngine()
        enhance_pairs = [
            ("wood", "fire"),   # wood enhances fire
            ("fire", "earth"),  # fire enhances earth
            ("earth", "metal"), # earth enhances metal
            ("metal", "water"), # metal enhances water
            ("water", "wood"),  # water enhances wood
        ]
        for current, memory in enhance_pairs:
            boost = engine._calculate_energy_boost(
                memory_energy=memory,
                current_energy=current
            )
            assert boost > 1.0, (
                f"FAIL: {current} should enhance {memory}, but boost={boost}"
            )

    def test_all_suppress_pairs(self):
        """Verify all five suppression pairs work."""
        engine = SpacetimeMultihopEngine()
        suppress_pairs = [
            ("wood", "earth"),   # wood suppresses earth
            ("earth", "water"),  # earth suppresses water
            ("water", "fire"),   # water suppresses fire
            ("fire", "metal"),   # fire suppresses metal
            ("metal", "wood"),   # metal suppresses wood
        ]
        for current, memory in suppress_pairs:
            boost = engine._calculate_energy_boost(
                memory_energy=memory,
                current_energy=current
            )
            assert boost < 1.0, (
                f"FAIL: {current} should suppress {memory}, but boost={boost}"
            )

    def test_infer_energy_type_returns_english(self):
        """_infer_energy_type must return English values, not Chinese."""
        engine = SpacetimeMultihopEngine()
        valid_english = {"wood", "fire", "earth", "metal", "water"}
        # Test with content that has clear keywords
        result = engine._infer_energy_type("春天生长发展绿色")
        assert result in valid_english, (
            f"FAIL: _infer_energy_type returned '{result}' - must be English"
            f" (one of {valid_english})"
        )
        # Test fallback
        result = engine._infer_energy_type("")
        assert result in valid_english, (
            f"FAIL: fallback value '{result}' must be English"
        )

    def test_dataclass_default_is_english(self):
        """SpacetimeHopResult default energy_type must be English."""
        from su_memory.sdk.spacetime_multihop import SpacetimeHopResult
        r = SpacetimeHopResult(node_id="test", content="test", score=1.0)
        valid_english = {"wood", "fire", "earth", "metal", "water"}
        assert r.energy_type in valid_english, (
            f"FAIL: Default energy_type='{r.energy_type}' is not English"
        )
