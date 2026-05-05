"""
P3: Energy Reasoning tests

Tests:
1. auto_link_by_energy() discovers and links memories by energy affinity
2. link_by_energy() creates energy-weighted causal edges
3. Energy-linked memories are discoverable via multihop
4. Energy-enhanced multihop finds connected memories through energy relations
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from su_memory._sys._energy_relations import (
    analyze_relation, get_affinity_score, RelationType,
    get_enhanced_energy, get_suppressed_energy
)


class TestEnergyRelationDiscovery:
    """Verify energy relationship analysis between memory pairs."""

    def test_wood_enhances_fire_relation(self):
        """Wood→Fire should detect as ENHANCE."""
        rel = analyze_relation("wood", "fire")
        assert rel.relation == RelationType.ENHANCE
        assert rel.strength > 1.0
        assert "enhance" in rel.description.lower()

    def test_metal_suppresses_wood_relation(self):
        """Metal→Wood should detect as SUPPRESS."""
        rel = analyze_relation("metal", "wood")
        assert rel.relation == RelationType.SUPPRESS
        assert rel.strength < 1.0

    def test_same_type_detected(self):
        """Earth→Earth should be SAME."""
        rel = analyze_relation("earth", "earth")
        assert rel.relation == RelationType.SAME
        assert rel.strength > 1.0

    def test_enhanced_energy_chain(self):
        """get_enhanced_energy follows the generation cycle."""
        assert get_enhanced_energy("wood") == "fire"
        assert get_enhanced_energy("fire") == "earth"
        assert get_enhanced_energy("earth") == "metal"
        assert get_enhanced_energy("metal") == "water"
        assert get_enhanced_energy("water") == "wood"

    def test_suppressed_energy_chain(self):
        """get_suppressed_energy follows the control cycle."""
        assert get_suppressed_energy("wood") == "earth"
        assert get_suppressed_energy("earth") == "water"
        assert get_suppressed_energy("water") == "fire"
        assert get_suppressed_energy("fire") == "metal"
        assert get_suppressed_energy("metal") == "wood"

    def test_affinity_hierarchy(self):
        """Affinity: ENHANCE(1.5) > SAME(1.2) > NEUTRAL(1.0) > SUPPRESS(0.6) > REVERSE(0.3)."""
        scores = {
            "enhance": get_affinity_score("wood", "fire"),    # 1.5
            "same": get_affinity_score("earth", "earth"),      # 1.2
            "suppress": get_affinity_score("water", "fire"),    # 0.6
            "reverse": get_affinity_score("fire", "water"),     # 0.3
        }
        assert scores["enhance"] > scores["same"] > scores["suppress"] > scores["reverse"]


class TestAutoLinkByEnergy:
    """Verify SuMemoryLitePro auto-links memories by energy type."""

    @pytest.mark.slow
    def test_auto_link_creates_energy_edges(self):
        """After adding memories, energy-linked ones should be connected."""
        from su_memory.sdk.lite_pro import SuMemoryLitePro

        pro = SuMemoryLitePro(
            enable_vector=False, enable_graph=True,
            enable_temporal=False, enable_session=False,
            enable_prediction=False, enable_explainability=False,
            storage_path="/tmp/p3_test_auto"
        )

        # Add wood memory
        wood_id = pro.add("Spring forest renewal and growth", 
                         metadata={"source": "test"})

        # Add fire memory (wood enhances fire → should be auto-linked)
        fire_id = pro.add("Summer passion and creative fire energy",
                         metadata={"source": "test"})

        # Fire should be linked as enhanced-by-wood
        assert hasattr(pro, 'auto_link_by_energy')

        count = pro.auto_link_by_energy()
        assert count > 0, f"Expected at least 1 energy link created, got {count}"

        # Fire should have wood as a parent via energy enhancement
        children = pro.get_children(wood_id)
        assert len(children) > 0, f"Wood should have energy-linked children"
        assert any(c.get('memory_id') == fire_id for c in children), \
            f"Fire should be a child of wood"

        pro.clear()

    @pytest.mark.slow
    def test_link_by_energy_manual(self):
        """link_by_energy() manually creates energy-weighted edges."""
        from su_memory.sdk.lite_pro import SuMemoryLitePro

        pro = SuMemoryLitePro(
            enable_vector=False, enable_graph=True,
            enable_temporal=False, enable_session=False,
            enable_prediction=False, enable_explainability=False,
            storage_path="/tmp/p3_test_manual"
        )

        water_id = pro.add("Winter wisdom and deep water flow",
                          metadata={"source": "test"})
        wood_id = pro.add("Spring renewal and wood growth",
                         metadata={"source": "test"})

        # Water enhances wood
        assert hasattr(pro, 'link_by_energy')
        success, weight = pro.link_by_energy(water_id, wood_id)

        assert success, "link_by_energy should succeed"
        assert weight > 1.0, f"Energy-enhanced link weight should be >1.0, got {weight}"

        # Verify edge exists
        children = pro.get_children(water_id)
        assert any(c.get('memory_id') == wood_id for c in children)

        pro.clear()

    @pytest.mark.slow
    def test_energy_multihop_follows_energy_links(self):
        """Multihop query follows energy-based causal links."""
        from su_memory.sdk.lite_pro import SuMemoryLitePro

        pro = SuMemoryLitePro(
            enable_vector=False, enable_graph=True,
            enable_temporal=False, enable_session=False,
            enable_prediction=False, enable_explainability=False,
            storage_path="/tmp/p3_test_mhop"
        )

        # Build an energy chain: water → wood → fire
        water_id = pro.add("Deep water wisdom flows through time")
        wood_id = pro.add("Ancient forests grow toward the light")
        fire_id = pro.add("Blazing inspiration ignites creation")

        # Link them by energy
        pro.link_by_energy(water_id, wood_id)  # water enhances wood
        pro.link_by_energy(wood_id, fire_id)   # wood enhances fire

        # Query from water should reach fire via multihop
        results = pro.query_multihop("wisdom", max_hops=3, top_k=5)
        result_ids = [r.get("memory_id") for r in results]

        assert len(results) > 0
        assert fire_id in result_ids, \
            f"Multihop should follow energy chain water→wood→fire, got {result_ids}"

        pro.clear()
