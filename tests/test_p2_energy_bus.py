"""
P2: Energy Bus + Ecology tests

Tests:
1. EnergyBus propagation activates related energy nodes
2. SuMemoryLitePro.add() registers nodes in EnergyBus
3. analyze_memory_ecology() returns energy balance analysis
4. Energy propagation influences search weights in query()
"""
import pytest
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from su_memory._sys._energy_bus import (
    EnergyBus, EnergyNode, EnergyLayer, PropagationConfig,
    EnergyState as BusEnergyState
)
from su_memory._sys._energy_core import EnergyCore
from su_memory._sys._energy_relations import (
    get_affinity_score, analyze_balance, analyze_relation,
    RelationType
)


class TestEnergyBusPropagation:
    """Verify EnergyBus network and propagation mechanics."""

    def test_create_five_elements_network(self):
        """Baseline five-element network is created with proper connections."""
        bus = EnergyBus()
        nodes = bus.create_five_elements_nodes()

        assert len(nodes) == 5
        for etype in ("wood", "fire", "earth", "metal", "water"):
            assert etype in nodes
            assert isinstance(nodes[etype], EnergyNode)

        # Wood should have channels to fire (enhance) and earth (suppress)
        wood = nodes["wood"]
        out_channels = bus.get_outgoing_channels(wood.node_id)
        assert len(out_channels) >= 2, f"Expected >=2 channels from wood, got {len(out_channels)}"

    def test_propagate_energy_activates_target(self):
        """Propagating from one node increases intensity of related nodes."""
        bus = EnergyBus()
        bus.create_five_elements_nodes()

        # Add a test node and connect to wood
        test_node = EnergyNode(
            node_id="test_mem", energy_type="fire",
            layer=EnergyLayer.FIVE_ELEMENTS, intensity=1.0
        )
        bus.add_node(test_node)

        # Propagate from wood (wood enhances fire)
        signals = bus.propagate_energy("element_wood", delta=0.5, max_hops=2)

        assert len(signals) > 0, "Propagation should generate signals"
        # Fire should receive enhancement from wood
        fire_signals = [s for s in signals if s.energy_type == "fire"]
        assert len(fire_signals) > 0, "Fire should receive propagated energy"

    def test_propagate_suppress_reduces_intensity(self):
        """Suppression relation reduces target intensity."""
        bus = EnergyBus()
        bus.create_five_elements_nodes()

        # Wood suppresses earth
        earth_node = bus.get_node("element_earth")
        original_intensity = earth_node.intensity

        bus.propagate_energy("element_wood", delta=1.0, max_hops=1)
        earth_node = bus.get_node("element_earth")

        assert earth_node.intensity <= original_intensity, (
            f"Earth intensity should decrease (wood suppresses earth), "
            f"was {original_intensity}, now {earth_node.intensity}"
        )

    def test_get_bus_state_balance(self):
        """Bus state reports energy balance."""
        bus = EnergyBus()
        bus.create_five_elements_nodes()

        state = bus.get_bus_state()
        assert "energy_balance" in state
        assert "balanced" in state["energy_balance"]
        assert "ratios" in state["energy_balance"]
        assert len(state["energy_balance"]["ratios"]) == 5


class TestEnergyEcology:
    """Verify energy balance analysis (pattern detection)."""

    def test_balanced_distribution(self):
        """Even distribution should report balanced."""
        result = analyze_balance({
            "wood": 0.2, "fire": 0.2, "earth": 0.2, "metal": 0.2, "water": 0.2
        })
        assert result["status"] == "balanced"

    def test_concentrated_distribution(self):
        """Concentration > 60% should report concentrated."""
        result = analyze_balance({
            "wood": 0.7, "fire": 0.1, "earth": 0.1, "metal": 0.05, "water": 0.05
        })
        assert result["status"] in ("concentrated", "imbalanced")

    def test_affinity_score_matrix(self):
        """Verify the affinity scoring used in re-ranking."""
        # Enhance (bidirectional): 1.5
        assert get_affinity_score("wood", "fire") == 1.5  # wood→fire
        assert get_affinity_score("fire", "wood") == 1.5  # fire→wood (enhanced_by)
        # Same: 1.2
        assert get_affinity_score("earth", "earth") == 1.2
        # Suppress: 0.6
        assert get_affinity_score("water", "fire") == 0.6  # water→fire
        # Reverse: 0.3 (suppressed_by)
        assert get_affinity_score("fire", "water") == 0.3  # fire→water


class TestEnergyBusIntegration:
    """Verify EnergyBus is wired into SuMemoryLitePro."""

    @pytest.mark.slow
    def test_add_registers_in_energy_bus(self):
        """Each add() should register node in EnergyBus."""
        from su_memory.sdk.lite_pro import SuMemoryLitePro

        pro = SuMemoryLitePro(
            enable_vector=False, enable_graph=False,
            enable_temporal=False, enable_session=False,
            enable_prediction=False, enable_explainability=False,
            storage_path="/tmp/p2_test_eb"
        )

        # Verify energy_bus is initialized
        assert hasattr(pro, '_energy_bus'), "EnergyBus not initialized"
        assert pro._energy_bus is not None

        # Add a memory - should be registered
        mid = pro.add("Spring renewal and growth energy")
        assert mid.startswith("mem_")

        # Verify node appears in bus
        node = pro._energy_bus.get_node(mid)
        assert node is not None, f"Memory {mid} not found in EnergyBus"
        assert node.energy_type in ("wood", "fire", "earth", "metal", "water")

        pro.clear()

    @pytest.mark.slow
    def test_analyze_memory_ecology(self):
        """analyze_memory_ecology() should return balance report."""
        from su_memory.sdk.lite_pro import SuMemoryLitePro

        pro = SuMemoryLitePro(
            enable_vector=False, enable_graph=False,
            enable_temporal=False, enable_session=False,
            enable_prediction=False, enable_explainability=False,
            storage_path="/tmp/p2_test_eco"
        )

        pro.add("Spring renewal and growth energy")     # wood
        pro.add("Summer passion and creative fire")      # fire
        pro.add("Central stability and grounding")       # earth
        pro.add("Autumn harvest and metal refinement")   # metal
        pro.add("Winter wisdom and water flow")          # water

        # Should have analyze_memory_ecology method
        assert hasattr(pro, 'analyze_memory_ecology')

        report = pro.analyze_memory_ecology()

        assert "balance" in report
        assert "distribution" in report
        assert "dominant" in report
        assert "node_count" in report

        pro.clear()

    @pytest.mark.slow
    def test_energy_bus_influences_query_ranking(self):
        """Query results should be influenced by energy bus propagation."""
        from su_memory.sdk.lite_pro import SuMemoryLitePro

        pro = SuMemoryLitePro(
            enable_vector=False, enable_graph=False,
            enable_temporal=False, enable_session=False,
            enable_prediction=False, enable_explainability=False,
            storage_path="/tmp/p2_test_rank"
        )

        # Add wood-related memory
        wood_id = pro.add("Forest spring renewal timber growth green east")

        # Add metal-related memory  
        metal_id = pro.add("Metal harvest autumn cutting west white")

        # Query with a wood-biased term
        results = pro.query("growth spring green", top_k=2)

        assert len(results) >= 1
        # Wood-related memory should rank higher for wood query
        assert results[0]["memory_id"] == wood_id, (
            f"Wood memory should rank #1 for wood query, "
            f"got {results[0].get('content', '?')[:50]}"
        )

        pro.clear()
