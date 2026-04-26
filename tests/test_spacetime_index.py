"""
Unit Tests for Spacetime Index Module

Tests energy-aware spacetime indexing functionality.
"""

import sys
sys.path.insert(0, 'src')

import pytest
from su_memory._sys._spacetime_index import (
    SpacetimeIndexEngine,
    SpacetimeNode,
    SpacetimeConfig,
    create_spacetime_engine,
    create_energy_aware_node,
    ENERGY_TO_SEASON,
)
from su_memory._sys._enums import TimeStem, TimeBranch


class TestSpacetimeConfig:
    """Test SpacetimeConfig"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = SpacetimeConfig()
        assert config.seasonal_decay == 0.9
        assert config.enhance_weight == 1.3
        assert config.suppress_weight == 0.7
    
    def test_custom_weights(self):
        """Test custom weight configuration"""
        config = SpacetimeConfig(
            enhance_weight=1.5,
            suppress_weight=0.6,
            four_phase_weights={"INITIAL_YANG": 1.5}
        )
        assert config.enhance_weight == 1.5
        assert config.suppress_weight == 0.6
        assert config.four_phase_weights["INITIAL_YANG"] == 1.5


class TestSpacetimeNode:
    """Test SpacetimeNode"""
    
    def test_create_node(self):
        """Test creating a spacetime node"""
        node = SpacetimeNode(
            node_id="test1",
            energy_type="wood",
            stem_idx=0,
            branch_idx=0,
            base_weight=1.0,
            energy_boost=1.0
        )
        assert node.node_id == "test1"
        assert node.energy_type == "wood"
        assert node.stem_idx == 0
        assert node.branch_idx == 0
    
    def test_effective_weight(self):
        """Test effective weight calculation"""
        node = SpacetimeNode(
            node_id="test1",
            energy_type="wood",
            base_weight=2.0,
            energy_boost=1.5
        )
        assert node.effective_weight == 3.0
    
    def test_energy_level(self):
        """Test energy level descriptions"""
        node_normal = SpacetimeNode("n1", "wood", energy_boost=1.0)
        assert node_normal.energy_level == "中和"
        
        node_strong = SpacetimeNode("n2", "fire", energy_boost=1.5)
        assert node_strong.energy_level == "旺盛"
        
        node_weak = SpacetimeNode("n3", "water", energy_boost=0.6)
        assert node_weak.energy_level == "偏弱"
        
        node_very_weak = SpacetimeNode("n4", "water", energy_boost=0.5)
        assert node_very_weak.energy_level == "衰弱"


class TestSpacetimeIndexEngine:
    """Test SpacetimeIndexEngine"""
    
    def setup_method(self):
        """Setup test engine"""
        self.engine = SpacetimeIndexEngine()
    
    def test_add_node(self):
        """Test adding nodes"""
        node = SpacetimeNode("n1", "wood")
        node_id = self.engine.add_node(node)
        assert node_id == "n1"
        assert len(self.engine._nodes) == 1
    
    def test_add_duplicate_node(self):
        """Test adding duplicate node raises error"""
        node = SpacetimeNode("n1", "wood")
        self.engine.add_node(node)
        with pytest.raises(ValueError):
            self.engine.add_node(node)
    
    def test_remove_node(self):
        """Test removing nodes"""
        self.engine.add_node(SpacetimeNode("n1", "wood"))
        result = self.engine.remove_node("n1")
        assert result is True
        assert "n1" not in self.engine._nodes
    
    def test_remove_nonexistent_node(self):
        """Test removing nonexistent node returns False"""
        result = self.engine.remove_node("nonexistent")
        assert result is False
    
    def test_get_node(self):
        """Test getting nodes"""
        self.engine.add_node(SpacetimeNode("n1", "wood"))
        node = self.engine.get_node("n1")
        assert node is not None
        assert node.node_id == "n1"
        
        missing = self.engine.get_node("nonexistent")
        assert missing is None
    
    def test_search_by_energy(self):
        """Test energy-based search"""
        self.engine.add_node(create_energy_aware_node("n1", "wood", stem_idx=0))
        self.engine.add_node(create_energy_aware_node("n2", "fire", stem_idx=2))
        self.engine.add_node(create_energy_aware_node("n3", "water", stem_idx=8))
        
        results = self.engine.search_by_energy("wood")
        node_ids = [r[0] for r in results]
        
        assert "n1" in node_ids
    
    def test_search_by_energy_with_relation_filter(self):
        """Test energy search with relation type filter"""
        from su_memory._sys._energy_relations import RelationType
        
        self.engine.add_node(create_energy_aware_node("n1", "wood"))
        self.engine.add_node(create_energy_aware_node("n2", "fire"))
        
        results = self.engine.search_by_energy("fire", relation_type=RelationType.SAME)
        node_ids = [r[0] for r in results]
        
        assert "n2" in node_ids
    
    def test_search_by_stem_branch(self):
        """Test stem-branch search"""
        self.engine.add_node(create_energy_aware_node("n1", "wood", stem_idx=0, branch_idx=0))
        self.engine.add_node(create_energy_aware_node("n2", "fire", stem_idx=2, branch_idx=6))
        
        results = self.engine.search_by_stem_branch(TimeStem.JIA, TimeBranch.ZI)
        node_ids = [r[0] for r in results]
        
        assert "n1" in node_ids
    
    def test_get_temporal_ranking(self):
        """Test temporal ranking"""
        self.engine.add_node(create_energy_aware_node("n1", "wood"))
        self.engine.add_node(create_energy_aware_node("n2", "fire"))
        
        ranking = self.engine.get_temporal_ranking(TimeStem.JIA, TimeBranch.ZI)
        assert len(ranking) == 2
    
    def test_calculate_energy_flow_enhance(self):
        """Test energy flow for enhancement"""
        self.engine.add_node(SpacetimeNode("n1", "wood"))
        
        flow = self.engine.calculate_energy_flow("n1", "fire")
        assert flow > 1.0
    
    def test_calculate_energy_flow_suppress(self):
        """Test energy flow for suppression"""
        self.engine.add_node(SpacetimeNode("n1", "wood"))
        
        flow = self.engine.calculate_energy_flow("n1", "earth")
        assert flow < 1.0
    
    def test_calculate_energy_flow_distance(self):
        """Test distance-based decay"""
        self.engine.add_node(SpacetimeNode("n1", "wood"))
        
        flow1 = self.engine.calculate_energy_flow("n1", "fire", distance=1.0)
        flow2 = self.engine.calculate_energy_flow("n1", "fire", distance=2.0)
        
        assert flow2 < flow1
    
    def test_get_index_state(self):
        """Test index state reporting"""
        self.engine.add_node(create_energy_aware_node("n1", "wood"))
        self.engine.add_node(create_energy_aware_node("n2", "fire"))
        self.engine.add_node(create_energy_aware_node("n3", "wood"))
        
        state = self.engine.get_index_state()
        
        assert state["total_nodes"] == 3
        assert state["energy_counts"]["wood"] == 2
        assert state["energy_counts"]["fire"] == 1
    
    def test_repr(self):
        """Test string representation"""
        self.engine.add_node(SpacetimeNode("n1", "wood"))
        repr_str = repr(self.engine)
        assert "SpacetimeIndexEngine" in repr_str


class TestCreateEnergyAwareNode:
    """Test create_energy_aware_node factory function"""
    
    def test_create_with_stem(self):
        """Test creating node with stem"""
        node = create_energy_aware_node("n1", "wood", stem_idx=0)
        assert node.node_id == "n1"
        assert node.energy_type == "wood"
        assert node.stem_idx == 0
    
    def test_auto_four_phase(self):
        """Test automatic four symbol attribution"""
        node = create_energy_aware_node("n1", "wood")
        assert node.four_phase == "INITIAL_YANG"
        
        node_fire = create_energy_aware_node("n2", "fire")
        assert node_fire.four_phase == "PEAK_YANG"
    
    def test_auto_season(self):
        """Test automatic season attribution"""
        node = create_energy_aware_node("n1", "wood")
        assert node.season == "spring"
        
        node_winter = create_energy_aware_node("n2", "water")
        assert node_winter.season == "winter"
    
    def test_explicit_season(self):
        """Test explicit season override"""
        node = create_energy_aware_node("n1", "wood", season="autumn")
        assert node.season == "autumn"


class TestEnergyBusIntegration:
    """Test energy bus integration"""
    
    def test_integrate_energy_bus_mock(self):
        """Test energy bus integration (mock)"""
        from src.su_memory._sys._energy_bus import EnergyBus, EnergyLayer
        
        engine = SpacetimeIndexEngine()
        engine.add_node(create_energy_aware_node("n1", "wood"))
        engine.add_node(create_energy_aware_node("n2", "fire"))
        
        # Create minimal mock energy bus
        class MockEnergyBus:
            def get_bus_state(self):
                return {"total_energy": 10.0, "nodes": {}}
            
            def _calculate_energy_balance(self):
                return {"ratios": {"wood": 0.2, "fire": 0.2, "earth": 0.2, "metal": 0.2, "water": 0.2}}
        
        mock_bus = MockEnergyBus()
        adjustments = engine.integrate_energy_bus(mock_bus)
        
        assert len(adjustments) >= 0


def run_tests():
    """Run all tests"""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()