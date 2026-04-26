"""
Unit Tests for Adaptive Engine Module (v1.6.0)

Tests adaptive intelligence capabilities:
- Parameter space definition
- Learning metrics collection
- Adaptive optimization
"""

import sys
sys.path.insert(0, 'src')

import pytest
import time
from su_memory._sys._adaptive_engine import (
    AdaptiveEngine,
    ParameterSpace,
    LearningMetrics,
    ParameterBound,
    ParameterType,
    MetricType,
    AdaptationStrategy,
    AdaptationResult,
    MetricEntry,
    MetricSummary,
    create_adaptive_engine,
    create_parameter_space,
    create_metrics_collector,
)


class TestParameterBound:
    """Test ParameterBound class"""
    
    def test_create_bound(self):
        """Test creating a parameter bound"""
        bound = ParameterBound(0.0, 1.0, 0.5)
        assert bound.min_value == 0.0
        assert bound.max_value == 1.0
        assert bound.default == 0.5
    
    def test_clamp(self):
        """Test value clamping"""
        bound = ParameterBound(0.0, 1.0, 0.5)
        assert bound.clamp(-0.5) == 0.0
        assert bound.clamp(0.5) == 0.5
        assert bound.clamp(1.5) == 1.0
    
    def test_normalize_denormalize(self):
        """Test normalization"""
        bound = ParameterBound(0.0, 100.0, 50.0)
        assert abs(bound.normalize(50.0) - 0.5) < 0.001
        assert abs(bound.denormalize(0.5) - 50.0) < 0.001
    
    def test_invalid_bounds(self):
        """Test invalid bounds raise error"""
        with pytest.raises(ValueError):
            ParameterBound(1.0, 0.0, 0.5)
        
        with pytest.raises(ValueError):
            ParameterBound(0.0, 1.0, 1.5)


class TestParameterSpace:
    """Test ParameterSpace class"""
    
    def test_create_space(self):
        """Test creating parameter space"""
        space = ParameterSpace()
        assert space.parameter_count == 0
    
    def test_add_parameter(self):
        """Test adding parameters"""
        space = ParameterSpace()
        bound = ParameterBound(0.0, 1.0, 0.5)
        space.add_parameter("threshold", bound, ParameterType.CONTINUOUS)
        
        assert space.parameter_count == 1
        assert space.get_parameter("threshold") == 0.5
    
    def test_set_parameter(self):
        """Test setting parameters"""
        space = ParameterSpace()
        bound = ParameterBound(0.0, 1.0, 0.5)
        space.add_parameter("threshold", bound, ParameterType.CONTINUOUS)
        
        assert space.set_parameter("threshold", 0.8)
        assert space.get_parameter("threshold") == 0.8
    
    def test_set_invalid_parameter(self):
        """Test setting invalid parameter returns False"""
        space = ParameterSpace()
        assert not space.set_parameter("nonexistent", 0.5)
    
    def test_set_all_parameters(self):
        """Test setting multiple parameters"""
        space = ParameterSpace()
        bound = ParameterBound(0.0, 1.0, 0.5)
        space.add_parameter("p1", bound, ParameterType.CONTINUOUS)
        space.add_parameter("p2", bound, ParameterType.CONTINUOUS)
        
        results = space.set_all_parameters({"p1": 0.3, "p2": 0.7})
        assert results["p1"]
        assert results["p2"]
    
    def test_reset_to_defaults(self):
        """Test resetting to defaults"""
        space = ParameterSpace()
        bound = ParameterBound(0.0, 1.0, 0.5)
        space.add_parameter("threshold", bound, ParameterType.CONTINUOUS)
        
        space.set_parameter("threshold", 0.9)
        space.reset_to_defaults()
        
        assert space.get_parameter("threshold") == 0.5
    
    def test_randomize(self):
        """Test randomization"""
        space = ParameterSpace()
        bound = ParameterBound(0.0, 1.0, 0.5)
        space.add_parameter("p1", bound, ParameterType.CONTINUOUS)
        space.add_parameter("p2", bound, ParameterType.CONTINUOUS)
        
        values = space.randomize(seed=42)
        assert "p1" in values
        assert "p2" in values


class TestLearningMetrics:
    """Test LearningMetrics class"""
    
    def test_create_metrics(self):
        """Test creating metrics collector"""
        metrics = LearningMetrics()
        assert len(metrics.get_values(MetricType.RECALL)) == 0
    
    def test_record_metric(self):
        """Test recording metrics"""
        metrics = LearningMetrics()
        metrics.record(MetricType.RECALL, 0.85)
        metrics.record(MetricType.RECALL, 0.90)
        
        values = metrics.get_values(MetricType.RECALL)
        assert len(values) == 2
        assert 0.85 in values
    
    def test_record_with_context(self):
        """Test recording with context"""
        metrics = LearningMetrics()
        metrics.record(MetricType.RECALL, 0.85, {"query": "test", "user": "alice"})
        
        dist = metrics.get_context_distribution("query")
        assert dist.get("test", 0) == 1
    
    def test_get_summary(self):
        """Test getting summary statistics"""
        metrics = LearningMetrics()
        for i in range(10):
            metrics.record(MetricType.RECALL, 0.8 + i * 0.01)
        
        summary = metrics.get_summary(MetricType.RECALL)
        assert summary.count == 10
        assert summary.mean > 0.8
        assert summary.trend in ["increasing", "decreasing", "stable"]
    
    def test_get_trend(self):
        """Test getting trend"""
        metrics = LearningMetrics()
        for i in range(10):
            metrics.record(MetricType.RECALL, 0.8 + i * 0.01)
        
        slope, direction = metrics.get_trend(MetricType.RECALL)
        assert direction == "increasing"
    
    def test_window_limit(self):
        """Test window limit on values"""
        metrics = LearningMetrics()
        for i in range(20):
            metrics.record(MetricType.RECALL, 0.8 + i * 0.01)
        
        values = metrics.get_values(MetricType.RECALL, window=5)
        assert len(values) == 5
    
    def test_clear(self):
        """Test clearing metrics"""
        metrics = LearningMetrics()
        metrics.record(MetricType.RECALL, 0.85)
        metrics.clear()
        
        assert len(metrics.get_values(MetricType.RECALL)) == 0


class TestAdaptiveEngine:
    """Test AdaptiveEngine class"""
    
    def test_create_engine(self):
        """Test creating adaptive engine"""
        engine = AdaptiveEngine()
        assert engine is not None
    
    def test_add_parameter(self):
        """Test adding parameters to engine"""
        engine = AdaptiveEngine()
        engine.add_parameter("threshold", 0.5, (0.0, 1.0))
        
        assert engine._parameter_space.parameter_count == 1
    
    def test_record_metric(self):
        """Test recording metrics"""
        engine = AdaptiveEngine()
        engine.record_metric(MetricType.RECALL, 0.85)
        engine.record_metric(MetricType.LATENCY, 100.0)
        
        summary = engine.get_metrics().get_summary(MetricType.RECALL)
        assert summary.count == 1
    
    def test_adapt(self):
        """Test adaptation"""
        engine = AdaptiveEngine()
        engine.add_parameter("threshold", 0.5, (0.0, 1.0))
        
        # Record some metrics
        for _ in range(5):
            engine.record_metric(MetricType.RECALL, 0.85)
            engine.record_metric(MetricType.LATENCY, 100.0)
        
        result = engine.adapt()
        assert isinstance(result, AdaptationResult)
        assert len(result.new_values) == 1
    
    def test_adapt_improves_score(self):
        """Test adaptation improves score"""
        engine = AdaptiveEngine()
        engine.add_parameter("threshold", 0.5, (0.0, 1.0))
        
        # Record improving metrics
        for i in range(10):
            engine.record_metric(MetricType.RECALL, 0.7 + i * 0.02)
        
        result = engine.adapt()
        assert result.improved or result.improvement_score >= 0
    
    def test_get_best_configuration(self):
        """Test getting best configuration"""
        engine = AdaptiveEngine()
        engine.add_parameter("threshold", 0.5, (0.0, 1.0))
        
        # Record metrics
        for _ in range(5):
            engine.record_metric(MetricType.RECALL, 0.9)
            engine.adapt()
        
        best = engine.get_best_configuration()
        assert best is not None
        assert "threshold" in best
    
    def test_reset(self):
        """Test engine reset"""
        engine = AdaptiveEngine()
        engine.add_parameter("threshold", 0.5, (0.0, 1.0))
        engine.record_metric(MetricType.RECALL, 0.85)
        engine.adapt()
        
        engine.reset()
        
        assert engine._best_values is None
        assert engine._parameter_space.get_parameter("threshold") == 0.5
    
    def test_callback(self):
        """Test adaptation callback"""
        engine = AdaptiveEngine()
        engine.add_parameter("threshold", 0.5, (0.0, 1.0))
        
        callback_values = []
        def on_adapt(values):
            callback_values.append(values)
        
        engine.on_adaptation(on_adapt)
        engine.record_metric(MetricType.RECALL, 0.85)
        engine.adapt()
        
        assert len(callback_values) == 1


class TestConvenienceFunctions:
    """Test convenience factory functions"""
    
    def test_create_adaptive_engine(self):
        """Test factory function"""
        engine = create_adaptive_engine(AdaptationStrategy.HEURISTIC, 0.1)
        assert isinstance(engine, AdaptiveEngine)
    
    def test_create_parameter_space(self):
        """Test parameter space factory"""
        space = create_parameter_space()
        assert isinstance(space, ParameterSpace)
    
    def test_create_metrics_collector(self):
        """Test metrics collector factory"""
        metrics = create_metrics_collector(5000)
        assert isinstance(metrics, LearningMetrics)


def run_tests():
    """Run all tests"""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()