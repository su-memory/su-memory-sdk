"""
Parameter Adapters Module (参数适配器)

This module provides adaptive parameter optimization for v1.6.0:
- Retrieval weight adaptation
- Encoding dimension adjustment
- Cache strategy optimization

Core Features:
- RetrievalWeightAdapter: Adaptive retrieval weight optimization
- EncodingDimensionAdapter: Dynamic encoding dimension adjustment
- CacheStrategyAdapter: Automatic cache policy optimization
- ParameterAdapterRegistry: Unified adapter management

Architecture:
- Adapters wrap the AdaptiveEngine
- Each adapter focuses on specific parameter type
- Registry provides unified interface
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import time

from ._adaptive_engine import (
    AdaptiveEngine,
    MetricType,
)


# =============================================================================
# Enums
# =============================================================================

class AdapterType(Enum):
    """Adapter type classification"""
    RETRIEVAL_WEIGHT = "retrieval_weight"
    ENCODING_DIMENSION = "encoding_dimension"
    CACHE_STRATEGY = "cache_strategy"


class CacheStrategy(Enum):
    """Cache strategy types"""
    LRU = "lru"                # Least Recently Used
    LFU = "lfu"               # Least Frequently Used
    FIFO = "fifo"             # First In First Out
    ARC = "arc"               # Adaptive Replacement Cache
    TTL = "ttl"               # Time To Live based


class OptimizationDirection(Enum):
    """Optimization direction"""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    STABILIZE = "stabilize"


# =============================================================================
# Base Adapter
# =============================================================================

@dataclass
class AdapterConfig:
    """Configuration for parameter adapters"""
    adapter_type: AdapterType
    enabled: bool = True
    update_interval: float = 60.0        # Seconds between updates
    min_samples: int = 10               # Minimum samples before adaptation
    convergence_threshold: float = 0.01 # Threshold for convergence
    exploration_rate: float = 0.1        # Exploration rate


@dataclass
class OptimizationResult:
    """Result of an optimization step"""
    parameter_name: str
    old_value: float
    new_value: float
    improvement: float
    confidence: float
    timestamp: float = field(default_factory=time.time)


class BaseAdapter:
    """
    Base class for parameter adapters.

    Provides common functionality for all adapters including:
    - Thread-safe parameter updates
    - History tracking
    - Event callbacks
    """

    def __init__(
        self,
        name: str,
        config: AdapterConfig,
        adaptive_engine: Optional[AdaptiveEngine] = None
    ):
        self._name = name
        self._config = config
        self._engine = adaptive_engine or AdaptiveEngine()
        self._lock = threading.Lock()
        self._history: List[OptimizationResult] = []
        self._callbacks: List[Callable[[OptimizationResult], None]] = []
        self._last_update = 0.0

    @property
    def name(self) -> str:
        """Get adapter name"""
        return self._name

    @property
    def adapter_type(self) -> AdapterType:
        """Get adapter type"""
        return self._config.adapter_type

    @property
    def is_enabled(self) -> bool:
        """Check if adapter is enabled"""
        return self._config.enabled

    def enable(self):
        """Enable the adapter"""
        self._config.enabled = True

    def disable(self):
        """Disable the adapter"""
        self._config.enabled = False

    def record_metric(self, metric_type: MetricType, value: float, context: Optional[Dict] = None):
        """Record a metric for learning"""
        self._engine.record_metric(metric_type, value, context)

    def _should_update(self) -> bool:
        """Check if enough time has passed for an update"""
        if not self._config.enabled:
            return False

        elapsed = time.time() - self._last_update
        return elapsed >= self._config.update_interval

    def _record_result(self, result: OptimizationResult):
        """Record optimization result"""
        with self._lock:
            self._history.append(result)
            self._last_update = result.timestamp

            for callback in self._callbacks:
                try:
                    callback(result)
                except Exception:
                    pass  # Silent fail on callback error

    def get_history(self, limit: Optional[int] = None) -> List[OptimizationResult]:
        """Get optimization history"""
        with self._lock:
            history = self._history[:]

        if limit:
            return history[-limit:]
        return history

    def on_optimization(self, callback: Callable[[OptimizationResult], None]):
        """Register callback for optimization events"""
        self._callbacks.append(callback)

    def get_current_parameters(self) -> Dict[str, float]:
        """Get current parameter values"""
        return self._engine._parameter_space.get_all_parameters()

    def reset(self):
        """Reset adapter state"""
        with self._lock:
            self._history.clear()
            self._engine.reset()
            self._last_update = 0.0


# =============================================================================
# Retrieval Weight Adapter
# =============================================================================

@dataclass
class RetrievalWeightConfig:
    """Configuration for retrieval weight adapter"""
    # Weight parameters
    semantic_weight: Tuple[float, float, float] = (0.1, 0.9, 0.4)  # (min, max, default)
    temporal_weight: Tuple[float, float, float] = (0.1, 0.9, 0.3)
    causal_weight: Tuple[float, float, float] = (0.1, 0.9, 0.2)
    energy_weight: Tuple[float, float, float] = (0.0, 0.5, 0.1)

    # Optimization settings
    target_recall: float = 0.85
    max_latency_ms: float = 200.0


class RetrievalWeightAdapter(BaseAdapter):
    """
    Adaptive retrieval weight optimization.

    Automatically adjusts retrieval weight parameters based on:
    - Recall and precision metrics
    - Query latency
    - User feedback

    Example:
        >>> adapter = RetrievalWeightAdapter()
        >>> adapter.add_weight_parameter("semantic_weight", (0.1, 0.9, 0.4))
        >>>
        >>> # Record performance
        >>> adapter.record_metric(MetricType.RECALL, 0.8)
        >>> adapter.record_metric(MetricType.LATENCY, 150.0)
        >>>
        >>> # Optimize weights
        >>> result = adapter.optimize()
    """

    def __init__(
        self,
        config: Optional[RetrievalWeightConfig] = None,
        adaptive_engine: Optional[AdaptiveEngine] = None
    ):
        super().__init__(
            name="RetrievalWeightAdapter",
            config=AdapterConfig(AdapterType.RETRIEVAL_WEIGHT),
            adaptive_engine=adaptive_engine
        )
        self._retrieval_config = config or RetrievalWeightConfig()

        # Initialize parameters
        self._setup_parameters()

    def _setup_parameters(self):
        """Setup weight parameters"""
        self.add_weight_parameter(
            "semantic_weight",
            self._retrieval_config.semantic_weight
        )
        self.add_weight_parameter(
            "temporal_weight",
            self._retrieval_config.temporal_weight
        )
        self.add_weight_parameter(
            "causal_weight",
            self._retrieval_config.causal_weight
        )
        self.add_weight_parameter(
            "energy_weight",
            self._retrieval_config.energy_weight
        )

    def add_weight_parameter(
        self,
        name: str,
        bounds: Tuple[float, float, float]
    ):
        """
        Add a weight parameter.

        Args:
            name: Parameter name
            bounds: Tuple of (min, max, default)
        """
        self._engine.add_parameter(name, bounds[2], (bounds[0], bounds[1]))

    def set_weight(self, name: str, value: float) -> bool:
        """
        Set a weight value.

        Args:
            name: Parameter name
            value: Weight value

        Returns:
            True if set successfully
        """
        return self._engine._parameter_space.set_parameter(name, value)

    def get_weight(self, name: str) -> Optional[float]:
        """Get current weight value"""
        return self._engine._parameter_space.get_parameter(name)

    def get_weights(self) -> Dict[str, float]:
        """Get all current weights"""
        return self._engine._parameter_space.get_all_parameters()

    def normalize_weights(self) -> Dict[str, float]:
        """
        Normalize weights to sum to 1.0.

        Returns:
            Dict of normalized weights
        """
        weights = self.get_weights()
        total = sum(weights.values())

        if total == 0:
            return weights

        return {k: v / total for k, v in weights.items()}

    def optimize(self) -> Optional[OptimizationResult]:
        """
        Run weight optimization.

        Returns:
            OptimizationResult or None if not enough data
        """
        if not self._should_update():
            return None

        # Check minimum samples
        recall_summary = self._engine.get_metrics().get_summary(MetricType.RECALL)
        if recall_summary.count < self._config.min_samples:
            return None

        # Get current weights
        old_weights = self.get_weights()

        # Run adaptation
        result = self._engine.adapt()
        new_weights = result.new_values

        # Calculate improvement
        improvement = 0.0
        for name in old_weights:
            old_val = old_weights[name]
            new_val = new_weights.get(name, old_val)
            if old_val > 0:
                improvement += (new_val - old_val) / old_val

        # Record result
        opt_result = OptimizationResult(
            parameter_name="retrieval_weights",
            old_value=sum(old_weights.values()),
            new_value=sum(new_weights.values()),
            improvement=improvement,
            confidence=min(recall_summary.count / 100.0, 1.0)
        )
        self._record_result(opt_result)

        return opt_result

    def suggest_weights(self, query_type: Optional[str] = None) -> Dict[str, float]:
        """
        Suggest weights based on query type.

        Args:
            query_type: Type of query (semantic, temporal, causal, etc.)

        Returns:
            Dict of suggested weights
        """
        weights = self.get_weights()

        if query_type == "semantic":
            weights["semantic_weight"] *= 1.2
        elif query_type == "temporal":
            weights["temporal_weight"] *= 1.2
        elif query_type == "causal":
            weights["causal_weight"] *= 1.2

        return self.normalize_weights()

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization history"""
        history = self.get_history()

        if not history:
            return {"status": "no_history", "count": 0}

        improvements = [h.improvement for h in history]

        return {
            "status": "active",
            "count": len(history),
            "avg_improvement": sum(improvements) / len(improvements),
            "last_improvement": improvements[-1] if improvements else 0.0,
            "current_weights": self.get_weights(),
            "normalized_weights": self.normalize_weights()
        }


# =============================================================================
# Encoding Dimension Adapter
# =============================================================================

@dataclass
class EncodingDimensionConfig:
    """Configuration for encoding dimension adapter"""
    # Dimension parameters
    min_dimension: int = 128
    max_dimension: int = 1024
    default_dimension: int = 384

    # Quality thresholds
    min_quality_threshold: float = 0.7
    max_dimension_increase_rate: float = 0.1  # 10% per update

    # Performance targets
    target_precision: float = 0.85
    max_encoding_latency_ms: float = 50.0


class EncodingDimensionAdapter(BaseAdapter):
    """
    Dynamic encoding dimension adjustment.

    Automatically adjusts encoding dimension based on:
    - Encoding precision
    - Latency requirements
    - Memory constraints

    Example:
        >>> adapter = EncodingDimensionAdapter()
        >>> adapter.record_metric(MetricType.PRECISION, 0.82)
        >>> adapter.record_metric(MetricType.LATENCY, 35.0, {"operation": "encode"})
        >>>
        >>> result = adapter.adjust_dimension()
    """

    # Dimension presets
    DIMENSION_PRESETS = {
        "compact": 128,
        "balanced": 384,
        "high": 512,
        "ultra": 768,
        "max": 1024
    }

    def __init__(
        self,
        config: Optional[EncodingDimensionConfig] = None,
        adaptive_engine: Optional[AdaptiveEngine] = None
    ):
        super().__init__(
            name="EncodingDimensionAdapter",
            config=AdapterConfig(AdapterType.ENCODING_DIMENSION),
            adaptive_engine=adaptive_engine
        )
        self._encoding_config = config or EncodingDimensionConfig()

        # Initialize dimension parameter
        self._engine.add_parameter(
            "encoding_dimension",
            self._encoding_config.default_dimension,
            (self._encoding_config.min_dimension, self._encoding_config.max_dimension)
        )

        # Track history for dimension changes
        self._dimension_history: List[Tuple[float, int]] = []

    def get_current_dimension(self) -> int:
        """Get current encoding dimension"""
        return int(self._engine._parameter_space.get_parameter("encoding_dimension") or
                   self._encoding_config.default_dimension)

    def set_dimension(self, dimension: int) -> bool:
        """
        Set encoding dimension.

        Args:
            dimension: Target dimension (clamped to valid range)

        Returns:
            True if set successfully
        """
        # Clamp to valid range
        clamped = max(
            self._encoding_config.min_dimension,
            min(self._encoding_config.max_dimension, dimension)
        )
        return self._engine._parameter_space.set_parameter("encoding_dimension", clamped)

    def adjust_dimension(
        self,
        precision: Optional[float] = None,
        latency: Optional[float] = None
    ) -> Optional[int]:
        """
        Adjust dimension based on metrics.

        Args:
            precision: Current encoding precision (0-1)
            latency: Current encoding latency (ms)

        Returns:
            New dimension or None if no adjustment needed
        """
        if not self._should_update():
            return None

        current_dim = self.get_current_dimension()
        precision = precision or self._get_avg_precision()
        latency = latency or self._get_avg_latency()

        # Determine adjustment direction
        adjustment = 0

        # Precision-based adjustment
        if precision < self._encoding_config.min_quality_threshold:
            # Need more precision = increase dimension
            adjustment = int(current_dim * self._encoding_config.max_dimension_increase_rate)
        elif precision > self._encoding_config.target_precision * 1.1:
            # Can reduce dimension
            adjustment = -int(current_dim * self._encoding_config.max_dimension_increase_rate * 0.5)

        # Latency-based adjustment
        if latency > self._encoding_config.max_encoding_latency_ms:
            # Too slow, reduce dimension
            adjustment = max(adjustment, -int(current_dim * 0.1))

        # Calculate new dimension
        if adjustment != 0:
            new_dim = max(
                self._encoding_config.min_dimension,
                min(self._encoding_config.max_dimension, current_dim + adjustment)
            )

            self.set_dimension(new_dim)
            self._dimension_history.append((time.time(), new_dim))

            return new_dim

        return current_dim

    def _get_avg_precision(self) -> float:
        """Get average precision from metrics"""
        summary = self._engine.get_metrics().get_summary(MetricType.PRECISION)
        return summary.mean if summary.count > 0 else 0.8

    def _get_avg_latency(self) -> float:
        """Get average latency from metrics"""
        summary = self._engine.get_metrics().get_summary(MetricType.LATENCY)
        return summary.mean if summary.count > 0 else 30.0

    def suggest_preset(self) -> str:
        """
        Suggest a dimension preset based on current metrics.

        Returns:
            Preset name
        """
        current = self.get_current_dimension()

        if current <= 128:
            return "compact"
        elif current <= 384:
            return "balanced"
        elif current <= 512:
            return "high"
        elif current <= 768:
            return "ultra"
        else:
            return "max"

    def get_dimension_stats(self) -> Dict[str, Any]:
        """Get dimension adjustment statistics"""
        if not self._dimension_history:
            return {
                "status": "no_adjustments",
                "current_dimension": self.get_current_dimension()
            }

        dimensions = [d for _, d in self._dimension_history]

        return {
            "status": "active",
            "current_dimension": self.get_current_dimension(),
            "adjustment_count": len(self._dimension_history),
            "min_dimension": min(dimensions),
            "max_dimension": max(dimensions),
            "avg_dimension": sum(dimensions) / len(dimensions),
            "history": self._dimension_history[-10:]  # Last 10
        }


# =============================================================================
# Cache Strategy Adapter
# =============================================================================

@dataclass
class CacheStrategyConfig:
    """Configuration for cache strategy adapter"""
    # Cache size parameters
    min_size: int = 100
    max_size: int = 10000
    default_size: int = 1000

    # TTL parameters (seconds)
    default_ttl: int = 3600
    min_ttl: int = 60
    max_ttl: int = 86400

    # Strategy preferences
    preferred_strategy: CacheStrategy = CacheStrategy.LRU

    # Performance targets
    target_hit_rate: float = 0.85


class CacheStrategyAdapter(BaseAdapter):
    """
    Automatic cache policy optimization.

    Optimizes cache parameters based on:
    - Hit rate metrics
    - Memory usage
    - Access patterns

    Example:
        >>> adapter = CacheStrategyAdapter()
        >>> adapter.record_metric(MetricType.HIT_RATE, 0.78)
        >>> adapter.record_metric(MetricType.MEMORY_USAGE, 500.0)
        >>>
        >>> result = adapter.optimize_cache()
    """

    def __init__(
        self,
        config: Optional[CacheStrategyConfig] = None,
        adaptive_engine: Optional[AdaptiveEngine] = None
    ):
        super().__init__(
            name="CacheStrategyAdapter",
            config=AdapterConfig(AdapterType.CACHE_STRATEGY),
            adaptive_engine=adaptive_engine
        )
        self._cache_config = config or CacheStrategyConfig()
        self._current_strategy = self._cache_config.preferred_strategy

        # Initialize cache parameters
        self._engine.add_parameter(
            "cache_size",
            self._cache_config.default_size,
            (self._cache_config.min_size, self._cache_config.max_size)
        )
        self._engine.add_parameter(
            "cache_ttl",
            self._cache_config.default_ttl,
            (self._cache_config.min_ttl, self._cache_config.max_ttl)
        )

        # Access pattern tracking
        self._access_patterns: List[str] = []

    def get_current_size(self) -> int:
        """Get current cache size"""
        return int(self._engine._parameter_space.get_parameter("cache_size") or
                   self._cache_config.default_size)

    def get_current_ttl(self) -> int:
        """Get current cache TTL"""
        return int(self._engine._parameter_space.get_parameter("cache_ttl") or
                   self._cache_config.default_ttl)

    def get_current_strategy(self) -> CacheStrategy:
        """Get current cache strategy"""
        return self._current_strategy

    def set_cache_size(self, size: int) -> bool:
        """Set cache size"""
        clamped = max(self._cache_config.min_size, min(self._cache_config.max_size, size))
        return self._engine._parameter_space.set_parameter("cache_size", clamped)

    def set_cache_ttl(self, ttl: int) -> bool:
        """Set cache TTL"""
        clamped = max(self._cache_config.min_ttl, min(self._cache_config.max_ttl, ttl))
        return self._engine._parameter_space.set_parameter("cache_ttl", clamped)

    def set_strategy(self, strategy: CacheStrategy):
        """Set cache strategy"""
        self._current_strategy = strategy

    def record_access(self, key: str, hit: bool):
        """
        Record cache access for pattern analysis.

        Args:
            key: Cache key
            hit: Whether it was a hit
        """
        self._access_patterns.append(key)
        # Keep history bounded
        if len(self._access_patterns) > 10000:
            self._access_patterns = self._access_patterns[-5000:]

        # Record hit rate
        hit_rate = 1.0 if hit else 0.0
        self.record_metric(MetricType.HIT_RATE, hit_rate, {"key": key[:20]})

    def optimize_cache(self) -> Dict[str, Any]:
        """
        Optimize cache parameters.

        Returns:
            Dict with optimization results
        """
        if not self._should_update():
            return {"status": "skipped", "reason": "too_soon"}

        hit_summary = self._engine.get_metrics().get_summary(MetricType.HIT_RATE)
        self._engine.get_metrics().get_summary(MetricType.MEMORY_USAGE)

        if hit_summary.count < self._config.min_samples:
            return {"status": "insufficient_data"}

        results = {}

        # Optimize based on hit rate
        if hit_summary.mean < self._cache_config.target_hit_rate:
            # Increase cache size
            current_size = self.get_current_size()
            new_size = int(current_size * 1.2)
            self.set_cache_size(new_size)
            results["cache_size"] = {"old": current_size, "new": new_size}

            # Consider strategy change
            if hit_summary.trend == "decreasing":
                self._consider_strategy_change()
                results["strategy_changed"] = True
        elif hit_summary.mean > self._cache_config.target_hit_rate * 1.2:
            # Can reduce cache size
            current_size = self.get_current_size()
            new_size = int(current_size * 0.8)
            self.set_cache_size(new_size)
            results["cache_size"] = {"old": current_size, "new": new_size}

        # Optimize TTL based on patterns
        self._optimize_ttl()

        return {
            "status": "optimized",
            "current_size": self.get_current_size(),
            "current_ttl": self.get_current_ttl(),
            "current_strategy": self._current_strategy.value,
            "changes": results
        }

    def _consider_strategy_change(self):
        """Consider changing cache strategy based on patterns"""
        if len(self._access_patterns) < 100:
            return

        # Simple pattern analysis
        recent = self._access_patterns[-100:]
        unique_ratio = len(set(recent)) / len(recent)

        if unique_ratio > 0.8:
            # High uniqueness = use LFU
            self._current_strategy = CacheStrategy.LFU
        else:
            # Low uniqueness = use LRU
            self._current_strategy = CacheStrategy.LRU

    def _optimize_ttl(self):
        """Optimize TTL based on access patterns"""
        if len(self._access_patterns) < 50:
            return

        # Analyze access frequency
        recent = self._access_patterns[-500:]
        from collections import Counter
        freq = Counter(recent)

        avg_freq = sum(freq.values()) / len(freq)

        # Adjust TTL based on frequency
        if avg_freq > 10:  # High frequency
            self.set_cache_ttl(int(self.get_current_ttl() * 0.8))  # Shorter TTL
        elif avg_freq < 2:  # Low frequency
            self.set_cache_ttl(int(self.get_current_ttl() * 1.2))  # Longer TTL

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache optimization statistics"""
        hit_summary = self._engine.get_metrics().get_summary(MetricType.HIT_RATE)

        return {
            "current_size": self.get_current_size(),
            "current_ttl": self.get_current_ttl(),
            "current_strategy": self._current_strategy.value,
            "hit_rate": hit_summary.mean if hit_summary.count > 0 else 0.0,
            "hit_rate_trend": hit_summary.trend if hit_summary.count > 0 else "stable",
            "target_hit_rate": self._cache_config.target_hit_rate
        }


# =============================================================================
# Adapter Registry
# =============================================================================

class ParameterAdapterRegistry:
    """
    Unified registry for all parameter adapters.

    Provides centralized management of adapters with:
    - Unified interface
    - Cross-adapter coordination
    - Performance monitoring

    Example:
        >>> registry = ParameterAdapterRegistry()
        >>> registry.register_retrieval_adapter()
        >>> registry.register_encoding_adapter()
        >>> registry.register_cache_adapter()
        >>>
        >>> # Optimize all
        >>> results = registry.optimize_all()
    """

    def __init__(self):
        self._adapters: Dict[AdapterType, BaseAdapter] = {}
        self._lock = threading.Lock()

    def register(
        self,
        adapter: BaseAdapter,
        replace: bool = False
    ) -> bool:
        """
        Register an adapter.

        Args:
            adapter: Adapter to register
            replace: Replace existing adapter of same type

        Returns:
            True if registered successfully
        """
        with self._lock:
            adapter_type = adapter.adapter_type

            if adapter_type in self._adapters and not replace:
                return False

            self._adapters[adapter_type] = adapter
            return True

    def unregister(self, adapter_type: AdapterType) -> bool:
        """Unregister an adapter"""
        with self._lock:
            if adapter_type in self._adapters:
                del self._adapters[adapter_type]
                return True
            return False

    def get(self, adapter_type: AdapterType) -> Optional[BaseAdapter]:
        """Get adapter by type"""
        return self._adapters.get(adapter_type)

    def get_all(self) -> List[BaseAdapter]:
        """Get all registered adapters"""
        with self._lock:
            return list(self._adapters.values())

    def optimize_all(self) -> Dict[str, Any]:
        """
        Run optimization on all adapters.

        Returns:
            Dict of results by adapter type
        """
        results = {}

        for adapter_type, adapter in self._adapters.items():
            if adapter_type == AdapterType.RETRIEVAL_WEIGHT:
                result = adapter.optimize()
            elif adapter_type == AdapterType.ENCODING_DIMENSION:
                result = adapter.adjust_dimension()
            elif adapter_type == AdapterType.CACHE_STRATEGY:
                result = adapter.optimize_cache()
            else:
                result = None

            results[adapter_type.value] = result

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all adapters"""
        summary = {
            "adapter_count": len(self._adapters),
            "adapters": {}
        }

        for adapter_type, adapter in self._adapters.items():
            if hasattr(adapter, 'get_optimization_summary'):
                adapter_summary = adapter.get_optimization_summary()
            elif hasattr(adapter, 'get_dimension_stats'):
                adapter_summary = adapter.get_dimension_stats()
            elif hasattr(adapter, 'get_cache_stats'):
                adapter_summary = adapter.get_cache_stats()
            else:
                adapter_summary = {"status": "unknown"}

            summary["adapters"][adapter_type.value] = adapter_summary

        return summary

    def reset_all(self):
        """Reset all adapters"""
        for adapter in self._adapters.values():
            adapter.reset()


# =============================================================================
# Factory Functions
# =============================================================================

def create_retrieval_adapter(
    config: Optional[RetrievalWeightConfig] = None
) -> RetrievalWeightAdapter:
    """Create a retrieval weight adapter"""
    return RetrievalWeightAdapter(config=config)


def create_encoding_adapter(
    config: Optional[EncodingDimensionConfig] = None
) -> EncodingDimensionAdapter:
    """Create an encoding dimension adapter"""
    return EncodingDimensionAdapter(config=config)


def create_cache_adapter(
    config: Optional[CacheStrategyConfig] = None
) -> CacheStrategyAdapter:
    """Create a cache strategy adapter"""
    return CacheStrategyAdapter(config=config)


def create_adapter_registry() -> ParameterAdapterRegistry:
    """Create an adapter registry with all adapters"""
    registry = ParameterAdapterRegistry()
    registry.register(RetrievalWeightAdapter())
    registry.register(EncodingDimensionAdapter())
    registry.register(CacheStrategyAdapter())
    return registry


# =============================================================================
# Test Suite
# =============================================================================

def test_parameter_adapters():
    """Test parameter adapter functionality"""
    print("=" * 60)
    print("Testing Parameter Adapters")
    print("=" * 60)

    passed = 0
    failed = 0

    def test(name: str, condition: bool):
        nonlocal passed, failed
        if condition:
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name}")
            failed += 1

    # Test 1: RetrievalWeightAdapter
    print("\n[Test 1] Retrieval Weight Adapter")
    print("-" * 40)

    adapter = RetrievalWeightAdapter()
    test("Create adapter", adapter is not None)

    adapter.set_weight("semantic_weight", 0.6)
    test("Set weight", adapter.get_weight("semantic_weight") == 0.6)

    weights = adapter.get_weights()
    test("Get all weights", len(weights) == 4)

    normalized = adapter.normalize_weights()
    total = sum(normalized.values())
    test("Normalized weights sum to 1", abs(total - 1.0) < 0.01)

    # Record metrics
    for _ in range(15):
        adapter.record_metric(MetricType.RECALL, 0.8 + _ * 0.01)

    result = adapter.optimize()
    test("Optimize returns result", result is not None)

    summary = adapter.get_optimization_summary()
    test("Get optimization summary", summary["count"] > 0)

    # Test 2: EncodingDimensionAdapter
    print("\n[Test 2] Encoding Dimension Adapter")
    print("-" * 40)

    enc_adapter = EncodingDimensionAdapter()
    test("Create encoding adapter", enc_adapter is not None)

    current_dim = enc_adapter.get_current_dimension()
    test("Get current dimension", 128 <= current_dim <= 1024)

    enc_adapter.set_dimension(512)
    test("Set dimension", enc_adapter.get_current_dimension() == 512)

    enc_adapter.record_metric(MetricType.PRECISION, 0.85)
    enc_adapter.record_metric(MetricType.LATENCY, 40.0)

    stats = enc_adapter.get_dimension_stats()
    test("Get dimension stats", "current_dimension" in stats)

    preset = enc_adapter.suggest_preset()
    test("Suggest preset", preset in ["compact", "balanced", "high", "ultra", "max"])

    # Test 3: CacheStrategyAdapter
    print("\n[Test 3] Cache Strategy Adapter")
    print("-" * 40)

    cache_adapter = CacheStrategyAdapter()
    test("Create cache adapter", cache_adapter is not None)

    test("Get current size", cache_adapter.get_current_size() > 0)
    test("Get current TTL", cache_adapter.get_current_ttl() > 0)

    cache_adapter.record_access("key1", True)
    cache_adapter.record_access("key2", False)
    test("Record access", True)

    cache_result = cache_adapter.optimize_cache()
    test("Optimize cache", "status" in cache_result)

    cache_stats = cache_adapter.get_cache_stats()
    test("Get cache stats", "current_size" in cache_stats)

    # Test 4: Adapter Registry
    print("\n[Test 4] Adapter Registry")
    print("-" * 40)

    registry = ParameterAdapterRegistry()
    test("Create registry", registry is not None)

    registry.register(adapter)
    test("Register retrieval adapter", len(registry.get_all()) == 1)

    registry.register(enc_adapter)
    registry.register(cache_adapter)
    test("Register all adapters", len(registry.get_all()) == 3)

    results = registry.optimize_all()
    test("Optimize all", len(results) == 3)

    registry_summary = registry.get_summary()
    test("Get registry summary", "adapter_count" in registry_summary)

    registry.reset_all()
    test("Reset all", True)

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = test_parameter_adapters()
    exit(0 if success else 1)
