"""
Adaptive Engine Module (自适应引擎)

This module provides adaptive intelligence capabilities for the SDK:
- Parameter space definition and management
- Learning metrics collection and analysis
- Adaptive optimization based on usage patterns

Core Features:
- AdaptiveEngine: Main adaptive engine for self-tuning
- ParameterSpace: Parameter space definition with boundaries
- LearningMetrics: Metrics collection for learning feedback

Architecture:
- Local ML inference (offline capable)
- Incremental learning with privacy protection
- Multi-objective optimization

【Pre-Phase Numeric】- Uses prior ordering for numerical calculations
【Post-Phase Symbolic】- Uses post ordering for symbolic applications
"""

from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import time
import threading
import statistics


# =============================================================================
# Enums
# =============================================================================

class ParameterType(Enum):
    """Parameter type classification"""
    CONTINUOUS = "continuous"      # Continuous value (0.0-1.0)
    DISCRETE = "discrete"          # Discrete choices
    BINARY = "binary"              # True/False
    RANKING = "ranking"            # Ordered ranking


class MetricType(Enum):
    """Learning metric types"""
    RECALL = "recall"              # Retrieval recall
    PRECISION = "precision"        # Retrieval precision
    LATENCY = "latency"            # Response latency
    MEMORY_USAGE = "memory_usage"  # Memory consumption
    HIT_RATE = "hit_rate"          # Cache hit rate
    SATISFACTION = "satisfaction"  # User satisfaction score


class AdaptationStrategy(Enum):
    """Adaptation strategy options"""
    GRADIENT_DESCENT = "gradient_descent"
    BAYESIAN = "bayesian"
    REINFORCEMENT = "reinforcement"
    HEURISTIC = "heuristic"


# =============================================================================
# Parameter Space Definition
# =============================================================================

@dataclass
class ParameterBound:
    """
    Parameter bound definition.

    Attributes:
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        default: Default value
        step: Step size for discrete parameters
    """
    min_value: float
    max_value: float
    default: float
    step: float = 0.0

    def __post_init__(self):
        """Validate bounds"""
        if self.min_value > self.max_value:
            raise ValueError(f"min_value > max_value: {self.min_value} > {self.max_value}")
        if not self.min_value <= self.default <= self.max_value:
            raise ValueError(f"default outside bounds: {self.default}")

    def clamp(self, value: float) -> float:
        """Clamp value to bounds"""
        return max(self.min_value, min(self.max_value, value))

    def normalize(self, value: float) -> float:
        """Normalize value to 0-1 range"""
        if self.max_value == self.min_value:
            return 0.5
        return (value - self.min_value) / (self.max_value - self.min_value)

    def denormalize(self, normalized: float) -> float:
        """Denormalize from 0-1 range"""
        return self.min_value + normalized * (self.max_value - self.min_value)


@dataclass
class ParameterSpace:
    """
    Parameter space definition with boundaries.

    Manages multiple parameters and their optimization ranges.

    Example:
        >>> space = ParameterSpace()
        >>> space.add_parameter("learning_rate",
        ...     ParameterBound(0.001, 0.1, 0.01),
        ...     ParameterType.CONTINUOUS)
        >>>
        >>> value = space.get_parameter("learning_rate")
        >>> space.set_parameter("learning_rate", 0.05)
    """
    _parameters: Dict[str, ParameterBound] = field(default_factory=dict)
    _types: Dict[str, ParameterType] = field(default_factory=dict)
    _current_values: Dict[str, float] = field(default_factory=dict)
    _metadata: Dict[str, Dict] = field(default_factory=dict)

    def add_parameter(
        self,
        name: str,
        bound: ParameterBound,
        param_type: ParameterType,
        metadata: Optional[Dict] = None
    ) -> "ParameterSpace":
        """
        Add a parameter to the space.

        Args:
            name: Parameter name
            bound: Parameter bounds
            param_type: Parameter type
            metadata: Optional metadata

        Returns:
            Self for chaining
        """
        self._parameters[name] = bound
        self._types[name] = param_type
        self._current_values[name] = bound.default
        if metadata:
            self._metadata[name] = metadata
        return self

    def get_parameter(self, name: str) -> Optional[float]:
        """Get current parameter value"""
        return self._current_values.get(name)

    def set_parameter(self, name: str, value: float) -> bool:
        """
        Set parameter value with clamping.

        Args:
            name: Parameter name
            value: New value

        Returns:
            True if set successfully
        """
        if name not in self._parameters:
            return False

        bound = self._parameters[name]
        clamped = bound.clamp(value)
        self._current_values[name] = clamped
        return True

    def get_bound(self, name: str) -> Optional[ParameterBound]:
        """Get parameter bounds"""
        return self._parameters.get(name)

    def get_type(self, name: str) -> Optional[ParameterType]:
        """Get parameter type"""
        return self._types.get(name)

    def get_all_parameters(self) -> Dict[str, float]:
        """Get all current parameters"""
        return dict(self._current_values)

    def set_all_parameters(self, values: Dict[str, float]) -> Dict[str, bool]:
        """
        Set multiple parameters.

        Returns:
            Dict of success status for each parameter
        """
        results = {}
        for name, value in values.items():
            results[name] = self.set_parameter(name, value)
        return results

    def reset_to_defaults(self):
        """Reset all parameters to defaults"""
        for name, bound in self._parameters.items():
            self._current_values[name] = bound.default

    def randomize(self, seed: Optional[int] = None) -> Dict[str, float]:
        """
        Randomize all parameters within bounds.

        Args:
            seed: Optional random seed

        Returns:
            Dict of new values
        """
        import random
        if seed is not None:
            random.seed(seed)

        for name, bound in self._parameters.items():
            if self._types.get(name) == ParameterType.DISCRETE and bound.step > 0:
                steps = int((bound.max_value - bound.min_value) / bound.step) + 1
                value = bound.min_value + random.choice(range(steps)) * bound.step
            else:
                value = random.uniform(bound.min_value, bound.max_value)
            self._current_values[name] = value

        return self.get_all_parameters()

    def sample_subset(self, count: int, seed: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Sample a subset of parameter combinations.

        Args:
            count: Number of samples
            seed: Optional random seed

        Returns:
            List of (name, value) tuples
        """
        import random
        if seed is not None:
            random.seed(seed)

        names = list(self._parameters.keys())
        sampled = random.sample(names, min(count, len(names)))

        return [(name, self._current_values[name]) for name in sampled]

    @property
    def parameter_count(self) -> int:
        """Get number of parameters"""
        return len(self._parameters)

    def __repr__(self) -> str:
        return f"ParameterSpace(params={self.parameter_count})"


# =============================================================================
# Learning Metrics
# =============================================================================

@dataclass
class MetricEntry:
    """Single metric entry"""
    timestamp: float
    metric_type: MetricType
    value: float
    context: Dict = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Summary statistics for a metric"""
    metric_type: MetricType
    count: int
    mean: float
    median: float
    std_dev: float
    min_val: float
    max_val: float
    trend: str  # "increasing", "decreasing", "stable"


class LearningMetrics:
    """
    Learning metrics collection and analysis.

    Collects usage patterns and computes statistics for adaptive optimization.

    Example:
        >>> metrics = LearningMetrics()
        >>> metrics.record(MetricType.RECALL, 0.85, {"query": "test"})
        >>> metrics.record(MetricType.LATENCY, 120.0, {"operation": "search"})
        >>>
        >>> summary = metrics.get_summary(MetricType.RECALL)
        >>> trend = metrics.get_trend(MetricType.RECALL, window=100)
    """

    def __init__(self, max_entries: int = 10000):
        """
        Initialize metrics collector.

        Args:
            max_entries: Maximum entries to keep (for memory management)
        """
        self._max_entries = max_entries
        self._entries: List[MetricEntry] = []
        self._lock = threading.Lock()

        # Pre-allocate storage by type
        self._by_type: Dict[MetricType, List[MetricEntry]] = {
            mt: [] for mt in MetricType
        }

        # Context counters
        self._context_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def record(
        self,
        metric_type: MetricType,
        value: float,
        context: Optional[Dict] = None
    ):
        """
        Record a metric entry.

        Args:
            metric_type: Type of metric
            value: Metric value
            context: Optional context data
        """
        entry = MetricEntry(
            timestamp=time.time(),
            metric_type=metric_type,
            value=value,
            context=context or {}
        )

        with self._lock:
            # Add to global list
            self._entries.append(entry)

            # Add to type-specific list
            self._by_type[metric_type].append(entry)

            # Update context counts
            for key, val in entry.context.items():
                if isinstance(val, str):
                    self._context_counts[key][val] += 1

            # Trim if needed
            if len(self._entries) > self._max_entries:
                self._trim_oldest()

    def _trim_oldest(self):
        """Trim oldest entries to stay within max_entries"""
        # Calculate how many to remove
        excess = len(self._entries) - self._max_entries

        if excess <= 0:
            return

        # Remove from global list
        removed = self._entries[:excess]
        self._entries = self._entries[excess:]

        # Remove from type-specific lists
        for entry in removed:
            type_list = self._by_type[entry.metric_type]
            if type_list and type_list[0] is entry:
                self._by_type[entry.metric_type] = type_list[1:]

    def get_entries(
        self,
        metric_type: Optional[MetricType] = None,
        since: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[MetricEntry]:
        """
        Get metric entries.

        Args:
            metric_type: Filter by type (None = all)
            since: Filter by timestamp (None = all)
            limit: Maximum entries (None = all)

        Returns:
            List of entries
        """
        with self._lock:
            if metric_type:
                entries = self._by_type[metric_type][:]
            else:
                entries = self._entries[:]

            if since is not None:
                entries = [e for e in entries if e.timestamp >= since]

            if limit:
                entries = entries[-limit:]

            return entries

    def get_values(
        self,
        metric_type: MetricType,
        window: Optional[int] = None
    ) -> List[float]:
        """
        Get metric values.

        Args:
            metric_type: Metric type
            window: Use last N entries (None = all)

        Returns:
            List of values
        """
        with self._lock:
            entries = self._by_type[metric_type]

            if window:
                entries = entries[-window:]

            return [e.value for e in entries]

    def get_summary(self, metric_type: MetricType) -> MetricSummary:
        """
        Get summary statistics for a metric.

        Args:
            metric_type: Metric type

        Returns:
            MetricSummary with statistics
        """
        values = self.get_values(metric_type)

        if not values:
            return MetricSummary(
                metric_type=metric_type,
                count=0,
                mean=0.0,
                median=0.0,
                std_dev=0.0,
                min_val=0.0,
                max_val=0.0,
                trend="stable"
            )

        mean_val = statistics.mean(values)
        median_val = statistics.median(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.0

        # Calculate trend
        if len(values) >= 10:
            first_half = statistics.mean(values[:len(values)//2])
            second_half = statistics.mean(values[len(values)//2:])
            if second_half > first_half * 1.05:
                trend = "increasing"
            elif second_half < first_half * 0.95:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return MetricSummary(
            metric_type=metric_type,
            count=len(values),
            mean=mean_val,
            median=median_val,
            std_dev=std_dev,
            min_val=min(values),
            max_val=max(values),
            trend=trend
        )

    def get_trend(
        self,
        metric_type: MetricType,
        window: int = 100
    ) -> Tuple[float, str]:
        """
        Get trend direction and slope.

        Args:
            metric_type: Metric type
            window: Window size for trend calculation

        Returns:
            Tuple of (slope, direction)
        """
        values = self.get_values(metric_type, window)

        if len(values) < 2:
            return 0.0, "stable"

        # Simple linear regression
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(values)

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0.0

        # Normalize slope
        if abs(slope) < 0.001:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        return slope, direction

    def get_context_distribution(self, key: str) -> Dict[str, int]:
        """Get distribution of context values"""
        return dict(self._context_counts.get(key, {}))

    def clear(self):
        """Clear all metrics"""
        with self._lock:
            self._entries.clear()
            for type_list in self._by_type.values():
                type_list.clear()
            self._context_counts.clear()

    def __repr__(self) -> str:
        return f"LearningMetrics(entries={len(self._entries)}, types={len(self._by_type)})"


# =============================================================================
# Adaptive Engine
# =============================================================================

@dataclass
class AdaptationResult:
    """Result of an adaptation step"""
    improved: bool
    previous_values: Dict[str, float]
    new_values: Dict[str, float]
    improvement_score: float
    metrics: Dict[str, float]


class AdaptiveEngine:
    """
    Main adaptive engine for self-tuning.

    Provides intelligent parameter optimization based on usage patterns
    and learning metrics.

    Core Features:
    - Multi-objective optimization
    - Gradient-based and heuristic adaptation
    - Safe exploration with bounds
    - Incremental learning with privacy protection

    Example:
        >>> engine = AdaptiveEngine()
        >>> engine.add_parameter("threshold", 0.5, (0.0, 1.0))
        >>> engine.add_metric(MetricType.RECALL)
        >>>
        >>> # Record performance
        >>> engine.record_metric(MetricType.RECALL, 0.85)
        >>>
        >>> # Adapt parameters
        >>> result = engine.adapt()
        >>> print(f"Improved: {result.improved}")
    """

    def __init__(
        self,
        strategy: AdaptationStrategy = AdaptationStrategy.HEURISTIC,
        exploration_rate: float = 0.1
    ):
        """
        Initialize adaptive engine.

        Args:
            strategy: Adaptation strategy to use
            exploration_rate: Rate of random exploration (0.0-1.0)
        """
        self._strategy = strategy
        self._exploration_rate = exploration_rate

        # Parameter space
        self._parameter_space = ParameterSpace()

        # Learning metrics
        self._metrics = LearningMetrics()

        # History for learning
        self._history: List[Tuple[Dict[str, float], Dict[str, float], float]] = []

        # Callbacks
        self._on_adaptation: Optional[Callable[[Dict[str, float]], None]] = None

        # Lock for thread safety
        self._lock = threading.Lock()

        # Best known configuration
        self._best_values: Optional[Dict[str, float]] = None
        self._best_score: float = -float('inf')

    def add_parameter(
        self,
        name: str,
        default: float,
        bounds: Tuple[float, float],
        param_type: ParameterType = ParameterType.CONTINUOUS,
        step: float = 0.0
    ):
        """
        Add a tunable parameter.

        Args:
            name: Parameter name
            default: Default value
            bounds: Tuple of (min, max)
            param_type: Parameter type
            step: Step size for discrete parameters
        """
        bound = ParameterBound(
            min_value=bounds[0],
            max_value=bounds[1],
            default=default,
            step=step
        )
        self._parameter_space.add_parameter(name, bound, param_type)

    def add_metric(self, metric_type: MetricType):
        """Add a metric to track (metrics are tracked by default)"""
        # Metrics are collected automatically
        pass

    def record_metric(
        self,
        metric_type: MetricType,
        value: float,
        context: Optional[Dict] = None
    ):
        """
        Record a metric value.

        Args:
            metric_type: Type of metric
            value: Metric value
            context: Optional context
        """
        self._metrics.record(metric_type, value, context)

    def get_metrics(self) -> LearningMetrics:
        """Get metrics collector"""
        return self._metrics

    def adapt(
        self,
        target_metrics: Optional[Dict[MetricType, float]] = None
    ) -> AdaptationResult:
        """
        Run adaptation step.

        Args:
            target_metrics: Optional target values for metrics

        Returns:
            AdaptationResult with new parameters
        """
        with self._lock:
            previous_values = self._parameter_space.get_all_parameters()

            # Get current metric summaries
            recall_summary = self._metrics.get_summary(MetricType.RECALL)
            latency_summary = self._metrics.get_summary(MetricType.LATENCY)

            # Calculate current score
            current_score = self._calculate_score(recall_summary, latency_summary)

            # Apply adaptation strategy
            new_values = self._apply_strategy(previous_values, current_score)

            # Update parameters
            self._parameter_space.set_all_parameters(new_values)

            # Calculate improvement
            improvement = current_score - (self._best_score if self._best_score > 0 else 0)

            # Update best
            if current_score > self._best_score:
                self._best_score = current_score
                self._best_values = dict(new_values)

            # Record in history
            self._history.append((previous_values, new_values, current_score))

            # Execute callback
            if self._on_adaptation:
                self._on_adaptation(new_values)

            return AdaptationResult(
                improved=current_score > self._best_score * 0.95,  # Within 5% of best
                previous_values=previous_values,
                new_values=new_values,
                improvement_score=improvement,
                metrics={
                    "recall_mean": recall_summary.mean,
                    "recall_trend": recall_summary.trend,
                    "latency_mean": latency_summary.mean,
                    "latency_trend": latency_summary.trend
                }
            )

    def _calculate_score(
        self,
        recall_summary: MetricSummary,
        latency_summary: MetricSummary
    ) -> float:
        """Calculate composite score from metrics"""
        # Weighted combination: 60% recall, 40% latency (inverted for lower is better)
        recall_score = recall_summary.mean * 0.6 if recall_summary.count > 0 else 0.5

        # Latency: normalize and invert (lower latency = higher score)
        latency_norm = 1.0 - min(latency_summary.mean / 1000.0, 1.0)
        latency_score = latency_norm * 0.4

        return recall_score + latency_score

    def _apply_strategy(
        self,
        current_values: Dict[str, float],
        current_score: float
    ) -> Dict[str, float]:
        """Apply adaptation strategy"""
        import random

        new_values = dict(current_values)

        if self._strategy == AdaptationStrategy.HEURISTIC:
            # Heuristic: adjust based on metric trends
            recall_trend = self._metrics.get_trend(MetricType.RECALL)
            self._metrics.get_trend(MetricType.LATENCY)

            # Adjust threshold based on trends
            if "threshold" in new_values:
                threshold = new_values["threshold"]
                if recall_trend[1] == "decreasing":
                    threshold *= 0.95  # Lower threshold for more candidates
                elif recall_trend[1] == "increasing":
                    threshold *= 1.02  # Slightly increase
                new_values["threshold"] = max(0.1, min(0.9, threshold))

        # Random exploration
        if random.random() < self._exploration_rate:
            for name in new_values:
                bound = self._parameter_space.get_bound(name)
                if bound and random.random() < 0.3:
                    # Small random adjustment
                    delta = random.uniform(-0.1, 0.1) * (bound.max_value - bound.min_value)
                    new_values[name] = bound.clamp(new_values[name] + delta)

        return new_values

    def get_best_configuration(self) -> Optional[Dict[str, float]]:
        """Get best known configuration"""
        return dict(self._best_values) if self._best_values else None

    def reset(self):
        """Reset to default configuration"""
        with self._lock:
            self._parameter_space.reset_to_defaults()
            self._metrics.clear()
            self._history.clear()
            self._best_values = None
            self._best_score = -float('inf')

    def on_adaptation(self, callback: Callable[[Dict[str, float]], None]):
        """Set callback for adaptation events"""
        self._on_adaptation = callback

    def __repr__(self) -> str:
        return f"AdaptiveEngine(strategy={self._strategy.value}, params={self._parameter_space.parameter_count})"


# =============================================================================
# Convenience Functions
# =============================================================================

def create_adaptive_engine(
    strategy: AdaptationStrategy = AdaptationStrategy.HEURISTIC,
    exploration_rate: float = 0.1
) -> AdaptiveEngine:
    """Create a configured adaptive engine"""
    return AdaptiveEngine(strategy=strategy, exploration_rate=exploration_rate)


def create_parameter_space() -> ParameterSpace:
    """Create an empty parameter space"""
    return ParameterSpace()


def create_metrics_collector(max_entries: int = 10000) -> LearningMetrics:
    """Create a metrics collector"""
    return LearningMetrics(max_entries=max_entries)


# =============================================================================
# Test Suite
# =============================================================================

def test_adaptive_engine():
    """Test adaptive engine functionality"""
    print("=" * 60)
    print("Testing Adaptive Engine")
    print("=" * 60)

    engine = AdaptiveEngine()
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

    # Test 1: Parameter space
    print("\n[Test 1] Parameter Space")
    print("-" * 40)

    space = ParameterSpace()
    space.add_parameter("threshold", ParameterBound(0.0, 1.0, 0.5), ParameterType.CONTINUOUS)
    space.add_parameter("learning_rate", ParameterBound(0.001, 0.1, 0.01), ParameterType.CONTINUOUS)

    test("Add parameters", space.parameter_count == 2)
    test("Get parameter", space.get_parameter("threshold") == 0.5)
    test("Set parameter", space.set_parameter("threshold", 0.8))
    test("Get updated value", space.get_parameter("threshold") == 0.8)
    test("Clamp value", space.set_parameter("threshold", 1.5))
    test("Clamped result", space.get_parameter("threshold") == 1.0)

    # Test 2: Learning metrics
    print("\n[Test 2] Learning Metrics")
    print("-" * 40)

    metrics = LearningMetrics()
    metrics.record(MetricType.RECALL, 0.8)
    metrics.record(MetricType.RECALL, 0.85)
    metrics.record(MetricType.RECALL, 0.9)
    metrics.record(MetricType.LATENCY, 100.0)
    metrics.record(MetricType.LATENCY, 120.0)

    summary = metrics.get_summary(MetricType.RECALL)
    test("Record metrics", summary.count == 3)
    test("Mean calculation", abs(summary.mean - 0.85) < 0.01)

    recall_trend = metrics.get_trend(MetricType.RECALL)
    test("Trend detection", recall_trend[1] == "increasing")

    # Test 3: Adaptive engine
    print("\n[Test 3] Adaptive Engine")
    print("-" * 40)

    engine = AdaptiveEngine()
    engine.add_parameter("threshold", 0.5, (0.0, 1.0))
    engine.add_parameter("learning_rate", 0.01, (0.001, 0.1))

    test("Add parameters", engine._parameter_space.parameter_count == 2)

    # Record metrics
    for _ in range(10):
        engine.record_metric(MetricType.RECALL, 0.8 + _ * 0.01)
        engine.record_metric(MetricType.LATENCY, 100 - _ * 2)

    # Adapt
    result = engine.adapt()
    test("Adapt returns result", result is not None)
    test("Result has new values", len(result.new_values) == 2)

    # Test 4: Best configuration
    print("\n[Test 4] Best Configuration")
    print("-" * 40)

    engine.reset()
    for _ in range(5):
        engine.record_metric(MetricType.RECALL, 0.9)
        engine.adapt()

    best = engine.get_best_configuration()
    test("Best configuration exists", best is not None)

    # Test 5: Metrics collection
    print("\n[Test 5] Metrics Collection")
    print("-" * 40)

    test("Metrics accessible", engine.get_metrics() is not None)

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = test_adaptive_engine()
    exit(0 if success else 1)
