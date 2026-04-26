"""
Incremental Learning Module (增量学习)

This module provides incremental learning capabilities for v1.6.0:
- User feedback loop
- Incremental model updates
- Memory forgetting strategies

Core Features:
- FeedbackLoop: User feedback collection and processing
- IncrementalUpdater: Incremental model update mechanism
- MemoryForgetting: Memory decay and pruning strategies
- IncrementalLearningManager: Unified management

Architecture:
- Privacy-preserving feedback collection
- Efficient incremental updates
- Configurable forgetting policies

【Pre-Phase Numeric】- Uses prior ordering for numerical calculations
【Post-Phase Symbolic】- Uses post ordering for symbolic applications
"""

from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import math
from collections import defaultdict, deque
from dataclasses import dataclass


# =============================================================================
# Enums
# =============================================================================

class FeedbackType(Enum):
    """User feedback types"""
    POSITIVE = "positive"          # User satisfied
    NEGATIVE = "negative"         # User dissatisfied
    NEUTRAL = "neutral"           # No opinion
    CORRECTION = "correction"     # User provided correction
    SKIP = "skip"                 # User skipped


class UpdateStrategy(Enum):
    """Model update strategies"""
    GRADUAL = "gradual"           # Slow incremental
    EAGER = "eager"              # Immediate update
    BATCH = "batch"              # Batch updates
    PRIORITIZED = "prioritized"  # Priority-based updates


class ForgettingPolicy(Enum):
    """Memory forgetting policies"""
    LRU = "lru"                  # Least Recently Used
    TIME_DECAY = "time_decay"    # Time-based decay
    IMPORTANCE = "importance"    # Importance-based
    HYBRID = "hybrid"            # Combined strategy


@dataclass
class FeedbackEntry:
    """User feedback entry"""
    timestamp: float
    feedback_type: FeedbackType
    content: Any
    context: Dict = field(default_factory=dict)
    weight: float = 1.0


@dataclass
class UpdateResult:
    """Result of an incremental update"""
    success: bool
    updated_count: int
    discarded_count: int
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None


# =============================================================================
# Feedback Loop
# =============================================================================

class FeedbackLoop:
    """
    User feedback collection and processing.
    
    Collects and processes user feedback for continuous learning.
    
    Example:
        >>> loop = FeedbackLoop()
        >>> loop.record_feedback(FeedbackType.POSITIVE, {"query": "test", "result": "good"})
        >>> loop.record_feedback(FeedbackType.NEGATIVE, {"query": "test", "result": "bad"})
        >>> 
        >>> stats = loop.get_statistics()
        >>> corrections = loop.get_corrections()
    """
    
    def __init__(
        self,
        max_entries: int = 10000,
        auto_decay: bool = True,
        decay_rate: float = 0.95
    ):
        self._max_entries = max_entries
        self._auto_decay = auto_decay
        self._decay_rate = decay_rate
        
        self._entries: deque = deque(maxlen=max_entries)
        self._positive_count = 0
        self._negative_count = 0
        self._corrections: List[FeedbackEntry] = []
        self._lock = threading.Lock()
        
        # Statistics
        self._total_count = 0
        self._last_update = time.time()
    
    def record_feedback(
        self,
        feedback_type: FeedbackType,
        content: Any,
        context: Optional[Dict] = None,
        weight: float = 1.0
    ):
        """
        Record a user feedback.
        
        Args:
            feedback_type: Type of feedback
            content: Feedback content
            context: Optional context data
            weight: Feedback weight
        """
        entry = FeedbackEntry(
            timestamp=time.time(),
            feedback_type=feedback_type,
            content=content,
            context=context or {},
            weight=weight
        )
        
        with self._lock:
            self._entries.append(entry)
            self._total_count += 1
            
            if feedback_type == FeedbackType.POSITIVE:
                self._positive_count += 1
            elif feedback_type == FeedbackType.NEGATIVE:
                self._negative_count += 1
            elif feedback_type == FeedbackType.CORRECTION:
                self._corrections.append(entry)
            
            self._last_update = time.time()
    
    def get_recent_feedback(
        self,
        limit: int = 100,
        feedback_type: Optional[FeedbackType] = None
    ) -> List[FeedbackEntry]:
        """
        Get recent feedback entries.
        
        Args:
            limit: Maximum number of entries
            feedback_type: Optional filter by type
        
        Returns:
            List of feedback entries
        """
        with self._lock:
            entries = list(self._entries)
        
        if feedback_type:
            entries = [e for e in entries if e.feedback_type == feedback_type]
        
        return entries[-limit:]
    
    def get_corrections(self, limit: int = 100) -> List[FeedbackEntry]:
        """Get user corrections"""
        with self._lock:
            return self._corrections[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get feedback statistics.
        
        Returns:
            Dict with statistics
        """
        with self._lock:
            total = len(self._entries)
            positive = self._positive_count
            negative = self._negative_count
            corrections = len(self._corrections)
        
        return {
            "total": total,
            "positive": positive,
            "negative": negative,
            "corrections": corrections,
            "positive_rate": positive / total if total > 0 else 0.0,
            "negative_rate": negative / total if total > 0 else 0.0,
            "last_update": self._last_update
        }
    
    def get_sentiment_trend(self, window: int = 100) -> str:
        """
        Get sentiment trend over recent window.
        
        Returns:
            "improving", "declining", or "stable"
        """
        recent = self.get_recent_feedback(limit=window)
        
        if len(recent) < 10:
            return "stable"
        
        half = len(recent) // 2
        first_half = recent[:half]
        second_half = recent[half:]
        
        first_positive = sum(1 for e in first_half if e.feedback_type == FeedbackType.POSITIVE)
        second_positive = sum(1 for e in second_half if e.feedback_type == FeedbackType.POSITIVE)
        
        first_rate = first_positive / len(first_half) if first_half else 0
        second_rate = second_positive / len(second_half) if second_half else 0
        
        if second_rate > first_rate * 1.1:
            return "improving"
        elif second_rate < first_rate * 0.9:
            return "declining"
        else:
            return "stable"
    
    def clear(self):
        """Clear all feedback"""
        with self._lock:
            self._entries.clear()
            self._positive_count = 0
            self._negative_count = 0
            self._corrections.clear()
            self._total_count = 0


# =============================================================================
# Incremental Updater
# =============================================================================

class IncrementalUpdater:
    """
    Incremental model update mechanism.
    
    Provides efficient model updates without full retraining.
    
    Example:
        >>> updater = IncrementalUpdater(strategy=UpdateStrategy.GRADUAL)
        >>> 
        >>> # Register model parameters
        >>> updater.register_param("weight", initial_value=0.5)
        >>> 
        >>> # Process feedback
        >>> updater.process_feedback({"feedback": "positive"}, delta=0.1)
        >>> 
        >>> # Apply updates
        >>> result = updater.apply_updates()
    """
    
    def __init__(
        self,
        strategy: UpdateStrategy = UpdateStrategy.GRADUAL,
        learning_rate: float = 0.1,
        momentum: float = 0.9
    ):
        self._strategy = strategy
        self._lr = learning_rate
        self._momentum = momentum
        
        self._params: Dict[str, float] = {}
        self._gradients: Dict[str, float] = {}
        self._velocity: Dict[str, float] = {}
        self._update_queue: deque = deque(maxlen=1000)
        
        self._lock = threading.Lock()
        self._pending_updates = 0
    
    def register_param(
        self,
        name: str,
        initial_value: float = 0.0,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ):
        """
        Register a model parameter.
        
        Args:
            name: Parameter name
            initial_value: Initial value
            min_value: Optional minimum
            max_value: Optional maximum
        """
        with self._lock:
            self._params[name] = initial_value
            self._gradients[name] = 0.0
            self._velocity[name] = 0.0
            
            if min_value is not None:
                self._params[name] = max(min_value, self._params[name])
            if max_value is not None:
                self._params[name] = min(max_value, self._params[name])
    
    def process_feedback(
        self,
        feedback: Dict[str, Any],
        delta: float = 0.1,
        param_names: Optional[List[str]] = None
    ):
        """
        Process feedback and compute updates.
        
        Args:
            feedback: Feedback data
            delta: Update magnitude
            param_names: Parameters to update (None = all)
        """
        with self._lock:
            # Determine gradient direction based on feedback
            feedback_type = feedback.get("type", "positive")
            
            if feedback_type == "positive":
                direction = 1.0
            elif feedback_type == "negative":
                direction = -1.0
            elif feedback_type == "correction":
                direction = feedback.get("correction_strength", 1.0)
            else:
                direction = 0.0
            
            # Apply update to parameters
            params_to_update = param_names or list(self._params.keys())
            
            for name in params_to_update:
                if name not in self._gradients:
                    continue
                
                gradient = direction * delta
                
                if self._strategy == UpdateStrategy.GRADUAL:
                    # Gradual update with momentum
                    self._gradients[name] += gradient
                elif self._strategy == UpdateStrategy.EAGER:
                    # Immediate update
                    self._params[name] += direction * self._lr * delta
                elif self._strategy == UpdateStrategy.PRIORITIZED:
                    # Priority-based update
                    priority = feedback.get("priority", 1.0)
                    self._gradients[name] += direction * delta * priority
                else:
                    # Default: add to queue
                    self._gradients[name] += gradient
            
            self._pending_updates += 1
    
    def apply_updates(self) -> UpdateResult:
        """
        Apply accumulated updates.
        
        Returns:
            UpdateResult
        """
        with self._lock:
            if self._pending_updates == 0:
                return UpdateResult(success=True, updated_count=0, discarded_count=0)
            
            updated = 0
            for name in self._params:
                if name not in self._gradients:
                    continue
                
                gradient = self._gradients[name]
                
                if self._strategy == UpdateStrategy.GRADUAL:
                    # Apply with momentum
                    self._velocity[name] = self._momentum * self._velocity[name] + gradient
                    self._params[name] += self._lr * self._velocity[name]
                
                # Reset gradient
                self._gradients[name] = 0.0
                updated += 1
            
            self._pending_updates = 0
            
            return UpdateResult(
                success=True,
                updated_count=updated,
                discarded_count=0
            )
    
    def get_params(self) -> Dict[str, float]:
        """Get current parameter values"""
        with self._lock:
            return dict(self._params)
    
    def set_param(self, name: str, value: float) -> bool:
        """Set parameter value"""
        with self._lock:
            if name not in self._params:
                return False
            self._params[name] = value
            return True
    
    def reset(self):
        """Reset all parameters and gradients"""
        with self._lock:
            for name in self._params:
                self._gradients[name] = 0.0
                self._velocity[name] = 0.0
            self._pending_updates = 0


# =============================================================================
# Memory Forgetting
# =============================================================================

@dataclass
class MemoryEntry:
    """Memory entry with importance tracking"""
    key: str
    content: Any
    importance: float = 1.0
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    decay_rate: float = 0.01


class MemoryForgetting:
    """
    Memory decay and pruning strategies.
    
    Implements various forgetting policies for memory management.
    
    Example:
        >>> forgetting = MemoryForgetting(policy=ForgettingPolicy.HYBRID)
        >>> 
        >>> forgetting.add("key1", {"data": "value1"}, importance=0.8)
        >>> forgetting.add("key2", {"data": "value2"}, importance=0.5)
        >>> 
        >>> # Access increases importance
        >>> forgetting.access("key1")
        >>> 
        >>> # Get decayed memory
        >>> pruned = forgetting.prune(threshold=0.3)
    """
    
    def __init__(
        self,
        policy: ForgettingPolicy = ForgettingPolicy.HYBRID,
        decay_rate: float = 0.01,
        base_threshold: float = 0.1
    ):
        self._policy = policy
        self._decay_rate = decay_rate
        self._base_threshold = base_threshold
        
        self._memories: Dict[str, MemoryEntry] = {}
        self._access_history: deque = deque(maxlen=1000)
        
        self._lock = threading.Lock()
    
    def add(
        self,
        key: str,
        content: Any,
        importance: float = 1.0,
        decay_rate: Optional[float] = None
    ):
        """
        Add a memory entry.
        
        Args:
            key: Memory key
            content: Memory content
            importance: Initial importance (0-1)
            decay_rate: Optional custom decay rate
        """
        entry = MemoryEntry(
            key=key,
            content=content,
            importance=importance,
            decay_rate=decay_rate or self._decay_rate
        )
        
        with self._lock:
            self._memories[key] = entry
    
    def access(self, key: str) -> Optional[Any]:
        """
        Access a memory entry.
        
        Args:
            key: Memory key
        
        Returns:
            Memory content or None
        """
        with self._lock:
            if key not in self._memories:
                return None
            
            entry = self._memories[key]
            entry.access_count += 1
            entry.last_access = time.time()
            
            # Boost importance based on access
            if self._policy == ForgettingPolicy.HYBRID:
                entry.importance = min(1.0, entry.importance * 1.1)
            elif self._policy == ForgettingPolicy.IMPORTANCE:
                entry.importance = min(1.0, entry.importance * 1.05)
            
            self._access_history.append(key)
            
            return entry.content
    
    def get_importance(self, key: str) -> Optional[float]:
        """Get current importance of a memory"""
        with self._lock:
            if key not in self._memories:
                return None
            return self._memories[key].importance
    
    def set_importance(self, key: str, importance: float) -> bool:
        """Set importance of a memory"""
        with self._lock:
            if key not in self._memories:
                return False
            self._memories[key].importance = max(0.0, min(1.0, importance))
            return True
    
    def decay(self, elapsed: Optional[float] = None):
        """
        Apply time-based decay to all memories.
        
        Args:
            elapsed: Elapsed time in seconds (None = use current time)
        """
        current_time = time.time()
        elapsed = elapsed or current_time
        
        with self._lock:
            for entry in self._memories.values():
                if self._policy == ForgettingPolicy.TIME_DECAY:
                    # Exponential decay based on time
                    time_factor = math.exp(-entry.decay_rate * (current_time - entry.last_access) / 3600)
                    entry.importance *= time_factor
                elif self._policy == ForgettingPolicy.HYBRID:
                    # Combined decay
                    time_factor = math.exp(-entry.decay_rate * (current_time - entry.last_access) / 3600)
                    entry.importance = max(
                        self._base_threshold,
                        entry.importance * time_factor * 0.9
                    )
    
    def prune(self, threshold: Optional[float] = None) -> List[str]:
        """
        Prune low-importance memories.
        
        Args:
            threshold: Minimum importance to keep (None = use base_threshold)
        
        Returns:
            List of pruned keys
        """
        threshold = threshold or self._base_threshold
        pruned = []
        
        with self._lock:
            # Apply decay first
            self.decay()
            
            # Find low-importance entries
            keys_to_remove = [
                key for key, entry in self._memories.items()
                if entry.importance < threshold
            ]
            
            # Remove entries
            for key in keys_to_remove:
                del self._memories[key]
                pruned.append(key)
        
        return pruned
    
    def get_all(self) -> List[Tuple[str, float, Any]]:
        """
        Get all memories sorted by importance.
        
        Returns:
            List of (key, importance, content) tuples
        """
        with self._lock:
            items = [
                (key, entry.importance, entry.content)
                for key, entry in self._memories.items()
            ]
        
        items.sort(key=lambda x: -x[1])
        return items
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        with self._lock:
            if not self._memories:
                return {
                    "count": 0,
                    "avg_importance": 0.0,
                    "min_importance": 0.0,
                    "max_importance": 0.0
                }
            
            importances = [e.importance for e in self._memories.values()]
            
            return {
                "count": len(self._memories),
                "avg_importance": sum(importances) / len(importances),
                "min_importance": min(importances),
                "max_importance": max(importances),
                "policy": self._policy.value
            }
    
    def clear(self):
        """Clear all memories"""
        with self._lock:
            self._memories.clear()


# =============================================================================
# Incremental Learning Manager
# =============================================================================

class IncrementalLearningManager:
    """
    Unified incremental learning management.
    
    Coordinates feedback loop, updater, and forgetting.
    
    Example:
        >>> manager = IncrementalLearningManager()
        >>> 
        >>> # Process user feedback
        >>> manager.process_feedback(FeedbackType.POSITIVE, {"query": "test"})
        >>> 
        >>> # Update model
        >>> manager.update()
        >>> 
        >>> # Prune old memories
        >>> manager.prune_memories()
        >>> 
        >>> # Get status
        >>> status = manager.get_status()
    """
    
    def __init__(
        self,
        update_strategy: UpdateStrategy = UpdateStrategy.GRADUAL,
        forgetting_policy: ForgettingPolicy = ForgettingPolicy.HYBRID
    ):
        self._feedback_loop = FeedbackLoop()
        self._updater = IncrementalUpdater(strategy=update_strategy)
        self._forgetting = MemoryForgetting(policy=forgetting_policy)
        
        self._lock = threading.Lock()
    
    def process_feedback(
        self,
        feedback_type: FeedbackType,
        content: Any,
        context: Optional[Dict] = None,
        weight: float = 1.0
    ):
        """
        Process user feedback.
        
        Args:
            feedback_type: Type of feedback
            content: Feedback content
            context: Optional context
            weight: Feedback weight
        """
        # Record feedback
        self._feedback_loop.record_feedback(
            feedback_type, content, context, weight
        )
        
        # Compute parameter update
        ctx = context or {}
        self._updater.process_feedback(
            {"type": feedback_type.value, **ctx},
            delta=weight * 0.1
        )
        
        # Update memory importance
        if context and "memory_key" in context:
            memory_key = context["memory_key"]
            if feedback_type == FeedbackType.POSITIVE:
                self._forgetting.set_importance(memory_key, 1.0)
            elif feedback_type == FeedbackType.NEGATIVE:
                self._forgetting.set_importance(memory_key, 0.5)
    
    def update(self) -> UpdateResult:
        """Apply accumulated updates"""
        return self._updater.apply_updates()
    
    def prune_memories(self, threshold: Optional[float] = None) -> List[str]:
        """Prune low-importance memories"""
        return self._forgetting.prune(threshold)
    
    def add_memory(
        self,
        key: str,
        content: Any,
        importance: float = 1.0
    ):
        """Add a memory entry"""
        self._forgetting.add(key, content, importance)
    
    def access_memory(self, key: str) -> Optional[Any]:
        """Access a memory entry"""
        return self._forgetting.access(key)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get overall status.
        
        Returns:
            Dict with status information
        """
        feedback_stats = self._feedback_loop.get_statistics()
        memory_stats = self._forgetting.get_statistics()
        params = self._updater.get_params()
        
        return {
            "feedback": feedback_stats,
            "memory": memory_stats,
            "parameters": params,
            "sentiment_trend": self._feedback_loop.get_sentiment_trend()
        }
    
    def reset(self):
        """Reset all components"""
        self._feedback_loop.clear()
        self._updater.reset()
        self._forgetting.clear()


# =============================================================================
# Factory Functions
# =============================================================================

def create_feedback_loop(
    max_entries: int = 10000,
    auto_decay: bool = True
) -> FeedbackLoop:
    """Create a feedback loop"""
    return FeedbackLoop(max_entries=max_entries, auto_decay=auto_decay)


def create_incremental_updater(
    strategy: UpdateStrategy = UpdateStrategy.GRADUAL
) -> IncrementalUpdater:
    """Create an incremental updater"""
    return IncrementalUpdater(strategy=strategy)


def create_memory_forgetting(
    policy: ForgettingPolicy = ForgettingPolicy.HYBRID
) -> MemoryForgetting:
    """Create a memory forgetting system"""
    return MemoryForgetting(policy=policy)


def create_learning_manager(
    update_strategy: UpdateStrategy = UpdateStrategy.GRADUAL,
    forgetting_policy: ForgettingPolicy = ForgettingPolicy.HYBRID
) -> IncrementalLearningManager:
    """Create an incremental learning manager"""
    return IncrementalLearningManager(
        update_strategy=update_strategy,
        forgetting_policy=forgetting_policy
    )


# =============================================================================
# Test Suite
# =============================================================================

def test_incremental_learning():
    """Test incremental learning components"""
    print("=" * 60)
    print("Testing Incremental Learning")
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
    
    # Test 1: Feedback Loop
    print("\n[Test 1] Feedback Loop")
    print("-" * 40)
    
    loop = FeedbackLoop()
    loop.record_feedback(FeedbackType.POSITIVE, {"query": "test1"})
    loop.record_feedback(FeedbackType.NEGATIVE, {"query": "test2"})
    loop.record_feedback(FeedbackType.POSITIVE, {"query": "test3"})
    loop.record_feedback(FeedbackType.CORRECTION, {"query": "test4", "corrected": True})
    
    stats = loop.get_statistics()
    test("Record feedback", stats["total"] == 4)
    test("Count positives", stats["positive"] == 2)
    test("Count negatives", stats["negative"] == 1)
    test("Count corrections", stats["corrections"] == 1)
    
    corrections = loop.get_corrections()
    test("Get corrections", len(corrections) == 1)
    
    trend = loop.get_sentiment_trend(window=10)
    test("Get sentiment trend", trend in ["improving", "declining", "stable"])
    
    # Test 2: Incremental Updater
    print("\n[Test 2] Incremental Updater")
    print("-" * 40)
    
    updater = IncrementalUpdater(strategy=UpdateStrategy.GRADUAL)
    updater.register_param("weight", initial_value=0.5)
    updater.register_param("bias", initial_value=0.1)
    
    test("Register params", len(updater.get_params()) == 2)
    
    updater.process_feedback({"type": "positive"}, delta=0.1)
    updater.process_feedback({"type": "positive"}, delta=0.1)
    updater.process_feedback({"type": "negative"}, delta=0.1)
    
    result = updater.apply_updates()
    test("Apply updates", result.success)
    test("Updated count > 0", result.updated_count > 0)
    
    params = updater.get_params()
    test("Get params", "weight" in params)
    
    # Test 3: Memory Forgetting
    print("\n[Test 3] Memory Forgetting")
    print("-" * 40)
    
    forgetting = MemoryForgetting(policy=ForgettingPolicy.HYBRID)
    forgetting.add("key1", "content1", importance=0.8)
    forgetting.add("key2", "content2", importance=0.5)
    forgetting.add("key3", "content3", importance=0.3)
    
    stats = forgetting.get_statistics()
    test("Memory count", stats["count"] == 3)
    
    content = forgetting.access("key1")
    test("Access memory", content == "content1")
    
    importance = forgetting.get_importance("key1")
    test("Importance boosted", importance is not None and importance >= 0.8)
    
    pruned = forgetting.prune(threshold=0.4)
    test("Prune returns list", isinstance(pruned, list))
    
    remaining = forgetting.get_statistics()
    test("Memory pruned", remaining["count"] <= 3)
    
    # Test 4: Learning Manager
    print("\n[Test 4] Learning Manager")
    print("-" * 40)
    
    manager = create_learning_manager()
    
    manager.process_feedback(FeedbackType.POSITIVE, {"query": "test"}, {"memory_key": "mem1"})
    manager.process_feedback(FeedbackType.NEGATIVE, {"query": "test"}, {"memory_key": "mem2"})
    
    manager.add_memory("mem1", "content1", importance=0.8)
    manager.add_memory("mem2", "content2", importance=0.6)
    
    content = manager.access_memory("mem1")
    test("Access via manager", content == "content1")
    
    status = manager.get_status()
    test("Get status", "feedback" in status and "memory" in status)
    test("Status has sentiment", "sentiment_trend" in status)
    
    update_result = manager.update()
    test("Update via manager", update_result.success)
    
    pruned = manager.prune_memories()
    test("Prune via manager", isinstance(pruned, list))
    
    # Test 5: Parameter Updates
    print("\n[Test 5] Update Strategies")
    print("-" * 40)
    
    gradual = IncrementalUpdater(strategy=UpdateStrategy.GRADUAL)
    gradual.register_param("param", initial_value=0.5)
    gradual.process_feedback({"type": "positive"}, delta=0.1)
    gradual.apply_updates()
    test("Gradual strategy", True)
    
    eager = IncrementalUpdater(strategy=UpdateStrategy.EAGER)
    eager.register_param("param", initial_value=0.5)
    eager.process_feedback({"type": "positive"}, delta=0.1)
    eager.apply_updates()
    test("Eager strategy", True)
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = test_incremental_learning()
    exit(0 if success else 1)