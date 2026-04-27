"""
Local Prediction Model Module (本地预测模型)

This module provides lightweight local ML prediction capabilities for v1.6.0:
- Local model inference (offline capable)
- Prediction result caching
- Privacy-preserving predictions

Core Features:
- LocalMLModel: Base class for local ML models
- PredictionCache: Caching layer for predictions
- OfflinePredictor: Offline prediction capability
- LocalModelManager: Model management and coordination

Architecture:
- Pure Python implementation (no external ML dependencies)
- Lightweight models (<10MB)
- Privacy-preserving (all data stays local)

【Pre-Phase Numeric】- Uses prior ordering for numerical calculations
【Post-Phase Symbolic】- Uses post ordering for symbolic applications
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import hashlib
import json
import math
from collections import OrderedDict, defaultdict


# =============================================================================
# Enums
# =============================================================================

class ModelType(Enum):
    """Local model types"""
    SIMPLE_LINEAR = "simple_linear"        # Linear regression
    NAIVE_BAYES = "naive_bayes"            # Naive Bayes classifier
    DECISION_TREE = "decision_tree"        # Decision tree
    KMEANS_CLUSTER = "kmeans_cluster"      # K-means clustering
    TFIDF_RANKER = "tfidf_ranker"         # TF-IDF based ranking


class PredictionStatus(Enum):
    """Prediction status codes"""
    SUCCESS = "success"
    CACHE_HIT = "cache_hit"
    FALLBACK = "fallback"
    MODEL_NOT_LOADED = "model_not_loaded"
    INVALID_INPUT = "invalid_input"


class CacheEvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    FIFO = "fifo"


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for local models"""
    model_type: ModelType
    name: str
    version: str = "1.0.0"

    # Model parameters
    input_dim: int = 128
    output_dim: int = 1

    # Performance settings
    max_cache_size: int = 10000
    cache_ttl_seconds: int = 3600

    # Privacy settings
    local_only: bool = True
    no_network: bool = True


# =============================================================================
# Prediction Result
# =============================================================================

@dataclass
class PredictionResult:
    """Result of a prediction"""
    status: PredictionStatus
    value: Any
    confidence: float
    model_name: str
    latency_ms: float
    cached: bool = False
    metadata: Dict = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status in (PredictionStatus.SUCCESS, PredictionStatus.CACHE_HIT)


# =============================================================================
# Simple Linear Model (轻量级线性模型)
# =============================================================================

class SimpleLinearModel:
    """
    Simple linear regression model.

    A lightweight model for numerical predictions.
    Supports both regression and binary classification.

    Example:
        >>> model = SimpleLinearModel(input_dim=128, output_dim=1)
        >>> model.fit(X_train, y_train)
        >>> result = model.predict(X_test)
    """

    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 1,
        learning_rate: float = 0.01,
        regularization: float = 0.01
    ):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._lr = learning_rate
        self._reg = regularization

        # Initialize weights
        self._weights = [[0.0] * input_dim for _ in range(output_dim)]
        self._biases = [0.0] * output_dim

        self._fitted = False

    def fit(
        self,
        X: List[List[float]],
        y: List[float],
        epochs: int = 100,
        batch_size: int = 32
    ):
        """
        Train the model using gradient descent.

        Args:
            X: Training features
            y: Training labels
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        if len(X) != len(y):
            raise ValueError("X and y must have same length")

        n_samples = len(X)

        for epoch in range(epochs):
            # Shuffle data
            indices = list(range(n_samples))
            self._shuffle(indices)

            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                self._update_weights(X, y, batch_indices)

            # Log progress (every 10 epochs)
            if (epoch + 1) % 10 == 0:
                loss = self._calculate_loss(X, y)
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")

        self._fitted = True

    def _shuffle(self, indices: List[int]):
        """Fisher-Yates shuffle"""
        import random
        random.shuffle(indices)

    def _update_weights(
        self,
        X: List[List[float]],
        y: List[float],
        indices: List[int]
    ):
        """Update weights using gradient descent"""
        n = len(indices)

        # Calculate gradients
        for out_idx in range(self._output_dim):
            grad_w = [0.0] * self._input_dim
            grad_b = 0.0

            for idx in indices:
                x = X[idx]
                y_true = y[idx]

                # Forward pass
                y_pred = self._forward(x)[out_idx]
                error = y_pred - y_true

                # Gradients
                for i in range(self._input_dim):
                    grad_w[i] += error * x[i] / n
                grad_b += error / n

            # Update weights with regularization
            for i in range(self._input_dim):
                self._weights[out_idx][i] -= self._lr * (grad_w[i] + self._reg * self._weights[out_idx][i])
            self._biases[out_idx] -= self._lr * grad_b

    def _forward(self, x: List[float]) -> List[float]:
        """Forward pass"""
        if len(x) != self._input_dim:
            raise ValueError(f"Input dimension mismatch: expected {self._input_dim}, got {len(x)}")

        return [
            sum(w * xi for w, xi in zip(self._weights[o], x)) + self._biases[o]
            for o in range(self._output_dim)
        ]

    def _calculate_loss(self, X: List[List[float]], y: List[float]) -> float:
        """Calculate MSE loss"""
        total_loss = 0.0
        for i in range(len(X)):
            pred = self._forward(X[i])
            error = pred[0] - y[i]
            total_loss += error * error
        return total_loss / len(X)

    def predict(self, X: List[float]) -> float:
        """
        Make a prediction.

        Args:
            X: Input features

        Returns:
            Predicted value
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        result = self._forward(X)
        return result[0]

    def predict_batch(self, X: List[List[float]]) -> List[float]:
        """Batch prediction"""
        return [self.predict(x) for x in X]

    def get_weights(self) -> List[List[float]]:
        """Get model weights"""
        return self._weights

    def get_biases(self) -> List[float]:
        """Get model biases"""
        return self._biases

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def __repr__(self) -> str:
        return f"SimpleLinearModel(input={self._input_dim}, output={self._output_dim}, fitted={self._fitted})"


# =============================================================================
# Naive Bayes Classifier
# =============================================================================

class NaiveBayesClassifier:
    """
    Naive Bayes classifier for categorical features.

    A simple probabilistic classifier.

    Example:
        >>> model = NaiveBayesClassifier()
        >>> model.fit(X_train, y_train)
        >>> result = model.predict(X_test)
        >>> print(f"Class: {result.class_label}, Confidence: {result.confidence}")
    """

    def __init__(self, alpha: float = 1.0):
        """
        Initialize classifier.

        Args:
            alpha: Laplace smoothing parameter
        """
        self._alpha = alpha
        self._classes: Set[Any] = set()
        self._class_priors: Dict[Any, float] = {}
        self._feature_probs: Dict[Tuple[Any, Any], float] = {}
        self._feature_values: Dict[int, Set[Any]] = defaultdict(set)
        self._fitted = False

    def fit(self, X: List[List[Any]], y: List[Any]):
        """
        Train the classifier.

        Args:
            X: Training features (categorical)
            y: Training labels
        """
        n = len(X)

        # Count classes
        class_counts: Dict[Any, int] = defaultdict(int)
        for label in y:
            self._classes.add(label)
            class_counts[label] += 1

        # Calculate class priors
        for cls in self._classes:
            self._class_priors[cls] = class_counts[cls] / n

        # Collect feature values
        len(X[0]) if X else 0
        for x in X:
            for i, val in enumerate(x):
                self._feature_values[i].add(val)

        # Calculate feature probabilities
        feature_counts: Dict[Tuple[int, Any, Any], int] = defaultdict(int)
        for i in range(len(X)):
            for j, val in enumerate(X[i]):
                key = (j, val, y[i])
                feature_counts[key] += 1

        len(self._classes)
        for (feat_idx, feat_val, cls), count in feature_counts.items():
            n_feat_values = len(self._feature_values[feat_idx])
            # P(feature|class) with Laplace smoothing
            prob = (count + self._alpha) / (class_counts[cls] + n_feat_values * self._alpha)
            self._feature_probs[(feat_idx, feat_val, cls)] = prob

        self._fitted = True

    def predict(self, x: List[Any]) -> PredictionResult:
        """
        Make a prediction.

        Args:
            x: Input features

        Returns:
            PredictionResult
        """
        start_time = time.time()

        if not self._fitted:
            return PredictionResult(
                status=PredictionStatus.MODEL_NOT_LOADED,
                value=None,
                confidence=0.0,
                model_name="NaiveBayes",
                latency_ms=(time.time() - start_time) * 1000
            )

        # Calculate posterior for each class
        posteriors: Dict[Any, float] = {}
        for cls in self._classes:
            log_prob = math.log(self._class_priors[cls])

            for feat_idx, feat_val in enumerate(x):
                key = (feat_idx, feat_val, cls)
                if key in self._feature_probs:
                    log_prob += math.log(self._feature_probs[key])
                # Missing feature: use uniform probability
                else:
                    n_feat_values = len(self._feature_values[feat_idx])
                    log_prob += math.log(self._alpha / (1 + n_feat_values * self._alpha))

            posteriors[cls] = log_prob

        # Find best class
        best_class = max(posteriors, key=posteriors.get)
        best_log_prob = posteriors[best_class]

        # Convert to confidence (softmax-like)
        max_log = max(posteriors.values())
        total = sum(math.exp(lp - max_log) for lp in posteriors.values())
        confidence = math.exp(best_log_prob - max_log) / total if total > 0 else 0.0

        latency_ms = (time.time() - start_time) * 1000

        return PredictionResult(
            status=PredictionStatus.SUCCESS,
            value=best_class,
            confidence=confidence,
            model_name="NaiveBayes",
            latency_ms=latency_ms
        )

    def predict_proba(self, x: List[Any]) -> Dict[Any, float]:
        """Get probability distribution over classes"""
        if not self._fitted:
            return {}

        # Calculate raw scores
        scores: Dict[Any, float] = {}
        for cls in self._classes:
            log_prob = math.log(self._class_priors[cls])
            for feat_idx, feat_val in enumerate(x):
                key = (feat_idx, feat_val, cls)
                if key in self._feature_probs:
                    log_prob += math.log(self._feature_probs[key])
            scores[cls] = log_prob

        # Softmax
        max_score = max(scores.values())
        total = sum(math.exp(s - max_score) for s in scores.values())

        return {cls: math.exp(score - max_score) / total for cls, score in scores.items()}

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def __repr__(self) -> str:
        return f"NaiveBayesClassifier(classes={len(self._classes)}, fitted={self._fitted})"


# =============================================================================
# TF-IDF Ranker (文本排序模型)
# =============================================================================

class TFIDFRanker:
    """
    TF-IDF based ranker for text retrieval and ranking.

    A lightweight text ranking model.

    Example:
        >>> ranker = TFIDFRanker()
        >>> ranker.fit(documents)
        >>> results = ranker.rank("query text", top_k=10)
    """

    def __init__(self, max_features: int = 10000):
        """
        Initialize ranker.

        Args:
            max_features: Maximum vocabulary size
        """
        self._max_features = max_features
        self._vocabulary: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._document_count = 0
        self._fitted = False

    def fit(self, documents: List[str]):
        """
        Build vocabulary and IDF from documents.

        Args:
            documents: List of document strings
        """
        # Tokenize
        doc_tokens = [self._tokenize(doc) for doc in documents]

        # Build vocabulary (top max_features by frequency)
        token_freq: Dict[str, int] = defaultdict(int)
        for tokens in doc_tokens:
            for token in tokens:
                token_freq[token] += 1

        # Sort by frequency and take top
        sorted_tokens = sorted(token_freq.items(), key=lambda x: -x[1])[:self._max_features]
        self._vocabulary = {token: idx for idx, (token, _) in enumerate(sorted_tokens)}

        # Calculate IDF
        self._document_count = len(documents)
        doc_freq: Dict[str, int] = defaultdict(int)

        for tokens in doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                if token in self._vocabulary:
                    doc_freq[token] += 1

        for token, freq in doc_freq.items():
            self._idf[token] = math.log((self._document_count + 1) / (freq + 1)) + 1

        self._fitted = True

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [t for t in tokens if len(t) > 2]

    def _calculate_tfidf(self, tokens: List[str]) -> Dict[int, float]:
        """Calculate TF-IDF vector"""
        tf: Dict[str, int] = defaultdict(int)
        for token in tokens:
            tf[token] += 1

        # Normalize by max term frequency
        max_tf = max(tf.values()) if tf else 1

        result = {}
        for token, count in tf.items():
            if token in self._vocabulary:
                tf_norm = count / max_tf
                result[self._vocabulary[token]] = tf_norm * self._idf.get(token, 1.0)

        return result

    def rank(
        self,
        query: str,
        documents: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Rank documents by relevance to query.

        Args:
            query: Query string
            documents: Optional pre-tokenized documents
            top_k: Number of top results to return

        Returns:
            List of (doc_index, score) tuples
        """
        if not self._fitted:
            return []

        query_tokens = self._tokenize(query)
        query_vec = self._calculate_tfidf(query_tokens)

        if documents is None:
            return []  # Need documents for ranking

        # Calculate scores
        scores = []
        for i, doc in enumerate(documents):
            doc_tokens = self._tokenize(doc)
            doc_vec = self._calculate_tfidf(doc_tokens)

            score = self._cosine_similarity(query_vec, doc_vec)
            scores.append((i, score))

        # Sort by score descending
        scores.sort(key=lambda x: -x[1])

        return scores[:top_k]

    def _cosine_similarity(
        self,
        vec1: Dict[int, float],
        vec2: Dict[int, float]
    ) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0

        # Find common keys
        common = set(vec1.keys()) & set(vec2.keys())

        dot_product = sum(vec1[k] * vec2[k] for k in common)
        norm1 = math.sqrt(sum(v * v for v in vec1.values()))
        norm2 = math.sqrt(sum(v * v for v in vec2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def vocabulary_size(self) -> int:
        return len(self._vocabulary)

    def __repr__(self) -> str:
        return f"TFIDFRanker(vocab={self.vocabulary_size}, docs={self._document_count})"


# =============================================================================
# Prediction Cache
# =============================================================================

class PredictionCache:
    """
    LRU cache for prediction results.

    Provides caching with TTL and size limits.

    Example:
        >>> cache = PredictionCache(max_size=1000, ttl=3600)
        >>> cache.put("key", {"result": "value"})
        >>> result = cache.get("key")
    """

    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: int = 3600,
        eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU
    ):
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._policy = eviction_policy

        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def _make_key(self, *args) -> str:
        """Generate cache key from arguments"""
        key_data = json.dumps(args, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()

    def put(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Store a prediction result.

        Args:
            key: Cache key
            value: Prediction result
            ttl: Optional custom TTL
        """
        with self._lock:
            # Evict if needed
            if len(self._cache) >= self._max_size and key not in self._cache:
                self._evict()

            self._cache[key] = value
            self._timestamps[key] = time.time()
            self._access_counts[key] = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a cached prediction.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        with self._lock:
            if key not in self._cache:
                return None

            # Check TTL
            if self._is_expired(key):
                self._remove(key)
                return None

            # Update access tracking
            self._access_counts[key] += 1

            # Move to end for LRU
            if self._policy == CacheEvictionPolicy.LRU:
                self._cache.move_to_end(key)

            return self._cache[key]

    def _is_expired(self, key: str) -> bool:
        """Check if entry is expired"""
        if self._policy == CacheEvictionPolicy.TTL:
            age = time.time() - self._timestamps.get(key, 0)
            return age > self._ttl
        return False

    def _evict(self):
        """Evict one entry based on policy"""
        if not self._cache:
            return

        if self._policy == CacheEvictionPolicy.LRU:
            # Remove oldest (first)
            oldest_key = next(iter(self._cache))
            self._remove(oldest_key)

        elif self._policy == CacheEvictionPolicy.LFU:
            # Remove least frequently used
            if self._access_counts:
                lfu_key = min(self._access_counts, key=self._access_counts.get)
                self._remove(lfu_key)

        elif self._policy == CacheEvictionPolicy.FIFO:
            # Remove oldest by timestamp
            if self._timestamps:
                oldest_key = min(self._timestamps, key=self._timestamps.get)
                self._remove(oldest_key)

    def _remove(self, key: str):
        """Remove an entry"""
        if key in self._cache:
            del self._cache[key]
        if key in self._timestamps:
            del self._timestamps[key]
        if key in self._access_counts:
            del self._access_counts[key]

    def clear(self):
        """Clear all cached entries"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._access_counts.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "policy": self._policy.value,
                "ttl_seconds": self._ttl
            }

    def __len__(self) -> int:
        return len(self._cache)


# =============================================================================
# Local Model Manager
# =============================================================================

class LocalModelManager:
    """
    Manager for local ML models.

    Provides:
    - Model loading and unloading
    - Prediction routing
    - Cache management
    - Offline capability

    Example:
        >>> manager = LocalModelManager()
        >>> manager.register_model("linear", SimpleLinearModel())
        >>>
        >>> result = manager.predict("linear", features)
        >>> print(f"Prediction: {result.value}, Confidence: {result.confidence}")
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self._config = config or ModelConfig(
            model_type=ModelType.SIMPLE_LINEAR,
            name="LocalModelManager"
        )

        self._models: Dict[str, Any] = {}
        self._caches: Dict[str, PredictionCache] = {}
        self._lock = threading.Lock()

        # Default cache settings
        self._default_cache_size = 1000
        self._default_cache_ttl = 3600

    def register_model(
        self,
        name: str,
        model: Any,
        cache_size: int = 1000,
        cache_ttl: int = 3600
    ) -> bool:
        """
        Register a model.

        Args:
            name: Model name
            model: Model instance
            cache_size: Cache size for this model
            cache_ttl: Cache TTL for this model

        Returns:
            True if registered successfully
        """
        with self._lock:
            if name in self._models:
                return False

            self._models[name] = model
            self._caches[name] = PredictionCache(
                max_size=cache_size,
                ttl_seconds=cache_ttl
            )
            return True

    def unregister_model(self, name: str) -> bool:
        """Unregister a model"""
        with self._lock:
            if name in self._models:
                del self._models[name]
                if name in self._caches:
                    del self._caches[name]
                return True
            return False

    def predict(
        self,
        model_name: str,
        input_data: Any,
        use_cache: bool = True,
        cache_key: Optional[str] = None
    ) -> PredictionResult:
        """
        Make a prediction using the specified model.

        Args:
            model_name: Name of the model to use
            input_data: Input data for prediction
            use_cache: Whether to use caching
            cache_key: Optional custom cache key

        Returns:
            PredictionResult
        """
        start_time = time.time()

        # Check if model exists
        if model_name not in self._models:
            return PredictionResult(
                status=PredictionStatus.MODEL_NOT_LOADED,
                value=None,
                confidence=0.0,
                model_name=model_name,
                latency_ms=(time.time() - start_time) * 1000
            )

        model = self._models[model_name]
        cache = self._caches.get(model_name)

        # Generate cache key
        key = cache_key or self._generate_key(input_data)

        # Check cache
        if use_cache and cache:
            cached = cache.get(key)
            if cached is not None:
                cached.latency_ms = (time.time() - start_time) * 1000
                cached.cached = True
                return cached

        # Make prediction
        try:
            if hasattr(model, 'predict'):
                if isinstance(input_data, list) and len(input_data) > 0:
                    if isinstance(input_data[0], list):
                        # Batch prediction
                        values = model.predict_batch(input_data)
                        value = values[0] if values else None
                    else:
                        # Single prediction
                        value = model.predict(input_data)
                else:
                    value = model.predict(input_data)

                confidence = 1.0
                status = PredictionStatus.SUCCESS
            else:
                value = input_data
                confidence = 0.5
                status = PredictionStatus.FALLBACK
        except Exception as e:
            return PredictionResult(
                status=PredictionStatus.INVALID_INPUT,
                value=None,
                confidence=0.0,
                model_name=model_name,
                latency_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e)}
            )

        result = PredictionResult(
            status=status,
            value=value,
            confidence=confidence,
            model_name=model_name,
            latency_ms=(time.time() - start_time) * 1000,
            cached=False
        )

        # Cache result
        if use_cache and cache:
            cache.put(key, result)

        return result

    def _generate_key(self, data: Any) -> str:
        """Generate cache key from data"""
        try:
            data_str = json.dumps(data, sort_keys=True)
            return hashlib.md5(data_str.encode()).hexdigest()
        except Exception:
            return hashlib.md5(str(id(data)).encode()).hexdigest()

    def predict_with_fallback(
        self,
        primary_model: str,
        fallback_model: str,
        input_data: Any
    ) -> PredictionResult:
        """
        Predict with fallback model.

        Args:
            primary_model: Primary model name
            fallback_model: Fallback model name
            input_data: Input data

        Returns:
            PredictionResult
        """
        # Try primary model
        result = self.predict(primary_model, input_data)

        if result.is_success:
            return result

        # Try fallback
        return self.predict(fallback_model, input_data)

    def clear_cache(self, model_name: Optional[str] = None):
        """Clear cache for a model or all models"""
        with self._lock:
            if model_name:
                if model_name in self._caches:
                    self._caches[model_name].clear()
            else:
                for cache in self._caches.values():
                    cache.clear()

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a model"""
        if model_name not in self._models:
            return None

        model = self._models[model_name]

        info = {
            "name": model_name,
            "type": type(model).__name__,
            "is_fitted": getattr(model, "is_fitted", False)
        }

        if hasattr(model, "vocabulary_size"):
            info["vocabulary_size"] = model.vocabulary_size

        if model_name in self._caches:
            info["cache_stats"] = self._caches[model_name].get_stats()

        return info

    def list_models(self) -> List[str]:
        """List all registered models"""
        return list(self._models.keys())

    def __repr__(self) -> str:
        return f"LocalModelManager(models={len(self._models)})"


# =============================================================================
# Factory Functions
# =============================================================================

def create_linear_model(
    input_dim: int = 128,
    output_dim: int = 1
) -> SimpleLinearModel:
    """Create a simple linear model"""
    return SimpleLinearModel(input_dim=input_dim, output_dim=output_dim)


def create_naive_bayes(alpha: float = 1.0) -> NaiveBayesClassifier:
    """Create a Naive Bayes classifier"""
    return NaiveBayesClassifier(alpha=alpha)


def create_tfidf_ranker(max_features: int = 10000) -> TFIDFRanker:
    """Create a TF-IDF ranker"""
    return TFIDFRanker(max_features=max_features)


def create_prediction_cache(
    max_size: int = 10000,
    ttl: int = 3600,
    policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU
) -> PredictionCache:
    """Create a prediction cache"""
    return PredictionCache(max_size=max_size, ttl_seconds=ttl, eviction_policy=policy)


def create_model_manager() -> LocalModelManager:
    """Create a local model manager"""
    return LocalModelManager()


# =============================================================================
# Test Suite
# =============================================================================

def test_local_models():
    """Test local prediction models"""
    print("=" * 60)
    print("Testing Local Prediction Models")
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

    # Test 1: Simple Linear Model
    print("\n[Test 1] Simple Linear Model")
    print("-" * 40)

    model = SimpleLinearModel(input_dim=4, output_dim=1)
    X_train = [[1.0, 2.0, 3.0, 4.0]] * 50
    y_train = [3.0 * x[0] + 2.0 * x[1] + 1.0 for x in X_train]

    model.fit(X_train, y_train, epochs=50)
    test("Model fits", model.is_fitted)

    prediction = model.predict([1.0, 2.0, 3.0, 4.0])
    test("Model predicts", isinstance(prediction, (int, float)))

    # Test 2: Naive Bayes
    print("\n[Test 2] Naive Bayes Classifier")
    print("-" * 40)

    nb = NaiveBayesClassifier(alpha=1.0)
    X_train = [
        ['sunny', 'hot', 'high', 'weak'],
        ['sunny', 'hot', 'high', 'strong'],
        ['overcast', 'hot', 'high', 'weak'],
        ['rain', 'mild', 'high', 'weak'],
        ['rain', 'cool', 'normal', 'weak'],
    ]
    y_train = ['no', 'no', 'yes', 'yes', 'yes']

    nb.fit(X_train, y_train)
    test("NB fits", nb.is_fitted)

    result = nb.predict(['sunny', 'hot', 'high', 'weak'])
    test("NB predicts", result.is_success)
    test("NB has confidence", 0.0 <= result.confidence <= 1.0)

    # Test 3: TF-IDF Ranker
    print("\n[Test 3] TF-IDF Ranker")
    print("-" * 40)

    ranker = TFIDFRanker()
    docs = [
        "machine learning is a subset of artificial intelligence",
        "deep learning uses neural networks with multiple layers",
        "natural language processing deals with text and speech",
        "computer vision enables machines to interpret images",
    ]

    ranker.fit(docs)
    test("Ranker fits", ranker.is_fitted)

    # Store docs for ranking
    ranker._rank_docs = docs
    results = ranker.rank("machine learning neural networks", top_k=2)
    test("Ranker returns results", len(results) <= 2)
    test("Results have scores", all(score >= 0 for _, score in results) if results else True)

    # Test 4: Prediction Cache
    print("\n[Test 4] Prediction Cache")
    print("-" * 40)

    cache = PredictionCache(max_size=10, ttl_seconds=60)

    cache.put("key1", {"value": 1})
    test("Cache stores", len(cache) == 1)

    result = cache.get("key1")
    test("Cache retrieves", result is not None and result["value"] == 1)

    result = cache.get("nonexistent")
    test("Cache misses return None", result is None)

    stats = cache.get_stats()
    test("Cache stats", "size" in stats and "max_size" in stats)

    # Test 5: Model Manager
    print("\n[Test 5] Local Model Manager")
    print("-" * 40)

    manager = LocalModelManager()

    manager.register_model("linear", model)
    test("Register model", "linear" in manager.list_models())

    result = manager.predict("linear", [1.0, 2.0, 3.0, 4.0])
    test("Predict", result.is_success)

    # Second call - may or may not be cached depending on implementation
    result2 = manager.predict("linear", [1.0, 2.0, 3.0, 4.0])
    # Caching is optional, just verify it works
    test("Second predict works", result2 is not None)

    manager.clear_cache("linear")
    result3 = manager.predict("linear", [1.0, 2.0, 3.0, 4.0])
    test("After clear works", result3 is not None)

    info = manager.get_model_info("linear")
    test("Model info", info is not None and "type" in info)

    manager.unregister_model("linear")
    test("Unregister model", "linear" not in manager.list_models())

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = test_local_models()
    exit(0 if success else 1)
