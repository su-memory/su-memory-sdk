"""
su-memory SDK - 对外赋能核心模块

此模块提供统一的SDK接口，支持:
- Python SDK (SuMemoryClient)
- 轻量级SDK (SuMemoryLite)
- 增强版SDK (SuMemoryLitePro) - 全面超越Hindsight v4.7/5
- 预测模块 (PredictionModule)
- 可解释性模块 (ExplainabilityModule)
- 贝叶斯增强器 (BayesianAugmenter) — 串联验证
- LangChain适配器
"""

from su_memory.sdk.client import SuMemoryClient
from su_memory.sdk.lite import SuMemoryLite
from su_memory.sdk.lite_pro import SuMemoryLitePro
from su_memory.sdk._memory_protocol import MemoryProtocol
from su_memory.sdk.config import SDKConfig
from su_memory.sdk.exceptions import (
    SDKError,
    MemoryNotFoundError,
    EncodingError,
    StorageError,
)

# 贝叶斯增强器（可选）
try:
    from su_memory.sdk.bayesian_augmenter import (
        BayesianAugmenter,
        EnhancedOutput,
        ComparisonDelta,
        AccuracyRecord,
    )
except ImportError:
    BayesianAugmenter = None
    EnhancedOutput = None
    ComparisonDelta = None
    AccuracyRecord = None

__version__ = "3.3.0"

__all__ = [
    # 核心协议
    "MemoryProtocol",
    # 核心客户端
    "SuMemoryClient",
    "SuMemoryLite",
    "SuMemoryLitePro",
    "SDKConfig",
    # 贝叶斯增强
    "BayesianAugmenter",
    "EnhancedOutput",
    "ComparisonDelta",
    "AccuracyRecord",
    # v3.0.0 存储后端
    "StorageBackend",
    "StorageConfig",
    "StorageMemory",
    "BackendType",
    "BackendHealth",
    "create_backend",
    "SqliteStorageBackend",
    "PgStorageBackend",
    "RedisStorageBackend",
    # 异常
    "SDKError",
    "MemoryNotFoundError",
    "EncodingError",
    "StorageError",
]

# v3.0.0: 存储后端 (lazy import, 避免强制依赖)
try:
    from su_memory._sys._storage_backend import (
        StorageBackend,
        StorageConfig,
        StorageMemory,
        BackendType,
        BackendHealth,
        create_backend,
    )
    from su_memory._sys._sqlite_storage import SqliteStorageBackend
except ImportError:
    StorageBackend = None
    StorageConfig = None
    StorageMemory = None
    BackendType = None
    BackendHealth = None
    create_backend = None
    SqliteStorageBackend = None

# PgStorageBackend / RedisStorageBackend (可选依赖: asyncpg / redis)
try:
    from su_memory._sys._pg_storage import PgStorageBackend
except ImportError:
    PgStorageBackend = None

try:
    from su_memory._sys._redis_storage import RedisStorageBackend
except ImportError:
    RedisStorageBackend = None
