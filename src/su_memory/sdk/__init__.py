"""
su-memory SDK - 对外赋能核心模块

此模块提供统一的SDK接口，支持:
- Python SDK (SuMemoryClient)
- 统一引擎 (SuMemory) — v4.0 单一产品线, 含向量/多跳/因果/时空全部能力
  (SuMemoryLite / SuMemoryLitePro 为向后兼容别名)
- 预测模块 (PredictionModule)
- 可解释性模块 (ExplainabilityModule)
- 贝叶斯增强器 (BayesianAugmenter) — 串联验证
- LangChain适配器
"""

from su_memory.sdk._memory_protocol import MemoryProtocol
from su_memory.sdk.client import SuMemoryClient
from su_memory.sdk.config import SDKConfig
from su_memory.sdk.exceptions import (
    EncodingError,
    MemoryNotFoundError,
    SDKError,
    StorageError,
)
from su_memory.sdk.lite import SuMemoryLite
from su_memory.sdk.lite_pro import SuMemoryLitePro
from su_memory.sdk.unified import SuMemory
from su_memory.sdk.multi_hop_reader import MultiHopReader

# v4.0: 统一产品线, 取消 Lite/LitePro 区分.
# SuMemory 是唯一主类 (含全部能力). Lite/LitePro 保留为向后兼容别名.
SuMemoryLite = SuMemory  # 兼容别名 (旧代码 `from su_memory.sdk import SuMemoryLite` 仍可用)
SuMemoryLitePro = SuMemory  # 兼容别名

# 贝叶斯增强器（可选）
try:
    from su_memory.sdk.bayesian_augmenter import (
        AccuracyRecord,
        BayesianAugmenter,
        ComparisonDelta,
        EnhancedOutput,
    )
except ImportError:
    BayesianAugmenter = None
    EnhancedOutput = None
    ComparisonDelta = None
    AccuracyRecord = None

__version__ = "4.0.0"

__all__ = [
    # 核心协议
    "MemoryProtocol",
    # 核心客户端
    "SuMemory",
    "MultiHopReader",
    "SuMemoryClient",
    "SuMemoryLite",  # 兼容别名
    "SuMemoryLitePro",  # 兼容别名
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
    from su_memory._sys._sqlite_storage import SqliteStorageBackend
    from su_memory._sys._storage_backend import (
        BackendHealth,
        BackendType,
        StorageBackend,
        StorageConfig,
        StorageMemory,
        create_backend,
    )
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
