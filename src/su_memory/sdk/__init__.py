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

__version__ = "1.7.7"

__all__ = [
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
    # 异常
    "SDKError",
    "MemoryNotFoundError",
    "EncodingError",
    "StorageError",
]
