"""
su-memory SDK 异常定义 — 向后兼容

自 v2.6.0 起，所有异常定义迁移至 su_memory.exceptions 统一模块。
本文件保持导出以兼容旧代码。
"""

# 向后兼容：从统一 exceptions 模块重新导出
from su_memory.exceptions import (  # noqa: F401
    ErrorCode,
    SuMemoryError,
    MemoryNotFoundError,
    EncodingError,
    StorageError,
    ConfigurationError,
    APIError,
)

# SDKError 别名为 SuMemoryError
SDKError = SuMemoryError
