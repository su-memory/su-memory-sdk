"""
su-memory SDK 异常定义
"""


class SDKError(Exception):
    """SDK基础异常"""
    pass


class MemoryNotFoundError(SDKError):
    """记忆未找到"""

    def __init__(self, memory_id: str):
        self.memory_id = memory_id
        super().__init__(f"Memory not found: {memory_id}")


class EncodingError(SDKError):
    """编码错误"""
    pass


class StorageError(SDKError):
    """存储错误"""
    pass


class ConfigurationError(SDKError):
    """配置错误"""
    pass


class APIError(SDKError):
    """API调用错误"""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API Error {status_code}: {message}")
