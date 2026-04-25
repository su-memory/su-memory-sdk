"""
su-memory SDK 配置管理
"""
import os
from typing import Literal
from dataclasses import dataclass


@dataclass
class SDKConfig:
    """
    SDK配置类

    支持多种运行模式:
    - local: 本地完整版
    - cloud: 云端API版
    - edge: 边缘计算版
    - embedded: 嵌入式版
    """

    # 运行模式
    mode: Literal["local", "cloud", "edge", "embedded"] = "local"

    # 存储配置
    storage: Literal["auto", "memory", "sqlite", "redis"] = "auto"
    persist_dir: str = "./su_memory_data"

    # API配置 (云端模式)
    api_url: str = "https://api.sumemory.io"
    api_key: str = ""
    timeout: int = 30

    # 向量模型配置
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # 性能配置
    max_memory_mb: int = 1024
    max_index_size: int = 100000
    max_hops: int = 3

    @classmethod
    def from_env(cls) -> "SDKConfig":
        """从环境变量创建配置"""
        return cls(
            mode=os.getenv("SDK_MODE", "local"),
            api_url=os.getenv("API_URL", "https://api.sumemory.io"),
            api_key=os.getenv("API_KEY", ""),
            persist_dir=os.getenv("PERSIST_DIR", "./su_memory_data"),
        )

    def is_cloud(self) -> bool:
        """是否为云端模式"""
        return self.mode == "cloud"

    def is_lite(self) -> bool:
        """是否为轻量模式"""
        return self.mode in ("edge", "embedded")
