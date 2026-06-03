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

    支持多种运行模式（环境变量驱动）:
    - local: 本地完整版 (默认)
    - cloud: 云端API版
    - edge: 边缘计算版
    - embedded: 嵌入式版

    环境变量:
    - SDK_MODE: 运行模式
    - SU_MEMORY_DATA_DIR: 数据持久化目录
    - SU_MEMORY_API_URL/API_URL: 云端API地址
    - SU_MEMORY_API_KEY/API_KEY: API密钥
    - SU_MEMORY_EMBEDDING_MODEL: 向量模型名称
    - OLLAMA_BASE_URL: Ollama服务地址 (local模式)
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
        """从环境变量创建配置 — F1-P1-4: docstring 与环境变量同步。

        读取的环境变量（11 项）：
        - SDK_MODE: 运行模式 (local/cloud/edge/embedded), 默认 local
        - SDK_STORAGE: 存储后端 (auto/sqlite/pgvector), 默认 auto
        - SU_MEMORY_DATA_DIR / PERSIST_DIR: 持久化目录, 默认 ./su_memory_data
        - SU_MEMORY_API_URL / API_URL: 云端 API 地址
        - SU_MEMORY_API_KEY / API_KEY: 云端 API 密钥
        - SDK_TIMEOUT: 请求超时秒数, 默认 30
        - SU_MEMORY_EMBEDDING_MODEL / EMBEDDING_MODEL: Embedding 模型名
        - EMBEDDING_DEVICE: Embedding 设备 (cpu/cuda), 默认 cpu
        - SDK_MAX_MEMORY_MB: 最大内存 MB, 默认 1024
        - SDK_MAX_INDEX_SIZE: 最大索引条目, 默认 100000
        - SDK_MAX_HOPS: 最大跳数, 默认 3

        Returns:
            SDKConfig 实例
        """
        return cls(
            mode=os.getenv("SDK_MODE", "local"),
            storage=os.getenv("SDK_STORAGE", "auto"),
            persist_dir=os.getenv("SU_MEMORY_DATA_DIR", os.getenv("PERSIST_DIR", "./su_memory_data")),
            api_url=os.getenv("SU_MEMORY_API_URL", os.getenv("API_URL", "https://api.sumemory.io")),
            api_key=os.getenv("SU_MEMORY_API_KEY", os.getenv("API_KEY", "")),
            timeout=int(os.getenv("SDK_TIMEOUT", "30")),
            embedding_model=os.getenv("SU_MEMORY_EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")),
            embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
            max_memory_mb=int(os.getenv("SDK_MAX_MEMORY_MB", "1024")),
            max_index_size=int(os.getenv("SDK_MAX_INDEX_SIZE", "100000")),
            max_hops=int(os.getenv("SDK_MAX_HOPS", "3")),
        )

    def is_cloud(self) -> bool:
        """是否为云端模式"""
        return self.mode == "cloud"

    def is_lite(self) -> bool:
        """是否为轻量模式"""
        return self.mode in ("edge", "embedded")
