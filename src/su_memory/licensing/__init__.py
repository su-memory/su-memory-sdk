"""
su-memory SDK 容量包与许可证管理

实现容量限制和容量包扩展机制，支持：
- 容量包类型定义
- 使用量追踪
- 许可证验证
- 升级提示
"""

from typing import Dict, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import time
import os


class LicenseType(Enum):
    """许可证类型"""
    COMMUNITY = "community"      # 社区版
    STARTER = "starter"         # 入门版
    PRO = "pro"                 # 专业版
    ENTERPRISE = "enterprise"   # 企业版


@dataclass
class CapacityPackage:
    """容量包配置"""
    name: str
    memories: int               # 最大记忆数量
    sessions: int               # 最大会话数
    api_calls_per_month: int    # 每月 API 调用次数 (-1 表示无限制)
    embedding_quota: int        # 嵌入生成配额 (-1 表示无限制)
    features: List[str]         # 启用的高级功能
    price: float                # 价格（仅用于展示）


# 预定义容量包
CAPACITY_PACKAGES: Dict[str, CapacityPackage] = {
    "community": CapacityPackage(
        name="社区版",
        memories=10000,           # 1万条记忆
        sessions=100,             # 100个会话
        api_calls_per_month=10000,
        embedding_quota=100000,
        features=["basic_query", "tfidf", "session_basic"],
        price=0
    ),
    "starter": CapacityPackage(
        name="入门版",
        memories=50000,           # 5万条记忆
        sessions=500,
        api_calls_per_month=50000,
        embedding_quota=500000,
        features=["basic_query", "tfidf", "session_basic", "vector_search"],
        price=29.9
    ),
    "pro": CapacityPackage(
        name="专业版",
        memories=200000,          # 20万条记忆
        sessions=-1,              # 无限制
        api_calls_per_month=-1,    # 无限制
        embedding_quota=-1,       # 无限制
        features=["basic_query", "tfidf", "session_basic", "vector_search",
                  "multihop", "causal_inference", "temporal", "prediction"],
        price=99.9
    ),
    "enterprise": CapacityPackage(
        name="企业版",
        memories=-1,              # 无限制
        sessions=-1,
        api_calls_per_month=-1,
        embedding_quota=-1,
        features=["*"],           # 所有功能
        price=0  # 定制价格
    )
}


@dataclass
class UsageStats:
    """使用统计"""
    current_memories: int = 0
    current_sessions: int = 0
    total_api_calls: int = 0
    api_calls_this_month: int = 0
    embeddings_generated: int = 0
    month_start: int = field(default_factory=lambda: int(time.time() // (30*24*3600) * (30*24*3600)))


@dataclass
class LicenseInfo:
    """许可证信息"""
    license_type: LicenseType
    package: CapacityPackage
    license_key: Optional[str] = None
    issued_at: int = field(default_factory=lambda: int(time.time()))
    expires_at: Optional[int] = None
    usage: UsageStats = field(default_factory=UsageStats)


class CapacityManager:
    """
    容量管理器

    追踪使用量，检查容量限制，提供升级建议

    Example:
        >>> from su_memory.licensing import CapacityManager, LicenseType
        >>>
        >>> # 初始化（社区版）
        >>> manager = CapacityManager()
        >>>
        >>> # 检查是否可添加记忆
        >>> if manager.can_add_memory():
        ...     pro.add("新记忆")
        ... else:
        ...     print("已达到容量限制")
        ...     print(manager.get_upgrade_suggestion())
        >>>
        >>> # 获取当前状态
        >>> stats = manager.get_usage_stats()
        >>> print(f"已使用: {stats.current_memories}/{manager.package.memories}")
    """

    def __init__(
        self,
        license_type: LicenseType = LicenseType.COMMUNITY,
        license_key: Optional[str] = None,
        storage_path: Optional[str] = None
    ):
        """
        初始化容量管理器

        Args:
            license_type: 许可证类型
            license_key: 许可证密钥（可选）
            storage_path: 存储路径
        """
        self._license_type = license_type
        self._license_key = license_key
        self._package = CAPACITY_PACKAGES.get(license_type.value, CAPACITY_PACKAGES["community"])

        # 加载使用统计
        self._usage = UsageStats()
        self._load_usage(storage_path)

        # 检查月结
        self._check_monthly_reset()

    def _load_usage(self, storage_path: Optional[str]):
        """加载使用统计"""
        if not storage_path:
            return

        usage_file = os.path.join(storage_path, ".usage_stats")
        if os.path.exists(usage_file):
            try:
                import json
                with open(usage_file, "r") as f:
                    data = json.load(f)
                    self._usage = UsageStats(**data)
            except Exception:
                pass

    def _save_usage(self, storage_path: Optional[str]):
        """保存使用统计"""
        if not storage_path:
            return

        try:
            import json
            usage_file = os.path.join(storage_path, ".usage_stats")
            with open(usage_file, "w") as f:
                json.dump({
                    "current_memories": self._usage.current_memories,
                    "current_sessions": self._usage.current_sessions,
                    "total_api_calls": self._usage.total_api_calls,
                    "api_calls_this_month": self._usage.api_calls_this_month,
                    "embeddings_generated": self._usage.embeddings_generated,
                    "month_start": self._usage.month_start
                }, f)
        except Exception:
            pass

    def _check_monthly_reset(self):
        """检查并重置月统计"""
        current_month = int(time.time() // (30*24*3600) * (30*24*3600))
        if current_month > self._usage.month_start:
            self._usage.api_calls_this_month = 0
            self._usage.month_start = current_month

    def can_add_memory(self) -> bool:
        """检查是否可以添加新记忆"""
        if self._package.memories < 0:
            return True  # 无限制
        return self._usage.current_memories < self._package.memories

    def can_create_session(self) -> bool:
        """检查是否可以创建新会话"""
        if self._package.sessions < 0:
            return True  # 无限制
        return self._usage.current_sessions < self._package.sessions

    def can_make_api_call(self) -> bool:
        """检查是否可以发起 API 调用"""
        if self._package.api_calls_per_month < 0:
            return True  # 无限制
        return self._usage.api_calls_this_month < self._package.api_calls_per_month

    def can_generate_embedding(self) -> bool:
        """检查是否可以生成嵌入"""
        if self._package.embedding_quota < 0:
            return True  # 无限制
        return self._usage.embeddings_generated < self._package.embedding_quota

    def record_memory_add(self):
        """记录添加记忆"""
        self._usage.current_memories += 1

    def record_session_create(self):
        """记录创建会话"""
        self._usage.current_sessions += 1

    def record_api_call(self):
        """记录 API 调用"""
        self._usage.total_api_calls += 1
        self._usage.api_calls_this_month += 1

    def record_embedding(self):
        """记录嵌入生成"""
        self._usage.embeddings_generated += 1

    def get_usage_stats(self) -> UsageStats:
        """获取使用统计"""
        return self._usage

    def get_package_info(self) -> CapacityPackage:
        """获取容量包信息"""
        return self._package

    def get_usage_percentage(self) -> Dict[str, float]:
        """获取各维度的使用百分比"""
        result = {}

        if self._package.memories >= 0:
            result["memories"] = self._usage.current_memories / self._package.memories * 100
        else:
            result["memories"] = 0

        if self._package.sessions >= 0:
            result["sessions"] = self._usage.current_sessions / self._package.sessions * 100
        else:
            result["sessions"] = 0

        if self._package.api_calls_per_month >= 0:
            result["api_calls"] = self._usage.api_calls_this_month / self._package.api_calls_per_month * 100
        else:
            result["api_calls"] = 0

        if self._package.embedding_quota >= 0:
            result["embeddings"] = self._usage.embeddings_generated / self._package.embedding_quota * 100
        else:
            result["embeddings"] = 0

        return result

    def is_feature_enabled(self, feature: str) -> bool:
        """检查功能是否启用"""
        features = self._package.features
        return "*" in features or feature in features

    def get_upgrade_suggestion(self) -> str:
        """获取升级建议"""
        usage = self.get_usage_percentage()

        suggestions = []

        # 检查各项使用量
        if usage.get("memories", 0) >= 80:
            suggestions.append("记忆容量已使用超过 80%")

        if usage.get("sessions", 0) >= 80:
            suggestions.append("会话数已使用超过 80%")

        if usage.get("api_calls", 0) >= 80:
            suggestions.append("API 调用已使用超过 80%")

        if usage.get("embeddings", 0) >= 80:
            suggestions.append("嵌入配额已使用超过 80%")

        if not suggestions:
            return "当前使用量正常，无需升级。"

        # 构建升级建议
        lines = [
            "\n" + "=" * 50,
            "📊 容量使用警告",
            "=" * 50,
            "",
            "当前版本: " + self._package.name,
            ""
        ]

        for s in suggestions:
            lines.append(f"⚠️  {s}")

        lines.extend([
            "",
            "💡 升级建议:",
        ])

        # 建议升级到更高的版本
        if self._license_type == LicenseType.COMMUNITY:
            lines.append("   升级到 入门版 (¥29.9/月) 获得 5万条记忆容量")
            lines.append("   升级到 专业版 (¥99.9/月) 获得 20万条记忆 + 多跳推理")
        elif self._license_type == LicenseType.STARTER:
            lines.append("   升级到 专业版 (¥99.9/月) 获得 20万条记忆 + 多跳推理 + 时序预测")
        elif self._license_type == LicenseType.PRO:
            lines.append("   升级到 企业版 获取无限制容量 + 专属支持")

        lines.extend([
            "",
            "升级地址: https://su-memory.ai/pricing",
            "=" * 50
        ])

        return "\n".join(lines)

    def verify_license(self, license_key: str) -> bool:
        """
        验证许可证密钥

        Args:
            license_key: 许可证密钥

        Returns:
            是否验证通过
        """
        # 简单的密钥验证（实际应使用服务端验证）
        if not license_key or len(license_key) < 10:
            return False

        # 生成密钥哈希
        hashlib.sha256(license_key.encode()).hexdigest()[:16]

        # 验证格式
        if not license_key.startswith("SM-"):
            return False

        # 更新许可证信息
        self._license_key = license_key

        # 根据密钥判断类型（简化版）
        if "ENT" in license_key.upper():
            self._license_type = LicenseType.ENTERPRISE
            self._package = CAPACITY_PACKAGES["enterprise"]
        elif "PRO" in license_key.upper():
            self._license_type = LicenseType.PRO
            self._package = CAPACITY_PACKAGES["pro"]
        elif "STD" in license_key.upper():
            self._license_type = LicenseType.STARTER
            self._package = CAPACITY_PACKAGES["starter"]
        else:
            self._license_type = LicenseType.COMMUNITY
            self._package = CAPACITY_PACKAGES["community"]

        return True


def create_capacity_manager(
    license_key: Optional[str] = None,
    storage_path: Optional[str] = None
) -> CapacityManager:
    """
    创建容量管理器的便捷函数

    自动从环境变量或配置文件读取许可证信息

    Args:
        license_key: 许可证密钥
        storage_path: 存储路径

    Returns:
        CapacityManager 实例
    """
    # 从环境变量获取许可证密钥
    if not license_key:
        license_key = os.environ.get("SU_MEMORY_LICENSE_KEY")

    # 从环境变量获取许可证类型
    license_type_str = os.environ.get("SU_MEMORY_LICENSE_TYPE", "community").lower()
    try:
        license_type = LicenseType(license_type_str)
    except ValueError:
        license_type = LicenseType.COMMUNITY

    manager = CapacityManager(
        license_type=license_type,
        license_key=license_key,
        storage_path=storage_path
    )

    # 如果提供了密钥，尝试验证
    if license_key:
        manager.verify_license(license_key)

    return manager


# 导出
__all__ = [
    "LicenseType",
    "CapacityPackage",
    "UsageStats",
    "LicenseInfo",
    "CapacityManager",
    "CAPACITY_PACKAGES",
    "create_capacity_manager",
]
