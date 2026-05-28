"""
su-memory 统一异常体系

提供 ErrorCode 枚举 + SuMemoryError 基类，统一所有模块的异常出口。

架构：
- ErrorCode: 按模块分组的错误码枚举（E=Error, W=Warning）
- SuMemoryError: 统一异常基类，携带 code/detail/hint
- 命名异常子类: 保持与 sdk/exceptions.py 向后兼容

使用方式：
    from su_memory.exceptions import SuMemoryError, ErrorCode
    raise SuMemoryError(ErrorCode.FAISS_INDEX_CORRUPTED, path="/tmp/index")
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional


# =============================================================================
# ErrorCode 枚举 — 按模块分组，含 code/标题/默认消息/建议
# =============================================================================

class ErrorCode(Enum):
    """统一错误码，格式: CATEGORY_TYPE### — E=Error, W=Warning"""

    # --- FAISS 向量索引 (FAISS_E001-E005) ---
    FAISS_INDEX_CORRUPTED = (
        "FAISS_E001",
        "FAISS索引文件已损坏",
        "请删除 {path} 后重新初始化",
    )
    FAISS_DIMENSION_MISMATCH = (
        "FAISS_E002",
        "向量维度不匹配",
        "期望 {expected} 维，实际 {actual} 维。请重建索引或统一嵌入后端",
    )
    FAISS_NOT_INSTALLED = (
        "FAISS_E003",
        "FAISS 未安装",
        "pip install faiss-cpu  # CPU版本\npip install faiss-gpu  # GPU版本",
    )
    FAISS_CREATE_FAILED = (
        "FAISS_E004",
        "FAISS 索引创建失败",
        "检查内存是否充足，或减小 max_memories 参数",
    )
    FAISS_PERSIST_FAILED = (
        "FAISS_E005",
        "FAISS 索引持久化失败",
        "检查磁盘空间和路径权限: {path}",
    )

    # --- 嵌入服务 (EMB_E001-E005) ---
    EMBED_UNAVAILABLE = (
        "EMB_E001",
        "所有嵌入后端均不可用",
        "请至少安装一个: pip install su-memory[ollama] 或设置 OPENAI_API_KEY",
    )
    EMBED_TIMEOUT = (
        "EMB_E002",
        "嵌入服务超时",
        "请检查 {backend} 服务是否运行，或增加超时配置",
    )
    EMBED_DIMENSION_MISMATCH = (
        "EMB_E003",
        "嵌入向量维度不匹配",
        "当前: {current}d，期望: {expected}d。请统一嵌入模型",
    )
    EMBED_CONVERSION_ERROR = (
        "EMB_E004",
        "向量转换失败",
        "原始数据格式异常: {error}",
    )
    EMBED_FALLBACK_EXHAUSTED = (
        "EMB_E005",
        "嵌入降级链已耗尽",
        "所有嵌入方案（{attempted}）均失败，请检查网络和API配置",
    )

    # --- 存储 (STO_E001-E004) ---
    STORAGE_WRITE_FAILED = (
        "STO_E001",
        "存储写入失败",
        "磁盘空间不足或路径权限问题: {path}",
    )
    STORAGE_READ_FAILED = (
        "STO_E002",
        "存储读取失败",
        "数据文件可能已损坏: {path}",
    )
    STORAGE_CORRUPT = (
        "STO_E003",
        "存储数据已损坏",
        "请备份后删除 {path}，重新初始化",
    )
    STORAGE_PATH_NOT_WRITABLE = (
        "STO_E004",
        "存储路径不可写",
        "请检查路径权限或使用其他目录: {path}",
    )

    # --- 查询 (QRY_E001-E003, QRY_W001-W003) ---
    QUERY_EMPTY = (
        "QRY_E001",
        "查询文本不能为空",
        "请提供有效的查询内容",
    )
    QUERY_NO_RESULTS = (
        "QRY_W001",
        "未找到匹配的记忆",
        "尝试更具体的查询词，或先添加一些记忆",
    )
    QUERY_INVALID_PARAMS = (
        "QRY_E002",
        "查询参数无效",
        "参数 {param} 的值 {value} 不在有效范围 [{min}, {max}]",
    )
    QUERY_TIMEOUT = (
        "QRY_W002",
        "查询操作超时",
        "可能是索引过大或嵌入服务响应慢，尝试缩小 top_k",
    )
    QUERY_EMPTY_KB = (
        "QRY_W003",
        "知识库为空",
        "请先使用 add() 或 add_batch() 添加记忆",
    )
    QUERY_FUSION_MODE_ERROR = (
        "QRY_E003",
        "不支持的融合模式",
        "fusion_mode 应为 vector_first / hybrid / graph_first 之一，实际: {mode}",
    )

    # --- 图谱 (GPH_E001-E003) ---
    GRAPH_NODE_NOT_FOUND = (
        "GPH_E001",
        "图谱节点未找到",
        "节点 ID {node_id} 不存在",
    )
    GRAPH_EDGE_NOT_FOUND = (
        "GPH_E002",
        "图谱边未找到",
        "边 ({source} -> {target}) 不存在",
    )
    GRAPH_CYCLE_DETECTED = (
        "GPH_E003",
        "检测到图谱环路",
        "添加边 ({source} -> {target}) 会导致环路，已拒绝",
    )

    # --- 并发 (CON_E001-E002) ---
    CONCURRENCY_DEADLOCK = (
        "CON_E001",
        "检测到死锁",
        "请减少并发写入线程数，当前: {threads}",
    )
    CONCURRENCY_RACE_CONDITION = (
        "CON_E002",
        "数据竞争检测",
        "并发操作导致数据不一致，操作: {operation}",
    )

    # --- 配置 (CFG_E001-E003) ---
    CONFIG_INVALID_PARAM = (
        "CFG_E001",
        "配置参数无效",
        "参数 {param} = {value}，原因: {reason}",
    )
    CONFIG_MISSING_REQUIRED = (
        "CFG_E002",
        "缺少必要配置项",
        "请设置 {param}",
    )
    CONFIG_TYPE_MISMATCH = (
        "CFG_E003",
        "配置类型不匹配",
        "参数 {param} 期望 {expected_type}，实际 {actual_type}",
    )

    # --- 时序 (TMP_E001-E002) ---
    TEMPORAL_INVALID_RANGE = (
        "TMP_E001",
        "时间范围无效",
        "start({start}) 不能晚于 end({end})",
    )
    TEMPORAL_EVENT_CONFLICT = (
        "TMP_E002",
        "时序事件冲突",
        "事件 {event_id} 与已有事件时间重叠",
    )

    # --- 会话 (SES_E001-E002) ---
    SESSION_NOT_FOUND = (
        "SES_E001",
        "会话未找到",
        "会话 ID {session_id} 不存在",
    )
    SESSION_BRIDGE_ERROR = (
        "SES_E002",
        "会话桥接失败",
        "无法在会话 {from_id} 和 {to_id} 之间建立桥接",
    )

    # --- 插件 (PLG_E001-E003) ---
    PLUGIN_LOAD_FAILED = (
        "PLG_E001",
        "插件加载失败",
        "插件 {plugin_name}: {error}",
    )
    PLUGIN_DEPENDENCY = (
        "PLG_E002",
        "插件依赖缺失",
        "插件 {plugin_name} 缺少依赖: {missing}",
    )
    PLUGIN_STATE_ERROR = (
        "PLG_E003",
        "插件状态错误",
        "插件 {plugin_name} 当前状态 {current}，期望 {expected}",
    )

    # --- 数据迁移 (MIG_E001-E002) ---
    MIGRATION_FAILED = (
        "MIG_E001",
        "数据迁移失败",
        "从 {source} 迁移时出错: {error}",
    )
    MIGRATION_FORMAT_ERROR = (
        "MIG_E002",
        "数据格式不支持",
        "不支持 {format} 格式，支持: json/csv/sqlite/obsidian",
    )

    # --- 记忆管理 (MEM_E001-E003) ---
    MEMORY_OVERFLOW = (
        "MEM_E001",
        "记忆数量超限",
        "当前 {current}/{max_memories}，请增加限制或启用自动清理",
    )
    MEMORY_NOT_FOUND = (
        "MEM_E002",
        "记忆未找到",
        "记忆 ID {memory_id} 不存在",
    )
    MEMORY_DUPLICATE = (
        "MEM_W001",
        "记忆重复检测",
        "记忆内容与已有记忆 (ID: {existing_id}) 高度相似 ({similarity:.1%})",
    )

    # --- 预测 (PRD_E001-E002) ---
    PREDICTION_MODEL_UNAVAILABLE = (
        "PRD_E001",
        "预测模型不可用",
        "请安装 scikit-learn: pip install scikit-learn",
    )
    PREDICTION_INSUFFICIENT_DATA = (
        "PRD_E002",
        "预测数据不足",
        "至少需要 {min_points} 个数据点，当前: {current}",
    )

    @property
    def code(self) -> str:
        """错误码字符串 (e.g. FAISS_E001)"""
        return self.value[0]

    @property
    def title(self) -> str:
        """错误标题 (中文)"""
        return self.value[1]

    @property
    def suggestion(self) -> str:
        """修复建议"""
        return self.value[2]

    @property
    def is_warning(self) -> bool:
        """是否为 Warning 级别"""
        return "_W" in self.code

    def format_message(self, **kwargs) -> str:
        """用参数格式化默认消息"""
        msg = self.title
        if kwargs:
            try:
                suggestion = self.suggestion.format(**kwargs)
            except KeyError:
                suggestion = self.suggestion
        else:
            suggestion = self.suggestion
        if suggestion:
            msg += f"。{suggestion}"
        return msg


# =============================================================================
# SuMemoryError — 统一异常基类
# =============================================================================

class SuMemoryError(Exception):
    """su-memory 所有异常的基类

    携带 ErrorCode、格式化详情、可选的修复提示。

    Attributes:
        code: ErrorCode 枚举值
        detail: 格式化后的错误详情
        hint: 可选的修复建议 dict
        context: 附加上下文字典

    Example:
        >>> raise SuMemoryError(
        ...     ErrorCode.FAISS_DIMENSION_MISMATCH,
        ...     expected=768, actual=1024
        ... )
        SuMemoryError: [FAISS_E002] 向量维度不匹配。期望 768 维，实际 1024 维。请重建索引或统一嵌入后端

        >>> raise SuMemoryError(
        ...     ErrorCode.MEMORY_OVERFLOW,
        ...     current=10000, max_memories=10000
        ... )
        SuMemoryError: [MEM_E001] 记忆数量超限。当前 10000/10000，请增加限制或启用自动清理
    """

    def __init__(
        self,
        code: ErrorCode,
        detail: Optional[str] = None,
        hint: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        self.code = code
        self.hint = hint

        if detail is not None:
            self.detail = detail
        else:
            self.detail = code.format_message(**kwargs)

        # 存储格式化参数，方便上层捕获后调整
        self.context = kwargs

        super().__init__(f"[{code.code}] {self.detail}")

    @classmethod
    def from_error(cls, error: Exception, code: ErrorCode, detail: Optional[str] = None) -> "SuMemoryError":
        """从现有异常创建 SuMemoryError，保留异常链"""
        return cls(code, detail=detail or str(error))


# =============================================================================
# 命名异常子类 — 向后兼容 sdk/exceptions.py
# =============================================================================

class MemoryNotFoundError(SuMemoryError):
    """记忆未找到"""

    def __init__(self, memory_id: str):
        super().__init__(
            ErrorCode.MEMORY_NOT_FOUND,
            memory_id=memory_id,
        )


class EncodingError(SuMemoryError):
    """编码错误"""

    def __init__(self, detail: str = "编码操作失败"):
        super().__init__(ErrorCode.EMBED_CONVERSION_ERROR, detail=detail)


class StorageError(SuMemoryError):
    """存储错误"""

    def __init__(self, detail: str = "存储操作失败", code: ErrorCode = ErrorCode.STORAGE_WRITE_FAILED):
        super().__init__(code, detail=detail)


class ConfigurationError(SuMemoryError):
    """配置错误"""

    def __init__(self, detail: str = "配置无效", param: str = None):
        if param:
            super().__init__(ErrorCode.CONFIG_INVALID_PARAM, param=param, reason=detail)
        else:
            super().__init__(ErrorCode.CONFIG_INVALID_PARAM, detail=detail)


class APIError(SuMemoryError):
    """API调用错误"""

    def __init__(self, status_code: int, message: str):
        super().__init__(
            ErrorCode.EMBED_TIMEOUT if status_code >= 500 else ErrorCode.EMBED_UNAVAILABLE,
            detail=f"API Error {status_code}: {message}",
        )


# =============================================================================
# 便捷导出
# =============================================================================

__all__ = [
    "ErrorCode",
    "SuMemoryError",
    "MemoryNotFoundError",
    "EncodingError",
    "StorageError",
    "ConfigurationError",
    "APIError",
]
