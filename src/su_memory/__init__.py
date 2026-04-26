"""
su_memory — Semantic Memory Engine

一行代码让AI应用拥有记忆能力。

Example:
    >>> from su_memory import SuMemory
    >>> client = SuMemory()
    >>> client.add("这个项目的ROI增长了25%")
    >>> results = client.query("投资汇报关")
"""

__version__ = "1.3.0"

# 环境检测：确保安装正确
import os
import sys
import shutil
import site as _site

# 检查 pip 和 python 环境一致性
def _check_installation():
    """检查安装环境"""
    python_path = os.path.dirname(os.path.dirname(sys.executable))
    pip_path = shutil.which("pip")
    
    if pip_path:
        pip_dir = os.path.dirname(os.path.dirname(pip_path))
        if python_path != pip_dir:
            import warnings
            warnings.warn(
                f"\n⚠️  su-memory 安装环境警告:\n"
                f"   Python: {sys.executable}\n"
                f"   pip:    {pip_path}\n"
                f"   pip 和 python 可能指向不同环境。\n"
                f"   建议使用: python -m pip install su-memory\n",
                UserWarning,
                stacklevel=2
            )

# 仅在首次导入时检测（可设置环境变量跳过）
if not os.environ.get("SU_MEMORY_SKIP_ENV_CHECK"):
    try:
        _check_installation()
    except Exception:
        pass  # 静默忽略检查错误，避免影响正常功能

from su_memory.client import SuMemory

from su_memory.sdk import SuMemoryLite, SuMemoryLitePro

# 导入增强检索器
try:
    from su_memory.sdk.enhanced_retriever import EnhancedRetriever
except ImportError:
    EnhancedRetriever = None

# 导入 VectorGraphRAG
try:
    from su_memory.sdk.vector_graph_rag import VectorGraphRAG, create_vector_graph_rag
except ImportError:
    VectorGraphRAG = None
    create_vector_graph_rag = None

from su_memory.core import (
    CausalChain,
    CausalInference,
    MetaCognition,
    SuCompressor,
    BeliefTracker,
    DynamicPriorityCalculator,
    SemanticEncoder,
    EncoderCore,
    EncodingInfo,
    IntentClassifier,
    IntentConfig,
    ProgressiveDisclosure,
    SessionBridge,
    SessionContext,
    RecencyFeedbackSystem,
    MultiHopRetriever,
    HopResult,
    WikiLinker,
    WikiResult,
    RecallTrigger,
    RecallResult,
    RecallResponse,
)
from su_memory.encoding import MemoryEncoding

__all__ = [
    # SDK客户端
    "SuMemoryLite",
    "SuMemoryLitePro",
    
    "SuMemory",
    
    # 增强检索器
    "EnhancedRetriever",
    
    # VectorGraphRAG 多跳推理
    "VectorGraphRAG",
    "create_vector_graph_rag",
    
    "CausalChain",
    "CausalInference",
    "MetaCognition",
    "SuCompressor",
    "BeliefTracker",
    "DynamicPriorityCalculator",
    "SemanticEncoder",
    "EncoderCore",
    "EncodingInfo",
    "MemoryEncoding",
    "IntentClassifier",
    "IntentConfig",
    "ProgressiveDisclosure",
    "SessionBridge",
    "SessionContext",
    "RecencyFeedbackSystem",
    "MultiHopRetriever",
    "HopResult",
    "WikiLinker",
    "WikiResult",
    "RecallTrigger",
    "RecallResult",
    "RecallResponse",
    # 数据迁移
    "MemoryMigrator",
    "MemoryRecord",
    "MigrationReport",
    "DataSourceType",
    "migrate_json",
    "migrate_csv",
    "migrate_sqlite",
    "migrate_obsidian",
]

# 数据迁移模块
from su_memory._sys.migrator import (
    MemoryMigrator,
    MemoryRecord,
    MigrationReport,
    DataSourceType,
    migrate_json,
    migrate_csv,
    migrate_sqlite,
    migrate_obsidian,
)

