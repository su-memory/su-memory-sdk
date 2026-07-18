"""
lite_pro 子模块共享基础设施

承载 lite_pro.py 拆分前的模块级全局（FAISS/numpy 探测、embedding 导入、停用词表）。
各子模块按需导入，避免重复定义与循环依赖。

设计原则（第一性原理）：
- 只放真正被多个子模块共享的对象；
- 独立类（Edge/MemoryNode/TemporalSystem/SessionManager）不依赖本模块，
  保持纯净、可单测；
- FAISS/np 的探测集中在此，子模块用 `from ._common import FAISS_AVAILABLE` 判断。
"""
from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger("su_memory.sdk.lite_pro")

# embedding 模块路径（兼容旧版 sys.path 注入）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from su_memory.sdk.embedding import (  # noqa: E402
    OllamaEmbedding,
    cosine_similarity,
    rrf_fusion,
)

# ── FAISS / numpy 可用性探测 ──────────────────────────────
try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError as e:  # pragma: no cover - 环境相关
    FAISS_AVAILABLE = False
    faiss = None
    np = None
    _faiss_import_error = e


def _check_and_suggest_faiss() -> bool:
    """检查 FAISS 是否可用，如不可用给出安装提示。"""
    if not FAISS_AVAILABLE:
        logger.warning(
            "⚠️  提示：FAISS 索引未安装\n"
            "当前状态：使用朴素搜索（线性扫描）\n"
            "安装 FAISS 可获得 O(log n) 搜索性能：\n"
            "  • pip install faiss-cpu        # CPU版本\n"
            "  • pip install faiss-gpu        # GPU加速（需CUDA）\n"
            "安装后请重启 Python 解释器以加载 FAISS"
        )
        return False
    return True


# ── 中文 / 英文停用词表 ──────────────────────────────────
STOP_WORDS = {
    '的', '了', '和', '是', '在', '有', '我', '你', '他', '她', '它',
    '这', '那', '都', '也', '就', '要', '会', '能', '对', '与', '及',
    '把', '被', '给', '但', '却', '而', '或', '而且', '并且', '所以',
    '因为', '如果', '虽然', '然后', '还是', '可以', '一个', '没有',
    '什么', '怎么', '这个', '那个', '一些', '已经', '非常', '可能',
}

__all__ = [
    "logger",
    "OllamaEmbedding",
    "cosine_similarity",
    "rrf_fusion",
    "FAISS_AVAILABLE",
    "faiss",
    "np",
    "_check_and_suggest_faiss",
    "STOP_WORDS",
    "ENGLISH_STOP_WORDS",
]


ENGLISH_STOP_WORDS = {
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
    'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been',
    'some', 'than', 'that', 'this', 'with', 'from', 'they', 'will',
    'when', 'what', 'which', 'their', 'about', 'into', 'other',
}
