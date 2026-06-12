"""
su-memory v4.1 — BM25 词法检索模块

基于 rank_bm25 库实现 OKapi BM25 词法检索，为 su-memory 提供精确
词法匹配信号，弥补纯语义检索在专有名词/数字/ID 等精确匹配上的不足。

核心能力:
- 英文分词 + 停用词过滤
- 增量索引更新（add 时同步写入）
- Top-K 检索 + 分数归一化
- 与现有 RRF 融合管道无缝集成

参考: agentmemory V4 (BM25 权重 0.12), Hindsight (稀疏+密集双路)
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 尝试导入 rank_bm25
# ---------------------------------------------------------------------------

_BM25_AVAILABLE = False
try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    BM25Okapi = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# 英文分词与停用词
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    'a', 'an', 'the', 'and', 'or', 'but', 'not', 'in', 'on', 'at', 'to',
    'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'can', 'may', 'might', 'must', 'shall',
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
    'she', 'her', 'it', 'its', 'they', 'them', 'their', 'this', 'that',
    'these', 'those', 'what', 'which', 'who', 'whom', 'how', 'when',
    'where', 'why', 'if', 'then', 'so', 'than', 'too', 'very', 'just',
    'also', 'about', 'up', 'out', 'all', 'no', 'nor', 'not', 'only',
    'own', 'same', 'such', 'other', 'more', 'most', 'some', 'any',
    'into', 'over', 'after', 'before', 'between', 'under', 'again',
    'further', 'once', 'here', 'there', 'both', 'each', 'few', 'many',
    'much', 'several', 'during', 'through', 'above', 'below',
})


def _tokenize(text: str) -> list[str]:
    """英文分词: 小写 + 去停用词 + 保留 2+ 字符 token"""
    tokens = re.findall(r'[a-z0-9]{2,}', text.lower())
    return [t for t in tokens if t not in _STOP_WORDS]


# ---------------------------------------------------------------------------
# BM25Searcher 主类
# ---------------------------------------------------------------------------

class BM25Searcher:
    """BM25 词法检索器 — 增量索引 + Top-K 检索

    用法:
        searcher = BM25Searcher()
        searcher.add("mem_001", "Alice moved to Boston in March")
        searcher.add("mem_002", "Bob works at Google in New York")
        results = searcher.search("Where does Alice live?", top_k=5)
        # → [("mem_001", 1.23), ...]
    """

    def __init__(self, max_docs: int = 50000):
        self._doc_ids: list[str] = []
        self._doc_texts: list[str] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: Any | None = None
        self._max_docs = max_docs
        self._dirty = False  # 标记是否需要重建索引

    @property
    def available(self) -> bool:
        """BM25 是否可用（rank_bm25 库已安装）"""
        return _BM25_AVAILABLE

    @property
    def doc_count(self) -> int:
        return len(self._doc_ids)

    def add(self, doc_id: str, text: str) -> None:
        """添加文档到 BM25 索引"""
        if not _BM25_AVAILABLE:
            return
        if not text or not text.strip():
            return

        self._doc_ids.append(doc_id)
        self._doc_texts.append(text)
        self._tokenized_corpus.append(_tokenize(text))
        self._dirty = True

        # 裁剪超量文档
        if len(self._doc_ids) > self._max_docs:
            excess = len(self._doc_ids) - self._max_docs
            self._doc_ids = self._doc_ids[excess:]
            self._doc_texts = self._doc_texts[excess:]
            self._tokenized_corpus = self._tokenized_corpus[excess:]

    def _ensure_index(self) -> None:
        """确保 BM25 索引是最新的"""
        if not _BM25_AVAILABLE:
            return
        if self._dirty or self._bm25 is None:
            if self._tokenized_corpus:
                self._bm25 = BM25Okapi(self._tokenized_corpus)
            else:
                self._bm25 = None
            self._dirty = False

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """检索与 query 最相关的 top_k 个文档

        Args:
            query: 查询文本
            top_k: 返回的最大文档数

        Returns:
            [(doc_id, score), ...] 按分数降序排列
        """
        if not _BM25_AVAILABLE or not self._doc_ids:
            return []

        self._ensure_index()
        if self._bm25 is None:
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)

        # 归一化分数到 [0, 1] 范围（min-max normalization）
        try:
            import numpy as np
            scores_np = np.array(scores, dtype=float)
        except ImportError:
            scores_np = None

        if scores_np is not None and len(scores_np) > 0:
            min_score = float(scores_np.min())
            max_score = float(scores_np.max())
        else:
            min_score = min(scores) if scores else 0.0
            max_score = max(scores) if scores else 0.0

        if max_score > min_score:
            if scores_np is not None:
                normalized = (scores_np - min_score) / (max_score - min_score)
            else:
                normalized = [(s - min_score) / (max_score - min_score) for s in scores]
        else:
            if scores_np is not None:
                normalized = scores_np
            else:
                normalized = scores

        # 按分数降序取 top_k
        if scores_np is not None:
            indexed = list(enumerate(normalized.tolist()))
        else:
            indexed = list(enumerate(normalized))
        indexed.sort(key=lambda x: x[1], reverse=True)

        results: list[tuple[str, float]] = []
        for idx, score in indexed[:top_k]:
            if score > 0:
                results.append((self._doc_ids[idx], float(score)))

        return results

    def search_with_content(
        self, query: str, top_k: int = 20
    ) -> list[tuple[str, str, float]]:
        """检索并返回文档内容

        Returns:
            [(doc_id, content, score), ...]
        """
        id_score_pairs = self.search(query, top_k)
        results: list[tuple[str, str, float]] = []
        for doc_id, score in id_score_pairs:
            # 通过 doc_id 查找 content
            try:
                idx = self._doc_ids.index(doc_id)
                content = self._doc_texts[idx]
                results.append((doc_id, content, score))
            except ValueError:
                continue
        return results

    def clear(self) -> None:
        """清空索引"""
        self._doc_ids.clear()
        self._doc_texts.clear()
        self._tokenized_corpus.clear()
        self._bm25 = None
        self._dirty = False
