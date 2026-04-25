"""
su-memory SDK 轻量级版本
适用于资源受限环境（嵌入式设备/移动端）
内存占用：<50MB
"""
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict
import uuid
import math
import re
import json
import os
from functools import lru_cache


# 中文停用词表
STOP_WORDS = {
    '的', '了', '和', '是', '在', '有', '我', '你', '他', '她', '它',
    '这', '那', '都', '也', '就', '要', '会', '能', '对', '与', '及',
    '把', '被', '给', '但', '却', '而', '或', '而且', '并且', '所以',
    '因为', '如果', '虽然', '然后', '还是', '可以', '一个', '没有',
    '什么', '怎么', '这个', '那个', '一些', '已经', '非常', '可能',
    '应该', '可能', '知道', '觉得', '现在', '时候', '这里', '那里',
    '他们', '她们', '我们', '自己', '不是', '只是', '不能', '如果',
    '通过', '进行', '使用', '支持', '提供', '需要', '根据', '按照',
    '由于', '关于', '对于', '以及', '或者', '而且', '不过', '然而',
    '因此', '所以', '那么', '因此', '之后', '之前', '之后', '当时',
    '一直', '一种', '这种', '两种', '每个', '各种', '其他', '另外',
    '其中', '之间', '以后', '以前', '只有', '才能', '只有', '一定',
    '比较', '更加', '特别', '尤其', '主要', '一般', '基本', '例如',
    '比如', '包括', '就是', '不是', '不同', '相同', '同时', '另外'
}


class SuMemoryLite:
    """
    轻量级SDK客户端

    适用于资源受限环境:
    - 嵌入式设备 (树莓派/Arduino)
    - 移动端 (iOS/Android)
    - 边缘计算节点

    特点:
    - 内存占用 <50MB
    - 模型大小 <20MB
    - 纯Python实现，无需额外依赖
    - 支持TF-IDF相似度计算
    - 支持持久化存储

    Example:
        >>> from su_memory.sdk import SuMemoryLite
        >>> client = SuMemoryLite()
        >>> mid = client.add("天气很好")
        >>> results = client.query("天气")
    """

    def __init__(
        self,
        max_memories: int = 10000,
        storage_path: Optional[str] = None,
        enable_tfidf: bool = True,
        enable_persistence: bool = True,
        cache_size: int = 128
    ):
        """
        初始化轻量级客户端

        Args:
            max_memories: 最大记忆数量
            storage_path: 持久化存储路径（可选）
            enable_tfidf: 是否启用TF-IDF评分（默认启用）
            enable_persistence: 是否启用持久化（默认启用）
            cache_size: 查询缓存大小（默认128）
        """
        self.max_memories = max_memories
        self.storage_path = storage_path
        self.enable_tfidf = enable_tfidf
        self.enable_persistence = enable_persistence
        self._memories: List[Dict[str, Any]] = []
        self._index: Dict[str, set] = {}  # 使用set去重
        self._doc_freq: Dict[str, int] = {}  # 文档频率（用于TF-IDF）
        self._total_docs: int = 0
        
        # LRU查询缓存
        self._cache_size = cache_size
        self._query_cache: OrderedDict[Tuple[str, int], List[Dict[str, Any]]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        
        # 加载已有数据
        if enable_persistence and storage_path:
            self._load()

    def _tokenize(self, text: str) -> List[str]:
        """
        中文分词（简单实现）
        
        使用N-gram滑动窗口进行简单分词。
        对于生产环境，建议使用jieba等专业分词库。
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果列表（去重）
        """
        # 去除标点符号
        text_clean = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text.lower())
        
        keywords = set()  # 使用set去重
        
        # 中文分词：使用2-4字滑动窗口
        chinese_chars = re.sub(r'[a-zA-Z0-9]', '', text_clean)
        
        # 处理中文：2-4字词滑动窗口
        for length in [2, 3, 4]:
            for i in range(len(chinese_chars) - length + 1):
                word = chinese_chars[i:i+length]
                if word:
                    keywords.add(word)
        
        # 处理英文/数字：按空格分割
        english_text = re.sub(r'[\u4e00-\u9fa5]', ' ', text.lower())
        english_words = english_text.split()
        for w in english_words:
            if len(w) > 1:
                keywords.add(w)
        
        # 过滤停用词和单字
        result = [
            kw for kw in keywords 
            if len(kw) >= 2 and kw not in STOP_WORDS
        ]
        
        return result

    def _extract_keywords(self, text: str) -> List[str]:
        """
        提取关键词
        
        使用N-gram分词，返回所有有效关键词。
        
        Args:
            text: 输入文本
            
        Returns:
            关键词列表
        """
        return self._tokenize(text)

    def add(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """
        添加记忆（优化版）

        Args:
            content: 记忆内容
            metadata: 元数据

        Returns:
            memory_id: 记忆唯一标识
        """
        memory_id = f"mem_{uuid.uuid4().hex[:8]}"
        timestamp = metadata.get('timestamp') if metadata else None

        memory = {
            "id": memory_id,
            "content": content,
            "metadata": metadata or {},
            "keywords": self._extract_keywords(content),
            "timestamp": timestamp
        }

        self._memories.append(memory)

        # 更新索引（使用set去重）
        unique_keywords = set(memory["keywords"])
        for keyword in unique_keywords:
            if keyword not in self._index:
                self._index[keyword] = set()
                self._doc_freq[keyword] = 0
            self._index[keyword].add(memory_id)
            self._doc_freq[keyword] += 1

        self._total_docs += 1

        # 内存限制
        if len(self._memories) > self.max_memories:
            self._evict_oldest()

        # 持久化
        if self.enable_persistence and self.storage_path:
            self._save()
        
        # 清除缓存（添加新记忆后需要重新查询）
        self._query_cache.clear()
        
        return memory_id

    def _evict_oldest(self) -> None:
        """
        淘汰最旧的记忆
        """
        if not self._memories:
            return
        
        oldest = self._memories.pop(0)
        
        # 更新索引
        for keyword in set(oldest.get("keywords", [])):
            if keyword in self._index:
                self._index[keyword].discard(oldest["id"])
                self._doc_freq[keyword] = max(0, self._doc_freq.get(keyword, 1) - 1)
                if not self._index[keyword]:
                    del self._index[keyword]
                    self._doc_freq.pop(keyword, None)

    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        查询记忆（TF-IDF优化版 + 缓存）

        Args:
            query: 查询内容
            top_k: 返回数量

        Returns:
            results: 检索结果列表
        """
        # 检查缓存
        cache_key = (query, top_k)
        if cache_key in self._query_cache:
            self._cache_hits += 1
            # 移动到末尾（LRU）
            self._query_cache.move_to_end(cache_key)
            return self._query_cache[cache_key].copy()
        
        self._cache_misses += 1
        query_keywords = self._extract_keywords(query)
        
        if not query_keywords or not self._memories:
            return []

        scores: Dict[str, float] = {}
        
        if self.enable_tfidf:
            # TF-IDF评分
            for keyword in query_keywords:
                if keyword in self._index:
                    # IDF = log(N / df)
                    df = len(self._index[keyword])
                    idf = math.log((self._total_docs + 1) / (df + 1)) + 1
                    
                    for memory_id in self._index[keyword]:
                        scores[memory_id] = scores.get(memory_id, 0) + idf
        else:
            # 简单计数
            for keyword in query_keywords:
                if keyword in self._index:
                    for memory_id in self._index[keyword]:
                        scores[memory_id] = scores.get(memory_id, 0) + 1

        # 归一化
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: round(v / max_score, 4) for k, v in scores.items()}

        # 排序
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 返回结果
        results = []
        memory_map = {m["id"]: m for m in self._memories}
        
        for memory_id, score in sorted_ids[:top_k]:
            if memory_id in memory_map:
                memory = memory_map[memory_id]
                results.append({
                    "memory_id": memory_id,
                    "content": memory["content"],
                    "score": score,
                    "metadata": memory["metadata"]
                })

        # 保存到缓存（LRU）
        self._query_cache[cache_key] = results
        if len(self._query_cache) > self._cache_size:
            self._query_cache.popitem(last=False)  # 删除最旧的
        
        return results

    def predict(self, situation: str, action: str) -> Dict[str, Any]:
        """
        预测（优化版）

        Args:
            situation: 当前情境
            action: 拟采取行动

        Returns:
            prediction: 预测结果
        """
        situation_keywords = self._extract_keywords(situation)
        action_keywords = self._extract_keywords(action)

        # 检索相关记忆
        related = set()
        for keyword in situation_keywords:
            if keyword in self._index:
                related.update(self._index[keyword])

        related_count = len(related)
        
        # 检查是否有相似行动的历史
        similar_actions = 0
        for keyword in action_keywords:
            if keyword in self._index:
                similar_actions += len(self._index[keyword])

        # 计算置信度
        if related_count == 0:
            confidence = 0.1
        else:
            confidence = min(related_count * 0.05 + similar_actions * 0.02, 0.95)

        return {
            "outcome": "基于TF-IDF相似度预测",
            "confidence": round(confidence, 4),
            "related_memories": related_count,
            "similar_actions": similar_actions,
            "mode": "lite_tfidf"
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计字典
        """
        cache_total = self._cache_hits + self._cache_misses
        cache_hit_rate = (
            self._cache_hits / cache_total * 100 
            if cache_total > 0 else 0
        )
        
        return {
            "total_memories": len(self._memories),
            "max_memories": self.max_memories,
            "index_size": len(self._index),
            "total_docs": self._total_docs,
            "tfidf_enabled": self.enable_tfidf,
            "persistence_enabled": self.enable_persistence,
            "storage_path": self.storage_path,
            "cache_size": len(self._query_cache),
            "cache_max_size": self._cache_size,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": round(cache_hit_rate, 2)
        }

    def _get_storage_file(self) -> Optional[str]:
        """
        获取存储文件路径
        """
        if self.storage_path:
            return os.path.join(self.storage_path, "su_memory_lite.json")
        return None

    def _save(self) -> bool:
        """
        保存记忆到磁盘
        
        Returns:
            是否保存成功
        """
        storage_file = self._get_storage_file()
        if not storage_file:
            return False
        
        try:
            os.makedirs(os.path.dirname(storage_file), exist_ok=True)
            data = {
                "memories": self._memories,
                "index": {k: list(v) for k, v in self._index.items()},
                "doc_freq": self._doc_freq,
                "total_docs": self._total_docs
            }
            with open(storage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Save failed: {e}")
            return False

    def _load(self) -> bool:
        """
        从磁盘加载记忆
        
        Returns:
            是否加载成功
        """
        storage_file = self._get_storage_file()
        if not storage_file or not os.path.exists(storage_file):
            return False
        
        try:
            with open(storage_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._memories = data.get("memories", [])
            self._index = {k: set(v) for k, v in data.get("index", {}).items()}
            self._doc_freq = data.get("doc_freq", {})
            self._total_docs = data.get("total_docs", len(self._memories))
            return True
        except Exception as e:
            print(f"Load failed: {e}")
            return False

    def clear(self) -> None:
        """
        清空所有记忆
        """
        self._memories.clear()
        self._index.clear()
        self._doc_freq.clear()
        self._total_docs = 0
        
        # 删除存储文件
        storage_file = self._get_storage_file()
        if storage_file and os.path.exists(storage_file):
            os.remove(storage_file)

    def __len__(self) -> int:
        """
        记忆数量
        """
        return len(self._memories)

    def __bool__(self) -> bool:
        """
        始终返回True，确保client对象在布尔上下文中为真
        """
        return True

    def __enter__(self):
        """
        上下文管理器入口
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        上下文管理器出口
        """
        if self.enable_persistence and self.storage_path:
            self._save()
