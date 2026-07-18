"""
_session — 会话管理器（lite_pro.py 拆分）

SessionManager: 对话上下文与会话记忆管理。无外部类依赖。
从 lite_pro.py 拆分，对外通过 lite_pro.py 再导出保持兼容。
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Any

from su_memory.sdk._common import logger


class SessionManager:
    """
    多会话管理系统
    支持会话隔离、跨会话召回和话题联想
    """

    def __init__(self, storage_path: str = None, embedding_manager = None):
        self.storage_path = storage_path
        self._sessions: dict[str, dict] = {}
        self._session_index: dict[str, list[str]] = defaultdict(list)  # topic -> memory_ids
        self._current_session: str | None = None
        self._memory_contents: dict[str, str] = {}  # memory_id -> content for cross-session recall
        self._embedding_manager = embedding_manager  # for semantic topic recall
        self._load()

    def set_embedding_manager(self, embedding_manager):
        """设置embedding管理器用于语义召回"""
        self._embedding_manager = embedding_manager

    def create_session(self, session_id: str = None, metadata: dict = None) -> str:
        """创建新会话"""
        sid = session_id or f"session_{int(time.time())}"

        self._sessions[sid] = {
            "id": sid,
            "created_at": int(time.time()),
            "memory_ids": [],
            "topics": set(),
            "metadata": metadata or {}
        }

        self._current_session = sid
        self._save()
        return sid

    def add_memory(self, session_id: str, memory_id: str, topic: str = None, content: str = None):
        """
        添加记忆到会话

        Args:
            session_id: 会话ID
            memory_id: 记忆ID
            topic: 话题标签
            content: 记忆内容（用于跨会话召回）
        """
        if session_id not in self._sessions:
            return

        session = self._sessions[session_id]
        session["memory_ids"].append(memory_id)

        # 存储记忆内容用于语义召回
        if content:
            self._memory_contents[memory_id] = content

        if topic and topic not in session["topics"]:
            session["topics"].add(topic)
            self._session_index[topic].append(memory_id)

        self._save()

    def get_session_memories(self, session_id: str) -> list[str]:
        """获取会话的所有记忆ID"""
        if session_id not in self._sessions:
            return []
        return self._sessions[session_id]["memory_ids"]

    def get_current_session(self) -> str | None:
        return self._current_session

    def set_current_session(self, session_id: str):
        self._current_session = session_id

    def get_topic_memories(self, topic: str) -> list[str]:
        """获取特定话题的所有记忆"""
        return self._session_index.get(topic, [])

    def get_all_topics(self) -> list[str]:
        """获取所有话题"""
        return list(self._session_index.keys())

    def get_cross_session_topics(self) -> list[str]:
        """获取跨会话话题（出现多次的话题）"""
        topic_count = defaultdict(int)
        for topic, memory_ids in self._session_index.items():
            topic_count[topic] = len(memory_ids)

        # 返回出现2次以上的话题
        return [t for t, count in topic_count.items() if count >= 2]

    def get_related_topics(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """
        获取与查询相关的话题

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            List of (topic, similarity_score)
        """
        if not self._embedding_manager:
            # Fallback to keyword matching
            all_topics = self.get_all_topics()
            results = []
            for topic in all_topics:
                if query in topic or topic in query:
                    results.append((topic, 1.0))
            return results[:top_k]

        # Semantic similarity using embedding
        try:
            query_vec = self._embedding_manager.encode(query)

            all_topics = self.get_all_topics()
            results = []

            for topic in all_topics:
                topic_vec = self._embedding_manager.encode(topic)
                sim = self._cosine_similarity(query_vec, topic_vec)
                results.append((topic, sim))

            # Sort by similarity
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

        except Exception:
            # Fallback
            return [(t, 1.0) for t in self.get_all_topics()[:top_k]]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """计算余弦相似度"""
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def get_cross_session_recall(self, topic: str, exclude_session: str = None, top_k: int = 10) -> list[str]:
        """
        跨会话召回

        Args:
            topic: 话题关键词
            exclude_session: 排除的会话ID
            top_k: 返回数量

        Returns:
            跨会话记忆ID列表
        """
        results = []

        # 1. 精确话题匹配
        topic_mems = self.get_topic_memories(topic)
        results.extend(topic_mems)

        # 2. 语义相似话题
        related = self.get_related_topics(topic, top_k=10)
        for related_topic, score in related:
            if score > 0.5:  # 相似度阈值
                related_mems = self.get_topic_memories(related_topic)
                results.extend(related_mems)

        # 3. 去重并过滤
        seen = set()
        final_results = []
        for mem_id in results:
            if mem_id in seen:
                continue
            seen.add(mem_id)

            # 检查是否属于被排除的会话
            if exclude_session:
                excluded = False
                for sid, session in self._sessions.items():
                    if sid != exclude_session and mem_id in session.get("memory_ids", []):
                        excluded = True
                        break
                if excluded:
                    continue

            final_results.append(mem_id)

        return final_results[:top_k]

    def get_session_summary(self, session_id: str) -> dict[str, Any]:
        """获取会话摘要"""
        if session_id not in self._sessions:
            return {}

        session = self._sessions[session_id]

        return {
            "id": session_id,
            "created_at": session["created_at"],
            "memory_count": len(session["memory_ids"]),
            "topics": list(session["topics"]),
            "metadata": session["metadata"]
        }

    def _save(self):
        if not self.storage_path:
            return

        os.makedirs(self.storage_path, exist_ok=True)
        path = os.path.join(self.storage_path, "sessions.json")

        # 转换set为list以便JSON序列化
        data = {
            k: {**v, "topics": list(v["topics"])}
            for k, v in self._sessions.items()
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump({"sessions": data, "index": dict(self._session_index)}, f)

    def _load(self):
        if not self.storage_path:
            return

        path = os.path.join(self.storage_path, "sessions.json")
        if not os.path.exists(path):
            return

        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)

            self._sessions = {
                k: {**v, "topics": set(v["topics"])}
                for k, v in data.get("sessions", {}).items()
            }
            self._session_index = defaultdict(list, data.get("index", {}))
        except Exception as e:
            logger.debug("降级处理: %s", e)


