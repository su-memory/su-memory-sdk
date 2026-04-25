"""
多会话上下文桥接器

为 su-memory 增加跨会话记忆关联能力：
1. SessionContext — 记录每个会话的 topic/memory_ids/queries
2. SessionBridge — 管理会话上下文，计算 context boost

与 Hindsight short-term-recall.json 机制兼容
"""

import time
import json
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set
from pathlib import Path


# ========================
# 配置
# ========================

SESSION_TTL = 24 * 3600  # 24小时内的会话视为"近期"
TOPIC_HISTORY_LIMIT = 100  # 每个会话最多保留的 topic 历史


# ========================
# 数据结构
# ========================

@dataclass
class SessionContext:
    """会话上下文"""
    session_id: str
    start_time: int  # Unix timestamp
    end_time: Optional[int] = None
    topics: List[str] = field(default_factory=list)
    memory_ids: List[str] = field(default_factory=list)  # 访问过的记忆 ID
    queries: List[str] = field(default_factory=list)
    intent_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def duration(self) -> int:
        """会话持续时间（秒）"""
        end = self.end_time or int(time.time())
        return end - self.start_time

    def add_topic(self, topic: str) -> None:
        if topic not in self.topics:
            self.topics.append(topic)
            if len(self.topics) > TOPIC_HISTORY_LIMIT:
                self.topics = self.topics[-TOPIC_HISTORY_LIMIT:]

    def add_memory_access(self, memory_id: str) -> None:
        if memory_id not in self.memory_ids:
            self.memory_ids.append(memory_id)

    def add_query(self, query: str) -> None:
        self.queries.append(query[:500])  # 截断长查询
        if len(self.queries) > TOPIC_HISTORY_LIMIT:
            self.queries = self.queries[-TOPIC_HISTORY_LIMIT:]

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "topics": self.topics,
            "memory_ids": self.memory_ids,
            "queries": self.queries,
            "intent_name": self.intent_name,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "SessionContext":
        return cls(
            session_id=d["session_id"],
            start_time=d["start_time"],
            end_time=d.get("end_time"),
            topics=d.get("topics", []),
            memory_ids=d.get("memory_ids", []),
            queries=d.get("queries", []),
            intent_name=d.get("intent_name"),
            metadata=d.get("metadata", {}),
        )


@dataclass
class MemoryVisit:
    """单次记忆访问记录"""
    memory_id: str
    session_id: str
    timestamp: int
    access_type: str = "recall"  # "recall" | "update" | "create"
    relevance_score: float = 0.0


# ========================
# SessionBridge
# ========================

class SessionBridge:
    """
    多会话上下文桥接器

    使用方法：
        bridge = SessionBridge()
        ctx = bridge.start_session("session-abc-123")

        # 记录用户的查询和记忆访问
        bridge.record_query("Nutri-Brain项目融资情况", intent_name="financing")
        bridge.record_memory_access("mem-001")
        bridge.record_topic("融资规划")

        # 召回时计算 boost
        boost = bridge.calculate_context_boost("mem-001")
        # boost > 1.0 表示该记忆与当前会话相关
    """

    def __init__(
        self,
        persist_path: Optional[str] = None,
        max_sessions: int = 100,
    ):
        """
        Args:
            persist_path: 会话历史持久化路径（用于重启后恢复）
            max_sessions: 内存中最多保留的会话数（超出时淘汰最老的）
        """
        self._persist_path = (
            Path(persist_path).expanduser()
            if persist_path
            else Path("~/.openclaw/workspace/memory/.hindsight/sessions.jsonl")
        )
        self._max_sessions = max_sessions

        self._sessions: Dict[str, SessionContext] = {}
        self._current_session: Optional[SessionContext] = None
        self._session_lock = threading.RLock()

        # 跨会话记忆访问索引：memory_id -> [MemoryVisit]
        self._memory_visits: Dict[str, List[MemoryVisit]] = {}

        # 加载持久化历史
        self._load_sessions()

    def start_session(self, session_id: Optional[str] = None) -> SessionContext:
        """
        开始新会话

        会自动结束当前会话（如果有）
        """
        with self._session_lock:
            # 结束当前会话
            if self._current_session:
                self._current_session.end_time = int(time.time())
                self._sessions[self._current_session.session_id] = self._current_session

            sid = session_id or f"session_{int(time.time())}"
            ctx = SessionContext(session_id=sid, start_time=int(time.time()))
            self._sessions[sid] = ctx
            self._current_session = ctx

            # 淘汰超出会话数量限制的老会话
            self._evict_old_sessions()

            return ctx

    def end_current_session(self) -> Optional[SessionContext]:
        """显式结束当前会话"""
        with self._session_lock:
            if self._current_session:
                self._current_session.end_time = int(time.time())
                self._sessions[self._current_session.session_id] = self._current_session
                self._current_session = None
            return self._current_session

    def get_current_session(self) -> Optional[SessionContext]:
        return self._current_session

    def get_session(self, session_id: str) -> Optional[SessionContext]:
        return self._sessions.get(session_id)

    def record_query(self, query: str, intent_name: Optional[str] = None) -> None:
        """记录当前会话的查询"""
        if not self._current_session:
            return
        self._current_session.add_query(query)
        if intent_name:
            self._current_session.intent_name = intent_name
            self._current_session.add_topic(intent_name)

    def record_memory_access(
        self,
        memory_id: str,
        access_type: str = "recall",
        relevance_score: float = 0.0,
    ) -> None:
        """记录当前会话访问过某记忆"""
        if not self._current_session:
            return
        self._current_session.add_memory_access(memory_id)

        visit = MemoryVisit(
            memory_id=memory_id,
            session_id=self._current_session.session_id,
            timestamp=int(time.time()),
            access_type=access_type,
            relevance_score=relevance_score,
        )
        self._memory_visits.setdefault(memory_id, []).append(visit)
        # 只保留最近 20 次访问
        if len(self._memory_visits[memory_id]) > 20:
            self._memory_visits[memory_id] = self._memory_visits[memory_id][-20:]

    def record_topic(self, topic: str) -> None:
        """记录当前会话的主题"""
        if not self._current_session:
            return
        self._current_session.add_topic(topic)

    def calculate_context_boost(
        self,
        memory_id: Optional[str],
        session_id: Optional[str] = None,
    ) -> float:
        """
        计算某记忆在当前（或指定）会话中的 context boost

        Boost 来源：
        1. 当前会话访问过 → ×1.2
        2. 近期会话（24h内）访问过 → ×1.05~1.1（按距今时间衰减）
        3. 与当前会话主题相关 → ×1.1
        4. 跨会话共现记忆（同一主题多次被访问）→ ×1.05
        """
        boost = 1.0
        now = int(time.time())

        # 确定目标会话
        ctx = self._current_session
        if session_id and session_id in self._sessions:
            ctx = self._sessions[session_id]
        if not ctx:
            return boost

        # Boost 1: 当前会话访问过
        if memory_id and memory_id in ctx.memory_ids:
            boost *= 1.2

        # Boost 2: 近期会话访问过（带时间衰减）
        if memory_id:
            recent_visits = [
                v for v in self._memory_visits.get(memory_id, [])
                if v.session_id != ctx.session_id
                and now - v.timestamp < SESSION_TTL
            ]
            if recent_visits:
                days_ago = min((now - recent_visits[-1].timestamp) / SESSION_TTL, 1.0)
                boost *= 1.0 + 0.1 * (1.0 - days_ago)  # 1.05 ~ 1.10

        # Boost 3: 与当前会话主题相关
        if memory_id and ctx.topics:
            # 简单策略：检查记忆 ID 是否在 topics 中出现
            for topic in ctx.topics:
                if topic in memory_id.lower() or (hasattr(memory_id, 'content') and topic in memory_id.content):
                    boost *= 1.1
                    break

        return min(boost, 1.5)  # 上限 1.5

    def get_recent_sessions(self, limit: int = 5) -> List[Dict]:
        """获取最近的会话（用于跨会话召回）"""
        cutoff = int(time.time()) - SESSION_TTL
        recent = [
            ctx.to_dict()
            for ctx in self._sessions.values()
            if ctx.start_time > cutoff
        ]
        recent.sort(key=lambda x: x["start_time"], reverse=True)
        return recent[:limit]

    def get_sessions_for_memory(self, memory_id: str) -> List[Dict]:
        """获取访问过某记忆的所有会话"""
        return [
            self._sessions[v.session_id].to_dict()
            for v in self._memory_visits.get(memory_id, [])
            if v.session_id in self._sessions
        ]

    def get_topic_evolution(self, session_id: str) -> List[str]:
        """获取某会话的主题演变历史"""
        ctx = self._sessions.get(session_id)
        return list(ctx.topics) if ctx else []

    def _evict_old_sessions(self) -> None:
        """淘汰最老的会话（超出数量限制时）"""
        if len(self._sessions) <= self._max_sessions:
            return
        # 按 start_time 排序，淘汰最老的
        sorted_sessions = sorted(
            self._sessions.items(),
            key=lambda x: x[1].start_time,
        )
        to_evict = len(self._sessions) - self._max_sessions
        for sid, _ in sorted_sessions[:to_evict]:
            del self._sessions[sid]

    def _load_sessions(self) -> None:
        """从持久化文件加载会话历史"""
        if not self._persist_path.exists():
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    ctx = SessionContext.from_dict(d)
                    self._sessions[ctx.session_id] = ctx
        except Exception:
            pass

    def persist_sessions(self) -> None:
        """将会话历史持久化到文件"""
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._persist_path, "w", encoding="utf-8") as f:
                for ctx in self._sessions.values():
                    f.write(json.dumps(ctx.to_dict(), ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ========================
    # 调试 / 管理接口
    # ========================

    @property
    def active_session_count(self) -> int:
        return len(self._sessions)

    def get_stats(self) -> Dict:
        return {
            "total_sessions": len(self._sessions),
            "current_session": (
                self._current_session.session_id
                if self._current_session
                else None
            ),
            "memory_visit_count": sum(len(v) for v in self._memory_visits.values()),
            "persist_path": str(self._persist_path),
        }

    def __len__(self) -> int:
        return len(self._sessions)
