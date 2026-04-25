"""
SuMemory Client — SDK 一行API
"""

from typing import Optional, List, Dict, Any, TYPE_CHECKING
import math
import json
import os
import shutil

if TYPE_CHECKING:
    from su_memory.encoding import MemoryEncoding


class MemoryResult:
    """记忆检索结果"""
    __slots__ = ("memory_id", "content", "score", "encoding", "metadata")

    def __init__(
        self,
        memory_id: str,
        content: str,
        score: float,
        encoding: "MemoryEncoding",
        metadata: Dict[str, Any],
    ):
        self.memory_id = memory_id
        self.content = content
        self.score = score
        self.encoding = encoding
        self.metadata = metadata


class SuMemory:
    """
    Semantic Memory Engine SDK Client

    一行代码初始化，本地运行，无需服务器。

    Example:
        >>> client = SuMemory()
        >>> mid = client.add("项目ROI增长了25%", metadata={"source": "finance"})
        >>> results = client.query("投资回报")
        >>> print(results[0].encoding.category)  # creative
    """

    def __init__(
        self,
        mode: str = "local",
        storage: str = "sqlite",
        persist_dir: str = "./su_memory_data",
        embedder = None,
    ):
        self.mode = mode
        self.storage = storage
        self.persist_dir = persist_dir
        self._embedder = embedder

        self._init_engine()

    def _init_engine(self):
        """初始化引擎（含 Phase 1&2 增强模块）"""
        from su_memory._sys.encoders import SemanticEncoder, EncoderCore
        from su_memory._sys.causal import CausalChain, CausalInference
        from su_memory._sys.codec import SuCompressor
        from su_memory._sys.chrono import TemporalSystem
        from su_memory._sys.intent_classifier import IntentClassifier, ProgressiveDisclosure
        from su_memory._sys.session_bridge import SessionBridge
        from su_memory._sys.recency_feedback import RecencyFeedbackSystem
        from su_memory._sys.wiki_linker import WikiLinker
        from su_memory._sys.multi_hop import MultiHopRetriever
        from su_memory._sys.recall_trigger import RecallTrigger
        # 原有核心
        self._causal = CausalChain()
        self._codec = SuCompressor()
        self._embedder = getattr(self, '_embedder', None)  # 来自 __init__ 注入

        self._encoder = SemanticEncoder()
        self._encoder_core = EncoderCore()
        self._causal_inference = CausalInference()
        self._temporal = TemporalSystem()
        # Phase 1&2 增强模块
        self._intent_classifier = IntentClassifier()
        self._session_bridge = SessionBridge(
            persist_path=os.path.join(self.persist_dir, "sessions.jsonl"))
        self._recency_feedback = RecencyFeedbackSystem(self._temporal)
        self._wiki_linker = WikiLinker()
        self._multi_hop = MultiHopRetriever(
            self._encoder_core, self._causal_inference, self._encoder)
        self._recall_trigger = RecallTrigger(
            intent_classifier=self._intent_classifier,
            session_bridge=self._session_bridge,
            wiki_linker=self._wiki_linker,
            semantic_encoder=self._encoder,
            encoder_core=self._encoder_core,
            memory_store=self)
        self._disclosure = ProgressiveDisclosure()
        # 存储结构
        self._memories: List[Dict] = []
        self._next_id = 1
        self._semantic_index: Dict[str, List[int]] = {}
        self._energy_index: Dict[str, List[int]] = {}
        self._vectors: List[Optional[List[float]]] = []
        # _load() 由外层按需调用

    def add(self, content: str, metadata: Optional[Dict] = None) -> str:
        """
        添加一条记忆

        Args:
            content: 记忆内容
            metadata: 可选元数据

        Returns:
            memory_id: 记忆唯一ID
        """
        enc = self._codec.compress(content)
        category = enc.get("category", "receptive")
        energy_type = enc.get("energy_type", "earth")
        energy = enc.get("energy", 1.0)

        memory_id = f"mem_{self._next_id}"
        self._next_id += 1

        memory = {
            "id": memory_id,
            "content": content,
            "category": category,
            "energy_type": energy_type,
            "energy": energy,
            "metadata": metadata or {},
        }

        self._memories.append(memory)
        self._causal.add(memory_id, category=category, energy_type=energy_type)

        return memory_id

    def query(self, text: str, top_k: int = 5) -> List[MemoryResult]:
        """
        语义检索记忆

        Args:
            text: 查询文本
            top_k: 返回数量

        Returns:
            按相关度排序的记忆列表
        """
        from su_memory.encoding import MemoryEncoding

        enc = self._codec.compress(text)
        query_category = enc.get("category", "receptive")
        query_energy_type = enc.get("energy_type", "earth")

        # 尝试使用向量检索
        vector_scores = {}
        if self._embedder:
            try:
                query_vec = self._embedder.encode(text)
                if query_vec:
                    for i, m in enumerate(self._memories):
                        if i < len(self._vectors) and self._vectors[i]:
                            vec = self._vectors[i]
                            # 计算余弦相似度
                            dot = sum(a * b for a, b in zip(query_vec, vec))
                            norm_q = sum(a * a for a in query_vec) ** 0.5
                            norm_m = sum(a * a for a in vec) ** 0.5
                            if norm_q > 0 and norm_m > 0:
                                vector_scores[m["id"]] = dot / (norm_q * norm_m)
            except Exception:
                pass

        results: List[MemoryResult] = []
        for m in self._memories:
            score = 0.0

            # 向量相似度（权重最高）
            if m["id"] in vector_scores:
                score += vector_scores[m["id"]] * 0.8

            # 类别匹配
            if m["category"] == query_category:
                score += 0.1

            # 能量类型匹配
            if m["energy_type"] == query_energy_type:
                score += 0.05

            # 内容包含关键词
            if any(w in m["content"] for w in text if len(w) > 1):
                score += 0.05

            if score > 0:
                results.append(MemoryResult(
                    memory_id=m["id"],
                    content=m["content"],
                    score=score,
                    encoding=MemoryEncoding(
                        category=m["category"],
                        energy=m["energy_type"],
                        pattern=0,
                        intensity=1.0,
                        time_stem="",
                        time_branch="",
                        causal_depth=0,
                    ),
                    metadata=m["metadata"],
                ))

        results.sort(key=lambda x: -x.score)
        return results[:top_k]

    def link(self, parent_id: str, child_id: str) -> bool:
        """建立两条记忆的因果关联"""
        return self._causal.link(parent_id, child_id)

    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计"""
        category_count: Dict[str, int] = {}
        energy_count: Dict[str, int] = {}
        for m in self._memories:
            category_count[m["category"]] = category_count.get(m["category"], 0) + 1
            energy_count[m["energy_type"]] = energy_count.get(m["energy_type"], 0) + 1

        return {
            "total_memories": len(self._memories),
            "category_distribution": category_count,
            "energy_distribution": energy_count,
        }

    # ── Phase 1&2: 新模块辅助方法 ─────────────────────────────────

    def get_all_memories(self) -> List[Dict]:
        """返回所有记忆（供 RecallTrigger 等使用）"""
        return self._memories

    def get_memory(self, memory_id: str) -> Optional[Dict]:
        """按 ID 获取单条记忆"""
        for m in self._memories:
            if m.get("id") == memory_id:
                return m
        return None

    def _classify_intent(self, query: str):
        """意图分类"""
        return self._intent_classifier.classify(query)

    def _build_candidates(self) -> List[Dict]:
        """构建检索候选集"""
        return [{
            "memory_id": m.get("id"),
            "content": m.get("content"),
            "memory_type": m.get("metadata", {}).get("type", "fact"),
            "hexagram_index": m.get("hexagram_index", 0),
            "hexagram_name": m.get("hexagram_name", ""),
            "energy_type": m.get("energy_type", ""),
            "category_probs": m.get("category_probs"),
            "energy_scores": m.get("energy_scores"),
            "vector": self._vectors[i] if i < len(self._vectors) else None,
            "timestamp": m.get("timestamp", 0),
        } for i, m in enumerate(self._memories)]

    def record_feedback(self, memory_id: str, was_useful: bool) -> None:
        """记录用户反馈"""
        self._recency_feedback.record_feedback(memory_id, was_useful)
        self._session_bridge.record_memory_access(
            memory_id, relevance_score=1.0 if was_useful else 0.0)

    def on_query(self, query: str) -> None:
        """每次查询前调用"""
        intent = self._intent_classifier.classify(query)
        self._session_bridge.record_query(query, intent.name)
        self._disclosure.record_query(query, intent.name)

    def get_next_disclosure_results(self, positive: bool) -> List["MemoryResult"]:
        """正反馈后获取更深阶段结果"""
        self._disclosure.get_next_stage(feedback="positive" if positive else "negative")
        resp = self._recall_trigger.last_response
        if not resp:
            return []
        return resp.results[:self._disclosure.current_stage.max_items]

    def __len__(self) -> int:
        return len(self._memories)
