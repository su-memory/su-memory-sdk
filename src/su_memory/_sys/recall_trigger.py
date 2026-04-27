"""
统一召回触发器

为 su-memory 提供完整的记忆召回流程：
1. 意图识别（调用 IntentClassifier）
2. 会话上下文管理（调用 SessionBridge）
3. 多源召回（Dreams + Wiki + 内存）
4. 渐进披露控制（调用 ProgressiveDisclosure）
5. 召回结果记录与反馈

核心流程（兼容 Hindsight recall-trigger.js）：
    should_recall(query) → recall(query) → update_disclosure() → log()
"""

import time
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, TYPE_CHECKING
from pathlib import Path

# 可选导入（避免循环依赖）
if TYPE_CHECKING:
    from su_memory._sys.intent_classifier import IntentClassifier, IntentConfig
    from su_memory._sys.session_bridge import SessionBridge
    from su_memory._sys.wiki_linker import WikiLinker
    from su_memory._sys.encoders import SemanticEncoder, EncoderCore

logger = logging.getLogger(__name__)


# ========================
# 召回结果结构
# ========================

@dataclass
class RecallResult:
    """单条召回结果"""
    source: str  # "dreams" / "obsidian" / "memex" / "memory" / "holographic"
    content: str
    memory_id: str = ""
    score: float = 0.0
    hexagram_index: int = 0
    hexagram_name: str = ""
    wuxing: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "content": self.content,
            "memory_id": self.memory_id,
            "score": self.score,
            "hexagram_index": self.hexagram_index,
            "hexagram_name": self.hexagram_name,
            "wuxing": self.wuxing,
            "tags": self.tags,
            "metadata": self.metadata,
        }


@dataclass
class RecallResponse:
    """完整召回响应"""
    query: str
    intent_name: str
    intent_level: int
    mode: str  # "none" / "simple" / "standard" / "deep"
    results: List[RecallResult]
    stage_name: str = ""
    processing_time_ms: float = 0.0
    sources_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "intent_name": self.intent_name,
            "intent_level": self.intent_level,
            "mode": self.mode,
            "results": [r.to_dict() for r in self.results],
            "stage_name": self.stage_name,
            "processing_time_ms": self.processing_time_ms,
            "sources_used": self.sources_used,
        }


# ========================
# RecallTrigger 主类
# ========================

class RecallTrigger:
    """
    统一召回触发器

    整合所有召回模块，提供单一入口。

    使用方法：
        trigger = RecallTrigger()
        response = trigger.recall("Nutri-Brain项目当前进度如何")
        for result in response.results:
            print(result.content, result.score)
    """

    def __init__(
        self,
        intent_classifier: Optional["IntentClassifier"] = None,
        session_bridge: Optional["SessionBridge"] = None,
        wiki_linker: Optional["WikiLinker"] = None,
        semantic_encoder: Optional["SemanticEncoder"] = None,
        encoder_core: Optional["EncoderCore"] = None,
        disclosure: Optional[Any] = None,  # ProgressiveDisclosure
        memory_store: Optional[Any] = None,  # SuMemory store
        recall_log_path: Optional[str] = None,
    ):
        self._intent_classifier = intent_classifier
        self._session_bridge = session_bridge
        self._wiki_linker = wiki_linker
        self._semantic_encoder = semantic_encoder
        self._encoder_core = encoder_core
        self._disclosure = disclosure
        self._memory_store = memory_store
        self._recall_log_path = recall_log_path

        # 内部状态
        self._last_response: Optional[RecallResponse] = None
        self._recall_count = 0

    def should_recall(self, query: str) -> bool:
        """
        判断某查询是否应该触发召回

        L0 意图（casual闲聊类）返回 False
        """
        if not self._intent_classifier:
            return True  # 没有分类器时默认召回
        return self._intent_classifier.should_recall(query)

    def recall(
        self,
        query: str,
        top_k: int = 5,
        force_level: Optional[int] = None,
    ) -> RecallResponse:
        """
        执行召回

        流程：
        1. 意图识别
        2. 按 level 路由到不同召回源
        3. 全息检索 + Wiki 查询
        4. 合并去重
        5. 记录召回日志
        """
        t0 = time.time()

        # 1. 意图识别
        if self._intent_classifier:
            intent = self._intent_classifier.classify(query)
            if force_level is not None and intent.level < force_level:
                intent = self._intent_classifier.classify_with_level(query, force_level)
        else:
            intent = None

        intent_name = intent.name if intent else "unknown"
        intent_level = intent.level if intent else 2
        mode = (
            self._intent_classifier.get_recall_mode(query)
            if self._intent_classifier
            else "standard"
        )

        # 2. 会话上下文 boost
        context_boost = 1.0
        if self._session_bridge:
            context_boost = self._session_bridge.calculate_context_boost(None, None)

        # 3. 确定 top_k（按披露阶段调整）
        actual_k = top_k
        if self._disclosure:
            stage = self._disclosure.get_current_stage()
            actual_k = min(top_k, stage.max_items)

        # 4. 执行多源召回
        all_results: List[RecallResult] = []
        sources_used: List[str] = []

        if mode in ("simple", "standard", "deep"):
            # L1+ : 先查内部记忆（全息检索）
            if self._semantic_encoder and self._encoder_core:
                memory_results = self._recall_from_memory(query, actual_k, context_boost)
                all_results.extend(memory_results)
                if memory_results:
                    sources_used.append("holographic")

        if mode in ("standard", "deep"):
            # L2+ : 查 Wiki
            if self._wiki_linker and intent and intent.wikis:
                wiki_results = self._recall_from_wiki(query, intent, actual_k)
                all_results.extend(wiki_results)
                if wiki_results:
                    sources_used.extend([r.wiki for r in wiki_results])

        if mode == "deep" and self._session_bridge:
            # L3 : 跨会话召回
            session_results = self._recall_from_sessions(query, actual_k)
            all_results.extend(session_results)
            if session_results:
                sources_used.append("sessions")

        # 5. 合并去重（按 content hash）
        seen: Dict[str, RecallResult] = {}
        for r in all_results:
            key = r.source + ":" + r.content[:100]
            if key not in seen or r.score > seen[key].score:
                seen[key] = r

        merged = sorted(seen.values(), key=lambda x: x.score, reverse=True)[:actual_k]

        # 6. 更新披露阶段
        if self._disclosure:
            self._disclosure.record_results_count(len(merged))
            if intent_name:
                self._disclosure.record_query(query, intent_name)

        # 7. 记录召回日志
        self._log_recall(query, intent_name, len(merged))

        self._recall_count += 1
        self._last_response = RecallResponse(
            query=query,
            intent_name=intent_name,
            intent_level=intent_level,
            mode=mode,
            results=merged,
            stage_name=(
                self._disclosure.current_stage.name
                if self._disclosure
                else "default"
            ),
            processing_time_ms=(time.time() - t0) * 1000,
            sources_used=sources_used,
        )

        return self._last_response

    def _recall_from_memory(
        self,
        query: str,
        top_k: int,
        context_boost: float,
    ) -> List[RecallResult]:
        """从内部记忆（全息检索）召回"""
        results: List[RecallResult] = []
        try:
            info, vec = self._semantic_encoder.encode_with_vector(query, "fact")
        except Exception:
            return results

        # 从 encoder_core 检索（需要有候选记忆）
        # 这里需要 memory_store 提供候选，暂时用简单编码
        if not self._encoder_core:
            # Fallback: 直接返回当前查询的编码结果
            return results

        # 获取候选（如果有 store）
        if self._memory_store:
            try:
                candidates = self._memory_store.get_all_memories()
                if not candidates:
                    return results

                cand_info_map = {}
                for mem in candidates:
                    mem_info, _ = self._semantic_encoder.encode_with_vector(
                        mem.content, mem.memory_type
                    )
                    cand_info_map[mem_info.index] = mem_info

                cand_indices = list(set(mem.hexagram_index for mem in candidates))
                encoder_results = self._encoder_core.retrieve_holographic(
                    info.index, cand_indices, top_k=top_k,
                    query_info=info, candidate_infos=cand_info_map,
                    use_vector_sim=True
                )

                mem_map = {mem.hexagram_index: mem for mem in candidates}
                for idx, score in encoder_results:
                    if idx in mem_map:
                        mem = mem_map[idx]
                        results.append(RecallResult(
                            source="holographic",
                            content=mem.content,
                            memory_id=mem.memory_id,
                            score=score * context_boost,
                            hexagram_index=idx,
                            hexagram_name=info.name,
                            wuxing=info.wuxing,
                            tags=mem.tags or [],
                        ))
            except Exception as e:
                logger.warning(f"Memory recall failed: {e}")

        return results

    def _recall_from_wiki(
        self,
        query: str,
        intent: "IntentConfig",
        top_k: int,
    ) -> List[RecallResult]:
        """从 Wiki（Obsidian/Memex）召回"""
        results: List[RecallResult] = []
        try:
            wiki_results = self._wiki_linker.query_wiki(
                query,
                wikis=intent.wikis,
                tags=intent.tags,
                max_results=top_k,
            )
            for r in wiki_results:
                results.append(RecallResult(
                    source=r.wiki,
                    content=r.excerpt or r.name,
                    memory_id="",  # Wiki 条目没有 memory_id
                    score=r.score,
                    hexagram_index=0,
                    hexagram_name="",
                    wuxing="",
                    tags=r.tags,
                    metadata={"wiki_path": r.path, "wiki_name": r.name},
                ))

            # 同步召回结果回写 Wiki
            if wiki_results:
                self._wiki_linker.batch_sync_recall(wiki_results)

        except Exception as e:
            logger.warning(f"Wiki recall failed: {e}")
        return results

    def _recall_from_sessions(
        self,
        query: str,
        top_k: int,
    ) -> List[RecallResult]:
        """跨会话召回（使用 session_bridge）"""
        results: List[RecallResult] = []
        if not self._session_bridge:
            return results

        try:
            recent_sessions = self._session_bridge.get_recent_sessions(limit=5)
            for ctx in recent_sessions:
                # 查询每个会话中与 query 相关的内容
                for mem_id in ctx.get("memory_ids", []):
                    boost = self._session_bridge.calculate_context_boost(mem_id, ctx["session_id"])
                    if boost > 1.0:
                        # 从 memory_store 补全内容
                        if self._memory_store:
                            mem = self._memory_store.get_memory(mem_id)
                            if mem:
                                results.append(RecallResult(
                                    source="sessions",
                                    content=mem.content,
                                    memory_id=mem_id,
                                    score=boost - 1.0,  # 相对增益
                                    hexagram_index=0,
                                    hexagram_name="",
                                    wuxing="",
                                    tags=[],
                                    metadata={"session_id": ctx["session_id"]},
                                ))
        except Exception as e:
            logger.warning(f"Session recall failed: {e}")
        return results

    def on_feedback(self, feedback: str) -> None:
        """
        处理用户反馈，驱动渐进披露

        Args:
            feedback: "positive" | "negative"
        """
        if self._disclosure:
            self._disclosure.get_next_stage(feedback=feedback)

    def get_next_results(self, top_k: int = 5) -> List[RecallResult]:
        """
        在用户正反馈后，获取下一批（更深阶段的）结果

        调用前应先调用 on_feedback("positive")
        """
        if not self._last_response:
            return []

        if self._disclosure:
            self._disclosure.get_next_stage(feedback="positive")
            stage = self._disclosure.current_stage
            return self._last_response.results[:stage.max_items]

        return self._last_response.results[:top_k]

    def _log_recall(self, query: str, intent_name: str, result_count: int) -> None:
        """记录召回日志到 disclosure-log.jsonl"""
        if not self._recall_log_path:
            return
        try:
            log_path = Path(self._recall_log_path).expanduser()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "type": "disclosure.recorded",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.{}Z".format(
                    int(time.time() * 1000) % 1000)),
                "query": query[:200],
                "intent": intent_name,
                "level": self._last_response.intent_level if self._last_response else 0,
                "resultCount": result_count,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log recall: {e}")

    @property
    def last_response(self) -> Optional[RecallResponse]:
        return self._last_response

    @property
    def recall_count(self) -> int:
        return self._recall_count

    # ========================
    # 工厂方法（快捷创建）
    # ========================

    @classmethod
    def create_default(cls) -> "RecallTrigger":
        """
        使用默认配置创建 RecallTrigger

        所有组件使用默认实现（懒加载）
        """
        from su_memory._sys.intent_classifier import IntentClassifier
        from su_memory._sys.wiki_linker import WikiLinker
        from su_memory._sys.session_bridge import SessionBridge
        from su_memory._sys.intent_classifier import ProgressiveDisclosure

        recall_log = os.path.expanduser("~/.openclaw/workspace/memory/.hindsight/disclosure-log.jsonl")

        return cls(
            intent_classifier=IntentClassifier(),
            wiki_linker=WikiLinker(),
            session_bridge=SessionBridge(),
            disclosure=ProgressiveDisclosure(),
            recall_log_path=recall_log,
        )


import os
