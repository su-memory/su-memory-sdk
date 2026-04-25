"""
MemoryRetriever - 记忆检索器（Phase 2增强版 + Task13多维融合）

Phase 0: 纯向量检索 + 时序重排
Phase 1: 全息六路检索（互卦/综卦/错卦）
Phase 2: su_core多视图融合检索
Task13: 5维连续融合检索（语义/八卦/五行/全息/因果）
"""

import logging
from typing import List, Dict, Any, Optional

from su_core import MultiViewRetriever, SemanticEncoder
from su_core._sys.causal import CausalInference, BAGUA_WUXING
from su_core._sys.yijing import YiJingInference

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """
    记忆检索器（Task13增强版）

    支持5维连续融合检索：
    - 语义向量相似度
    - 八卦概率分布软匹配
    - 五行能量匹配
    - 全息结构匹配
    - 因果链关联度
    """

    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.semantic_encoder = SemanticEncoder()
        self.multi_view_retriever = MultiViewRetriever()
        self.causal_inference = CausalInference()
        self.yijing_inference = YiJingInference()

    async def retrieve(
        self,
        collection: str,
        query_vector: list,
        user_id: str,
        query_text: str = "",
        limit: int = 8,
        time_window: Optional[int] = None,
        use_holographic: bool = True
    ) -> List[Dict[str, Any]]:
        """
        检索记忆

        Args:
            collection: 租户ID
            query_vector: 查询向量
            user_id: 用户ID
            query_text: 查询原文（用于生成查询的 EncodingInfo）
            limit: 返回数量
            time_window: 时间窗口（秒）
            use_holographic: 是否使用全息检索
        """
        # 构建过滤条件
        filter_conditions = {"user_id": user_id}

        # Phase 0: 向量检索（基线）
        vector_results = await self.vector_db.search(
            collection=collection,
            query_vector=query_vector,
            limit=limit * 3,
            filter=filter_conditions
        )

        # 时序过滤
        if time_window:
            import time
            cutoff = time.time() - time_window
            vector_results = [
                r for r in vector_results
                if r["payload"].get("timestamp", 0) > cutoff
            ]

        # 多维融合检索
        if use_holographic:
            results = self._holographic_rerank(
                query_vector=query_vector,
                query_text=query_text,
                candidates=vector_results,
                limit=limit
            )
        else:
            results = self._simple_rerank(vector_results, limit)

        return results

    def _holographic_rerank(
        self,
        query_vector: list,
        query_text: str,
        candidates: List[Dict],
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Task13: 5维连续融合重排

        使用 query_text 直接编码生成查询的 EncodingInfo，
        携带 bagua_probs/wuxing_scores 传入 MultiViewRetriever。
        """
        if not candidates:
            return []

        # 1. 生成查询的 EncodingInfo（从 query_text 直接编码）
        if query_text:
            query_hexagram = self.semantic_encoder.encode(query_text, "fact")
        else:
            # fallback: 从第一个候选的内容编码
            query_hexagram = self.semantic_encoder.encode(
                candidates[0]["payload"].get("content", ""),
                candidates[0]["payload"].get("memory_type", "fact")
            )

        # 2. 提取候选信息
        for i, cand in enumerate(candidates):
            hex_idx = cand["payload"].get("hexagram_index", 0)
            cand["hexagram_index"] = hex_idx
            # Qdrant cosine similarity 直接使用
            cand["vector_score"] = cand.get("score", 0.0)
            # 提取 bagua_probs 和 wuxing_scores（如果 payload 中有）
            cand["bagua_probs"] = cand["payload"].get("bagua_probs")
            cand["wuxing_scores"] = cand["payload"].get("wuxing_scores")

        # 3. 多维融合检索
        fused_results = self.multi_view_retriever.retrieve(
            query_content=query_text or candidates[0]["payload"].get("content", ""),
            query_hexagram=query_hexagram,
            candidates=candidates,
            top_k=limit
        )

        # 3.5 三层推理 bonus（YiJingInference）
        if self.yijing_inference and query_hexagram:
            try:
                cand_indices = [c.get("hexagram_index", 0) for c in fused_results]
                three_layer_scores = self.yijing_inference.three_layer_retrieve(
                    query_index=query_hexagram.index,
                    candidate_indices=cand_indices,
                    top_k=len(fused_results)
                )
                tl_map = {s["index"]: s for s in three_layer_scores}
                for cand in fused_results:
                    hex_idx = cand.get("hexagram_index", 0)
                    tl_info = tl_map.get(hex_idx % 64)
                    if tl_info:
                        cand["yijing_trend"] = tl_info.get("trend", "")
                        cand["yijing_layer_scores"] = tl_info.get("layer_scores", {})
                        # bonus: 三层推理得分 * 0.05 作为额外加成
                        bonus = tl_info.get("score", 0) * 0.05
                        cand["holographic_score"] = cand.get("holographic_score", 0) + bonus
            except Exception:
                pass

        # 按更新后的分数重排
        fused_results.sort(key=lambda x: x.get("holographic_score", 0), reverse=True)

        # 4. 转换格式
        memories = []
        for r in fused_results:
            memories.append({
                "id": r["id"],
                "content": r["payload"]["content"],
                "score": r["score"],
                "timestamp": r["payload"].get("timestamp", 0),
                "memory_type": r["payload"].get("memory_type", "fact"),
                "metadata": r["payload"].get("metadata", {}),
                "holographic_score": r.get("holographic_score", 0),
                "hexagram_index": r.get("hexagram_index", 0),
                "fusion_detail": r.get("fusion_detail", {}),
            })

        return memories

    def _simple_rerank(self, results: List[Dict], limit: int) -> List[Dict[str, Any]]:
        """Phase 0: 简单时序重排"""
        import time
        now = time.time()

        for r in results:
            r["holographic_score"] = 0
            r["hexagram_index"] = r["payload"].get("hexagram_index", 0)

        # 按时间排序（新的优先）
        results.sort(key=lambda x: x["payload"].get("timestamp", 0), reverse=True)

        memories = []
        for r in results[:limit]:
            memories.append({
                "id": r["id"],
                "content": r["payload"]["content"],
                "score": r["score"],
                "timestamp": r["payload"].get("timestamp", 0),
                "memory_type": r["payload"].get("memory_type", "fact"),
                "metadata": r["payload"].get("metadata", {}),
                "holographic_score": 0,
                "hexagram_index": r.get("hexagram_index", 0),
            })

        return memories

    async def retrieve_multi_hop(
        self,
        collection: str,
        query_vector: list,
        user_id: str,
        query_text: str = "",
        limit: int = 8,
        max_hops: int = 2,
        use_holographic: bool = True
    ) -> List[Dict[str, Any]]:
        """
        多跳检索：通过因果推理发现间接关联的记忆
        
        流程：
        1. 第一跳：标准 retrieve()，获取 top-k 直接匹配
        2. 对每个直接匹配的记忆，提取其八卦/五行属性
        3. 使用 CausalInference.multi_hop_inference() 扩展检索
        4. 去重 + 按综合分排序
        """
        # 第一跳：标准检索
        first_hop = await self.retrieve(
            collection=collection,
            query_vector=query_vector,
            user_id=user_id,
            query_text=query_text,
            limit=limit * 2,
            use_holographic=use_holographic,
        )

        if not first_hop or not self.causal_inference:
            return first_hop[:limit]

        # 提取查询的八卦/五行
        if query_text:
            query_hexagram = self.semantic_encoder.encode(query_text, "fact")
            from su_core._sys.encoders import BAGUA_NAMES, HEXAGRAM_TRIGRAMS_BELOW
            below_idx = HEXAGRAM_TRIGRAMS_BELOW[query_hexagram.index]
            query_bagua = BAGUA_NAMES[below_idx] if 0 <= below_idx < 8 else "坤"
            query_wuxing = BAGUA_WUXING.get(query_bagua, "土")
        else:
            query_bagua = "坤"
            query_wuxing = "土"

        # 构建 memories 列表供多跳推理
        memories_for_hop = []
        for m in first_hop:
            bagua = m.get("metadata", {}).get("bagua_name", "")
            wuxing = m.get("metadata", {}).get("wuxing", "")
            if not bagua:
                # 从 hexagram_index 推断
                from su_core._sys.encoders import BAGUA_NAMES, HEXAGRAM_TRIGRAMS_BELOW
                hex_idx = m.get("hexagram_index", 0)
                below = HEXAGRAM_TRIGRAMS_BELOW[hex_idx]
                bagua = BAGUA_NAMES[below] if 0 <= below < 8 else ""
            if not wuxing and bagua:
                wuxing = BAGUA_WUXING.get(bagua, "")
            memories_for_hop.append({
                "id": m.get("id", ""),
                "bagua_name": bagua,
                "wuxing": wuxing,
                **m,
            })

        # 多跳推理
        hop_results = self.causal_inference.multi_hop_inference(
            query_bagua=query_bagua,
            query_wuxing=query_wuxing,
            memories=memories_for_hop,
            max_hops=max_hops,
        )

        # 合并分数
        seen = set()
        final = []
        for r in hop_results:
            mid = r.get("id", "")
            if mid in seen:
                continue
            seen.add(mid)
            # 综合分 = 原始分 * 0.7 + 多跳分 * 0.3
            orig_score = r.get("score", 0)
            hop_score = r.get("hop_score", 0)
            r["combined_score"] = round(orig_score * 0.7 + hop_score * 0.3, 4)
            r["hop_count"] = r.get("hop_count", 0)
            r["hop_path"] = r.get("hop_path", [])
            final.append(r)

        final.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        return final[:limit]
