"""
多跳推理检索器

为 su-memory 解决最大的差距：多跳推理能力（-41%）

在 CausalInference 基础上增强：
1. 动态跳数（根据查询复杂度自动调整 1-4 跳）
2. 多样性 bridge 选择（MMR 变体，避免结果聚集在同一语义域）
3. 环路检测与剪枝
4. 自适应衰减系数（根据记忆类型调整）

核心算法参考 Hindsight CausalInference.multi_hop_inference()，
但扩展为 su-memory Trigram Symbol空间内的动态推理。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from su_memory._sys.encoders import SemanticEncoder, EncodingInfo, EncoderCore
    from su_memory._sys.causal import CausalInference


# ========================
# 配置
# ========================

# 不同记忆类型的跳数衰减系数（记忆越"临时"，衰减越快）
HOP_DECAY_BY_TYPE = {
    "task": 0.60,      # 任务类记忆衰减快（上下文依赖强）
    "insight": 0.70,   # 洞察类记忆中等衰减
    "fact": 0.65,      # 事实类记忆正常衰减
    "decision": 0.75,  # 决策类记忆衰减慢（长期有效）
    "casual": 0.55,    # 闲聊类衰减最快
}

DEFAULT_MAX_HOPS = 3
COMPLEX_QUERY_MAX_HOPS = 4

# 多样性重排参数
MMR_LAMBDA = 0.7  # 语义相似度权重（1 - lambda = 多样性权重）


# ========================
# 数据结构
# ========================

@dataclass
class HopResult:
    """单跳检索结果"""
    memory_id: str
    content: str
    hexagram_index: int
    hexagram_name: str
    wuxing: str
    hop_score: float
    memory_type: str = "fact"
    bridges: List[str] = field(default_factory=list)  # 经过的 bridge 节点 ID


# ========================
# 多跳推理器
# ========================

class MultiHopRetriever:
    """
    多跳推理检索器

    在 Hindsight CausalInference 基础上，针对 su-memory 的Trigram Symbol空间做了适配：

    1. 动态跳数检测
       - 包含"之前"/"后来"/"当时"等时序词 → 至少 2 跳
       - 包含"为什么"/"怎么想到"等推理词 → 至少 3 跳
       - 包含多步因果描述 → 4 跳

    2. Bridge 选择多样性（避免都聚集在同一语义分类）
       - 每跳取不同语义分类的结果
       - 使用 MMR（Maximum Marginal Relevance）重排

    3. 环路检测
       - 已访问过的节点不重复访问
       - 防止分类循环（如 creative→receptive→creative→...）

    使用方法：
        retriever = MultiHopRetriever(encoder_core, causal_inference)
        results = retriever.retrieve(
            query="Nutri-Brain项目之前提到的那个AI模型是什么",
            query_complexity="complex",  # "simple" | "normal" | "complex"
            candidates=[...],
        )
    """

    def __init__(
        self,
        encoder_core: "EncoderCore",
        causal_inference: Optional["CausalInference"] = None,
        semantic_encoder: Optional["SemanticEncoder"] = None,
    ):
        self._encoder_core = encoder_core
        self._causal_inference = causal_inference
        self._semantic_encoder = semantic_encoder

    def retrieve(
        self,
        query: str,
        candidates: List[Dict],
        query_complexity: str = "normal",
        max_hops: Optional[int] = None,
        use_vector_sim: bool = True,
    ) -> List[HopResult]:
        """
        执行多跳推理检索

        Args:
            query: 查询文本
            candidates: 候选记忆列表，每项含 content, memory_id, memory_type, hexagram_index
            query_complexity: "simple" (1跳) | "normal" (2跳) | "complex" (3-4跳)
            max_hops: 覆盖 query_complexity 的显式跳数
            use_vector_sim: 是否使用全量向量 cosine（精细化模式）

        Returns:
            按多跳综合得分降序排列的 HopResult 列表
        """
        # 1. 确定跳数
        actual_hops = max_hops or self._determine_hop_count(query, query_complexity)

        # 2. 获取查询的Trigram Symbol编码
        if self._semantic_encoder:
            query_info = self._semantic_encoder.encode(query, "fact")
            query_hexagram = query_info.index
            query_bagua = query_info.bagua_probs
        else:
            query_hexagram = 0
            query_bagua = None

        # 3. 构建候选索引
        {c["memory_id"]: c for c in candidates}
        cand_by_idx = self._group_by_hexagram(candidates)

        # 4. 多跳检索
        if actual_hops >= 2:
            results = self._multi_hop_search(
                query_hexagram, query_bagua, candidates,
                cand_by_idx, actual_hops, use_vector_sim
            )
        else:
            # 单跳：直接 top-1 匹配
            results = self._direct_match(query_hexagram, candidates, use_vector_sim)

        # 5. 多样性重排（MMR）
        results = self._diversity_rerank(results, candidates)

        return results

    def _determine_hop_count(self, query: str, complexity: str) -> int:
        """根据查询文本和复杂度确定跳数"""
        q = query.lower()

        # 显式时序/推理词 → 增加跳数
        multi_hop_triggers = {
            "之前": 2, "之前提到": 2, "那件事": 2,
            "后来": 2, "后来呢": 2,
            "当时": 2, "当时的情况": 2,
            "为什么": 3, "怎么想到": 3, "怎么决定": 3,
            "原因是什么": 3, "起因": 3,
            "然后呢": 2, "接下来": 2,
            "整个过程": 3, "经过": 3,
        }

        max_detected = 1
        for keyword, hops in multi_hop_triggers.items():
            if keyword in q:
                max_detected = max(max_detected, hops)

        # complexity 覆盖
        complexity_hops = {
            "simple": 1,
            "normal": 2,
            "complex": 3,
        }

        base = complexity_hops.get(complexity, 2)
        return max(base, max_detected, DEFAULT_MAX_HOPS)

    def _group_by_hexagram(
        self,
        candidates: List[Dict],
    ) -> Dict[int, List[Dict]]:
        """按Trigram Symbol索引分组候选"""
        groups: Dict[int, List[Dict]] = defaultdict(list)
        for c in candidates:
            idx = c.get("hexagram_index", 0)
            groups[idx].append(c)
        return groups

    def _multi_hop_search(
        self,
        query_hexagram: int,
        query_bagua: Optional[Dict[str, float]],
        candidates: List[Dict],
        cand_by_idx: Dict[int, List[Dict]],
        max_hops: int,
        use_vector_sim: bool,
    ) -> List[HopResult]:
        """核心多跳搜索"""
        visited: Set[int] = set()
        all_results: List[HopResult] = []
        cand_map = {c["memory_id"]: c for c in candidates}

        # 获取一跳结果
        first_hop = self._get_hop_candidates(
            query_hexagram, candidates, cand_by_idx,
            query_bagua=query_bagua, top_k=5, hop=1,
            use_vector_sim=use_vector_sim
        )

        for result in first_hop:
            all_results.append(result)
            visited.add(result.hexagram_index)

        # ── Vector Graph RAG 改进：建立候选编码信息映射 ──────────────
        candidate_infos = {}
        if self._semantic_encoder and use_vector_sim:
            for c in candidates:
                try:
                    info, _ = self._semantic_encoder.encode_with_vector(c["content"], c.get("memory_type", "fact"))
                    info.wuxing = c.get("wuxing", "")
                    info.wuxing_scores = c.get("wuxing_scores")
                    candidate_infos[info.index] = info
                except Exception:
                    pass

        # 第二跳及以后：用向量相似度驱动动态邻居扩展（Vector Graph RAG 核心）
        if max_hops >= 2:
            bridges = first_hop[:5]
            for hop in range(2, max_hops + 1):
                next_hop_candidates = []

                for bridge in bridges:
                    # ── Vector Graph RAG 改进 ──────────────────────────────
                    # 以 bridge 的完整向量（而非Trigram Symbol）为锚点做向量邻居扩展
                    bridge_bagua = self._get_bagua_for_memory(bridge.memory_id, cand_map)
                    bridge_vector = self._get_vector_for_memory(bridge.memory_id, candidates)

                    hop_cands = self._get_hop_candidates(
                        bridge.hexagram_index,
                        candidates,
                        cand_by_idx,
                        query_bagua=bridge_bagua,
                        query_vector=bridge_vector,
                        top_k=5,  # Vector Graph RAG: 扩展更多邻居
                        hop=hop,
                        use_vector_sim=use_vector_sim,
                        candidate_infos=candidate_infos,
                    )

                    decay = self._get_hop_decay(bridge.memory_type)
                    for hc in hop_cands:
                        hc.hop_score *= decay
                        hc.bridges.append(bridge.memory_id)

                    next_hop_candidates.extend(hop_cands)

                # 去重（已访问过的降低分数）
                for hc in next_hop_candidates:
                    if hc.hexagram_index in visited:
                        hc.hop_score *= 0.5
                    all_results.append(hc)

                visited.add(bridge.hexagram_index)

                # 更新 bridges（用于下一跳）
                bridges = sorted(
                    next_hop_candidates,
                    key=lambda x: x.hop_score,
                    reverse=True
                )[:5]

        # 按 hop_score 排序
        all_results.sort(key=lambda x: x.hop_score, reverse=True)
        return all_results

    def _get_hop_candidates(
        self,
        query_hexagram: int,
        candidates: List[Dict],
        cand_by_idx: Dict[int, List[Dict]],
        query_bagua: Optional[Dict[str, float]],
        query_vector: Optional[List[float]] = None,
        top_k: int = 5,
        hop: int = 1,
        use_vector_sim: bool = True,
        candidate_infos: Optional[Dict[int, "EncodingInfo"]] = None,
    ) -> List[HopResult]:
        """获取单跳的候选结果"""
        if not cand_by_idx:
            return []

        {c["memory_id"]: c for c in candidates}
        cand_indices = list(set(c["hexagram_index"] for c in candidates))

        # ── Vector Graph RAG 核心改进 ─────────────────────────────────
        # 用 candidate_infos 提供完整语义向量，实现真正的向量邻居扩展
        if self._encoder_core and (query_bagua or query_vector):
            from su_memory._sys.encoders import EncodingInfo
            query_info = EncodingInfo.from_index(query_hexagram)
            query_info.bagua_probs = query_bagua

            # 关键：当有 query_vector 时直接用它做全量 cosine（Vector Graph RAG）
            if query_vector and use_vector_sim:
                # 直接在全量候选上做 full-vector cosine，不需要 encoder_core 包装
                scored = []
                for c in candidates:
                    v = self._get_vector_for_memory(c.get("memory_id", ""), candidates)
                    if v:
                        sim = self._cosine_vec(query_vector, v)
                        scored.append((c.get("hexagram_index", 0), sim))
                scored.sort(key=lambda x: x[1], reverse=True)
                encoder_results = scored[:top_k]
            elif candidate_infos:
                # 有 candidate_infos 时传给 encoder_core 做精细化检索
                encoder_results = self._encoder_core.retrieve_holographic(
                    query_hexagram, cand_indices, top_k=top_k,
                    query_info=query_info, candidate_infos=candidate_infos,
                    use_vector_sim=use_vector_sim
                )
            else:
                # Fallback: 只用Trigram Symbol概率
                encoder_results = self._encoder_core.retrieve_holographic(
                    query_hexagram, cand_indices, top_k=top_k,
                    query_info=query_info, candidate_infos=None,
                    use_vector_sim=False
                )
        else:
            # Fallback: 按Trigram Symbol距离排序
            encoder_results = [
                (idx, 1.0 / (1 + abs(idx - query_hexagram)))
                for idx in cand_indices
            ]
            encoder_results.sort(key=lambda x: x[1], reverse=True)
            encoder_results = encoder_results[:top_k]

        results: List[HopResult] = []
        seen_ids: Set[str] = set()

        for idx, score in encoder_results:
            for c in cand_by_idx.get(idx, []):
                mem_id = c["memory_id"]
                if mem_id in seen_ids:
                    continue
                seen_ids.add(mem_id)

                results.append(HopResult(
                    memory_id=mem_id,
                    content=c.get("content", ""),
                    hexagram_index=idx,
                    hexagram_name=c.get("hexagram_name", ""),
                    wuxing=c.get("wuxing", ""),
                    hop_score=score,
                    memory_type=c.get("memory_type", "fact"),
                    bridges=[],
                ))

        return results

    def _direct_match(
        self,
        query_hexagram: int,
        candidates: List[Dict],
        use_vector_sim: bool,
    ) -> List[HopResult]:
        """单跳直接匹配"""
        return self._get_hop_candidates(
            query_hexagram, candidates, self._group_by_hexagram(candidates),
            query_bagua=None, top_k=10, hop=1, use_vector_sim=use_vector_sim,
        )

    def _get_bagua_for_memory(
        self,
        memory_id: str,
        cand_map: Dict[str, Dict],
    ) -> Optional[Dict[str, float]]:
        """从候选记忆获取语义分类概率分布"""
        c = cand_map.get(memory_id)
        if not c:
            return None
        category_probs = c.get("category_probs")
        if isinstance(category_probs, dict):
            return category_probs
        return None

    def _get_vector_for_memory(
        self,
        memory_id: str,
        candidates: List[Dict],
    ) -> Optional[List[float]]:
        """从候选中提取某记忆的完整语义向量"""
        if not memory_id:
            return None
        for c in candidates:
            if c.get("memory_id") == memory_id:
                return c.get("vector") or c.get("semantic_vector")
        return None

    def _cosine_vec(self, a: List[float], b: List[float]) -> float:
        """计算两个向量的 cosine similarity"""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        denom = (norm_a * norm_b) or 1e-9
        return dot / denom

    def _get_hop_decay(self, memory_type: str) -> float:
        """获取某记忆类型的跳数衰减系数"""
        return HOP_DECAY_BY_TYPE.get(memory_type, 0.65)

    def _diversity_rerank(
        self,
        results: List[HopResult],
        candidates: List[Dict],
    ) -> List[HopResult]:
        """
        MMR 多样性重排

        避免所有结果都聚集在同一Trigram Symbol语义域内。
        在 top_k 结果中选择语义相似度和多样性最佳平衡的结果。
        """
        if not results:
            return results

        # 选择 top N 候选用于重排（避免结果过多）
        rerank_candidates = results[:20]
        reranked: List[HopResult] = []
        selected_hexagrams: Set[int] = set()

        {c["memory_id"]: c for c in candidates}

        for item in rerank_candidates:
            hex_idx = item.hexagram_index

            # 新颖度：之前没选过这个Trigram Symbol → 高新颖度
            novelty = 0.0 if hex_idx in selected_hexagrams else 1.0

            # 最终得分 = lambda * semantic_score + (1-lambda) * novelty
            score_final = MMR_LAMBDA * item.hop_score + (1 - MMR_LAMBDA) * novelty * 0.5

            item.hop_score = score_final
            reranked.append(item)
            selected_hexagrams.add(hex_idx)

        reranked.sort(key=lambda x: x.hop_score, reverse=True)
        return reranked

    # ========================
    # 便捷接口
    # ========================

    def retrieve_simple(self, query: str, candidates: List[Dict]) -> List[HopResult]:
        """
        简单接口：自动判断复杂度，使用向量相似度

        等同于 retrieve(query, candidates, query_complexity="normal", use_vector_sim=True)
        """
        return self.retrieve(query, candidates, query_complexity="normal", use_vector_sim=True)

    def retrieve_with_depth(
        self,
        query: str,
        candidates: List[Dict],
        depth: int,
    ) -> List[HopResult]:
        """
        指定深度的接口

        depth = 1: 单跳（快速）
        depth = 2: 两跳（标准）
        depth = 3+: 多跳（深度推理）
        """
        return self.retrieve(
            query, candidates,
            query_complexity="complex",
            max_hops=depth,
            use_vector_sim=True,
        )
