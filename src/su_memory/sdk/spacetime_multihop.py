"""
时空多跳融合引擎 - SpacetimeMultihopEngine

功能：
- 融合 VectorGraphRAG（语义引导）+ SpacetimeIndex（时空加权）
- 时空多跳推理：结合语义相似度、时间衰减、Energy System能量
- RRF融合排序：统一不同引擎的得分

架构：
┌─────────────────────────────────────────────────────────────┐
│                    SpacetimeMultihopEngine                   │
├─────────────────────────────────────────────────────────────┤
│  输入: query, max_hops                                       │
│                                                             │
│  ┌───────────────────┐    ┌───────────────────────────┐    │
│  │   VectorGraphRAG   │    │     SpacetimeIndex         │    │
│  │   - 语义种子搜索   │    │     - 时空联合搜索          │    │
│  │   - 向量扩展       │    │     - 时间衰减              │    │
│  │   - 因果推理       │    │     - Energy System能量增强          │    │
│  └─────────┬──────────┘    └─────────────┬───────────────┘    │
│            │                            │                   │
│            ▼                            ▼                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              时空加权融合模块                          │  │
│  │  - vector_score × time_decay × energy_boost            │  │
│  │  - hop_count 衰减                                       │  │
│  └────────────────────────┬─────────────────────────────┘  │
│                           │                                 │
│                           ▼                                 │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              RRF 融合排序                              │  │
│  │  - rank_score = Σ(1 / (k + rank_i))                    │  │
│  │  - k = 60 (标准RRF参数)                                │  │
│  └────────────────────────┬─────────────────────────────┘  │
│                           │                                 │
│                           ▼                                 │
│  输出: List[SpacetimeHopResult]                             │
└─────────────────────────────────────────────────────────────┘
"""

import time
import math
from typing import List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field


@dataclass
class SpacetimeHopResult:
    """
    时空多跳结果

    包含：
    - 节点信息（id, content）
    - 得分（综合得分、向量得分、时间衰减、能量增强）
    - 跳数信息（hops, path）
    - 时空信息（timestamp, energy_type）
    """
    node_id: str
    content: str
    score: float  # 综合得分
    vector_score: float = 0.0  # 向量相似度
    time_decay: float = 1.0  # 时间衰减因子
    energy_boost: float = 1.0  # 能量增强因子
    hops: int = 1  # 跳数
    path: List[str] = field(default_factory=list)  # 路径
    causal_type: str = "semantic"  # 因果类型
    timestamp: int = 0  # 时间戳
    energy_type: str = "土"  # Energy System类型
    source: str = "unknown"  # 来源引擎


class SpacetimeMultihopEngine:
    """
    时空多跳融合引擎

    核心特点：
    1. 双引擎并行：VectorGraphRAG（语义） + SpacetimeIndex（时空）
    2. 时空加权：为语义结果添加时间衰减和能量增强
    3. RRF融合：统一不同引擎的得分进行排序

    使用方法：
        engine = SpacetimeMultihopEngine(
            vector_graph=vg,  # VectorGraphRAG实例
            spacetime=st,    # SpacetimeIndex实例
            embedding_func=encode_func
        )

        results = engine.search(
            query="深度学习的影响",
            max_hops=3,
            top_k=5,
            use_spacetime_weight=True
        )
    """

    # RRF融合参数
    RRF_K = 60  # 标准RRF参数

    # 时空权重参数
    DEFAULT_TIME_DECAY_BASE = 0.02  # λ 参数
    DEFAULT_ENERGY_BOOST_MAX = 1.3  # 最大能量增强

    def __init__(
        self,
        vector_graph=None,  # VectorGraphRAG实例
        spacetime=None,  # SpacetimeIndex实例
        memory_nodes: Dict[str, Any] = None,  # 主存储的节点映射
        embedding_func: Callable[[str], List[float]] = None,
        rrf_k: int = 60,
        time_decay_base: float = 0.02,
        energy_boost_max: float = 1.3
    ):
        self.vector_graph = vector_graph
        self.spacetime = spacetime
        self.memory_nodes = memory_nodes or {}  # memory_id -> node对象
        self.embedding_func = embedding_func

        # 融合参数
        self.rrf_k = rrf_k
        self.time_decay_base = time_decay_base
        self.energy_boost_max = energy_boost_max

        # Energy System增强映射
        self.ENERGY_ENHANCE = {"木": "火", "火": "土", "土": "金", "金": "水", "水": "木"}
        self.ENERGY_SUPPRESS = {"木": "土", "土": "水", "水": "火", "火": "金", "金": "木"}
        self.BRANCH_ENERGY = {
            "子": "水", "丑": "土", "寅": "木", "卯": "木",
            "辰": "土", "巳": "火", "午": "火", "未": "土",
            "申": "金", "酉": "金", "戌": "土", "亥": "水"
        }

        # Energy System关键词
        self.ENERGY_KEYWORDS = {
            "木": ["生长", "发展", "树木", "森林", "绿色", "东方", "春季", "肝", "筋"],
            "火": ["热情", "炎热", "红色", "南方", "夏季", "心", "血液"],
            "土": ["稳定", "黄色", "中央", "四季", "脾", "消化"],
            "金": ["收敛", "白色", "西方", "秋季", "肺", "呼吸"],
            "水": ["流动", "蓝色", "北方", "冬季", "肾", "泌尿"]
        }

    def _infer_energy_type(self, content: str) -> str:
        """从内容推断Energy System类型"""
        scores = {e: 0 for e in self.ENERGY_KEYWORDS}
        for e, kws in self.ENERGY_KEYWORDS.items():
            for kw in kws:
                if kw in content:
                    scores[e] += 1
        return max(scores, key=scores.get) if max(scores.values()) > 0 else "土"

    def _get_time_code(self, timestamp: int = None) -> Dict[str, str]:
        """获取时间编码"""
        ts = timestamp or int(time.time())
        year = 1970 + (ts // 31556926)
        jiazi_year = (year - 1984) % 60

        stems = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
        branches = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]

        return {
            "stem": stems[jiazi_year % 10],
            "branch": branches[jiazi_year % 12],
            "energy": self.BRANCH_ENERGY.get(branches[jiazi_year % 12], "土")
        }

    def _calculate_time_decay(self, memory_ts: int, current_ts: int = None) -> float:
        """计算时间衰减因子"""
        ts = current_ts or int(time.time())
        days = (ts - memory_ts) / 86400

        decay = math.exp(-self.time_decay_base * days)

        # 短期记忆增强
        if days < 1:
            decay *= 1.2
        elif days < 7:
            decay *= 1.1

        return max(0.1, min(1.0, decay))

    def _calculate_energy_boost(self, memory_energy: str, current_energy: str = None) -> float:
        """计算能量增强因子"""
        current_energy = current_energy or "土"
        boost = 1.0

        if self.ENERGY_ENHANCE.get(current_energy) == memory_energy:
            boost = 1.2
        elif self.ENERGY_SUPPRESS.get(current_energy) == memory_energy:
            boost = 0.8
        elif memory_energy == current_energy:
            boost = 1.1

        return boost

    def _get_node_info(self, node_id: str) -> Tuple[str, int, str]:
        """获取节点信息（content, timestamp, energy_type）"""
        # 从主存储获取
        if self.memory_nodes and node_id in self.memory_nodes:
            node = self.memory_nodes[node_id]
            content = getattr(node, 'content', '')
            timestamp = getattr(node, 'timestamp', int(time.time()))
            energy_type = getattr(node, 'energy_type', self._infer_energy_type(content))
            return content, timestamp, energy_type

        # 从 SpacetimeIndex 获取
        if self.spacetime and node_id in self.spacetime.nodes:
            node = self.spacetime.nodes[node_id]
            return node.content, node.timestamp, node.energy_type

        # 从 VectorGraphRAG 获取
        if self.vector_graph and node_id in self.vector_graph.nodes:
            node = self.vector_graph.nodes[node_id]
            return node.content, int(time.time()), self._infer_energy_type(node.content)

        return "", int(time.time()), "土"

    def _spacetime_weight(
        self,
        node_id: str,
        vector_score: float,
        hops: int = 1
    ) -> SpacetimeHopResult:
        """
        为节点添加时空加权

        Args:
            node_id: 节点ID
            vector_score: 向量相似度
            hops: 跳数

        Returns:
            SpacetimeHopResult，包含综合得分
        """
        content, timestamp, energy_type = self._get_node_info(node_id)

        current_ts = int(time.time())
        time_code = self._get_time_code(current_ts)

        # 计算时空因子
        time_decay = self._calculate_time_decay(timestamp, current_ts)
        energy_boost = self._calculate_energy_boost(energy_type, time_code["energy"])

        # 跳数衰减
        hop_decay = 0.95 ** (hops - 1)

        # 综合得分
        final_score = vector_score * time_decay * energy_boost * hop_decay

        return SpacetimeHopResult(
            node_id=node_id,
            content=content,
            score=final_score,
            vector_score=vector_score,
            time_decay=time_decay,
            energy_boost=energy_boost,
            hops=hops,
            timestamp=timestamp,
            energy_type=energy_type
        )

    def _vector_graph_search(self, query: str, max_hops: int, top_k: int) -> List[SpacetimeHopResult]:
        """
        VectorGraphRAG 多跳搜索

        Args:
            query: 查询文本
            max_hops: 最大跳数
            top_k: 返回数量

        Returns:
            SpacetimeHopResult 列表
        """
        if not self.vector_graph or len(self.vector_graph) == 0:
            return []

        try:
            vg_results = self.vector_graph.multi_hop_query(
                query, max_hops=max_hops, top_k=top_k * 2
            )

            results = []
            for r in vg_results:
                # 添加时空加权
                st_result = self._spacetime_weight(r.node_id, r.score, r.hops)
                st_result.path = r.path
                st_result.causal_type = r.causal_type
                st_result.source = "vector_graph"
                results.append(st_result)

            return results
        except Exception as e:
            print(f"[SpacetimeMultihopEngine] VectorGraphRAG 搜索失败: {e}")
            return []

    def _spacetime_search(self, query: str, max_hops: int, top_k: int) -> List[SpacetimeHopResult]:
        """
        SpacetimeIndex 多跳搜索

        Args:
            query: 查询文本
            max_hops: 最大跳数
            top_k: 返回数量

        Returns:
            SpacetimeHopResult 列表
        """
        if not self.spacetime or len(self.spacetime.nodes) == 0:
            return []

        try:
            st_results = self.spacetime.search_multihop(
                query, max_hops=max_hops, top_k=top_k * 2
            )

            results = []
            for r in st_results:
                results.append(SpacetimeHopResult(
                    node_id=r["node_id"],
                    content=r["content"],
                    score=r["score"],
                    vector_score=r.get("vector_score", r["score"]),
                    time_decay=r.get("time_decay", 1.0),
                    energy_boost=r.get("energy_boost", 1.0),
                    hops=r.get("hop", 1),
                    timestamp=r.get("timestamp", 0),
                    energy_type=r.get("energy_type", "土"),
                    source="spacetime"
                ))

            return results
        except Exception as e:
            print(f"[SpacetimeMultihopEngine] SpacetimeIndex 搜索失败: {e}")
            return []

    def _rrf_fusion(
        self,
        results_list: List[List[SpacetimeHopResult]],
        weights: List[float] = None
    ) -> List[SpacetimeHopResult]:
        """
        RRF融合（Reciprocal Rank Fusion）

        公式：score(q) = Σ(1 / (k + rank_i(q)))

        Args:
            results_list: 多个引擎的结果列表
            weights: 引擎权重（可选）

        Returns:
            融合后的排序结果
        """
        if not results_list:
            return []

        if weights is None:
            weights = [1.0] * len(results_list)

        # 收集所有节点
        all_nodes: Dict[str, SpacetimeHopResult] = {}

        for results in results_list:
            for rank, result in enumerate(results):
                node_id = result.node_id

                if node_id in all_nodes:
                    # 累加RRF得分
                    rrf_contribution = weights[results_list.index(results)] / (self.rrf_k + rank + 1)
                    all_nodes[node_id].score += rrf_contribution
                else:
                    # 添加新节点
                    result.score = weights[results_list.index(results)] / (self.rrf_k + rank + 1)
                    all_nodes[node_id] = result

        # 排序
        sorted_results = sorted(
            all_nodes.values(),
            key=lambda x: x.score,
            reverse=True
        )

        return sorted_results

    def search(
        self,
        query: str,
        max_hops: int = 3,
        top_k: int = 5,
        use_vector_graph: bool = True,
        use_spacetime: bool = True,
        spacetime_weight: float = 1.0,  # 时空权重（0-2）
        vector_weight: float = 1.0,  # 向量权重（0-2）
        fusion_mode: str = "auto"  # "auto", "spacetime_first", "vector_first", "hybrid"
    ) -> List[SpacetimeHopResult]:
        """
        时空多跳融合搜索

        Args:
            query: 查询文本
            max_hops: 最大跳数
            top_k: 返回数量
            use_vector_graph: 是否使用 VectorGraphRAG
            use_spacetime: 是否使用 SpacetimeIndex
            spacetime_weight: 时空引擎权重（0-2）
            vector_weight: 向量引擎权重（0-2）
            fusion_mode: 融合模式
                - "auto": 自动选择（基于数据可用性）
                - "spacetime_first": SpacetimeIndex 优先
                - "vector_first": VectorGraphRAG 优先
                - "hybrid": 两者均衡融合

        Returns:
            SpacetimeHopResult 列表
        """
        results_list = []
        weights = []

        # VectorGraphRAG 搜索
        if use_vector_graph and self.vector_graph:
            vg_results = self._vector_graph_search(query, max_hops, top_k)
            if vg_results:
                results_list.append(vg_results)
                weights.append(vector_weight)

        # SpacetimeIndex 搜索
        if use_spacetime and self.spacetime:
            st_results = self._spacetime_search(query, max_hops, top_k)
            if st_results:
                results_list.append(st_results)
                weights.append(spacetime_weight)

        if not results_list:
            return []

        # 融合模式调整
        if fusion_mode == "spacetime_first":
            weights = [spacetime_weight * 2, vector_weight]
        elif fusion_mode == "vector_first":
            weights = [spacetime_weight, vector_weight * 2]
        elif fusion_mode == "hybrid":
            weights = [spacetime_weight, vector_weight]

        # RRF融合
        fused = self._rrf_fusion(results_list, weights)

        return fused[:top_k]

    def search_with_filter(
        self,
        query: str,
        max_hops: int = 3,
        top_k: int = 5,
        time_range: Tuple[int, int] = None,
        energy_filter: str = None,
        min_time_decay: float = 0.3
    ) -> List[SpacetimeHopResult]:
        """
        带过滤的时空多跳搜索

        Args:
            query: 查询文本
            max_hops: 最大跳数
            top_k: 返回数量
            time_range: 时间范围 (start_ts, end_ts)
            energy_filter: Energy System过滤
            min_time_decay: 最小时间衰减阈值

        Returns:
            过滤后的 SpacetimeHopResult 列表
        """
        # 执行搜索
        results = self.search(query, max_hops, top_k * 3)

        # 应用过滤
        filtered = []
        for r in results:
            # 时间范围过滤
            if time_range:
                if r.timestamp < time_range[0] or r.timestamp > time_range[1]:
                    continue

            # Energy System过滤
            if energy_filter and r.energy_type != energy_filter:
                continue

            # 时间衰减过滤
            if r.time_decay < min_time_decay:
                continue

            filtered.append(r)

        return filtered[:top_k]

    def get_temporal_context(self) -> Dict[str, Any]:
        """获取当前时空上下文"""
        if self.spacetime:
            return self.spacetime.get_temporal_context()

        current_ts = int(time.time())
        time_code = self._get_time_code(current_ts)

        return {
            "timestamp": current_ts,
            "time_code": time_code,
            "current_energy": time_code["energy"],
            "n_nodes": len(self.memory_nodes) if self.memory_nodes else 0
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        vg_nodes = len(self.vector_graph.nodes) if self.vector_graph else 0
        st_nodes = len(self.spacetime.nodes) if self.spacetime else 0
        main_nodes = len(self.memory_nodes) if self.memory_nodes else 0

        ctx = self.get_temporal_context()

        return {
            "vector_graph_nodes": vg_nodes,
            "spacetime_nodes": st_nodes,
            "main_nodes": main_nodes,
            "current_energy": ctx.get("current_energy", "土"),
            "time_code": ctx.get("time_code", {}),
            "fusion_modes": ["auto", "spacetime_first", "vector_first", "hybrid"]
        }


# ============================================================
# 工厂函数
# ============================================================

def create_spacetime_multihop_engine(
    vector_graph=None,
    spacetime=None,
    memory_nodes: Dict[str, Any] = None,
    embedding_func: Callable[[str], List[float]] = None
) -> SpacetimeMultihopEngine:
    """
    创建时空多跳融合引擎

    Args:
        vector_graph: VectorGraphRAG实例
        spacetime: SpacetimeIndex实例
        memory_nodes: 主存储节点映射
        embedding_func: 编码函数

    Returns:
        SpacetimeMultihopEngine 实例
    """
    return SpacetimeMultihopEngine(
        vector_graph=vector_graph,
        spacetime=spacetime,
        memory_nodes=memory_nodes,
        embedding_func=embedding_func
    )
