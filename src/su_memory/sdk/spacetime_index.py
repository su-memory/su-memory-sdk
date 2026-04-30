"""
时空索引模块 - SpatiotemporalIndex

功能：
- 时间编码（基于时间周期引擎）
- 时空联合检索
- 时间衰减权重
- 历史记忆增强

架构：
- 时间分桶索引：按天/周/月建立时间桶
- 时空联合搜索：向量相似度 × 时间衰减因子
- 历史增强：根据时间上下文调整检索结果
"""

import time
import math
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SpacetimeNode:
    """时空节点 - 包含时间信息"""
    id: str
    content: str
    vector: Optional[List[float]] = None
    timestamp: int = 0  # Unix 时间戳
    energy_type: str = "土"  # Energy System类型
    time_bucket: str = ""  # 时间桶标识
    neighbors: Dict[str, float] = field(default_factory=dict)


class TimeBucketIndex:
    """
    时间桶索引

    将记忆按时间分桶存储，支持快速时间范围查询
    """

    # 时间桶大小（秒）
    BUCKET_HOUR = 3600
    BUCKET_DAY = 86400
    BUCKET_WEEK = 86400 * 7
    BUCKET_MONTH = 86400 * 30

    def __init__(self, bucket_size: int = BUCKET_DAY):
        self.bucket_size = bucket_size
        self.buckets: Dict[int, List[str]] = defaultdict(list)  # bucket_key -> node_ids

    def get_bucket_key(self, timestamp: int) -> int:
        """获取时间桶键"""
        return timestamp // self.bucket_size

    def add_node(self, node_id: str, timestamp: int):
        """添加节点到时间桶"""
        bucket_key = self.get_bucket_key(timestamp)
        if node_id not in self.buckets[bucket_key]:
            self.buckets[bucket_key].append(node_id)

    def get_nodes_in_range(
        self,
        start_ts: int,
        end_ts: int,
        include_neighbors: bool = True
    ) -> List[str]:
        """获取时间范围内的节点"""
        start_key = self.get_bucket_key(start_ts)
        end_key = self.get_bucket_key(end_ts)

        nodes = []
        for key in range(start_key, end_key + 1):
            if key in self.buckets:
                nodes.extend(self.buckets[key])

        return nodes

    def get_recent_nodes(self, current_ts: int, hours: int = 24) -> List[str]:
        """获取最近的节点"""
        start_ts = current_ts - hours * 3600
        return self.get_nodes_in_range(start_ts, current_ts)

    def get_bucket_stats(self) -> Dict[str, Any]:
        """获取桶统计信息"""
        return {
            "n_buckets": len(self.buckets),
            "bucket_size": self.bucket_size,
            "total_nodes": sum(len(n) for n in self.buckets.values())
        }


class SpatiotemporalIndex:
    """
    时空联合索引

    结合：
    - 向量相似度搜索（VectorGraphRAG）
    - 时间衰减（TemporalSystem）
    - Energy System能量增强（Energy System）

    检索公式：
    final_score = vector_similarity × time_decay × energy_boost

    其中：
    - time_decay = exp(-λ × days_since_creation)
    - energy_boost = 根据当前时辰/Energy System调整
    """

    def __init__(
        self,
        embedding_func: Callable[[str], List[float]],
        dims: int = 1024,
        time_bucket_size: int = TimeBucketIndex.BUCKET_DAY,
        decay_base: float = 0.02,
        energy_boost_max: float = 1.3
    ):
        self.embedding_func = embedding_func
        self.dims = dims

        # 时间索引
        self.time_index = TimeBucketIndex(time_bucket_size)

        # 节点存储
        self.nodes: Dict[str, SpacetimeNode] = {}
        self.node_vectors: Dict[str, np.ndarray] = {}

        # 时间衰减参数
        self.decay_base = decay_base  # λ 参数
        self.energy_boost_max = energy_boost_max

        # 时间上下文
        self.current_energy = "土"
        self.current_stem = "甲"
        self.current_branch = "子"

        # 能量增强映射
        self.ENERGY_ENHANCE = {"木": "火", "火": "土", "土": "金", "金": "水", "水": "木"}
        self.ENERGY_SUPPRESS = {"木": "土", "土": "水", "水": "火", "火": "金", "金": "木"}

        # Energy System关键词
        self.ENERGY_KEYWORDS = {
            "木": ["生长", "发展", "树木", "森林", "绿色", "东方", "春季", "肝", "筋"],
            "火": ["热情", "炎热", "红色", "南方", "夏季", "心", "血液"],
            "土": ["稳定", "黄色", "中央", "四季", "脾", "消化"],
            "金": ["收敛", "白色", "西方", "秋季", "肺", "呼吸"],
            "水": ["流动", "蓝色", "北方", "冬季", "肾", "泌尿"]
        }

        # 时间分支能量映射
        self.BRANCH_ENERGY = {
            "子": "水", "丑": "土", "寅": "木", "卯": "木",
            "辰": "土", "巳": "火", "午": "火", "未": "土",
            "申": "金", "酉": "金", "戌": "土", "亥": "水"
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
        """获取时间编码（八字）"""
        ts = timestamp or int(time.time())

        # 简化计算：使用时间戳的循环特性
        # 60年甲子循环
        year = 1970 + (ts // 31556926)
        jiazi_year = (year - 1984) % 60

        stems = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
        branches = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]

        stem = stems[jiazi_year % 10]
        branch = branches[jiazi_year % 12]

        return {
            "stem": stem,
            "branch": branch,
            "energy": self.BRANCH_ENERGY.get(branch, "土")
        }

    def _calculate_time_decay(self, memory_ts: int, current_ts: int = None) -> float:
        """
        计算时间衰减因子

        公式：decay = exp(-λ × days)

        Args:
            memory_ts: 记忆创建时间戳
            current_ts: 当前时间戳
            lambda: 衰减系数（默认 0.02）

        Returns:
            衰减因子 (0, 1]
        """
        ts = current_ts or int(time.time())
        days = (ts - memory_ts) / 86400

        # 指数衰减
        decay = math.exp(-self.decay_base * days)

        # 短期记忆增强（7天内增强）
        if days < 1:
            decay *= 1.2
        elif days < 7:
            decay *= 1.1

        return max(0.1, min(1.0, decay))

    def _calculate_energy_boost(
        self,
        memory_energy: str,
        current_energy: str = None,
        time_code: Dict[str, str] = None
    ) -> float:
        """
        计算能量增强因子

        根据Energy System生克关系调整检索权重：
        - 增强：当前能量匹配记忆能量 → 增强 1.2x
        - 相克：当前能量克记忆能量 → 削弱 0.8x
        - 同气：同类能量 → 中等增强 1.1x
        """
        if time_code:
            current_energy = time_code.get("energy", current_energy or "土")
        else:
            current_energy = current_energy or "土"

        boost = 1.0

        # 增强关系
        if self.ENERGY_ENHANCE.get(current_energy) == memory_energy:
            boost = 1.2
        # 相克关系
        elif self.ENERGY_SUPPRESS.get(current_energy) == memory_energy:
            boost = 0.8
        # 同气
        elif memory_energy == current_energy:
            boost = 1.1

        return boost

    def update_temporal_context(self, timestamp: int = None):
        """更新当前时间上下文"""
        time_code = self._get_time_code(timestamp)
        self.current_energy = time_code.get("energy", "土")
        self.current_stem = time_code.get("stem", "甲")
        self.current_branch = time_code.get("branch", "子")

    def add_node(
        self,
        node_id: str,
        content: str,
        timestamp: int = None,
        vector: List[float] = None,
        energy_type: str = None
    ) -> bool:
        """
        添加时空节点

        Args:
            node_id: 节点ID
            content: 内容
            timestamp: 时间戳（默认当前）
            vector: 向量（可选，自动编码）
            energy_type: Energy System类型（可选，自动推断）

        Returns:
            是否成功
        """
        ts = timestamp or int(time.time())

        # 自动编码
        if vector is None and self.embedding_func:
            vector = self.embedding_func(content)

        if vector is None:
            return False

        # 推断Energy System
        if energy_type is None:
            energy_type = self._infer_energy_type(content)

        # 创建节点
        node = SpacetimeNode(
            id=node_id,
            content=content,
            vector=vector,
            timestamp=ts,
            energy_type=energy_type,
            time_bucket=str(ts // self.time_index.bucket_size)
        )

        self.nodes[node_id] = node
        self.node_vectors[node_id] = np.array(vector, dtype=np.float32)

        # 添加到时间索引
        self.time_index.add_node(node_id, ts)

        # 更新当前时间上下文
        self.update_temporal_context(ts)

        return True

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        weight: float = 1.0
    ):
        """添加边"""
        if source_id in self.nodes and target_id in self.nodes:
            self.nodes[source_id].neighbors[target_id] = weight
            self.nodes[target_id].neighbors[source_id] = weight

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_temporal: bool = True,
        time_range: Tuple[int, int] = None,
        energy_filter: str = None
    ) -> List[Dict]:
        """
        时空联合搜索

        Args:
            query: 查询文本
            top_k: 返回数量
            use_temporal: 是否使用时间加权
            time_range: 时间范围 (start_ts, end_ts)
            energy_filter: Energy System过滤

        Returns:
            List of {node_id, content, score, vector_score, time_decay, energy_boost}
        """
        # 编码查询
        query_vec = self.embedding_func(query) if self.embedding_func else None
        if query_vec is None:
            return []

        query_vec = np.array(query_vec, dtype=np.float32)

        # 候选节点
        if time_range:
            candidate_ids = self.time_index.get_nodes_in_range(time_range[0], time_range[1])
        else:
            candidate_ids = list(self.nodes.keys())

        # 过滤Energy System
        if energy_filter:
            candidate_ids = [
                nid for nid in candidate_ids
                if self.nodes[nid].energy_type == energy_filter
            ]

        # 计算时空得分
        results = []
        current_ts = int(time.time())
        time_code = self._get_time_code(current_ts)

        for node_id in candidate_ids:
            node = self.nodes[node_id]

            # 向量相似度
            if node.vector is not None:
                node_vec = self.node_vectors[node_id]
                vec_sim = float(np.dot(query_vec, node_vec) /
                              (np.linalg.norm(query_vec) * np.linalg.norm(node_vec) + 1e-8))
            else:
                vec_sim = 0.0

            # 时间衰减
            if use_temporal:
                time_decay = self._calculate_time_decay(node.timestamp, current_ts)

                # 能量增强
                energy_boost = self._calculate_energy_boost(
                    node.energy_type,
                    time_code=time_code
                )

                # 综合得分
                final_score = vec_sim * time_decay * energy_boost

                results.append({
                    "node_id": node_id,
                    "content": node.content,
                    "score": final_score,
                    "vector_score": vec_sim,
                    "time_decay": time_decay,
                    "energy_boost": energy_boost,
                    "timestamp": node.timestamp,
                    "energy_type": node.energy_type
                })
            else:
                results.append({
                    "node_id": node_id,
                    "content": node.content,
                    "score": vec_sim,
                    "vector_score": vec_sim,
                    "time_decay": 1.0,
                    "energy_boost": 1.0,
                    "timestamp": node.timestamp,
                    "energy_type": node.energy_type
                })

        # 排序
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    def search_multihop(
        self,
        query: str,
        max_hops: int = 3,
        top_k: int = 5,
        use_temporal: bool = True
    ) -> List[Dict]:
        """
        时空多跳搜索

        结合向量搜索和时空上下文的链式推理
        """
        # 第一跳：时空搜索
        seeds = self.search(query, top_k=top_k * 2, use_temporal=use_temporal)

        all_results = seeds[:]
        visited = {r["node_id"] for r in seeds}

        # 多跳扩展
        for hop in range(max_hops - 1):
            new_seeds = []

            for seed in seeds[:top_k]:
                seed_id = seed["node_id"]

                if seed_id not in self.nodes:
                    continue

                # 遍历邻居
                neighbors = self.nodes[seed_id].neighbors
                for neighbor_id, edge_weight in neighbors.items():
                    if neighbor_id in visited:
                        continue

                    if neighbor_id not in self.nodes:
                        continue

                    neighbor = self.nodes[neighbor_id]

                    # 计算时空得分
                    if neighbor.vector is not None:
                        query_vec = self.embedding_func(query) if self.embedding_func else None
                        if query_vec:
                            vec_sim = float(np.dot(
                                np.array(query_vec, dtype=np.float32),
                                self.node_vectors[neighbor_id]
                            ))
                        else:
                            vec_sim = 0.0
                    else:
                        vec_sim = 0.0

                    # 时间衰减
                    current_ts = int(time.time())
                    time_decay = self._calculate_time_decay(neighbor.timestamp, current_ts)

                    # 能量增强
                    energy_boost = self._calculate_energy_boost(neighbor.energy_type)

                    # 综合得分（带跳数衰减）
                    hop_decay = 0.9 ** (hop + 1)
                    final_score = seed["score"] * edge_weight * vec_sim * time_decay * energy_boost * hop_decay

                    result = {
                        "node_id": neighbor_id,
                        "content": neighbor.content,
                        "score": final_score,
                        "vector_score": vec_sim,
                        "time_decay": time_decay,
                        "energy_boost": energy_boost,
                        "hop": hop + 2,
                        "timestamp": neighbor.timestamp,
                        "energy_type": neighbor.energy_type
                    }

                    all_results.append(result)
                    new_seeds.append(result)
                    visited.add(neighbor_id)

            seeds = new_seeds

        # 最终排序
        all_results.sort(key=lambda x: x["score"], reverse=True)

        return all_results[:top_k]

    def get_temporal_context(self, timestamp: int = None) -> Dict[str, Any]:
        """获取时间上下文"""
        ts = timestamp or int(time.time())
        time_code = self._get_time_code(ts)

        # 获取近期节点
        recent_nodes = self.time_index.get_recent_nodes(ts, hours=24)

        return {
            "timestamp": ts,
            "time_code": time_code,
            "current_energy": self.current_energy,
            "recent_count": len(recent_nodes),
            "energy_distribution": self._get_energy_distribution(recent_nodes)
        }

    def _get_energy_distribution(self, node_ids: List[str]) -> Dict[str, int]:
        """获取Energy System分布"""
        dist = {"木": 0, "火": 0, "土": 0, "金": 0, "水": 0}
        for nid in node_ids:
            if nid in self.nodes:
                e = self.nodes[nid].energy_type
                if e in dist:
                    dist[e] += 1
        return dist

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "n_nodes": len(self.nodes),
            "time_buckets": self.time_index.get_bucket_stats(),
            "current_energy": self.current_energy,
            "current_time_code": self._get_time_code()
        }


# ============================================================
# 工厂函数
# ============================================================

def create_spatiotemporal_index(
    embedding_func: Callable[[str], List[float]],
    dims: int = 1024,
    bucket_size: int = TimeBucketIndex.BUCKET_DAY
) -> SpatiotemporalIndex:
    """创建时空索引"""
    return SpatiotemporalIndex(
        embedding_func=embedding_func,
        dims=dims,
        time_bucket_size=bucket_size
    )
