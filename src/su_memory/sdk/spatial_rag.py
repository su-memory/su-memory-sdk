"""
SpatialRAG 三维世界模型模块

功能：
- 空间坐标索引（2D/3D）
- 时空轨迹追踪
- 三维检索融合（空间+时间+语义）
- 场景理解

架构：
- SpatialIndex: 空间索引（基于 R-tree 或 KD-tree）
- SpatialNode: 带空间坐标的记忆节点
- TrajectoryTracker: 轨迹追踪器
- SpatialRAG: 三维世界模型融合引擎

使用方式：
    from su_memory.sdk.spatial_rag import SpatialRAG
    
    sr = SpatialRAG(
        embedding_func=encode_func,
        spacetime=spacetime_index,  # 现有时空索引
        enable_3d=True
    )
    
    # 添加带空间坐标的记忆
    sr.add_spatial_memory(
        memory_id="mem_001",
        content="在会议室A发生的事件",
        position=(10.0, 20.0, 0.0),  # x, y, z 坐标
        timestamp=1704067200
    )
    
    # 空间检索：查找位置附近的记忆
    results = sr.search_nearby(position=(10.0, 20.0, 0.0), radius=5.0)
    
    # 三维检索：空间+时间+语义
    results = sr.search_3d(
        query="会议",
        position=(10.0, 20.0, 0.0),
        time_range=(start_ts, end_ts),
        max_distance=10.0
    )
"""

import math
import time
from typing import Dict, List, Tuple, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


# ============================================================
# 数据结构
# ============================================================

@dataclass
class SpatialNode:
    """带空间坐标的记忆节点"""
    memory_id: str
    content: str
    position: Tuple[float, float, float]  # (x, y, z) 或 (lat, lon, alt)
    timestamp: int = 0
    energy_type: str = "土"
    semantic_vector: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpatialSearchResult:
    """空间检索结果"""
    memory_id: str
    content: str
    score: float
    distance: float  # 距离
    position: Tuple[float, float, float]
    timestamp: int
    source: str = "spatial"  # "spatial", "temporal", "semantic", "3d"


@dataclass
class TrajectoryPoint:
    """轨迹点"""
    memory_id: str
    position: Tuple[float, float, float]
    timestamp: int
    content: str = ""


# ============================================================
# KD-Tree 空间索引实现
# ============================================================

class KDTreeNode:
    """KD-Tree 节点"""
    
    def __init__(self, point: Tuple[float, float, float], memory_id: str, depth: int = 0):
        self.point = point  # (x, y, z)
        self.memory_id = memory_id
        self.left: Optional[KDTreeNode] = None
        self.right: Optional[KDTreeNode] = None
        self.depth = depth


class KDTree:
    """
    简化版 KD-Tree 实现
    
    用于空间最近邻搜索
    """
    
    def __init__(self, dim: int = 3):
        self.root: Optional[KDTreeNode] = None
        self.dim = dim
        self.n_nodes = 0
    
    def insert(self, point: Tuple[float, float, float], memory_id: str):
        """插入节点"""
        new_node = KDTreeNode(point, memory_id)
        
        if self.root is None:
            self.root = new_node
            self.n_nodes += 1
            return
        
        node = self.root
        depth = 0
        
        while True:
            axis = depth % self.dim
            
            if point[axis] < node.point[axis]:
                if node.left is None:
                    node.left = new_node
                    self.n_nodes += 1
                    return
                node = node.left
            else:
                if node.right is None:
                    node.right = new_node
                    self.n_nodes += 1
                    return
                node = node.right
            
            depth += 1
    
    def _distance(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """计算欧氏距离"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    
    def _search_nearby(
        self,
        node: Optional[KDTreeNode],
        target: Tuple[float, float, float],
        radius: float,
        results: List[Tuple[float, str, float]],  # (distance, memory_id, point)
        max_results: int = 100
    ):
        """递归搜索附近节点"""
        if node is None or len(results) >= max_results:
            return
        
        # 计算距离
        dist = self._distance(node.point, target)
        
        if dist <= radius:
            results.append((dist, node.memory_id, node.point))
        
        # 计算分割维度
        axis = node.depth % self.dim
        
        # 确定搜索顺序
        if target[axis] < node.point[axis]:
            first, second = node.left, node.right
        else:
            first, second = node.right, node.left
        
        # 优先搜索更可能的方向
        if first is not None:
            self._search_nearby(first, target, radius, results, max_results)
        
        # 检查是否需要搜索另一侧
        if second is not None:
            # 计算到分割超平面的距离
            diff = abs(target[axis] - node.point[axis])
            if diff <= radius:
                self._search_nearby(second, target, radius, results, max_results)
    
    def search_nearby(
        self,
        target: Tuple[float, float, float],
        radius: float,
        max_results: int = 100
    ) -> List[Tuple[float, str, Tuple[float, float, float]]]:
        """搜索附近节点"""
        results: List[Tuple[float, str, Tuple[float, float, float]]] = []
        self._search_nearby(self.root, target, radius, results, max_results)
        
        # 按距离排序
        results.sort(key=lambda x: x[0])
        
        return results
    
    def search_k_nearest(
        self,
        target: Tuple[float, float, float],
        k: int = 5
    ) -> List[Tuple[float, str, Tuple[float, float, float]]]:
        """搜索 K 近邻"""
        # 使用半径搜索，从大到小缩小
        radius = 1.0
        results = []
        
        while len(results) < k and radius < 10000:
            results = self.search_nearby(target, radius, k)
            radius *= 2
        
        return results[:k]


# ============================================================
# 轨迹追踪器
# ============================================================

class TrajectoryTracker:
    """
    轨迹追踪器
    
    追踪实体在空间中的移动轨迹
    """
    
    def __init__(self, entity_id: str):
        self.entity_id = entity_id
        self.points: List[TrajectoryPoint] = []
    
    def add_point(
        self,
        memory_id: str,
        position: Tuple[float, float, float],
        timestamp: int,
        content: str = ""
    ):
        """添加轨迹点"""
        self.points.append(TrajectoryPoint(
            memory_id=memory_id,
            position=position,
            timestamp=timestamp,
            content=content
        ))
    
    def get_trajectory(
        self,
        start_time: int = None,
        end_time: int = None
    ) -> List[TrajectoryPoint]:
        """获取轨迹"""
        if start_time is None and end_time is None:
            return self.points.copy()
        
        result = []
        for p in self.points:
            if start_time is not None and p.timestamp < start_time:
                continue
            if end_time is not None and p.timestamp > end_time:
                continue
            result.append(p)
        
        return result
    
    def get_total_distance(self) -> float:
        """计算轨迹总长度"""
        if len(self.points) < 2:
            return 0.0
        
        total = 0.0
        for i in range(1, len(self.points)):
            p1 = self.points[i - 1].position
            p2 = self.points[i].position
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
            total += dist
        
        return total
    
    def get_average_speed(self) -> float:
        """计算平均速度（单位/秒）"""
        if len(self.points) < 2:
            return 0.0
        
        total_dist = self.get_total_distance()
        time_span = self.points[-1].timestamp - self.points[0].timestamp
        
        if time_span == 0:
            return 0.0
        
        return total_dist / time_span


# ============================================================
# SpatialRAG 三维世界模型
# ============================================================

class SpatialRAG:
    """
    SpatialRAG 三维世界模型
    
    融合空间、时间、语义三个维度的检索能力
    
    使用方式：
        sr = SpatialRAG(
            embedding_func=encode_func,
            spacetime=spacetime_index,
            dim=3
        )
        
        # 添加空间记忆
        sr.add_spatial_memory(memory_id, content, position, timestamp)
        
        # 三维检索
        results = sr.search_3d(query, position, time_range, max_distance)
    """
    
    def __init__(
        self,
        embedding_func: Callable[[str], List[float]] = None,
        spacetime=None,  # SpacetimeIndex 实例
        dim: int = 3,  # 维度 2D 或 3D
        enable_trajectory: bool = True
    ):
        self.embedding_func = embedding_func
        self.spacetime = spacetime
        self.dim = dim
        
        # 空间索引
        self._spatial_index = KDTree(dim=dim)
        
        # 记忆存储
        self._spatial_nodes: Dict[str, SpatialNode] = {}
        
        # 轨迹追踪
        self._trajectories: Dict[str, TrajectoryTracker] = {}
        self.enable_trajectory = enable_trajectory
        
        # 空间权重参数
        self._spatial_weight = 0.3
        self._temporal_weight = 0.3
        self._semantic_weight = 0.4
        
        print(f"[SpatialRAG] 三维世界模型已初始化 (dim={dim})")
        print(f"  - 空间索引: KD-Tree")
        print(f"  - 轨迹追踪: {'启用' if enable_trajectory else '禁用'}")
    
    @property
    def n_nodes(self) -> int:
        return len(self._spatial_nodes)
    
    def add_spatial_memory(
        self,
        memory_id: str,
        content: str,
        position: Tuple[float, float, float],
        timestamp: int = None,
        energy_type: str = "土",
        semantic_vector: List[float] = None,
        entity_id: str = None,
        metadata: Dict = None
    ) -> bool:
        """
        添加带空间坐标的记忆
        
        Args:
            memory_id: 记忆ID
            content: 内容
            position: 空间坐标 (x, y, z)
            timestamp: 时间戳
            energy_type: 五行类型
            semantic_vector: 语义向量
            entity_id: 实体ID（用于轨迹追踪）
            metadata: 元数据
        
        Returns:
            是否成功
        """
        ts = timestamp or int(time.time())
        
        # 如果没有提供语义向量，使用 embedding_func
        if semantic_vector is None and self.embedding_func:
            semantic_vector = self.embedding_func(content)
        
        # 创建空间节点
        node = SpatialNode(
            memory_id=memory_id,
            content=content,
            position=position,
            timestamp=ts,
            energy_type=energy_type,
            semantic_vector=semantic_vector,
            metadata=metadata or {}
        )
        
        self._spatial_nodes[memory_id] = node
        
        # 插入空间索引
        self._spatial_index.insert(position, memory_id)
        
        # 更新轨迹
        if self.enable_trajectory and entity_id:
            if entity_id not in self._trajectories:
                self._trajectories[entity_id] = TrajectoryTracker(entity_id)
            
            self._trajectories[entity_id].add_point(
                memory_id=memory_id,
                position=position,
                timestamp=ts,
                content=content
            )
        
        return True
    
    def search_nearby(
        self,
        position: Tuple[float, float, float],
        radius: float,
        max_results: int = 10
    ) -> List[SpatialSearchResult]:
        """
        空间邻域搜索
        
        Args:
            position: 查询位置
            radius: 搜索半径
            max_results: 最大结果数
        
        Returns:
            附近记忆列表
        """
        # 使用 KD-Tree 搜索
        nearby = self._spatial_index.search_nearby(position, radius, max_results)
        
        results = []
        for dist, memory_id, pos in nearby:
            if memory_id in self._spatial_nodes:
                node = self._spatial_nodes[memory_id]
                results.append(SpatialSearchResult(
                    memory_id=memory_id,
                    content=node.content,
                    score=1.0 / (1.0 + dist),  # 距离转相似度
                    distance=dist,
                    position=node.position,
                    timestamp=node.timestamp,
                    source="spatial"
                ))
        
        return results
    
    def search_3d(
        self,
        query: str,
        position: Tuple[float, float, float],
        time_range: Tuple[int, int] = None,
        max_distance: float = 10.0,
        max_results: int = 10
    ) -> List[SpatialSearchResult]:
        """
        三维检索（空间+时间+语义）
        
        Args:
            query: 查询文本
            position: 查询位置
            time_range: 时间范围 (start_ts, end_ts)
            max_distance: 最大空间距离
            max_results: 最大结果数
        
        Returns:
            三维检索结果
        """
        results: Dict[str, SpatialSearchResult] = {}
        
        # 1. 空间邻域搜索
        spatial_results = self.search_nearby(position, max_distance, max_results * 2)
        for r in spatial_results:
            results[r.memory_id] = r
        
        # 2. 时间范围过滤
        if time_range:
            filtered = {}
            for memory_id, r in results.items():
                if time_range[0] <= r.timestamp <= time_range[1]:
                    filtered[memory_id] = r
            results = filtered
        
        # 3. 语义增强
        if self.embedding_func:
            query_vec = self.embedding_func(query)
            if query_vec:
                for memory_id, node in self._spatial_nodes.items():
                    if memory_id in results:
                        r = results[memory_id]
                        
                        # 计算语义相似度
                        if node.semantic_vector:
                            semantic_sim = self._cosine_similarity(query_vec, node.semantic_vector)
                            
                            # 综合得分
                            spatial_score = 1.0 / (1.0 + r.distance)
                            final_score = (
                                self._spatial_weight * spatial_score +
                                self._semantic_weight * semantic_sim
                            )
                            
                            r.score = final_score
        
        # 排序并返回
        sorted_results = sorted(results.values(), key=lambda x: x.score, reverse=True)
        
        for r in sorted_results:
            r.source = "3d"
        
        return sorted_results[:max_results]
    
    def get_trajectory(self, entity_id: str) -> Optional[TrajectoryTracker]:
        """获取实体轨迹"""
        return self._trajectories.get(entity_id)
    
    def search_path(
        self,
        start_pos: Tuple[float, float, float],
        end_pos: Tuple[float, float, float],
        max_distance: float = 5.0
    ) -> List[SpatialSearchResult]:
        """
        路径搜索：查找从起点到终点路径上的记忆
        
        Args:
            start_pos: 起点
            end_pos: 终点
            max_distance: 路径搜索半径
        
        Returns:
            路径上的记忆
        """
        # 线性插值获取中间点
        n_steps = 10
        path_points = []
        
        for i in range(n_steps + 1):
            t = i / n_steps
            point = (
                start_pos[0] + (end_pos[0] - start_pos[0]) * t,
                start_pos[1] + (end_pos[1] - start_pos[1]) * t,
                start_pos[2] + (end_pos[2] - start_pos[2]) * t
            )
            path_points.append(point)
        
        # 收集路径附近的记忆
        results: Dict[str, SpatialSearchResult] = {}
        
        for point in path_points:
            nearby = self.search_nearby(point, max_distance, 5)
            for r in nearby:
                if r.memory_id not in results or r.distance < results[r.memory_id].distance:
                    results[r.memory_id] = r
        
        # 按沿路径位置排序
        sorted_results = sorted(
            results.values(),
            key=lambda x: self._distance_to_path(x.position, start_pos, end_pos)
        )
        
        return sorted_results
    
    def _distance_to_path(
        self,
        point: Tuple[float, float, float],
        start: Tuple[float, float, float],
        end: Tuple[float, float, float]
    ) -> float:
        """计算点到线段的距离"""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dz = end[2] - start[2]
        
        length_sq = dx*dx + dy*dy + dz*dz
        
        if length_sq == 0:
            return self._distance(point, start)
        
        t = max(0, min(1, (
            (point[0] - start[0]) * dx +
            (point[1] - start[1]) * dy +
            (point[2] - start[2]) * dz
        ) / length_sq))
        
        projection = (
            start[0] + t * dx,
            start[1] + t * dy,
            start[2] + t * dz
        )
        
        return self._distance(point, projection)
    
    def _distance(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """计算欧氏距离"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if not vec1 or not vec2:
            return 0.0
        
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        return dot / (norm1 * norm2 + 1e-8)
    
    def get_spatial_context(self, position: Tuple[float, float, float], radius: float = 10.0) -> Dict[str, Any]:
        """获取空间上下文"""
        nearby = self.search_nearby(position, radius, 20)
        
        return {
            "position": position,
            "radius": radius,
            "n_nearby": len(nearby),
            "nodes": [
                {
                    "memory_id": r.memory_id,
                    "content": r.content[:50] + "..." if len(r.content) > 50 else r.content,
                    "distance": r.distance,
                    "timestamp": r.timestamp
                }
                for r in nearby[:5]
            ]
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "n_spatial_nodes": self.n_nodes,
            "n_trajectories": len(self._trajectories),
            "dim": self.dim,
            "weights": {
                "spatial": self._spatial_weight,
                "temporal": self._temporal_weight,
                "semantic": self._semantic_weight
            }
        }


# ============================================================
# 工厂函数
# ============================================================

def create_spatial_rag(
    embedding_func: Callable[[str], List[float]] = None,
    spacetime=None,
    dim: int = 3,
    enable_trajectory: bool = True
) -> SpatialRAG:
    """
    创建 SpatialRAG 实例
    
    Args:
        embedding_func: 文本嵌入函数
        spacetime: SpacetimeIndex 实例（可选）
        dim: 维度 (2 或 3)
        enable_trajectory: 是否启用轨迹追踪
    
    Returns:
        SpatialRAG 实例
    """
    return SpatialRAG(
        embedding_func=embedding_func,
        spacetime=spacetime,
        dim=dim,
        enable_trajectory=enable_trajectory
    )