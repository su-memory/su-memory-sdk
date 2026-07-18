"""
_memory_graph — 记忆节点与因果图谱（lite_pro.py 拆分）

本模块承载记忆的核心数据结构：
- Edge: 记忆间关联边，带"成色"（置信度 + 证据类型）
- MemoryNode: 记忆节点（含双时间模型、版本化、来源溯源）
- MemoryGraph: 因果图谱，支持多跳推理与路径置信度剪枝

从 lite_pro.py 拆分而来，对外 API 通过 lite_pro.py 再导出保持向后兼容。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Edge:
    """记忆间的关联边, 带"成色"(置信度 + 证据类型)。

    evidence_type 四档成色 (第一性原理: 边的可信度必须可区分):
    - "explicit":  用户显式声明 parent_ids=...     confidence=0.95
    - "causal":    数据验证的因果 (委托 mci 发现)   confidence=0.80
    - "semantic":  embedding 相似度 + 共现实体      confidence=0.50
    - "heuristic": 关键词/时序邻近 (启发式猜测)     confidence=0.20
    """
    source: str
    target: str
    confidence: float = 0.20
    evidence_type: str = "heuristic"
    causal_type: str = "sequence"  # legacy: condition/cause/result/sequence

    # 成色 → 默认置信度映射
    _TIER_CONFIDENCE = {
        "explicit": 0.95,
        "causal": 0.80,
        "semantic": 0.50,
        "heuristic": 0.20,
    }


@dataclass
class MemoryNode:
    """记忆节点"""
    id: str
    content: str
    metadata: dict[str, Any]
    embedding: list[float] | None = None
    keywords: list[str] = field(default_factory=list)
    timestamp: int = 0
    parent_ids: list[str] = field(default_factory=list)
    child_ids: list[str] = field(default_factory=list)
    energy_type: str = "earth"  # Default energy type
    # 带成色的关联边 (新增; parent_ids/child_ids 保留向后兼容)
    edges: dict[str, Edge] = field(default_factory=dict)  # target_id -> Edge
    # 访问追踪 (频率加权衰减的地基)
    access_count: int = 0
    last_accessed: int = 0  # timestamp of last query hit
    # 来源溯源 provenance (C5: 医疗级合规审计要求)
    source_type: str = "unknown"        # order|lab_report|patient|ai_inferred|imported
    source_id: str = ""                 # 原始记录 ID（病历号/对话ID/FHIR Resource ID）
    source_confidence: float = 1.0      # 来源可信度（医嘱1.0/患者自述0.6/AI推断0.4）
    # 双时间模型 (C4: 区分事件发生时间 vs 记录入库时间)
    event_time: int = 0                 # 事件实际发生时间（缺省=timestamp 入库时间）
    # 版本化 (C6: 同一事实多次更新的版本链)
    version: int = 1                    # 版本号（从 1 起）
    prev_version_id: str = ""           # 上一版本 memory_id（版本链前驱）
    superseded_by: str = ""             # 被哪个新版本取代（空=当前活跃版本）

    @property
    def effective_time(self) -> int:
        """有效时间：优先事件时间，缺省回退入库时间（C4 双时间模型）。

        V9: 用 > 0 而非 truthy 判断——0 在医疗场景无意义（1970），
        与"未设置"合并是可接受语义；负数/未来时间由 add 层拦截。
        """
        return self.event_time if self.event_time > 0 else self.timestamp


class MemoryGraph:
    """
    记忆因果图谱
    支持多跳推理和因果关系推断
    """

    # 因果关系关键词
    CAUSAL_KEYWORDS = {
        # 因果连接词
        "cause": ["如果", "因为", "由于", "既然", "由于", "因"],
        "effect": ["所以", "因此", "导致", "使得", "就会", "结果", "于是", "那么", "就", "便"],
        "condition": ["当", "只要", "除非", "假如", "倘若", "要是"],
        "result": ["就会", "便会", "便会", "就会", "就会", "则", "必然", "一定"]
    }

    def __init__(self):
        self._nodes: dict[str, MemoryNode] = {}
        self._adjacency: dict[str, set[str]] = defaultdict(set)  # parent -> children
        # (parent, child) -> Edge (含 confidence + evidence_type + causal_type)
        self._causal_edges: dict[tuple[str, str], Edge] = {}

    def detect_causal_type(self, parent_content: str, child_content: str) -> str | None:
        """
        检测两个记忆之间的因果关系类型

        Args:
            parent_content: 父记忆内容
            child_content: 子记忆内容

        Returns:
            因果关系类型: "condition", "cause", "sequence", 或 None
        """
        # 检查条件关系
        for kw in self.CAUSAL_KEYWORDS["condition"]:
            if kw in parent_content:
                return "condition"

        # 检查因果关系
        for kw in self.CAUSAL_KEYWORDS["cause"]:
            if kw in parent_content:
                for effect_kw in self.CAUSAL_KEYWORDS["effect"]:
                    if effect_kw in child_content:
                        return "cause"


        # 检查结果关系
        for kw in self.CAUSAL_KEYWORDS["result"]:
            if kw in child_content:
                return "result"

        # 默认时序关系
        return "sequence"

    def infer_causal_links(self, node: MemoryNode) -> list[tuple[str, float]]:
        """根据内容推断可能的关联 (heuristic 成色, confidence=0.20)。

        ⚠️ 这是启发式推断——仅基于中文连接词的字面共现, 不可作为因果依据。
        检索时会被自动降权。真正的因果关联请用显式声明或委托 mci 发现算法。

        Returns:
            list of (node_id, confidence) — 已按相似度排序, 限制前 3 条。
        """
        inferred: list[tuple[str, float]] = []
        content = node.content

        # 双向配对: 当前节点含 cause/condition 词 → 找含 effect 词的旧节点;
        #           当前节点含 effect 词 → 找含 cause/condition 词的旧节点
        has_cause_kw = any(kw in content for kw in self.CAUSAL_KEYWORDS["cause"])
        has_cond_kw = any(kw in content for kw in self.CAUSAL_KEYWORDS["condition"])
        has_effect_kw = any(kw in content for kw in self.CAUSAL_KEYWORDS["effect"])
        if not (has_cause_kw or has_cond_kw or has_effect_kw):
            return inferred

        for nid, n in self._nodes.items():
            if nid == node.id:
                continue
            n_content = n.content
            n_has_cause = any(kw in n_content for kw in self.CAUSAL_KEYWORDS["cause"])
            n_has_cond = any(kw in n_content for kw in self.CAUSAL_KEYWORDS["condition"])
            n_has_effect = any(kw in n_content for kw in self.CAUSAL_KEYWORDS["effect"])
            # cause/condition ←→ effect 的配对 (任一方向)
            paired = ((has_cause_kw or has_cond_kw) and n_has_effect) or \
                     (has_effect_kw and (n_has_cause or n_has_cond))
            if paired:
                inferred.append((nid, 0.20))

        # 排序 + 限 3 条 (避免噪声边泛滥)
        inferred.sort(key=lambda x: x[1], reverse=True)
        return inferred[:3]

    def add_node(self, node: MemoryNode):
        """添加节点"""
        self._nodes[node.id] = node

        # 构建邻接表
        for parent_id in node.parent_ids:
            self._adjacency[parent_id].add(node.id)

        # 自动推断关联 (heuristic 成色, 自动降权)
        inferred = self.infer_causal_links(node)
        for parent_id, conf in inferred:
            if parent_id not in node.parent_ids:
                causal_type = self.detect_causal_type(
                    self._nodes[parent_id].content,
                    node.content
                )
                self.add_edge(
                    parent_id, node.id, causal_type,
                    confidence=conf, evidence_type="heuristic",
                )

    def add_edge(
        self,
        parent_id: str,
        child_id: str,
        causal_type: str | None = None,
        confidence: float | None = None,
        evidence_type: str = "heuristic",
    ):
        """添加带成色的关联边。

        Parameters
        ----------
        confidence : float, optional
            边置信度 [0,1]。None 时按 evidence_type 取默认值。
        evidence_type : str
            "explicit" / "causal" / "semantic" / "heuristic"。
        """
        # 确定置信度
        if confidence is None:
            confidence = Edge._TIER_CONFIDENCE.get(evidence_type, 0.20)

        # 防重复: 若边已存在, 保留更高成色的那条
        key = (parent_id, child_id)
        if key in self._causal_edges:
            existing = self._causal_edges[key]
            if existing.confidence >= confidence:
                return  # 已有更高成色的边, 不降级

        edge = Edge(
            source=parent_id, target=child_id,
            confidence=confidence, evidence_type=evidence_type,
            causal_type=causal_type or "sequence",
        )

        # 更新邻接表 (去重)
        if parent_id in self._nodes and child_id not in self._nodes[parent_id].child_ids:
            self._nodes[parent_id].child_ids.append(child_id)
        if child_id in self._nodes and parent_id not in self._nodes[child_id].parent_ids:
            self._nodes[child_id].parent_ids.append(parent_id)
        self._adjacency[parent_id].add(child_id)

        # 存储 Edge 对象 (同时挂在 target 节点上便于回溯)
        self._causal_edges[key] = edge
        if child_id in self._nodes:
            self._nodes[child_id].edges[parent_id] = edge

    def get_parents(self, node_id: str) -> list[str]:
        """获取父节点"""
        return self._nodes.get(node_id, MemoryNode("", "", {})).parent_ids

    def get_children(self, node_id: str) -> list[str]:
        """获取子节点"""
        return list(self._adjacency.get(node_id, set()))

    def get_causal_type(self, parent_id: str, child_id: str) -> str:
        """获取因果类型"""
        edge = self._causal_edges.get((parent_id, child_id))
        return edge.causal_type if edge else "sequence"

    def get_edge_confidence(self, parent_id: str, child_id: str) -> float:
        """获取边置信度 (无边时返回 0)。"""
        edge = self._causal_edges.get((parent_id, child_id))
        return edge.confidence if edge else 0.0

    def get_edge(self, parent_id: str, child_id: str) -> Edge | None:
        """获取完整 Edge 对象。"""
        return self._causal_edges.get((parent_id, child_id))

    def bfs_hops(
        self,
        start_ids: list[str],
        max_hops: int = 3,
        causal_only: bool = False,
        min_path_confidence: float = 0.1,
    ) -> list[tuple[str, int, list[str], str, float]]:
        """BFS 多跳遍历, 带路径置信度剪枝。

        路径置信度 = ∏(沿途各边的 confidence)。低于 ``min_path_confidence``
        的路径被剪枝——这会让 heuristic(0.20) 的边在 2 跳后自动淘汰
        (0.20² = 0.04 < 0.1), 而 explicit(0.95) 的边 3 跳仍存活
        (0.95³ = 0.857)。

        Returns:
            List of (node_id, hop_count, path, last_causal, path_confidence)
        """
        results = []
        visited = {}  # node_id -> best path_confidence seen
        # (node_id, hops, path, last_causal, path_confidence)
        queue = [(sid, 0, [sid], "start", 1.0) for sid in start_ids]

        while queue:
            node_id, hops, path, last_causal, path_conf = queue.pop(0)

            if hops > max_hops:
                continue
            # 剪枝: 路径置信度太低
            if path_conf < min_path_confidence and hops > 0:
                continue
            # 去重: 只保留到达同一节点的更高置信度路径
            if node_id in visited and visited[node_id] >= path_conf:
                continue
            visited[node_id] = path_conf

            results.append((node_id, hops, path, last_causal, path_conf))

            # 扩展子节点
            for child_id in self.get_children(node_id):
                edge_conf = self.get_edge_confidence(node_id, child_id)
                causal_type = self.get_causal_type(node_id, child_id)
                new_conf = path_conf * edge_conf
                if new_conf >= min_path_confidence:
                    if not causal_only or causal_type != "sequence":
                        queue.append((child_id, hops + 1, path + [child_id], causal_type, new_conf))

            # 扩展父节点（双向遍历）
            for parent_id in self.get_parents(node_id):
                edge_conf = self.get_edge_confidence(parent_id, node_id)
                causal_type = self.get_causal_type(parent_id, node_id)
                new_conf = path_conf * edge_conf
                if new_conf >= min_path_confidence:
                    if not causal_only or causal_type != "sequence":
                        queue.append((parent_id, hops + 1, [parent_id] + path, causal_type, new_conf))

        return results


