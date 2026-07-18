"""
su-memory SDK 增强版
支持向量检索、多跳推理、时序理解、会话管理
对标 Hindsight LongMemEval 基准

模块结构:
- MemoryNode: 记忆节点数据结构
- MemoryGraph: 因果图谱，支持多跳推理
- TemporalSystem: 时序编码和衰减系统
- PredictionModule: 预测模块
- ExplainabilityModule: 可解释性模块
- SessionManager: 会话管理器
- SuMemoryLitePro: 主客户端类
"""
import json
import logging
import math
import os
import sys
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any

from su_memory.sdk._bridge_recall import EntityBridgeRecaller
from su_memory.sdk._topic_clusterer import TopicClusterer
from su_memory.sdk._memory_protocol import MemoryProtocol

logger = logging.getLogger(__name__)

# 导入embedding模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from su_memory.sdk.embedding import OllamaEmbedding, cosine_similarity, rrf_fusion

# 尝试导入 FAISS
try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError as e:
    FAISS_AVAILABLE = False
    faiss = None
    np = None
    _faiss_import_error = e

# FAISS 安装提示（延迟检测）
def _check_and_suggest_faiss():
    """检查 FAISS 是否可用，如不可用给出安装提示"""
    if not FAISS_AVAILABLE:
        logger.warning(
            "⚠️  提示：FAISS 索引未安装\n"
            "当前状态：使用朴素搜索（线性扫描）\n"
            "安装 FAISS 可获得 O(log n) 搜索性能：\n"
            "  • pip install faiss-cpu        # CPU版本\n"
            "  • pip install faiss-gpu        # GPU加速（需CUDA）\n"
            "安装后请重启 Python 解释器以加载 FAISS"
        )
        return False
    return True

# 尝试导入 VectorGraphRAG
try:
    from su_memory.sdk.vector_graph_rag import VectorGraphRAG, create_vector_graph_rag
    VECTOR_GRAPH_AVAILABLE = True
except ImportError:
    VECTOR_GRAPH_AVAILABLE = False
    VectorGraphRAG = None
    create_vector_graph_rag = None

# 尝试导入 SpacetimeIndex
try:
    from su_memory.sdk.spacetime_index import SpatiotemporalIndex, create_spatiotemporal_index
    SPACETIME_AVAILABLE = True
except ImportError:
    SPACETIME_AVAILABLE = False
    SpatiotemporalIndex = None
    create_spatiotemporal_index = None

# 尝试导入 SpacetimeMultihopEngine
try:
    from su_memory.sdk.spacetime_multihop import (
        SpacetimeMultihopEngine,
        create_spacetime_multihop_engine,
    )
    SPACETIME_MULTIHOP_AVAILABLE = True
except ImportError:
    SPACETIME_MULTIHOP_AVAILABLE = False
    SpacetimeMultihopEngine = None
    create_spacetime_multihop_engine = None

# 尝试导入 MultimodalEmbedding
try:
    from su_memory.sdk.multimodal import MultimodalEmbeddingManager, create_multimodal_manager
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False
    MultimodalEmbeddingManager = None
    create_multimodal_manager = None

# 尝试导入 SpatialRAG
try:
    from su_memory.sdk.spatial_rag import SpatialRAG, create_spatial_rag
    SPATIAL_RAG_AVAILABLE = True
except ImportError:
    SPATIAL_RAG_AVAILABLE = False
    SpatialRAG = None
    create_spatial_rag = None


# 中文停用词表
STOP_WORDS = {
    '的', '了', '和', '是', '在', '有', '我', '你', '他', '她', '它',
    '这', '那', '都', '也', '就', '要', '会', '能', '对', '与', '及',
    '把', '被', '给', '但', '却', '而', '或', '而且', '并且', '所以',
    '因为', '如果', '虽然', '然后', '还是', '可以', '一个', '没有',
    '什么', '怎么', '这个', '那个', '一些', '已经', '非常', '可能',
}

ENGLISH_STOP_WORDS = {
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
    'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been',
    'some', 'than', 'that', 'this', 'with', 'from', 'they', 'will',
    'when', 'what', 'which', 'their', 'about', 'into', 'other',
}


class _STBatchEmbedding:
    """sentence-transformers 包装, 支持原生 batch encode (10× 加速).

    与 OllamaEmbedding 接口兼容 (dims 属性 + encode 方法), 但 encode 多条时
    赞 sentence-transformers 的批量前向传播, 而非逐条请求.
    """
    def __init__(self, st_model, ndim):
        self._model = st_model
        self.dims = ndim

    def encode(self, text):
        arr = self._model.encode(text, convert_to_numpy=True)
        if arr.ndim == 2:
            arr = arr[0]
        return arr.tolist()


@dataclass
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
    edges: dict[str, "Edge"] = field(default_factory=dict)  # target_id -> Edge
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


class TemporalSystem:
    """
    Temporal encoding and time-aware recall system
    Enhanced temporal dimension retrieval capability
    Supports energy strength state calculation and time decay weighting
    """

    # Time stems (pinyin)
    TIME_STEMS = ["jia", "yi", "bing", "ding", "wu", "ji", "geng", "xin", "ren", "gui"]
    # Time branches (pinyin)
    TIME_BRANCHES = ["zi", "chou", "yin", "mao", "chen", "si", "wu", "wei", "shen", "you", "xu", "hai"]

    # Energy types mapped to branches
    BRANCH_ENERGY = {
        "zi": "water", "chou": "earth", "yin": "wood", "mao": "wood",
        "chen": "earth", "si": "fire", "wu": "fire", "wei": "earth",
        "shen": "metal", "you": "metal", "xu": "earth", "hai": "water"
    }

    # Energy enhancement cycle
    ENERGY_ENHANCE = {"wood": "fire", "fire": "earth", "earth": "metal", "metal": "water", "water": "wood"}

    # Energy suppression cycle
    ENERGY_SUPPRESS = {"wood": "earth", "earth": "water", "water": "fire", "fire": "metal", "metal": "wood"}

    # Strength state mapping: strong/thriving/resting/restrained/dormant
    STRENGTH_MAP = {
        "wood": {"strong": "yin mao", "thriving": "hai zi", "resting": "si wu", "restrained": "shen you", "dormant": "chen xu chou wei"},
        "fire": {"strong": "si wu", "thriving": "yin mao", "resting": "shen you", "restrained": "hai zi", "dormant": "chen xu chou wei"},
        "earth": {"strong": "chen xu chou wei", "thriving": "si wu", "resting": "hai zi", "restrained": "yin mao", "dormant": "shen you"},
        "metal": {"strong": "shen you", "thriving": "chen xu chou wei", "resting": "yin mao", "restrained": "si wu", "dormant": "hai zi"},
        "water": {"strong": "hai zi", "thriving": "shen you", "resting": "chen xu chou wei", "restrained": "si wu", "dormant": "yin mao"}
    }

    # Month to dominant energy mapping
    MONTH_ENERGY = {
        1: "water", 2: "wood", 3: "wood", 4: "fire", 5: "fire",
        6: "earth", 7: "earth", 8: "metal", 9: "metal", 10: "water", 11: "water", 12: "wood"
    }

    def get_time_code(self, timestamp: int = None) -> dict[str, Any]:
        """
        Get current time stem and branch encoding

        Args:
            timestamp: Unix timestamp (seconds)

        Returns:
            Dictionary containing time_stem, time_branch, energy_type, year_code
        """
        ts = timestamp or int(time.time())

        # Calculate year (based on Unix timestamp)
        year = 1970 + (ts // 31556926)  # approximate year

        # Jiazi year cycle calculation (60-year cycle)
        jiazi_year = (year - 1984) % 60

        time_stem_idx = jiazi_year % 10
        time_branch_idx = jiazi_year % 12

        time_stem = self.TIME_STEMS[time_stem_idx]
        time_branch = self.TIME_BRANCHES[time_branch_idx]

        return {
            "time_stem": time_stem,
            "time_branch": time_branch,
            "energy_type": self.BRANCH_ENERGY[time_branch],
            "year_code": f"{time_stem}{time_branch}"
        }

    def get_strength_state(self, energy_type: str, month: int = None) -> str:
        """
        Get energy strength state for current time period

        Args:
            energy_type: Energy type (wood/fire/earth/metal/water)
            month: Month (1-12), defaults to current month

        Returns:
            Strength state: strong/thriving/resting/restrained/dormant
        """
        if month is None:
            import datetime
            month = datetime.datetime.now().month

        current_energy = self.MONTH_ENERGY.get(month, "earth")

        if energy_type == current_energy:
            return "strong"  # Same element dominant
        elif self.ENERGY_ENHANCE.get(current_energy) == energy_type:
            return "thriving"  # Enhanced by current
        elif self.ENERGY_SUPPRESS.get(current_energy) == energy_type:
            return "dormant"  # Suppressed by current
        else:
            return "resting"  # Neutral

    def infer_energy_from_content(self, content: str) -> str:
        """
        Infer energy type from memory content

        Args:
            content: Memory content text

        Returns:
            Energy type classification (wood/fire/earth/metal/water)
        """
        energy_keywords = {
            "wood": ["生长", "发展", "树木", "森林", "绿色", "东方", "春季",
                   "肝", "筋", "希望", "创造", "开始", "健康"],
            "fire": ["热情", "炎热", "红色", "南方", "夏季", "心",
                   "血液", "高温", "活力", "能量", "动力", "激情"],
            "earth": ["稳定", "黄色", "中央", "四季", "脾", "消化",
                   "土地", "基础", "踏实", "信任", "稳定", "持续"],
            "metal": ["收敛", "白色", "西方", "秋季", "肺", "呼吸",
                   "金属", "价值", "收获", "总结", "结束", "财"],
            "water": ["流动", "蓝色", "北方", "冬季", "肾", "泌尿",
                   "智慧", "灵活", "变化", "适应", "学习", "思考"]
        }

        scores = dict.fromkeys(energy_keywords, 0)
        for e, kws in energy_keywords.items():
            for kw in kws:
                if kw in content:
                    scores[e] += 1

        return max(scores, key=scores.get) if max(scores.values()) > 0 else "earth"

    def calculate_recency_score(
        self,
        memory_timestamp: int,
        memory_energy_type: str = "earth",
        current_timestamp: int = None
    ) -> float:
        """
        Calculate recency/decay score considering energy strength state

        Args:
            memory_timestamp: Memory creation timestamp
            memory_energy_type: Memory energy type
            current_timestamp: Current timestamp

        Returns:
            Recency score (0-1)
        """
        ts = current_timestamp or int(time.time())
        # V15/V16: 时间合法性防御——负数/零/未来时间归一化
        # 负数或零（黑洞记忆）回退到当前时间（衰减最小，不霸占也不消失）；
        # 未来时间 clamp 到当前（防止 recency>1.0 永久霸占召回）。
        if not memory_timestamp or memory_timestamp < 0:
            memory_timestamp = ts
        elif memory_timestamp > ts:
            memory_timestamp = ts
        days = (ts - memory_timestamp) / 86400

        # Exponential decay（days>=0 保证 decay<=1.0，不再被未来时间抬高）
        decay = math.exp(-0.02 * days)

        # Get current energy state
        time_code = self.get_time_code(ts)
        current_energy = time_code["energy_type"]

        # Energy enhancement/suppression effects
        if self.ENERGY_ENHANCE.get(current_energy) == memory_energy_type:
            # Memory energy enhances current energy - strengthen decay
            decay *= 1.3
        elif self.ENERGY_SUPPRESS.get(current_energy) == memory_energy_type:
            # Memory energy is suppressed by current energy - weaken decay
            decay *= 0.7
        elif memory_energy_type == current_energy:
            # Same energy type - moderate strengthen
            decay *= 1.1

        # Apply strength state modifier
        strength_state = self.get_strength_state(memory_energy_type)
        if strength_state == "strong":
            decay *= 1.2
        elif strength_state == "thriving":
            decay *= 1.1
        elif strength_state == "dormant":
            decay *= 0.8

        # Short-term memory bonus (within 30 days)
        if days < 1:
            decay *= 1.2
        elif days < 7:
            decay *= 1.1
        elif days < 30:
            decay *= 1.0
        else:
            decay *= 0.9

        return max(0.1, min(1.0, decay))

    def get_temporal_context(self, timestamp: int = None) -> dict[str, Any]:
        """
        Get temporal context for a given timestamp

        Args:
            timestamp: Unix timestamp

        Returns:
            Temporal context information
        """
        ts = timestamp or int(time.time())

        import datetime
        dt = datetime.datetime.fromtimestamp(ts)

        time_code = self.get_time_code(ts)

        return {
            "datetime": dt.isoformat(),
            "timestamp": ts,
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "weekday": dt.strftime("%A"),
            "time_code": time_code,
            "strength_state": self.get_strength_state(time_code["energy_type"], dt.month)
        }


class PredictionModule:
    """
    时序预测模块
    基于历史记忆模式预测未来事件和趋势
    
    V1.7.7: 支持贝叶斯置信度增强（通过 enable_bayesian=True 启用）
    """

    def __init__(self, temporal_system: 'TemporalSystem' = None, enable_bayesian: bool = True):
        self._temporal = temporal_system or TemporalSystem()
        self._pattern_cache: dict[str, list[float]] = defaultdict(list)
        self._event_sequences: list[dict] = []

        # 贝叶斯增强
        self._enable_bayesian = enable_bayesian
        self._bayesian_engine = None
        self._prediction_feedback: dict[str, dict] = {}  # {pred_type: {"success": n, "failure": n}}

        if enable_bayesian:
            try:
                from su_memory._sys.bayesian import BayesianEngine
                self._bayesian_engine = BayesianEngine(default_prior_type="weak")
                # 为每种预测类型注册信念
                for pred_type in ["enhancement_prediction", "suppression_warning",
                                  "frequency_prediction", "trend_prediction",
                                  "historical_causal", "energy_enhancement"]:
                    self._bayesian_engine.register_belief(
                        pred_type,
                        content_summary=f"{pred_type} prediction accuracy",
                        prior_belief=0.65,
                        prior_strength=3.0
                    )
            except ImportError:
                self._enable_bayesian = False

    def _get_confidence(self, pred_type: str, fallback: float = 0.7) -> float:
        """获取贝叶斯置信度（若启用），否则回退到固定值"""
        if self._enable_bayesian and self._bayesian_engine:
            belief = self._bayesian_engine.get_belief(pred_type)
            if belief and belief.posterior.effective_sample_size > 3:
                return belief.posterior.mean
        return fallback

    def feedback(self, pred_type: str, was_correct: bool):
        """提供预测反馈，更新贝叶斯先验"""
        if not self._enable_bayesian or not self._bayesian_engine:
            return
        self._bayesian_engine.observe(pred_type, success=was_correct, weight=1.0, source="prediction_feedback")

    def record_event(self, content: str, timestamp: int = None, metadata: dict = None):
        """
        Record event for subsequent prediction

        Args:
            content: Event content
            timestamp: Event timestamp
            metadata: Event metadata
        """
        ts = timestamp or int(time.time())
        energy_type = self._temporal.infer_energy_from_content(content)
        self._event_sequences.append({
            "content": content,
            "timestamp": ts,
            "metadata": metadata or {},
            "energy_type": energy_type
        })

        # Update pattern cache
        self._pattern_cache[energy_type].append(ts)

    def predict_next_events(self, current_context: str, top_k: int = 3) -> list[dict[str, Any]]:
        """
        预测下一个可能的事件

        Args:
            current_context: 当前上下文
            top_k: 返回预测数量

        Returns:
            List of predicted events with confidence
        """
        # 1. Based on energy enhancement prediction
        current_energy = self._temporal.infer_energy_from_content(current_context)
        current_time = int(time.time())

        predictions = []

        enhanced = self._temporal.ENERGY_ENHANCE.get(current_energy, "earth")
        enhanced_events = [e for e in self._event_sequences
                       if e["energy_type"] == enhanced and e["timestamp"] > current_time - 86400 * 30]

        if enhanced_events:
            predictions.append({
                "type": "enhancement_prediction",
                "content": f"{enhanced} related events may occur",
                "confidence": self._get_confidence("enhancement_prediction", 0.75),
                "confidence_source": "bayesian" if self._enable_bayesian else "heuristic",
                "basis": f"Current {current_energy} enhances {enhanced}, historically {enhanced} events are frequent"
            })

        # 2. Based on temporal pattern prediction
        suppressed = self._temporal.ENERGY_SUPPRESS.get(current_energy, "earth")
        predictions.append({
            "type": "suppression_warning",
            "content": f"Pay attention to {suppressed} related matters",
            "confidence": self._get_confidence("suppression_warning", 0.65),
            "confidence_source": "bayesian" if self._enable_bayesian else "heuristic",
            "basis": f"Current {current_energy} may be affected by {suppressed}"
        })

        # 3. Based on historical frequency prediction
        energy_counts = defaultdict(int)
        for e in self._event_sequences[-100:]:  # Recent 100
            if e["timestamp"] > current_time - 86400 * 7:
                energy_counts[e["energy_type"]] += 1

        if energy_counts:
            most_common = max(energy_counts, key=energy_counts.get)
            if most_common != current_energy:
                predictions.append({
                    "type": "frequency_prediction",
                    "content": f"Recent {most_common} type events are high frequency",
                    "confidence": self._get_confidence("frequency_prediction", 0.70),
                    "confidence_source": "bayesian" if self._enable_bayesian else "heuristic",
                    "basis": f"In past 7 days, {most_common} events appeared {energy_counts[most_common]} times"
                })

        # 按置信度排序
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        return predictions[:top_k]

    def predict_temporal_trend(self, metric: str = "activity", days: int = 7) -> dict[str, Any]:
        """
        预测时间趋势

        Args:
            metric: Metric type (activity/recency/energy)
            days: 预测天数

        Returns:
            趋势预测结果
        """
        current_time = int(time.time())

        if metric == "activity":
            # 活动趋势预测
            recent_events = [e for e in self._event_sequences
                           if e["timestamp"] > current_time - 86400 * 7]
            prev_events = [e for e in self._event_sequences
                          if current_time - 86400 * 14 < e["timestamp"] <= current_time - 86400 * 7]

            recent_count = len(recent_events)
            prev_count = len(prev_events)

            if prev_count > 0:
                change_rate = (recent_count - prev_count) / prev_count
            else:
                change_rate = 0.0

            # 预测趋势
            if change_rate > 0.2:
                trend = "上升"
                confidence = self._get_confidence("trend_prediction", min(0.9, 0.6 + abs(change_rate)))
            elif change_rate < -0.2:
                trend = "下降"
                confidence = self._get_confidence("trend_prediction", min(0.9, 0.6 + abs(change_rate)))
            else:
                trend = "平稳"
                confidence = self._get_confidence("trend_prediction", 0.75)

            return {
                "metric": "activity",
                "trend": trend,
                "change_rate": change_rate,
                "confidence": confidence,
                "recent_count": recent_count,
                "prev_count": prev_count,
                "prediction": f"未来{days}天活动量预计{trend}"
            }

        elif metric == "energy_type":
            # Energy trend prediction
            energy_distribution = defaultdict(int)
            for e in self._event_sequences[-50:]:
                if e["timestamp"] > current_time - 86400 * 14:
                    energy_distribution[e["energy_type"]] += 1

            # Current time period energy
            current_time_code = self._temporal.get_time_code(current_time)
            current_energy = current_time_code["energy_type"]

            return {
                "metric": "energy_type",
                "current_energy": current_energy,
                "distribution": dict(energy_distribution),
                "prediction": f"Current energy {current_energy}, recommend focusing on {self._temporal.ENERGY_ENHANCE.get(current_energy)} related"
            }

        return {"error": "Unknown metric"}

    def get_causal_predictions(self, cause_content: str) -> list[dict[str, Any]]:
        """
        基于因果关系预测结果

        Args:
            cause_content: 原因内容

        Returns:
            可能的结果列表
        """
        cause_energy = self._temporal.infer_energy_from_content(cause_content)

        # Causal keyword detection
        causal_keywords = {
            "如果": ["就", "那么", "则", "会"],
            "因为": ["所以", "因此", "导致", "使得"],
            "当": ["就", "便", "则"],
        }

        results = []

        for cause_kw, effect_kws in causal_keywords.items():
            if cause_kw in cause_content:
                # Find historical causal pairs
                for event in self._event_sequences:
                    for effect_kw in effect_kws:
                        if effect_kw in event["content"]:
                            confidence = self._get_confidence("historical_causal", 0.70)
                            results.append({
                                "cause": cause_content,
                                "effect": event["content"],
                                "confidence": round(confidence, 3),
                                "type": "historical_causal",
                                "confidence_source": "bayesian" if self._enable_bayesian else "heuristic"
                            })

        # Based on energy enhancement prediction
        enhanced = self._temporal.ENERGY_ENHANCE.get(cause_energy)
        if enhanced:
            confidence = self._get_confidence("energy_enhancement", 0.60)
            results.append({
                "cause": cause_content,
                "effect": f"May trigger {enhanced} related events",
                "confidence": round(confidence, 3),
                "type": "energy_enhancement",
                "basis": f"{cause_energy} enhances {enhanced}",
                "confidence_source": "bayesian" if self._enable_bayesian else "heuristic"
            })

        return results[:3]


class ExplainabilityModule:
    """
    可解释性模块
    提供决策路径追溯和因果链可视化
    """

    def __init__(self, memory_graph: 'MemoryGraph' = None):
        self._graph = memory_graph
        self._reasoning_trace: list[dict] = []

    def record_reasoning_step(self, step_type: str, content: str, metadata: dict = None):
        """
        记录推理步骤

        Args:
            step_type: 步骤类型 (perception/recall/reasoning/action)
            content: 步骤内容
            metadata: 额外元数据
        """
        self._reasoning_trace.append({
            "step_type": step_type,
            "content": content,
            "timestamp": int(time.time()),
            "metadata": metadata or {}
        })

    def explain_query(self, query: str, results: list[dict], memory_ids: list[str] = None) -> dict[str, Any]:
        """
        生成查询可解释性报告

        Args:
            query: 查询文本
            results: 查询结果
            memory_ids: 涉及的memory ID列表

        Returns:
            可解释性报告
        """
        report = {
            "query": query,
            "result_count": len(results),
            "reasoning_chain": [],
            "confidence_factors": [],
            "explanation": ""
        }

        # 构建推理链
        for i, result in enumerate(results[:5]):
            chain_item = {
                "rank": i + 1,
                "memory_id": result.get("memory_id"),
                "content_preview": result["content"][:50] + "..." if len(result["content"]) > 50 else result["content"],
                "score": result.get("score", 0),
                "factors": []
            }

            # 分析得分因素
            if result.get("score"):
                chain_item["factors"].append({
                    "factor": "语义相似度",
                    "contribution": f"{result['score']:.2%}"
                })

            if result.get("hops"):
                chain_item["factors"].append({
                    "factor": "多跳推理",
                    "contribution": f"{result['hops']}跳",
                    "path": result.get("path", [])
                })

            if result.get("causal_type"):
                chain_item["factors"].append({
                    "factor": "因果类型",
                    "contribution": result["causal_type"]
                })

            # 时空维度因素
            if result.get("time_decay"):
                chain_item["factors"].append({
                    "factor": "时间衰减",
                    "contribution": f"{result['time_decay']:.2%}"
                })

            if result.get("energy_boost"):
                chain_item["factors"].append({
                    "factor": "能量增强",
                    "contribution": f"{result['energy_boost']:.2f}x",
                    "energy_type": result.get("energy_type", "earth")
                })

            if result.get("energy_type"):
                chain_item["factors"].append({
                    "factor": "Energy System类型",
                    "contribution": result["energy_type"]
                })

            report["reasoning_chain"].append(chain_item)

        # 置信度因素
        if results:
            top_score = results[0].get("score", 0)
            report["confidence_factors"] = [
                {"factor": "语义匹配", "weight": 0.4, "value": f"{top_score:.2%}"},
                {"factor": "因果关联", "weight": 0.3, "value": "基于图谱推理"},
                {"factor": "时序相关性", "weight": 0.2, "value": "时效性已计算"},
                {"factor": "会话上下文", "weight": 0.1, "value": "会话已隔离"}
            ]

        # 生成自然语言解释
        report["explanation"] = self._generate_explanation(query, results, report)

        return report

    def _generate_explanation(self, query: str, results: list[dict], report: dict) -> str:
        """生成自然语言解释"""
        if not results:
            return f"未找到与'{query}'相关的记忆。"

        explanation = f"针对查询'{query}'，系统检索到{len(results)}条相关记忆。\n\n"

        # Top结果解释
        top = results[0]
        explanation += f"最相关记忆：{top['content'][:100]}...\n"
        explanation += f"相关度得分：{top.get('score', 0):.2%}\n\n"

        # 推理路径解释
        if top.get('hops', 0) > 0:
            path = top.get('path', [])
            explanation += f"推理路径：{' → '.join(path[:5])}\n"
            explanation += f"经过{top['hops']}跳推理找到此记忆\n\n"

        # 时空维度解释
        if top.get('time_decay') and top.get('time_decay') < 1.0:
            explanation += f"时间衰减：{top['time_decay']:.2%}（记忆时效性影响）\n"

        if top.get('energy_boost') and top.get('energy_boost') != 1.0:
            energy_type = top.get('energy_type', '土')
            boost = top['energy_boost']
            explanation += f"能量增强：{boost:.2f}x（Energy System类型：{energy_type}）\n"

        # 置信度说明
        explanation += "\n检索因素：\n"
        for factor in report.get("confidence_factors", []):
            explanation += f"  • {factor['factor']}（权重{factor['weight']:.0%}）：{factor['value']}\n"

        return explanation

    def explain_multihop(self, start_memory: str, end_memory: str, path: list[str]) -> dict[str, Any]:
        """
        解释多跳推理路径

        Args:
            start_memory: 起始记忆ID
            end_memory: 结束记忆ID
            path: 推理路径

        Returns:
            多跳解释报告
        """
        if not self._graph:
            return {"error": "MemoryGraph not available"}

        explanation = {
            "path": path,
            "hops": len(path) - 1,
            "edges": [],
            "total_confidence": 1.0
        }

        # 分析每条边
        for i in range(len(path) - 1):
            parent_id, child_id = path[i], path[i + 1]
            causal_type = self._graph.get_causal_type(parent_id, child_id)

            # 获取边上的节点内容
            parent_node = self._graph._nodes.get(parent_id)
            child_node = self._graph._nodes.get(child_id)

            edge_info = {
                "from": parent_id,
                "from_content": parent_node.content[:50] + "..." if parent_node else "",
                "to": child_id,
                "to_content": child_node.content[:50] + "..." if child_node else "",
                "causal_type": causal_type,
                "confidence": self._get_causal_confidence(causal_type)
            }

            explanation["edges"].append(edge_info)
            explanation["total_confidence"] *= edge_info["confidence"]

        # 生成自然语言解释
        explanation["narrative"] = self._generate_path_narrative(explanation)

        return explanation

    def _get_causal_confidence(self, causal_type: str) -> float:
        """获取因果类型置信度"""
        confidence_map = {
            "cause": 0.85,
            "condition": 0.80,
            "result": 0.75,
            "sequence": 0.60,
            "start": 1.0
        }
        return confidence_map.get(causal_type, 0.5)

    def _generate_path_narrative(self, explanation: dict) -> str:
        """生成路径叙事"""
        narrative = f"推理路径共{explanation['hops']}跳\n\n"

        for i, edge in enumerate(explanation["edges"]):
            causal_verb = {
                "cause": "导致",
                "condition": "条件触发",
                "result": "结果产生",
                "sequence": "随后发生"
            }.get(edge["causal_type"], "关联到")

            narrative += f"第{i + 1}跳：{edge['from_content']}\n"
            narrative += f"   {causal_verb} → {edge['to_content']}\n\n"

        narrative += f"综合置信度：{explanation['total_confidence']:.1%}\n"

        return narrative

    def visualize_reasoning_tree(self, query: str, results: list[dict]) -> dict[str, Any]:
        """
        生成推理树可视化数据

        Args:
            query: 查询文本
            results: 结果列表

        Returns:
            树形结构数据（可用于前端渲染）
        """
        tree = {
            "name": query,
            "type": "query",
            "children": []
        }

        for result in results[:5]:
            node = {
                "name": result["content"][:30] + "...",
                "type": "memory",
                "score": result.get("score", 0),
                "metadata": {
                    "memory_id": result.get("memory_id"),
                    "hops": result.get("hops", 0),
                    "causal_type": result.get("causal_type", "unknown")
                }
            }

            # 如果有路径，展开子节点
            if result.get("path") and len(result["path"]) > 1:
                node["children"] = [
                    {
                        "name": f"跳{i}: {pid[:20]}...",
                        "type": "hop",
                        "hop_index": i
                    }
                    for i, pid in enumerate(result["path"][1:], 1)
                ]

            tree["children"].append(node)


        return tree

    def get_reasoning_summary(self) -> dict[str, Any]:
        """
        获取推理过程摘要

        Returns:
            推理摘要统计
        """
        if not self._reasoning_trace:
            return {"total_steps": 0, "message": "暂无推理记录"}

        # 统计各类型步骤
        step_counts = defaultdict(int)
        for step in self._reasoning_trace:
            step_counts[step["step_type"]] += 1

        return {
            "total_steps": len(self._reasoning_trace),
            "step_distribution": dict(step_counts),
            "first_step": self._reasoning_trace[0] if self._reasoning_trace else None,
            "last_step": self._reasoning_trace[-1] if self._reasoning_trace else None
        }


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
        dot = sum(x * y for x, y in zip(a, b))
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


class SuMemoryLitePro(MemoryProtocol):
    """
    su-memory SDK 增强版

    特性:
    - Dense embedding向量检索
    - 记忆因果图谱
    - 多跳推理
    - time_code时序系统
    - 多会话管理
    - 混合检索(RRF融合)
    - 时序预测(PredictionModule)
    - 可解释性(ExplainabilityModule)

    目标: 对标 Hindsight LongMemEval 基准

    ⚠️ 多实例限制：默认 JSON 文件存储（su_memory_pro.json）不支持多进程/多实例
    并发写同一目录（后写者覆盖先写者数据）。多 worker / 多进程部署请使用
    ``storage_backend="postgresql"`` 或 ``"redis"``（支持并发写）。

    ⚠️ 持久化权衡：``save_interval`` 控制落盘频率。默认 10（每 10 次变更落盘一次），
    在进程被 SIGKILL（OOM/容器强杀）时最多丢失 save_interval-1 条（__del__ 不执行）。
    要求数据零丢失的场景设 ``save_interval=1``（每次 add 落盘，有写放大开销）。
    正常退出（SIGTERM/close）会自动 flush，无丢失。
    """


    def __init__(
        self,
        max_memories: int = 5000,
        storage_path: str = None,
        embedding_backend: str = "minimax",
        enable_tfidf: bool = True,
        enable_vector: bool = True,
        enable_graph: bool = True,
        enable_temporal: bool = True,
        enable_session: bool = True,
        enable_prediction: bool = True,
        enable_explainability: bool = True,
        enable_llm_energy: bool = False,
        autosave: bool = True,
        save_interval: int = 10,
        cache_size: int = 128,
        storage_backend: str = "default",
        **embedding_kwargs
    ):
        self.max_memories = max_memories
        self.enable_tfidf = enable_tfidf
        self.enable_vector = enable_vector
        self.enable_graph = enable_graph
        self.enable_temporal = enable_temporal
        self.enable_session = enable_session
        self.enable_prediction = enable_prediction
        self.enable_explainability = enable_explainability
        self.enable_llm_energy = enable_llm_energy or os.environ.get("SU_MEMORY_LLM_ENERGY", "") == "1"
        # 持久化节流：避免每次 add 全量序列化（O(n) 写放大）
        self._autosave = autosave
        self._save_interval = max(1, save_interval)
        self._dirty_count = 0
        self._dirty = False
        self._save_lock = threading.Lock()  # 保护持久化写入，防止并发 save 竞态

        # 自动设置默认存储路径
        if not storage_path:
            storage_path = self._get_default_storage_path()

        self.storage_path = storage_path

        # 核心数据结构
        self._memories: list[MemoryNode] = []
        self._memory_map: dict[str, int] = {}  # id -> index
        self._index: dict[str, set] = defaultdict(set)

        # Embedding - 优先检测 Ollama (V3.16: 延迟检测, 超时2s)
        self._embedding = None
        self._embedding_backend_type = "none"
        self._ollama_checked = False

        if enable_vector:
            # 延迟检测: 仅当真正需要embedding时才连接
            pass  # embedding在_ensure_embedding()中懒加载

        # FAISS 索引 (V3.16: 优先从磁盘加载已持久化的索引)
        self._faiss_index = None
        self._faiss_id_map: dict[int, str] = {}
        self._faiss_lock = threading.RLock()
        self._id_faiss_map: dict[str, int] = {}
        self._faiss_index_path = os.path.join(storage_path, "faiss_hnsw.index") if storage_path else None

        if enable_vector and FAISS_AVAILABLE:
            # 尝试从磁盘加载已有索引
            loaded = self._load_faiss_index()
            if not loaded:
                # 懒加载: 首次 add() 时创建
                pass

        # 记忆图谱
        self._graph = MemoryGraph() if enable_graph else None
        # 软分类: 增量主题分桶 (与 algebra 硬分类正交)
        self._topic_clusterer = TopicClusterer()

        # Vector Graph RAG 多跳推理引擎
        self._vector_graph = None
        if enable_graph and VECTOR_GRAPH_AVAILABLE and self._embedding:
            try:
                def embed_func(text):
                    return self._embedding.encode(text) if self._embedding else None

                # 获取 HNSW 参数（与主索引保持一致）
                hnsw_m = 32
                hnsw_ef = 64
                if self._faiss_index and hasattr(self._faiss_index, 'hnsw'):
                    try:
                        hnsw_m = self._faiss_index.hnsw.m
                        hnsw_ef = self._faiss_index.hnsw.efConstruction
                    except Exception as e:
                        logger.debug("降级处理: %s", e)

                self._vector_graph = VectorGraphRAG(
                    embedding_func=embed_func,
                    dims=getattr(self._embedding, 'dims', 1024),
                    enable_faiss=True,  # 启用 FAISS 索引
                    hnsw_m=hnsw_m,
                    hnsw_ef_construction=hnsw_ef,
                    hnsw_ef_search=hnsw_ef
                )
                logger.info("[SuMemoryLitePro] VectorGraphRAG 多跳推理引擎已初始化 (FAISS enabled)")
            except Exception as e:
                logger.error(f"[SuMemoryLitePro] VectorGraphRAG 初始化失败: {e}")

        # 时序系统
        self._temporal = TemporalSystem() if enable_temporal else None

        # 时空联合索引（融合 TemporalSystem 与 VectorGraphRAG）
        self._spacetime = None
        if enable_temporal and SPACETIME_AVAILABLE and self._embedding:
            try:
                def embed_func_st(text):
                    return self._embedding.encode(text) if self._embedding else None

                self._spacetime = SpatiotemporalIndex(
                    embedding_func=embed_func_st,
                    dims=getattr(self._embedding, 'dims', 1024)
                )
                logger.info("[SuMemoryLitePro] SpacetimeIndex 时空索引已初始化")
            except Exception as e:
                logger.error(f"[SuMemoryLitePro] SpacetimeIndex 初始化失败: {e}")

        # 时空多跳融合引擎（融合 VectorGraphRAG + SpacetimeIndex）
        self._spacetime_engine = None
        if SPACETIME_MULTIHOP_AVAILABLE and self._embedding:
            try:
                # 创建节点映射（memory_id -> node）
                memory_map = {node.id: node for node in self._memories}

                self._spacetime_engine = SpacetimeMultihopEngine(
                    vector_graph=self._vector_graph,
                    spacetime=self._spacetime,
                    memory_nodes=memory_map,
                    embedding_func=self._embedding.encode if self._embedding else None
                )
                logger.info("[SuMemoryLitePro] SpacetimeMultihopEngine 时空多跳融合引擎已初始化")
            except Exception as e:
                logger.error(f"[SuMemoryLitePro] SpacetimeMultihopEngine 初始化失败: {e}")

        # 多模态嵌入管理器（图像+音频支持）
        self._multimodal = None
        if MULTIMODAL_AVAILABLE and self._embedding:
            try:
                self._multimodal = MultimodalEmbeddingManager(
                    text_embedding_func=self._embedding.encode if self._embedding else None,
                    enable_image=False,  # 默认关闭，需要时手动启用
                    enable_audio=False
                )
                logger.info("[SuMemoryLitePro] MultimodalEmbedding 多模态管理器已初始化")
            except Exception as e:
                logger.error(f"[SuMemoryLitePro] MultimodalEmbedding 初始化失败: {e}")

        # SpatialRAG 三维世界模型
        self._spatial = None
        if SPATIAL_RAG_AVAILABLE and self._embedding:
            try:
                self._spatial = SpatialRAG(
                    embedding_func=self._embedding.encode if self._embedding else None,
                    spacetime=self._spacetime,  # 与 SpacetimeIndex 集成
                    dim=3,  # 3D 空间
                    enable_trajectory=False  # 默认关闭轨迹追踪
                )
                logger.info("[SuMemoryLitePro] SpatialRAG 三维世界模型已初始化")
            except Exception as e:
                logger.error(f"[SuMemoryLitePro] SpatialRAG 初始化失败: {e}")

        # 会话管理
        self._sessions = SessionManager(storage_path, self._embedding) if enable_session else None

        # 预测模块
        self._prediction = PredictionModule(self._temporal) if enable_prediction else None

        # 可解释性模块
        self._explainability = ExplainabilityModule(self._graph) if enable_explainability else None

        # Energy re-ranking engine (causal energy affinity scoring)
        try:
            from su_memory._sys._causal_engine import CategoryCausalEngine
            self._causal = CategoryCausalEngine()
        except Exception:
            self._causal = None

        # Unified energy label factory
        try:
            from su_memory._sys._unified_unit import UnifiedInfoFactory
            self._unified_factory = UnifiedInfoFactory()
        except Exception:
            self._unified_factory = None

        # Energy bus: three-layer propagation network
        try:
            from su_memory._sys._energy_bus import EnergyBus
            self._energy_bus = EnergyBus()
            self._energy_bus.create_five_elements_nodes()
        except Exception:
            self._energy_bus = None

        # Energy core: balance analysis & pattern detection
        try:
            from su_memory._sys._energy_core import EnergyCore
            self._energy_core = EnergyCore()
        except Exception:
            self._energy_core = None

        # LRU缓存
        self._cache_size = cache_size
        self._query_cache: OrderedDict[tuple[str, int], list[dict]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

        # FAISS 安装提示
        _check_and_suggest_faiss()

        # v3.0.0: 可选分布式存储后端
        self._storage_backend = None
        self._storage_backend_type = storage_backend
        if storage_backend != "default":
            self._init_storage_backend(storage_backend)

        # 持久化
        if storage_path:
            os.makedirs(storage_path, exist_ok=True)
            self._load()

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text (Chinese + English)."""
        import re
        text_lower = text.lower()

        keywords = set()

        # Extract English words (3+ chars, filter stop words)
        english_words = re.findall(r'[a-z]{3,}', text_lower)
        for word in english_words:
            if word not in ENGLISH_STOP_WORDS:
                keywords.add(word)

        # Extract Chinese bigrams/trigrams
        chinese = re.sub(r'[a-zA-Z0-9]', '', text_lower)
        chinese = re.sub(r'[^\u4e00-\u9fa5]', '', chinese)
        for length in [2, 3, 4]:
            for i in range(len(chinese) - length + 1):
                word = chinese[i:i+length]
                if word and word not in STOP_WORDS:
                    keywords.add(word)

        return list(keywords)

    # ═══════════════════ v3.0.0 分布式存储后端 ═══════════════════

    def _init_storage_backend(self, backend_type: str) -> None:
        """
        初始化分布式存储后端 (委托共享模块)。

        Args:
            backend_type: 后端类型 ("sqlite" / "postgresql" / "redis" / "auto")
        """
        from su_memory.sdk._storage_helpers import init_storage_backend
        init_storage_backend(self, backend_type, self.storage_path, label="SuMemoryLitePro")

    def get_storage_backend(self):
        """
        获取当前存储后端实例。

        Returns:
            StorageBackend 或 None（使用默认 JSON 持久化时）
        """
        return self._storage_backend

    @property
    def storage_backend_type(self) -> str:
        """当前存储后端类型"""
        return self._storage_backend_type

    # ═══════════════════ V3.16 懒加载方法 ═══════════════════

    def _ensure_embedding(self):
        """懒加载: 首次需要embedding时才初始化向量服务（永不返回None）.

        优先级 (性能优先):
        1. sentence-transformers + 本地 BAAI/bge-m3 — 原生 batch encode,
           比 Ollama 逐条请求快约 10×, 是生产检索路径首选.
        2. Ollama bge-m3 (HTTP) — 离线 fallback, 逐条 encode 较慢.
        3. sentence-transformers MiniLM — 最后兜底.
        """
        if self._embedding is not None:
            return self._embedding

        # 1. 优先: sentence-transformers + 本地 bge-m3 (原生 batch, 最快)
        try:
            import sentence_transformers

            from su_memory._sys.encoders import _resolve_local_bge_m3
            local_bge = _resolve_local_bge_m3()
            if local_bge:
                model = sentence_transformers.SentenceTransformer(local_bge)
                dims = model.get_sentence_embedding_dimension()
                self._embedding = _STBatchEmbedding(model, dims)
                self._embedding_backend_type = "sentence-transformers-bge-m3"
                logger.debug(f"[SuMemoryLitePro] sentence-transformers bge-m3 就绪 (dim={dims}, 原生 batch)")
                return self._embedding
        except Exception as e:
            logger.error(f"[SuMemoryLitePro] sentence-transformers bge-m3 加载失败: {e}")

        # 2. Fallback: Ollama（本地离线）
        if not self._ollama_checked:
            ollama_available = self._check_ollama()
            self._ollama_checked = True
            if ollama_available:
                try:
                    self._embedding = OllamaEmbedding()
                    self._embedding_backend_type = "ollama"
                    logger.info("[SuMemoryLitePro] 使用 Ollama 向量服务")
                    return self._embedding
                except Exception as e:
                    logger.error(f"[SuMemoryLitePro] Ollama 初始化失败: {e}")

        # 3. 最后兜底: sentence-transformers MiniLM
        try:
            import sentence_transformers
            model_name = os.environ.get(
                "SU_MEMORY_EMBEDDING_MODEL",
                "paraphrase-multilingual-MiniLM-L12-v2"
            )
            logger.debug(f"[SuMemoryLitePro] 加载 sentence-transformers 模型: {model_name}")
            model = sentence_transformers.SentenceTransformer(model_name)
            dims = model.get_sentence_embedding_dimension()
            self._embedding = _STBatchEmbedding(model, dims)
            self._embedding_backend_type = "sentence-transformers"
            logger.debug(f"[SuMemoryLitePro] sentence-transformers 就绪 (dim={dims})")
            return self._embedding
        except Exception as e:
            logger.warning(f"[SuMemoryLitePro] sentence-transformers 不可用: {e}")

        # 3. 轻量级 TF-IDF fallback（依赖 sklearn）
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            class TfidfEmbedding:
                """轻量级 TF-IDF 向量器 — 零依赖、零配置、总是可用"""
                def __init__(self):
                    self.dims = 256
                    self._vectorizer = None
                    self._fitted = False
                    self._corpus = []

                def encode(self, text: str):
                    if not self._fitted or len(self._corpus) < 50:
                        # 累积语料
                        self._corpus.append(text)
                        return self._hash_vector(text)
                    return self._tfidf_vector(text)

                def _hash_vector(self, text: str):
                    """基于字符的 hash vector，256维，保持语义近似"""
                    import hashlib
                    import struct
                    vec = [0.0] * self.dims
                    chars = list(text)
                    for i, ch in enumerate(chars):
                        h = hashlib.sha256(f"{i}:{ch}".encode()).digest()[:2]
                        idx = struct.unpack('<H', h)[0] % self.dims
                        vec[idx] += 1.0
                    norm = (sum(v * v for v in vec)) ** 0.5
                    if norm > 0:
                        vec = [v / norm for v in vec]
                    return vec

                def _tfidf_vector(self, text: str):
                    """当累积足够语料后，切换到真实 TF-IDF"""
                    if self._vectorizer is None:
                        self._vectorizer = TfidfVectorizer(
                            max_features=self.dims,
                            analyzer='char_wb',
                            ngram_range=(2, 4)
                        )
                        self._vectorizer.fit(self._corpus + [text])
                    try:
                        v = self._vectorizer.transform([text]).toarray()[0]
                        vec = list(v[:self.dims])
                        if len(vec) < self.dims:
                            vec += [0.0] * (self.dims - len(vec))
                        norm = (sum(x * x for x in vec)) ** 0.5
                        if norm > 0:
                            vec = [x / norm for x in vec]
                        return vec
                    except Exception:
                        return self._hash_vector(text)

            self._embedding = TfidfEmbedding()
            self._embedding_backend_type = "tfidf"
            logger.info("[SuMemoryLitePro] 使用 TF-IDF 轻量向量服务 (dim=256)")
            return self._embedding
        except Exception as e:
            logger.debug("降级处理: %s", e)

        # 4. 最终兜底: 纯 Hash vector (保证 dims 永远非None)
        class HashFallback:
            """终极兜底向量器 — 保证 dims 可用"""
            def __init__(self):
                self.dims = 128
            def encode(self, text: str):
                import hashlib
                import struct
                vec = [0.0] * self.dims
                for i, ch in enumerate(text):
                    h = hashlib.sha256(f"{i}:{ch}".encode()).digest()[:2]
                    idx = struct.unpack('<H', h)[0] % self.dims
                    vec[idx] += 1.0
                norm = (sum(v * v for v in vec)) ** 0.5
                if norm > 0:
                    vec = [v / norm for v in vec]
                return vec

        self._embedding = HashFallback()
        self._embedding_backend_type = "hash"
        logger.info("[SuMemoryLitePro] 使用 Hash 兜底向量服务 (dim=128)")
        return self._embedding

    def _ensure_faiss_index(self):
        """懒加载: 确保FAISS索引已创建（线程安全）。"""
        with self._faiss_lock:
            if self._faiss_index is not None:
                return self._faiss_index
            emb = self._ensure_embedding()
            if not emb or not FAISS_AVAILABLE:
                return None
            dims = getattr(emb, 'dims', 1024)
            self._faiss_index = faiss.IndexHNSWFlat(dims, 32)
            self._faiss_index.hnsw.efConstruction = 40
            logger.info(f"[SuMemoryLitePro] FAISS HNSW 索引已创建，维度={dims}")
            return self._faiss_index

    def _add_to_faiss(self, memory_id: str, embedding) -> None:
        """将一条记忆的向量加入 FAISS 索引并维护 id 双向映射。

        此前 add() 只把 embedding 存进 node，从不填充 FAISS 索引，
        导致 _faiss_index 始终为 None、_id_faiss_map 为空，向量检索永远回退到朴素搜索。
        """
        if not self.enable_vector or not FAISS_AVAILABLE:
            return
        if embedding is None:
            return
        with self._faiss_lock:
            try:
                import numpy as _np
                idx = self._ensure_faiss_index()
                if idx is None:
                    return
                vec = _np.array([embedding], dtype=_np.float32)
                # 已存在则跳过（避免重复插入）
                if memory_id in self._id_faiss_map:
                    return
                idx.add(vec)
                pos = idx.ntotal - 1
                self._id_faiss_map[memory_id] = pos
                self._faiss_id_map[pos] = memory_id
            except Exception as e:
                # 记录而非静默吞掉，避免向量插入失败导致检索静默回退
                logger.error(f"[SuMemoryLitePro] FAISS 插入失败 memory_id={memory_id}: {e}", file=sys.stderr)


    def _load_faiss_index(self) -> bool:
        """从磁盘加载已持久化的FAISS索引 (V3.16)"""
        if not self._faiss_index_path:
            return False
        idmap_path = self._faiss_index_path + ".idmap"
        try:
            if os.path.exists(self._faiss_index_path) and os.path.exists(idmap_path):
                self._faiss_index = faiss.read_index(self._faiss_index_path)
                with open(idmap_path) as f:
                    self._faiss_id_map = json.loads(f.read())
                self._id_faiss_map = {v: int(k) for k, v in self._faiss_id_map.items()}
                logger.debug(f"[SuMemoryLitePro] FAISS 索引从磁盘加载: {self._faiss_index.ntotal} 条向量")
                return True
        except Exception as e:
            logger.error(f"[SuMemoryLitePro] FAISS 索引加载失败: {e}")
        return False

    def _save_faiss_index(self):
        """持久化FAISS索引到磁盘 (V3.16)"""
        if not self._faiss_index or not self._faiss_index_path:
            return
        try:
            faiss.write_index(self._faiss_index, self._faiss_index_path)
            idmap_path = self._faiss_index_path + ".idmap"
            with open(idmap_path, 'w') as f:
                f.write(json.dumps(self._faiss_id_map, ensure_ascii=False))
        except Exception as e:
            logger.error(f"[SuMemoryLitePro] FAISS 索引保存失败: {e}")

    # ═══════════════════ 原有方法 ═══════════════════

    def _check_ollama(self) -> bool:
        """检查 Ollama 是否可用（2s超时）"""
        try:
            import urllib.request
            req = urllib.request.Request(
                "http://localhost:11434/api/tags",
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read())
                models = [m['name'] for m in data.get('models', [])]
                for model in models:
                    if 'bge' in model.lower() or 'embed' in model.lower():
                        return True
                return len(models) > 0
        except Exception:
            return False

    def _get_default_storage_path(self) -> str:
        """
        获取默认存储路径

        优先级：
        1. 环境变量 SU_MEMORY_DATA_DIR
        2. OpenClaw 环境: ~/.openclaw/su_memory_data/
        3. ~/.su_memory/
        4. 当前目录 ./su_memory_data/
        """
        import os as _os

        # 1. 检查环境变量
        env_path = _os.environ.get("SU_MEMORY_DATA_DIR")
        if env_path:
            return env_path

        home_path = _os.path.expanduser("~")

        # 2. OpenClaw 环境检测: 检查 OPENCLAW_DIR 或 ~/.openclaw
        openclaw_dir = _os.environ.get("OPENCLAW_DIR", _os.path.join(home_path, ".openclaw"))
        if _os.path.exists(openclaw_dir):
            openclaw_data = _os.path.join(openclaw_dir, "su_memory_data")
            try:
                _os.makedirs(openclaw_data, exist_ok=True)
                return openclaw_data
            except (OSError, PermissionError) as e:
                logger.debug("降级处理: %s", e)

        # 3. 使用默认用户目录
        default_path = _os.path.join(home_path, ".su_memory")
        try:
            _os.makedirs(default_path, exist_ok=True)
            test_file = _os.path.join(default_path, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            _os.remove(test_file)
            return default_path
        except (OSError, PermissionError) as e:
            logger.debug("降级处理: %s", e)

        # 3. 使用当前目录
        return _os.path.join(_os.getcwd(), "su_memory_data")

    def _trim_energy_cache(self, max_size: int = 2000):
        """LRU 裁剪 energy_cache，防止长时间运行下无限增长。"""
        while len(self._energy_cache) > max_size:
            self._energy_cache.popitem(last=False)

    def _infer_energy(self, content: str) -> str:
        """Infer energy type from content using LLM with keyword fallback.

        Uses local Ollama model (qwen3.5:9b-nothink) for semantic understanding.
        Falls back to keyword matching if LLM is unavailable.
        Results are cached by MD5 hash of content.
        """
        import hashlib
        import os


        # Check cache
        content_hash = hashlib.md5(content.encode()).hexdigest()
        if not hasattr(self, '_energy_cache'):
            self._energy_cache = OrderedDict()
        if content_hash in self._energy_cache:
            return self._energy_cache[content_hash]

        # P0-1 修复：LLM 能量推断默认关闭（opt-in），避免 add() 主路径同步阻塞远程调用。
        # 生产安全：默认走关键词分类（零网络依赖、亚毫秒级）。
        # 显式开启方式：构造函数 enable_llm_energy=True 或环境变量 SU_MEMORY_LLM_ENERGY=1
        use_llm = getattr(self, "enable_llm_energy", False)
        if use_llm:
            try:
                result = self._llm_infer_energy(content)
                if result in ("wood", "fire", "earth", "metal", "water"):
                    self._energy_cache[content_hash] = result
                    self._trim_energy_cache()
                    return result
            except Exception as e:
                logger.debug("LLM 能量推断降级到关键词分类: %s", e)

        # Keyword fallback (默认路径)
        energy_keywords = {
            "wood": ["生长", "发展", "树木", "森林", "绿色", "东方", "春季", "肝", "筋",
                     "wood", "growth", "spring", "forest", "green", "east", "tree", "plant", "leaf", "garden", "liver"],
            "fire": ["热情", "炎热", "红色", "南方", "夏季", "心", "血液", "高温",
                     "fire", "passion", "heat", "summer", "red", "south", "flame", "blaze", "warm", "burning", "heart", "blood"],
            "earth": ["稳定", "黄色", "中央", "四季", "脾", "消化", "土地",
                      "earth", "stability", "center", "balance", "yellow", "ground", "foundation", "soil", "terrain", "spleen"],
            "metal": ["收敛", "白色", "西方", "秋季", "肺", "呼吸", "金属",
                      "metal", "autumn", "white", "west", "structure", "refinement", "steel", "iron", "gold", "silver", "sword", "lung"],
            "water": ["流动", "蓝色", "北方", "冬季", "肾", "泌尿", "智慧",
                      "water", "flow", "winter", "blue", "north", "wisdom", "ocean", "river", "rain", "stream", "lake", "ice", "cold", "fluid", "kidney"]
        }

        scores = dict.fromkeys(energy_keywords, 0)
        for e, kws in energy_keywords.items():
            for kw in kws:
                if kw in content:
                    scores[e] += 1

        result = max(scores, key=scores.get) if max(scores.values()) > 0 else "earth"
        self._energy_cache[content_hash] = result
        self._trim_energy_cache()
        return result

    def _llm_infer_energy(self, content: str) -> str:
        """Use LLM to infer energy type from content semantics.

        Provider priority: DeepSeek API → MiniMax API → Ollama local → empty.
        Returns empty string on failure (caller falls back to keyword).
        Each provider is tried with a short timeout; first valid result wins.
        """
        import os

        import requests

        prompt = (
            "Classify this text into exactly one of five categories.\n"
            "wood: growth, plants, spring, east, green, expansion, creativity\n"
            "fire: passion, summer, south, red, heat, energy, enthusiasm\n"
            "earth: stability, center, yellow, grounding, nurturing, balance\n"
            "metal: structure, autumn, west, white, precision, refinement\n"
            "water: wisdom, winter, north, blue, flow, adaptability, depth\n\n"
            f"Text: {content[:300]}\n\n"
            "Respond with exactly one word: wood, fire, earth, metal, or water."
        )

        # ── Provider 1: DeepSeek API (OpenAI-compatible) ──
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if deepseek_key:
            try:
                resp = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {deepseek_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 10,
                        "temperature": 0,
                    },
                    timeout=(5, 30)
                )
                if resp.status_code == 200:
                    text = resp.json()["choices"][0]["message"]["content"].strip().lower()
                    for et in ("wood", "fire", "earth", "metal", "water"):
                        if et in text:
                            return et
            except Exception as e:
                logger.debug("能量推断 API 降级: %s", e)

        # ── Provider 2: MiniMax API (OpenAI-compatible) ──
        minimax_key = os.environ.get("MINIMAX_API_KEY", "")
        if minimax_key:
            try:
                resp = requests.post(
                    "https://api.minimax.chat/v1/text/chatcompletion_v2",
                    headers={
                        "Authorization": f"Bearer {minimax_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "abab6.5s-chat",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 10,
                        "temperature": 0,
                    },
                    timeout=(5, 30)
                )
                if resp.status_code == 200:
                    text = resp.json()["choices"][0]["message"]["content"].strip().lower()
                    for et in ("wood", "fire", "earth", "metal", "water"):
                        if et in text:
                            return et
            except Exception as e:
                logger.debug("能量推断 API 降级: %s", e)

        # ── Provider 3: Ollama local ──
        # Cache availability check: only probe once per instance
        if not hasattr(self, '_ollama_checked'):
            self._ollama_available = False
            try:
                r = requests.get("http://localhost:11434/api/tags", timeout=0.2)
                self._ollama_available = r.status_code == 200
            except Exception as e:
                logger.debug("能量推断 API 降级: %s", e)
            self._ollama_checked = True

        if self._ollama_available:
            models_to_try = ["qwen3.5:9b-nothink", "tinyllama", "gemma3:4b"]
            for model in models_to_try:
                try:
                    resp = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": model,
                            "prompt": prompt,
                            "stream": False,
                            "raw": True,
                            "options": {
                                "temperature": 0,
                                "num_predict": 20,
                                "stop": ["\n", ".", ","]
                            },
                        },
                        timeout=(5, 30)
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        text = (data.get("response") or "").strip().lower()
                        if not text:
                            text = (data.get("thinking") or "").strip().lower()
                        for et in ("wood", "fire", "earth", "metal", "water"):
                            if et in text:
                                return et
                except Exception:
                    continue

        return ""

    # ==================== 会话管理 ====================
    def create_session(self, session_id: str = None, metadata: dict = None) -> str:
        """
        创建新会话

        Args:
            session_id: 会话ID
            metadata: 会话元数据

        Returns:
            会话ID
        """
        if not self._sessions:
            self._sessions = SessionManager(self.storage_path, self._embedding)
        return self._sessions.create_session(session_id, metadata)

    def add(
        self,
        content: str,
        metadata: dict = None,
        parent_ids: list[str] = None,
        topic: str = None,
        session_id: str = None,
        source_type: str = "unknown",
        source_id: str = "",
        source_confidence: float = 1.0,
        event_time: int | None = None,
    ) -> str:
        """添加记忆

        Args:
            source_type: 来源类型 order|lab_report|patient|ai_inferred|imported
            source_id: 原始记录 ID（病历号/对话ID/FHIR Resource ID）
            source_confidence: 来源可信度 [0,1]
            event_time: 事件实际发生时间（Unix秒），缺省=入库时间（C4 双时间模型）
        """
        if not isinstance(content, str):
            raise TypeError(f"content 必须是 str，收到 {type(content).__name__}")
        import uuid

        memory_id = f"mem_{uuid.uuid4().hex[:8]}"
        timestamp = int(time.time())
        # C4: 双时间模型——event_time 缺省回退到入库时间
        # V9/V15/V16: event_time 合法性校验（拒绝负数；未来时间 clamp 到 now）
        if event_time is not None:
            et = int(event_time)
            if et < 0:
                logger.warning("event_time=%d 为负数，回退到入库时间", et)
                node_event_time = timestamp
            elif et > timestamp:
                logger.warning("event_time=%d 为未来时间，clamp 到 now", et)
                node_event_time = timestamp
            else:
                node_event_time = et
        else:
            node_event_time = timestamp

        # 推断energy_type
        energy_type = self._infer_energy(content)

        # 获取embedding (V3.16: 懒加载)
        embedding = None
        if self.enable_vector:
            emb = self._ensure_embedding()
            if emb:
                try:
                    embedding = emb.encode(content)
                except Exception as e:
                    logger.debug("降级处理: %s", e)

        # 创建节点（复制parent_ids避免引用问题）
        node_parent_ids = list(parent_ids) if parent_ids else []
        node = MemoryNode(
            id=memory_id,
            content=content,
            metadata=metadata or {},
            embedding=embedding,
            keywords=self._tokenize(content),
            timestamp=timestamp,
            parent_ids=node_parent_ids,
            energy_type=energy_type,
            source_type=source_type,
            source_id=source_id,
            source_confidence=source_confidence,
            event_time=node_event_time,
        )

        # P1: Register in energy causal engine for re-ranking
        if self._causal is not None:
            try:
                self._causal.add_node(memory_id, content, energy_type=energy_type)
            except Exception as e:
                logger.debug("降级处理: %s", e)

        # P1: Attach unified energy label to metadata
        if self._unified_factory is not None and metadata is not None:
            try:
                energy_int = {"wood": 0, "fire": 1, "earth": 2, "metal": 3, "water": 4}.get(energy_type, 2)
                tc = self._temporal.get_time_code(timestamp)
                stem_idx = self._temporal.TIME_STEMS.index(tc.get("stem", "jia"))
                branch_idx = self._temporal.TIME_BRANCHES.index(tc.get("branch", "zi"))
                unit = self._unified_factory.create_from_content(
                    content, stem_idx=stem_idx, branch_idx=branch_idx,
                    energy_type=energy_int
                )
                metadata["_energy_label"] = unit.to_dict()
            except Exception as e:
                logger.debug("降级处理: %s", e)

        # P2: Register in EnergyBus propagation network
        if self._energy_bus is not None:
            try:
                from su_memory._sys._energy_bus import EnergyLayer, EnergyNode
                eb_node = EnergyNode(
                    node_id=memory_id, energy_type=energy_type,
                    layer=EnergyLayer.FIVE_ELEMENTS, intensity=1.0
                )
                self._energy_bus.add_node(eb_node, auto_connect=True)
            except Exception as e:
                logger.debug("降级处理: %s", e)

        # 存储
        self._memories.append(node)
        self._memory_map[memory_id] = len(self._memories) - 1

        # 更新索引
        for kw in node.keywords:
            self._index[kw].add(memory_id)

        # 软分类: 分配主题簇
        cluster_id = self._topic_clusterer.assign(memory_id, node.keywords)
        node.metadata["_topic_cluster"] = cluster_id

        # 将向量加入 FAISS 索引（修复：此前 add 从不填充 FAISS，向量检索实际未启用）
        if embedding is not None:
            self._add_to_faiss(memory_id, embedding)

        # 更新图谱
        if self._graph:
            self._graph.add_node(node)
            for parent_id in (parent_ids or []):
                # 显式声明的关联 = 最高成色 explicit(0.95)
                self._graph.add_edge(
                    parent_id, memory_id,
                    causal_type="explicit",
                    confidence=0.95, evidence_type="explicit",
                )

        # 更新 VectorGraphRAG
        if self._vector_graph is not None:
            try:
                # 推断因果类型
                causal_type = None
                if parent_ids:
                    # 检查是否包含因果关键词
                    causal_keywords = ["导致", "因为", "所以", "因此", "如果"]
                    for kw in causal_keywords:
                        if kw in content:
                            causal_type = "cause"
                            break

                self._vector_graph.add_memory(
                    memory_id=memory_id,
                    content=content,
                    parent_ids=parent_ids,
                    causal_type=causal_type
                )
            except Exception as e:
                logger.error(f"[SuMemoryLitePro] VectorGraphRAG 添加失败: {e}")

        # 更新 SpacetimeIndex 时空索引
        if self._spacetime is not None:
            try:
                self._spacetime.add_node(
                    node_id=memory_id,
                    content=content,
                    timestamp=node_event_time,  # C4: 时空索引用事件时间
                    energy_type=energy_type
                )
                # 添加边关系
                for parent_id in (parent_ids or []):
                    if parent_id in self._memory_map:
                        self._spacetime.add_edge(parent_id, memory_id)
            except Exception as e:
                logger.error(f"[SuMemoryLitePro] SpacetimeIndex 添加失败: {e}")

        # 同步更新 SpacetimeMultihopEngine 的 memory_nodes
        if self._spacetime_engine and node:
            try:
                self._spacetime_engine.memory_nodes[memory_id] = node
            except Exception as e:  # 静默失败，不影响主流程
                logger.debug("降级: %s", e)

        # 同步更新 MultimodalEmbedding 的 memory_nodes
        if self._multimodal and node:
            try:
                self._multimodal.add_multimodal_memory(
                    memory_id=memory_id,
                    content=content,
                    text_vector=embedding if embedding is not None else None,
                    timestamp=timestamp,
                    energy_type=energy_type,
                    metadata=metadata
                )
            except Exception as e:  # 静默失败，不影响主流程
                logger.debug("降级: %s", e)

        # 更新会话
        if self._sessions:
            current = self._sessions.get_current_session()
            if current:
                self._sessions.add_memory(current, memory_id, topic)

        # 记录预测事件
        if self._prediction:
            self._prediction.record_event(content, timestamp, metadata)

        # 记录推理步骤
        if self._explainability:
            self._explainability.record_reasoning_step(
                "action",
                f"添加记忆: {content[:30]}...",
                {"memory_id": memory_id, "energy_type": energy_type}
            )

        # 内存限制
        if len(self._memories) > self.max_memories:
            self._evict_oldest()

        # 清除缓存
        self._query_cache.clear()

        # 持久化（节流：避免每次 add 全量序列化的 O(n) 写放大）
        self._maybe_save()

        return memory_id

    def add_batch(
        self,
        items: list,
        metadata: dict = None,
        parent_ids: list[str] = None,
        session_id: str = None
    ) -> list[str]:
        """
        批量添加记忆

        Args:
            items: 可以是字符串列表 ["记忆1", "记忆2"] 
                  或dict列表 [{"content": "..."}, {"content": "...", "topic": "..."}]
            metadata: 全局元数据（所有记忆共享）
            parent_ids: 全局父节点ID
            session_id: 会话ID

        Returns:
            记忆ID列表
        """
        ids = []
        max_len = 8000  # 单条记忆最大长度

        for item in items:
            if isinstance(item, str):
                content = item[:max_len]
                topic = None
                item_meta = (metadata or {}).copy()
            elif isinstance(item, dict):
                content = item.get("content", "")[:max_len]
                topic = item.get("topic")
                item_meta = (metadata or {}).copy()
                if "metadata" in item and isinstance(item["metadata"], dict):
                    item_meta.update(item["metadata"])
            else:
                continue

            if not content.strip():
                continue

            mid = self.add(
                content=content,
                metadata=item_meta,
                parent_ids=parent_ids,
                topic=topic,
                session_id=session_id
            )
            ids.append(mid)

        return ids

    def _evict_oldest(self):
        """淘汰最旧的记忆（基于时间戳），同步清理所有关联索引。"""
        if not self._memories:
            return
        oldest_idx = 0
        oldest_ts = float('inf')
        for i, node in enumerate(self._memories):
            ts = node.timestamp if hasattr(node, 'timestamp') else 0
            if ts < oldest_ts:
                oldest_ts = ts
                oldest_idx = i
        removed = self._memories.pop(oldest_idx)
        rid = removed.id
        if rid in self._memory_map:
            del self._memory_map[rid]
        self._memory_map = {node.id: i for i, node in enumerate(self._memories)}
        # 同步清理倒排索引（防止幽灵 ID 残留导致内存泄漏与查询膨胀）
        for kw in removed.keywords:
            if kw in self._index:
                self._index[kw].discard(rid)
                if not self._index[kw]:
                    del self._index[kw]
        # 同步清理主题分桶
        if hasattr(self, '_topic_clusterer'):
            self._topic_clusterer.remove(rid)

    def query(
        self,
        query: str,
        top_k: int = 5,
        use_vector: bool = None,
        use_keyword: bool = None,
        use_spacetime: bool = False,  # 新增：是否使用时空索引
        session_id: str = None,
        time_range: tuple[int, int] = None,  # 新增：时间范围过滤
        energy_filter: str = None,  # 新增：Energy System过滤
        **kwargs,  # 容忍/忽略上层（如 BayesianAugmenter）透传的未知参数
    ) -> list[dict]:
        """
        混合检索

        Args:
            query: 查询文本
            top_k: 返回数量
            use_vector: 是否使用向量检索
            use_keyword: 是否使用关键词检索
            use_spacetime: 是否使用时空索引（融合时间衰减+Energy System能量）
            session_id: 限定会话
            time_range: 时间范围 (start_ts, end_ts)
            energy_filter: Energy System类型过滤
        """
        if not isinstance(query, str):
            raise TypeError(f"query 必须是 str，收到 {type(query).__name__}")
        use_vector = use_vector if use_vector is not None else self.enable_vector
        use_keyword = use_keyword if use_keyword is not None else self.enable_tfidf

        # 检查缓存
        cache_key = (query, top_k, use_vector, use_keyword, session_id, use_spacetime, energy_filter, time_range)
        if cache_key in self._query_cache:
            self._cache_hits += 1
            return self._query_cache[cache_key].copy()

        self._cache_misses += 1

        # ── 软分类路由: 先定位 top-3 主题桶, 桶内候选优先 ──
        # 当记忆量大时, 把 O(全部) 检索缩减为 O(桶大小)
        cluster_candidates: set[str] | None = None
        if self._topic_clusterer.n_clusters > 3:
            query_kws = self._tokenize(query)
            top_clusters = self._topic_clusterer.query_clusters(query_kws, top_k=3)
            if top_clusters:
                cluster_candidates = set()
                for cid, _sim in top_clusters:
                    cluster_candidates.update(self._topic_clusterer.cluster_members(cid))

        results = []

        # ========================================
        # 时空索引检索（优先级最高）
        # ========================================
        if use_spacetime and self._spacetime:
            try:
                st_results = self._spacetime.search(
                    query=query,
                    top_k=top_k * 2,
                    use_temporal=True,
                    time_range=time_range,
                    energy_filter=energy_filter
                )

                if st_results:
                    # 转换为标准格式
                    st_dict = {}
                    for r in st_results:
                        idx = self._memory_map.get(r["node_id"])
                        if idx is not None:
                            node = self._memories[idx]
                            st_dict[r["node_id"]] = {
                                "memory_id": r["node_id"],
                                "content": r["content"],
                                "score": r["score"],
                                "metadata": node.metadata,
                                "timestamp": r["timestamp"],
                                "time_decay": r.get("time_decay", 1.0),
                                "energy_boost": r.get("energy_boost", 1.0),
                                "energy_type": r.get("energy_type", "earth")
                            }

                    if st_dict:
                        results.append(("spacetime", list(st_dict.values())))
            except Exception as e:
                logger.error(f"[SuMemoryLitePro] SpacetimeIndex 查询失败: {e}")

        # 关键词检索
        if use_keyword:
            kw_results = self._keyword_search(query)
            results.append(("keyword", kw_results))

        # 向量检索
        if use_vector and self._embedding:
            vec_results = self._vector_search(query)
            results.append(("vector", vec_results))

        # 融合结果
        if len(results) > 1:
            fused = self._fusion_search(results, top_k)
        elif results:
            fused = results[0][1][:top_k]
        else:
            fused = []

        # 时序重排（使用 TemporalSystem）
        if self._temporal:
            fused = self._temporal_rerank(fused)

        # P1: Energy affinity re-ranking via causal engine
        if self._causal is not None and fused:
            try:
                query_energy = energy_filter or self._infer_energy(query)
                qid = f"_query_{hash(query) % 100000}"
                self._causal.add_node(qid, query, energy_type=query_energy)
                candidate_ids = [r["memory_id"] for r in fused]
                base_scores = {r["memory_id"]: r.get("score", 0.5) for r in fused}
                boosted = self._causal.query_with_energy_boost(
                    qid, candidate_ids, base_scores
                )
                if boosted:
                    boost_map = {b["node_id"]: b["boosted_score"] for b in boosted}
                    for r in fused:
                        if r["memory_id"] in boost_map:
                            r["score"] = boost_map[r["memory_id"]]
                    fused.sort(key=lambda x: x.get("score", 0), reverse=True)
            except Exception as e:
                logger.debug("降级处理: %s", e)

        # P2: EnergyBus propagation + balance boost
        if self._energy_bus is not None and fused:
            try:
                query_energy = energy_filter or self._infer_energy(query)
                qid = f"_eb_q_{hash(query) % 100000}"
                from su_memory._sys._energy_bus import EnergyLayer, EnergyNode
                qnode = EnergyNode(
                    node_id=qid, energy_type=query_energy,
                    layer=EnergyLayer.FIVE_ELEMENTS, intensity=1.5
                )
                self._energy_bus.add_node(qnode, auto_connect=False)
                self._energy_bus.propagate_energy(qid, delta=0.3, max_hops=2)

                # Apply energy balance bonus to scores
                bus_state = self._energy_bus.get_bus_state()
                eb = bus_state.get("energy_balance", {})
                ratios = eb.get("ratios", {})
                if ratios:
                    for r in fused:
                        mem_energy = r.get("energy_type", "earth")
                        ratio = ratios.get(mem_energy, 0.2)
                        r["score"] = r.get("score", 0.5) * (0.85 + ratio * 0.5)
                    fused.sort(key=lambda x: x.get("score", 0), reverse=True)
            except Exception as e:
                logger.debug("降级处理: %s", e)

        # 软分类提权: 在主题桶内的候选记忆分数 ×1.2 (桶内更可能是相关信号)
        if cluster_candidates:
            for r in fused:
                if r.get("memory_id") in cluster_candidates:
                    r["score"] = r.get("score", 0.5) * 1.2
            fused.sort(key=lambda x: x.get("score", 0), reverse=True)

        # 会话过滤
        if session_id and self._sessions:
            session_mids = set(self._sessions.get_session_memories(session_id))
            fused = [r for r in fused if r["memory_id"] in session_mids]

        # 访问计数: 被检索命中的记忆自增 access_count (频率加权衰减的地基)
        # 同时统一补 provenance 字段到检索结果（C5: 来源溯源）
        now = int(time.time())
        for r in fused[:top_k]:
            idx = self._memory_map.get(r.get("memory_id"))
            if idx is not None:
                node = self._memories[idx]
                node.access_count += 1
                node.last_accessed = now
                r["source_type"] = node.source_type
                r["source_id"] = node.source_id
                r["source_confidence"] = node.source_confidence
                r["event_time"] = node.effective_time
                r["version"] = node.version
                r["superseded_by"] = node.superseded_by
            else:
                # 历史兼容：无 node 的结果补默认值
                r.setdefault("source_type", "unknown")
                r.setdefault("source_id", "")
                r.setdefault("source_confidence", 1.0)
                r.setdefault("event_time", r.get("timestamp", 0))
                r.setdefault("version", 1)
                r.setdefault("superseded_by", "")

        # 缓存
        self._query_cache[cache_key] = fused
        if len(self._query_cache) > self._cache_size:
            self._query_cache.popitem(last=False)

        return fused[:top_k]

    def _keyword_search(self, query: str, top_k: int = 20) -> list[dict]:
        """关键词检索"""
        query_kws = self._tokenize(query)

        scores = defaultdict(float)
        for kw in query_kws:
            if kw in self._index:
                idf = math.log(len(self._memories) / (len(self._index[kw]) + 1)) + 1
                for mem_id in self._index[kw]:
                    scores[mem_id] += idf

        if not scores:
            return []

        max_score = max(scores.values())
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for mem_id, score in sorted_ids[:top_k]:
            idx = self._memory_map.get(mem_id)
            if idx is not None:
                node = self._memories[idx]
                results.append({
                    "memory_id": mem_id,
                    "content": node.content,
                    "score": score / max_score,
                    "metadata": node.metadata,
                    "timestamp": node.timestamp
                })

        return results

    def _vector_search(self, query: str, top_k: int = 20) -> list[dict]:
        """向量检索（优先使用 FAISS 索引）"""
        # V3.16: 懒加载embedding
        emb = self._ensure_embedding()
        if not emb:
            return []

        try:
            query_vec = emb.encode(query)
        except Exception:
            return []

        # 优先使用 FAISS 索引 (V3.16: 懒加载 + 持久化)
        if self._ensure_faiss_index() and self._id_faiss_map:
            return self._faiss_vector_search(query_vec, top_k)

        # 回退到朴素搜索
        return self._naive_vector_search(query_vec, top_k)

    def _faiss_vector_search(self, query_vec: list[float], top_k: int = 20) -> list[dict]:
        """使用 FAISS HNSW 索引的向量搜索（读锁，与写入互斥）。"""
        try:
            query_np = np.array([query_vec], dtype=np.float32)
            with self._faiss_lock:
                if self._faiss_index is None or not self._id_faiss_map:
                    return []
                # 设置搜索参数
                if hasattr(self._faiss_index, 'hnsw'):
                    self._faiss_index.hnsw.efSearch = 64

                # 搜索
                distances, indices = self._faiss_index.search(query_np, min(top_k * 2, len(self._id_faiss_map)))

            results = []
            max_dist = max(distances[0]) if distances[0][0] > 0 else 1.0

            for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                if idx < 0:
                    continue

                memory_id = self._faiss_id_map.get(int(idx))
                if not memory_id:
                    continue

                # 关键修复：HNSW 返回 L2 距离，不是相似度！
                # 距离越小越相似，需要转换
                if max_dist > 0:
                    similarity = 1.0 - (dist / max_dist)
                else:
                    similarity = 1.0

                # 获取完整记忆信息
                mem_idx = self._memory_map.get(memory_id)
                if mem_idx is not None:
                    node = self._memories[mem_idx]
                    results.append({
                        "memory_id": memory_id,
                        "content": node.content,
                        "score": similarity,
                        "metadata": node.metadata,
                        "timestamp": node.timestamp
                    })

            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.error(f"[SuMemoryLitePro] FAISS 搜索失败: {e}")
            return self._naive_vector_search(query_vec, top_k)

    def _naive_vector_search(self, query_vec: list[float], top_k: int = 20) -> list[dict]:
        """朴素向量搜索（O(n)线性扫描）"""
        results = []
        for node in self._memories:
            if node.embedding:
                sim = cosine_similarity(query_vec, node.embedding)
                results.append({
                    "memory_id": node.id,
                    "content": node.content,
                    "score": sim,
                    "metadata": node.metadata,
                    "timestamp": node.timestamp
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _fusion_search(self, results_list: list[tuple[str, list[dict]]], top_k: int) -> list[dict]:
        """
        增强版RRF融合
        使用改进的融合算法，支持方法权重
        """
        # 方法权重（向量检索权重更高）
        method_weights = []
        rrf_results = []

        for method, results in results_list:
            rrf_results.append([
                (r["memory_id"], r["score"]) for r in results
            ])
            # 根据方法类型设置权重
            if method == "vector":
                method_weights.append(2.0)  # 向量检索权重更高
            elif method == "keyword":
                method_weights.append(1.0)
            elif method == "session":
                method_weights.append(1.5)  # 会话检索中等权重
            else:
                method_weights.append(1.0)

        # 使用增强版RRF融合
        fused = rrf_fusion(
            rrf_results,
            k=60,
            use_score_weight=True,
            method_weights=method_weights
        )

        # 构建返回结果
        final_results = []
        for mem_id, rrf_score in fused[:top_k]:
            idx = self._memory_map.get(mem_id)
            if idx is not None:
                node = self._memories[idx]
                final_results.append({
                    "memory_id": mem_id,
                    "content": node.content,
                    "score": rrf_score,
                    "metadata": node.metadata,
                    "timestamp": node.timestamp
                })

        return final_results

    def _temporal_rerank(self, results: list[dict]) -> list[dict]:
        """Temporal recency-based reranking（V15: 用 effective_time 而非入库时间）。

        event_time（事件实际发生时间）优先，缺省回退入库时间。
        这是双时间模型的核心——"上周的就诊"应按上周衰减，而非按今天入库算新。
        """
        ts = int(time.time())

        for r in results:
            node = self._memories[self._memory_map.get(r["memory_id"])]
            energy_type = getattr(node, 'energy_type', 'earth')
            # V15: 优先事件时间（effective_time 已含 >0 回退逻辑）
            mem_ts = r.get("event_time") or r.get("timestamp") or getattr(node, 'effective_time', 0)
            recency = self._temporal.calculate_recency_score(
                mem_ts, energy_type, ts
            )
            # Combine original score and temporal score
            r["score"] = r["score"] * 0.7 + recency * 0.3

        return sorted(results, key=lambda x: x["score"], reverse=True)

    def query_multihop(
        self,
        query: str,
        max_hops: int = 3,
        top_k: int = 5,
        use_vector: bool = None,
        causal_only: bool = False,
        fusion_mode: str = "hybrid",  # hybrid: vector 60% + graph 40% for better multi-hop
        energy_filter: str = None  # filter by energy category (wood/fire/earth/metal/water)
    ) -> list[dict]:
        """
        多跳推理查询

        融合 VectorGraphRAG（语义引导）+ MemoryGraph（因果结构）双重推理能力

        Args:
            query: 查询文本
            max_hops: 最大跳数
            top_k: 返回数量
            use_vector: 是否使用向量检索
            causal_only: 是否只返回因果相关结果
            fusion_mode: 融合模式
                - "vector_first": VectorGraphRAG 优先
                - "graph_first": MemoryGraph BFS 优先（保证多跳展开）
                - "hybrid": 两者融合（向量化60% + 图谱40%，推荐）
        """
        # ========================================
        # 模式1: VectorGraphRAG 语义引导推理（默认）
        # ========================================
        if fusion_mode in ["vector_first", "hybrid"]:
            if self._vector_graph and len(self._vector_graph) > 0:
                try:
                    vg_results = self._vector_graph.multi_hop_query(
                        query, max_hops=max_hops, top_k=top_k * 2
                    )

                    if vg_results:
                        vg_dict = {}
                        for r in vg_results:
                            idx = self._memory_map.get(r.node_id)
                            if idx is not None:
                                node = self._memories[idx]
                                vg_dict[r.node_id] = {
                                    "memory_id": r.node_id,
                                    "content": r.content,
                                    "score": r.score,
                                    "metadata": node.metadata,
                                    "hops": r.hops,
                                    "path": r.path,
                                    "causal_type": r.causal_type,
                                    "source": "vector_graph"
                                }

                        # 融合模式：结合 MemoryGraph 增强因果结构
                        if fusion_mode == "hybrid" and self._graph:
                            vg_dict = self._enhance_with_graph(vg_dict, query, max_hops, top_k)

                        # 按得分排序
                        results = list(vg_dict.values())
                        results.sort(key=lambda x: x["score"], reverse=True)

                        # 清理临时字段
                        for r in results:
                            r.pop("source", None)

                        # 记录推理步骤
                        if self._explainability:
                            self._explainability.record_reasoning_step(
                                "recall",
                                f"多跳推理: {query}, {len(results)}个结果, 模式={fusion_mode}",
                                {"hops": max_hops, "top_k": top_k, "engine": "vector_graph"}
                            )

                        # P1: Apply energy filter
                        if energy_filter and results:
                            results = [r for r in results
                                      if r.get("energy_type", "earth") == energy_filter]

                        return results[:top_k]
                except Exception as e:
                    logger.error(f"[SuMemoryLitePro] VectorGraphRAG 查询失败: {e}")

        # ========================================
        # 模式2: MemoryGraph 因果结构推理
        # ========================================
        if fusion_mode == "graph_first" or not self._vector_graph:
            return self._query_multihop_graph(query, max_hops, top_k, use_vector, causal_only)

        return []

    def query_multihop_spacetime(
        self,
        query: str,
        max_hops: int = 3,
        top_k: int = 5,
        use_spacetime_weight: bool = True,
        fusion_mode: str = "hybrid",
        energy_filter: str = None  # filter by energy category
    ) -> list[dict]:
        """
        时空多跳融合推理（融合 VectorGraphRAG + SpacetimeIndex）

        这是 query_multihop() 的增强版本，增加了时空维度支持：
        - 时间衰减：越早的记忆衰减越大
        - Energy System能量：根据当前时辰调整检索权重
        - RRF融合：统一不同引擎的得分

        Args:
            query: 查询文本
            max_hops: 最大跳数
            top_k: 返回数量
            use_spacetime_weight: 是否使用时空调权
            fusion_mode: 融合模式
                - "auto": 自动选择
                - "spacetime_first": SpacetimeIndex 优先
                - "vector_first": VectorGraphRAG 优先
                - "hybrid": 两者均衡融合

        Returns:
            List[Dict]，包含时空信息的检索结果
        """
        if not self._spacetime_engine:
            # 回退到普通多跳
            return self.query_multihop(query, max_hops, top_k, fusion_mode="vector_first")

        try:
            # 调用时空多跳融合引擎
            st_results = self._spacetime_engine.search(
                query=query,
                max_hops=max_hops,
                top_k=top_k * 2,
                use_vector_graph=True,
                use_spacetime=use_spacetime_weight,
                fusion_mode=fusion_mode
            )

            if not st_results:
                return []

            # 转换为标准格式
            results = []
            for r in st_results:
                idx = self._memory_map.get(r.node_id)
                if idx is not None:
                    node = self._memories[idx]
                    results.append({
                        "memory_id": r.node_id,
                        "content": r.content,
                        "score": r.score,
                        "metadata": node.metadata,
                        "timestamp": r.timestamp,
                        "time_decay": r.time_decay,
                        "energy_boost": r.energy_boost,
                        "energy_type": r.energy_type,
                        "hops": r.hops,
                        "path": r.path,
                        "vector_score": r.vector_score,
                        "source": r.source
                    })

            # 清理临时字段
            for r in results:
                r.pop("source", None)

            # 记录推理步骤
            if self._explainability:
                self._explainability.record_reasoning_step(
                    "recall",
                    f"时空多跳: {query}, {len(results)}个结果, 模式={fusion_mode}",
                    {"hops": max_hops, "top_k": top_k, "engine": "spacetime_multihop"}
                )

            return results[:top_k]

        except Exception as e:
            logger.error(f"[SuMemoryLitePro] 时空多跳推理失败: {e}")
            # 回退到普通多跳
            return self.query_multihop(query, max_hops, top_k)

    def _enhance_with_graph(
        self,
        vg_dict: dict[str, dict],
        query: str,
        max_hops: int,
        top_k: int
    ) -> dict[str, dict]:
        """利用 MemoryGraph + 实体桥接图增强 VectorGraphRAG 结果.

        两层增强:
        1. MemoryGraph 因果链接: 有因果关联的节点 +10% 得分.
        2. 实体桥接召回 (CausalDAG): 从 vector top 结果出发, 召回共享
           命名实体的桥接段落 — 即与 query 无直接语义重叠、但通过
           seed 段落的实体桥接才能找到的多跳证据 (HotpotQA bridge 结构).
           桥接段落按实体特异性 idf 融合进 vg_dict, 补足 vector 召回盲区.
        """
        if not vg_dict:
            return vg_dict

        # (1) MemoryGraph 因果增强
        if self._graph:
            for node_id, result in vg_dict.items():
                parents = self._graph.get_parents(node_id)
                children = self._graph.get_children(node_id)
                if parents or children:
                    result["score"] *= 1.1
                    result["has_causal_links"] = True

        # (2) 实体桥接召回 — 补足 vector 召不回的多跳证据
        try:
            vg_dict = self._bridge_enhance(vg_dict, top_k)
        except Exception as e:  # 桥接失败不影响主路径
            logger.debug("降级: %s", e)

        return vg_dict

    def _bridge_enhance(
        self, vg_dict: dict[str, dict], top_k: int
    ) -> dict[str, dict]:
        """用实体桥接图召回多跳证据, 融合进 vector 结果.

        从 vg_dict 得分最高的若干 seed 出发, 召回共享实体的桥接段落.
        新召回的段落以 *特异性融合分* (桥接特异性归一化) 计入, 低于
        vector 原始得分但补足召回盲区; 已在 vg_dict 中的段落获得桥接加分.
        """
        if not self._memories or len(vg_dict) >= top_k:
            return vg_dict  # 已满, 无需桥接

        ids = [n.id for n in self._memories]
        contents = [n.content for n in self._memories]
        recaller = EntityBridgeRecaller(ids=ids, contents=contents)

        # seed = vector top 结果 (得分降序前 3); 先构建桥接图索引
        recaller._ensure_built()
        seeds = sorted(vg_dict.items(), key=lambda x: -x[1].get("score", 0.0))
        seed_ids = [nid for nid, _ in seeds[:3] if nid in recaller._idx]
        if not seed_ids:
            return vg_dict

        # 计算所有段落的桥接特异性
        spec = recaller.specificity_scores(seed_ids)
        if not spec:
            return vg_dict
        max_spec = max(spec.values()) or 1.0

        # vector 结果的最高分作为桥接分数的基准
        base_score = max((r.get("score", 0.0) for r in vg_dict.values()), default=0.0)

        for mid, sp in spec.items():
            norm_spec = sp / max_spec  # [0,1]
            if mid in vg_dict:
                # 已召回的段落: 桥接加分 (最多 +15%)
                vg_dict[mid]["score"] *= (1.0 + 0.15 * norm_spec)
                vg_dict[mid]["bridge_boost"] = round(0.15 * norm_spec, 3)
            else:
                # 新桥接段落: 以 vector 基准分的 60-85% 计入 (低于直接命中, 但补足盲区)
                bridge_score = base_score * (0.60 + 0.25 * norm_spec)
                idx = recaller._idx.get(mid)
                if idx is not None and idx < len(self._memories):
                    node = self._memories[idx]
                    vg_dict[mid] = {
                        "memory_id": mid,
                        "content": node.content,
                        "score": bridge_score,
                        "metadata": node.metadata,
                        "hops": 2,
                        "path": seed_ids[:1] + [mid],
                        "causal_type": "entity_bridge",
                        "bridge_specificity": round(norm_spec, 3),
                    }
        return vg_dict

    def _query_multihop_graph(
        self,
        query: str,
        max_hops: int,
        top_k: int,
        use_vector: bool,
        causal_only: bool
    ) -> list[dict]:
        """传统 MemoryGraph BFS 多跳查询"""
        if not self._graph:
            return []

        # 第一跳：语义检索找起始节点
        actual_vector = (use_vector if use_vector is not None else self.enable_vector) and self._embedding is not None

        start_results = self.query(
            query,
            top_k=5,
            use_vector=actual_vector,
            use_keyword=not actual_vector or self.enable_tfidf
        )

        if not start_results:
            start_results = self.query(query, top_k=5, use_vector=False, use_keyword=True)

        if not start_results:
            return []

        start_ids = [r["memory_id"] for r in start_results]

        # BFS多跳遍历
        hop_results = self._graph.bfs_hops(start_ids, max_hops, causal_only=causal_only)

        results = []
        for node_id, hops, path, causal_type, path_conf in hop_results:
            idx = self._memory_map.get(node_id)
            if idx is not None:
                node = self._memories[idx]

                # 得分 = 距离衰减 × 因果权重 × 路径置信度
                causal_weight = 1.5 if causal_type in ["cause", "condition", "explicit"] else 1.0
                base_score = 0.5 ** hops * causal_weight * path_conf

                results.append({
                    "memory_id": node_id,
                    "content": node.content,
                    "score": base_score,
                    "metadata": node.metadata,
                    "hops": hops,
                    "path": path,
                    "causal_type": causal_type,
                    "path_confidence": round(path_conf, 4),
                })

        results.sort(key=lambda x: (x["hops"], -x["score"]))

        return results[:top_k]

    def predict(
        self,
        query: str = None,
        top_k: int = 3,
        metric: str = "activity"
    ) -> dict[str, Any]:
        """
        时序预测接口

        Args:
            query: 当前上下文（用于因果预测）
            top_k: 返回预测数量
            metric: Prediction metric (activity/energy)

        Returns:
            预测结果
        """
        if not self._prediction:
            return {"error": "Prediction module not enabled"}

        predictions = {}

        # 基于当前上下文预测
        if query:
            predictions["event_predictions"] = self._prediction.predict_next_events(query, top_k)
            predictions["causal_predictions"] = self._prediction.get_causal_predictions(query)

        # 时序趋势预测
        predictions["temporal_trend"] = self._prediction.predict_temporal_trend(metric)

        # 记录推理步骤
        self._explainability.record_reasoning_step(
            "reasoning",
            f"时序预测: {query or 'general'}",
            {"predictions": len(predictions)}
        )

        return predictions

    def explain_query(
        self,
        query: str,
        results: list[dict] = None,
        top_k: int = 5
    ) -> dict[str, Any]:
        """
        查询可解释性接口

        Args:
            query: 查询文本
            results: 查询结果（可选，会自动执行查询）
            top_k: 解释的TopK结果

        Returns:
            可解释性报告
        """
        if not self._explainability:
            return {"error": "Explainability module not enabled"}

        # 如果没有提供结果，执行查询
        if results is None:
            results = self.query(query, top_k=top_k * 2)

        # 生成解释报告
        report = self._explainability.explain_query(query, results[:top_k])

        # 生成推理树
        report["reasoning_tree"] = self._explainability.visualize_reasoning_tree(query, results[:top_k])

        # 记录推理步骤
        self._explainability.record_reasoning_step(
            "reasoning",
            f"生成解释: {query}",
            {"result_count": len(results)}
        )

        return report

    def explain_multihop(self, path: list[str]) -> dict[str, Any]:
        """
        解释多跳推理路径

        Args:
            path: 推理路径（memory_id列表）

        Returns:
            多跳解释报告
        """
        if not self._explainability:
            return {"error": "Explainability module not enabled"}

        if len(path) < 2:
            return {"error": "Path too short"}

        return self._explainability.explain_multihop(path[0], path[-1], path)

    def analyze_memory_ecology(self) -> dict[str, Any]:
        """
        Analyze the energy balance and health of the memory ecosystem.

        Returns a report with:
        - balance: overall energy balance status
        - distribution: energy type ratios across all memories
        - dominant: the dominant energy type
        - node_count: total memories analyzed
        - bus_state: EnergyBus network state (if available)
        - suggestions: balance improvement suggestions
        """
        report = {"node_count": len(self._memories)}

        # Energy distribution from all memories
        distribution = {"wood": 0, "fire": 0, "earth": 0, "metal": 0, "water": 0}
        for node in self._memories:
            et = getattr(node, 'energy_type', 'earth')
            if et in distribution:
                distribution[et] += 1

        total = sum(distribution.values()) or 1
        ratios = {k: v / total for k, v in distribution.items()}

        report["distribution"] = distribution
        report["ratios"] = ratios
        report["dominant"] = max(distribution, key=distribution.get)

        # Balance analysis via _energy_relations
        try:
            from su_memory._sys._energy_relations import analyze_balance
            balance = analyze_balance(ratios)
            report["balance"] = balance
        except Exception:
            report["balance"] = {"status": "unknown"}

        # EnergyCore pattern detection
        if self._energy_core is not None:
            try:
                pattern_result = self._energy_core.analyze_balance(ratios)
                report["pattern"] = pattern_result.to_dict() if hasattr(pattern_result, 'to_dict') else str(pattern_result)
            except Exception as e:
                logger.debug("降级处理: %s", e)

        # EnergyBus network state
        if self._energy_bus is not None:
            try:
                report["bus_state"] = self._energy_bus.get_bus_state()
            except Exception as e:
                logger.debug("降级处理: %s", e)

        # Suggestions based on imbalances
        suggestions = []
        for et, ratio in ratios.items():
            if ratio > 0.40:
                suggestions.append(f"Energy {et} is dominant ({ratio:.0%}). Consider adding variety.")
            elif ratio < 0.05 and distribution[et] == 0:
                suggestions.append(f"Energy {et} is missing. Consider adding {et}-type memories.")

        report["suggestions"] = suggestions
        return report

    def link_by_energy(self, source_id: str, target_id: str) -> tuple:
        """
        Create an energy-weighted link between two memories.

        The link weight is automatically adjusted based on the energy
        relationship between the two memories:
        - ENHANCE: weight × 1.2 (source enhances target)
        - SUPPRESS: weight × 0.8 (source suppresses target)
        - SAME: weight × 1.1 (same energy type)
        - NEUTRAL: weight × 1.0 (no direct relation)

        Returns (success: bool, weight: float).
        """
        source_node = self._memory_map.get(source_id)
        target_node = self._memory_map.get(target_id)
        if source_node is None or target_node is None:
            return False, 0.0

        src = self._memories[source_node]
        tgt = self._memories[target_node]

        src_energy = getattr(src, 'energy_type', 'earth')
        tgt_energy = getattr(tgt, 'energy_type', 'earth')

        from su_memory._sys._energy_relations import analyze_relation, calculate_link_weight
        weight = calculate_link_weight(src_energy, tgt_energy, base_weight=1.0)
        relation = analyze_relation(src_energy, tgt_energy)

        # Add to MemoryGraph with energy-weighted edge
        # energy-based link = semantic 成色 (基于能量亲和度计算, 非 keyword 猜测)
        self._graph.add_edge(
            parent_id=source_id, child_id=target_id,
            causal_type=f"energy_{relation.relation.value}",
            confidence=max(0.5, min(weight / 1.5, 0.8)),  # weight 0.8~1.5 → conf 0.5~0.8
            evidence_type="semantic",
        )

        # Also register in causal engine if available
        if self._causal is not None:
            try:
                self._causal.add_node(source_id, src.content, energy_type=src_energy)
                self._causal.add_node(target_id, tgt.content, energy_type=tgt_energy)
                self._causal.link(source_id, target_id, base_weight=1.0, use_energy=True)
            except Exception as e:
                logger.debug("降级处理: %s", e)

        return True, weight

    def auto_link_by_energy(self, threshold: float = 0.5) -> int:
        """
        Automatically discover and create energy-based links between all memories.

        Scans all memory pairs and creates links when the energy affinity
        exceeds the threshold. Returns number of links created.

        Affinity scores:
        - ENHANCE: 1.5 (always linked)
        - SAME: 1.2 (always linked)
        - SUPPRESS: 0.6 (linked if threshold <= 0.6)
        - REVERSE: 0.3 (never linked)
        """
        from su_memory._sys._energy_relations import get_affinity_score

        link_count = 0
        n = len(self._memories)
        if n < 2:
            return 0

        for i in range(n):
            for j in range(i + 1, n):
                src = self._memories[i]
                tgt = self._memories[j]

                src_energy = getattr(src, 'energy_type', 'earth')
                tgt_energy = getattr(tgt, 'energy_type', 'earth')

                # Check forward affinity
                fwd_affinity = get_affinity_score(src_energy, tgt_energy)
                if fwd_affinity >= 1.2:  # ENHANCE or SAME
                    self.link_by_energy(src.id, tgt.id)
                    link_count += 1

                # Check reverse affinity
                rev_affinity = get_affinity_score(tgt_energy, src_energy)
                if rev_affinity >= 1.2:
                    self.link_by_energy(tgt.id, src.id)
                    link_count += 1

        return link_count

    # ═══════════════════════════════════════════════════════════════
    # Layer 2: Knowledge Distillation
    # ═══════════════════════════════════════════════════════════════

    def distill_patterns(self) -> dict[str, Any]:
        """
        Distill common patterns from memory clusters grouped by energy type.

        Returns a report with pattern summaries for each energy category,
        cluster sizes, and common keyword overlaps.
        """
        from collections import Counter

        clusters = {"wood": [], "fire": [], "earth": [], "metal": [], "water": []}
        for node in self._memories:
            et = getattr(node, 'energy_type', 'earth')
            if et in clusters:
                clusters[et].append(node)

        patterns = {}
        for energy_type, nodes in clusters.items():
            if len(nodes) < 2:
                continue

            # Find common keywords across this cluster
            keyword_counter = Counter()
            for n in nodes:
                for kw in n.keywords:
                    keyword_counter[kw] += 1

            # Find keywords appearing in >50% of nodes
            threshold = max(1, len(nodes) // 2)
            common_kws = [kw for kw, count in keyword_counter.most_common(10)
                         if count >= threshold]

            patterns[energy_type] = {
                "size": len(nodes),
                "common_keywords": common_kws[:5],
                "sample_contents": [n.content[:60] for n in nodes[:3]],
                "dominant_themes": self._extract_themes(common_kws, energy_type),
            }

        return {
            "patterns": patterns,
            "cluster_count": len([p for p in patterns.values() if p["size"] > 0]),
            "total_memories": len(self._memories),
        }

    def _extract_themes(self, keywords: list, energy_type: str) -> list:
        """Heuristic theme extraction from common keywords."""
        theme_map = {
            "wood": {"growth": "expansion", "spring": "renewal", "green": "nature",
                     "east": "direction", "tree": "nature", "forest": "nature"},
            "fire": {"heat": "energy", "summer": "season", "red": "color",
                     "passion": "emotion", "south": "direction"},
            "earth": {"stability": "balance", "center": "position", "yellow": "color",
                      "ground": "foundation", "soil": "nature"},
            "metal": {"structure": "order", "autumn": "season", "white": "color",
                      "west": "direction", "precision": "quality"},
            "water": {"wisdom": "knowledge", "winter": "season", "blue": "color",
                      "north": "direction", "flow": "movement"},
        }
        themes = set()
        et_themes = theme_map.get(energy_type, {})
        for kw in keywords:
            if kw in et_themes:
                themes.add(et_themes[kw])
        return sorted(themes)[:5]

    def extract_rules(self, min_cluster_size: int = 3) -> list[dict]:
        """
        Extract general rules from memory clusters.

        Rules are derived from: energy type × common keyword overlap.
        Returns a list of rule dicts with energy, pattern, and confidence.
        """
        patterns = self.distill_patterns()
        rules = []

        for energy_type, info in patterns.get("patterns", {}).items():
            if info["size"] < min_cluster_size:
                continue

            kws = info["common_keywords"]
            themes = info.get("dominant_themes", [])

            if kws:
                rules.append({
                    "energy": energy_type,
                    "pattern": f"{energy_type}-type memories ({info['size']} items)",
                    "keywords": kws,
                    "themes": themes,
                    "confidence": min(1.0, info["size"] / 10),
                    "sample": info["sample_contents"][0] if info["sample_contents"] else "",
                })

        return sorted(rules, key=lambda r: -r["confidence"])

    # ═══════════════════════════════════════════════════════════════
    # Layer 3: Memory Routing
    # ═══════════════════════════════════════════════════════════════

    def route_memory(self, content: str) -> dict[str, Any]:
        """
        Route a new memory to the appropriate energy cluster.

        Uses energy inference + affinity scoring against existing clusters
        to determine the best routing destination.
        """
        energy_type = self._infer_energy(content)

        # Check affinity with existing clusters
        from su_memory._sys._energy_relations import get_affinity_score

        cluster_affinities = {}
        for node in self._memories:
            et = getattr(node, 'energy_type', 'earth')
            score = get_affinity_score(energy_type, et)
            cluster_affinities[et] = max(cluster_affinities.get(et, 0), score)

        best_cluster = max(cluster_affinities, key=cluster_affinities.get) if cluster_affinities else energy_type

        return {
            "energy": energy_type,
            "routed_to": best_cluster,
            "affinity_score": cluster_affinities.get(best_cluster, 1.0),
            "cluster_sizes": {
                et: sum(1 for n in self._memories if getattr(n, 'energy_type', 'earth') == et)
                for et in ("wood", "fire", "earth", "metal", "water")
            }
        }

    def get_importance_scores(self) -> dict[str, float]:
        """
        Calculate importance scores for all memories.

        Score factors:
        - Query frequency (from cache hits)
        - Recency (newer = higher base)
        - Energy type balance contribution
        """
        scores = {}
        now = int(time.time())

        for node in self._memories:
            score = 0.5  # Base importance

            # Recency bonus
            age_days = (now - node.timestamp) / 86400
            score += max(0, 0.3 * (1 - age_days / 365))

            # Query frequency bonus (heuristic)
            if hasattr(self, '_query_stats'):
                score += self._query_stats.get(node.id, 0) * 0.1

            scores[node.id] = round(score, 3)

        return scores

    # ═══════════════════════════════════════════════════════════════
    # Layer 4: Self-Reflection
    # ═══════════════════════════════════════════════════════════════

    def reflect_and_optimize(self) -> dict[str, Any]:
        """
        Periodic self-reflection: audit memory quality and suggest optimizations.

        Checks: energy balance, stale memories, cluster health, link density.
        Returns a health report with actionable suggestions.
        """
        n = len(self._memories)
        if n == 0:
            return {"health_score": 100, "suggestions": [], "memory_count": 0}

        suggestions = []
        health_deductions = 0

        # Check 1: Energy balance
        eco = self.analyze_memory_ecology()
        balance = eco.get("balance", {})
        if balance.get("status") == "concentrated":
            suggestions.append(
                f"Energy concentrated on {balance.get('dominant', '?')}. "
                f"Add variety to improve retrieval diversity."
            )
            health_deductions += 15

        # Check 2: Stale memories (unqueried for >30 days)
        now = int(time.time())
        stale_count = sum(1 for n in self._memories if (now - n.timestamp) > 86400 * 30)
        if stale_count > n * 0.5:
            suggestions.append(f"{stale_count}/{n} memories are stale (>30 days). Consider decay.")
            health_deductions += 10

        # Check 3: Orphan nodes (no energy links)
        orphan_count = sum(1 for n in self._memories
                         if len(n.parent_ids) == 0 and len(n.child_ids) == 0)
        if orphan_count > n * 0.7 and n > 3:
            suggestions.append(f"{orphan_count}/{n} memories have no links. Run auto_link_by_energy().")
            health_deductions += 10

        # Check 4: Missing energy types
        eco_dist = eco.get("distribution", {})
        missing = [et for et, count in eco_dist.items() if count == 0]
        if missing:
            suggestions.append(f"Missing energy types: {missing}. Add diverse content.")
            health_deductions += len(missing) * 5

        health_score = max(0, 100 - health_deductions)

        return {
            "health_score": health_score,
            "suggestions": suggestions,
            "memory_count": n,
            "stale_count": stale_count,
            "orphan_count": orphan_count,
            "ecology": eco,
        }

    def evolution_pipeline(self) -> dict[str, Any]:
        """
        Full evolution pipeline: distill → route → reflect → optimize.

        Runs all four layers in sequence and returns a consolidated report.
        This is the entry point for the AGI-level continuous learning loop.
        """
        result = {"success": True}

        # Step 1: Distill patterns (Layer 2)
        try:
            patterns = self.distill_patterns()
            result["distilled_patterns"] = patterns
        except Exception as e:
            result["distilled_patterns"] = {"error": str(e)}

        # Step 2: Extract rules (Layer 2)
        try:
            rules = self.extract_rules(min_cluster_size=2)
            result["rules_extracted"] = len(rules)
        except Exception:
            result["rules_extracted"] = 0

        # Step 3: Auto-link by energy (Layer 3)
        try:
            links = self.auto_link_by_energy()
            result["routing_suggestions"] = links
        except Exception:
            result["routing_suggestions"] = 0

        # Step 4: Reflect and optimize (Layer 4)
        try:
            reflection = self.reflect_and_optimize()
            result["reflection_report"] = reflection
        except Exception as e:
            result["reflection_report"] = {"error": str(e)}

        return result

    def get_reasoning_summary(self) -> dict[str, Any]:
        """
        获取推理过程摘要

        Returns:
            推理摘要
        """
        if not self._explainability:
            return {"error": "Explainability module not enabled"}

        return self._explainability.get_reasoning_summary()

    def get_memory(self, memory_id: str) -> dict | None:
        """获取单条记忆"""
        idx = self._memory_map.get(memory_id)
        if idx is None:
            return None

        node = self._memories[idx]
        return {
            "memory_id": node.id,
            "content": node.content,
            "metadata": node.metadata,
            "timestamp": node.timestamp,
            "energy_type": node.energy_type,
            "parent_ids": node.parent_ids,
            "child_ids": node.child_ids
        }

    def get_children(self, memory_id: str) -> list[dict]:
        """获取子记忆"""
        if not self._graph:
            return []

        child_ids = self._graph.get_children(memory_id)
        results = []
        for cid in child_ids:
            mem = self.get_memory(cid)
            if mem:
                results.append(mem)
        return results

    def get_parents(self, memory_id: str) -> list[dict]:
        """获取父记忆"""
        if not self._graph:
            return []

        parent_ids = self._graph.get_parents(memory_id)
        results = []
        for pid in parent_ids:
            mem = self.get_memory(pid)
            if mem:
                results.append(mem)
        return results

    def link_memories(self, parent_id: str, child_id: str):
        """链接两条记忆 (用户显式声明 = explicit 成色 0.95)。

        显式声明的因果链在多跳检索中不会被置信度剪枝误杀。
        """
        if not self._graph:
            return

        self._graph.add_edge(
            parent_id, child_id,
            causal_type="explicit",
            confidence=0.95, evidence_type="explicit",
        )

        idx_p = self._memory_map.get(parent_id)
        idx_c = self._memory_map.get(child_id)

        if idx_p is not None and idx_c is not None:
            self._memories[idx_p].child_ids.append(child_id)
            self._memories[idx_c].parent_ids.append(parent_id)

        self._save()

    def count(self) -> int:
        """获取记忆总数"""
        return len(self._memories)

    def get_stats(self) -> dict:
        """获取统计信息"""
        cache_total = self._cache_hits + self._cache_misses
        return {
            "total_memories": len(self._memories),
            "max_memories": self.max_memories,
            "index_size": len(self._index),
            "graph_nodes": len(self._graph._nodes) if self._graph else 0,
            "sessions": len(self._sessions._sessions) if self._sessions else 0,
            "vector_enabled": self.enable_vector,
            "cache_size": len(self._query_cache),
            "cache_hit_rate": self._cache_hits / cache_total if cache_total > 0 else 0
        }

    def _maybe_save(self):
        """节流持久化：变更累计到 save_interval 时落盘，避免 O(n) 写放大。

        持久化失败（磁盘满/权限/IO错误）时记 error 日志但不中断写入——
        数据保留在内存中仍可用，下次 _maybe_save 会重试落盘。
        """
        self._dirty = True
        self._dirty_count += 1
        if self._autosave and self._dirty_count >= self._save_interval:
            try:
                self._save()
                self._dirty = False
                self._dirty_count = 0
            except Exception as e:
                logger.error("持久化失败，数据保留内存待重试: %s", e)
                # 不重置 _dirty_count，下次 add 会再次尝试落盘

    def flush(self):
        """显式将所有未落盘变更写入磁盘。生产环境建议在关键检查点调用。"""
        if self._dirty:
            self._save()
            self._dirty = False
            self._dirty_count = 0

    def close(self):
        """关闭客户端，确保数据落盘。"""
        self.flush()

    def __del__(self):
        try:
            self.flush()
        except Exception:
            pass

    def _save(self):
        if not self.storage_path:
            return

        with self._save_lock:
            return self._save_locked()

    def _save_locked(self):
        """实际持久化逻辑（已持锁）。"""
        os.makedirs(self.storage_path, exist_ok=True)
        path = os.path.join(self.storage_path, "su_memory_pro.json")

        # 简化持久化（不保存embedding）
        data = {
            "memories": [
                {
                    "id": n.id,
                    "content": n.content,
                    "metadata": n.metadata,
                    "keywords": n.keywords,
                    "timestamp": n.timestamp,
                    "parent_ids": n.parent_ids,
                    "child_ids": n.child_ids,
                    "energy_type": n.energy_type
                }
                for n in self._memories
            ],
            "graph": {
                "edges": [
                    (p, c)
                    for node in self._memories
                    for p in node.parent_ids
                    for c in node.child_ids
                ]
            } if self._graph else {}
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 同步持久化 FAISS 索引 + id 映射（此前遗漏，导致重启后向量索引丢失）
        try:
            self._save_faiss_index()
        except Exception as e:
            logger.debug("降级处理: %s", e)

    def _load(self):
        if not self.storage_path:
            return

        path = os.path.join(self.storage_path, "su_memory_pro.json")
        if not os.path.exists(path):
            return

        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)

            for mem_data in data.get("memories", []):
                node = MemoryNode(
                    id=mem_data["id"],
                    content=mem_data["content"],
                    metadata=mem_data.get("metadata", {}),
                    keywords=mem_data.get("keywords", []),
                    timestamp=mem_data.get("timestamp", 0),
                    parent_ids=mem_data.get("parent_ids", []),
                    child_ids=mem_data.get("child_ids", []),
                    energy_type=mem_data.get("energy_type", "earth")
                )
                self._memories.append(node)
                self._memory_map[node.id] = len(self._memories) - 1

                if self._graph:
                    self._graph.add_node(node)

                for kw in node.keywords:
                    self._index[kw].add(node.id)

            # 重建图谱边 (parent_ids = 用户显式声明 = explicit 成色)
            if self._graph:
                for node in self._memories:
                    for parent_id in node.parent_ids:
                        self._graph.add_edge(
                            parent_id, node.id,
                            causal_type="explicit",
                            confidence=0.95, evidence_type="explicit",
                        )

        except Exception as e:
            logger.debug(f"Load failed: {e}")

    def clear(self):
        """清空所有记忆"""
        self._memories.clear()
        self._memory_map.clear()
        self._index.clear()
        self._query_cache.clear()

        if self._graph:
            self._graph = MemoryGraph()

        if self._sessions:
            self._sessions = SessionManager(self.storage_path)

        self._save()

    # ==================== 双轨分类: 软分类 introspection ====================

    def get_topics(self) -> list[dict]:
        """获取所有自动发现的主题簇 (软分类)。

        与硬分类 (energy_type / algebra 能量维度) 正交:
        - 硬分类: 记忆的结构性归属 (语义/因果/时空/生成/信任), 稳定可解释
        - 软分类: 记忆的主题归属 (从关键词相似度涌现), 动态自适应

        Returns
        -------
        list[dict]
            每个主题: {cluster_id, label, size, centroid_keywords}
        """
        if not hasattr(self, '_topic_clusterer'):
            return []
        return self._topic_clusterer.get_topics()

    # ==================== 超长记忆: 语义归纳 ====================

    def consolidate(
        self,
        similarity_threshold: float = 0.6,
        min_cluster_size: int = 5,
        max_clusters: int = 50,
    ) -> dict[str, Any]:
        """语义归纳: 将大量同主题细节记忆合并为摘要记忆。

        对抗"高维诅咒"的核心机制——当记忆数超过阈值, embedding 空间出现
        假近邻, 检索信噪比坍塌。归纳把 N 条细节压缩成 1 条摘要, 让检索
        先命中摘要 (信号强), 需要细节时再回溯原文。

        实现: 用关键词 Jaccard 相似度做轻量聚类 (无重型依赖), 每簇生成
        一条摘要记忆 (取最长内容为代表 + 标注成员数), 原始记忆标记为
        ``_consolidated=True`` 但不删除 (可回溯)。

        Parameters
        ----------
        similarity_threshold : float
            关键词 Jaccard 相似度高于此值的记忆归入同簇。
        min_cluster_size : int
            少于此数的簇不归纳 (不值得合并)。
        max_clusters : int
            最多生成的摘要记忆数 (防止过度归纳)。

        Returns
        -------
        dict
            {consolidated_clusters, summary_memories_created, details_archived}
        """
        if len(self._memories) < min_cluster_size:
            return {"consolidated_clusters": 0, "summary_memories_created": 0,
                    "details_archived": 0, "note": "记忆数不足, 无需归纳"}

        # 1. 构建关键词集合 (跳过已归纳的)
        candidates = [
            (node.id, set(node.keywords), node)
            for node in self._memories
            if not node.metadata.get("_consolidated")
        ]
        if len(candidates) < min_cluster_size:
            return {"consolidated_clusters": 0, "summary_memories_created": 0,
                    "details_archived": 0, "note": "候选记忆不足"}

        # 2. 贪心聚类: 按 Jaccard 相似度归簇
        clusters: list[list[tuple[str, MemoryNode]]] = []
        used = set()
        for mid, kws, node in candidates:
            if mid in used:
                continue
            cluster = [(mid, node)]
            used.add(mid)
            for other_mid, other_kws, other_node in candidates:
                if other_mid in used:
                    continue
                # Jaccard 相似度
                union = kws | other_kws
                if not union:
                    continue
                sim = len(kws & other_kws) / len(union)
                if sim >= similarity_threshold:
                    cluster.append((other_mid, other_node))
                    used.add(other_mid)
            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)

        # 限制簇数
        clusters = clusters[:max_clusters]

        # 3. 每簇生成摘要记忆
        summary_created = 0
        details_archived = 0
        for cluster in clusters:
            members = [(mid, node) for mid, node in cluster]
            # 摘要内容: 取最长的那条 + 标注成员数
            best = max(members, key=lambda x: len(x[1].content))
            member_ids = [mid for mid, _ in members]
            summary_content = (
                f"[归纳·{len(members)}条] {best[1].content}"
            )

            summary_id = self.add(
                summary_content,
                metadata={
                    "_consolidated_summary": True,
                    "_member_ids": member_ids,
                    "_cluster_size": len(members),
                },
                topic="consolidated",
            )

            # 标记原始记忆为已归纳 (不删除, 可回溯)
            for mid, node in members:
                node.metadata["_consolidated"] = True
                node.metadata["_summary_id"] = summary_id
                details_archived += 1

            summary_created += 1

        self._save()
        return {
            "consolidated_clusters": len(clusters),
            "summary_memories_created": summary_created,
            "details_archived": details_archived,
        }

    # ==================== V2.0 Energy Engine API ====================

    def query_energy(self, energy_type: str = None, limit: int = 10) -> list:
        """Query memories filtered by energy type.

        Args:
            energy_type: Filter by energy (wood/fire/earth/metal/water). None = all.
            limit: Max results

        Returns:
            List of memory nodes with energy metadata
        """
        results = []
        for node in self._memories:
            et = getattr(node, 'energy_type', 'earth')
            if energy_type is None or et == energy_type:
                results.append({
                    "id": node.id,
                    "content": node.content,
                    "energy_type": et,
                    "timestamp": node.timestamp
                })
        return results[:limit]

    def reason(self, query: str, max_hops: int = 3) -> dict:
        """Execute multi-layer reasoning on memories using energy dynamics.

        Performs parallel reasoning across energy layers:
        - Layer 1: Direct keyword/vector match
        - Layer 2: Energy propagation (sheng-ke dynamics)
        - Layer 3: Temporal relevance adjustment

        Args:
            query: Query string
            max_hops: Maximum reasoning hops

        Returns:
            Dict with reasoning chain, supporting evidence, and confidence
        """
        import requests

        # Collect relevant memories
        memories = self.query(query, top_k=10)
        query_energy = self._infer_energy(query)

        # Build reasoning context
        ctx_parts = [f"Query: {query}\nQuery energy type: {query_energy}\n\nRelevant memories:"]
        for i, m in enumerate(memories):
            et = getattr(m, 'energy_type', self._infer_energy(m.content))
            ctx_parts.append(f"[{i}] {m.content} (energy: {et})")

        context = "\n".join(ctx_parts)

        # Try LLM reasoning
        try:
            prompt = (
                "You are a reasoning engine. Analyze the query against the memories below. "
                "Consider how different energy types interact (generating/controlling cycles). "
                "Provide:\\n"
                "1. Direct matches\\n"
                "2. Inferred connections via energy dynamics\\n"
                "3. Temporal relevance notes\\n\\n"
                f"{context}\\n\\n"
                "Output JSON with keys: direct_matches (list of indices), "
                "inferred_connections (list of {from, to, reason}), "
                "confidence (0-1), summary (string)"
            )

            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen3.5:9b-nothink",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 500},
                    "raw": True
                },
                timeout=30
            )
            if resp.status_code == 200:
                data = resp.json()
                text = data.get("response") or data.get("thinking", "")
                import json as _json
                try:
                    result = _json.loads(text)
                    return result
                except Exception as e:
                    logger.debug("降级处理: %s", e)
        except Exception as e:
            logger.debug("能量推理 LLM 降级到关键词: %s", e)

        # Fallback: simple keyword-based reasoning
        return {
            "direct_matches": list(range(min(len(memories), 3))),
            "inferred_connections": [],
            "confidence": 0.3,
            "summary": f"Found {len(memories)} relevant memories (keyword fallback)"
        }

    def diagnose(self) -> dict:
        """Diagnose the energy distribution of the memory system.

        Returns:
            Dict with energy balance analysis, gaps, and suggestions
        """
        from collections import Counter

        if not self._memories:
            return {"balance": "empty", "distribution": {}, "gaps": [], "suggestions": []}

        # Count energy distribution
        counter = Counter()
        for node in self._memories:
            et = getattr(node, 'energy_type', 'earth')
            counter[et] += 1

        total = len(self._memories)
        distribution = {e: round(counter.get(e, 0) / total, 3) for e in
                        ("wood", "fire", "earth", "metal", "water")}

        # Check balance (ideal: each ~20%)
        gaps = []
        for e, pct in distribution.items():
            if pct < 0.10:
                gaps.append(f"{e}: critically low ({pct:.1%})")
            elif pct < 0.15:
                gaps.append(f"{e}: low ({pct:.1%})")

        # Generate suggestions
        suggestions = []
        if gaps:
            suggestions.append("Consider adding memories in under-represented energy types")
        if distribution.get("earth", 0) > 0.40:
            suggestions.append("Memory system is earth-heavy; may benefit from more dynamic content")

        balance = "balanced" if not gaps else "imbalanced"

        return {
            "balance": balance,
            "total_memories": total,
            "distribution": distribution,
            "gaps": gaps,
            "suggestions": suggestions
        }

    def export(self, fmt: str = "jsonl", path: str = None) -> str:
        """Export all memories in specified format.

        Args:
            fmt: Format - 'jsonl', 'markdown', or 'obsidian'
            path: Output file path (optional)

        Returns:
            Exported content as string
        """
        import json as _json

        if fmt == "jsonl":
            lines = []
            for node in self._memories:
                record = {
                    "id": node.id,
                    "content": node.content,
                    "energy_type": getattr(node, 'energy_type', 'earth'),
                    "timestamp": node.timestamp,
                    "metadata": node.metadata if hasattr(node, 'metadata') else {}
                }
                lines.append(_json.dumps(record, ensure_ascii=False))
            output = "\n".join(lines)

        elif fmt == "markdown":
            lines = ["# Memory Export", "", f"Total: {len(self._memories)} memories", ""]
            for node in self._memories:
                et = getattr(node, 'energy_type', 'earth')
                ts = time.strftime('%Y-%m-%d %H:%M', time.localtime(node.timestamp))
                lines.append(f"## [{et}] {node.id}")
                lines.append(f"*{ts}*")
                lines.append("")
                lines.append(node.content)
                lines.append("")
            output = "\n".join(lines)

        elif fmt == "obsidian":
            lines = []
            for node in self._memories:
                et = getattr(node, 'energy_type', 'earth')
                ts = time.strftime('%Y-%m-%d %H:%M', time.localtime(node.timestamp))
                lines.append("---")
                lines.append(f"id: {node.id}")
                lines.append(f"energy: {et}")
                lines.append(f"timestamp: {ts}")
                lines.append("---")
                lines.append("")
                lines.append(node.content)
                lines.append("")
            output = "\n".join(lines)

        else:
            raise ValueError(f"Unknown format: {fmt}")

        if path:
            with open(path, 'w') as f:
                f.write(output)
            return f"Exported to {path}"

        return output

    def __len__(self):
        return len(self._memories)

    def __bool__(self):
        return True
