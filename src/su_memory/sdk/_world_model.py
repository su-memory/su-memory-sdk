"""
su-memory v4.4.1 — MCI World Model JEPA
===========================================

神经-符号因果推理系统的统一接口，
v4.4.1: JEPA 潜空间预测替代 QLoRA/Transformer。
融合三层因果量化管道 + JEPA 编码器-预测器 + Pearl do-calculus 干预 +
Pearl counterfactual 反事实推理 (L3)。

核心能力:
- discover():        三层因果发现 → 加权因果图 → JEPA 编码
- predict_effect():  纯检索路径 + JEPA 预测路径（v4.4.1 统一）
- jepa_predict():    JEPA 潜空间预测 (encoder→state→predictor→next_state)
- intervene():       Pearl do-operator 干预预测（v3.7.0 L2）
- decompose_effect(): 因果效应三分解 NDE/NIE/TE
- query_counterfactual(): Pearl 反事实推理（v3.8.0 L3）
- train_jepa():      JEPA 端到端训练（替代 train_parametric）
- explain():         因果链回溯，人类可读解释
- health_check():    全系统健康诊断

架构层次:
    ┌───────────────────────────────────────────┐
    │        MCIWorldModel (v4.0.0 JEPA)        │
    │  ┌───────────────────────────────────┐    │
    │  │  JEPA Encoder + Predictor         │    │
    │  │  (潜空间因果图编码 → GNN/基线预测) │    │
    │  │  + EnergyConsistencyLoss          │    │
    │  └──────────┬────────────────────────┘    │
    │             │ 潜空间状态编码                │
    │  ┌──────────▼────────────────────────┐    │
    │  │  三层因果管道                     │    │
    │  │  FourierCausal → GaussianDAG     │    │
    │  │  → BayesianCausal                │    │
    │  └───────────────────────────────────┘    │
    │  ┌───────────────────────────────────┐    │
    │  │  Entity Surfacing + SIGReg        │    │
    │  └───────────────────────────────────┘    │
    └───────────────────────────────────────────┘

用法:
    from su_memory.sdk._world_model import MCIWorldModel

    wm = MCIWorldModel(su_lite_pro_instance)
    causal_graph = wm.discover()
    effects = wm.predict_effect("价格上涨")
    jepa_effects = wm.jepa_predict("价格上涨")
    explanation = wm.explain("为什么库存下降?")
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CausalWorldModelState
# =============================================================================


@dataclass
class CausalWorldModelState:
    """
    因果世界模型状态 — 事实世界 + 反事实世界双图结构。

    事实世界 G: 观测到的因果关系图
    反事实世界 G_not_X: 干预后的反事实图
    """

    # ── 因果图 ──
    causal_edges: list[dict] = field(default_factory=list)
    # [{"cause": str, "effect": str, "rho": float, "confidence": float,
    #   "verdict": str, "energy_relation": str, "bayes_factor": float}, ...]

    # ── 状态覆盖 ──
    active_states: set[str] = field(default_factory=set)
    # 当前活跃的五范畴状态

    # ── 置信度统计 ──
    n_confirmed: int = 0
    n_novel: int = 0
    n_suppressed: int = 0

    # ── 元信息 ──
    n_memories: int = 0
    n_qa_pairs: int = 0
    parametric_enhanced: bool = False
    timestamp: str = ""

    # ── 反事实世界（v3.8.0 L3）─
    counterfactual_graph: dict | None = None
    do_interventions: list[dict] = field(default_factory=list)

    # ── v4.0.0 JEPA: 时空 + 信念 + 元认知元数据 ──
    temporal_info: object | None = None
    # TemporalInfo from _sys/chrono.py(stem-branch support info)
    belief_tracker: object | None = None
    # BayesianBeliefTracker from _sys/states.py（信念演化）
    cognitive_gaps: list = field(default_factory=list)
    # list[CognitiveGap] from _sys/awareness.py（认知空洞）

    @classmethod
    def empty(cls) -> CausalWorldModelState:
        return cls()

    def to_dict(self) -> dict:
        return {
            "n_causal_edges": len(self.causal_edges),
            "n_confirmed": self.n_confirmed,
            "n_novel": self.n_novel,
            "n_suppressed": self.n_suppressed,
            "active_states": list(self.active_states),
            "n_memories": self.n_memories,
            "n_qa_pairs": self.n_qa_pairs,
            "parametric_enhanced": self.parametric_enhanced,
            "timestamp": self.timestamp,
            "has_counterfactual_graph": self.counterfactual_graph is not None,
            "n_do_interventions": len(self.do_interventions),
            "has_temporal_info": self.temporal_info is not None,
            "has_belief_tracker": self.belief_tracker is not None,
            "n_cognitive_gaps": len(self.cognitive_gaps),
        }

    # ────────────────────────────────────────────────
    # v4.0.0 JEPA: 因果图距离度量
    # ────────────────────────────────────────────────

    def _build_node_index(self) -> dict[str, int]:
        """从 causal_edges 中提取所有唯一节点并建立索引。

        支持两种边格式:
        - entity-level: cause="entity_name", effect="entity_name"
        - memory-level: cause_idx=0, effect_idx=1 (GaussianDAG 输出)
        """
        nodes: dict[str, int] = {}
        has_named_edges = False
        for e in self.causal_edges:
            for key in ("cause", "effect"):
                name = str(e.get(key, ""))
                if name and name not in nodes:
                    nodes[name] = len(nodes)
                    has_named_edges = True
        # Fallback: index-based edges from GaussianDAG
        if not has_named_edges:
            for e in self.causal_edges:
                for key in ("cause_idx", "effect_idx"):
                    idx = e.get(key, -1)
                    if idx >= 0:
                        name = f"n{idx}"
                        if name not in nodes:
                            nodes[name] = len(nodes)
        return nodes

    def _get_node_name(self, e: dict, key: str) -> str:
        """从边中提取节点名称，兼容 cause/effect 和 cause_idx/effect_idx 两种格式。"""
        name = str(e.get(key, ""))
        if name:
            return name
        idx_key = f"{key}_idx"
        idx = e.get(idx_key, -1)
        if idx >= 0:
            return f"n{idx}"
        return ""

    def to_adjacency_matrix(self) -> np.ndarray:
        """
        构建 N×N 加权邻接矩阵。

        有因果边 → 权重 = rho（偏相关系数）
        无因果边 → 权重 = 0.0
        自环 → 0.0

        Returns:
            shape=(N, N) 的 float32 邻接矩阵
        """
        node_index = self._build_node_index()
        n = len(node_index)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float32)
        adj = np.zeros((n, n), dtype=np.float32)
        for e in self.causal_edges:
            cause_name = self._get_node_name(e, "cause")
            effect_name = self._get_node_name(e, "effect")
            if cause_name in node_index and effect_name in node_index:
                i = node_index[cause_name]
                j = node_index[effect_name]
                adj[i, j] = float(e.get("rho", 0.0))
        return adj

    def to_node_feature_matrix(self) -> np.ndarray:
        """
        构建 N×D 节点特征矩阵。

        特征维度 D = 5 (活跃状态 one-hot) + 3 (度统计)
        - 活跃状态 one-hot (5): semantic/causal/spacetime/generative/trust
        - 出度 (1): 该节点作为 cause 的次数
        - 入度 (1): 该节点作为 effect 的次数
        - 度中心性 (1): (出度+入度) / (2*N)

        Returns:
            shape=(N, 8) 的 float32 特征矩阵
        """
        node_index = self._build_node_index()
        n = len(node_index)
        if n == 0:
            return np.zeros((0, 8), dtype=np.float32)

        features = np.zeros((n, 8), dtype=np.float32)
        five_states = ["semantic", "causal", "spacetime", "generative", "trust"]
        state_to_idx = {s: i for i, s in enumerate(five_states)}

        # 统计度
        out_degree = dict.fromkeys(node_index, 0)
        in_degree = dict.fromkeys(node_index, 0)
        for e in self.causal_edges:
            cause_name = self._get_node_name(e, "cause")
            effect_name = self._get_node_name(e, "effect")
            if cause_name in out_degree:
                out_degree[cause_name] += 1
            if effect_name in in_degree:
                in_degree[effect_name] += 1

        for name, idx in node_index.items():
            d_out = out_degree[name]
            d_in = in_degree[name]
            features[idx, 5] = float(d_out)
            features[idx, 6] = float(d_in)
            features[idx, 7] = (d_out + d_in) / max(2 * n, 1)

            # 活跃状态 one-hot（基于已有的 active_states）
            for state_name, si in state_to_idx.items():
                features[idx, si] = 1.0 if state_name in self.active_states else 0.0

        return features

    def state_distance(
        self,
        other: CausalWorldModelState,
        alpha_edges: float = 0.5,
        alpha_structure: float = 0.3,
        alpha_energy: float = 0.2,
        alpha_temporal: float = 0.15,
        alpha_belief: float = 0.15,
    ) -> float:
        """
        计算两个因果世界状态之间的距离。

        JEPA 训练损失的主项：
            L_pred = state_distance(s_pred, s_actual)

        v4.0.0: 融合因果图距离 + 时空距离 + 信念距离。
            L_total = (α_causal·L_causal + α_temporal·L_temporal + α_belief·L_belief) / Σα

        因果图距离子项：
        1. 边权重 L1 距离 (alpha_edges): 同一条边在两个状态间 rho 的差异
        2. 图结构 Jaccard 差异 (alpha_structure): 边集合的重叠度
        3. 能量守恒差异 (alpha_energy): 因果图总能量变化率

        时空/信念距离（仅在双方均有数据时激活）：
        4. 时空距离 (alpha_temporal): energy_type 不匹配比例
        5. 信念距离 (alpha_belief): 信念轨迹置信度变化

        Args:
            other: 另一个 CausalWorldModelState
            alpha_edges: 边权重距离权重
            alpha_structure: 图结构差异权重
            alpha_energy: 能量守恒差异权重
            alpha_temporal: 时空距离权重
            alpha_belief: 信念距离权重

        Returns:
            0.0 到 1.0 之间的距离标量
        """
        if not self.causal_edges and not other.causal_edges:
            return 0.0
        if not self.causal_edges or not other.causal_edges:
            return 1.0

        # ── 1. 边权重 L1 距离 ──
        self_adj = self.to_adjacency_matrix()
        other_adj = other.to_adjacency_matrix()
        n_max = max(self_adj.shape[0], other_adj.shape[0])
        if self_adj.shape[0] < n_max:
            padded = np.zeros((n_max, n_max), dtype=np.float32)
            padded[: self_adj.shape[0], : self_adj.shape[1]] = self_adj
            self_adj = padded
        if other_adj.shape[0] < n_max:
            padded = np.zeros((n_max, n_max), dtype=np.float32)
            padded[: other_adj.shape[0], : other_adj.shape[1]] = other_adj
            other_adj = padded

        edge_l1 = float(np.sum(np.abs(self_adj - other_adj)))
        total_rho = max(float(np.sum(self_adj) + np.sum(other_adj)), 1e-10)
        dist_edges = min(edge_l1 / total_rho, 1.0)

        # ── 2. 图结构 Jaccard 差异 ──
        self_edges_set = {
            (self._get_node_name(e, "cause"), self._get_node_name(e, "effect"))
            for e in self.causal_edges
        }
        other_edges_set = {
            (other._get_node_name(e, "cause"), other._get_node_name(e, "effect"))
            for e in other.causal_edges
        }
        intersection = len(self_edges_set & other_edges_set)
        union = len(self_edges_set | other_edges_set)
        if union > 0:
            jaccard_sim = intersection / union
            dist_structure = 1.0 - jaccard_sim
        else:
            dist_structure = 0.0

        # ── 3. 能量守恒差异 ──
        self_total_energy = sum(
            abs(e.get("rho", 0.0)) for e in self.causal_edges
        )
        other_total_energy = sum(
            abs(e.get("rho", 0.0)) for e in other.causal_edges
        )
        max_energy = max(self_total_energy, other_total_energy, 1e-10)
        dist_energy = abs(self_total_energy - other_total_energy) / max_energy

        # ── 4. v4.0.0: 时空距离 (energy_type 对齐) ──
        dist_temporal = 0.0
        has_temporal = False
        if self.temporal_info is not None and other.temporal_info is not None:
            has_temporal = True
            try:
                self_et = getattr(self.temporal_info, "energy_type", "")
                other_et = getattr(other.temporal_info, "energy_type", "")
                dist_temporal = 0.0 if self_et == other_et else 1.0
            except Exception:
                pass

        # ── 5. v4.0.0: 信念距离 (置信度轨迹差异) ──
        dist_belief = 0.0
        has_belief = False
        if self.belief_tracker is not None and other.belief_tracker is not None:
            has_belief = True
            try:
                self_states = getattr(self.belief_tracker, "belief_states", {})
                other_states = getattr(other.belief_tracker, "belief_states", {})
                all_keys = set(self_states.keys()) | set(other_states.keys())
                if all_keys:
                    diffs = []
                    for k in all_keys:
                        sc = getattr(self_states.get(k), "confidence", 0.5) if isinstance(self_states.get(k), object) else 0.5
                        oc = getattr(other_states.get(k), "confidence", 0.5) if isinstance(other_states.get(k), object) else 0.5
                        diffs.append(abs(sc - oc))
                    dist_belief = sum(diffs) / len(diffs) if diffs else 0.0
            except Exception:
                pass

        # ── 加权求和 ──
        # 归一化: 只对活跃的组件分配权重
        causal_weight = alpha_edges + alpha_structure + alpha_energy
        total_weight = causal_weight
        distance = (
            alpha_edges * dist_edges
            + alpha_structure * dist_structure
            + alpha_energy * dist_energy
        )
        if has_temporal:
            total_weight += alpha_temporal
            distance += alpha_temporal * dist_temporal
        if has_belief:
            total_weight += alpha_belief
            distance += alpha_belief * dist_belief

        return min(float(distance / total_weight), 1.0)

    def __sub__(self, other: CausalWorldModelState) -> float:
        """
        操作符重载：`distance = abs(s_t1 - s_t)` 返回距离标量。

        等价于 self.state_distance(other)。
        """
        if not isinstance(other, CausalWorldModelState):
            return NotImplemented
        return self.state_distance(other)


# =============================================================================
# MCIWorldModel
# =============================================================================


class MCIWorldModel:
    """
    MCI World Model v4.0.0 JEPA — 神经-符号因果推理系统。

    v4.0.0: JEPA 潜空间预测替代 QLoRA/Transformer。
    统一了检索增强 + JEPA 编码器-预测器两种路径，
    提供 Pearl 因果层级（关联→干预→反事实）的完整接口。

    Example:
        >>> wm = MCIWorldModel(lite_pro)
        >>> graph = wm.discover()
        >>> print(f"发现 {len(graph.causal_edges)} 条因果边")

        >>> # JEPA 潜空间预测
        >>> predictions = wm.jepa_predict("产品价格上涨")
        >>> for p in predictions:
        ...     print(f"→ {p['effect']} (置信度: {p['confidence']})")

        >>> # 干预分析 (v3.7.0 L2)
        >>> result = wm.intervene(
        ...     do_x={"price": 1.5},
        ...     target="demand",
        ... )

        >>> # 反事实推理 (v3.8.0 L3)
        >>> cf = wm.query_counterfactual(
        ...     evidence={"price": 1.0, "demand": 100},
        ...     do_x={"price": 0.8},
        ...     target="demand",
        ... )
    """

    # ── 五范畴状态系统 ──
    FIVE_STATES = ["semantic", "causal", "spacetime", "generative", "trust"]

    def __init__(
        self,
        lite_pro=None,
        config: dict | None = None,
    ):
        """
        Args:
            lite_pro: SuMemoryLitePro 实例（可选）
            config: 配置字典
        """
        self._lite_pro = lite_pro
        self._config = config or {}
        self._state = CausalWorldModelState.empty()
        self._parametric: object | None = None  # 降级为惰性加载 (v4.0.0)
        self._energy_loss: object | None = None  # EnergyConsistencyLoss
        self._initialized: bool = False

        # v4.0.0 JEPA: 编码器 + 预测器 (懒加载)
        self._jepa_encoder: object | None = None
        self._jepa_predictor: object | None = None

        # v3.7.0: do-calculus 干预引擎 (懒加载)
        self._do_calculus: object | None = None
        self._do_calculus_lock: threading.Lock = threading.Lock()
        self._intervention_history: list[dict] = []

        # P1 并发加固: 初始化锁 (防止多线程重复 initialize)
        self._init_lock: threading.Lock = threading.Lock()

        # P2 并发加固: 因果发现锁 (防止多线程同时 discover() 状态交错)
        self._discover_lock: threading.Lock = threading.Lock()

        # 如果传入了 lite_pro，自动初始化
        if lite_pro is not None:
            self.initialize()

    # ────────────────────────────────────────────────
    # 初始化
    # ────────────────────────────────────────────────

    def initialize(self) -> dict:
        """
        初始化世界模型组件（幂等安全）。

        自动检测并组装:
        - 四层因果管道（_spectral_causal）
        - Reflection QA 合成器
        - Entity Surfacing + SIGReg
        - ParametricMemory（按需加载）

        Returns:
            初始化状态报告
        """
        # 幂等: 已初始化则直接返回缓存报告
        if self._initialized:
            return {
                "modules": {"causal_pipeline": "available"},
                "warnings": [],
                "ready": True,
                "initialized": True,
                "_cached": True,
            }

        with self._init_lock:
            # 双重检查
            if self._initialized:
                return {
                    "modules": {"causal_pipeline": "available"},
                    "warnings": [],
                    "ready": True,
                    "initialized": True,
                    "_cached": True,
                }

            report: dict = {
                "modules": {},
                "warnings": [],
                "ready": False,
            }

            # ── 检查四层因果管道 ──
            try:
                from su_memory.sdk._spectral_causal import (  # noqa: F401
                    BayesianCausal,
                    FourierCausal,
                    GaussianDAG,
                )
                report["modules"]["causal_pipeline"] = "available"
            except ImportError:
                report["modules"]["causal_pipeline"] = "unavailable"
                report["warnings"].append("四层因果管道不可用 — 因果发现将受限")

            # ── 检查 Reflection QA ──
            try:
                from su_memory.sdk._reflection_synthesizer import (
                    ReflectionSynthesizer,  # noqa: F401
                )
                report["modules"]["reflection_qa"] = "available"
            except ImportError:
                report["modules"]["reflection_qa"] = "unavailable"

            # ── 检查 SIGReg ──
            try:
                from su_memory.sdk._sigreg import SIGReg  # noqa: F401
                report["modules"]["sigreg"] = "available"
            except ImportError:
                report["modules"]["sigreg"] = "unavailable"

            # ── v4.0.0 JEPA: 检查编码器 ──
            try:
                from su_memory.sdk._jepa_encoder import JEPAEncoder  # noqa: F401
                report["modules"]["jepa_encoder"] = "available"
            except ImportError:
                report["modules"]["jepa_encoder"] = "unavailable"

            # ── v4.0.0 JEPA: 检查预测器 ──
            try:
                from su_memory.sdk._jepa_predictor import (  # noqa: F401
                    BeliefPropagationPredictor,
                )
                report["modules"]["jepa_predictor"] = "available"
            except ImportError:
                report["modules"]["jepa_predictor"] = "unavailable"

            # ── v4.0.0 M2: 检查 GNN 预测器 ──
            try:
                from su_memory.sdk._jepa_gnn import GNNPredictor  # noqa: F401
                report["modules"]["jepa_gnn"] = "available"
            except ImportError:
                report["modules"]["jepa_gnn"] = "unavailable"

            # ── 检查能量损失 ──
            try:
                from su_memory.sdk._energy_loss import (
                    EnergyConsistencyLoss,
                )
                report["modules"]["energy_loss"] = "available"
                self._energy_loss = EnergyConsistencyLoss()
            except ImportError:
                report["modules"]["energy_loss"] = "unavailable"

            # ── v4.0.0: 初始化 JEPA 编码器 ──
            if report["modules"]["jepa_encoder"] == "available":
                try:
                    from su_memory.sdk._jepa_encoder import JEPAEncoder
                    self._jepa_encoder = JEPAEncoder(self)
                    report["jepa_encoder"] = "initialized"
                except Exception as e:
                    report["warnings"].append(f"JEPA 编码器初始化失败: {e}")

            # ── v4.0.0: 初始化 JEPA 预测器（默认为 BeliefPropagation 基线） ──
            if report["modules"]["jepa_predictor"] == "available":
                try:
                    from su_memory.sdk._jepa_predictor import (
                        BeliefPropagationPredictor,
                    )
                    self._jepa_predictor = BeliefPropagationPredictor()
                    report["jepa_predictor"] = "initialized"
                except Exception as e:
                    report["warnings"].append(f"JEPA 预测器初始化失败: {e}")

            report["ready"] = report["modules"]["causal_pipeline"] == "available"
            self._initialized = report["ready"]

            if report["ready"]:
                logger.info("MCIWorldModel v4.0.0 JEPA 初始化完成")
            else:
                logger.warning("MCIWorldModel 初始化不完整: %s", report["warnings"])

            return report

    # ────────────────────────────────────────────────
    # 因果发现
    # ────────────────────────────────────────────────

    def discover(
        self,
        memories: list[dict] | None = None,
        use_parametric: bool = False,
        verbose: bool = True,
    ) -> CausalWorldModelState:
        """
        三层因果发现流水线（线程安全）。

        执行完整流程:
        Layer 1: FourierCausal 频域过滤
        Layer 2: GaussianDAG 偏相关发现
        Layer 3: BayesianCausal 后验量化

        （F1-P1-1: 原 docstring 声称"四层"，但 CausalProbability 未实际导入，修正为三层）

        并发安全 (P2 加固):
        - 使用 ``self._discover_lock`` 序列化对 ``self._state`` 的原地修改
        - 多个线程同时调用 ``discover()`` 不会产生状态交错
        - JEPADataset.from_memories() 在并发场景下应拷贝返回的 state 以避免交叉污染

        Args:
            memories: 记忆列表（None 时从 lite_pro 自动获取）
            use_parametric: 是否启用参数化先验增强
            verbose: 是否输出 INFO 日志（训练时设为 False）

        Returns:
            CausalWorldModelState 含所有发现的因果边 (为 ``self._state`` 引用)
        """
        if memories is None and self._lite_pro is not None:
            memories = self._get_memories_from_lite_pro()

        if not memories or len(memories) < 3:
            logger.warning("记忆不足（需要 ≥ 3 条）")
            return self._state

        # P2 并发加固: 序列化对 self._state 的原地修改
        # 多线程同时调用 discover() 会导致 _state.causal_edges/n_memories 等字段交错
        with self._discover_lock:
            try:
                from su_memory.sdk._spectral_causal import (  # noqa: F401
                    BayesianCausal,
                    FourierCausal,
                    GaussianDAG,
                )

                # ── 获取 TF-IDF 索引 ──
                index = None
                if self._lite_pro and hasattr(self._lite_pro, "_index"):
                    index = self._lite_pro._index

                # ── 获取 EnergyBus ──
                energy_bus = None
                if self._lite_pro and hasattr(self._lite_pro, "_energy_bus"):
                    energy_bus = self._lite_pro._energy_bus

                # ── Layer 1+2: GaussianDAG ──
                dag = GaussianDAG(memories, index, energy_bus)

                # ── v4.4.1: JEPA 先验增强 ──
                if use_parametric and self._jepa_encoder is not None:
                    self._apply_parametric_prior(dag, memories)

                # ── Reflection Prior ──
                try:
                    from su_memory.sdk._reflection_synthesizer import (
                        ReflectionSynthesizer,  # noqa: F401
                    )
                    syn = ReflectionSynthesizer(
                        energy_bus=energy_bus,
                        min_confidence=0.4,
                        max_pairs=200,
                    )
                    _, prior_matrix = syn.run_pipeline(memories)
                    dag.with_reflection_prior(prior_matrix)
                except ImportError:
                    pass

                # ── 发现隐藏因果边 ──
                edges = dag.discover_hidden_edges()

                # ── v4.4.1: 补充 cause/effect 实体名称 ──
                # GaussianDAG 输出边使用 TF-IDF 词表索引 (cause_idx/effect_idx)，
                # 后续代码 (BayesianCausal) 依赖这些索引。同时补充 cause/effect
                # 实体名称，使 GAT 编码器和 align_adjacency 能正确对齐。
                if hasattr(dag, "_vocab") and dag._vocab:
                    for e in edges:
                        ci = e.get("cause_idx")
                        ei = e.get("effect_idx")
                        if ci is not None and ci < len(dag._vocab):
                            e["cause"] = dag._vocab[ci]
                        if ei is not None and ei < len(dag._vocab):
                            e["effect"] = dag._vocab[ei]

                # ── Layer 3: BayesianCausal 量化 ──
                bayesian = BayesianCausal(energy_bus)
                edges = bayesian.batch_update(edges)

                # ── 更新状态 ──
                self._state.causal_edges = edges
                self._state.n_memories = len(memories)
                self._state.parametric_enhanced = use_parametric

                # ── 统计 ──
                self._state.n_confirmed = sum(
                    1 for e in edges if e.get("verdict") == "confirmed"
                )
                self._state.n_novel = sum(
                    1 for e in edges if e.get("verdict") == "novel"
                )
                self._state.n_suppressed = sum(
                    1 for e in edges if e.get("verdict") == "suppressed"
                )

                # ── 活跃状态 ──
                active = set()
                for e in edges:
                    if e.get("energy_relation"):
                        active.add(e["energy_relation"])
                self._state.active_states = active

                from datetime import datetime
                self._state.timestamp = datetime.now().isoformat()

                if verbose:
                    logger.info(
                        "因果发现完成: %d 条边 (确认: %d, 新发现: %d, 抑制: %d)",
                        len(edges), self._state.n_confirmed,
                        self._state.n_novel, self._state.n_suppressed,
                    )

            except ImportError as e:
                logger.error("因果发现失败 — 缺少依赖: %s", e)
            except Exception as e:
                logger.error("因果发现失败: %s", e)

        return self._state

    # ────────────────────────────────────────────────
    # 因果预测
    # ────────────────────────────────────────────────

    def predict_effect(
        self,
        cause: str,
        memories: list[dict] | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """
        检索路径因果预测（v3.5.0 能力）。

        基于 CausalEngine 关键词 + 偏相关统计，
        不依赖参数化模型。

        Args:
            cause: 原因文本
            memories: 记忆列表
            top_k: 返回前 K 个效应

        Returns:
            [{"effect": str, "confidence": float, "causal_type": str}, ...]
        """
        if memories is None and self._lite_pro is not None:
            memories = self._get_memories_from_lite_pro()

        if not memories:
            return []

        try:
            from su_memory.sdk._causal import CausalEngine
            engine = CausalEngine(min_confidence=0.4)
            effects = engine.predict_effects(cause, memories, top_k=top_k)
            return effects
        except Exception as e:
            logger.error("检索预测失败: %s", e)
            return []

    def jepa_predict(
        self,
        cause: str,
        target_category: str | None = None,
        top_k: int = 3,
        memories: list[dict] | None = None,
    ) -> list[dict]:
        """
        JEPA 潜空间因果预测（v4.4.1）。

        流程: 编码器(记忆 → 因果图状态) → 预测器(状态 → 下一状态) →
              差分分析(原因 → 效应)

        Args:
            cause: 原因文本
            target_category: 目标状态类别（可选）
            top_k: 返回前 K 个预测
            memories: 记忆列表（None 时从 lite_pro 获取）

        Returns:
            [{"effect": str, "confidence": float, "energy_relation": str}, ...]
        """
        if self._jepa_encoder is None or self._jepa_predictor is None:
            logger.warning("JEPA 编码器/预测器未初始化，回退到检索路径")
            return self.predict_effect(cause, top_k=top_k)

        # ── 获取记忆并编码 ──
        if memories is None and self._lite_pro is not None:
            memories = self._get_memories_from_lite_pro()

        if not memories or len(memories) < 3:
            logger.warning("记忆不足 JEPA 预测（需要 ≥ 3 条）")
            return self.predict_effect(cause, top_k=top_k)

        try:
            # 0. 检测 M3 模式
            is_m3 = (self._jepa_encoder is not None and
                     hasattr(self._jepa_encoder, '_differentiable') and
                     self._jepa_encoder._differentiable)

            # 1. 编码: 记忆 → 因果图状态
            state = self._jepa_encoder.encode(memories)

            # 2. 预测: 状态 → 下一状态 (GNN)
            next_state = self._jepa_predictor.predict(state)

            # 3. 差分: 找出新增/增强的因果边
            predictions = []
            if state.causal_edges and next_state.causal_edges:
                current_edge_keys = {
                    (e.get("cause", ""), e.get("effect", ""))
                    for e in state.causal_edges
                }
                for edge in next_state.causal_edges:
                    ee = edge.get("effect", "")
                    ec = edge.get("cause", "")
                    # 只返回与 cause 相关的新增/变化边
                    if cause.lower() in ec.lower() or cause.lower() in ee.lower():
                        key = (ec, ee)
                        is_new = key not in current_edge_keys
                        predictions.append({
                            "effect": ee,
                            "confidence": edge.get("confidence", 0.5) * (1.1 if is_new else 0.9),
                            "energy_relation": edge.get("energy_relation", "neutral"),
                            "cause": ec,
                            "verdict": edge.get("verdict", "predicted"),
                            "_mode": "m3_gat_gnn" if is_m3 else "jepa_baseline",
                        })

            # 按置信度排序
            predictions.sort(key=lambda x: x["confidence"], reverse=True)
            return predictions[:top_k] if predictions else self.predict_effect(
                cause, top_k=top_k
            )

        except Exception as e:
            logger.error("JEPA 预测失败: %s，回退到检索路径", e)
            return self.predict_effect(cause, top_k=top_k)

    def parametric_predict(
        self,
        cause: str,
        target_category: str | None = None,
        top_k: int = 3,
    ) -> list[dict]:
        """
        参数化路径因果预测（v3.6.0 — v4.4.1 降级为 jepa_predict 别名）。

        v4.4.1: 重路由到 JEPA 潜空间预测。
        保留接口兼容性，内部调用 jepa_predict()。

        Args:
            cause: 原因文本
            target_category: 目标状态类别（可选）
            top_k: 返回前 K 个预测

        Returns:
            [{"effect": str, "confidence": float, "energy_relation": str}, ...]
        """
        return self.jepa_predict(cause, target_category, top_k)

    def predict_from_memories_m3(
        self,
        memories: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        M3 专用推理：从记忆列表直接预测因果边（GAT + GNN）。

        与 jepa_predict() 的区别:
        - 不依赖 cause 文本过滤，返回所有预测的因果边
        - 直接输出 GNN 预测的 (cause, effect, rho) 三元组
        - 可用于端到端评估和可视化

        Args:
            memories: 记忆列表（至少 3 条）
            top_k: 返回前 K 条最显著的因果边

        Returns:
            [{"cause": str, "effect": str, "rho": float, "confidence": float}, ...]
        """
        if self._jepa_encoder is None or self._jepa_predictor is None:
            logger.warning("JEPA 编码器/预测器未初始化")
            return []

        if not memories or len(memories) < 3:
            logger.warning("记忆不足 M3 预测（需要 ≥ 3 条）")
            return []

        try:
            # 1. GAT 编码
            state = self._jepa_encoder.encode(memories)

            if not state.causal_edges:
                logger.info("GAT 编码未发现因果边")
                return []

            # 2. GNN 预测下一状态
            next_state = self._jepa_predictor.predict(state)

            # 3. 提取预测边
            predictions = []
            for edge in next_state.causal_edges:
                predictions.append({
                    "cause": edge.get("cause", ""),
                    "effect": edge.get("effect", ""),
                    "rho": edge.get("rho", 0.0),
                    "confidence": edge.get("confidence", 0.5),
                    "verdict": edge.get("verdict", "predicted"),
                    "energy_relation": edge.get("energy_relation", "neutral"),
                })

            predictions.sort(key=lambda x: abs(x["rho"]), reverse=True)
            return predictions[:top_k]

        except Exception as e:
            logger.error("M3 预测失败: %s", e)
            return []

    def fused_predict(
        self,
        cause: str,
        memories: list[dict] | None = None,
        top_k: int = 5,
        retrieval_weight: float = 0.4,
        parametric_weight: float = 0.6,
    ) -> list[dict]:
        """
        融合预测（v4.4.1: 检索 + JEPA 加权）。

        v4.4.1: 将"参数化"路径替换为 JEPA 潜空间预测。
        parametric_weight 参数保留但语义变为 JEPA 预测权重。

        Args:
            cause: 原因文本
            memories: 记忆列表
            top_k: 返回数量
            retrieval_weight: 检索路径权重
            parametric_weight: JEPA 预测路径权重

        Returns:
            加权融合后的预测列表
        """
        retrieval_results = self.predict_effect(cause, memories, top_k=top_k)
        jepa_results = self.jepa_predict(cause, top_k=top_k, memories=memories)

        # 融合策略: JEPA 结果在前，检索结果补充
        fused = []
        seen_effects: set[str] = set()

        for r in jepa_results:
            effect_key = r.get("effect", "")
            if effect_key not in seen_effects:
                seen_effects.add(effect_key)
                fused.append({
                    "effect": effect_key,
                    "confidence": r.get("confidence", 0.5) * parametric_weight,
                    "source": "jepa",
                    "energy_relation": r.get("energy_relation", "neutral"),
                })

        for r in retrieval_results:
            content = r.get("content", "")
            if content not in seen_effects:
                seen_effects.add(content)
                fused.append({
                    "effect": content,
                    "confidence": r.get("confidence", 0.5) * retrieval_weight,
                    "source": "retrieval",
                    "causal_type": r.get("causal_type", ""),
                })

        fused.sort(key=lambda x: x["confidence"], reverse=True)
        return fused[:top_k]

    # ────────────────────────────────────────────────
    # 私有方法：因果图构建
    # ────────────────────────────────────────────────

    def _build_causal_graph_from_state(self) -> object | None:
        """
        从当前因果边构建 CausalGraph。

        统一 intervene()/decompose_effect()/query_counterfactual()
        的图构建逻辑，消除重复。

        Returns:
            CausalGraph 实例或 None
        """
        if not self._state or not self._state.causal_edges:
            return None
        from su_memory.sdk._do_calculus import DoCalculus
        n_nodes = max(
            max(e.get("cause_idx", 0), e.get("effect_idx", 0))
            for e in self._state.causal_edges
        ) + 1
        return DoCalculus.build_from_gaussian_dag(
            self._state.causal_edges, n_nodes
        )

    # ────────────────────────────────────────────────
    # Pearl do-operator 干预（v3.7.0 完整实现）
    # ────────────────────────────────────────────────

    def intervene(
        self,
        state: str = "current",
        do_x: dict | None = None,
        target: str | None = None,
        method: str = "auto",
    ) -> dict:
        """
        Pearl do-operator 干预预测（v3.7.0 完整实现）。

        计算公式:
            P(Y | do(X=x)) = Σ_z P(Y | X=x, Z=z) · P(Z=z)

        工作流:
        1. 从当前因果图构建 CausalGraph
        2. 识别调整变量集 (后门准则)
        3. 估计 ATE (平均处理效应)
        4. 返回干预分析结果

        Args:
            state: 世界状态标识
            do_x: 干预 {"变量名": 干预值}
            target: 目标变量名
            method: "auto"|"backdoor"|"frontdoor"

        Returns:
            InterventionResult 字典
        """
        if do_x is None or target is None:
            return {
                "status": "insufficient_input",
                "message": "需要 do_x 和 target 参数",
            }

        # ── 懒加载 DoCalculus 引擎 ──
        if self._do_calculus is None:
            with self._do_calculus_lock:
                if self._do_calculus is None:
                    try:
                        from su_memory.sdk._do_calculus import DoCalculus
                        self._do_calculus = DoCalculus()
                    except ImportError:
                        return {
                            "status": "error",
                            "message": "DoCalculus 引擎不可用",
                        }

        # ── 从因果边构建 CausalGraph ──
        try:
            from su_memory.sdk._do_calculus import CausalGraph
            cg = self._build_causal_graph_from_state()
            if cg is None:
                cg = CausalGraph(
                    nodes=list(do_x.keys()) + [target],
                    edges=[],
                )
        except Exception:
            logger.warning("CausalGraph 构建失败，回退到默认空图", exc_info=True)
            cg = CausalGraph(
                nodes=list(do_x.keys()) + [target],
                edges=[],
            )

        self._do_calculus.set_graph(cg)

        # ── 执行干预分析 ──
        x_name = list(do_x.keys())[0]
        x_value = float(list(do_x.values())[0])

        # F2-P0-1: 拒绝 NaN/Inf 干预值 — 保证浮点边界洁污不污染下游计算
        if not np.isfinite(x_value):
            return {
                "status": "error",
                "message": (
                    f"intervention value must be finite (NaN/Inf rejected), "
                    f"got: x_value={x_value}"
                ),
            }

        try:
            result = self._do_calculus.estimate_ate(
                X=x_name,
                Y=target,
                x_value=x_value,
                x_baseline=0.0,
                method=method,
            )
        except Exception as e:
            logger.error("干预分析失败: %s", e)
            return {
                "status": "error",
                "message": f"干预分析失败: {e}",
            }

        # ── 记录干预历史 ──
        intervention_record = {
            "state": state,
            "do": do_x,
            "target": target,
            "result": result.to_dict(),
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }
        self._intervention_history.append(intervention_record)
        self._state.do_interventions.append(intervention_record)

        # ── 构建反事实图 (干预边被切断) ──
        try:
            if cg.n_nodes > 0 and x_name in cg.nodes:
                x_idx = cg.node_index(x_name)
                if x_idx is not None and cg.adjacency is not None:
                    cf_adj = cg.adjacency.copy()
                    # 切断所有指向 X 的边 (do-operator 语义)
                    cf_adj[:, x_idx] = 0.0
                    self._state.counterfactual_graph = {
                        "nodes": list(cg.nodes),
                        "cf_adjacency": cf_adj.tolist(),
                        "intervention": do_x,
                    }
        except Exception:
            logger.warning("反事实图构建失败，跳过", exc_info=True)

        # ── 返回结果 ──
        output = result.to_dict()
        output["status"] = "ok"
        output["history_count"] = len(self._intervention_history)
        return output

    # ────────────────────────────────────────────────
    # 因果效应分解（v3.7.0 新增）
    # ────────────────────────────────────────────────

    def decompose_effect(
        self,
        cause: str,
        effect: str,
        mediator: str | None = None,
    ) -> dict:
        """
        因果效应三分解:
        - NDE: 自然直接效应 (Natural Direct Effect)
        - NIE: 自然间接效应 (Natural Indirect Effect)
        - TE:  总效应 (Total Effect = NDE + NIE)

        使用 Pearl 的 mediation formula:
            NDE = E[Y_{x,M_{x*}} - Y_{x*}]
            NIE = E[Y_{x,M_x} - Y_{x,M_{x*}}]

        Args:
            cause: 原因变量
            effect: 结果变量
            mediator: 中介变量 (None 时自动检测)

        Returns:
            {"nde": float, "nie": float, "te": float, "mediator": str, ...}
        """
        # ── 自动检测中介变量 ──
        if mediator is None:
            if self._do_calculus is None:
                with self._do_calculus_lock:
                    if self._do_calculus is None:
                        try:
                            from su_memory.sdk._do_calculus import DoCalculus
                            self._do_calculus = DoCalculus()
                        except ImportError:
                            return {
                                "nde": 0.0, "nie": 0.0, "te": 0.0,
                        "mediator": None,
                        "note": "do_calculus_unavailable",
                    }

            # 尝试从因果图中识别中介
            if self._state and self._state.causal_edges:
                try:
                    cg = self._build_causal_graph_from_state()
                    if cg is not None:
                        mediators = cg.get_mediators(cause, effect)
                        mediator = mediators[0] if mediators else None
                    else:
                        mediator = None
                except Exception:
                    logger.warning(
                        "中介变量识别失败，回退为无中介", exc_info=True
                    )
                    mediator = None

        if mediator is None:
            return {
                "nde": 0.0, "nie": 0.0, "te": 0.0,
                "mediator": None,
                "note": "no_mediator_identified",
            }

        # ── 使用 do-calculus 分解 ──
        try:
            # 直接效应: do(mediator) 固定时的 cause → effect
            direct_result = self.intervene(
                do_x={cause: 1.0},
                target=effect,
                method="direct",
            )
            nde = direct_result.get("ate", 0.0)

            # 间接效应: cause → mediator 的 ATE × mediator → effect 的 ATE
            cause_to_med = self.intervene(
                do_x={cause: 1.0},
                target=mediator,
                method="direct",
            )
            med_to_eff = self.intervene(
                do_x={mediator: 1.0},
                target=effect,
                method="direct",
            )
            nie = cause_to_med.get("ate", 0.0) * med_to_eff.get("ate", 0.0)
        except Exception as e:
            logger.error("因果分解失败: %s", e)
            nde = 0.0
            nie = 0.0

        te = nde + nie

        return {
            "nde": round(nde, 6),
            "nie": round(nie, 6),
            "te": round(te, 6),
            "mediator": mediator,
            "nde_pct": round(abs(nde) / max(abs(te), 1e-10) * 100, 1),
            "nie_pct": round(abs(nie) / max(abs(te), 1e-10) * 100, 1),
            "method": "mediation_formula",
        }

    # ────────────────────────────────────────────────
    # 反事实推理（v3.8.0 新增 — Pearl L3）
    # ────────────────────────────────────────────────

    def query_counterfactual(
        self,
        evidence: dict[str, float],
        do_x: dict[str, float],
        target: str,
        compute_pns: bool = True,
    ) -> dict:
        """
        Pearl 三步反事实推理（v3.8.0 L3 新增）。

        基于当前因果图，回答反事实问题:
            "如果当初 X=x' 而非 X=x，Y 会是多少？"

        三步算法:
            1. Abduction (溯因): 从事实证据推断不可观测噪声
            2. Action (干预): 用 do(X=x') 构建 mutilated graph
            3. Prediction (预测): 用溯因噪声 + mutilated graph 计算 Y_{x'}

        输出:
            - counterfactual_value: 反事实结果
            - individual_effect: 个体因果效应 (Y_{x'} - Y)
            - PN/PS/PNS: 必然性/充分性概率
            - noise_terms: 溯因推断的噪声项

        Args:
            evidence: 事实证据 {"X": 1.0, "Y": 3.0, ...}
            do_x: 反事实干预 {"X": 0.0}
            target: 目标变量 (反事实结果)
            compute_pns: 是否计算 PN/PS/PNS

        Returns:
            CounterfactualResult dict

        Example:
            >>> wm = MCIWorldModel(su_lite_pro)
            >>> wm.discover()
            >>> result = wm.query_counterfactual(
            ...     evidence={"手术量": 100, "收入": 50},
            ...     do_x={"手术量": 80},
            ...     target="收入",
            ... )
            >>> print(f"反事实收入: {result['counterfactual_value']}")
        """
        try:
            from su_memory.sdk._counterfactual import CounterfactualEngine
            from su_memory.sdk._do_calculus import CausalGraph
        except ImportError as e:
            return {
                "status": "error",
                "note": f"counterfactual_engine_unavailable: {e}",
            }

        # ── 从因果图构建 CausalGraph ──
        cg = self._build_causal_graph_from_state()
        if cg is None:
            all_nodes = (
                list(evidence.keys())
                + list(do_x.keys())
                + [target]
            )
            cg = CausalGraph(nodes=list(set(all_nodes)), edges=[])

        # ── 构建反事实引擎并查询 ──
        engine = CounterfactualEngine.from_causal_graph(cg)
        if engine is None:
            return {
                "status": "error",
                "note": "failed_to_build_counterfactual_engine",
            }

        try:
            result = engine.query(
                evidence=evidence,
                do_x=do_x,
                target=target,
                compute_pns=compute_pns,
            )
        except Exception as e:
            logger.error("反事实查询失败: %s", e, exc_info=True)
            return {
                "status": "error",
                "note": f"counterfactual_query_failed: {e}",
            }

        return result.to_dict()

    # ────────────────────────────────────────────────
    # 因果解释
    # ────────────────────────────────────────────────

    def explain(
        self,
        query: str,
        max_depth: int = 3,
    ) -> dict:
        """
        因果链回溯解释。

        返回从 query 出发的因果路径，
        含每一步的置信度和能量关系类型。

        Args:
            query: 查询文本
            max_depth: 最大因果跳数

        Returns:
            {
                "query": str,
                "chains": [{"path": [...], "confidence": float}, ...],
                "summary": str,
            }
        """
        if not self._state.causal_edges:
            return {
                "query": query,
                "chains": [],
                "summary": "暂无因果图数据。请先运行 discover()。",
            }

        chains = self._trace_causal_chains(query, max_depth)
        summary = self._generate_explanation_summary(chains, query)

        return {
            "query": query,
            "chains": chains[:5],  # 最多 5 条链
            "summary": summary,
        }

    def _trace_causal_chains(
        self, query: str, max_depth: int
    ) -> list[dict]:
        """追踪因果链。"""
        chains = []
        # 简单的关键词匹配追溯
        for edge in self._state.causal_edges:
            if "cause_idx" in edge and "effect_idx" in edge:
                # 基于索引的边 — 需要映射回文本
                chains.append({
                    "path": [
                        f"节点 {edge['cause_idx']}",
                        f"→ 节点 {edge['effect_idx']}",
                    ],
                    "confidence": edge.get("confidence", 0.5),
                    "verdict": edge.get("verdict", "unknown"),
                    "energy_relation": edge.get("energy_relation", "neutral"),
                    "depth": 1,
                })

        chains.sort(key=lambda c: c["confidence"], reverse=True)
        return chains

    def _generate_explanation_summary(
        self, chains: list[dict], query: str
    ) -> str:
        """生成可读解释摘要。"""
        if not chains:
            return f"未找到与「{query}」相关的因果链。"

        n_confirmed = sum(1 for c in chains if c.get("verdict") == "confirmed")
        n_novel = sum(1 for c in chains if c.get("verdict") == "novel")

        parts = [f"共发现 {len(chains)} 条与「{query}」相关的因果链。"]
        if n_confirmed > 0:
            parts.append(f"其中 {n_confirmed} 条被拓扑先验确认。")
        if n_novel > 0:
            parts.append(f"{n_novel} 条为潜在新发现。")

        top_chain = chains[0]
        parts.append(
            f"最高置信度链 (置信度: {top_chain['confidence']:.2f}): "
            f"{' → '.join(top_chain['path'])}"
        )

        return " ".join(parts)

    # ────────────────────────────────────────────────
    # JEPA 训练
    # ────────────────────────────────────────────────

    def train_jepa(
        self,
        dataset: object | None = None,
        qa_pairs: list | None = None,
        output_dir: str = "./checkpoints/mci-world-model",
        n_epochs: int = 10,
        learning_rate: float = 0.01,
    ) -> dict:
        """
        JEPA 端到端训练（v4.4.1，替代 train_parametric）。

        Args:
            dataset: JEPADataset 实例（优先）
            qa_pairs: Reflection QA 对列表（备选，自动转 JEPADataset）
            output_dir: checkpoint 输出目录
            n_epochs: 训练轮数
            learning_rate: 学习率

        Returns:
            训练统计
        """
        # ── 构造 JEPADataset ──
        if dataset is None and qa_pairs is not None:
            try:
                from su_memory.sdk._jepa_dataset import JEPADataset
                # 从 QA 对提取记忆并构造数据集
                memories = []
                for qa in qa_pairs:
                    if isinstance(qa, dict):
                        memories.append({
                            "content": qa.get("question", ""),
                            "answer": qa.get("answer", ""),
                        })
                if memories:
                    dataset = JEPADataset.from_memories(
                        memories, self
                    )
            except ImportError as e:
                return {"error": f"jepa_dataset_unavailable: {e}"}

        if dataset is None:
            return {
                "error": "no_training_data",
                "message": "需要 JEPADataset 或 qa_pairs",
            }

        # ── 构造训练器并训练 ──
        try:
            from su_memory.sdk._jepa_trainer import JEPATrainer

            trainer = JEPATrainer(
                encoder=self._jepa_encoder,
                predictor=self._jepa_predictor,
                dataset=dataset,
            )
            stats = trainer.train(
                n_epochs=n_epochs,
                learning_rate=learning_rate,
            )

            self._state.n_qa_pairs = len(dataset.pairs) if hasattr(
                dataset, "pairs"
            ) else 0

            return {
                "n_pairs": len(dataset.pairs) if hasattr(dataset, "pairs") else 0,
                "n_epochs": n_epochs,
                "training_stats": stats.to_dict() if hasattr(stats, "to_dict") else stats,
                "adapter_path": output_dir,
                "mode": "e2e" if trainer._is_e2e else ("gnn" if trainer._is_gnn else "baseline"),
            }
        except ImportError as e:
            return {"error": f"jepa_trainer_unavailable: {e}"}
        except Exception as e:
            logger.error("JEPA 训练失败: %s", e)
            return {"error": str(e)}

    def train_parametric(
        self,
        qa_pairs: list,
        output_dir: str = "./checkpoints/mci-world-model",
    ) -> dict:
        """
        一键参数化训练（v3.6.0 — v4.4.1 降级为 train_jepa 别名）。

        v4.4.1: 重路由到 JEPA 训练循环。

        Args:
            qa_pairs: Reflection QA 对列表
            output_dir: checkpoint 输出目录

        Returns:
            训练统计
        """
        return self.train_jepa(qa_pairs=qa_pairs, output_dir=output_dir)

    # ────────────────────────────────────────────────
    # 健康检查
    # ────────────────────────────────────────────────

    def health_check(self) -> dict:
        """全系统健康诊断。"""
        check = {
            "version": "4.4.1",
            "code_name": "MCI World Model v4.4.1 JEPA",
            "initialized": self._initialized,
            "causal_pipeline": {
                "edges_discovered": len(self._state.causal_edges),
                "confirmed": self._state.n_confirmed,
                "novel": self._state.n_novel,
                "suppressed": self._state.n_suppressed,
                "has_counterfactual_graph": self._state.counterfactual_graph is not None,
                "n_do_interventions": len(self._state.do_interventions),
            },
            "jepa_predictor": {
                "available": self._jepa_predictor is not None,
                "encoder_available": self._jepa_encoder is not None,
                "is_gnn": self._is_gnn_predictor(),
                "predictor_type": type(self._jepa_predictor).__name__ if self._jepa_predictor else "none",
            },
            "energy_loss": {
                "available": self._energy_loss is not None,
            },
            "integration": {
                "lite_pro_connected": self._lite_pro is not None,
                "n_memories": self._state.n_memories,
            },
            "roadmap": {
                "v3.6.0": "parametric_causal_discovery ✓",
                "v3.7.0": "do_operator_intervention ✓",
                "v3.8.0": "counterfactual_reasoning_l3 ✓",
                "v4.4.1": "jepa_world_model_closed_loop ✓",
                "v4.4.1-m2": "jepa_gnn_trainable ✓" if self._is_gnn_predictor() else "jepa_gnn_trainable (use GNNPredictor)",
                "v4.4.1-m3": "jepa_e2e_differentiable ✓" if self._is_e2e_mode() else "jepa_e2e_differentiable (use enable_m3())",
            },
            "status": self._compute_health_status(),
        }
        return check

    def _compute_health_status(self) -> str:
        """计算整体健康状态。"""
        if not self._initialized:
            return "not_initialized"
        if len(self._state.causal_edges) == 0:
            return "no_causal_data"
        if self._state.n_confirmed > 0 and self._jepa_predictor is not None:
            if self._is_gnn_predictor():
                return "fully_operational_gnn"
            return "fully_operational"
        if self._state.n_confirmed > 0:
            return "operational_retrieval_only"
        return "degraded"

    def _is_gnn_predictor(self) -> bool:
        """检测是否使用可微 GNN 预测器。"""
        if self._jepa_predictor is None:
            return False
        return hasattr(self._jepa_predictor, 'training_predict')

    def _is_e2e_mode(self) -> bool:
        """检测是否启用 M3 端到端可微模式。"""
        if self._jepa_encoder is None or self._jepa_predictor is None:
            return False
        return (hasattr(self._jepa_encoder, 'training_encode') and
                self._is_gnn_predictor())

    # ────────────────────────────────────────────────
    # M3: 端到端可微模式
    # ────────────────────────────────────────────────

    def enable_m3(
        self,
        encoder_key_dim: int = 16,
        predictor_hidden_dim: int = 16,
    ) -> dict:
        """
        启用 M3 端到端可微训练模式。

        安装 GAT 编码器 + GNN 预测器，替代基线预测器。
        调用后，train_jepa() 走 e2e 训练路径。

        Args:
            encoder_key_dim: GAT 注意力键维度
            predictor_hidden_dim: GNN 隐层维度

        Returns:
            状态报告
        """
        report = {"encoder": "unchanged", "predictor": "unchanged"}

        # ── GAT 编码器 ──
        if self._jepa_encoder is not None:
            try:
                from su_memory.sdk._jepa_encoder import JEPAEncoder
                encoder = JEPAEncoder(self, differentiable=True, gat_key_dim=encoder_key_dim)
                self._jepa_encoder = encoder
                report["encoder"] = "gat_encoder_initialized"
                logger.info("M3 GAT 编码器已安装 (key_dim=%d)", encoder_key_dim)
            except Exception as e:
                report["encoder"] = f"failed: {e}"
                logger.warning("M3 GAT 编码器失败: %s", e)

        # ── GNN 预测器 ──
        try:
            from su_memory.sdk._jepa_gnn import GNNPredictor
            self._jepa_predictor = GNNPredictor(hidden_dim=predictor_hidden_dim)
            report["predictor"] = "gnn_predictor_installed"
            logger.info("M3 GNN 预测器已安装 (hidden_dim=%d)", predictor_hidden_dim)
        except Exception as e:
            report["predictor"] = f"failed: {e}"
            logger.warning("M3 GNN 预测器失败: %s", e)

        return report

    # ────────────────────────────────────────────────
    # 内部工具
    # ────────────────────────────────────────────────

    def _get_memories_from_lite_pro(self) -> list[dict]:
        """从 lite_pro 获取记忆列表。"""
        if self._lite_pro is None:
            return []
        try:
            # SuMemoryLitePro 可能通过不同方式暴露记忆
            if hasattr(self._lite_pro, "_store"):
                store = self._lite_pro._store
                if isinstance(store, dict):
                    return [
                        {"id": k, "content": v.get("content", "")}
                        for k, v in store.items()
                    ]
            # 通过 query 获取
            if hasattr(self._lite_pro, "query"):
                results = self._lite_pro.query("*", top_k=100)
                return [
                    {"id": r.get("id", str(i)), "content": r.get("content", "")}
                    for i, r in enumerate(results)
                ]
        except Exception as e:
            logger.warning("从 lite_pro 获取记忆失败: %s", e)
        return []

    def _apply_parametric_prior(
        self, dag, memories: list[dict]
    ):
        """
        v4.0.0: JEPADataset 先验注入（替代 TopologicalEnergyMatrix 回退）。

        通过 JEPADataset 从历史状态转移中提取因果边先验权重。
        不可用时回退到均匀弱先验。

        Args:
            dag: GaussianDAG 实例
            memories: 记忆列表 (≥ 1 条)
        """
        n = min(len(memories), 50)
        parametric_prior = np.zeros((n, n), dtype=np.float32)

        # v4.0.0: 优先使用 JEPADataset 统计信息构造先验
        prior_source = "uniform"
        try:
            from su_memory.sdk._jepa_dataset import JEPADataset
            dataset = JEPADataset.from_memories(memories, self)
            if dataset.pairs and len(dataset.pairs) >= 1:
                avg_dist = dataset.stats.get("avg_distance", 0.5)
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            parametric_prior[i, j] = max(
                                0.05, 1.0 - avg_dist
                            ) * 0.2
                prior_source = "jepa_dataset"
            else:
                parametric_prior.fill(0.1)
                for i in range(n):
                    parametric_prior[i, i] = 0.0
        except ImportError:
            for i in range(n):
                for j in range(n):
                    if i != j:
                        parametric_prior[i, j] = 0.1

        dag.with_parametric_prior(parametric_prior)
        logger.debug(
            "JEPA 先验已注入 GaussianDAG (%dx%d, source=%s)",
            n, n, prior_source,
        )

    @property
    def state(self) -> CausalWorldModelState:
        """当前世界模型状态。"""
        return self._state

    def __repr__(self) -> str:
        status = self._compute_health_status()
        jepa_ready = self._jepa_predictor is not None
        gnn_label = "[GNN]" if self._is_gnn_predictor() else ""
        return (
            f"MCIWorldModel(v4.0.0 JEPA{gnn_label}, {len(self._state.causal_edges)} edges, "
            f"jepa={'✓' if jepa_ready else '✗'}, "
            f"status={status})"
        )
