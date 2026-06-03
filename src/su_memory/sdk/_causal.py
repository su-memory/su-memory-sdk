"""
su-memory v3.8.0 — Lightweight Causal Engine

基于中文关键词模式的因果关系检测和推理。
v3.4.0: 新增双路径因果发现 — 关键词匹配 (语法层) + 偏相关统计 (数值层)
v3.6.0: 新增参数化路径 — QLoRA 模型推理 (语义层，三路径加权融合)
v3.7.0: 新增 do-calculus 干预预测 — Pearl do-operator 因果干预
独立于 SuMemoryLitePro 的 MemoryGraph，为 SuMemoryLite 提供因果推理能力。

用法:
    from su_memory.sdk._causal import CausalEngine

    engine = CausalEngine()
    pairs = engine.find_causal_pairs(memories_list)

    # v3.6.0 新增: 参数化路径 (三路径融合)
    pairs = engine.find_causal_pairs(memories_list, use_parametric=True, parametric_model=pm)

    # v3.7.0 新增: 干预预测
    effects = engine.predict_effects(cause, memories, use_intervention=True, do_value=1.5)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 中文因果关系关键词模式
# ---------------------------------------------------------------------------

CAUSAL_PATTERNS: dict[str, dict[str, list[str]]] = {
    "cause": {
        "markers": ["如果", "因为", "由于", "既然", "因", "由"],
        "effect_markers": ["所以", "因此", "导致", "使得", "促使", "引发",
                          "就会", "结果", "于是", "那么", "就", "便"],
        "type": "cause",
    },
    "condition": {
        "markers": ["当", "只要", "除非", "假如", "倘若", "要是"],
        "effect_markers": ["就", "便", "则", "会"],
        "type": "condition",
    },
    "result": {
        "markers": [""],  # 无前置 marker，检测 effect_markers 在效应侧
        "effect_markers": ["所以", "因此", "导致", "使得", "促使", "则", "必然", "一定"],
        "type": "result",
    },
}

# 共享词汇因果（无需连接词，靠共享主语/主题）
SHARED_CAUSAL_PATTERNS = [
    "导致", "造成", "引起", "引发", "促使", "使得",
    "带来", "产生", "触发", "推动", "带动", "影响",
]


def detect_causal_link(
    text_a: str,
    text_b: str,
) -> tuple[str, float] | None:
    """
    检测两段文本之间是否存在因果关系。

    返回 (causal_type, confidence) 或 None。

    causal_type: "cause" | "condition" | "result" | "shared"
    """
    # 1. 关键词模式匹配
    for _pattern_name, pattern in CAUSAL_PATTERNS.items():
        for marker in pattern["markers"]:
            if marker and marker in text_a:
                for eff_marker in pattern["effect_markers"]:
                    if eff_marker in text_b:
                        # 分数基于标记组合强度
                        score = 0.6 + 0.1 * (len(marker) + len(eff_marker)) / 2
                        return (pattern["type"], round(min(score, 0.95), 3))

    # 2. 共享关键词因果
    for shared_kw in SHARED_CAUSAL_PATTERNS:
        if shared_kw in text_a and shared_kw in text_b:
            return ("shared", 0.7)

    # 3. 检查反向（text_b 是原因，text_a 是结果）
    for _pattern_name, pattern in CAUSAL_PATTERNS.items():
        for marker in pattern["markers"]:
            if marker and marker in text_b:
                for eff_marker in pattern["effect_markers"]:
                    if eff_marker in text_a:
                        score = 0.6 + 0.1 * (len(marker) + len(eff_marker)) / 2
                        return (f"reverse_{pattern['type']}", round(min(score, 0.95), 3))

    return None


# ---------------------------------------------------------------------------
# v3.4.0: 统计路径辅助
# ---------------------------------------------------------------------------
def _hash_pair_id_360(cause: str, effect: str) -> str:
    """v3.6.0: 生成参数化因果对哈希 ID。"""
    import hashlib
    h = hashlib.sha256(f"{cause}:::{effect}".encode()).hexdigest()
    return f"p3_{h[:8]}"


def _is_duplicate(
    pairs: list[tuple[dict, dict, str, float]], id_a: str, id_b: str
) -> bool:
    """检查因果对是否已在列表中 (双向去重)。"""
    for p in pairs:
        pid_a = p[0].get("id", "")
        pid_b = p[1].get("id", "")
        if (pid_a == id_a and pid_b == id_b) or (pid_a == id_b and pid_b == id_a):
            return True
    return False


# ---------------------------------------------------------------------------
# CausalEngine
# ---------------------------------------------------------------------------

class CausalEngine:
    """
    轻量级因果推理引擎。

    基于中文关键词模式检测记忆间的因果关系，
    不依赖外部模型或向量嵌入。

    Example:
        >>> engine = CausalEngine()
        >>> pairs = engine.find_causal_pairs([
        ...     {"id": "1", "content": "暴雨导致城市内涝"},
        ...     {"id": "2", "content": "城市内涝促使排水系统升级"},
        ... ])
        >>> len(pairs) >= 1
        True
    """

    def __init__(self, min_confidence: float = 0.5):
        """
        Args:
            min_confidence: 最低置信度阈值
        """
        self.min_confidence = min_confidence
        self._causal_pairs_cache: list[tuple[dict, dict, str, float]] = []

    def find_causal_pairs(
        self,
        memories: list[dict],
        use_statistical: bool = False,
        energy_bus=None,
        index: dict[str, set] | None = None,
        use_reflection_prior: bool = False,  # v3.5.0 新增
        use_parametric: bool = False,  # v3.6.0 新增
        parametric_model=None,  # v3.6.0 新增: ParametricMemory 实例
    ) -> list[tuple[dict, dict, str, float]]:
        """
        在记忆列表中查找因果关系对。

        时间复杂度: O(n²) — 适用于 n ≤ 100 的分析场景。

        Args:
            memories: 记忆列表，每项含 "id" 和 "content"
            use_statistical: v3.4.0 — 启用偏相关统计路径
            energy_bus: 可选 EnergyBus 实例，用于能量先验交叉验证
            index: 可选 TF-IDF 索引，用于统计路径向量化
            use_reflection_prior: v3.5.0 — 启用 Reflection QA 因果先验增强
            use_parametric: v3.6.0 — 启用参数化模型推理路径
            parametric_model: v3.6.0 — ParametricMemory 实例 (三路径融合时使用)

        Returns:
            [(cause_memory, effect_memory, causal_type, confidence), ...]
            按置信度降序排列。
        """
        if len(memories) > 500:
            # 超过 500 条时采样最近 100 条
            memories = memories[-100:]

        pairs: list[tuple[dict, dict, str, float]] = []

        # ── 路径 1: 关键词匹配 (保留全部现有逻辑) ──
        for i, mem_a in enumerate(memories):
            for j, mem_b in enumerate(memories):
                if i == j:
                    continue
                result = detect_causal_link(
                    mem_a.get("content", ""),
                    mem_b.get("content", ""),
                )
                if result:
                    causal_type, confidence = result
                    if confidence >= self.min_confidence:
                        pairs.append((mem_a, mem_b, causal_type, confidence))

        # ── 路径 2: 偏相关统计因果发现 (v3.4.0 新增) ──
        if use_statistical and len(memories) >= 10:
            try:
                from su_memory.sdk._spectral_causal import GaussianDAG
                dag = GaussianDAG(memories, index, energy_bus)

                # ── v3.5.0: Reflection Prior 增强 ──
                if use_reflection_prior:
                    try:
                        from su_memory.sdk._reflection_synthesizer import (
                            ReflectionSynthesizer,
                        )
                        syn = ReflectionSynthesizer(
                            energy_bus=energy_bus,
                            min_confidence=0.4,
                            max_pairs=200,
                        )
                        _, prior_matrix = syn.run_pipeline(memories)
                        dag.with_reflection_prior(prior_matrix)
                    except ImportError:
                        logger.debug(
                            "ReflectionSynthesizer 不可用，降级为纯统计路径"
                        )

                # ── v3.6.0: 参数化先验增强 ──
                if use_parametric and parametric_model is not None:
                    try:
                        from su_memory.sdk._energy_loss import TopologicalEnergyMatrix
                        topo = TopologicalEnergyMatrix.build()
                        dag.with_parametric_prior(topo)
                    except ImportError:
                        logger.debug(
                            "TopologicalEnergyMatrix 不可用，跳过参数化先验"
                        )

                stat_edges = dag.discover_hidden_edges()

                for edge in stat_edges:
                    i, j = edge["cause_idx"], edge["effect_idx"]
                    mem_a = memories[i]
                    mem_b = memories[j]

                    # 去重: 避免与关键词结果重复
                    if _is_duplicate(pairs, mem_a.get("id", ""), mem_b.get("id", "")):
                        continue

                    causal_type = (
                        f"stat_{edge.get('verdict', '')}"
                        f"_{edge.get('energy_relation', '')}"
                    )
                    pairs.append((
                        mem_a, mem_b,
                        causal_type,
                        edge["confidence"],
                    ))
            except ImportError:
                logger.debug(
                    "GaussianDAG 统计模块不可用，降级为纯关键词模式"
                )

        # ── 路径 3: 参数化模型推理 (v3.6.0 新增) ──
        if use_parametric and parametric_model is not None and hasattr(parametric_model, 'predict'):
            try:
                for _i, mem_a in enumerate(memories):
                    cause_text = mem_a.get("content", "")
                    if len(cause_text) < 5:
                        continue
                    predictions = parametric_model.predict(cause_text, top_k=2)
                    for pred in predictions:
                        pred_confidence = pred.get("confidence", 0.5)
                        if pred_confidence < self.min_confidence:
                            continue
                        # 构造虚拟记忆对象作为效应
                        pred_id = f"parametric_{_hash_pair_id_360(cause_text, pred.get('effect', ''))}"
                        effect_mem = {
                            "id": pred_id,
                            "content": pred.get("effect", ""),
                            "parametric": True,
                        }
                        if not _is_duplicate(pairs, mem_a.get("id", ""), pred_id):
                            pairs.append((
                                mem_a, effect_mem,
                                f"parametric_{pred.get('energy_relation', 'neutral')}",
                                pred_confidence,
                            ))
            except Exception as e:
                logger.debug("参数化推理失败: %s", e)

        # 按置信度降序
        pairs.sort(key=lambda x: x[3], reverse=True)
        return pairs

    def predict_effects(
        self,
        cause_content: str,
        memories: list[dict],
        top_k: int = 3,
        use_intervention: bool = False,  # v3.7.0 新增
        do_value: float | None = None,    # v3.7.0 新增
    ) -> list[dict]:
        """
        基于历史记忆预测给定原因的效应。

        Args:
            cause_content: 原因文本
            memories: 历史记忆列表
            top_k: 返回数量
            use_intervention: v3.7.0 — 使用 do-calculus 替代纯统计预测
            do_value: v3.7.0 — do(X) 干预值

        Returns:
            [{"memory_id", "content", "confidence", "causal_type"}, ...]
        """
        # ── v3.7.0: 干预增强路径 ──
        if use_intervention:
            try:
                from su_memory.sdk._world_model import MCIWorldModel
                wm = MCIWorldModel()
                wm.initialize()
                # 使用记忆构建因果图
                if memories:
                    wm.discover(memories=memories)
                # 干预预测
                result = wm.intervene(
                    do_x={cause_content: do_value if do_value is not None else 1.0},
                    target="effect",
                )
                if result.get("status") == "ok":
                    return [{
                        "memory_id": "do_intervention",
                        "content": f"ATE={result.get('ate', 0):.4f}",
                        "confidence": 1.0 - result.get("p_value", 0.5),
                        "causal_type": f"do_calculus_{result.get('method', 'auto')}",
                        "intervention_result": result,
                    }]
            except ImportError:
                logger.debug("MCIWorldModel 不可用，回退到关键词模式")
            except Exception as e:
                logger.debug("干预预测失败: %s，回退到关键词模式", e)

        # ── 关键词路径 (现有逻辑) ──
        causes = []
        for mem in memories:
            result = detect_causal_link(cause_content, mem.get("content", ""))
            if result:
                causal_type, confidence = result
                causes.append({
                    "memory_id": mem.get("id", ""),
                    "content": mem["content"],
                    "confidence": confidence,
                    "causal_type": causal_type,
                })

        causes.sort(key=lambda x: x["confidence"], reverse=True)
        return causes[:top_k]

    def query_causal_chain(
        self,
        query: str,
        memories: list[dict],
        max_depth: int = 2,
    ) -> list[dict]:
        """
        查询因果链：查询 → 直接效应 → 二级效应。

        Args:
            query: 查询文本
            memories: 记忆列表
            max_depth: 最大因果跳数

        Returns:
            [{"depth", "memory_id", "content", "confidence"}, ...]
        """
        chain = []
        seen: set[str] = set()

        # Depth 1: direct effects
        for mem in memories:
            result = detect_causal_link(query, mem.get("content", ""))
            if result:
                _, confidence = result
                mid = mem.get("id", "")
                if mid not in seen:
                    seen.add(mid)
                    chain.append({
                        "depth": 1,
                        "memory_id": mid,
                        "content": mem["content"],
                        "confidence": confidence,
                    })

        # Depth 2: effects of effects
        if max_depth >= 2:
            for item in chain[:]:  # iterate copy
                for mem in memories:
                    result = detect_causal_link(
                        item["content"],
                        mem.get("content", ""),
                    )
                    if result:
                        _, confidence = result
                        mid = mem.get("id", "")
                        if mid not in seen:
                            seen.add(mid)
                            chain.append({
                                "depth": 2,
                                "memory_id": mid,
                                "content": mem["content"],
                                "confidence": round(confidence * 0.8, 3),
                                "parent": item["memory_id"],
                            })

        chain.sort(key=lambda x: (x["depth"], -x["confidence"]))
        return chain
