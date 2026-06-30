"""
su-memory v3.3.0 — Lightweight Causal Engine

基于中文关键词模式的因果关系检测和推理。
独立于 SuMemoryLitePro 的 MemoryGraph，为 SuMemoryLite 提供因果推理能力。

用法:
    from su_memory.sdk._causal import CausalEngine

    engine = CausalEngine()
    pairs = engine.find_causal_pairs(memories_list)
    # → [(cause_memory, effect_memory, confidence, causal_type), ...]
"""

from __future__ import annotations

from ..algebra.belief_net import BeliefNetwork, BeliefPropagator

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
    for pattern_name, pattern in CAUSAL_PATTERNS.items():
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
    for pattern_name, pattern in CAUSAL_PATTERNS.items():
        for marker in pattern["markers"]:
            if marker and marker in text_b:
                for eff_marker in pattern["effect_markers"]:
                    if eff_marker in text_a:
                        score = 0.6 + 0.1 * (len(marker) + len(eff_marker)) / 2
                        return (f"reverse_{pattern['type']}", round(min(score, 0.95), 3))

    return None


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
        # --- Bayesian belief backend (algebra layer) ---
        # Pairs detected by the lightweight keyword patterns are fed as
        # observations into a Beta-Bernoulli belief network, so that genuine
        # posterior causal probabilities (not just keyword match scores) can be
        # queried via infer_belief().
        self._belief_net = BeliefNetwork()
        self._belief_prop = BeliefPropagator(max_iterations=30, damping=0.5)
        self._belief_dirty = False

    def find_causal_pairs(
        self,
        memories: list[dict],
    ) -> list[tuple[dict, dict, str, float]]:
        """
        在记忆列表中查找因果关系对。

        时间复杂度: O(n²) — 适用于 n ≤ 100 的分析场景。

        Args:
            memories: 记忆列表，每项含 "id" 和 "content"

        Returns:
            [(cause_memory, effect_memory, causal_type, confidence), ...]
            按置信度降序排列。
        """
        if len(memories) > 500:
            # 超过 500 条时采样最近 100 条
            memories = memories[-100:]

        pairs: list[tuple[dict, dict, str, float]] = []

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

        # 按置信度降序
        pairs.sort(key=lambda x: x[3], reverse=True)

        # Feed detected cause->effect pairs into the Bayesian belief network.
        # Each (cause, effect, confidence) becomes a weighted positive
        # co-occurrence observation on the edge cause -> effect.
        for cause_mem, effect_mem, _ctype, confidence in pairs:
            cause_id = cause_mem.get("id", cause_mem.get("content", ""))
            effect_id = effect_mem.get("id", effect_mem.get("content", ""))
            if cause_id and effect_id and cause_id != effect_id:
                self._belief_net.observe(
                    cause_id, effect_id,
                    parent_state=True, child_state=True,
                    weight=max(confidence, 0.01),
                )
        self._belief_dirty = True

        return pairs

    def predict_effects(
        self,
        cause_content: str,
        memories: list[dict],
        top_k: int = 3,
    ) -> list[dict]:
        """
        基于历史记忆预测给定原因的效应。

        Args:
            cause_content: 原因文本
            memories: 历史记忆列表
            top_k: 返回数量

        Returns:
            [{"memory_id", "content", "confidence", "causal_type"}, ...]
        """
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

    def infer_belief(
        self,
        evidence: dict[str, bool],
        query_nodes: list[str],
    ) -> dict[str, float]:
        """Bayesian posterior inference over the cause/effect graph.

        Given observed Boolean states at some memory nodes (``evidence``),
        compute the posterior probability that each ``query_node`` is "active"
        via loopy belief propagation on the Beta-Bernoulli network learned
        from :meth:`find_causal_pairs`.

        This upgrades the keyword-only causal detection to genuine
        probabilistic causal inference: edges carry learned conditional
        probabilities P(effect | cause), and evidence propagates to
        query nodes through the network topology.

        Parameters
        ----------
        evidence : dict
            memory_id -> observed Boolean state.
        query_nodes : list
            memory ids whose posterior P(active) is requested.

        Returns
        -------
        dict
            memory_id -> posterior mean in [0, 1].
        """
        posteriors = self._belief_prop.infer(
            self._belief_net, query_nodes, evidence=evidence
        )
        return {nid: float(b.mean) for nid, b in posteriors.items()}

    def causal_strength(self, cause_id: str, effect_id: str) -> float | None:
        """Learned causal strength P(effect | cause=1) - P(effect | cause=0).

        Returns None if no edge has been learned between the two nodes. A
        positive value means the cause promotes the effect; negative means it
        inhibits it. This is the data-driven replacement for the fixed keyword
        confidence scores.
        """
        edge = self._belief_net.get_edge(cause_id, effect_id)
        if edge is None:
            return None
        return float(edge.causal_strength)
