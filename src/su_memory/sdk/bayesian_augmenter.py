"""
BayesianAugmenter — 贝叶斯推理串联增强包装器

非侵入式地将贝叶斯推理增强串联到现有 SuMemoryLitePro 系统。

核心设计原则：
1. ⛓️  串联集成 — 原系统完整保留，贝叶斯作为增强管道
2. 🔄 双路径执行 — 每个 API 同时运行原始版和贝叶斯版
3. 📊 对比输出 — 返回原始结果 + 贝叶斯结果 + 差异分析
4. 🎯 反馈闭环 — 通过 feedback() 对比验证，追踪准确度提升
5. 🔌 零侵入 — SuMemoryLitePro 源码一行不改

架构:
    User Input
        │
        ▼
    ┌─────────────────────────────────────┐
    │       BayesianAugmenter             │
    │  ┌───────────────────────────────┐  │
    │  │ Path 1: SuMemoryLitePro       │  │
    │  │   query() / predict() /       │  │
    │  │   reason() / add()            │  │
    │  └─────────────┬─────────────────┘  │
    │                │                    │
    │  ┌─────────────▼─────────────────┐  │
    │  │ Path 2: BayesianEngine +       │  │
    │  │   BayesianNetwork +            │  │
    │  │   EvidenceCollector            │  │
    │  └─────────────┬─────────────────┘  │
    │                │                    │
    │  ┌─────────────▼─────────────────┐  │
    │  │ Compare & Merge                │  │
    │  │ → comparison_delta             │  │
    │  │ → accuracy_tracking            │  │
    │  └───────────────────────────────┘  │
    └─────────────────────────────────────┘
        │
        ▼
    EnhancedOutput {original, bayesian, comparison}
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import time
import json
import math

from su_memory._sys.bayesian import (
    BayesianEngine,
    BetaDistribution,
    LikelihoodFunctions,
)
from su_memory._sys.bayesian_network import BayesianNetwork
from su_memory._sys.evidence import EvidenceCollector
from su_memory._sys.bayesian_reasoning import BayesianReasoningSystem, BayesianPredictor
from su_memory._sys.states import BayesianBeliefTracker


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ComparisonDelta:
    """双路径对比差异"""
    field: str
    original_value: Any
    bayesian_value: Any
    difference_description: str = ""
    improvement_indicator: str = ""  # "positive" | "negative" | "neutral"

    def to_dict(self) -> Dict:
        return {
            "field": self.field,
            "original_value": self.original_value,
            "bayesian_value": self.bayesian_value,
            "difference": self.difference_description,
            "improvement": self.improvement_indicator,
        }


@dataclass
class EnhancedOutput:
    """增强输出 — 同时包含原始和贝叶斯结果"""
    original: Dict              # 原始系统输出
    bayesian: Dict             # 贝叶斯增强输出
    comparisons: List[ComparisonDelta]  # 差异列表
    meta: Dict = field(default_factory=dict)  # 元信息

    def to_dict(self) -> Dict:
        return {
            "original_result": self.original,
            "bayesian_result": self.bayesian,
            "comparison_deltas": [c.to_dict() for c in self.comparisons],
            "meta": self.meta,
        }


@dataclass
class AccuracyRecord:
    """准确度追踪记录"""
    timestamp: float
    method: str              # "original" | "bayesian"
    query: str
    predicted_value: float
    actual_value: float
    error: float
    absolute_error: float


# ============================================================
# BayesianAugmenter
# ============================================================

class BayesianAugmenter:
    """
    贝叶斯推理串联增强器

    包装 SuMemoryLitePro，提供双路径对比验证。

    使用方式:
        >>> from su_memory.sdk.lite_pro import SuMemoryLitePro
        >>> from su_memory.sdk.bayesian_augmenter import BayesianAugmenter
        >>> 
        >>> client = SuMemoryLitePro()
        >>> augmenter = BayesianAugmenter(client)
        >>> 
        >>> # 双路径查询
        >>> result = augmenter.query("投资回报")
        >>> print(result.original)    # 原始结果
        >>> print(result.bayesian)    # 贝叶斯增强结果
        >>> print(result.comparisons) # 差异分析
        >>> 
        >>> # 反馈验证
        >>> augmenter.feedback(query="投资回报", expected_memory_ids=["mem_abc"])
        >>> 
        >>> # 查看准确度报告
        >>> report = augmenter.get_accuracy_report()
    """

    def __init__(
        self,
        client,  # SuMemoryLitePro 实例
        enable_network: bool = True,
        enable_predictor: bool = True,
        enable_auto_sync: bool = True,
        prior_type: str = "uniform",
        verbose: bool = False,
    ):
        """
        Args:
            client: SuMemoryLitePro 实例（必须）
            enable_network: 是否启用贝叶斯网络
            enable_predictor: 是否启用贝叶斯预测器
            enable_auto_sync: 是否自动从 client.add() 同步到贝叶斯
            prior_type: 先验类型
            verbose: 是否输出详细日志
        """
        self._client = client
        self._verbose = verbose

        # 贝叶斯推理系统
        self._brs = BayesianReasoningSystem(
            name="augmenter",
            prior_type=prior_type,
            enable_network=enable_network,
            enable_predictor=enable_predictor,
            enable_advisor=False,  # 暂不需要建议
        )

        # 快捷引用
        self.engine = self._brs.engine
        self.network = self._brs.network
        self.evidence = self._brs.evidence_collector
        self.predictor = self._brs.predictor

        # 准确度追踪
        self._accuracy_records: List[AccuracyRecord] = []
        self._feedback_count = 0

        # 自动同步配置
        self._enable_auto_sync = enable_auto_sync
        self._synced_memory_ids: Set[str] = set()

        # 如果启用自动同步，hook client 的 add 方法
        if enable_auto_sync:
            self._hook_client_add()

        if verbose:
            print(f"[BayesianAugmenter] 初始化完成 "
                  f"(网络={'启用' if enable_network else '禁用'}, "
                  f"预测={'启用' if enable_predictor else '禁用'}, "
                  f"自动同步={'启用' if enable_auto_sync else '禁用'})")

    def _hook_client_add(self):
        """Hook client.add() 以自动同步记忆到贝叶斯系统"""
        original_add = self._client.add

        def augmented_add(*args, **kwargs):
            memory_id = original_add(*args, **kwargs)

            # 同步到贝叶斯系统
            content = args[0] if args else kwargs.get("content", "")
            metadata = kwargs.get("metadata", {}) if "metadata" in kwargs else (args[1] if len(args) > 1 else {})

            if memory_id and memory_id not in self._synced_memory_ids:
                try:
                    self._sync_memory_to_bayesian(memory_id, content, metadata)
                except Exception as e:
                    if self._verbose:
                        print(f"[BayesianAugmenter] 同步失败: {e}")
                self._synced_memory_ids.add(memory_id)

            return memory_id

        self._client.add = augmented_add
        self._original_add = original_add

    def _sync_memory_to_bayesian(self, memory_id: str, content: str, metadata: Dict = None):
        """将记忆同步到贝叶斯系统"""
        metadata = metadata or {}

        # 注册信念
        tags = metadata.get("tags", [])
        category = metadata.get("category", metadata.get("type", "general"))
        self._brs.register_belief(
            belief_id=memory_id,
            content=content,
            category=category,
            tags=tags,
        )

        # 建立因果网络连接
        parent_ids = metadata.get("parent_ids", [])
        for parent_id in parent_ids:
            if parent_id in self._synced_memory_ids:
                try:
                    self._brs.add_causal_link(parent_id, memory_id)
                except ValueError:
                    pass  # 环路，跳过

    def _vlog(self, msg: str):
        """Verbose 日志"""
        if self._verbose:
            print(f"[BayesianAugmenter] {msg}")

    # ================================================================
    # 双路径查询 — query()
    # ================================================================

    def query(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> EnhancedOutput:
        """
        双路径查询

        路径1: client.query() — 原始混合检索（关键词+向量+时空）
        路径2: BayesianEngine — 后验概率排序 + 不确定性量化

        Returns:
            EnhancedOutput {original, bayesian, comparisons}
        """
        self._vlog(f"双路径查询: '{query}'")

        # ── 路径1: 原始查询 ──
        original_results = self._client.query(query, top_k=top_k, **kwargs)

        # ── 路径2: 贝叶斯增强 ──
        bayesian_results = self._bayesian_query(query, top_k, original_results)

        # ── 对比分析 ──
        comparisons = self._compare_query_results(
            query, original_results, bayesian_results
        )

        return EnhancedOutput(
            original={"results": original_results, "count": len(original_results)},
            bayesian={
                "results": bayesian_results,
                "count": len(bayesian_results),
                "engine_stats": self.engine.get_statistics(),
            },
            comparisons=comparisons,
            meta={
                "query": query,
                "top_k": top_k,
                "method": "dual_path_query",
                "timestamp": time.time(),
            }
        )

    def _bayesian_query(
        self,
        query: str,
        top_k: int,
        original_results: List[Dict]
    ) -> List[Dict]:
        """贝叶斯增强查询"""
        results = []

        for item in original_results:
            mem_id = item.get("memory_id", "")
            original_score = item.get("score", 0.0)

            # 从贝叶斯引擎获取后验置信度
            belief = self.engine.get_belief(mem_id)

            if belief and belief.posterior.effective_sample_size > 2:
                # 贝叶斯增强得分：融合原始得分和后验置信度
                bayesian_confidence = belief.posterior.mean
                bayesian_uncertainty = belief.posterior.std

                # 融合得分 = 原始得分 × 0.6 + 后验置信度 × 0.4
                # （保留原始检索信号，用贝叶斯信念修正）
                fused_score = original_score * 0.6 + bayesian_confidence * 0.4

                results.append({
                    **item,
                    "bayesian_confidence": bayesian_confidence,
                    "bayesian_uncertainty": bayesian_uncertainty,
                    "credible_interval_95": list(belief.posterior.credible_interval(0.95)),
                    "evidence_strength": belief.posterior.effective_sample_size,
                    "stage": belief.get_stage(),
                    "score": fused_score,  # 覆盖得分
                    "original_score": original_score,
                })
            else:
                # 无贝叶斯信息，保持原始
                results.append({
                    **item,
                    "bayesian_confidence": None,
                    "bayesian_uncertainty": None,
                    "stage": "no_bayesian_data",
                })

        # 按融合得分重新排序
        results.sort(key=lambda x: x.get("score", x.get("original_score", 0)), reverse=True)
        return results[:top_k]

    def _compare_query_results(
        self,
        query: str,
        original: List[Dict],
        bayesian: List[Dict]
    ) -> List[ComparisonDelta]:
        """对比原始和贝叶斯查询结果"""
        comparisons = []

        # 对比排序变化
        orig_ids = [r.get("memory_id") for r in original]
        bayes_ids = [r.get("memory_id") for r in bayesian]

        # Top-1 是否一致
        if orig_ids and bayes_ids:
            top1_same = orig_ids[0] == bayes_ids[0]
            comparisons.append(ComparisonDelta(
                field="top1_match",
                original_value=orig_ids[0],
                bayesian_value=bayes_ids[0],
                difference_description="一致" if top1_same else "贝叶斯重新排序, Top-1 不同",
                improvement_indicator="neutral" if top1_same else "positive"
            ))

        # 排序变化
        changed = sum(1 for i, bid in enumerate(bayes_ids) if i < len(orig_ids) and orig_ids[i] != bid)
        comparisons.append(ComparisonDelta(
            field="ranking_changes",
            original_value=f"{len(original)} results",
            bayesian_value=f"{changed} ranking changes in top-{min(len(bayesian), len(original))}",
            difference_description=f"贝叶斯调整了 {changed} 个结果的排序",
            improvement_indicator="positive" if changed > 0 else "neutral"
        ))

        # 有贝叶斯增强的结果数
        bayes_enhanced = sum(1 for r in bayesian if r.get("bayesian_confidence") is not None)
        comparisons.append(ComparisonDelta(
            field="bayesian_enhanced_count",
            original_value=0,
            bayesian_value=bayes_enhanced,
            difference_description=f"{bayes_enhanced}/{len(bayesian)} 结果有贝叶斯置信度增强",
            improvement_indicator="positive" if bayes_enhanced > 0 else "neutral"
        ))

        return comparisons

    # ================================================================
    # 双路径预测 — predict()
    # ================================================================

    def predict(
        self,
        query: str = None,
        top_k: int = 3,
        **kwargs
    ) -> EnhancedOutput:
        """
        双路径预测

        路径1: client.predict() — 原始预测（固定置信度）
        路径2: BayesianPredictor — 后验概率预测 + 置信区间 + 校准

        Returns:
            EnhancedOutput
        """
        self._vlog(f"双路径预测: '{query}'")

        # ── 路径1: 原始预测 ──
        try:
            original = self._client.predict(query=query, top_k=top_k, **kwargs)
        except Exception as e:
            original = {"error": str(e)}

        # ── 路径2: 贝叶斯预测 ──
        bayesian = self._bayesian_predict(query, top_k, original)

        # ── 对比分析 ──
        comparisons = self._compare_predictions(query, original, bayesian)

        return EnhancedOutput(
            original=original,
            bayesian=bayesian,
            comparisons=comparisons,
            meta={
                "query": query,
                "top_k": top_k,
                "method": "dual_path_predict",
                "timestamp": time.time(),
            }
        )

    def _bayesian_predict(
        self,
        query: str,
        top_k: int,
        original: Dict
    ) -> Dict:
        """贝叶斯增强预测"""
        results = {}

        if self.predictor is None:
            return {"error": "Predictor not enabled"}

        # 对原始预测中的事件做贝叶斯增强
        if isinstance(original, dict) and "event_predictions" in original:
            enhanced_events = []
            for pred in original.get("event_predictions", []):
                pred_content = pred.get("content", "")
                pred_confidence = pred.get("confidence", 0.5)

                # 为每个预测创建/更新贝叶斯信念
                belief_id = f"pred_{hash(pred_content) % 100000}"
                belief = self.engine.get_or_create(belief_id, content_summary=pred_content)

                # 贝叶斯增强的置信度
                bayes_conf = belief.posterior.mean if belief.posterior.effective_sample_size > 2 else pred_confidence
                bayes_uncert = belief.posterior.std if belief.posterior.effective_sample_size > 2 else None

                ci = belief.posterior.credible_interval(0.95) if belief.posterior.effective_sample_size > 2 else None

                enhanced_events.append({
                    **pred,
                    "original_confidence": pred_confidence,
                    "bayesian_confidence": bayes_conf,
                    "bayesian_uncertainty": bayes_uncert,
                    "credible_interval_95": list(ci) if ci else None,
                    "confidence_delta": bayes_conf - pred_confidence,
                    "stage": belief.get_stage(),
                })

            results["event_predictions"] = enhanced_events
        else:
            # 使用贝叶斯预测器直接预测
            if query:
                pred_result = self.predictor.predict_event_probability(query)
                results["bayesian_prediction"] = pred_result

        # 校准报告
        if self.predictor:
            results["calibration"] = self.predictor.get_calibration_report()

        results["engine_stats"] = self.engine.get_statistics()
        return results

    def _compare_predictions(
        self,
        query: str,
        original: Dict,
        bayesian: Dict
    ) -> List[ComparisonDelta]:
        """对比预测结果"""
        comparisons = []

        # 对比置信度
        if "event_predictions" in bayesian:
            for pred in bayesian["event_predictions"]:
                orig_conf = pred.get("original_confidence", 0)
                bayes_conf = pred.get("bayesian_confidence", 0)
                delta = pred.get("confidence_delta", 0)

                comparisons.append(ComparisonDelta(
                    field=f"prediction_confidence:{pred.get('content', '')[:30]}",
                    original_value=orig_conf,
                    bayesian_value=bayes_conf,
                    difference_description=(
                        f"贝叶斯{'上调' if delta > 0 else '下调'}置信度 {abs(delta):.3f}"
                    ),
                    improvement_indicator="positive" if abs(delta) > 0.05 else "neutral"
                ))

        # 是否有不确定性量化
        has_uncertainty = any(
            p.get("bayesian_uncertainty") is not None
            for p in bayesian.get("event_predictions", [])
        )
        comparisons.append(ComparisonDelta(
            field="uncertainty_quantification",
            original_value=False,
            bayesian_value=has_uncertainty,
            difference_description="原始预测无不确定性量化，贝叶斯提供标准差和置信区间",
            improvement_indicator="positive" if has_uncertainty else "neutral"
        ))

        return comparisons

    # ================================================================
    # 双路径推理 — reason()
    # ================================================================

    def reason(
        self,
        query: str,
        max_hops: int = 3,
        **kwargs
    ) -> EnhancedOutput:
        """
        双路径推理

        路径1: client.reason() — 原始推理（Energy动力学）
        路径2: BayesianNetwork — 后验概率推理 + 因果链置信度

        Returns:
            EnhancedOutput
        """
        self._vlog(f"双路径推理: '{query}'")

        # ── 路径1: 原始推理 ──
        try:
            original = self._client.reason(query, max_hops=max_hops, **kwargs)
        except Exception as e:
            original = {"error": str(e)}

        # ── 路径2: 贝叶斯推理 ──
        bayesian = self._bayesian_reason(query, max_hops, original)

        # ── 对比分析 ──
        comparisons = self._compare_reasoning(query, original, bayesian)

        return EnhancedOutput(
            original=original,
            bayesian=bayesian,
            comparisons=comparisons,
            meta={
                "query": query,
                "max_hops": max_hops,
                "method": "dual_path_reason",
                "timestamp": time.time(),
            }
        )

    def _bayesian_reason(
        self,
        query: str,
        max_hops: int,
        original: Dict
    ) -> Dict:
        """贝叶斯增强推理"""
        results = {}

        # 获取相关记忆
        query_results = self._client.query(query, top_k=10)

        # 为每个相关记忆注册信念
        memory_beliefs = []
        for item in query_results:
            mem_id = item.get("memory_id", "")
            belief = self.engine.get_belief(mem_id)

            if belief:
                memory_beliefs.append({
                    "memory_id": mem_id,
                    "content": item.get("content", ""),
                    "confidence": belief.posterior.mean,
                    "uncertainty": belief.posterior.std,
                    "credible_interval_95": list(belief.posterior.credible_interval(0.95)),
                    "stage": belief.get_stage(),
                    "evidence_strength": belief.posterior.effective_sample_size,
                })
            else:
                memory_beliefs.append({
                    "memory_id": mem_id,
                    "content": item.get("content", ""),
                    "confidence": None,
                    "stage": "no_bayesian_data",
                })

        results["memory_beliefs"] = memory_beliefs

        # 贝叶斯网络因果推理
        if self.network and query_results:
            # 尝试通过网络推断因果关系
            related_nodes = [r.get("memory_id") for r in query_results[:5]]
            causal_chains = []

            for i, node_a in enumerate(related_nodes):
                for node_b in related_nodes[i + 1:]:
                    # 检查双向因果强度
                    strength_ab = self.network.query_causal_strength(node_a, node_b)
                    strength_ba = self.network.query_causal_strength(node_b, node_a)

                    if strength_ab and strength_ab.get("evidence_count", 0) > 0:
                        causal_chains.append({
                            "from": node_a,
                            "to": node_b,
                            "strength": strength_ab["causal_strength"],
                            "relative_risk": strength_ab["relative_risk"],
                            "evidence_count": strength_ab["evidence_count"],
                        })
                    if strength_ba and strength_ba.get("evidence_count", 0) > 0:
                        causal_chains.append({
                            "from": node_b,
                            "to": node_a,
                            "strength": strength_ba["causal_strength"],
                            "relative_risk": strength_ba["relative_risk"],
                            "evidence_count": strength_ba["evidence_count"],
                        })

            # 按因果强度排序
            causal_chains.sort(key=lambda x: abs(x["strength"]), reverse=True)
            results["causal_chains"] = causal_chains[:10]

        # 原始推理的置信度 vs 贝叶斯置信度
        original_confidence = original.get("confidence") if isinstance(original, dict) else None
        if original_confidence is not None:
            results["original_confidence"] = original_confidence
            # 基于相关记忆的平均后验置信度
            bayes_confidences = [
                b["confidence"] for b in memory_beliefs
                if b["confidence"] is not None
            ]
            if bayes_confidences:
                results["bayesian_confidence"] = sum(bayes_confidences) / len(bayes_confidences)
                results["confidence_delta"] = results["bayesian_confidence"] - original_confidence

        results["engine_stats"] = self.engine.get_statistics()
        if self.network:
            results["network_stats"] = self.network.get_statistics()

        return results

    def _compare_reasoning(
        self,
        query: str,
        original: Dict,
        bayesian: Dict
    ) -> List[ComparisonDelta]:
        """对比推理结果"""
        comparisons = []

        # 置信度对比
        orig_conf = original.get("confidence") if isinstance(original, dict) else None
        bayes_conf = bayesian.get("bayesian_confidence")
        if orig_conf is not None and bayes_conf is not None:
            delta = bayes_conf - orig_conf
            comparisons.append(ComparisonDelta(
                field="reasoning_confidence",
                original_value=orig_conf,
                bayesian_value=bayes_conf,
                difference_description=(
                    f"贝叶斯{'上调' if delta > 0 else '下调'}推理置信度 {abs(delta):.3f}"
                ),
                improvement_indicator="positive"
            ))

        # 因果链是否存在
        causal_chains = bayesian.get("causal_chains", [])
        comparisons.append(ComparisonDelta(
            field="causal_chain_count",
            original_value="无因果链",
            bayesian_value=f"{len(causal_chains)} 条概率化因果链",
            difference_description="贝叶斯网络补充了因果强度量化",
            improvement_indicator="positive" if causal_chains else "neutral"
        ))

        # 记忆信念覆盖
        memory_beliefs = bayesian.get("memory_beliefs", [])
        with_beliefs = sum(1 for m in memory_beliefs if m.get("confidence") is not None)
        comparisons.append(ComparisonDelta(
            field="belief_coverage",
            original_value="无信念覆盖",
            bayesian_value=f"{with_beliefs}/{len(memory_beliefs)} 记忆有贝叶斯信念",
            difference_description=f"为 {with_beliefs} 条记忆提供了概率化信念状态",
            improvement_indicator="positive" if with_beliefs > 0 else "neutral"
        ))

        return comparisons

    # ================================================================
    # 反馈闭环
    # ================================================================

    def feedback(
        self,
        query: str,
        expected_memory_ids: List[str] = None,
        expected_outcome: bool = None,
        is_correct: bool = None,
        ground_truth_value: float = None,
    ) -> Dict:
        """
        用户反馈 — 闭合贝叶斯更新回路

        用于对比验证两个系统的准确度：
        1. 更新贝叶斯引擎的后验概率
        2. 记录原始 vs 贝叶斯的准确度差异
        3. 如果提供了 ground_truth_value，更新预测校准

        Args:
            query: 之前的查询
            expected_memory_ids: 期望检索到的记忆ID
            expected_outcome: 期望的预测结果（True/False）
            is_correct: 预测是否正确
            ground_truth_value: 真实值（用于评估概率准确度）

        Returns:
            反馈处理结果
        """
        self._feedback_count += 1
        self._vlog(f"收到反馈 #{self._feedback_count}: query='{query}'")

        result = {"feedback_id": self._feedback_count}

        # 1. 更新贝叶斯信念
        if expected_memory_ids:
            # 将命中的记忆作为正面证据
            for mem_id in expected_memory_ids:
                self.engine.observe(
                    belief_id=mem_id,
                    success=True,
                    weight=1.0,
                    source="user_feedback",
                    note=f"Relevant to: {query}"
                )

            # 获取之前查询的结果，将未命中的标记为负面证据
            try:
                prev_results = self._client.query(query, top_k=10)
                for item in prev_results:
                    mem_id = item.get("memory_id", "")
                    if mem_id and mem_id not in expected_memory_ids:
                        self.engine.observe(
                            belief_id=mem_id,
                            success=False,
                            weight=0.3,  # 未命中证据权重较低
                            source="user_feedback",
                            note=f"Not relevant to: {query}"
                        )
            except Exception:
                pass

            result["beliefs_updated"] = len(expected_memory_ids)

        # 2. 记录准确度（支持 is_correct + ground_truth，也支持仅 ground_truth）
        if ground_truth_value is not None:
            # 评估原始系统的概率
            original_prob = 0.5  # 默认
            try:
                prev_pred = self._client.predict(query=query)
                events = prev_pred.get("event_predictions", [])
                if events:
                    original_prob = events[0].get("confidence", 0.5)
            except Exception:
                pass

            # 评估贝叶斯系统的概率
            bayesian_prob = 0.5
            belief = self.engine.get_belief(f"pred_{hash(query) % 100000}")
            if belief and belief.posterior.effective_sample_size > 2:
                bayesian_prob = belief.posterior.mean

            # 记录两条路径的误差
            error_original = abs(original_prob - ground_truth_value)
            error_bayesian = abs(bayesian_prob - ground_truth_value)

            self._accuracy_records.append(AccuracyRecord(
                timestamp=time.time(),
                method="original",
                query=query,
                predicted_value=original_prob,
                actual_value=ground_truth_value,
                error=original_prob - ground_truth_value,
                absolute_error=error_original,
            ))
            self._accuracy_records.append(AccuracyRecord(
                timestamp=time.time(),
                method="bayesian",
                query=query,
                predicted_value=bayesian_prob,
                actual_value=ground_truth_value,
                error=bayesian_prob - ground_truth_value,
                absolute_error=error_bayesian,
            ))

            result["accuracy"] = {
                "original_error": error_original,
                "bayesian_error": error_bayesian,
                "improvement": (
                    (error_original - error_bayesian) / max(error_original, 0.001) * 100
                    if error_original > 0 else 0
                )
            }

        # 3. 预测校准反馈
        if expected_outcome is not None:
            try:
                prev_pred = self._client.predict(query=query)
                events = prev_pred.get("event_predictions", [])
                if events:
                    pred_prob = events[0].get("confidence", 0.5)
                    self._brs.record_outcome(
                        event_id=f"pred_{hash(query) % 100000}",
                        predicted_prob=pred_prob,
                        actual_outcome=expected_outcome,
                    )
                    result["calibration_updated"] = True
            except Exception:
                pass

        return result

    # ================================================================
    # 准确度报告
    # ================================================================

    def get_accuracy_report(self) -> Dict:
        """
        获取双路径准确度对比报告

        Returns:
            {
                "summary": {...},
                "original_stats": {...},
                "bayesian_stats": {...},
                "improvement": float,
                "records_count": int,
                "recommendation": str,
            }
        """
        if not self._accuracy_records:
            return {
                "status": "no_data",
                "summary": {
                    "total_feedback": self._feedback_count,
                    "total_records": 0,
                    "improvement_pct": 0,
                    "verdict": "尚无反馈数据",
                    "recommendation": "对若干查询调用 augmenter.feedback() 来收集对比数据"
                },
                "message": "尚无反馈数据。使用 feedback() 方法提供验证数据。",
                "suggestion": "对若干查询调用 augmenter.feedback() 来收集对比数据。"
            }

        original_records = [r for r in self._accuracy_records if r.method == "original"]
        bayesian_records = [r for r in self._accuracy_records if r.method == "bayesian"]

        def compute_stats(records: List[AccuracyRecord]) -> Dict:
            if not records:
                return {}
            mae = sum(r.absolute_error for r in records) / len(records)
            mse = sum(r.error ** 2 for r in records) / len(records)
            rmse = math.sqrt(mse)
            errors = sorted([r.absolute_error for r in records])

            return {
                "count": len(records),
                "mae": mae,
                "rmse": rmse,
                "median_error": errors[len(errors) // 2] if errors else 0,
                "p90_error": errors[int(len(errors) * 0.9)] if len(errors) >= 10 else errors[-1] if errors else 0,
                "mean_bias": sum(r.error for r in records) / len(records),
            }

        orig_stats = compute_stats(original_records)
        bayes_stats = compute_stats(bayesian_records)

        improvement = (
            (orig_stats.get("mae", 0) - bayes_stats.get("mae", 0))
            / max(orig_stats.get("mae", 0.001), 0.001) * 100
        )

        # 判定置信度
        if improvement > 10:
            verdict = "贝叶斯方法显著优于原始方法"
            recommendation = "建议在生产环境中启用贝叶斯增强"
        elif improvement > 3:
            verdict = "贝叶斯方法略有优势"
            recommendation = "建议继续积累反馈数据验证稳定性"
        elif improvement > -3:
            verdict = "两者表现相当"
            recommendation = "贝叶斯方法提供额外的不确定性量化能力"
        else:
            verdict = "原始方法在当前数据上更优"
            recommendation = "建议检查贝叶斯先验配置或增加反馈数据量"

        return {
            "summary": {
                "total_feedback": self._feedback_count,
                "total_records": len(self._accuracy_records),
                "improvement_pct": improvement,
                "verdict": verdict,
                "recommendation": recommendation,
            },
            "original_stats": orig_stats,
            "bayesian_stats": bayes_stats,
            "engine_stats": self.engine.get_statistics(),
            "network_stats": self.network.get_statistics() if self.network else None,
            "calibration": (
                self.predictor.get_calibration_report() if self.predictor else None
            ),
        }

    def print_accuracy_report(self):
        """打印格式化的准确度报告"""
        report = self.get_accuracy_report()
        print("\n" + "=" * 60)
        print("📊 BayesianAugmenter 双路径准确度对比报告")
        print("=" * 60)

        if report.get("status") == "no_data":
            print(f"\n  ⚠️  {report['message']}")
            print(f"  💡 {report['suggestion']}")
            print("\n" + "=" * 60)
            return

        summary = report["summary"]
        orig = report["original_stats"]
        bayes = report["bayesian_stats"]

        print(f"\n  📈 总体: {summary['total_feedback']} 次反馈, {summary['total_records']} 条准确度记录")
        print(f"\n  {'指标':<15} {'原始':>10} {'贝叶斯':>10} {'改善':>10}")
        print(f"  {'─'*15} {'─'*10} {'─'*10} {'─'*10}")
        print(f"  {'MAE':<15} {orig.get('mae', 0):>10.4f} {bayes.get('mae', 0):>10.4f} {summary['improvement_pct']:>+9.1f}%")
        print(f"  {'RMSE':<15} {orig.get('rmse', 0):>10.4f} {bayes.get('rmse', 0):>10.4f}")
        print(f"  {'Median Error':<15} {orig.get('median_error', 0):>10.4f} {bayes.get('median_error', 0):>10.4f}")
        print(f"  {'P90 Error':<15} {orig.get('p90_error', 0):>10.4f} {bayes.get('p90_error', 0):>10.4f}")
        print(f"  {'Bias':<15} {orig.get('mean_bias', 0):>10.4f} {bayes.get('mean_bias', 0):>10.4f}")

        print(f"\n  🎯 判定: {summary['verdict']}")
        print(f"  💡 建议: {summary['recommendation']}")

        if report.get("calibration") and report["calibration"].get("status") != "insufficient_data":
            cal = report["calibration"]
            print(f"\n  📏 预测校准:")
            print(f"     状态: {cal.get('status', 'N/A')}")
            print(f"     Brier Score: {cal.get('brier_score', 0):.4f}")
            print(f"     校准偏置: {cal.get('calibration_bias', 0):.4f}")

        print("\n" + "=" * 60)

    # ================================================================
    # 批量对比验证
    # ================================================================

    def run_validation_suite(
        self,
        test_queries: List[Dict],
        verbose: bool = True
    ) -> Dict:
        """
        运行批量对比验证

        Args:
            test_queries: [
                {
                    "query": str,
                    "expected_memory_ids": [str, ...],
                    "expected_outcome": bool (optional),
                    "ground_truth_value": float (optional),
                },
                ...
            ]

        Returns:
            验证结果汇总
        """
        results = []
        for tc in test_queries:
            query = tc["query"]

            # 双路径查询
            result = self.query(query)

            # 反馈
            feedback = self.feedback(
                query=query,
                expected_memory_ids=tc.get("expected_memory_ids"),
                expected_outcome=tc.get("expected_outcome"),
                ground_truth_value=tc.get("ground_truth_value"),
            )

            results.append({
                "query": query,
                "top1_match": (
                    result.original["results"][0]["memory_id"]
                    if result.original["results"] else None
                ),
                "bayesian_top1": (
                    result.bayesian["results"][0]["memory_id"]
                    if result.bayesian["results"] else None
                ),
                "expected": tc.get("expected_memory_ids", [None])[0] if tc.get("expected_memory_ids") else None,
                "original_correct": (
                    result.original["results"][0]["memory_id"] in tc.get("expected_memory_ids", [])
                    if result.original["results"] and tc.get("expected_memory_ids") else None
                ),
                "bayesian_correct": (
                    result.bayesian["results"][0]["memory_id"] in tc.get("expected_memory_ids", [])
                    if result.bayesian["results"] and tc.get("expected_memory_ids") else None
                ),
                "feedback": feedback,
            })

        # 统计
        orig_correct = sum(1 for r in results if r["original_correct"] is True)
        bayes_correct = sum(1 for r in results if r["bayesian_correct"] is True)
        total_tested = sum(1 for r in results if r["original_correct"] is not None)

        report = self.get_accuracy_report()

        if verbose:
            print("\n" + "=" * 60)
            print("📋 批量对比验证结果")
            print("=" * 60)
            print(f"\n  测试用例: {len(test_queries)}")
            if total_tested > 0:
                print(f"  原始 Top-1 准确率: {orig_correct}/{total_tested} ({orig_correct/total_tested*100:.1f}%)")
                print(f"  贝叶斯 Top-1 准确率: {bayes_correct}/{total_tested} ({bayes_correct/total_tested*100:.1f}%)")
                improv = (bayes_correct - orig_correct) / max(total_tested, 1) * 100
                print(f"  准确率差异: {improv:+.1f}%")
            print(f"\n  误差改善: {report.get('summary', {}).get('improvement_pct', 0):+.1f}%")
            print("=" * 60)

        return {
            "results": results,
            "summary": {
                "test_count": len(test_queries),
                "original_accuracy": orig_correct / max(total_tested, 1) if total_tested > 0 else None,
                "bayesian_accuracy": bayes_correct / max(total_tested, 1) if total_tested > 0 else None,
            },
            "accuracy_report": report,
        }

    # ================================================================
    # 访问原始客户端（透传未包装的方法）
    # ================================================================

    def __getattr__(self, name):
        """
        未包装的方法直接透传到原始客户端

        如：add(), query_multihop(), explain_query() 等
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._client, name)

    # ================================================================
    # 持久化
    # ================================================================

    def save_state(self, path: str = None) -> str:
        """保存贝叶斯增强器状态"""
        path = path or f"bayesian_augmenter_state_{int(time.time())}.json"
        state = {
            "brs": self._brs.to_dict(),
            "accuracy_records": [
                {
                    "timestamp": r.timestamp,
                    "method": r.method,
                    "query": r.query,
                    "predicted_value": r.predicted_value,
                    "actual_value": r.actual_value,
                    "error": r.error,
                    "absolute_error": r.absolute_error,
                }
                for r in self._accuracy_records
            ],
            "feedback_count": self._feedback_count,
            "synced_ids": list(self._synced_memory_ids),
        }
        with open(path, "w") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        return path

    def load_state(self, path: str):
        """恢复贝叶斯增强器状态"""
        with open(path, "r") as f:
            state = json.load(f)

        self._brs = BayesianReasoningSystem.from_dict(state.get("brs", {}))
        self.engine = self._brs.engine
        self.network = self._brs.network
        self.evidence = self._brs.evidence_collector
        self.predictor = self._brs.predictor

        self._accuracy_records = [
            AccuracyRecord(**r) for r in state.get("accuracy_records", [])
        ]
        self._feedback_count = state.get("feedback_count", 0)
        self._synced_memory_ids = set(state.get("synced_ids", []))

    # ================================================================
    # 工具方法
    # ================================================================

    def reset_bayesian(self):
        """重置贝叶斯系统（保留原始系统不变）"""
        self._brs.reset()
        self._accuracy_records.clear()
        self._feedback_count = 0
        self._vlog("贝叶斯系统已重置，原始系统未受影响")

    def get_bayesian_engine(self):
        """获取底层 BayesianEngine（高级用途）"""
        return self.engine

    def get_bayesian_network(self):
        """获取底层 BayesianNetwork（高级用途）"""
        return self.network
