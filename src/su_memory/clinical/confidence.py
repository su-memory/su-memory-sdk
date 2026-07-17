"""
confidence — 记忆置信度增强

用贝叶斯推理评估医疗记忆的可靠性，影响检索排序。

⚠️ 项目区隔：本模块评估的是「这条记忆有多可靠」（检索置信度），
不是「患者风险概率」（临床预测由 MCI World Model 负责）。

工作原理：
  Beta-Binomial 模型评估记忆可靠性：
  - 先验: α=1, β=1（均匀先验，新记忆置信度 = 0.5）
  - 正反馈（被引用/被好评）: α += 1 → 置信度↑
  - 负反馈（被纠错/被差评）: β += 1 → 置信度↓
  - 后验均值 = α / (α + β)

  时间衰减：旧记忆置信度按半衰期衰减（默认 180 天）

Example:
  >>> from su_memory.clinical.confidence import ConfidenceTracker
  >>> tracker = ConfidenceTracker(memory_client)
  >>> tracker.record_positive("mem_abc123")  # 正反馈
  >>> score = tracker.get_confidence("mem_abc123")  # → 0.67
  >>> boosted = tracker.rerank_by_confidence(query_results)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from su_memory.sdk.lite_pro import SuMemoryLitePro

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 置信度记录
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ConfidenceRecord:
    """单条记忆的贝叶斯置信度记录。

    Attributes:
        memory_id: 记忆 ID
        alpha: Beta 分布 α（正证据计数 + 1）
        beta: Beta 分布 β（负证据计数 + 1）
        last_updated: 最后更新时间戳
        evidence_sources: 证据来源记录 ["feedback_5star", "cited_by_doctor", ...]
    """

    memory_id: str
    alpha: float = 1.0
    beta: float = 1.0
    last_updated: float = field(default_factory=time.time)
    evidence_sources: list[str] = field(default_factory=list)

    @property
    def posterior_mean(self) -> float:
        """Beta(α,β) 后验均值 = α/(α+β)"""
        return self.alpha / (self.alpha + self.beta)

    def time_decay(self, half_life_days: float = 180.0) -> float:
        """时间衰减因子（指数衰减，半衰期默认 180 天）

        Returns:
            衰减因子 [0, 1]，新记忆 = 1.0
        """
        age_days = (time.time() - self.last_updated) / 86400.0
        if age_days <= 0:
            return 1.0
        return math.exp(-0.693 * age_days / half_life_days)


# ═══════════════════════════════════════════════════════════════════
# 置信度追踪器
# ═══════════════════════════════════════════════════════════════════


class ConfidenceTracker:
    """记忆置信度追踪器 — 贝叶斯 Beta-Binomial 模型。

    用法：
        tracker = ConfidenceTracker(memory_client)
        tracker.inject_hooks()  # 注入 query 后置钩子，自动重排序

    或手动：
        tracker.record_positive("mem_001", source="doctor_cited")
        score = tracker.get_confidence("mem_001")
        reranked = tracker.rerank_by_confidence(results)
    """

    def __init__(
        self,
        client: SuMemoryLitePro | None = None,
        half_life_days: float = 180.0,
    ):
        self._client = client
        self._half_life_days = half_life_days
        self._records: dict[str, ConfidenceRecord] = {}

    def get_confidence(self, memory_id: str) -> float:
        """获取记忆置信度（贝叶斯后验 × 时间衰减）

        Returns:
            置信度 [0, 1]
        """
        rec = self._records.get(memory_id)
        if rec is None:
            return 0.5  # 新记忆默认置信度
        return rec.posterior_mean * rec.time_decay(self._half_life_days)

    def record_positive(
        self, memory_id: str, source: str = "feedback", weight: float = 1.0
    ) -> float:
        """记录正反馈（记忆被引用/好评/被采纳）

        Returns:
            更新后的置信度
        """
        rec = self._records.setdefault(memory_id, ConfidenceRecord(memory_id=memory_id))
        rec.alpha += weight
        rec.last_updated = time.time()
        if source not in rec.evidence_sources:
            rec.evidence_sources.append(source)
        return self.get_confidence(memory_id)

    def record_negative(
        self, memory_id: str, source: str = "rejection", weight: float = 1.0
    ) -> float:
        """记录负反馈（记忆被纠错/差评/被拒绝）

        Returns:
            更新后的置信度
        """
        rec = self._records.setdefault(memory_id, ConfidenceRecord(memory_id=memory_id))
        rec.beta += weight
        rec.last_updated = time.time()
        if source not in rec.evidence_sources:
            rec.evidence_sources.append(source)
        return self.get_confidence(memory_id)

    def rerank_by_confidence(
        self, results: list[dict], blend: float = 0.3
    ) -> list[dict]:
        """用置信度重排序检索结果。

        将原始检索分数与贝叶斯置信度按 blend 比例混合：
            final_score = (1-blend) * original_score + blend * confidence

        Args:
            results: query 返回的 dict 列表（含 memory_id 和 score）
            blend: 置信度权重 [0, 1]。0 = 纯原始排序，1 = 纯置信度排序

        Returns:
            重排序后的结果列表
        """
        if not results:
            return results

        enriched = []
        for r in results:
            mem_id = r.get("memory_id", r.get("id", ""))
            conf = self.get_confidence(mem_id)
            orig_score = r.get("score", 0.5)
            final_score = (1 - blend) * orig_score + blend * conf
            enriched.append({**r, "confidence": conf, "score": final_score})

        enriched.sort(key=lambda x: x["score"], reverse=True)
        return enriched

    def inject_hooks(self, client: SuMemoryLitePro) -> None:
        """注入 query 后置钩子，自动重排序。

        在 client.query() 返回结果后，自动用置信度重排序。

        Args:
            client: SuMemoryLitePro 实例
        """
        self._client = client
        original_query = client.query

        def hooked_query(query: str, top_k: int = 5, **kwargs) -> list[dict]:
            results = original_query(query, top_k=top_k, **kwargs)
            try:
                return self.rerank_by_confidence(results)
            except Exception as e:
                logger.debug("置信度重排序降级（非阻塞）: %s", e)
                return results

        client.query = hooked_query
        logger.info("[ConfidenceTracker] 已注入置信度重排序钩子")

    def get_stats(self) -> dict[str, Any]:
        """获取置信度统计"""
        if not self._records:
            return {"total": 0, "avg_confidence": 0.5}

        confidences = [self.get_confidence(mid) for mid in self._records]
        return {
            "total": len(self._records),
            "avg_confidence": sum(confidences) / len(confidences),
            "high_confidence": sum(1 for c in confidences if c > 0.7),
            "low_confidence": sum(1 for c in confidences if c < 0.3),
        }

    def export_records(self) -> list[dict]:
        """导出置信度记录（供持久化）"""
        return [
            {
                "memory_id": r.memory_id,
                "alpha": r.alpha,
                "beta": r.beta,
                "posterior_mean": r.posterior_mean,
                "confidence": self.get_confidence(r.memory_id),
                "evidence_sources": r.evidence_sources,
                "last_updated": r.last_updated,
            }
            for r in self._records.values()
        ]
