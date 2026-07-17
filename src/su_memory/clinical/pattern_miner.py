"""
pattern_miner — 临床模式提炼

从大量患者记忆中发现临床决策模式，转化为可执行的临床规则。

⚠️ 项目区隔：模式发现基于记忆聚类统计，不做因果建模（后者归 World Model）。

工作原理：
  1. 调用 SDK 已有的 distill_patterns() / extract_rules() 做记忆聚类
  2. 用 P1-S1 的 MedicalAssociationKB 识别聚类中的医疗实体
  3. 提炼为结构化 ClinicalPattern（条件→结论 + 支持度 + 置信度）
  4. 高质量模式可注入关联图强化 P1-S1 的关联边权重

Example:
  >>> from su_memory.clinical import MedicalAssociationKB, ClinicalPatternMiner
  >>> miner = ClinicalPatternMiner(client, MedicalAssociationKB())
  >>> patterns = miner.mine_patterns()
  >>> for p in patterns:
  ...     print(f"{p.condition} → {p.conclusion} (support={p.support})")
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from su_memory.clinical.association_kb import MedicalAssociationKB
    from su_memory.sdk.lite_pro import SuMemoryLitePro

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 临床模式数据结构
# ═══════════════════════════════════════════════════════════════════


@dataclass(eq=False)
class ClinicalPattern:
    """一个临床决策模式。

    Attributes:
        pattern_id: 模式 ID
        condition: 条件描述（如"糖尿病肾病患者"）
        conclusion: 结论描述（如"适合低蛋白饮食"）
        support: 支持度（有多少条记忆支撑）
        confidence: 置信度 [0,1]
        sample_memory_ids: 样例记忆 ID 列表
        associated_rules: 关联到的医疗知识规则 ID
    """

    pattern_id: str
    condition: str = ""
    conclusion: str = ""
    support: int = 0
    confidence: float = 0.0
    sample_memory_ids: list[str] = field(default_factory=list)
    associated_rules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "condition": self.condition,
            "conclusion": self.conclusion,
            "support": self.support,
            "confidence": round(self.confidence, 3),
            "sample_count": len(self.sample_memory_ids),
            "associated_rules": self.associated_rules,
        }


# ═══════════════════════════════════════════════════════════════════
# 临床模式提炼器
# ═══════════════════════════════════════════════════════════════════


class ClinicalPatternMiner:
    """临床模式提炼器 — 从记忆聚类中发现临床决策模式。

    用法：
        miner = ClinicalPatternMiner(client, association_kb)
        patterns = miner.mine_patterns(min_support=3)
    """

    def __init__(
        self,
        client: SuMemoryLitePro,
        association_kb: MedicalAssociationKB | None = None,
    ):
        self._client = client
        self._kb = association_kb

    def mine_patterns(
        self,
        min_support: int = 3,
        min_confidence: float = 0.5,
    ) -> list[ClinicalPattern]:
        """从记忆库中提炼临床模式。

        Args:
            min_support: 最小支持度（至少多少条记忆支撑一个模式）
            min_confidence: 最小置信度

        Returns:
            ClinicalPattern 列表，按支持度降序
        """
        # Step 1: 调用 SDK 已有的 distill_patterns 做聚类
        try:
            raw = self._client.distill_patterns()
        except Exception as e:
            logger.debug("[PatternMiner] distill_patterns 降级: %s", e)
            return []

        clusters = raw.get("patterns", {})
        if not clusters:
            return []

        patterns: list[ClinicalPattern] = []

        # Step 2: 每个聚类转化为临床模式
        for cluster_key, cluster_data in clusters.items():
            size = cluster_data.get("size", 0)
            if size < min_support:
                continue

            sample_contents = cluster_data.get("sample_contents", [])
            if not sample_contents:
                continue

            # Step 3: 识别聚类中的医疗实体
            condition, conclusion, rule_ids = self._extract_medical_semantics(
                sample_contents
            )

            # 置信度 = 聚类纯度 × 关联规则匹配率
            confidence = min(size / 10.0, 1.0)  # 支持度归一化
            if rule_ids:
                confidence = min(confidence + 0.2, 1.0)  # 有医疗关联加成

            if confidence < min_confidence and not condition:
                continue

            pattern = ClinicalPattern(
                pattern_id=f"pat_{cluster_key}_{size}",
                condition=condition,
                conclusion=conclusion,
                support=size,
                confidence=confidence,
                sample_memory_ids=self._find_memory_ids(sample_contents),
                associated_rules=rule_ids,
            )
            patterns.append(pattern)

        patterns.sort(key=lambda p: p.support, reverse=True)
        logger.info(
            "[PatternMiner] 提炼出 %d 个临床模式（min_support=%d）",
            len(patterns), min_support,
        )
        return patterns

    def _extract_medical_semantics(
        self, contents: list[str]
    ) -> tuple[str, str, list[str]]:
        """从聚类内容中提取医疗语义（条件/结论/关联规则）。

        Returns:
            (condition, conclusion, associated_rule_ids)
        """
        if not self._kb:
            # 无知识库时用关键词频率做简单提取
            return self._keyword_based_extraction(contents)

        # 用关联知识库识别医疗实体
        all_source_hits: Counter = Counter()
        all_target_hits: Counter = Counter()
        rule_ids: set[str] = set()

        for content in contents:
            matches = self._kb.match(content)
            for rule, end in matches:
                if end == "source":
                    all_source_hits[rule.source_patterns[0]] += 1
                else:
                    all_target_hits[rule.target_patterns[0]] += 1
                rule_ids.add(rule.rule_id)

        # 高频 source = 条件，高频 target = 结论
        condition = all_source_hits.most_common(1)[0][0] if all_source_hits else ""
        conclusion = all_target_hits.most_common(1)[0][0] if all_target_hits else ""

        if not condition and not conclusion:
            return self._keyword_based_extraction(contents)

        return condition, conclusion, list(rule_ids)

    def _keyword_based_extraction(
        self, contents: list[str]
    ) -> tuple[str, str, list[str]]:
        """无知识库时的关键词频率提取"""
        # 取所有内容的公共子串片段
        word_freq: Counter = Counter()
        for content in contents:
            # 简单 2-gram 提取
            for i in range(len(content) - 1):
                word_freq[content[i:i + 2]] += 1

        top_words = [w for w, _ in word_freq.most_common(3) if len(w) >= 2]
        condition = top_words[0] if top_words else ""
        conclusion = top_words[1] if len(top_words) > 1 else ""
        return condition, conclusion, []

    def _find_memory_ids(self, sample_contents: list[str]) -> list[str]:
        """根据内容反查 memory_id"""
        ids: list[str] = []
        graph = getattr(self._client, "_graph", None)
        if graph is None:
            return ids[:5]
        for mem_id, node in graph._nodes.items():
            if node.content in sample_contents:
                ids.append(mem_id)
        return ids[:5]

    def patterns_to_rules(
        self, patterns: list[ClinicalPattern]
    ) -> list[dict[str, Any]]:
        """将模式转化为可执行的临床规则。

        Returns:
            [{"if": condition, "then": conclusion, "support": N, "confidence": P}, ...]
        """
        rules: list[dict[str, Any]] = []
        for p in patterns:
            if not p.condition or not p.conclusion:
                continue
            rules.append({
                "rule_id": p.pattern_id,
                "if": p.condition,
                "then": p.conclusion,
                "support": p.support,
                "confidence": round(p.confidence, 3),
                "evidence": f"基于{p.support}条临床记忆提炼",
                "associated_rules": p.associated_rules,
            })
        return rules
