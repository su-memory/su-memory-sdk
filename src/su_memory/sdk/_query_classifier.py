"""
su-memory v4.0 — Query-Aware Dynamic Retrieval Classifier (查询感知动态检索分类器)

根据问题类型动态选择检索通道、参数和后处理策略，
替代 one-size-fits-all 检索管道。

核心能力:
- 问题类型分类: TR/KU/MS/SSU/SSA/SSP → 结构化 QueryPlan
- 动态检索通道选择: event_time / vector / keyword / graph
- 后处理策略选择: recency / entity / session_ndcg / aggregation
- 上下文格式选择: temporal_chain / latest_first / aggregation

参考: Chronos (95.6%), EmergenceMem (86%), ByteRover (92.8%)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from su_memory.sdk._temporal_parser import TemporalParser, create_temporal_parser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class QueryPlan:
    """结构化检索计划 — 根据问题类型动态生成"""
    # 检索通道优先级
    primary_channels: list[str] = field(default_factory=lambda: ["vector", "keyword"])
    # 时间过滤（如有）
    time_filter: Any | None = None  # TemporalRange
    # 目标实体
    target_entities: list[str] = field(default_factory=list)
    # 重排策略
    rerank_strategy: str = "default"  # default / recency / entity / session_ndcg
    # 上下文格式
    context_format: str = "default"  # default / temporal_chain / latest_first / aggregation
    # 是否需要多子查询分解
    needs_decomposition: bool = False
    # 子查询列表（MS 问题分解后填充）
    sub_queries: list[str] = field(default_factory=list)
    # top_k 倍率（相对于基础 top_k）
    top_k_multiplier: float = 1.0
    # 事实优先权重（0-1，1=完全使用事实）
    fact_priority: float = 0.5


# ---------------------------------------------------------------------------
# 问题类型检测关键词
# ---------------------------------------------------------------------------

# 时序推理 — 包含时间约束
_TR_PATTERNS = [
    r'\bwhen\b', r'\bbefore\b', r'\bafter\b', r'\bfirst\b', r'\blast\b',
    r'\bmost recent\b', r'\bearliest\b', r'\bprevious\b', r'\bnext\b',
    r'\bhow (?:many|much|long|often|far)\b',
    r'\bthen\b', r'\bsince\b', r'\buntil\b', r'\bduring\b',
    r'\b(in|on|at)\s+(January|February|March|April|May|June|July|August|September|October|November|December)\b',
    r'\b\d{4}\b',  # 年份
    r'\b(last|past|this|previous)\s+(week|month|year)\b',
    r'\b(days?|weeks?|months?|years?)\s+ago\b',
    r'\bchronolog', r'\border\b', r'\bsequence\b', r'\btimeline\b',
]

# 知识更新 — 最新值查询
_KU_PATTERNS = [
    r'\bcurrent\b', r'\bnow\b', r'\bupdated\b', r'\bchanged\b',
    r'\bnew\b', r'\blatest\b', r'\bcurrently\b', r'\bpresent\b',
    r'\bwhat (?:is|are)\s+(?:the\s+)?(?:current|latest|new|present)\b',
    r'\bwhere (?:is|does)\s+(?:.*?\s+)?(?:now|currently|lately)\b',
    r'\bwho (?:is|does)\s+(?:.*?\s+)?(?:now|currently)\b',
]

# 多会话推理 — 聚合/计数
_MS_PATTERNS = [
    r'\bhow many\b', r'\bhow much\b', r'\bhow often\b',
    r'\ball (?:the\s+)?(?:times|occasions|instances)\b',
    r'\bevery\s+(?:time|occasion)\b',
    r'\btotal\b', r'\bcount\b', r'\bnumber of\b',
    r'\beach\b', r'\blist\s+(?:all|every)\b',
    r'\bcompare\b', r'\bdifference\b', r'\bsimilarit',
]

# 单会话偏好
_SSP_PATTERNS = [
    r'\bfavorite\b', r'\bfavourite\b', r'\bprefer\b', r'\bpreference\b',
    r'\blike\b.*\b(?:most|best)\b', r'\blove\b.*\b(?:most|best)\b',
    r'\bbest\b', r'\bfavorite\b', r'\btop\b.*\b(?:pick|choice|favorite)\b',
    r'\bopinion\b', r'\bthink\b.*\b(?:about|of)\b',
    r'\bdislike\b', r'\bhate\b', r'\bavoid\b',
]


# ---------------------------------------------------------------------------
# QueryClassifier 主类
# ---------------------------------------------------------------------------

class QueryClassifier:
    """查询感知动态检索分类器 — 根据问题类型生成结构化检索计划

    分类规则（基于 LongMemEval 问题模式）:
    - temporal-reasoning → 时间过滤 + 事件索引 + 正序排列
    - knowledge-update → 实体匹配 + 时间倒序 + 最新值优先
    - multi-session → 多子查询分解 + 聚合
    - single-session-preference → 全量session扫描 + LLM判断
    - single-session-user/assistant → 标准向量+关键词检索

    用法:
        classifier = QueryClassifier()
        plan = classifier.classify("When did I last visit Paris?")
        # plan.rerank_strategy = "recency"
        # plan.context_format = "temporal_chain"
    """

    def __init__(self, temporal_parser: TemporalParser | None = None):
        self._parser = temporal_parser or create_temporal_parser()
        # 编译正则
        self._tr_patterns = [re.compile(p, re.IGNORECASE) for p in _TR_PATTERNS]
        self._ku_patterns = [re.compile(p, re.IGNORECASE) for p in _KU_PATTERNS]
        self._ms_patterns = [re.compile(p, re.IGNORECASE) for p in _MS_PATTERNS]
        self._ssp_patterns = [re.compile(p, re.IGNORECASE) for p in _SSP_PATTERNS]

    def classify(
        self,
        question: str,
        qtype: str | None = None,
        reference_date: str | None = None,
    ) -> QueryPlan:
        """根据问题类型生成结构化检索计划

        Args:
            question: 查询问题文本
            qtype: 问题类型标签（如已知则直接使用，否则自动分类）
            reference_date: 参考日期（用于时间表达式解析）

        Returns:
            QueryPlan 结构化检索计划
        """
        # 如果有明确的 qtype，直接使用
        if qtype and qtype != "unknown":
            return self._plan_for_type(qtype, question, reference_date)

        # 否则自动分类
        detected_type = self._detect_type(question)
        return self._plan_for_type(detected_type, question, reference_date)

    def _detect_type(self, question: str) -> str:
        """自动检测问题类型"""
        scores = {
            "temporal-reasoning": 0,
            "knowledge-update": 0,
            "multi-session": 0,
            "single-session-preference": 0,
        }

        for p in self._tr_patterns:
            if p.search(question):
                scores["temporal-reasoning"] += 1
        for p in self._ku_patterns:
            if p.search(question):
                scores["knowledge-update"] += 1
        for p in self._ms_patterns:
            if p.search(question):
                scores["multi-session"] += 1
        for p in self._ssp_patterns:
            if p.search(question):
                scores["single-session-preference"] += 1

        max_type = max(scores, key=scores.get)
        max_score = scores[max_type]

        if max_score == 0:
            return "single-session-user"  # 默认

        return max_type

    def _plan_for_type(
        self,
        qtype: str,
        question: str,
        reference_date: str | None,
    ) -> QueryPlan:
        """根据问题类型生成检索计划"""
        if qtype == "temporal-reasoning":
            return self._plan_tr(question, reference_date)
        elif qtype == "knowledge-update":
            return self._plan_ku(question, reference_date)
        elif qtype == "multi-session":
            return self._plan_ms(question, reference_date)
        elif qtype == "single-session-preference":
            return self._plan_ssp(question, reference_date)
        elif qtype in ("single-session-user", "single-session-assistant"):
            return self._plan_ss(question, reference_date, qtype)
        else:
            return QueryPlan()

    def _plan_tr(self, question: str, ref_date: str | None) -> QueryPlan:
        """TR: 时间过滤 + 事件索引 + 正序排列"""
        time_filter = self._parser.parse(question, reference_date=ref_date)
        entities = self._extract_entities(question)
        return QueryPlan(
            primary_channels=["event_time", "vector", "keyword"],
            time_filter=time_filter,
            target_entities=entities,
            rerank_strategy="recency",
            context_format="temporal_chain",
            top_k_multiplier=1.5,
            fact_priority=0.7,
        )

    def _plan_ku(self, question: str, ref_date: str | None) -> QueryPlan:
        """KU: 实体匹配 + 时间倒序 + 最新值优先"""
        time_filter = self._parser.parse(question, reference_date=ref_date)
        entities = self._extract_entities(question)
        return QueryPlan(
            primary_channels=["vector", "keyword", "event_time"],
            time_filter=time_filter,
            target_entities=entities,
            rerank_strategy="entity",
            context_format="latest_first",
            top_k_multiplier=1.2,
            fact_priority=0.6,
        )

    def _plan_ms(self, question: str, ref_date: str | None) -> QueryPlan:
        """MS: 多子查询分解 + 聚合"""
        time_filter = self._parser.parse(question, reference_date=ref_date)
        entities = self._extract_entities(question)
        sub_queries = self._decompose(question, entities)
        return QueryPlan(
            primary_channels=["vector", "keyword", "event_time"],
            time_filter=time_filter,
            target_entities=entities,
            rerank_strategy="session_ndcg",
            context_format="aggregation",
            needs_decomposition=True,
            sub_queries=sub_queries,
            top_k_multiplier=2.0,  # MS 需要更大的召回窗口
            fact_priority=0.5,
        )

    def _plan_ssp(self, question: str, ref_date: str | None) -> QueryPlan:
        """SSP: 全量session扫描 + LLM判断"""
        entities = self._extract_entities(question)
        return QueryPlan(
            primary_channels=["vector", "keyword"],
            target_entities=entities,
            rerank_strategy="default",
            context_format="default",
            top_k_multiplier=1.0,
            fact_priority=0.8,  # preference 信息在事实中更明确
        )

    def _plan_ss(self, question: str, ref_date: str | None, qtype: str) -> QueryPlan:
        """SSU/SSA: 标准检索"""
        entities = self._extract_entities(question)
        return QueryPlan(
            primary_channels=["vector", "keyword"],
            target_entities=entities,
            rerank_strategy="default",
            context_format="default",
            top_k_multiplier=1.0,
            fact_priority=0.6,
        )

    def _extract_entities(self, question: str) -> list[str]:
        """从问题中提取目标实体（简单启发式）"""
        entities = []
        # 大写开头的连续词 → 可能是实体
        for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', question):
            name = m.group(1)
            # 过滤常见非实体词
            non_entities = {
                "What", "When", "Where", "Who", "How", "Why", "Which",
                "The", "This", "That", "These", "Those", "My", "Your",
                "How Many", "How Much", "How Often",
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December",
            }
            if name not in non_entities and len(name) > 2:
                entities.append(name)
        return entities

    def _decompose(self, question: str, entities: list[str]) -> list[str]:
        """MS 问题分解 — 将聚合问题分解为多个子查询

        示例:
        "How many times did I exercise in May?"
        → ["exercise exercising workout gym", "May 2024"]
        """
        sub_queries = []

        # 策略1：按实体分解
        for entity in entities:
            sub_queries.append(entity)

        # 策略2：提取核心动词 + 时间约束
        # "How many times did I exercise in May?" → "exercise" + May
        verb_match = re.search(
            r'\b(did|do|does|have|has|had)\s+(?:I|you|they|we|he|she)\s+(\w+)',
            question, re.IGNORECASE,
        )
        if verb_match:
            verb = verb_match.group(2)
            sub_queries.append(verb)

        # 时间约束子查询
        if self._parser.has_temporal_expression(question):
            sub_queries.append(question)  # 原始问题本身就是时间过滤子查询

        return sub_queries if sub_queries else [question]


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def create_query_classifier(**kwargs: Any) -> QueryClassifier:
    """工厂函数：创建 QueryClassifier 实例"""
    return QueryClassifier(**kwargs)
