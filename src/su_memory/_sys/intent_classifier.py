"""
意图分类器 + 渐进披露控制器

为 su-memory 增加：
1. IntentClassifier — 基于关键词的意图识别 + level 分层
2. ProgressiveDisclosure — 查询复杂度驱动的渐进披露

与 Hindsight intent-map.json 兼容，支持扩展自定义意图
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
import re
import time


# ========================
# 配置数据结构
# ========================

@dataclass
class IntentConfig:
    """意图配置（兼容 intent-map.json 格式）"""
    name: str
    keywords: List[str]
    level: int  # 0=闲聊(L0), 1=简单(L1), 2=标准(L2), 3=深度(L3)
    wikis: List[str] = field(default_factory=list)  # obsidian, memex
    dreams_boost: float = 0.0
    tags: List[str] = field(default_factory=list)
    description: str = ""
    priority: int = 0  # 类别优先级，越大越优先（用于同分时 tie-break）

    def match_score(self, query: str) -> tuple[float, int]:
        """
        计算查询与本意图的匹配得分
        Returns: (primary_score, max_keyword_len) 元组
        
        领域意图（level >= 1）命中领域词时加分；
        casual（level=0）即使命中也降权，因为领域词优先。
        """
        q = query.lower()
        score = 0.0
        max_kw_len = 0
        for kw in self.keywords:
            if kw.lower() in q:
                kw_len = len(kw)
                # Level-0 (casual) 的通用词降权，领域词优先
                if self.level == 0:
                    score += kw_len * 0.3  # casual 关键词权重降低
                else:
                    score += kw_len
                max_kw_len = max(max_kw_len, kw_len)
        return score, max_kw_len


# ========================
# 内置意图集
# ========================

DEFAULT_INTENTS: Dict[str, IntentConfig] = {
    "project-status": IntentConfig(
        name="project-status",
        keywords=["项目状态", "进度", "任务", "当前在做", "进行中", "完成情况"],
        level=2,
        wikis=["obsidian"],
        dreams_boost=0.3,
        priority=5,
        tags=["nutri-brain", "task", "progress"],
        description="查询项目当前进度和任务状态"
    ),
    "financing": IntentConfig(
        name="financing",
        keywords=["融资", "路演", "投资", "BP", "商业计划", "投资人", "资金", "估值"],
        level=2,
        wikis=["obsidian", "memex"],
        dreams_boost=0.2,
        priority=10,
        tags=["nutri-brain", "financing", "investor"],
        description="融资相关查询"
    ),
    "technical": IntentConfig(
        name="technical",
        keywords=["技术", "架构", "代码", "开发", "AI模型", "训练", "部署", "算法"],
        level=2,
        wikis=["obsidian", "memex"],
        dreams_boost=0.2,
        tags=["nutri-brain", "technical", "ai", "architecture"],
        description="技术相关查询"
    ),
    "cooperation": IntentConfig(
        name="cooperation",
        keywords=["合作", "协议", "签约", "医院", "合作方", "伙伴", "佛山", "广西", "坪山"],
        level=2,
        wikis=["obsidian", "memex"],
        dreams_boost=0.3,
        tags=["nutri-brain", "cooperation", "hospital"],
        description="商务合作相关查询"
    ),
    "team": IntentConfig(
        name="team",
        keywords=["团队", "招聘", "人员", "成员", "代理", "角色", "分工"],
        level=1,
        wikis=["obsidian"],
        dreams_boost=0.1,
        tags=["nutri-brain", "team", "agent"],
        description="团队相关查询"
    ),
    "clinical-knowledge": IntentConfig(
        name="clinical-knowledge",
        keywords=["临床", "营养", "指南", "指标", "共识", "治疗", "评估", "量表"],
        level=3,
        wikis=["memex"],
        dreams_boost=0.2,
        tags=["nutri-brain", "clinical", "medical"],
        description="临床营养专业知识查询"
    ),
    "casual": IntentConfig(
        name="casual",
        keywords=["你好", "天气", "闲聊", "怎么样", "在吗", "嗨"],
        level=0,
        wikis=[],
        dreams_boost=0.0,
        tags=[],
        description="闲聊/打招呼，不触发深度召回"
    ),
    "multi-hop": IntentConfig(
        name="multi-hop",
        keywords=["之前", "后来", "之前提到", "那件事", "当时", "为什么", "怎么想到"],
        level=2,
        wikis=["obsidian"],
        dreams_boost=0.4,
        tags=["reasoning", "temporal"],
        description="多跳推理查询"
    ),
}


# ========================
# IntentClassifier
# ========================

class IntentClassifier:
    """
    意图分类器

    使用方法：
        classifier = IntentClassifier()
        intent = classifier.classify("Nutri-Brain项目当前进度如何")
        # intent.name = "project-status", intent.level = 2
    """

    def __init__(self, custom_intents: Optional[Dict[str, IntentConfig]] = None):
        self._intents = {**DEFAULT_INTENTS, **(custom_intents or {})}
        self._patterns: Dict[str, re.Pattern] = {}
        self._compile_patterns()
        self._last_classified: Optional[IntentConfig] = None

    def _compile_patterns(self) -> None:
        """预编译所有关键词正则"""
        for name, config in self._intents.items():
            pattern_str = "|".join(re.escape(kw) for kw in config.keywords)
            if pattern_str:
                self._patterns[name] = re.compile(pattern_str, re.IGNORECASE)

    def classify(self, query: str) -> IntentConfig:
        """
        分类查询并返回最佳匹配的 IntentConfig

        匹配策略：
        1. 精确关键词匹配（计算得分）
        2. 正则全文搜索
        3. Fallback 到 casual
        """
        if not query or not query.strip():
            return self._intents["casual"]

        q = query.lower()
        best_intent: Optional[IntentConfig] = None
        best_score = 0.0
        best_kw_len = 0
        best_priority = 0

        # 策略1：关键词精确匹配，排序键为 (score, kw_len, priority)
        for name, config in self._intents.items():
            score, kw_len = config.match_score(query)
            priority = config.priority
            if (score > best_score or
                (score == best_score and kw_len > best_kw_len) or
                (score == best_score and kw_len == best_kw_len and priority > best_priority)):
                best_score = score
                best_kw_len = kw_len
                best_priority = priority
                best_intent = config

        # 策略2：正则匹配（捕获更多变体）
        for name, pattern in self._patterns.items():
            if pattern.search(query):
                config = self._intents[name]
                matched = sum(1 for kw in config.keywords if kw.lower() in q)
                if matched >= 2:
                    score = matched * 0.5
                    kw_len = max(len(kw) for kw in config.keywords if kw.lower() in q)
                    priority = config.priority
                    if (score > best_score or
                        (score == best_score and kw_len > best_kw_len) or
                        (score == best_score and kw_len == best_kw_len and priority > best_priority)):
                        best_score = score
                        best_kw_len = kw_len
                        best_priority = priority
                        best_intent = config

        if best_intent is None:
            best_intent = self._intents["casual"]

        self._last_classified = best_intent
        return best_intent

    def classify_with_level(self, query: str, min_level: int = 0) -> IntentConfig:
        """
        按最低层级约束分类

        用于多轮对话中强制使用某个 level 以上的结果
        """
        intent = self.classify(query)
        if intent.level < min_level:
            # 返回一个满足最低 level 的 generic intent
            return self._get_generic_intent_for_level(min_level)
        return intent

    def _get_generic_intent_for_level(self, min_level: int) -> IntentConfig:
        """返回符合最低 level 要求的通用 intent"""
        if min_level >= 2:
            return self._intents["project-status"]
        elif min_level >= 1:
            return self._intents["team"]
        return self._intents["casual"]

    @property
    def last_intent(self) -> Optional[IntentConfig]:
        """返回上次分类结果"""
        return self._last_classified

    def should_recall(self, query: str) -> bool:
        """
        判断某查询是否应该触发记忆召回

        L0 (level=0) 不触发召回
        """
        intent = self.classify(query)
        return intent.level >= 1

    def get_wiki_sources(self, query: str) -> List[str]:
        """获取某查询应查询的 Wiki 源列表"""
        intent = self.classify(query)
        return intent.wikis

    def get_recall_mode(self, query: str) -> str:
        """
        获取召回模式

        Returns:
            "none": 不触发召回 (L0)
            "simple": 简单召回，只查 Dreams (L1)
            "standard": 标准召回，Dreams + Wiki (L2)
            "deep": 深度召回，Dreams + Wiki + 外部搜索 (L3)
        """
        intent = self.classify(query)
        if intent.level == 0:
            return "none"
        elif intent.level == 1:
            return "simple"
        elif intent.level == 2:
            return "standard"
        else:
            return "deep"


# ========================
# ProgressiveDisclosure
# ========================

@dataclass
class DisclosureStage:
    """单个披露阶段"""
    name: str
    max_items: int
    wait_feedback: bool = True
    summary_mode: bool = False


DISCLOSURE_STAGES: List[DisclosureStage] = [
    DisclosureStage(name="summary", max_items=3, wait_feedback=True, summary_mode=True),
    DisclosureStage(name="details", max_items=5, wait_feedback=True, summary_mode=False),
    DisclosureStage(name="context", max_items=8, wait_feedback=True, summary_mode=False),
    DisclosureStage(name="evidence", max_items=12, wait_feedback=False, summary_mode=False),
    DisclosureStage(name="full", max_items=20, wait_feedback=False, summary_mode=False),
]


class ProgressiveDisclosure:
    """
    渐进披露控制器

    根据用户反馈动态调整披露深度：
    - 用户正反馈（positive）：快速推进到下一阶段
    - 用户负反馈（negative）：回退到上一阶段
    - 无反馈：保持当前阶段

    使用方法：
        disclosure = ProgressiveDisclosure()
        stage = disclosure.get_current_stage()  # 获取当前阶段配置

        results = retrieve(query, top_k=stage.max_items, ...)
        disclosure.record_results_count(len(results))

        # 用户反馈后
        disclosure.on_feedback("positive")  # 或 "negative" 或 None
        next_stage = disclosure.get_next_stage()
    """

    def __init__(
        self,
        stages: Optional[List[DisclosureStage]] = None,
        start_stage: int = 0,
    ):
        self._stages = stages or DISCLOSURE_STAGES
        self._stage_index = start_stage
        self._feedback_history: List[str] = []  # "positive", "negative", None
        self._result_counts: List[int] = []
        self._last_query: Optional[str] = None
        self._last_intent_name: Optional[str] = None

    @property
    def current_stage(self) -> DisclosureStage:
        return self._stages[max(0, min(self._stage_index, len(self._stages) - 1))]

    @property
    def stage_index(self) -> int:
        return self._stage_index

    def get_current_stage(self) -> DisclosureStage:
        """获取当前阶段配置"""
        return self.current_stage

    def get_next_stage(
        self,
        feedback: Optional[str] = None,
        result_count: Optional[int] = None,
    ) -> DisclosureStage:
        """
        根据反馈获取下一阶段

        Args:
            feedback: "positive" | "negative" | None
            result_count: 上次结果数量（用于判断是否需要退阶）
        """
        if feedback is not None:
            self._feedback_history.append(feedback)

        # 分析最近的反馈模式
        recent_feedback = self._feedback_history[-3:] if self._feedback_history else []

        if feedback == "positive":
            # 正反馈：加速推进
            consecutive_pos = sum(1 for f in recent_feedback if f == "positive")
            if consecutive_pos >= 2:
                self._stage_index = min(self._stage_index + 2, len(self._stages) - 1)
            else:
                self._stage_index = min(self._stage_index + 1, len(self._stages) - 1)

        elif feedback == "negative":
            # 负反馈：回退
            self._stage_index = max(self._stage_index - 1, 0)
            # 清空历史避免干扰
            self._feedback_history.clear()

        elif result_count is not None:
            self._result_counts.append(result_count)
            # 如果结果为空且不是 L0，尝试退阶
            if result_count == 0 and self._stage_index > 0:
                self._stage_index = max(self._stage_index - 1, 0)

        return self.current_stage

    def record_query(self, query: str, intent_name: str) -> None:
        """记录当前查询和意图（用于调试/分析）"""
        self._last_query = query
        self._last_intent_name = intent_name

    def record_results_count(self, count: int) -> None:
        """记录本次返回结果数量"""
        self._result_counts.append(count)

    def reset(self) -> None:
        """重置到初始状态"""
        self._stage_index = 0
        self._feedback_history.clear()
        self._result_counts.clear()
        self._last_query = None
        self._last_intent_name = None

    def force_stage(self, stage_index: int) -> DisclosureStage:
        """强制跳转到指定阶段"""
        self._stage_index = max(0, min(stage_index, len(self._stages) - 1))
        return self.current_stage

    def get_summary_needed(self, query: str) -> bool:
        """判断某查询是否需要返回摘要模式"""
        if self.current_stage.summary_mode:
            return True
        # 特定类型查询自动摘要
        summary_triggers = ["总结", "概述", "核心", "重点", "概要"]
        return any(trigger in query for trigger in summary_triggers)

    def should_wait_feedback(self) -> bool:
        """当前阶段是否需要等待反馈"""
        return self.current_stage.wait_feedback

    @property
    def stats(self) -> Dict:
        """当前状态统计"""
        return {
            "stage": self.current_stage.name,
            "stage_index": self._stage_index,
            "feedback_history_len": len(self._feedback_history),
            "last_query": self._last_query,
            "last_intent": self._last_intent_name,
            "avg_result_count": (
                sum(self._result_counts) / len(self._result_counts)
                if self._result_counts else 0
            ),
        }
