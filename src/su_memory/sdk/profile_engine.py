"""
su-memory SDK — User Profile Engine (v3.5.5 P1-2)

从记忆库自动抽取用户画像：偏好、领域关键词、交互模式、专业水平、
约束条件、历史目标。支持增量更新。

核心类：
- UserProfile: 用户画像数据模型
- UserProfileEngine: 画像提取引擎

使用示例:
    >>> from su_memory.sdk.profile_engine import UserProfileEngine
    >>> engine = UserProfileEngine(client)
    >>> profile = engine.extract_from_memories()
    >>> print(profile.domain_keywords[:5])
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# ── 可选依赖 ──────────────────────────────────────────────────────────────────

_INTENT_CLASSIFIER_AVAILABLE = False
try:
    from su_memory._sys.intent_classifier import IntentClassifier  # noqa: F401
    _INTENT_CLASSIFIER_AVAILABLE = True
except ImportError:
    logger.debug("IntentClassifier 不可用，将使用简单关键词分析")

_BAYESIAN_AVAILABLE = False
try:
    from su_memory._sys.bayesian import BayesianBelief, BayesianEngine  # noqa: F401, E501
    _BAYESIAN_AVAILABLE = True
except ImportError:
    logger.debug("BayesianEngine 不可用，跳过信念追踪分析")


# ── 数据模型 ──────────────────────────────────────────────────────────────────


@dataclass
class InteractionPattern:
    """交互模式统计"""
    total_queries: int = 0
    active_hours: list[int] = field(default_factory=list)     # 活跃时段 (0-23)
    avg_query_length: float = 0.0                              # 平均查询长度
    query_topics: dict[str, int] = field(default_factory=dict) # 主题频率
    feedback_ratio: float = 0.0                                # 正反馈比例
    session_count: int = 0                                     # 会话数


@dataclass
class UserProfile:
    """用户画像数据模型

    从记忆库中自动抽取的完整用户画像。
    """
    # 基础信息
    extracted_at: str = ""                              # 提取时间 ISO 格式
    total_memories: int = 0                             # 总记忆数

    # 偏好
    preferences: list[str] = field(default_factory=list)          # 显式偏好列表
    implicit_preferences: list[str] = field(default_factory=list) # 隐式偏好（从行为推断）
    preferred_sources: list[str] = field(default_factory=list)    # 偏好信息来源

    # 领域知识
    domain_keywords: list[str] = field(default_factory=list)      # 领域关键词 (top N)
    domain_distribution: dict[str, float] = field(default_factory=dict)  # 领域分布

    # 交互模式
    interaction_pattern: InteractionPattern = field(default_factory=InteractionPattern)

    # 专业水平
    expertise_level: str = "unknown"                    # "novice" | "intermediate" | "advanced" | "expert"
    expertise_domains: dict[str, str] = field(default_factory=dict)  # 领域→水平

    # 约束与目标
    constraints: list[str] = field(default_factory=list)          # 显式约束条件
    historical_goals: list[str] = field(default_factory=list)     # 历史目标
    active_goals: list[str] = field(default_factory=list)         # 当前活跃目标

    # 学习状态
    learning_velocity: float = 0.0                      # 学习速率 (新知/总量)
    knowledge_gaps: list[str] = field(default_factory=list)       # 知识盲区

    # 原始统计
    category_distribution: dict[str, int] = field(default_factory=dict)
    energy_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转为可序列化字典"""
        return {
            "extracted_at": self.extracted_at,
            "total_memories": self.total_memories,
            "preferences": self.preferences,
            "implicit_preferences": self.implicit_preferences,
            "preferred_sources": self.preferred_sources,
            "domain_keywords": self.domain_keywords,
            "domain_distribution": self.domain_distribution,
            "interaction_pattern": {
                "total_queries": self.interaction_pattern.total_queries,
                "active_hours": self.interaction_pattern.active_hours,
                "avg_query_length": self.interaction_pattern.avg_query_length,
                "query_topics": self.interaction_pattern.query_topics,
                "feedback_ratio": self.interaction_pattern.feedback_ratio,
                "session_count": self.interaction_pattern.session_count,
            },
            "expertise_level": self.expertise_level,
            "expertise_domains": self.expertise_domains,
            "constraints": self.constraints,
            "historical_goals": self.historical_goals,
            "active_goals": self.active_goals,
            "learning_velocity": self.learning_velocity,
            "knowledge_gaps": self.knowledge_gaps,
            "category_distribution": self.category_distribution,
            "energy_distribution": self.energy_distribution,
        }

    def summary(self) -> str:
        """生成可读摘要"""
        parts = [
            f"总记忆数: {self.total_memories}",
            f"专业水平: {self.expertise_level}",
            f"核心领域: {', '.join(self.domain_keywords[:5]) or '未知'}",
            f"偏好: {', '.join(self.preferences[:5]) or '未检测到'}",
            f"活跃目标: {', '.join(self.active_goals[:3]) or '未检测到'}",
            f"学习速率: {self.learning_velocity:.2%}",
        ]
        return "\n".join(parts)


# ── 画像引擎 ──────────────────────────────────────────────────────────────────


class UserProfileEngine:
    """用户画像引擎 (v3.5.5 P1-2)

    从记忆库中自动抽取用户画像，支持增量更新。

    Args:
        client: SuMemory / SuMemoryLite / SuMemoryLitePro 实例
        domain_keywords_config: 自定义领域关键词映射 {领域名: [关键词列表]}
        expertise_thresholds: 专业水平判定阈值 {"advanced": 100, "expert": 500}

    Example:
        >>> engine = UserProfileEngine(client)
        >>> profile = engine.extract_from_memories()
        >>> engine.update_incremental(new_memories)
        >>> summary = engine.get_profile().summary()
    """

    # 默认领域关键词配置
    DEFAULT_DOMAIN_KEYWORDS: dict[str, list[str]] = {
        "programming": [
            "python", "java", "go", "rust", "typescript", "javascript",
            "react", "vue", "django", "fastapi", "flask",
            "api", "database", "sql", "nosql", "redis",
            "docker", "kubernetes", "ci/cd", "git", "devops",
            "machine learning", "ai", "deep learning", "nlp",
        ],
        "product_design": [
            "用户研究", "交互设计", "原型", "figma", "sketch",
            "产品经理", "prd", "需求分析", "用户画像", "ab测试",
        ],
        "business": [
            "融资", "路演", "bp", "商业计划", "投资人",
            "市场分析", "竞品", "商业模式", "营收", "增长",
        ],
        "healthcare": [
            "临床", "营养", "医疗", "患者", "诊断",
            "治疗", "康复", "护理", "药品", "手术",
        ],
        "education": [
            "学习", "课程", "教程", "论文", "研究",
            "阅读", "知识", "笔记", "考试", "认证",
        ],
        "food_dietary": [
            "辛辣", "川菜", "湘菜", "粤菜", "咖啡", "茶",
            "花生", "过敏", "坚果", "牛奶", "鸡蛋", "海鲜",
            "糖尿病", "低盐", "低糖", "低脂", "碳水化合物", "蛋白质",
            "蔬菜", "水果", "肉类", "鱼类", "豆类", "谷物",
            "早餐", "午餐", "晚餐", "零食", "饮料", "烹饪",
            "饮食", "食物", "膳食", "营养", "热量", "卡路里",
            "辣", "甜", "咸", "酸", "苦", "油腻",
        ],
        "daily_life": [
            "北京", "上海", "深圳", "旅游", "美食",
            "运动", "健身", "电影", "音乐", "游戏",
        ],
    }

    def __init__(
        self,
        client,  # SuMemory / SuMemoryLite / SuMemoryLitePro
        domain_keywords_config: dict[str, list[str]] | None = None,
        expertise_thresholds: dict[str, int] | None = None,
    ):
        self._client = client
        self._domain_keywords = domain_keywords_config or self.DEFAULT_DOMAIN_KEYWORDS
        self._expertise_thresholds = expertise_thresholds or {
            "advanced": 100,
            "expert": 500,
        }
        self._profile: UserProfile | None = None
        self._profile_version: int = 0
        self._last_memory_count: int = 0

    # ── 公开 API ──────────────────────────────────────────────────────────

    def extract_from_memories(
        self,
        top_keywords: int = 30,
        include_intent_analysis: bool = True,
    ) -> UserProfile:
        """从记忆库全量提取用户画像

        扫描所有记忆内容，提取偏好、关键词、领域分布、专业水平等。

        Args:
            top_keywords: 提取的关键词数量（默认 30）
            include_intent_analysis: 是否启用意图分类分析

        Returns:
            UserProfile 对象
        """
        memories = self._get_memories()
        if not memories:
            logger.warning("记忆库为空，返回空画像")
            return UserProfile(extracted_at=datetime.now().isoformat())

        contents = [m.get("content", "") for m in memories]

        # 1. 关键词提取
        all_keywords = self._extract_keywords(contents, top_keywords)

        # 2. 领域分布
        domain_dist = self._classify_domains(contents)

        # 3. 偏好提取
        preferences = self._extract_preferences(contents)
        implicit_prefs = self._infer_implicit_preferences(contents, domain_dist)

        # 4. 专业水平评估
        expertise_level, expertise_domains = self._assess_expertise(
            len(memories), domain_dist, contents
        )

        # 5. 约束与目标提取
        constraints = self._extract_constraints(contents)
        goals = self._extract_goals(contents)

        # 6. 交互模式
        interaction = self._build_interaction_pattern()

        # 7. 学习状态
        learning_velocity = self._compute_learning_velocity(memories)
        knowledge_gaps = self._identify_knowledge_gaps(domain_dist, contents)

        # 8. 统计信息
        stats = self._get_stats()
        category_dist = stats.get("category_distribution", {})
        energy_dist = stats.get("energy_distribution", {})

        profile = UserProfile(
            extracted_at=datetime.now().isoformat(),
            total_memories=len(memories),
            preferences=self._deduplicate_list(preferences),
            implicit_preferences=self._deduplicate_list(implicit_prefs),
            preferred_sources=self._extract_sources(memories),
            domain_keywords=all_keywords,
            domain_distribution=domain_dist,
            interaction_pattern=interaction,
            expertise_level=expertise_level,
            expertise_domains=expertise_domains,
            constraints=self._deduplicate_list(constraints),
            historical_goals=self._deduplicate_list(goals),
            active_goals=self._deduplicate_list(goals[-5:] if goals else []),
            learning_velocity=learning_velocity,
            knowledge_gaps=knowledge_gaps,
            category_distribution=category_dist,
            energy_distribution=energy_dist,
        )

        self._profile = profile
        self._profile_version += 1
        self._last_memory_count = len(memories)

        logger.info(
            f"画像提取完成: {profile.total_memories} 条记忆, "
            f"{len(profile.domain_keywords)} 个关键词, "
            f"水平={profile.expertise_level}"
        )
        return profile

    def update_incremental(
        self,
        new_memories: list[dict[str, Any]] | None = None,
    ) -> UserProfile:
        """增量更新画像（适合高频调用）

        仅处理新增的记忆，合并到现有画像中。

        Args:
            new_memories: 新增的记忆列表（为 None 时自动从 client 获取）

        Returns:
            UserProfile 对象
        """
        current_memories = self._get_memories()

        # 快速路径：没有变化
        if len(current_memories) == self._last_memory_count:
            return self._profile or self.extract_from_memories()

        # 增量更新：只分析最近新增的记忆
        delta_count = len(current_memories) - self._last_memory_count
        if delta_count > 0 and delta_count < 10:
            recent = current_memories[-delta_count:]
            logger.debug(f"增量更新: +{delta_count} 条新记忆")
            # 简单合并关键词和领域分布
            if self._profile:
                recent_contents = [m.get("content", "") for m in recent]
                new_keywords = self._extract_keywords(recent_contents, 10)
                # 合并关键词
                existing = set(self._profile.domain_keywords)
                for kw in new_keywords:
                    if kw not in existing:
                        self._profile.domain_keywords.append(kw)
                self._profile.total_memories = len(current_memories)
                self._last_memory_count = len(current_memories)
                self._profile_version += 1
                return self._profile

        # 变化较大，全量重算
        return self.extract_from_memories()

    def get_profile(self) -> UserProfile | None:
        """获取当前缓存的画像"""
        if self._profile is None:
            self._profile = self.extract_from_memories()
        return self._profile

    @property
    def version(self) -> int:
        """画像版本号（每次 extract/update 递增）"""
        return self._profile_version

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _get_memories(self) -> list[dict[str, Any]]:
        """从客户端获取全部记忆 (v3.5.5-p0: 三级回退)"""
        # 第一层: get_all_memories() 方法 (MemoryProtocol 接口)
        if hasattr(self._client, "get_all_memories"):
            result = self._client.get_all_memories()
            if result:
                return result

        # 第二层: get_stats().recent_memories (部分客户端实现)
        if hasattr(self._client, "get_stats"):
            stats = self._client.get_stats()
            recent = stats.get("recent_memories", [])
            if recent:
                return recent

        # 第三层: 直接访问内部 _memories (兜底回退)
        # P0-3修复: SuMemoryClient 通过 self._client 代理, 需穿透检查
        actual_client = getattr(self._client, '_client', self._client)
        if hasattr(actual_client, "_memories"):
            raw_memories = actual_client._memories
            if raw_memories:
                result = []
                for m in raw_memories:
                    if isinstance(m, dict):
                        result.append(m)
                    elif hasattr(m, 'id'):
                        # MemoryNode 对象 → dict
                        result.append({
                            "id": getattr(m, 'id', ''),
                            "content": getattr(m, 'content', ''),
                            "metadata": getattr(m, 'metadata', {}),
                            "keywords": getattr(m, 'keywords', []),
                            "timestamp": getattr(m, 'timestamp', 0),
                            "energy_type": getattr(m, 'energy_type', None),
                        })
                if result:
                    return result

        return []

    def _get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        if hasattr(self._client, "get_stats"):
            return self._client.get_stats()
        return {}

    def _extract_keywords(
        self,
        contents: list[str],
        top_n: int = 30,
    ) -> list[str]:
        """从内容中提取顶级关键词"""
        word_freq: Counter = Counter()
        for text in contents:
            words = self._tokenize(text)
            word_freq.update(words)

        # 过滤太短或太长的词
        filtered = {
            w: c for w, c in word_freq.items()
            if 2 <= len(w) <= 15 and not w.isdigit()
        }
        return [w for w, _ in Counter(filtered).most_common(top_n)]

    def _tokenize(self, text: str) -> list[str]:
        """简单分词（中文 + 英文混合）"""
        # 中文字符单独分割，英文按空白/标点分割
        tokens: list[str] = []
        # 替换标点为空格
        cleaned = re.sub(r'[,，。.!！?？;；:：、""'r'「」\[\]{}()（）\s]+', ' ', text)
        for word in cleaned.split():
            word = word.strip().lower()
            if len(word) >= 2:
                # 混合中英文：拆分连续中文
                if re.search(r'[\u4e00-\u9fff]', word):
                    # 对含中文的词进一步拆分
                    sub_tokens = re.findall(r'[\u4e00-\u9fff]{1,4}|[a-zA-Z]+', word)
                    tokens.extend(t for t in sub_tokens if len(t) >= 2)
                else:
                    tokens.append(word)
        return tokens

    def _classify_domains(self, contents: list[str]) -> dict[str, float]:
        """根据关键词匹配进行领域分类"""
        all_text = " ".join(contents).lower()
        domain_scores: dict[str, float] = {}

        total_hits = 0
        for domain, keywords in self._domain_keywords.items():
            hits = sum(1 for kw in keywords if kw.lower() in all_text)
            if hits > 0:
                domain_scores[domain] = hits
                total_hits += hits

        # 归一化为比例
        if total_hits > 0:
            domain_scores = {
                d: round(s / total_hits, 4) for d, s in domain_scores.items()
            }

        return domain_scores

    def _extract_preferences(self, contents: list[str]) -> list[str]:
        """提取显式偏好（"我喜欢"、"我偏好" 等模式）"""
        patterns = [
            r'我(?:喜欢|偏好|倾向于|常用|习惯用?)\s*[：:]*\s*([^。.!！?？\n]{2,40})',
            r'(?:我的|本人)(?:首选|偏好|习惯)\s*[：:是]*\s*([^。.!！?？\n]{2,40})',
        ]
        results: list[str] = []
        for text in contents:
            for pat in patterns:
                matches = re.findall(pat, text)
                results.extend(m.strip() for m in matches if m.strip())
        return self._deduplicate_list(results)

    def _infer_implicit_preferences(
        self,
        contents: list[str],
        domain_dist: dict[str, float],
    ) -> list[str]:
        """从行为推断隐式偏好"""
        implicit: list[str] = []

        # 高频领域 → 隐式偏好
        top_domains = sorted(domain_dist, key=domain_dist.get, reverse=True)[:3]
        for d in top_domains:
            implicit.append(f"偏好领域: {d}")

        # 从关键词推断技术栈偏好
        all_text = " ".join(contents).lower()
        tech_indicators = [
            ("python", "Python 生态"),
            ("golang", "Go 语言"),
            ("rust", "Rust 语言"),
            ("react", "React 前端"),
            ("vue", "Vue 前端"),
            ("fastapi", "FastAPI 框架"),
            ("docker", "容器化部署"),
            ("kubernetes", "K8s 编排"),
        ]
        for kw, label in tech_indicators:
            if kw in all_text:
                implicit.append(label)

        return implicit

    def _assess_expertise(
        self,
        total: int,
        domain_dist: dict[str, float],
        contents: list[str],
    ) -> tuple[str, dict[str, str]]:
        """评估专业水平"""
        # 整体水平
        if total >= self._expertise_thresholds.get("expert", 500):
            overall = "expert"
        elif total >= self._expertise_thresholds.get("advanced", 100):
            overall = "advanced"
        elif total >= 20:
            overall = "intermediate"
        else:
            overall = "novice"

        # 分领域水平
        domain_levels: dict[str, str] = {}
        all_text = " ".join(contents).lower()
        for domain, keywords in self._domain_keywords.items():
            hits = sum(1 for kw in keywords if kw.lower() in all_text)
            if hits >= 50:
                domain_levels[domain] = "expert"
            elif hits >= 20:
                domain_levels[domain] = "advanced"
            elif hits >= 5:
                domain_levels[domain] = "intermediate"
            elif hits > 0:
                domain_levels[domain] = "novice"

        return overall, domain_levels

    def _extract_constraints(self, contents: list[str]) -> list[str]:
        """提取约束条件（"我不能"、"限制"、"必须" 等模式）"""
        patterns = [
            r'(?:我)?(?:不能|不可以|禁止|无法|受限[于制]?)\s*[：:]*\s*([^。.!！?？\n]{3,50})',
            r'(?:需要|必须|应该|务必)\s*(?:遵守|满足|遵循|符合)\s*([^。.!！?？\n]{3,50})',
        ]
        results: list[str] = []
        for text in contents:
            for pat in patterns:
                matches = re.findall(pat, text)
                results.extend(m.strip() for m in matches if m.strip())
        return self._deduplicate_list(results)

    def _extract_goals(self, contents: list[str]) -> list[str]:
        """提取目标（"我想"、"目标"、"计划" 等模式）"""
        patterns = [
            r'(?:我)?(?:想|要|希望|打算|目标[是:：]|计划)\s*([^。.!！?？\n]{3,60})',
            r'目标[：:是]\s*([^。.!！?？\n]{3,60})',
        ]
        results: list[str] = []
        for text in contents:
            for pat in patterns:
                matches = re.findall(pat, text)
                results.extend(m.strip() for m in matches if m.strip())
        return self._deduplicate_list(results)

    def _extract_sources(self, memories: list[dict[str, Any]]) -> list[str]:
        """提取偏好信息来源"""
        sources: Counter = Counter()
        for m in memories:
            meta = m.get("metadata") or {}
            source = meta.get("source") or meta.get("source_file") or meta.get("ingest_source")
            if source:
                sources[str(source)] += 1
        return [s for s, _ in sources.most_common(10)]

    def _build_interaction_pattern(self) -> InteractionPattern:
        """构建交互模式统计"""
        pattern = InteractionPattern()

        stats = self._get_stats()
        if "total_queries" in stats:
            pattern.total_queries = stats["total_queries"]

        return pattern

    def _compute_learning_velocity(
        self,
        memories: list[dict[str, Any]],
    ) -> float:
        """计算学习速率（新知识 / 总量知识的比率）"""
        if len(memories) < 10:
            return 0.0

        # 简单估计：最近 20% 记忆中有多少新关键词
        split = max(len(memories) // 5, 1)
        recent = [m.get("content", "") for m in memories[-split:]]
        older = [m.get("content", "") for m in memories[:-split]]

        recent_kw = set(self._extract_keywords(recent, 50))
        older_kw = set(self._extract_keywords(older, 50))

        if not recent_kw:
            return 0.0

        new_ratio = len(recent_kw - older_kw) / len(recent_kw)
        return round(min(new_ratio, 1.0), 4)

    def _identify_knowledge_gaps(
        self,
        domain_dist: dict[str, float],
        contents: list[str],
    ) -> list[str]:
        """识别知识盲区"""
        gaps: list[str] = []

        # 覆盖率不足的领域
        all_text = " ".join(contents).lower()
        for domain, keywords in self._domain_keywords.items():
            hits = sum(1 for kw in keywords if kw.lower() in all_text)
            if hits == 0 and len(keywords) > 3:
                gaps.append(f"未覆盖领域: {domain}")

        # 领域内稀疏关键词
        for domain, _score in domain_dist.items():
            keywords = self._domain_keywords.get(domain, [])
            if keywords and len(keywords) > 5:
                hit_ratio = sum(1 for kw in keywords if kw.lower() in all_text) / len(keywords)
                if hit_ratio < 0.3:
                    gaps.append(f"领域覆盖不足: {domain} ({hit_ratio:.0%})")

        return gaps[:10]

    @staticmethod
    def _deduplicate_list(items: list[str]) -> list[str]:
        """去重并保持顺序"""
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            key = item.lower().strip()
            if key and key not in seen:
                seen.add(key)
                result.append(item.strip())
        return result
