"""
P0: ProfileEngine 单元测试 (v3.5.5 P1-2)
===========================================
覆盖: UserProfileEngine, UserProfile, InteractionPattern
      关键词提取, 领域分类, 偏好提取, 约束提取, 目标提取,
      专业水平评估, 学习速率, 增量更新

运行: pytest tests/test_profile_engine.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from su_memory.sdk.lite_pro import SuMemoryLitePro
from su_memory.sdk.profile_engine import (
    InteractionPattern,
    UserProfile,
    UserProfileEngine,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def empty_client():
    """空的记忆客户端"""
    return SuMemoryLitePro(max_memories=200)


@pytest.fixture
def dietary_client():
    """含饮食/医疗领域数据的客户端"""
    client = SuMemoryLitePro(max_memories=200)
    memories = [
        ("用户偏好辛辣食物，喜欢川菜和湘菜", {"domain": "dietary"}),
        ("血压偏高，医生建议低盐饮食控制钠摄入", {"domain": "medical"}),
        ("每周跑步3次，每次5公里，运动习惯良好", {"domain": "exercise"}),
        ("对花生过敏，需避免含花生成分的食物", {"domain": "allergy"}),
        ("工作压力大，经常加班到晚上10点", {"domain": "lifestyle"}),
        ("喜欢阅读科幻小说，最近在看三体和基地系列", {"domain": "hobby"}),
        ("有糖尿病家族史，需控制糖分和碳水化合物摄入", {"domain": "medical"}),
        ("每天喝2-3杯美式咖啡，喜欢深度烘焙", {"domain": "dietary"}),
    ]
    for content, meta in memories:
        client.add(content, metadata=meta)
    return client


@pytest.fixture
def tech_client():
    """含编程/技术领域数据的客户端"""
    client = SuMemoryLitePro(max_memories=300)
    memories = [
        ("我习惯使用Python进行后端开发，常用FastAPI框架", {}),
        ("项目使用Docker进行容器化部署，Kubernetes管理集群", {}),
        ("我需要学习Rust语言以提升系统编程能力", {}),
        ("机器学习和深度学习是AI的核心技术", {}),
        ("数据库使用PostgreSQL和Redis做缓存", {}),
    ]
    for content, meta in memories:
        client.add(content, metadata=meta)
    return client


# ============================================================
# UserProfile 数据模型
# ============================================================

class TestUserProfile:
    """UserProfile 数据模型"""

    def test_default_profile(self):
        """默认构造"""
        profile = UserProfile()
        assert profile.total_memories == 0
        assert profile.preferences == []
        assert profile.expertise_level == "unknown"

    def test_to_dict(self):
        """序列化"""
        profile = UserProfile(
            extracted_at="2026-01-01T00:00:00",
            total_memories=10,
            preferences=["偏好A", "偏好B"],
            domain_keywords=["python", "fastapi"],
            expertise_level="intermediate",
        )
        d = profile.to_dict()
        assert d["total_memories"] == 10
        assert "偏好A" in d["preferences"]
        assert d["expertise_level"] == "intermediate"

    def test_summary_contains_key_info(self):
        """摘要包含关键信息"""
        profile = UserProfile(
            total_memories=100,
            expertise_level="advanced",
            domain_keywords=["python", "docker", "kubernetes"],
            preferences=["喜欢川菜"],
        )
        summary = profile.summary()
        assert "100" in summary
        assert "advanced" in summary
        assert "python" in summary


class TestInteractionPattern:
    """InteractionPattern 数据模型"""

    def test_default(self):
        pattern = InteractionPattern()
        assert pattern.total_queries == 0
        assert pattern.active_hours == []

    def test_with_data(self):
        pattern = InteractionPattern(
            total_queries=50,
            active_hours=[9, 14, 20],
            avg_query_length=25.5,
        )
        assert pattern.total_queries == 50
        assert 9 in pattern.active_hours


# ============================================================
# UserProfileEngine 核心功能
# ============================================================

class TestProfileEngineKeywords:
    """关键词提取"""

    def test_extract_keywords_basic(self, dietary_client):
        engine = UserProfileEngine(dietary_client)
        contents = [m.content for m in dietary_client._memories]
        keywords = engine._extract_keywords(contents, 15)
        assert len(keywords) >= 5, f"关键词太少: {keywords}"
        # 应该包含食物相关词
        food_keywords = [k for k in keywords if any(
            w in k for w in ["食物", "饮食", "辣", "咖啡", "茶"]
        )]
        assert len(food_keywords) >= 1, f"未检测到食物关键词: {keywords}"

    def test_extract_keywords_top_n(self, dietary_client):
        engine = UserProfileEngine(dietary_client)
        contents = [m.content for m in dietary_client._memories]
        keywords = engine._extract_keywords(contents, top_n=5)
        assert len(keywords) == 5

    def test_extract_keywords_empty_contents(self, empty_client):
        engine = UserProfileEngine(empty_client)
        keywords = engine._extract_keywords([], 10)
        assert keywords == []

    def test_tokenize_chinese(self, empty_client):
        engine = UserProfileEngine(empty_client)
        tokens = engine._tokenize("人工智能技术在医疗领域的突破")
        assert len(tokens) >= 2

    def test_tokenize_mixed(self, empty_client):
        engine = UserProfileEngine(empty_client)
        tokens = engine._tokenize("Python和Go是常用编程语言")
        assert "python" in tokens or "go" in tokens or "编程" in tokens or "语言" in tokens


class TestProfileEngineDomains:
    """领域分类"""

    def test_classify_domains_dietary(self, dietary_client):
        engine = UserProfileEngine(dietary_client)
        contents = [m.content for m in dietary_client._memories]
        domains = engine._classify_domains(contents)
        assert isinstance(domains, dict)
        # 应该有 healthcare 和 daily_life 领域
        assert len(domains) >= 1

    def test_classify_domains_tech(self, tech_client):
        engine = UserProfileEngine(tech_client)
        contents = [m.content for m in tech_client._memories]
        domains = engine._classify_domains(contents)
        # 技术内容应匹配 programming 领域
        assert "programming" in domains, f"未匹配编程领域: {domains}"

    def test_classify_domains_empty(self, empty_client):
        engine = UserProfileEngine(empty_client)
        domains = engine._classify_domains([])
        assert domains == {}


class TestProfileEnginePreferences:
    """偏好提取"""

    def test_extract_preferences(self, dietary_client):
        engine = UserProfileEngine(dietary_client)
        contents = [m.content for m in dietary_client._memories]
        prefs = engine._extract_preferences(contents)
        assert isinstance(prefs, list)

    def test_infer_implicit_preferences(self, tech_client):
        engine = UserProfileEngine(tech_client)
        contents = [m.content for m in tech_client._memories]
        domains = engine._classify_domains(contents)
        implicit = engine._infer_implicit_preferences(contents, domains)
        assert isinstance(implicit, list)
        # 应在 programming 领域有隐式偏好
        programming_prefs = [p for p in implicit if "programming" in p.lower() or "Python" in p]
        assert len(programming_prefs) >= 1, f"未检测到编程隐式偏好: {implicit}"


class TestProfileEngineExpertise:
    """专业水平评估"""

    def test_assess_expertise_novice(self, empty_client):
        engine = UserProfileEngine(empty_client)
        level, domains = engine._assess_expertise(5, {}, [])
        assert level == "novice"

    def test_assess_expertise_intermediate(self, empty_client):
        engine = UserProfileEngine(empty_client)
        level, domains = engine._assess_expertise(50, {}, ["a" * 100])
        assert level == "intermediate"

    def test_assess_expertise_advanced(self, empty_client):
        engine = UserProfileEngine(empty_client)
        level, domains = engine._assess_expertise(150, {}, ["a" * 100])
        assert level == "advanced"

    def test_assess_expertise_expert(self, empty_client):
        engine = UserProfileEngine(empty_client)
        level, domains = engine._assess_expertise(600, {}, ["a" * 100])
        assert level == "expert"


class TestProfileEngineConstraints:
    """约束与目标提取"""

    def test_extract_constraints_empty(self, empty_client):
        engine = UserProfileEngine(empty_client)
        constraints = engine._extract_constraints([])
        assert constraints == []

    def test_extract_goals_empty(self, empty_client):
        engine = UserProfileEngine(empty_client)
        goals = engine._extract_goals([])
        assert goals == []


class TestProfileEngineLearning:
    """学习状态"""

    def test_learning_velocity_few_memories(self, empty_client):
        engine = UserProfileEngine(empty_client)
        # < 10 条记忆 → 0.0
        memories = [{"content": f"记忆{i}"} for i in range(5)]
        velocity = engine._compute_learning_velocity(memories)
        assert velocity == 0.0

    def test_learning_velocity_normal(self, dietary_client):
        engine = UserProfileEngine(dietary_client)
        memories = [{"content": m.content} for m in dietary_client._memories]
        velocity = engine._compute_learning_velocity(memories)
        # 8 条记忆 → 仍小于 10，返回 0
        assert velocity == 0.0

    def test_knowledge_gaps(self, dietary_client):
        engine = UserProfileEngine(dietary_client)
        contents = [m.content for m in dietary_client._memories]
        domains = engine._classify_domains(contents)
        gaps = engine._identify_knowledge_gaps(domains, contents)
        assert isinstance(gaps, list)


class TestProfileEngineCore:
    """核心提取流程"""

    def test_extract_from_memories_dietary(self, dietary_client):
        """从饮食数据提取画像"""
        engine = UserProfileEngine(dietary_client)
        profile = engine.extract_from_memories()
        assert isinstance(profile, UserProfile)
        assert profile.total_memories == 8
        assert profile.extracted_at != ""

    def test_extract_from_memories_empty(self, empty_client):
        """空记忆库返回空画像（不崩溃）"""
        engine = UserProfileEngine(empty_client)
        profile = engine.extract_from_memories()
        assert isinstance(profile, UserProfile)
        assert profile.total_memories == 0

    def test_get_profile_cached(self, dietary_client):
        """缓存画像返回"""
        engine = UserProfileEngine(dietary_client)
        p1 = engine.get_profile()
        p2 = engine.get_profile()
        assert p1 is p2  # 相同对象（缓存）
        assert engine.version >= 1

    def test_update_incremental_no_change(self, dietary_client):
        """无变化时增量更新不触发全量重算"""
        engine = UserProfileEngine(dietary_client)
        engine.extract_from_memories()  # 首次提取
        v1 = engine.version
        engine.update_incremental()
        assert engine.version == v1  # 无变化，版本不变

    def test_deduplicate_list(self):
        """去重辅助"""
        items = ["a", "b", "A", "a", "c"]
        result = UserProfileEngine._deduplicate_list(items)
        assert len(result) <= 4  # "a" 和 "A" 被视为重复（小写相同）

    def test_extract_sources(self, dietary_client):
        engine = UserProfileEngine(dietary_client)
        memories = [{"metadata": {"source": "test_source"}}]
        sources = engine._extract_sources(memories)
        assert "test_source" in sources

    def test_version_increment(self, dietary_client):
        engine = UserProfileEngine(dietary_client)
        assert engine.version == 0
        engine.extract_from_memories()
        assert engine.version == 1
        engine.extract_from_memories()
        assert engine.version == 2


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
