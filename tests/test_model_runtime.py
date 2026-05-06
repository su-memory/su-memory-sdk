#!/usr/bin/env python3
"""
M2-T2: Model Runtime 测试 — LocalModels + EnergyCore

覆盖：
- SimpleLinearModel: 线性回归完整生命周期
- NaiveBayesClassifier: 分类器 + predict_proba
- TFIDFRanker: 文本排序 + 余弦相似度
- PredictionCache: LRU/LFU/TTL/FIFO 策略
- LocalModelManager: 模型注册/预测/回退
- Factory functions
- EnergyCore: 五行能量全部方法
"""

import sys
import os
import time
import math
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from su_memory._sys._local_models import (
    SimpleLinearModel,
    NaiveBayesClassifier,
    TFIDFRanker,
    PredictionCache,
    LocalModelManager,
    ModelConfig,
    ModelType,
    PredictionStatus,
    PredictionResult,
    CacheEvictionPolicy,
    create_linear_model,
    create_naive_bayes,
    create_tfidf_ranker,
    create_prediction_cache,
    create_model_manager,
)
from su_memory._sys._energy_core import (
    EnergyCore,
    EnergyState,
    EnergyBalanceResult,
    EnergyFlow,
    EnergyType,
    EnergyRelation,
    StrengthState,
    EnergyPattern,
)


# ============================================================
# SimpleLinearModel 测试
# ============================================================

class TestSimpleLinearModel:
    """SimpleLinearModel 全面测试"""

    def test_initialization(self):
        """初始化参数"""
        model = SimpleLinearModel(input_dim=4, output_dim=1)
        assert model._input_dim == 4
        assert model._output_dim == 1
        assert model._lr == 0.01
        assert model._reg == 0.01
        assert model.is_fitted is False

    def test_fit_predict_basic(self):
        """基本训练预测"""
        model = SimpleLinearModel(input_dim=2, output_dim=1, learning_rate=0.1)
        X = [[1.0, 2.0]] * 20 + [[3.0, 4.0]] * 20
        y = [5.0] * 20 + [11.0] * 20
        model.fit(X, y, epochs=50)
        assert model.is_fitted
        pred = model.predict([1.0, 2.0])
        assert isinstance(pred, float)

    def test_predict_before_fit_raises(self):
        """未训练时预测抛出异常"""
        model = SimpleLinearModel(input_dim=4)
        import pytest
        with pytest.raises(RuntimeError):
            model.predict([1.0, 2.0, 3.0, 4.0])

    def test_dimension_mismatch_raises(self):
        """维度不匹配抛出异常"""
        model = SimpleLinearModel(input_dim=4, output_dim=1)
        X = [[1.0, 2.0, 3.0, 4.0]] * 10
        y = [10.0] * 10
        model.fit(X, y, epochs=10)
        import pytest
        with pytest.raises(ValueError):
            model._forward([1.0, 2.0, 3.0])  # 3维输入

    def test_data_length_mismatch(self):
        """X和y长度不匹配"""
        model = SimpleLinearModel(input_dim=4)
        import pytest
        with pytest.raises(ValueError):
            model.fit([[1.0, 2.0, 3.0, 4.0]] * 5, [1.0] * 3)

    def test_predict_batch(self):
        """批量预测"""
        model = SimpleLinearModel(input_dim=2, learning_rate=0.1)
        X = [[1.0, 1.0]] * 10 + [[3.0, 3.0]] * 10
        y = [2.0] * 10 + [6.0] * 10
        model.fit(X, y, epochs=50)
        preds = model.predict_batch([[1.0, 1.0], [3.0, 3.0]])
        assert len(preds) == 2
        assert isinstance(preds[0], float)
        assert isinstance(preds[1], float)

    def test_loss_decreases(self):
        """损失随训练下降"""
        model = SimpleLinearModel(input_dim=4, learning_rate=0.001)
        X = [[0.1, 0.2, 0.3, 0.4] for _ in range(20)]
        y = [2*x[0] + 3*x[1] + x[2] - x[3] for x in X]
        initial_loss = model._calculate_loss(X, y)
        model.fit(X, y, epochs=100)
        final_loss = model._calculate_loss(X, y)
        assert final_loss < initial_loss, f"Loss should decrease: {final_loss} >= {initial_loss}"

    def test_get_weights_and_biases(self):
        """获取权重和偏置"""
        model = SimpleLinearModel(input_dim=3, output_dim=2)
        X = [[1.0, 2.0, 3.0]] * 15 + [[4.0, 5.0, 6.0]] * 15
        y = [6.0] * 15 + [15.0] * 15
        model.fit(X, y, epochs=20)
        w = model.get_weights()
        b = model.get_biases()
        assert len(w) == 2  # output_dim=2
        assert len(w[0]) == 3  # input_dim=3
        assert len(b) == 2

    def test_multi_output_fit(self):
        """多输出维度训练"""
        model = SimpleLinearModel(input_dim=2, output_dim=2, learning_rate=0.1)
        X = [[1.0, 2.0]] * 10 + [[3.0, 4.0]] * 10
        y = [5.0] * 10 + [11.0] * 10
        model.fit(X, y, epochs=30)
        assert model.is_fitted

    def test_repr(self):
        """字符串表示"""
        model = SimpleLinearModel(input_dim=4)
        r = repr(model)
        assert "SimpleLinearModel" in r
        assert "fitted=False" in r


# ============================================================
# NaiveBayesClassifier 测试
# ============================================================

class TestNaiveBayesClassifier:
    """NaiveBayesClassifier 全面测试"""

    def test_fit_predict(self):
        """基本训练预测"""
        nb = NaiveBayesClassifier(alpha=1.0)
        X = [
            ['sunny', 'hot', 'high', 'weak'],
            ['sunny', 'hot', 'high', 'strong'],
            ['overcast', 'hot', 'high', 'weak'],
            ['rain', 'mild', 'high', 'weak'],
            ['rain', 'cool', 'normal', 'weak'],
            ['rain', 'cool', 'normal', 'strong'],
            ['overcast', 'cool', 'normal', 'strong'],
            ['sunny', 'mild', 'high', 'weak'],
        ]
        y = ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no']
        nb.fit(X, y)
        assert nb.is_fitted
        result = nb.predict(['sunny', 'hot', 'high', 'weak'])
        assert result.is_success
        assert 0.0 <= result.confidence <= 1.0

    def test_predict_before_fit(self):
        """未训练时预测"""
        nb = NaiveBayesClassifier()
        result = nb.predict(['sunny', 'hot', 'high', 'weak'])
        assert result.status == PredictionStatus.MODEL_NOT_LOADED

    def test_predict_proba(self):
        """概率分布预测"""
        nb = NaiveBayesClassifier(alpha=1.0)
        X = [
            ['a', 'x'], ['a', 'x'], ['a', 'y'],
            ['b', 'x'], ['b', 'y'], ['b', 'y'],
        ]
        y = ['pos', 'pos', 'pos', 'neg', 'neg', 'neg']
        nb.fit(X, y)
        proba = nb.predict_proba(['a', 'x'])
        assert len(proba) >= 1

    def test_missing_feature_handling(self):
        """缺失特征处理"""
        nb = NaiveBayesClassifier(alpha=1.0)
        X = [['a', 'x'], ['a', 'x'], ['b', 'y']]
        y = ['yes', 'yes', 'no']
        nb.fit(X, y)
        result = nb.predict(['a', 'z'])  # 'z' 未在训练中出现
        assert result.is_success

    def test_alpha_smoothing(self):
        """拉普拉斯平滑"""
        nb1 = NaiveBayesClassifier(alpha=0.1)
        nb2 = NaiveBayesClassifier(alpha=5.0)
        X = [['a']] * 3 + [['b']] * 2
        y = ['yes'] * 3 + ['no'] * 2
        nb1.fit(X, y)
        nb2.fit(X, y)
        # 不同 alpha 产生不同概率
        p1 = nb1.predict(['a']).confidence
        p2 = nb2.predict(['a']).confidence
        # 两者都应 > 0
        assert p1 > 0 and p2 > 0

    def test_repr(self):
        """字符串表示"""
        nb = NaiveBayesClassifier()
        r = repr(nb)
        assert "NaiveBayesClassifier" in r


# ============================================================
# TFIDFRanker 测试
# ============================================================

class TestTFIDFRanker:
    """TFIDFRanker 全面测试"""

    def test_fit_and_rank(self):
        """基本训练和排序"""
        ranker = TFIDFRanker(max_features=100)
        docs = [
            "machine learning is a subset of artificial intelligence",
            "deep learning uses neural networks with multiple layers",
            "natural language processing deals with text and speech",
            "computer vision enables machines to interpret images",
            "machine learning and deep learning are related fields",
        ]
        ranker.fit(docs)
        assert ranker.is_fitted
        assert ranker.vocabulary_size > 0

        results = ranker.rank("machine learning neural networks", docs, top_k=3)
        assert len(results) <= 3

    def test_rank_before_fit(self):
        """未训练时排序"""
        ranker = TFIDFRanker()
        results = ranker.rank("test", ["doc1"], top_k=5)
        assert results == []

    def test_cosine_similarity_identical(self):
        """余弦相似度 — 相同向量"""
        ranker = TFIDFRanker()
        vec = {0: 0.5, 1: 0.3, 2: 0.2}
        sim = ranker._cosine_similarity(vec, vec)
        assert 0.99 <= sim <= 1.01  # 应接近 1.0

    def test_cosine_similarity_orthogonal(self):
        """余弦相似度 — 正交向量"""
        ranker = TFIDFRanker()
        sim = ranker._cosine_similarity({0: 1.0}, {1: 1.0})
        assert sim == 0.0

    def test_cosine_similarity_empty(self):
        """余弦相似度 — 空向量"""
        ranker = TFIDFRanker()
        sim = ranker._cosine_similarity({}, {0: 1.0})
        assert sim == 0.0

    def test_tokenize(self):
        """分词功能"""
        ranker = TFIDFRanker()
        tokens = ranker._tokenize("Hello World test document")
        assert isinstance(tokens, list)
        assert all(len(t) > 2 for t in tokens)

    def test_no_documents_rank(self):
        """无文档时排序"""
        ranker = TFIDFRanker()
        docs = ["doc one", "doc two"]
        ranker.fit(docs)
        results = ranker.rank("query", None, top_k=3)  # None docs
        assert results == []

    def test_vocabulary_limit(self):
        """词汇表大小限制"""
        ranker = TFIDFRanker(max_features=5)
        docs = ["word" + str(i) for i in range(20)]
        ranker.fit(docs)
        assert ranker.vocabulary_size <= 5

    def test_repr(self):
        """字符串表示"""
        ranker = TFIDFRanker()
        r = repr(ranker)
        assert "TFIDFRanker" in r


# ============================================================
# PredictionCache 测试
# ============================================================

class TestPredictionCache:
    """PredictionCache 全面测试"""

    def test_put_and_get(self):
        """基本存储和获取"""
        cache = PredictionCache(max_size=10)
        cache.put("key1", {"value": 42})
        result = cache.get("key1")
        assert result == {"value": 42}

    def test_miss(self):
        """缓存未命中"""
        cache = PredictionCache()
        result = cache.get("nonexistent")
        assert result is None

    def test_max_size_eviction(self):
        """达到最大容量后淘汰"""
        cache = PredictionCache(max_size=5, eviction_policy=CacheEvictionPolicy.LRU)
        for i in range(10):
            cache.put(f"key{i}", i)
        assert len(cache) <= 5

    def test_lru_policy(self):
        """LRU 淘汰策略"""
        cache = PredictionCache(max_size=3, eviction_policy=CacheEvictionPolicy.LRU)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.get("a")  # 访问 a，使其成为最近使用
        cache.put("d", 4)  # 应淘汰 b
        assert cache.get("a") == 1  # a 仍存在
        assert cache.get("b") is None  # b 可能被淘汰
        assert cache.get("d") == 4

    def test_lfu_policy(self):
        """LFU 淘汰策略"""
        cache = PredictionCache(max_size=3, eviction_policy=CacheEvictionPolicy.LFU)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.get("a")
        cache.get("a")
        cache.get("b")
        cache.put("d", 4)
        assert len(cache) <= 3

    def test_fifo_policy(self):
        """FIFO 淘汰策略"""
        cache = PredictionCache(max_size=3, eviction_policy=CacheEvictionPolicy.FIFO)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.put("d", 4)
        assert cache.get("a") is None  # a 最先进入

    def test_clear(self):
        """清空缓存"""
        cache = PredictionCache(max_size=10)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()
        assert len(cache) == 0
        assert cache.get("a") is None

    def test_stats(self):
        """缓存统计"""
        cache = PredictionCache(max_size=100)
        cache.put("a", 1)
        stats = cache.get_stats()
        assert stats["size"] == 1
        assert stats["max_size"] == 100
        assert "policy" in stats


# ============================================================
# LocalModelManager 测试
# ============================================================

class TestLocalModelManager:
    """LocalModelManager 全面测试"""

    def test_register_and_predict(self):
        """注册模型并预测"""
        manager = LocalModelManager()
        model = SimpleLinearModel(input_dim=4)
        X = [[1.0, 2.0, 3.0, 4.0]] * 20 + [[5.0, 6.0, 7.0, 8.0]] * 20
        y = [10.0] * 20 + [26.0] * 20
        model.fit(X, y, epochs=30)
        manager.register_model("linear", model)
        assert "linear" in manager.list_models()
        result = manager.predict("linear", [1.0, 2.0, 3.0, 4.0])
        assert result.is_success

    def test_predict_unregistered_model(self):
        """预测未注册模型"""
        manager = LocalModelManager()
        result = manager.predict("nonexistent", [1.0, 2.0, 3.0])
        assert result.status == PredictionStatus.MODEL_NOT_LOADED

    def test_duplicate_register(self):
        """重复注册"""
        manager = LocalModelManager()
        model = create_linear_model(4)
        assert manager.register_model("test", model) is True
        assert manager.register_model("test", model) is False

    def test_unregister(self):
        """注销模型"""
        manager = LocalModelManager()
        model = create_linear_model(4)
        manager.register_model("test", model)
        assert manager.unregister_model("test") is True
        assert manager.unregister_model("test") is False
        assert "test" not in manager.list_models()

    def test_clear_cache(self):
        """清空缓存"""
        manager = LocalModelManager()
        model = SimpleLinearModel(input_dim=4)
        X = [[1.0, 2.0, 3.0, 4.0]] * 10
        y = [10.0] * 10
        model.fit(X, y, epochs=20)
        manager.register_model("linear", model)
        manager.predict("linear", [1.0, 2.0, 3.0, 4.0])
        manager.clear_cache("linear")
        manager.clear_cache()  # 清空所有

    def test_model_info(self):
        """模型信息"""
        manager = LocalModelManager()
        model = create_linear_model(8)
        manager.register_model("test_linear", model)
        info = manager.get_model_info("test_linear")
        assert info is not None
        assert info["type"] == "SimpleLinearModel"
        assert info["is_fitted"] is False

    def test_model_info_nonexistent(self):
        """不存在的模型信息"""
        manager = LocalModelManager()
        info = manager.get_model_info("nonexistent")
        assert info is None

    def test_predict_with_fallback(self):
        """回退预测"""
        manager = LocalModelManager()
        model = SimpleLinearModel(input_dim=4)
        X = [[1.0, 2.0, 3.0, 4.0]] * 10
        y = [10.0] * 10
        model.fit(X, y, epochs=20)
        manager.register_model("primary", model)
        result = manager.predict_with_fallback("primary", "nonexistent", [1.0, 2.0, 3.0, 4.0])
        assert result.is_success

    def test_repr(self):
        """字符串表示"""
        manager = LocalModelManager()
        r = repr(manager)
        assert "LocalModelManager" in r


# ============================================================
# Factory functions 测试
# ============================================================

class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_linear_model(self):
        model = create_linear_model(input_dim=8, output_dim=2)
        assert model._input_dim == 8
        assert model._output_dim == 2

    def test_create_naive_bayes(self):
        nb = create_naive_bayes(alpha=2.0)
        assert nb._alpha == 2.0

    def test_create_tfidf_ranker(self):
        ranker = create_tfidf_ranker(max_features=500)
        assert ranker._max_features == 500

    def test_create_prediction_cache(self):
        cache = create_prediction_cache(max_size=500, ttl=1800)
        assert len(cache) == 0
        stats = cache.get_stats()
        assert stats["max_size"] == 500

    def test_create_model_manager(self):
        manager = create_model_manager()
        assert isinstance(manager, LocalModelManager)
        assert manager.list_models() == []


# ============================================================
# EnergyCore 测试
# ============================================================

class TestEnergyCore:
    """EnergyCore 五行能量核心引擎测试"""

    def setup_method(self):
        self.ec = EnergyCore()

    def test_enhance_relations(self):
        """相生关系测试"""
        assert self.ec.get_enhance_relation("wood", "fire") is True
        assert self.ec.get_enhance_relation("fire", "earth") is True
        assert self.ec.get_enhance_relation("earth", "metal") is True
        assert self.ec.get_enhance_relation("metal", "water") is True
        assert self.ec.get_enhance_relation("water", "wood") is True
        # 反向不应为相生
        assert self.ec.get_enhance_relation("fire", "wood") is False

    def test_suppress_relations(self):
        """相克关系测试（双向）"""
        assert self.ec.get_suppress_relation("wood", "earth") is True
        assert self.ec.get_suppress_relation("earth", "wood") is True  # bidirectional
        assert self.ec.get_suppress_relation("earth", "water") is True
        assert self.ec.get_suppress_relation("water", "fire") is True
        assert self.ec.get_suppress_relation("fire", "metal") is True
        assert self.ec.get_suppress_relation("metal", "wood") is True

    def test_energy_state_by_month(self):
        """按月查询能量状态（旺相休囚死）"""
        # 寅月(2): 木旺
        state = self.ec.get_energy_state("wood", 2)
        assert state.strength == StrengthState.WANG
        # 巳月(5): 火旺
        state = self.ec.get_energy_state("fire", 5)
        assert state.strength == StrengthState.WANG
        # 子月(0): 水旺
        state = self.ec.get_energy_state("water", 0)
        assert state.strength == StrengthState.WANG
        # 申月(8): 金旺
        state = self.ec.get_energy_state("metal", 8)
        assert state.strength == StrengthState.WANG

    def test_intensity_values(self):
        """强度值计算"""
        state = self.ec.get_energy_state("wood", 2)  # WANG
        assert abs(state.intensity - 1.2) < 0.01
        state = self.ec.get_energy_state("wood", 0)  # XIANG
        assert abs(state.intensity - 1.0) < 0.01

    def test_invalid_branch(self):
        """无效地支"""
        import pytest
        with pytest.raises(ValueError):
            self.ec.get_energy_state("wood", 12)
        with pytest.raises(ValueError):
            self.ec.get_energy_state("wood", -1)

    def test_strength_from_branch(self):
        """从地支获取所有五行强度"""
        strengths = self.ec.get_strength_from_branch(2)  # 寅月
        assert len(strengths) == 5
        assert strengths["wood"] == StrengthState.WANG

    def test_balance_analysis(self):
        """平衡分析"""
        energies = {"wood": 0.3, "fire": 0.2, "earth": 0.2, "metal": 0.15, "water": 0.15}
        result = self.ec.analyze_balance(energies)
        assert result.status in ("balanced", "imbalanced")
        assert result.pattern is not None
        assert result.dominant in self.ec.ENERGY_ORDER
        assert len(result.suggestions) > 0

    def test_balance_zero_energies(self):
        """零能量值异常"""
        import pytest
        with pytest.raises(ValueError):
            self.ec.analyze_balance({"wood": 0, "fire": 0, "earth": 0, "metal": 0, "water": 0})

    def test_energy_attributes(self):
        """能量属性查询"""
        attrs = self.ec.get_energy_attributes("wood")
        assert "name" in attrs
        assert "chinese_name" in attrs
        assert "season" in attrs
        assert "enhances" in attrs
        assert "suppresses" in attrs

    def test_compatibility(self):
        """兼容性计算"""
        e1 = {"wood": 0.4, "fire": 0.2, "earth": 0.2, "metal": 0.1, "water": 0.1}
        e2 = {"wood": 0.3, "fire": 0.3, "earth": 0.2, "metal": 0.1, "water": 0.1}
        compat = self.ec.calculate_compatibility(e1, e2)
        assert 0.0 <= compat <= 1.0

    def test_compatibility_empty(self):
        """空输入的兼容性"""
        assert self.ec.calculate_compatibility({}, {"wood": 1.0}) == 0.0
        assert self.ec.calculate_compatibility({"wood": 0}, {"fire": 0}) == 0.0

    def test_interaction_analysis(self):
        """交互分析"""
        interactions = self.ec.analyze_interaction("wood", "fire")
        assert EnergyRelation.ENHANCE in interactions
        interactions = self.ec.analyze_interaction("wood", "earth")
        assert EnergyRelation.SUPPRESS in interactions

    def test_same_energy_interaction(self):
        """相同能量的交互"""
        interactions = self.ec.analyze_interaction("wood", "wood")
        assert EnergyRelation.SAME in interactions

    def test_energy_flow_simulation(self):
        """能量流转模拟"""
        initial = {"wood": 0.3, "fire": 0.2, "earth": 0.2, "metal": 0.15, "water": 0.15}
        history = self.ec.simulate_energy_flow(initial, steps=5)
        assert len(history) == 6  # initial + 5 steps
        for step in history:
            assert sum(step.values()) > 0

    def test_energy_cycle(self):
        """五行相生循环"""
        cycle = self.ec.get_energy_cycle()
        assert len(cycle) == 5
        assert cycle[0] == ("wood", "fire")

    def test_control_cycle(self):
        """五行相克循环"""
        cycle = self.ec.get_control_cycle()
        assert len(cycle) == 5

    def test_opposing_pair(self):
        """敌对配对"""
        pair = self.ec.get_opposing_pair("wood")
        assert pair[0] == "fire"  # enhance to
        assert pair[1] == "earth"  # suppress to

    def test_apply_balance_rules(self):
        """应用平衡规则"""
        energies = {"wood": 0.3, "fire": 0.2, "earth": 0.2, "metal": 0.15, "water": 0.15}
        for pattern in EnergyPattern:
            result = self.ec.apply_balance_rules(energies, pattern)
            assert len(result) == 5

    def test_normalize_energy_with_enum(self):
        """EnergyType 枚举标准化"""
        result = self.ec._normalize_energy(EnergyType.WOOD)
        assert result == "wood"

    def test_energy_state_is_enhanced(self):
        """is_enhanced 属性"""
        state = EnergyState(
            energy_type=EnergyType.WOOD,
            strength=StrengthState.WANG,
            intensity=1.2,
        )
        assert state.is_enhanced is True
        state2 = EnergyState(
            energy_type=EnergyType.WATER,
            strength=StrengthState.SI,
            intensity=0.3,
        )
        assert state2.is_enhanced is False

    def test_energy_balance_result_to_dict(self):
        """EnergyBalanceResult.to_dict()"""
        result = EnergyBalanceResult(
            status="balanced",
            pattern=EnergyPattern.PEI_HE,
            ratios={"wood": 0.2, "fire": 0.2, "earth": 0.2, "metal": 0.2, "water": 0.2},
            dominant="wood",
            suggestions=["保持当前状态"],
        )
        d = result.to_dict()
        assert d["status"] == "balanced"
        assert d["pattern"] == "PEI_HE"

    def test_overconstraint_relation(self):
        """相乘关系"""
        assert self.ec.get_overconstraint_relation("wood", "earth") is True
        assert self.ec.get_overconstraint_relation("wood", "fire") is False

    def test_reverse_relation(self):
        """相侮关系"""
        assert self.ec.get_reverse_relation("earth", "wood") is True
        assert self.ec.get_reverse_relation("wood", "earth") is False
