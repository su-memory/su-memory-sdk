"""
embedding.py 纯算法与兜底路径测试

覆盖: 余弦相似度、RRF 融合、加权组合融合、哈希/简易/兜底嵌入
(均为确定性纯计算, 不依赖任何外部模型服务)。
"""
import asyncio
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from su_memory.sdk.embedding import (
    EmbeddingResult,
    HashFallbackEmbedding,
    MiniMaxEmbedding,
    OllamaEmbedding,
    OpenAIEmbedding,
    cosine_similarity,
    rrf_fusion,
    weighted_combination_fusion,
)


def _bare(cls, dims):
    """绕过 __init__ 直接构造, 只设置维度(纯算法路径无外部依赖)"""
    obj = cls.__new__(cls)
    obj.dims = dims
    return obj


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert math.isclose(cosine_similarity([1.0, 2.0], [1.0, 2.0]), 1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
        assert cosine_similarity([1.0, 1.0], [0.0, 0.0]) == 0.0

    def test_partial_similarity(self):
        s = cosine_similarity([1.0, 1.0], [1.0, 0.0])
        assert math.isclose(s, 1 / math.sqrt(2))


class TestHashEmbeddings:
    def test_hash_embedding_normalized_and_deterministic(self):
        a = _bare(MiniMaxEmbedding, 16)
        v1 = a._hash_embedding("医疗记忆测试")
        v2 = a._hash_embedding("医疗记忆测试")
        assert v1 == v2
        assert len(v1) == 16
        norm = sum(x * x for x in v1) ** 0.5
        assert math.isclose(norm, 1.0)

    def test_hash_embedding_empty_text(self):
        a = _bare(MiniMaxEmbedding, 8)
        assert a._hash_embedding("") == [0.0] * 8

    def test_ollama_fallback_embedding(self):
        a = _bare(OllamaEmbedding, 8)
        v = a._fallback_embedding("abc")
        assert len(v) == 8
        assert any(x != 0 for x in v)

    def test_openai_simple_embedding(self):
        a = _bare(OpenAIEmbedding, 8)
        v = a._simple_embedding("你好世界")
        assert len(v) == 8
        norm = sum(x * x for x in v) ** 0.5
        assert math.isclose(norm, 1.0)


class TestHashFallbackEmbedding:
    def test_encode(self):
        e = HashFallbackEmbedding(dims=32)
        v1, v2 = e.encode("同一文本"), e.encode("同一文本")
        assert v1 == v2 and len(v1) == 32

    def test_aencode(self):
        e = HashFallbackEmbedding(dims=16)
        out = asyncio.run(e.aencode("异步测试"))
        assert isinstance(out, EmbeddingResult)
        assert out.model == "hash_fallback"
        assert out.embedding == e.encode("异步测试")


class TestRrfFusion:
    def test_single_method(self):
        out = rrf_fusion([[("a", 1.0), ("b", 0.5)]])
        assert out[0][0] == "a"

    def test_multiple_methods_merge_scores(self):
        out = rrf_fusion([
            [("a", 1.0), ("b", 0.5)],
            [("b", 2.0), ("c", 1.0)],
        ])
        ids = [doc for doc, _ in out]
        assert ids[0] == "b"  # 两路都命中的文档 RRF 得分更高

    def test_no_score_weight(self):
        out = rrf_fusion([[("a", 1.0), ("b", 0.5)]], use_score_weight=False)
        # 不使用分数权重时得分仅与排序位置相关
        assert out[0][0] == "a"

    def test_method_weights(self):
        out = rrf_fusion([
            [("a", 1.0), ("b", 1.0)],
            [("b", 1.0), ("a", 1.0)],
        ], method_weights=[2.0, 1.0])
        assert out[0][0] == "a"  # 方法0权重更高

    def test_empty_inputs(self):
        assert rrf_fusion([]) == []
        assert rrf_fusion([[]]) == []


class TestWeightedCombinationFusion:
    def test_equal_weights_by_default(self):
        out = weighted_combination_fusion([
            [("a", 1.0), ("b", 0.0)],
            [("a", 0.0), ("b", 1.0)],
        ])
        assert out[0][1] == out[1][1]  # 等权时 a/b 平分

    def test_custom_weights(self):
        out = weighted_combination_fusion([
            [("a", 1.0), ("b", 0.5)],
            [("b", 1.0), ("a", 0.5)],
        ], weights=[1.0, 0.0])
        assert out[0][0] == "a"

    def test_nonpositive_scores_safe(self):
        out = weighted_combination_fusion([
            [("a", 0.0), ("b", -1.0)],
        ])
        # 负分/零分归一化为 0, 不抛错
        assert isinstance(out, list)
