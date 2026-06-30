"""LLMReader + 标准 EM 归一化 单元测试.

覆盖纯函数 (squad_normalize/em/f1) 与 MultiHopReader 的 LLM 接入路径
(LLM 失败时回退启发式, 不破坏可用性). 不依赖本地模型加载.
"""
import numpy as np
import pytest

from su_memory.sdk.llm_reader import squad_normalize, squad_em, squad_f1
from su_memory.sdk.multi_hop_reader import MultiHopReader


class TestSquadNormalize:
    def test_lowercase(self):
        assert squad_normalize("Hello") == "hello"

    def test_strip_punctuation(self):
        assert squad_normalize("hello, world!") == "hello world"

    def test_strip_articles(self):
        assert squad_normalize("the answer") == "answer"
        assert squad_normalize("a an the") == ""

    def test_collapse_spaces(self):
        assert squad_normalize("  a   b  ") == "b"

    def test_empty(self):
        assert squad_normalize("") == ""


class TestSquadEM:
    def test_exact(self):
        assert squad_em("New York", "New York")

    def test_case_insensitive(self):
        assert squad_em("new york", "New York")

    def test_punctuation_insensitive(self):
        assert squad_em("New York!", "New York")

    def test_article_insensitive(self):
        assert squad_em("the United States", "United States")

    def test_mismatch(self):
        assert not squad_em("Paris", "London")

    def test_gold_in_pred_not_em(self):
        # 严格相等: pred 超集不算 EM
        assert not squad_em("New York City", "New York")


class TestSquadF1:
    def test_perfect(self):
        assert f1_token_partial("New York", "New York") == 1.0

    def test_partial_overlap(self):
        f = f1_token_partial("New York City", "New York")
        assert 0 < f < 1.0

    def test_no_overlap(self):
        assert f1_token_partial("Paris", "London") == 0.0


def f1_token_partial(pred, gold):
    return squad_f1(pred, gold)


class TestMultiHopReaderLLMIntegration:
    """LLM reader 接入路径: 有 LLM 时走 LLM, 失败时回退启发式."""

    def _fake_embed(self, dim=8):
        def embed(text):
            h = hash(text) % 1000
            rng = np.random.default_rng(h)
            return rng.standard_normal(dim).astype(np.float32)

        def embed_batch(texts):
            return np.stack([embed(t) for t in texts])

        return embed, embed_batch

    def test_no_llm_falls_back_to_heuristic(self):
        """无 llm_reader 时 extract_answer 用启发式 (不报错)."""
        e, eb = self._fake_embed()
        reader = MultiHopReader(e, eb, llm_reader=None)
        ans = reader.extract_answer("Who is Alpha?", "Alpha Corp is big.")
        assert isinstance(ans, str)

    def test_llm_failure_falls_back(self):
        """LLM reader 抛异常时, extract_answer 回退启发式, 不破坏可用性."""
        e, eb = self._fake_embed()

        class BrokenLLM:
            def extract_answer(self, q, c):
                raise RuntimeError("model unavailable")

        reader = MultiHopReader(e, eb, llm_reader=BrokenLLM())
        ans = reader.extract_answer("Who is Alpha?", "Alpha Corp is big.")
        assert isinstance(ans, str)  # 回退成功, 无异常

    def test_llm_used_when_available(self):
        """LLM reader 正常时, extract_answer 返回 LLM 结果."""
        e, eb = self._fake_embed()

        class StubLLM:
            def __init__(self):
                self.called = False

            def extract_answer(self, q, c):
                self.called = True
                return "stub answer"

        stub = StubLLM()
        reader = MultiHopReader(e, eb, llm_reader=stub)
        ans = reader.extract_answer("q", "context")
        assert ans == "stub answer"
        assert stub.called

    def test_answer_em_with_llm_uses_standard_em(self):
        """有 LLM 时 answer_em 走标准 EM (reader span == gold)."""
        e, eb = self._fake_embed()

        class ExactLLM:
            def extract_answer(self, q, c):
                return "the United States"  # 归一化后 == "United States"

        reader = MultiHopReader(e, eb, llm_reader=ExactLLM())
        assert reader.answer_em("q", "ctx", "United States")  # 去 the 后相等
        assert not reader.answer_em("q", "ctx", "France")
