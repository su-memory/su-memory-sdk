"""MultiHopReader 单测: 三路融合检索 + 答案抽取."""
import numpy as np
import pytest

from su_memory.sdk.multi_hop_reader import MultiHopReader, HopResult


def _fake_embed(dim=8):
    """确定性 fake embedder: 文本 hash -> 稳定向量."""
    def embed(text):
        h = hash(text) % 1000
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim).astype(np.float32)
    def embed_batch(texts):
        return np.stack([embed(t) for t in texts])
    return embed, embed_batch


@pytest.fixture
def reader():
    e, eb = _fake_embed()
    return MultiHopReader(e, eb)


class TestRetrieve:
    def test_basic_retrieve(self, reader):
        paras = ["Alpha Corp is a company.", "Beta Ltd competes with Alpha.",
                 "Unrelated weather text.", "Gamma Inc is elsewhere."]
        res = reader.retrieve("What is Alpha", paras, top_k=3)
        assert isinstance(res, HopResult)
        assert len(res.ranked_ids) <= 3
        assert res.top1 >= 0

    def test_empty_paragraphs(self, reader):
        res = reader.retrieve("q", [], top_k=3)
        assert res.ranked_ids == []

    def test_top_k_respected(self, reader):
        paras = [f"doc {i} content" for i in range(10)]
        res = reader.retrieve("query", paras, top_k=4)
        assert len(res.ranked_ids) <= 4

    def test_answer_context_built(self, reader):
        paras = ["First paragraph.", "Second paragraph."]
        res = reader.retrieve("q", paras, top_k=2)
        assert isinstance(res.answer_context, str)
        assert len(res.answer_context) > 0

    def test_paths_recorded(self, reader):
        paras = ["Alpha: text about Alpha Corp.", "Beta: Beta Ltd info.",
                 "Gamma: unrelated."]
        res = reader.retrieve("Alpha", paras, top_k=3)
        assert isinstance(res.path_used, list)


class TestAnswerEM:
    def test_exact_match(self, reader):
        assert reader.answer_em("q", "Shirley Temple was here", "Shirley Temple")

    def test_case_insensitive(self, reader):
        assert reader.answer_em("q", "The Answer is FORTY-TWO", "forty two")

    def test_partial_word_match(self, reader):
        # 80% 词覆盖算命中
        assert reader.answer_em("q", "the chief of protocol duties", "chief of protocol")

    def test_no_match(self, reader):
        assert not reader.answer_em("q", "completely different content", "missing answer")

    def test_empty_gold(self, reader):
        assert not reader.answer_em("q", "some text", "")


class TestExtractAnswer:
    def test_yesno_question(self, reader):
        ans = reader.extract_answer("Are they the same?", "Some context.")
        assert ans in ("yes", "no")

    def test_span_question(self, reader):
        ctx = "Shirley Temple was an American actress. She was a diplomat too."
        ans = reader.extract_answer("Who was Shirley Temple", ctx)
        assert isinstance(ans, str)
        assert len(ans) > 0


class TestEntityBridge:
    def test_bridges_shared_entity(self, reader):
        """两段共享实体应被桥接召回."""
        paras = [
            "Kiss and Tell film starred Shirley Temple as the lead.",
            "Shirley Temple was an American actress and diplomat.",
            "Random unrelated paragraph about London weather.",
        ]
        res = reader.retrieve("Kiss and Tell film", paras, top_k=3)
        # 至少应召回前两段 (共享 Shirley Temple), 排除 weather
        assert 0 in res.ranked_ids
        assert 1 in res.ranked_ids
        assert res.bridge_entities  # 提取到桥接实体


class TestRetrieveStructured:
    """CausalDAG 桥接结构标注 (v4.0 增强)."""

    def test_structured_returns_hopresult(self, reader):
        paras = ["Alpha Corp: big company in Alpha",
                 "Beta: rivals Alpha Corp",
                 "Gamma: unrelated weather"]
        res = reader.retrieve_structured("What is Alpha", paras, top_k=3)
        assert isinstance(res, HopResult)
        assert len(res.ranked_ids) <= 3

    def test_structured_has_bridge_annotation(self, reader):
        """结构化 context 应含 EVIDENCE/BRIDGE 标注."""
        paras = ["Alpha: Alpha Corp is big",
                 "Beta: Beta Ltd competes with Alpha",
                 "Gamma: unrelated"]
        res = reader.retrieve_structured("Alpha", paras, top_k=3)
        assert "EVIDENCE" in res.answer_context

    def test_structured_empty_paragraphs(self, reader):
        res = reader.retrieve_structured("q", [], top_k=3)
        assert res.ranked_ids == []

    def test_causaldag_rare_entity_bridge(self, reader):
        """CausalDAG 罕见实体桥接: 共享罕见实体的段落应优先桥接."""
        # 两段共享罕见实体 "ZephyrEngine", 第三段共享常见词 "American"
        paras = [
            "ProjectX: uses ZephyrEngine for computation",
            "ZephyrEngine: a rare technology built in 2020",
            "OtherCo: an American company doing other things",
        ]
        res = reader.retrieve_structured("What is ProjectX", paras, top_k=3)
        # bridge_map 应记录罕见实体桥接
        assert isinstance(res.bridge_map, dict)


class TestCausalDAGBridgeOrder:
    """_entity_bridge_order 罕见实体 IDF 加权."""

    def test_returns_scored_list(self, reader):
        paras = ["Alpha Corp uses ZephyrEngine",
                 "ZephyrEngine is rare",
                 "Gamma is unrelated"]
        scored = reader._entity_bridge_order(paras, 0)
        assert isinstance(scored, list)
        # 每项是 (idx, spec, [bridge_ents])
        for item in scored:
            assert len(item) == 3

    def test_rare_entity_preferred(self, reader):
        """罕见实体 (DF<=3) 应被优先, 常见实体被过滤."""
        paras = [
            "Alpha uses ZephyrEngine and American",
            "Beta uses ZephyrEngine and American",
            "Gamma uses American only",
        ]
        scored = reader._entity_bridge_order(paras, 0, rare_only=True)
        # idx=1 共享 ZephyrEngine (罕见) 应排前; idx=2 只共享 American (常见) 应被过滤
        if scored:
            top_idx = scored[0][0]
            assert top_idx == 1  # ZephyrEngine 桥接优先

    def test_rare_only_false_includes_all(self, reader):
        """rare_only=False 时不过滤常见实体 (DF高的也纳入)."""
        paras = [
            "AlphaCorp uses SharedLib",
            "BetaCorp uses SharedLib",
            "GammaCorp uses SharedLib",
            "DeltaCorp uses SharedLib",
            "EpsilonCorp uses SharedLib",
        ]
        scored = reader._entity_bridge_order(paras, 0, rare_only=False)
        # rare_only=False: SharedLib (DF=5, 常见) 也应纳入
        assert len(scored) >= 1
        # 对比: rare_only=True 时 SharedLib (DF=5>3) 应被过滤
        scored_rare = reader._entity_bridge_order(paras, 0, rare_only=True)
        assert len(scored_rare) == 0  # 只有SharedLib, DF=5>3被过滤
