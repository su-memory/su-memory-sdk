"""Regression test: SuMemoryLite.query must not return duplicate content.

Reproduces the bug found during the A/B causal experiment where the warm-tier
fallback appended duplicate entries (same content, different memory_id, with
the warm copy carrying score=0.1). The fix deduplicates by content at both
the hot-tier result construction and the warm-tier fallback, plus a defensive
content-level dedup in TieredStorage.query_warm.
"""
import tempfile

from su_memory.sdk.lite import SuMemoryLite


def _fresh_store() -> SuMemoryLite:
    tmp = tempfile.mkdtemp()
    return SuMemoryLite(storage_path=tmp, enable_persistence=True)


class TestQueryDedup:
    def test_no_duplicate_content_in_results(self):
        m = _fresh_store()
        for d in ["由于持续暴雨", "因此水库水位暴涨", "持续暴雨是常见现象"]:
            m.add(d)
        res = m.query("持续暴雨", top_k=10)
        contents = [r["content"] for r in res]
        assert len(contents) == len(set(contents)), f"duplicate contents: {contents}"

    def test_no_duplicate_memory_id(self):
        m = _fresh_store()
        for d in ["alpha", "beta", "gamma"]:
            m.add(d)
        res = m.query("alpha", top_k=10)
        ids = [r["memory_id"] for r in res]
        assert len(ids) == len(set(ids))

    def test_warm_fallback_does_not_reintroduce_hot_content(self):
        # When hot results are fewer than top_k, warm fallback must not append
        # a duplicate of an already-returned hot content.
        m = _fresh_store()
        for d in ["uniqueA 暴雨", "uniqueB 暴雨"]:
            m.add(d)
        res = m.query("暴雨", top_k=10)
        contents = [r["content"] for r in res]
        assert len(contents) == len(set(contents))

    def test_repeated_add_same_content_no_duplicates(self):
        m = _fresh_store()
        for _ in range(5):
            m.add("same content keyword")
        res = m.query("keyword", top_k=10)
        contents = [r["content"] for r in res]
        assert len(contents) == len(set(contents))

    def test_consecutive_queries_stable(self):
        # The original symptom: consecutive queries on the same instance.
        m = _fresh_store()
        docs = ["持续暴雨", "水库泄洪", "大坝决堤", "村庄被淹", "暴雨预警"]
        for d in docs:
            m.add(d)
        for q in ["暴雨", "大坝", "村庄", "水库"]:
            res = m.query(q, top_k=5)
            contents = [r["content"] for r in res]
            assert len(contents) == len(set(contents)), f"dup for q={q}: {contents}"

    def test_results_count_does_not_exceed_top_k_after_dedup(self):
        m = _fresh_store()
        for d in ["x1", "x2", "x3"]:
            m.add(d)
        res = m.query("x", top_k=2)
        assert len(res) <= 2
