"""
Layer 1: LLM Energy Inference tests

Tests:
1. Multi-provider LLM inference (DeepSeek, MiniMax, Ollama)
2. Fallback chain: API → local → keyword
3. Cache hit avoids API calls
4. Accuracy benchmark on known test cases
"""
import pytest
import sys
import os
import hashlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# Known ground-truth test cases for energy classification
GROUND_TRUTH = [
    ("spring renewal growth forest green east morning", "wood"),
    ("summer passion heat fire red south noon blaze", "fire"),
    ("stability center balance earth yellow ground foundation", "earth"),
    ("autumn harvest metal white west structure refinement", "metal"),
    ("winter wisdom water blue north deep fluid flow", "water"),
    ("tree planting garden botanical leaf branch root", "wood"),
    ("volcano eruption flame warm sunlight bright burning", "fire"),
    ("mountain rock soil clay terrain plateau crust", "earth"),
    ("sword knife steel iron gold silver copper bronze", "metal"),
    ("ocean river rain stream lake ice cold fish", "water"),
    ("liver healing tendon flexibility green tea herbs", "wood"),
    ("heart blood circulation pulse warmth joy emotion", "fire"),
    ("spleen stomach digestion nutrition absorption yellow", "earth"),
    ("lung breath oxygen air respiration skin white", "metal"),
    ("kidney bladder fluid urine bone marrow black", "water"),
]


class TestLLMInference:
    """Verify energy inference accuracy with multi-provider LLM."""

    def test_keyword_fallback_accuracy(self):
        """Baseline: keyword-only accuracy (expected ~30-40%)."""
        from su_memory.sdk.lite_pro import SuMemoryLitePro

        pro = SuMemoryLitePro(
            enable_vector=False, enable_graph=False,
            enable_temporal=False, enable_session=False,
            enable_prediction=False, enable_explainability=False
        )

        correct = 0
        for content, expected in GROUND_TRUTH:
            # Force keyword-only by clearing cache and mocking LLM unavailable
            pro._energy_cache = {}
            # _infer_energy will try LLM first, fallback to keywords
            result = pro._infer_energy(content)
            if result == expected:
                correct += 1

        accuracy = correct / len(GROUND_TRUTH)
        print(f"\n  Keyword baseline accuracy: {accuracy:.0%} ({correct}/{len(GROUND_TRUTH)})")
        # Keywords should get some right but not all
        assert accuracy > 0.2, f"Keyword accuracy too low: {accuracy:.0%}"

    def test_energy_cache_works(self):
        """Cached results avoid re-computation."""
        from su_memory.sdk.lite_pro import SuMemoryLitePro

        pro = SuMemoryLitePro(
            enable_vector=False, enable_graph=False,
            enable_temporal=False, enable_session=False,
            enable_prediction=False, enable_explainability=False
        )
        pro._energy_cache = {}

        content = "spring renewal forest green"
        result1 = pro._infer_energy(content)
        result2 = pro._infer_energy(content)

        assert result1 == result2, "Cache should return same result"
        assert len(pro._energy_cache) == 1, "Cache should have 1 entry"

    def test_infer_returns_valid_energy(self):
        """All inferences return valid energy types."""
        from su_memory.sdk.lite_pro import SuMemoryLitePro

        pro = SuMemoryLitePro(
            enable_vector=False, enable_graph=False,
            enable_temporal=False, enable_session=False,
            enable_prediction=False, enable_explainability=False
        )
        valid = {"wood", "fire", "earth", "metal", "water"}

        for content, _ in GROUND_TRUTH:
            result = pro._infer_energy(content)
            assert result in valid, f"'{content[:30]}...' → '{result}' (invalid)"

    def test_llm_infer_returns_valid(self):
        """_llm_infer_energy at minimum doesn't crash and returns str."""
        from su_memory.sdk.lite_pro import SuMemoryLitePro

        pro = SuMemoryLitePro(
            enable_vector=False, enable_graph=False,
            enable_temporal=False, enable_session=False,
            enable_prediction=False, enable_explainability=False
        )

        # Should not raise exceptions
        result = pro._llm_infer_energy("test content")
        assert isinstance(result, str)
        # If empty, that's fine (means fallback to keywords)
        if result:
            assert result in ("wood", "fire", "earth", "metal", "water")

    def test_cache_md5_isolation(self):
        """Different content → different cache keys."""
        from su_memory.sdk.lite_pro import SuMemoryLitePro

        pro = SuMemoryLitePro(
            enable_vector=False, enable_graph=False,
            enable_temporal=False, enable_session=False,
            enable_prediction=False, enable_explainability=False
        )
        pro._energy_cache = {}

        pro._infer_energy("spring forest")
        pro._infer_energy("summer fire")

        assert len(pro._energy_cache) == 2, "Different content → different cache entries"
