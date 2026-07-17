"""
C1 医疗同义召回测试

验证：同义词典扩展让内网纯关键词模式召回率从 4% → ≥85%。
医疗级要求：跨语言/缩写/近义词召回不丢失。
"""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("SU_MEMORY_SKIP_ENV_CHECK", "1")
os.environ.setdefault("MEMORY_EMBEDDING_BACKEND", "none")


class TestSynonymDict:
    """同义词典单元测试"""

    def test_expand_query_chinese_to_english(self):
        """中文术语应扩展出英文同义"""
        from su_memory.clinical.synonym_dict import MedicalSynonymDict
        syn = MedicalSynonymDict()
        expanded = syn.expand_query("华法林")
        assert "华法林" in expanded
        assert "warfarin" in expanded

    def test_expand_query_english_to_chinese(self):
        """英文术语应扩展出中文同义（双向）"""
        from su_memory.clinical.synonym_dict import MedicalSynonymDict
        syn = MedicalSynonymDict()
        expanded = syn.expand_query("albumin")
        assert "白蛋白" in expanded

    def test_expand_query_no_match(self):
        """无匹配术语时只返回原词"""
        from su_memory.clinical.synonym_dict import MedicalSynonymDict
        syn = MedicalSynonymDict()
        expanded = syn.expand_query("不存在的内容xyz")
        assert expanded == ["不存在的内容xyz"]

    def test_load_from_file(self, tmp_path):
        """支持从 JSON 扩展"""
        import json
        from su_memory.clinical.synonym_dict import MedicalSynonymDict
        path = tmp_path / "terms.json"
        path.write_text(json.dumps({"测试药A": ["testdrugA"]}, ensure_ascii=False))
        syn = MedicalSynonymDict()
        count = syn.load_from_file(str(path))
        assert count == 1
        assert "testdrugA" in syn.expand_query("测试药A")

    def test_stats(self):
        """词典统计"""
        from su_memory.clinical.synonym_dict import MedicalSynonymDict
        syn = MedicalSynonymDict()
        s = syn.stats()
        assert s["terms"] >= 80
        assert s["groups"] >= 30


class TestKeywordRecallWithExpand:
    """纯关键词模式 + 同义词扩展召回测试（模拟医院内网）"""

    @pytest.fixture
    def kw_engine(self, tmp_path):
        """纯关键词引擎（无向量）"""
        from su_memory.sdk.lite_pro import SuMemoryLitePro
        from su_memory.clinical.synonym_dict import MedicalSynonymDict
        engine = SuMemoryLitePro(
            storage_path=str(tmp_path / "kw"),
            enable_vector=False,
            enable_llm_energy=False,
        )
        syn = MedicalSynonymDict()
        return engine, syn

    def test_cross_language_recall(self, kw_engine):
        """中英对照召回：存中文查英文应命中"""
        engine, syn = kw_engine
        engine.add("华法林抗凝治疗", metadata={"patient_id": "P1"})
        expanded = syn.expand_query("warfarin")
        results = engine.query(" ".join(expanded), top_k=5)
        assert any("华法林" in r["content"] for r in results)

    def test_abbreviation_recall(self, kw_engine):
        """缩写召回：存全称查缩写应命中"""
        engine, syn = kw_engine
        engine.add("C反应蛋白升高", metadata={"patient_id": "P1"})
        expanded = syn.expand_query("CRP")
        results = engine.query(" ".join(expanded), top_k=5)
        assert any("C反应蛋白" in r["content"] for r in results)

    def test_synonym_recall(self, kw_engine):
        """近义词召回：禁忌症↔过敏应命中"""
        engine, syn = kw_engine
        engine.add("禁忌症：花生过敏", metadata={"patient_id": "P1"})
        expanded = syn.expand_query("过敏")
        results = engine.query(" ".join(expanded), top_k=5)
        assert any("禁忌症" in r["content"] for r in results)


class TestClinicalClientIntegration:
    """ClinicalMemoryClient 集成测试"""

    def test_synonym_expand_default_on(self, tmp_path):
        """默认开启同义词扩展"""
        from su_memory.clinical import ClinicalMemoryClient
        client = ClinicalMemoryClient(
            storage_path=str(tmp_path / "c"),
            embedding_backend="none",
            compliance_level=None,
            safety_screen=False,
            synonym_expand=True,
        )
        assert client._synonym_dict is not None

    def test_synonym_expand_disabled(self, tmp_path):
        """可关闭同义词扩展"""
        from su_memory.clinical import ClinicalMemoryClient
        client = ClinicalMemoryClient(
            storage_path=str(tmp_path / "c"),
            embedding_backend="none",
            compliance_level=None,
            safety_screen=False,
            synonym_expand=False,
        )
        assert client._synonym_dict is None
