"""
领域知识层测试 — P2-S2 验证

药物交互查询、检验参考值、过敏禁忌映射。
"""
from __future__ import annotations

import pytest


class TestDrugInteraction:
    """药物-营养交互查询测试"""

    def test_check_single_drug(self):
        from su_memory.clinical import MedicalKnowledgeBase
        kb = MedicalKnowledgeBase()
        results = kb.check_drug_interaction(["华法林"])
        assert len(results) >= 1
        assert any(r.nutrient == "维生素K" for r in results)

    def test_check_multiple_drugs(self):
        from su_memory.clinical import MedicalKnowledgeBase
        kb = MedicalKnowledgeBase()
        results = kb.check_drug_interaction(["华法林", "二甲双胍"])
        assert len(results) >= 2

    def test_get_contraindicated_nutrients(self):
        from su_memory.clinical import MedicalKnowledgeBase
        kb = MedicalKnowledgeBase()
        nutrients = kb.get_contraindicated_nutrients(["华法林"])
        assert "维生素K" in nutrients.get("华法林", [])

    def test_unknown_drug(self):
        from su_memory.clinical import MedicalKnowledgeBase
        kb = MedicalKnowledgeBase()
        results = kb.check_drug_interaction(["不存在的药物"])
        assert len(results) == 0


class TestLabReference:
    """检验参考值测试"""

    def test_get_reference(self):
        from su_memory.clinical import MedicalKnowledgeBase
        kb = MedicalKnowledgeBase()
        ref = kb.get_lab_reference("白蛋白")
        assert ref is not None
        assert ref.unit == "g/L"

    def test_is_abnormal_low(self):
        from su_memory.clinical import MedicalKnowledgeBase
        kb = MedicalKnowledgeBase()
        assert kb.is_abnormal("白蛋白", 28.0) is True

    def test_is_abnormal_normal(self):
        from su_memory.clinical import MedicalKnowledgeBase
        kb = MedicalKnowledgeBase()
        assert kb.is_abnormal("白蛋白", 40.0) is False

    def test_is_critical(self):
        from su_memory.clinical import MedicalKnowledgeBase
        kb = MedicalKnowledgeBase()
        assert kb.is_critical("钾", 7.0) is True
        assert kb.is_critical("钾", 4.0) is False

    def test_find_abnormal_from_dict(self):
        from su_memory.clinical import MedicalKnowledgeBase
        kb = MedicalKnowledgeBase()
        abnormal = kb.find_abnormal_from_dict({
            "白蛋白": 28.0,   # 低
            "钾": 4.0,       # 正常
            "血红蛋白": 100,  # 低
        })
        names = [a["name"] for a in abnormal]
        assert "白蛋白" in names
        assert "血红蛋白" in names
        assert "钾" not in names


class TestAllergy:
    """过敏-禁忌测试"""

    def test_check_allergy(self):
        from su_memory.clinical import MedicalKnowledgeBase
        kb = MedicalKnowledgeBase()
        entry = kb.check_allergy("花生")
        assert entry is not None
        assert "花生蛋白" in entry.contraindicated_substances

    def test_check_substance_allergy(self):
        from su_memory.clinical import MedicalKnowledgeBase
        kb = MedicalKnowledgeBase()
        results = kb.check_substance_allergy(["花生油", "大豆蛋白"])
        allergens = [r.allergen for r in results]
        assert "花生" in allergens
        assert "大豆" in allergens

    def test_unknown_allergy(self):
        from su_memory.clinical import MedicalKnowledgeBase
        kb = MedicalKnowledgeBase()
        assert kb.check_allergy("不存在的过敏原") is None


class TestStats:
    """知识库统计"""

    def test_stats(self):
        from su_memory.clinical import MedicalKnowledgeBase
        kb = MedicalKnowledgeBase()
        stats = kb.stats()
        assert stats["drug_interactions"] >= 8
        assert stats["lab_references"] >= 10
        assert stats["allergies"] >= 6
