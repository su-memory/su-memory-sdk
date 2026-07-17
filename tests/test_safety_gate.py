"""
C3 风险门控检索测试

验证：recall 结果经风险门控，禁忌零泄露、风险标记全覆盖。
医疗级安全要求：含药物交互/过敏冲突的记忆被标记或拦截。
"""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("SU_MEMORY_SKIP_ENV_CHECK", "1")
os.environ.setdefault("MEMORY_EMBEDDING_BACKEND", "none")


@pytest.fixture
def client(tmp_path):
    from su_memory.clinical import ClinicalMemoryClient
    return ClinicalMemoryClient(
        storage_path=str(tmp_path / "safety_test"),
        embedding_backend="none",
        compliance_level="mask",
        safety_screen=True,
        safety_policy="mark",
    )


@pytest.fixture
def block_client(tmp_path):
    from su_memory.clinical import ClinicalMemoryClient
    return ClinicalMemoryClient(
        storage_path=str(tmp_path / "safety_block"),
        embedding_backend="none",
        compliance_level="mask",
        safety_screen=True,
        safety_policy="block",
    )


class TestSafetyGateDirect:
    """直接测 SafetyGate"""

    def test_safe_memory_passthrough(self):
        """无药物/过敏的记忆应 safe 放行"""
        from su_memory.clinical import MedicalKnowledgeBase, SafetyGate
        gate = SafetyGate(MedicalKnowledgeBase())
        results = [{"memory_id": "m1", "content": "普通营养建议"}]
        screened = gate.screen(results)
        assert len(screened) == 1
        assert screened[0]["risk_level"] == "safe"
        assert screened[0]["risk_flags"] == []

    def test_major_drug_interaction_contraindicated(self):
        """重大药物交互应标 contraindicated"""
        from su_memory.clinical import MedicalKnowledgeBase, SafetyGate
        gate = SafetyGate(MedicalKnowledgeBase())
        results = [{"memory_id": "m1", "content": "华法林抗凝，建议多食菠菜"}]
        screened = gate.screen(results)
        assert screened[0]["risk_level"] == "contraindicated"
        assert any("华法林" in f for f in screened[0]["risk_flags"])

    def test_moderate_interaction_caution(self):
        """中度交互应标 caution"""
        from su_memory.clinical import MedicalKnowledgeBase, SafetyGate
        gate = SafetyGate(MedicalKnowledgeBase())
        results = [{"memory_id": "m1", "content": "二甲双胍治疗中"}]
        screened = gate.screen(results)
        assert screened[0]["risk_level"] == "caution"

    def test_allergy_conflict_detected(self):
        """患者过敏原匹配记忆内容应 contraindicated"""
        from su_memory.clinical import MedicalKnowledgeBase, SafetyGate
        gate = SafetyGate(MedicalKnowledgeBase())
        results = [{"memory_id": "m1", "content": "建议补充花生蛋白"}]
        screened = gate.screen(results, patient_allergies=["花生"])
        assert screened[0]["risk_level"] == "contraindicated"
        assert any("花生" in f for f in screened[0]["risk_flags"])

    def test_block_policy_filters_contraindicated(self):
        """block 策略应移除 contraindicated"""
        from su_memory.clinical import MedicalKnowledgeBase, SafetyGate
        gate = SafetyGate(MedicalKnowledgeBase(), policy="block")
        results = [
            {"memory_id": "m1", "content": "华法林配菠菜"},
            {"memory_id": "m2", "content": "普通建议"},
        ]
        screened = gate.screen(results)
        assert len(screened) == 1
        assert screened[0]["memory_id"] == "m2"


class TestRecallIntegration:
    """recall 集成测试"""

    def test_recall_marks_drug_interaction(self, client):
        """recall 应自动标记药物交互风险"""
        client.add_patient_event("P001", "华法林抗凝医嘱，建议多食深色蔬菜", "order")
        hits = client.recall("P001", "华法林")
        assert len(hits) >= 1
        assert hits[0]["risk_level"] == "contraindicated"
        assert len(hits[0]["risk_flags"]) >= 1

    def test_recall_safe_memory_no_flags(self, client):
        """安全记忆应无风险标记"""
        client.add_patient_event("P001", "高蛋白营养方案1800kcal", "plan")
        hits = client.recall("P001", "营养方案")
        assert len(hits) >= 1
        assert hits[0]["risk_level"] == "safe"

    def test_recall_block_policy_filters(self, block_client):
        """block 策略 recall 应过滤禁忌"""
        block_client.add_patient_event("P001", "华法林配菠菜方案", "order")
        block_client.add_patient_event("P001", "普通营养建议", "plan")
        hits = block_client.recall("P001", "方案", top_k=5, max_fetch=500)
        # 禁忌记忆被拦截
        assert all(h["risk_level"] != "contraindicated" for h in hits)

    def test_safety_screen_disabled(self, tmp_path):
        """safety_screen=False 时无风险字段"""
        from su_memory.clinical import ClinicalMemoryClient
        c = ClinicalMemoryClient(
            storage_path=str(tmp_path / "no_gate"),
            embedding_backend="none",
            safety_screen=False,
        )
        c.add_patient_event("P001", "华法林抗凝", "order")
        hits = c.recall("P001", "华法林")
        assert len(hits) >= 1
        assert "risk_level" not in hits[0]


class TestStats:
    """门控器状态"""

    def test_stats(self):
        from su_memory.clinical import MedicalKnowledgeBase, SafetyGate
        gate = SafetyGate(MedicalKnowledgeBase())
        s = gate.stats()
        assert s["drug_names_tracked"] >= 8
        assert s["allergens_tracked"] >= 6
        assert s["policy"] == "mark"
