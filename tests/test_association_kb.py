"""
医疗关联知识库测试 — P1-S1 验证

验证关联规则匹配、自动注入、多跳检索增强。
严格区分：这是检索关联增强测试，不是因果推断测试。
"""
from __future__ import annotations

import os
import tempfile

import pytest

os.environ.setdefault("SU_MEMORY_SKIP_ENV_CHECK", "1")
os.environ.setdefault("MEMORY_EMBEDDING_BACKEND", "none")


@pytest.fixture
def memory_client(tmp_path):
    """带医疗关联注入的记忆客户端"""
    from su_memory.sdk.lite_pro import SuMemoryLitePro
    from su_memory.clinical import MedicalAssociationKB

    client = SuMemoryLitePro(
        storage_path=str(tmp_path / "med_test"),
        embedding_backend="none",
        enable_llm_energy=False,
    )
    kb = MedicalAssociationKB()
    kb.inject_hooks(client)
    yield client, kb


class TestAssociationMatching:
    """关联规则匹配测试"""

    def test_drug_nutrient_match(self, memory_client):
        """药物-营养交互规则匹配"""
        _, kb = memory_client
        matches = kb.match("患者服用华法林")
        assert len(matches) > 0
        rule_ids = [r.rule_id for r, _ in matches]
        assert "dn_warfarin" in rule_ids

    def test_deficiency_symptom_match(self, memory_client):
        """营养缺乏-症状规则匹配"""
        _, kb = memory_client
        matches = kb.match("患者白蛋白偏低")
        assert any(r.rule_id == "ds_protein" for r, _ in matches)

    def test_allergy_match(self, memory_client):
        """过敏-禁忌规则匹配"""
        _, kb = memory_client
        matches = kb.match("花生过敏史")
        assert any(r.rule_id == "ac_peanut" for r, _ in matches)

    def test_no_false_match(self, memory_client):
        """无关内容不误匹配"""
        _, kb = memory_client
        matches = kb.match("今天天气很好，适合散步")
        assert len(matches) == 0


class TestAutoInjection:
    """自动注入关联边测试"""

    def test_warfarin_vitk_link_created(self, memory_client):
        """华法林记忆与维K记忆自动建立关联边"""
        client, _ = memory_client
        mid1 = client.add("患者服用华法林进行抗凝治疗")
        mid2 = client.add("建议限制深色蔬菜摄入，因富含维生素K")

        edge = client._graph._causal_edges.get((mid1, mid2))
        assert edge is not None, "华法林→维K 关联边未创建"
        assert edge.confidence == 0.9
        assert edge.evidence_type == "explicit"

    def test_allergy_contraindication_link(self, memory_client):
        """过敏记忆与禁忌制剂记忆自动建立关联（置信度更高）"""
        client, _ = memory_client
        mid1 = client.add("患者有花生过敏史")
        mid2 = client.add("该肠内营养制剂含花生蛋白成分")

        edge = client._graph._causal_edges.get((mid1, mid2))
        assert edge is not None
        assert edge.confidence == 0.95  # 过敏-禁忌更高置信度

    def test_no_link_for_unrelated(self, memory_client):
        """无关记忆不创建关联边"""
        client, _ = memory_client
        client.add("患者喜欢散步")
        client.add("今天出院")

        assert len(client._graph._causal_edges) == 0

    def test_multihop_reaches_linked_memory(self, memory_client):
        """多跳查询能通过关联边到达相关记忆"""
        client, _ = memory_client
        client.add("患者服用甲氨蝶呤治疗")
        client.add("需补充叶酸预防副作用")

        results = client.query_multihop("甲氨蝶呤", top_k=5)
        contents = [r.get("content", "") for r in results]
        assert any("叶酸" in c for c in contents), "多跳查询未通过关联边到达叶酸记忆"


class TestExplainAssociation:
    """关联解释测试"""

    def test_explain_warfarin_vitk(self, memory_client):
        """能解释华法林与维K的关联"""
        _, kb = memory_client
        desc = kb.explain_association(
            "患者服用华法林",
            "限制深色蔬菜因含维生素K",
        )
        assert desc is not None
        assert "华法林" in desc or "维生素K" in desc

    def test_explain_no_association(self, memory_client):
        """无关联时返回 None"""
        _, kb = memory_client
        desc = kb.explain_association(
            "今天天气很好",
            "患者出院了",
        )
        assert desc is None


class TestCustomRules:
    """自定义规则测试"""

    def test_add_custom_rule(self, memory_client):
        """添加自定义关联规则"""
        from su_memory.clinical import AssociationRule, AssociationType

        _, kb = memory_client
        custom = AssociationRule(
            rule_id="custom_test",
            assoc_type=AssociationType.DRUG_NUTRIENT,
            source_patterns=["测试药物A"],
            target_patterns=["测试营养素B"],
            relation_desc="自定义测试关联",
        )
        kb.add_rule(custom)
        assert any(r.rule_id == "custom_test" for r in kb.rules)
