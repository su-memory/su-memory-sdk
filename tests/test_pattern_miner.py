"""
临床模式提炼测试 — P1-S4 验证

从患者记忆聚类中发现临床决策模式。
"""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("SU_MEMORY_SKIP_ENV_CHECK", "1")
os.environ.setdefault("MEMORY_EMBEDDING_BACKEND", "none")


@pytest.fixture
def seeded_client(tmp_path):
    """预填充临床记忆的客户端"""
    from su_memory.sdk.lite_pro import SuMemoryLitePro
    from su_memory.clinical import MedicalAssociationKB

    client = SuMemoryLitePro(
        storage_path=str(tmp_path / "pattern_test"),
        embedding_backend="none",
        enable_llm_energy=False,
    )
    # 写入两类临床记忆
    for i in range(5):
        client.add(f"糖尿病肾病患者适合低蛋白饮食方案变体{i}")
    for i in range(4):
        client.add(f"花生过敏患者禁用含花生蛋白制剂变体{i}")
    return client


class TestPatternMining:
    """模式提炼测试"""

    def test_mine_finds_patterns(self, seeded_client):
        """能从记忆中提炼出模式"""
        from su_memory.clinical import ClinicalPatternMiner
        miner = ClinicalPatternMiner(seeded_client)
        patterns = miner.mine_patterns(min_support=3)
        assert len(patterns) >= 1

    def test_pattern_has_support(self, seeded_client):
        """模式包含支持度"""
        from su_memory.clinical import ClinicalPatternMiner
        miner = ClinicalPatternMiner(seeded_client)
        patterns = miner.mine_patterns(min_support=3)
        for p in patterns:
            assert p.support >= 3

    def test_pattern_with_kb_recognizes_entities(self, seeded_client):
        """带知识库时识别医疗实体"""
        from su_memory.clinical import ClinicalPatternMiner, MedicalAssociationKB
        kb = MedicalAssociationKB()
        miner = ClinicalPatternMiner(seeded_client, kb)
        patterns = miner.mine_patterns(min_support=3)
        # 至少一个模式有关联规则
        has_rule = any(p.associated_rules for p in patterns)
        assert has_rule, "知识库未识别到医疗实体"


class TestPatternToRules:
    """模式转规则测试"""

    def test_patterns_to_rules(self, seeded_client):
        """模式转化为可执行规则"""
        from su_memory.clinical import ClinicalPatternMiner
        miner = ClinicalPatternMiner(seeded_client)
        patterns = miner.mine_patterns(min_support=3)
        rules = miner.patterns_to_rules(patterns)
        assert len(rules) >= 1
        assert "if" in rules[0]
        assert "then" in rules[0]

    def test_rule_has_confidence(self, seeded_client):
        """规则包含置信度"""
        from su_memory.clinical import ClinicalPatternMiner
        miner = ClinicalPatternMiner(seeded_client)
        patterns = miner.mine_patterns(min_support=3)
        rules = miner.patterns_to_rules(patterns)
        for r in rules:
            assert 0 <= r["confidence"] <= 1


class TestEdgeCases:
    """边界情况测试"""

    def test_empty_memory(self, tmp_path):
        """空记忆库返回空"""
        from su_memory.sdk.lite_pro import SuMemoryLitePro
        from su_memory.clinical import ClinicalPatternMiner
        client = SuMemoryLitePro(
            storage_path=str(tmp_path / "empty"),
            embedding_backend="none",
            enable_llm_energy=False,
        )
        miner = ClinicalPatternMiner(client)
        patterns = miner.mine_patterns()
        assert patterns == []

    def test_insufficient_support(self, tmp_path):
        """记忆太少不形成模式"""
        from su_memory.sdk.lite_pro import SuMemoryLitePro
        from su_memory.clinical import ClinicalPatternMiner
        client = SuMemoryLitePro(
            storage_path=str(tmp_path / "sparse"),
            embedding_backend="none",
            enable_llm_energy=False,
        )
        client.add("单条记忆不足以形成模式")
        miner = ClinicalPatternMiner(client)
        patterns = miner.mine_patterns(min_support=3)
        assert patterns == []
