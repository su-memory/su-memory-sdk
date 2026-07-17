"""
C6 版本化冲突消解测试

验证：同一事实多次更新建立版本链，可回溯历史，旧版本不丢失。
医疗级要求：诊疗方案变更可追溯到每一版。
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
        storage_path=str(tmp_path / "ver"),
        embedding_backend="none",
        compliance_level=None,
        safety_screen=False,
    )


class TestVersionChain:
    """版本链测试"""

    def test_first_update_creates_v1(self, client):
        """首次更新创建 v1"""
        client.update_clinical_fact(
            "P001", "nutrition_plan", "高蛋白方案1800kcal"
        )
        active = client.get_active_fact("P001", "nutrition_plan")
        assert active is not None
        assert active["version"] == 1
        assert active["active"] is True
        assert "高蛋白方案1800kcal" in active["content"]

    def test_second_update_creates_v2_links_v1(self, client):
        """第二次更新创建 v2，v1 被 superseded"""
        client.update_clinical_fact("P001", "nutrition_plan", "方案v1")
        client.update_clinical_fact("P001", "nutrition_plan", "方案v2")

        active = client.get_active_fact("P001", "nutrition_plan")
        assert active["version"] == 2
        assert active["active"] is True

        history = client.get_fact_history("P001", "nutrition_plan")
        assert len(history) == 2
        # v1 在前，v2 在后
        assert history[0]["version"] == 1
        assert history[1]["version"] == 2
        # v1 不活跃，v2 活跃
        assert history[0]["active"] is False
        assert history[1]["active"] is True

    def test_three_versions_full_chain(self, client):
        """三版本完整链条可回溯"""
        client.update_clinical_fact("P001", "diagnosis", "糖尿病")
        client.update_clinical_fact("P001", "diagnosis", "糖尿病+营养不良")
        client.update_clinical_fact("P001", "diagnosis", "糖尿病+营养不良+贫血")

        history = client.get_fact_history("P001", "diagnosis")
        assert len(history) == 3
        versions = [h["version"] for h in history]
        assert versions == [1, 2, 3]
        # 最新版是活跃的
        assert history[-1]["active"] is True
        # 旧版都不活跃
        assert all(not h["active"] for h in history[:-1])

    def test_old_version_preserved(self, client):
        """旧版本内容不被覆盖"""
        client.update_clinical_fact("P001", "plan", "原始方案")
        client.update_clinical_fact("P001", "plan", "更新方案")

        history = client.get_fact_history("P001", "plan")
        # v1 的内容仍可查到
        assert "原始方案" in history[0]["content"]
        assert "更新方案" in history[1]["content"]

    def test_different_facts_independent(self, client):
        """不同 fact_key 的版本链独立"""
        client.update_clinical_fact("P001", "nutrition_plan", "营养v1")
        client.update_clinical_fact("P001", "diagnosis", "诊断v1")
        client.update_clinical_fact("P001", "nutrition_plan", "营养v2")

        # nutrition_plan 有 2 版本
        nut_history = client.get_fact_history("P001", "nutrition_plan")
        assert len(nut_history) == 2
        # diagnosis 仍 1 版本
        diag_history = client.get_fact_history("P001", "diagnosis")
        assert len(diag_history) == 1

    def test_list_fact_keys(self, client):
        """列出所有事实键"""
        client.update_clinical_fact("P001", "nutrition_plan", "x")
        client.update_clinical_fact("P001", "diagnosis", "y")
        keys = client.list_clinical_facts("P001")
        assert "nutrition_plan" in keys
        assert "diagnosis" in keys

    def test_patient_isolation(self, client):
        """不同患者的事实链隔离"""
        client.update_clinical_fact("P001", "plan", "P001方案")
        client.update_clinical_fact("P002", "plan", "P002方案")
        assert client.get_active_fact("P001", "plan")["content"] != \
               client.get_active_fact("P002", "plan")["content"]

    def test_nonexistent_fact_returns_empty(self, client):
        """不存在的事实返回空"""
        assert client.get_active_fact("P001", "nope") is None
        assert client.get_fact_history("P001", "nope") == []
