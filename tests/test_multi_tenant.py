"""
多租户隔离测试 — P3-S3 验证

验证多个医疗项目共享同一记忆引擎时的数据隔离。
"""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("SU_MEMORY_SKIP_ENV_CHECK", "1")
os.environ.setdefault("MEMORY_EMBEDDING_BACKEND", "none")


class TestTenantIsolation:
    """租户数据隔离测试"""

    def test_tenants_cannot_see_each_other(self, tmp_path):
        """租户A 看不到租户B 的数据"""
        from su_memory.clinical import MultiTenantClient

        shared_path = str(tmp_path / "shared")
        tenant_a = MultiTenantClient("T001", storage_path=shared_path)
        tenant_b = MultiTenantClient("T002", storage_path=shared_path)

        # T001 写入
        tenant_a.add_patient_event("P001", "肿瘤营养方案A", "plan")
        # T002 写入
        tenant_b.add_patient_event("P001", "营养系统方案B", "plan")

        # T001 只能看到自己的
        hits_a = tenant_a.recall("P001", "方案", top_k=10)
        contents_a = [h.get("content", "") for h in hits_a]
        assert any("肿瘤" in c for c in contents_a)
        assert not any("营养系统" in c for c in contents_a)

        # T002 只能看到自己的
        hits_b = tenant_b.recall("P001", "方案", top_k=10)
        contents_b = [h.get("content", "") for h in hits_b]
        assert any("营养系统" in c for c in contents_b)
        assert not any("肿瘤" in c for c in contents_b)

    def test_lab_isolation(self, tmp_path):
        """检验值租户隔离"""
        from su_memory.clinical import MultiTenantClient

        shared_path = str(tmp_path / "lab_shared")
        tenant_a = MultiTenantClient("T001", storage_path=shared_path)
        tenant_b = MultiTenantClient("T002", storage_path=shared_path)

        tenant_a.add_lab_value("P001", "白蛋白", 28.0, "g/L", "35-55")
        tenant_b.add_lab_value("P001", "白蛋白", 50.0, "g/L", "35-55")

        trend_a = tenant_a.get_lab_trend("P001", "白蛋白")
        trend_b = tenant_b.get_lab_trend("P001", "白蛋白")

        assert trend_a.first_value == 28.0
        assert trend_b.first_value == 50.0

    def test_purge_only_deletes_tenant_data(self, tmp_path):
        """删除只影响当前租户"""
        from su_memory.clinical import MultiTenantClient

        shared_path = str(tmp_path / "purge_shared")
        tenant_a = MultiTenantClient("T001", storage_path=shared_path)
        tenant_b = MultiTenantClient("T002", storage_path=shared_path)

        tenant_a.add_patient_event("P001", "T001数据", "plan")
        tenant_b.add_patient_event("P001", "T002数据", "plan")

        # T001 删除 P001
        report = tenant_a.purge_patient("P001")
        assert report.memories_deleted >= 1

        # T002 的 P001 数据仍在
        hits_b = tenant_b.recall("P001", "数据", top_k=10)
        assert len(hits_b) >= 1

    def test_tenant_id_in_metadata(self, tmp_path):
        """写入的数据包含 tenant_id 元数据"""
        from su_memory.clinical import MultiTenantClient

        tenant = MultiTenantClient("T003", storage_path=str(tmp_path / "meta"))
        mid = tenant.add_patient_event("P001", "测试", "plan")

        node = tenant.inner._engine._graph._nodes.get(mid)
        assert node is not None
        assert node.metadata.get("tenant_id") == "T003"
        assert node.metadata.get("patient_id") == "T003:P001"


class TestSharedKnowledge:
    """共享知识库测试（药物交互/检验参考不隔离）"""

    def test_drug_interaction_shared(self, tmp_path):
        """药物交互知识全局共享"""
        from su_memory.clinical import MultiTenantClient

        tenant_a = MultiTenantClient("T001", storage_path=str(tmp_path / "kb"))
        results = tenant_a.check_drug_interaction(["华法林"])
        assert len(results) >= 1

    def test_lab_reference_shared(self, tmp_path):
        """检验参考值全局共享"""
        from su_memory.clinical import MultiTenantClient

        tenant_b = MultiTenantClient("T002", storage_path=str(tmp_path / "kb2"))
        ref = tenant_b.get_lab_reference("白蛋白")
        assert ref is not None


class TestHealthCheck:
    """健康检查测试"""

    def test_health_check_includes_tenant(self, tmp_path):
        from su_memory.clinical import MultiTenantClient
        tenant = MultiTenantClient("T001", storage_path=str(tmp_path / "hc"))
        health = tenant.health_check()
        assert health["tenant_id"] == "T001"


class TestEdgeCases:
    """边界测试"""

    def test_empty_tenant_id_raises(self):
        from su_memory.clinical import MultiTenantClient
        with pytest.raises(ValueError):
            MultiTenantClient("")

    def test_idempotent_scoping(self, tmp_path):
        """重复前缀不叠加"""
        from su_memory.clinical import MultiTenantClient
        tenant = MultiTenantClient("T001", storage_path=str(tmp_path / "idem"))
        scoped = tenant._scoped_pid("T001:P001")  # 已有前缀
        assert scoped == "T001:P001"  # 不变成 T001:T001:P001
