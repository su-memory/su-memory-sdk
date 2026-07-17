"""
合规层测试 — P2-S3 验证

PHI 脱敏、审计日志、删除权。
"""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("SU_MEMORY_SKIP_ENV_CHECK", "1")
os.environ.setdefault("MEMORY_EMBEDDING_BACKEND", "none")


class TestPHISanitizer:
    """PHI 脱敏测试"""

    def test_mask_name(self):
        from su_memory.clinical.compliance import mask_name
        assert mask_name("张三") == "张*"
        assert mask_name("欧阳修") == "欧*修"

    def test_mask_id_card(self):
        from su_memory.clinical.compliance import mask_id_card
        masked = mask_id_card("330102199001011234")
        assert masked.startswith("3301")
        assert masked.endswith("1234")
        assert "*" in masked

    def test_mask_phone(self):
        from su_memory.clinical.compliance import mask_phone
        masked = mask_phone("13812345678")
        assert masked.startswith("138")
        assert masked.endswith("5678")
        assert "*" in masked

    def test_mask_email(self):
        from su_memory.clinical.compliance import mask_email
        masked = mask_email("test@example.com")
        assert "@" in masked
        assert masked.startswith("t")
        assert "*" in masked

    def test_sanitize_mask_level(self):
        from su_memory.clinical.compliance import PHISanitizer
        s = PHISanitizer(level="mask")
        result = s.sanitize({"patient_name": "张三", "id_card": "330102199001011234", "diagnosis": "糖尿病"})
        assert result["patient_name"] == "张*"
        assert "*" in result["id_card"]
        assert result["diagnosis"] == "糖尿病"  # 非 PHI 不脱敏

    def test_sanitize_hash_level(self):
        from su_memory.clinical.compliance import PHISanitizer
        s = PHISanitizer(level="hash")
        result = s.sanitize({"patient_name": "张三"})
        assert result["patient_name"] != "张三"
        assert len(result["patient_name"]) == 16

    def test_sanitize_remove_level(self):
        from su_memory.clinical.compliance import PHISanitizer
        s = PHISanitizer(level="remove")
        result = s.sanitize({"patient_name": "张三", "diagnosis": "糖尿病"})
        assert "patient_name" not in result
        assert "diagnosis" in result


class TestAuditLogger:
    """审计日志测试"""

    def test_log_and_query(self):
        from su_memory.clinical.compliance import AuditLogger
        audit = AuditLogger()
        audit.log("add", patient_id="P001", memory_id="mem_001")
        audit.log("query", patient_id="P001", memory_id="mem_001")
        entries = audit.query(patient_id="P001")
        assert len(entries) == 2

    def test_query_by_action(self):
        from su_memory.clinical.compliance import AuditLogger
        audit = AuditLogger()
        audit.log("add", patient_id="P001")
        audit.log("query", patient_id="P001")
        adds = audit.query(action="add")
        assert len(adds) == 1
        assert adds[0].action == "add"

    def test_file_persistence(self, tmp_path):
        """审计日志持久化到文件"""
        from su_memory.clinical.compliance import AuditLogger
        log_path = str(tmp_path / "audit.jsonl")
        audit = AuditLogger(log_path=log_path)
        audit.log("add", patient_id="P001", memory_id="mem_001")
        # 验证文件写入
        import json
        with open(log_path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["action"] == "add"


class TestComplianceManager:
    """合规管理器集成测试"""

    def test_auto_sanitize_on_add(self, tmp_path):
        """add 时自动脱敏 PHI"""
        from su_memory.sdk.lite_pro import SuMemoryLitePro
        from su_memory.clinical import ComplianceManager

        client = SuMemoryLitePro(
            storage_path=str(tmp_path / "comp_test"),
            embedding_backend="none",
            enable_llm_energy=False,
        )
        cm = ComplianceManager(client, phi_level="mask")
        cm.inject_hooks()

        mid = client.add("患者入院", metadata={
            "patient_id": "P001",
            "patient_name": "张三",
            "id_card": "330102199001011234",
            "diagnosis": "糖尿病",
        })

        # 验证 metadata 被脱敏
        node = client._graph._nodes[mid]
        meta = node.metadata
        assert meta["patient_name"] == "张*"
        assert "*" in meta["id_card"]
        assert meta["diagnosis"] == "糖尿病"  # 非 PHI 保留

    def test_auto_audit_on_operations(self, tmp_path):
        """add/query 自动记录审计"""
        from su_memory.sdk.lite_pro import SuMemoryLitePro
        from su_memory.clinical import ComplianceManager

        client = SuMemoryLitePro(
            storage_path=str(tmp_path / "audit_test"),
            embedding_backend="none",
            enable_llm_energy=False,
        )
        cm = ComplianceManager(client)
        cm.inject_hooks()

        client.add("test", metadata={"patient_id": "P001"})
        client.query("test", top_k=3)

        entries = cm.audit.query(patient_id="P001")
        actions = [e.action for e in entries]
        assert "add" in actions
        assert "query" in actions

    def test_purge_patient(self, tmp_path):
        """删除权：彻底删除患者所有记忆"""
        from su_memory.sdk.lite_pro import SuMemoryLitePro
        from su_memory.clinical import ComplianceManager

        client = SuMemoryLitePro(
            storage_path=str(tmp_path / "purge_test"),
            embedding_backend="none",
            enable_llm_energy=False,
        )
        cm = ComplianceManager(client)
        cm.inject_hooks()

        # 写入 P001 和 P002 的记忆
        for i in range(3):
            client.add(f"P001记忆{i}", metadata={"patient_id": "P001"})
        client.add("P002记忆", metadata={"patient_id": "P002"})

        # 删除 P001
        report = cm.purge_patient("P001")
        assert report.success
        assert report.memories_deleted == 3

        # 验证 P001 完全清除，P002 保留
        p001_count = sum(
            1 for n in client._graph._nodes.values()
            if (n.metadata or {}).get("patient_id") == "P001"
        )
        p002_count = sum(
            1 for n in client._graph._nodes.values()
            if (n.metadata or {}).get("patient_id") == "P002"
        )
        assert p001_count == 0
        assert p002_count == 1

        # 审计日志记录了删除操作
        deletes = cm.audit.query(action="delete")
        assert len(deletes) >= 1
