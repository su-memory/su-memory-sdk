"""
P0 安全修复对抗性验证测试

验证 V1/V4/V5/V6/V8 五个高危漏洞已修复。
"""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("SU_MEMORY_SKIP_ENV_CHECK", "1")
os.environ.setdefault("MEMORY_EMBEDDING_BACKEND", "none")


class TestV8EmptyPatientId:
    """V8: 空 patient_id 拒绝（防串扰）"""

    def test_empty_patient_id_add_rejected(self, tmp_path):
        from su_memory.clinical import ClinicalMemoryClient
        c = ClinicalMemoryClient(
            storage_path=str(tmp_path / "v8"),
            embedding_backend="none", compliance_level=None,
        )
        # 空 patient_id 写入应被拒绝
        mid = c.add_patient_event("", "患者A秘密", "plan")
        assert mid is None
        # 空格 patient_id 也拒绝
        mid2 = c.add_patient_event("   ", "患者B数据", "plan")
        assert mid2 is None

    def test_empty_patient_id_recall_rejected(self, tmp_path):
        from su_memory.clinical import ClinicalMemoryClient
        c = ClinicalMemoryClient(
            storage_path=str(tmp_path / "v8r"),
            embedding_backend="none", compliance_level=None,
        )
        # 空 patient_id 查询返回空
        hits = c.recall("", "任何查询")
        assert hits == []
        hits2 = c.recall("   ", "查询")
        assert hits2 == []


class TestV1FailClosed:
    """V1: 门控异常时 fail-closed（不返回未门控结果）"""

    def test_gate_exception_marks_unknown(self, tmp_path):
        from su_memory.clinical import ClinicalMemoryClient
        c = ClinicalMemoryClient(
            storage_path=str(tmp_path / "v1"),
            embedding_backend="none", compliance_level=None,
        )
        # 正常写入华法林
        c.add_patient_event("P1", "华法林抗凝", "order")
        # 破坏 safety_gate 让 screen 抛异常
        original_screen = c._safety_gate.screen
        def broken_screen(results, patient_allergies=None):
            raise RuntimeError("模拟知识库损坏")
        c._safety_gate.screen = broken_screen
        try:
            hits = c.recall("P1", "华法林")
            # fail-closed: 不应返回干净的 safe 结果，应标 unknown
            assert len(hits) >= 1
            assert hits[0]["risk_level"] == "unknown"
            assert any("门控异常" in f for f in hits[0]["risk_flags"])
        finally:
            c._safety_gate.screen = original_screen


class TestV4ContentSanitization:
    """V4: content 正文 PHI 脱敏"""

    def test_id_card_in_content_masked(self):
        from su_memory.clinical.compliance import PHISanitizer
        s = PHISanitizer(level="mask")
        content = "患者身份证330102199001011234诊断营养不良"
        result = s.sanitize_content(content)
        assert "330102199001011234" not in result
        assert "3301" in result  # 保留前4位

    def test_phone_in_content_masked(self):
        from su_memory.clinical.compliance import PHISanitizer
        s = PHISanitizer(level="mask")
        content = "联系电话13812345678请回访"
        result = s.sanitize_content(content)
        assert "13812345678" not in result
        assert "138" in result and "5678" in result

    def test_email_in_content_masked(self):
        from su_memory.clinical.compliance import PHISanitizer
        s = PHISanitizer(level="mask")
        content = "邮箱zhangsan@hospital.com联系"
        result = s.sanitize_content(content)
        assert "zhangsan@hospital.com" not in result

    def test_content_sanitized_on_add(self, tmp_path):
        """开启合规层后，content 里 PHI 被脱敏存储"""
        from su_memory.clinical import ClinicalMemoryClient
        c = ClinicalMemoryClient(
            storage_path=str(tmp_path / "v4"),
            embedding_backend="none", compliance_level="mask",
        )
        c.add_patient_event("P1", "患者电话13812345678需随访", "plan")
        hits = c.recall("P1", "电话")
        assert len(hits) >= 1
        # 存储的 content 里手机号应已脱敏
        assert "13812345678" not in hits[0]["content"]
        assert "138****5678" in hits[0]["content"]


class TestV6PurgeClearsIndexes:
    """V6: purge 后向量/倒排索引清理，删除权有效"""

    def test_purged_memory_not_recallable(self, tmp_path):
        from su_memory.clinical import ClinicalMemoryClient
        c = ClinicalMemoryClient(
            storage_path=str(tmp_path / "v6"),
            embedding_backend="none", compliance_level="mask",
        )
        c.add_patient_event("P1", "华法林特殊方案唯一标记XYZ", "order")
        # 确认可召回
        assert len(c.recall("P1", "华法林")) >= 1
        # purge
        report = c.purge_patient("P1")
        assert report.memories_deleted >= 1
        # purge 后不应再召回（倒排/缓存已清）
        hits = c.recall("P1", "华法林特殊方案唯一标记XYZ")
        assert len(hits) == 0


class TestV5AuditPhiSanitization:
    """V5: 审计日志 PHI 脱敏"""

    def test_audit_phi_in_patient_id_masked(self, tmp_path):
        """patient_id 含手机号时审计脱敏"""
        from su_memory.clinical import ClinicalMemoryClient
        c = ClinicalMemoryClient(
            storage_path=str(tmp_path / "v5"),
            embedding_backend="none", compliance_level="mask",
        )
        # 用手机号当 patient_id（模拟脏数据）
        c.add_patient_event("13812345678", "事件记录", "plan")
        entries = c.audit.query(action="add")
        assert len(entries) >= 1
        # 审计里的 patient_id 应脱敏
        audit_pid = entries[-1].patient_id
        assert "13812345678" not in audit_pid
        assert "138" in audit_pid or "*" in audit_pid
