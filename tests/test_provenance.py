"""
C5 来源溯源 provenance 测试

验证：写入→检索→审计 三环 source 链完整。
医疗级合规要求：每条记忆可溯源到原始来源。
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
        storage_path=str(tmp_path / "prov_test"),
        embedding_backend="none",
        compliance_level="mask",
    )


class TestSourceFields:
    """MemoryNode source 字段测试"""

    def test_default_source_is_unknown(self, client):
        """不传 source 时默认 unknown"""
        client.add_patient_event("P001", "默认来源记忆", "plan")
        hits = client.recall("P001", "默认来源")
        assert len(hits) >= 1
        assert hits[0]["source_type"] == "unknown"
        assert hits[0]["source_confidence"] == 1.0

    def test_explicit_source_passthrough(self, client):
        """显式传 source 应透传到检索结果"""
        client.add_patient_event(
            "P001", "华法林抗凝医嘱", "order",
            source_type="order",
            source_id="EMR-2026-001",
            source_confidence=1.0,
        )
        hits = client.recall("P001", "华法林")
        assert len(hits) >= 1
        assert hits[0]["source_type"] == "order"
        assert hits[0]["source_id"] == "EMR-2026-001"
        assert hits[0]["source_confidence"] == 1.0

    def test_patient_self_report_low_confidence(self, client):
        """患者自述应低可信度"""
        client.add_patient_event(
            "P001", "我感觉最近头晕", "symptom",
            source_type="patient",
            source_id="dialog-001",
            source_confidence=0.6,
        )
        hits = client.recall("P001", "头晕")
        assert len(hits) >= 1
        assert hits[0]["source_type"] == "patient"
        assert hits[0]["source_confidence"] == 0.6

    def test_ai_inferred_lowest_confidence(self, client):
        """AI 推断应最低可信度"""
        client.add_patient_event(
            "P001", "推测存在营养不良风险", "assessment",
            source_type="ai_inferred",
            source_id="infer-001",
            source_confidence=0.4,
        )
        hits = client.recall("P001", "营养不良")
        assert len(hits) >= 1
        assert hits[0]["source_type"] == "ai_inferred"
        assert hits[0]["source_confidence"] == 0.4


class TestAuditSourceChain:
    """审计日志来源链测试"""

    def test_audit_records_source(self, client):
        """审计日志应记录来源链"""
        client.add_patient_event(
            "P001", "华法林医嘱", "order",
            source_type="order",
            source_id="EMR-001",
        )
        entries = client.audit.query(patient_id="P001", action="add")
        assert len(entries) >= 1
        assert entries[-1].source_type == "order"
        assert entries[-1].source_id == "EMR-001"

    def test_audit_default_source_empty(self, client):
        """不传 source 时审计 source_type 为空（向后兼容）"""
        client.add_patient_event("P001", "无来源标记", "plan")
        entries = client.audit.query(patient_id="P001", action="add")
        assert len(entries) >= 1
        # 默认 unknown 在 node 层，审计层记录为空字符串（区分"未提供"与"未知"）
        assert entries[-1].source_type in ("", "unknown")  # 向后兼容


class TestBackwardCompat:
    """向后兼容测试"""

    def test_old_call_signature_works(self, client):
        """旧调用方式（无 source 参数）仍正常工作"""
        mid = client.add_patient_event("P001", "旧式调用", "plan")
        assert mid is not None
        hits = client.recall("P001", "旧式")
        assert len(hits) >= 1

    def test_engine_add_without_source(self, client):
        """引擎层 add() 不传 source 仍正常"""
        mid = client._engine.add("直接引擎调用", metadata={"patient_id": "P002"})
        assert mid is not None
        hits = client.recall("P002", "直接引擎")
        assert len(hits) >= 1
        assert hits[0]["source_type"] == "unknown"
