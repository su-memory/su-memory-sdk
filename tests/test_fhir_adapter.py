"""
FHIR R4 适配器测试 — P3-S2 验证
"""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("SU_MEMORY_SKIP_ENV_CHECK", "1")
os.environ.setdefault("MEMORY_EMBEDDING_BACKEND", "none")


@pytest.fixture
def adapter(tmp_path):
    from su_memory.clinical import ClinicalMemoryClient
    from su_memory.fhir import FHIRAdapter
    client = ClinicalMemoryClient(
        storage_path=str(tmp_path / "fhir_test"),
        embedding_backend="none",
    )
    return FHIRAdapter(client)


class TestWriteResources:
    """FHIR Resource 写入测试"""

    def test_write_observation(self, adapter):
        """写入 Observation（检验值）"""
        mid = adapter.write_resource({
            "resourceType": "Observation",
            "subject": {"reference": "Patient/P001"},
            "code": {"text": "白蛋白"},
            "valueQuantity": {"value": 30, "unit": "g/L"},
        })
        assert mid is not None

    def test_write_condition(self, adapter):
        """写入 Condition（诊断）"""
        mid = adapter.write_resource({
            "resourceType": "Condition",
            "subject": {"reference": "Patient/P001"},
            "code": {"text": "2型糖尿病"},
        })
        assert mid is not None

    def test_write_allergy(self, adapter):
        """写入 AllergyIntolerance"""
        mid = adapter.write_resource({
            "resourceType": "AllergyIntolerance",
            "patient": {"reference": "Patient/P001"},
            "code": {"text": "花生"},
            "criticality": "high",
        })
        assert mid is not None

    def test_write_medication(self, adapter):
        """写入 MedicationRequest"""
        mid = adapter.write_resource({
            "resourceType": "MedicationRequest",
            "subject": {"reference": "Patient/P001"},
            "medicationCodeableConcept": {"text": "华法林"},
            "dosageInstruction": [{"text": "2.5mg qd"}],
        })
        assert mid is not None

    def test_write_nutrition_order(self, adapter):
        """写入 NutritionOrder"""
        mid = adapter.write_resource({
            "resourceType": "NutritionOrder",
            "patient": {"reference": "Patient/P001"},
            "oralDiet": {"texture": [{"modifier": {"text": "高蛋白饮食"}}]},
        })
        assert mid is not None

    def test_write_unsupported_type(self, adapter):
        """不支持类型走通用写入"""
        mid = adapter.write_resource({
            "resourceType": "Encounter",
            "id": "enc-001",
            "subject": {"reference": "Patient/P001"},
        })
        assert mid is not None  # 通用写入不报错


class TestBatchWrite:
    """批量写入测试"""

    def test_write_bundle(self, adapter):
        """写入 FHIR Bundle"""
        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {"resource": {
                    "resourceType": "Observation",
                    "subject": {"reference": "Patient/P001"},
                    "code": {"text": "白蛋白"},
                    "valueQuantity": {"value": 35, "unit": "g/L"},
                }},
                {"resource": {
                    "resourceType": "Condition",
                    "subject": {"reference": "Patient/P001"},
                    "code": {"text": "糖尿病"},
                }},
            ],
        }
        results = adapter.write_batch(bundle)
        assert len(results) == 2
        assert all(r is not None for r in results)


class TestQueryToBundle:
    """查询转 FHIR Bundle 测试"""

    def test_query_returns_bundle(self, adapter):
        """查询返回 FHIR Bundle"""
        adapter.write_resource({
            "resourceType": "Observation",
            "subject": {"reference": "Patient/P001"},
            "code": {"text": "白蛋白"},
            "valueQuantity": {"value": 30, "unit": "g/L"},
        })
        bundle = adapter.query_to_bundle("P001")
        assert bundle["resourceType"] == "Bundle"
        assert bundle["total"] >= 1
        assert len(bundle["entry"]) >= 1

    def test_query_filtered_by_type(self, adapter):
        """按类型过滤"""
        adapter.write_resource({
            "resourceType": "Condition",
            "subject": {"reference": "Patient/P001"},
            "code": {"text": "高血压"},
        })
        bundle = adapter.query_to_bundle("P001", resource_type="Observation")
        # 没有 Observation 类型，应该返回空或不含 Condition
        for entry in bundle["entry"]:
            assert entry["resource"]["resourceType"] == "Observation"


class TestObservationLabIntegration:
    """Observation → LabValue 集成测试"""

    def test_observation_becomes_lab_trend(self, adapter, tmp_path):
        """FHIR Observation 写入后可查检验趋势"""
        for val in [28.0, 32.0, 36.0]:
            adapter.write_resource({
                "resourceType": "Observation",
                "subject": {"reference": "Patient/P001"},
                "code": {"text": "白蛋白"},
                "valueQuantity": {"value": val, "unit": "g/L"},
            })

        # 通过 ClinicalMemoryClient 查趋势
        trend = adapter._client.get_lab_trend("P001", "白蛋白")
        assert trend.direction.value == "up"
        assert trend.count == 3
