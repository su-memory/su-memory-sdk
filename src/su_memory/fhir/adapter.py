"""
FHIR R4 适配器 — 将 FHIR Resource 映射到 su-memory 记忆引擎。

支持双向转换：
  FHIR Resource → su-memory 记忆（写入）
  su-memory 查询 → FHIR Bundle（输出）

不依赖 fhir.resources 库，使用原生 dict 操作，
保证零额外依赖即可工作。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from su_memory.clinical.client import ClinicalMemoryClient

logger = logging.getLogger(__name__)


def _extract_patient_id(resource: dict) -> str:
    """从 FHIR Resource 提取 patient_id"""
    # 优先从 subject.reference 提取 "Patient/P001" → "P001"
    subject = resource.get("subject", {})
    if isinstance(subject, dict):
        ref = subject.get("reference", "")
        if ref.startswith("Patient/"):
            return ref.split("/", 1)[1]
        return ref
    # 从 patient.reference 提取
    patient = resource.get("patient", {})
    if isinstance(patient, dict):
        ref = patient.get("reference", "")
        if ref.startswith("Patient/"):
            return ref.split("/", 1)[1]
        return ref
    return "unknown"


def _extract_value(resource: dict) -> tuple[float | None, str]:
    """从 Observation 提取 (value, unit)"""
    vq = resource.get("valueQuantity", {})
    if isinstance(vq, dict):
        return vq.get("value"), vq.get("unit", "")
    return None, ""


class FHIRAdapter:
    """FHIR R4 适配器 — 桥接 FHIR Resource 与 su-memory。

    用法：
        adapter = FHIRAdapter(clinical_client)
        adapter.write_resource(fhir_dict)
        bundle = adapter.query_to_bundle("P001", "Observation")
    """

    # 支持的 Resource 类型 → 处理方法映射
    _HANDLERS = {
        "Patient": "_write_patient",
        "Observation": "_write_observation",
        "MedicationRequest": "_write_medication",
        "AllergyIntolerance": "_write_allergy",
        "Condition": "_write_condition",
        "NutritionOrder": "_write_nutrition_order",
    }

    def __init__(self, client: ClinicalMemoryClient):
        self._client = client

    def write_resource(self, resource: dict) -> str | None:
        """写入 FHIR Resource 到 su-memory。

        Args:
            resource: FHIR R4 Resource dict（必须含 resourceType）

        Returns:
            memory_id，失败返回 None
        """
        rtype = resource.get("resourceType", "")
        handler_name = self._HANDLERS.get(rtype)

        if handler_name is None:
            logger.warning("[FHIR] 不支持的 Resource 类型: %s", rtype)
            return self._write_generic(resource)

        handler = getattr(self, handler_name)
        try:
            return handler(resource)
        except Exception as e:
            logger.error("[FHIR] 写入 %s 失败: %s", rtype, e)
            return None

    def write_batch(self, bundle: dict) -> list[str | None]:
        """批量写入 FHIR Bundle。

        Args:
            bundle: FHIR Bundle dict（含 entry[].resource）

        Returns:
            memory_id 列表
        """
        entries = bundle.get("entry", [])
        results: list[str | None] = []
        for entry in entries:
            resource = entry.get("resource", entry)
            results.append(self.write_resource(resource))
        return results

    def query_to_bundle(
        self,
        patient_id: str,
        resource_type: str | None = None,
        top_k: int = 20,
    ) -> dict:
        """查询患者记忆并转为 FHIR Bundle 输出。

        Args:
            patient_id: 患者 ID
            resource_type: 可选过滤（如 "Observation"）
            top_k: 最大返回数

        Returns:
            FHIR Bundle dict
        """
        hits = self._client.recall(patient_id, patient_id, top_k=top_k)

        entries: list[dict] = []
        for hit in hits:
            meta = hit.get("metadata", {})
            event_type = meta.get("event_type", "")

            # 按类型过滤
            fhir_type = self._event_type_to_fhir(event_type)
            if resource_type and fhir_type != resource_type:
                continue

            entry = self._memory_to_fhir(hit, fhir_type, patient_id)
            entries.append({"resource": entry})

        return {
            "resourceType": "Bundle",
            "type": "searchset",
            "total": len(entries),
            "entry": entries,
        }

    # ── 各 Resource 写入方法 ─────────────────────────────

    def _write_patient(self, resource: dict) -> str | None:
        """写入 Patient Resource"""
        patient_id = resource.get("id", "unknown")
        name_info = resource.get("name", [{}])
        if isinstance(name_info, list) and name_info:
            name = " ".join(name_info[0].get("text", "").split() or
                           name_info[0].get("family", ""))
        else:
            name = str(name_info)

        content = f"患者入院: {name}, ID={patient_id}"
        return self._client.add_patient_event(
            patient_id=str(patient_id),
            content=content,
            event_type="admission",
        )

    def _write_observation(self, resource: dict) -> str | None:
        """写入 Observation Resource（检验值/生命体征）"""
        patient_id = _extract_patient_id(resource)
        code_text = resource.get("code", {}).get("text", "未知检验")
        value, unit = _extract_value(resource)

        if value is not None:
            return self._client.add_lab_value(
                patient_id=patient_id,
                lab_name=code_text,
                value=value,
                unit=unit,
            )
        # 无数值的 Observation 作为普通事件
        content = f"观察记录: {code_text}"
        return self._client.add_patient_event(
            patient_id=patient_id,
            content=content,
            event_type="lab_result",
        )

    def _write_medication(self, resource: dict) -> str | None:
        """写入 MedicationRequest Resource"""
        patient_id = _extract_patient_id(resource)
        med_ref = resource.get("medicationCodeableConcept", {}).get("text", "")
        if not med_ref:
            med_ref = resource.get("medicationReference", {}).get("display", "未知药物")
        dosage = resource.get("dosageInstruction", [{}])
        dosage_text = ""
        if isinstance(dosage, list) and dosage:
            dosage_text = dosage[0].get("text", "")

        content = f"医嘱: {med_ref}"
        if dosage_text:
            content += f" {dosage_text}"

        return self._client.add_patient_event(
            patient_id=patient_id,
            content=content,
            event_type="medication",
        )

    def _write_allergy(self, resource: dict) -> str | None:
        """写入 AllergyIntolerance Resource"""
        patient_id = _extract_patient_id(resource)
        code = resource.get("code", {}).get("text", "未知过敏原")
        severity = resource.get("criticality", "unknown")

        content = f"过敏: {code} (严重程度: {severity})"
        return self._client.add_patient_event(
            patient_id=patient_id,
            content=content,
            event_type="allergy",
        )

    def _write_condition(self, resource: dict) -> str | None:
        """写入 Condition Resource（诊断）"""
        patient_id = _extract_patient_id(resource)
        code = resource.get("code", {}).get("text", "未知诊断")

        content = f"诊断: {code}"
        return self._client.add_patient_event(
            patient_id=patient_id,
            content=content,
            event_type="diagnosis",
        )

    def _write_nutrition_order(self, resource: dict) -> str | None:
        """写入 NutritionOrder Resource"""
        patient_id = _extract_patient_id(resource)
        oral = resource.get("oralDiet", {})
        texture = oral.get("texture", [{}])
        if isinstance(texture, list) and texture:
            diet = texture[0].get("modifier", {}).get("text", "营养处方")
        else:
            diet = "营养处方"

        content = f"营养处方: {diet}"
        return self._client.add_patient_event(
            patient_id=patient_id,
            content=content,
            event_type="plan",
        )

    def _write_generic(self, resource: dict) -> str | None:
        """通用写入（不识别的 Resource 类型）"""
        rtype = resource.get("resourceType", "Unknown")
        patient_id = _extract_patient_id(resource)
        content = f"{rtype}: {resource.get('id', 'no-id')}"
        return self._client.add_patient_event(
            patient_id=patient_id,
            content=content,
            event_type="generic",
        )

    # ── 记忆 → FHIR 转换 ─────────────────────────────────

    def _event_type_to_fhir(self, event_type: str) -> str:
        """临床事件类型 → FHIR Resource 类型"""
        mapping = {
            "lab_result": "Observation",
            "medication": "MedicationRequest",
            "allergy": "AllergyIntolerance",
            "diagnosis": "Condition",
            "plan": "NutritionOrder",
            "admission": "Patient",
        }
        return mapping.get(event_type, "Observation")

    def _memory_to_fhir(
        self, memory: dict, fhir_type: str, patient_id: str
    ) -> dict:
        """将记忆条目转为 FHIR Resource dict"""
        meta = memory.get("metadata", {})
        base: dict[str, Any] = {
            "resourceType": fhir_type,
            "id": memory.get("memory_id", memory.get("id", "")),
            "subject": {"reference": f"Patient/{patient_id}"},
        }

        if fhir_type == "Observation":
            base["code"] = {"text": meta.get("lab_name", "")}
            if "lab_value" in meta:
                base["valueQuantity"] = {
                    "value": meta["lab_value"],
                    "unit": meta.get("lab_unit", ""),
                }
        elif fhir_type == "Condition":
            base["code"] = {"text": memory.get("content", "")}
        elif fhir_type == "AllergyIntolerance":
            base["code"] = {"text": memory.get("content", "")}
        else:
            base["text"] = {"status": "generated", "div": memory.get("content", "")}

        return base
