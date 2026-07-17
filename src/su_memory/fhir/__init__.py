"""
su_memory.fhir — FHIR R4 适配器

支持 FHIR Resource 直接写入/查询 su-memory 记忆引擎。

支持的 Resource 类型：
  - Patient: 患者基本信息
  - Observation: 检验值/生命体征
  - MedicationRequest: 医嘱
  - AllergyIntolerance: 过敏
  - Condition: 诊断
  - NutritionOrder: 营养处方

Example:
  >>> from su_memory.fhir import FHIRAdapter
  >>> adapter = FHIRAdapter(client)
  >>> adapter.write_observation({
  ...     "resourceType": "Observation",
  ...     "subject": {"reference": "Patient/P001"},
  ...     "code": {"text": "白蛋白"},
  ...     "valueQuantity": {"value": 30, "unit": "g/L"},
  ... })
"""
from su_memory.fhir.adapter import FHIRAdapter

__all__ = ["FHIRAdapter"]
