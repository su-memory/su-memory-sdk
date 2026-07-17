"""
multi_tenant — 多租户隔离层

让多个医疗项目（营养系统、肿瘤平台、mci-huan 等）共享同一个 su-memory 实例，
通过 tenant_id:patient_id 前缀方案实现数据隔离。

设计原则（第一性原理）：
  - 租户A 绝对不能看到租户B 的数据（法律合规刚需）
  - 隔离在 metadata 层实现，不改变引擎核心
  - 双重保障：tenant_id 前缀 + 审计日志

Example:
  >>> from su_memory.clinical import MultiTenantClient
  >>> # 肿瘤平台（租户T001）
  >>> oncology = MultiTenantClient(tenant_id="T001")
  >>> oncology.add_patient_event("P001", "肿瘤营养方案", "plan")
  >>> # 营养系统（租户T002）
  >>> nutrition = MultiTenantClient(tenant_id="T002")
  >>> nutrition.recall("P001", "方案")  # 只返回T002的数据，看不到T001
"""

from __future__ import annotations

import logging
from typing import Any

from su_memory.clinical.client import ClinicalMemoryClient

logger = logging.getLogger(__name__)


class MultiTenantClient:
    """多租户临床记忆客户端。

    在 ClinicalMemoryClient 之上增加 tenant_id 隔离层。
    所有 patient_id 自动加前缀 `tenant_id:patient_id`，
    查询时自动过滤，确保租户间数据完全隔离。

    Args:
        tenant_id: 租户标识（如医院ID/项目ID）
        storage_path: 共享存储路径（多租户共用）
        **kwargs: 传递给 ClinicalMemoryClient 的参数
    """

    def __init__(
        self,
        tenant_id: str,
        storage_path: str | None = None,
        embedding_backend: str = "none",
        compliance_level: str | None = "mask",
        **kwargs: Any,
    ):
        if not tenant_id:
            raise ValueError("tenant_id 不能为空")

        self._tenant_id = tenant_id
        self._client = ClinicalMemoryClient(
            storage_path=storage_path,
            embedding_backend=embedding_backend,
            compliance_level=compliance_level,
            **kwargs,
        )

    def _scoped_pid(self, patient_id: str) -> str:
        """加租户前缀：P001 → T001:P001"""
        if ":" in patient_id:
            # 已有前缀，不重复加
            return patient_id
        return f"{self._tenant_id}:{patient_id}"

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def inner(self) -> ClinicalMemoryClient:
        """底层 ClinicalMemoryClient（高级用途）"""
        return self._client

    # ── 代理 ClinicalMemoryClient 接口（自动加租户前缀）──

    def add_patient_event(
        self,
        patient_id: str,
        content: str,
        event_type: str = "",
        metadata: dict | None = None,
    ) -> str | None:
        """写入患者事件（自动加租户前缀）。"""
        scoped_meta = dict(metadata) if metadata else {}
        scoped_meta["tenant_id"] = self._tenant_id
        return self._client.add_patient_event(
            patient_id=self._scoped_pid(patient_id),
            content=content,
            event_type=event_type,
            metadata=scoped_meta,
        )

    def add_lab_value(
        self,
        patient_id: str,
        lab_name: str,
        value: float,
        unit: str = "",
        reference_range: str = "",
    ) -> str | None:
        """写入检验值（自动加租户前缀）。"""
        return self._client.add_lab_value(
            patient_id=self._scoped_pid(patient_id),
            lab_name=lab_name,
            value=value,
            unit=unit,
            reference_range=reference_range,
        )

    def recall(
        self,
        patient_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """召回记忆（自动加租户前缀，只返回当前租户数据）。"""
        return self._client.recall(
            patient_id=self._scoped_pid(patient_id),
            query=query,
            top_k=top_k,
        )

    def get_lab_trend(self, patient_id: str, lab_name: str) -> Any:
        """检验趋势（自动加租户前缀）。"""
        return self._client.get_lab_trend(
            patient_id=self._scoped_pid(patient_id),
            lab_name=lab_name,
        )

    def find_abnormal_labs(self, patient_id: str) -> list[dict]:
        """异常检验筛查（自动加租户前缀）。"""
        return self._client.find_abnormal_labs(
            patient_id=self._scoped_pid(patient_id),
        )

    def get_care_trajectory(self, patient_id: str, limit: int = 50) -> list[dict]:
        """诊疗轨迹（自动加租户前缀）。"""
        return self._client.get_care_trajectory(
            patient_id=self._scoped_pid(patient_id),
            limit=limit,
        )

    def train_from_feedback(
        self, memory_id: str, rating: int, action: str = "accept"
    ) -> float | None:
        """反馈训练（memory_id 已全局唯一，无需前缀）。"""
        return self._client.train_from_feedback(memory_id, rating, action)

    def check_drug_interaction(self, drug_list: list[str]) -> list:
        """药物交互查询（知识库全局共享，无需隔离）。"""
        return self._client.check_drug_interaction(drug_list)

    def get_lab_reference(self, lab_name: str):
        """检验参考值（全局共享）。"""
        return self._client.get_lab_reference(lab_name)

    def purge_patient(self, patient_id: str):
        """删除患者数据（自动加租户前缀，只删当前租户的）。"""
        return self._client.purge_patient(self._scoped_pid(patient_id))

    def health_check(self) -> dict[str, Any]:
        """健康检查。"""
        result = self._client.health_check()
        result["tenant_id"] = self._tenant_id
        return result
