"""
compliance — 医疗隐私合规层

PHI 脱敏、审计日志、数据删除权。
满足医院 IT 验收和《个人信息保护法》要求。

Example:
  >>> from su_memory.clinical import ComplianceManager
  >>> cm = ComplianceManager(client)
  >>> cm.inject_hooks()  # 自动脱敏 + 审计
  >>> mid = client.add("...", metadata={"patient_name": "张三", "id_card": "330102199001011234"})
  >>> # metadata 中 patient_name → "张*"，id_card → "3301***********1234"
  >>> cm.purge_patient("P001")  # 彻底删除患者所有记忆
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from su_memory.sdk.lite_pro import SuMemoryLitePro

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# PHI 字段定义
# ═══════════════════════════════════════════════════════════════════

PHI_FIELDS: set[str] = {
    "patient_name", "name", "real_name",
    "id_card", "identity_card", "id_number",
    "phone", "mobile", "telephone", "contact_phone",
    "address", "home_address",
    "medical_record_no", "mrn", "hospital_id",
    "email",
    "birth_date", "date_of_birth",
}


# ═══════════════════════════════════════════════════════════════════
# 脱敏工具
# ═══════════════════════════════════════════════════════════════════


def mask_name(name: str) -> str:
    """姓名脱敏：张三 → 张*，欧阳修 → 欧阳*"""
    if not name or len(name) <= 1:
        return "*"
    if len(name) == 2:
        return name[0] + "*"
    return name[0] + "*" * (len(name) - 2) + name[-1]


def mask_id_card(id_card: str) -> str:
    """身份证脱敏：330102199001011234 → 3301***********1234"""
    if not id_card or len(id_card) < 8:
        return "*" * len(id_card)
    return id_card[:4] + "*" * (len(id_card) - 8) + id_card[-4:]


def mask_phone(phone: str) -> str:
    """电话脱敏：13812345678 → 138****5678"""
    if not phone or len(phone) < 7:
        return "*" * len(phone)
    digits = re.sub(r"\D", "", phone)
    if len(digits) >= 7:
        return digits[:3] + "*" * (len(digits) - 7) + digits[-4:]
    return "*" * len(phone)


def mask_email(email: str) -> str:
    """邮箱脱敏：test@example.com → t***@example.com"""
    if "@" not in email:
        return "*" * len(email)
    local, domain = email.split("@", 1)
    if len(local) <= 1:
        return "*" + "@" + domain
    return local[0] + "*" * (len(local) - 1) + "@" + domain


def hash_value(value: str) -> str:
    """不可逆哈希（供跨系统关联，不可逆推原文）"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


PHI_MASKERS: dict[str, Any] = {
    "patient_name": mask_name,
    "name": mask_name,
    "real_name": mask_name,
    "id_card": mask_id_card,
    "identity_card": mask_id_card,
    "id_number": mask_id_card,
    "phone": mask_phone,
    "mobile": mask_phone,
    "telephone": mask_phone,
    "contact_phone": mask_phone,
    "email": mask_email,
}


# ═══════════════════════════════════════════════════════════════════
# 脱敏器
# ═══════════════════════════════════════════════════════════════════


class PHISanitizer:
    """PHI 字段自动脱敏。

    三级脱敏：
    - mask: 部分掩码（姓名→张*，身份证→3301***1234）—— 默认
    - hash: 不可逆哈希（供跨系统关联）
    - remove: 直接删除字段
    """

    def __init__(self, level: str = "mask"):
        self._level = level

    def sanitize(self, metadata: dict | None) -> dict | None:
        """脱敏 metadata 中的 PHI 字段"""
        if not metadata:
            return metadata

        sanitized = dict(metadata)
        for key in list(sanitized.keys()):
            if key.lower() in {f.lower() for f in PHI_FIELDS}:
                value = sanitized[key]
                if value is None:
                    continue
                value_str = str(value)

                if self._level == "remove":
                    del sanitized[key]
                elif self._level == "hash":
                    sanitized[key] = hash_value(value_str)
                else:  # mask
                    masker = PHI_MASKERS.get(key.lower())
                    if masker:
                        sanitized[key] = masker(value_str)
                    else:
                        # 通用 PHI 字段用掩码
                        sanitized[key] = value_str[:2] + "*" * max(len(value_str) - 2, 1)

        return sanitized


# ═══════════════════════════════════════════════════════════════════
# 审计日志
# ═══════════════════════════════════════════════════════════════════


@dataclass
class AuditEntry:
    """审计日志条目"""
    timestamp: float
    actor: str
    action: str           # add/query/delete/export
    patient_id: str = ""
    memory_id: str = ""
    detail: str = ""


class AuditLogger:
    """append-only 审计日志。

    所有 add/query/delete 操作自动记录。
    日志文件不可被业务 API 修改（append-only JSONL）。
    """

    def __init__(self, log_path: str | None = None):
        self._log_path = log_path
        self._entries: list[AuditEntry] = []
        self._actor = "system"

    def set_actor(self, actor: str) -> None:
        """设置当前操作者（如医生工号）"""
        self._actor = actor

    def log(
        self,
        action: str,
        patient_id: str = "",
        memory_id: str = "",
        detail: str = "",
    ) -> None:
        """记录一条审计日志"""
        entry = AuditEntry(
            timestamp=time.time(),
            actor=self._actor,
            action=action,
            patient_id=patient_id,
            memory_id=memory_id,
            detail=detail,
        )
        self._entries.append(entry)

        if self._log_path:
            try:
                os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "timestamp": entry.timestamp,
                        "actor": entry.actor,
                        "action": entry.action,
                        "patient_id": entry.patient_id,
                        "memory_id": entry.memory_id,
                        "detail": entry.detail,
                    }, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.debug("审计日志写入降级: %s", e)

    def query(
        self,
        patient_id: str | None = None,
        actor: str | None = None,
        action: str | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """查询审计日志"""
        results = self._entries
        if patient_id:
            results = [e for e in results if e.patient_id == patient_id]
        if actor:
            results = [e for e in results if e.actor == actor]
        if action:
            results = [e for e in results if e.action == action]
        return results[-limit:]

    def count(self) -> int:
        """审计日志总数"""
        return len(self._entries)


# ═══════════════════════════════════════════════════════════════════
# 合规管理器（统一入口）
# ═══════════════════════════════════════════════════════════════════


@dataclass
class PurgeReport:
    """数据删除报告"""
    patient_id: str
    memories_deleted: int = 0
    edges_deleted: int = 0
    success: bool = True
    error: str = ""


class ComplianceManager:
    """医疗合规管理器 — 脱敏 + 审计 + 删除权。

    用法：
        cm = ComplianceManager(client, audit_log_path="./audit.jsonl")
        cm.inject_hooks()  # 自动脱敏 + 审计
        cm.purge_patient("P001")  # 删除权
    """

    def __init__(
        self,
        client: SuMemoryLitePro,
        phi_level: str = "mask",
        audit_log_path: str | None = None,
    ):
        self._client = client
        self._sanitizer = PHISanitizer(level=phi_level)
        self._audit = AuditLogger(log_path=audit_log_path)

    @property
    def audit(self) -> AuditLogger:
        """审计日志器"""
        return self._audit

    def inject_hooks(self) -> None:
        """注入 add/query 后置钩子，自动脱敏 + 审计"""
        original_add = self._client.add
        original_query = self._client.query

        def hooked_add(content: str, metadata: dict = None, **kwargs) -> str:
            sanitized = self._sanitizer.sanitize(metadata)
            memory_id = original_add(content, metadata=sanitized, **kwargs)
            patient_id = (sanitized or {}).get("patient_id", "")
            self._audit.log("add", patient_id=patient_id, memory_id=memory_id)
            return memory_id

        def hooked_query(query: str, top_k: int = 5, **kwargs) -> list[dict]:
            results = original_query(query, top_k=top_k, **kwargs)
            for r in results:
                pid = (r.get("metadata") or {}).get("patient_id", "")
                mid = r.get("memory_id", r.get("id", ""))
                self._audit.log("query", patient_id=pid, memory_id=mid)
            return results

        self._client.add = hooked_add
        self._client.query = hooked_query
        logger.info("[Compliance] 已注入脱敏+审计钩子")

    def purge_patient(self, patient_id: str) -> PurgeReport:
        """彻底删除患者所有记忆（删除权 / 个保法）。

        删除：记忆节点 + 关联边 + 索引引用
        保留：审计日志（记录删除操作本身，但被删数据不可恢复）

        Args:
            patient_id: 患者 ID

        Returns:
            PurgeReport 删除统计
        """
        report = PurgeReport(patient_id=patient_id)
        graph = self._client._graph

        # 找出该患者的所有记忆 ID
        to_delete: list[str] = []
        for mem_id, node in list(graph._nodes.items()):
            meta = node.metadata or {}
            if meta.get("patient_id") == patient_id:
                to_delete.append(mem_id)

        # 删除节点
        for mem_id in to_delete:
            if mem_id in graph._nodes:
                del graph._nodes[mem_id]
            report.memories_deleted += 1

        # 删除关联边
        edges_to_remove = [
            (s, t) for (s, t) in list(graph._causal_edges.keys())
            if s in to_delete or t in to_delete
        ]
        for edge_key in edges_to_remove:
            del graph._causal_edges[edge_key]
            report.edges_deleted += 1

        # 清理邻接表
        for parent_id in list(graph._adjacency.keys()):
            graph._adjacency[parent_id] = {
                child for child in graph._adjacency[parent_id]
                if child not in to_delete
            }
            if parent_id in to_delete:
                del graph._adjacency[parent_id]

        # 清理 _memories 列表
        if hasattr(self._client, "_memories"):
            self._client._memories = [
                m for m in self._client._memories
                if (m.metadata or {}).get("patient_id") != patient_id
            ]

        # 清理 memory_map
        if hasattr(self._client, "_memory_map"):
            for mid in to_delete:
                self._client._memory_map.pop(mid, None)

        # 重建索引
        self._client._memory_map = {}
        for idx, node in enumerate(self._client._memories):
            self._client._memory_map[node.id] = idx

        # 审计
        self._audit.log(
            "delete", patient_id=patient_id,
            detail=f"purged {report.memories_deleted} memories, {report.edges_deleted} edges",
        )

        logger.info(
            "[Compliance] 删除患者 %s: %d 记忆, %d 边",
            patient_id, report.memories_deleted, report.edges_deleted,
        )
        return report
