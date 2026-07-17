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
from dataclasses import dataclass
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

    def sanitize_content(self, content: str) -> str:
        """脱敏 content 正文中的 PHI 模式（V4: 正文是最大泄露面）。

        识别并脱敏：
        - 身份证号（18位/15位）：330102199001011234 → 3301***********1234
        - 手机号（11位）：13812345678 → 138****5678
        - 邮箱：test@example.com → t***@example.com
        - 银行卡号（16-19位连续数字）：622202xxxxxxxxxxxx → 6222************xxxx
        """
        if not content or self._level == "remove":
            # remove 级别不处理 content（无法删除正文，只能标记）
            return content

        result = content

        # 身份证号（18位：前6位地区+8位生日+3位顺序+1位校验）
        result = re.sub(
            r"[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]",
            lambda m: mask_id_card(m.group()) if self._level == "mask" else hash_value(m.group()),
            result,
        )
        # 手机号（1开头的11位数字，前后非数字边界）
        result = re.sub(
            r"(?<!\d)1[3-9]\d{9}(?!\d)",
            lambda m: mask_phone(m.group()) if self._level == "mask" else hash_value(m.group()),
            result,
        )
        # 邮箱
        result = re.sub(
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            lambda m: mask_email(m.group()) if self._level == "mask" else hash_value(m.group()),
            result,
        )
        # 银行卡号（16-19位连续数字）
        result = re.sub(
            r"(?<!\d)\d{16,19}(?!\d)",
            lambda m: (m.group()[:4] + "*" * (len(m.group()) - 8) + m.group()[-4:])
            if self._level == "mask" else hash_value(m.group()),
            result,
        )

        return result


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
    # C5: 来源溯源链（审计可追溯记忆源自哪份病历/对话）
    source_type: str = ""         # order|lab_report|patient|ai_inferred|imported
    source_id: str = ""           # 原始记录 ID


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
        source_type: str = "",
        source_id: str = "",
    ) -> None:
        """记录一条审计日志（含来源溯源链）。

        V5: patient_id / source_id 若含 PHI 模式（身份证/手机号）自动脱敏，
        审计日志不再成为 PHI 明文二次泄露点。
        """
        # V5: 审计字段脱敏（patient_id/source_id 可能是身份证/病历号）
        sanitizer = PHISanitizer(level="mask")
        safe_pid = sanitizer.sanitize_content(patient_id) if patient_id else patient_id
        safe_sid = sanitizer.sanitize_content(source_id) if source_id else source_id
        entry = AuditEntry(
            timestamp=time.time(),
            actor=self._actor,
            action=action,
            patient_id=safe_pid,
            memory_id=memory_id,
            detail=detail,
            source_type=source_type,
            source_id=safe_sid,
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
                        "source_type": entry.source_type,
                        "source_id": entry.source_id,
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

        def hooked_add(content: str, metadata: dict | None = None, **kwargs) -> str:
            sanitized = self._sanitizer.sanitize(metadata) or {}
            # V4: content 正文 PHI 脱敏（正文是最大泄露面）
            sanitized_content = self._sanitizer.sanitize_content(content)
            memory_id = original_add(sanitized_content, metadata=sanitized, **kwargs)
            patient_id = (sanitized or {}).get("patient_id", "")
            # C5: 审计日志记录来源链（从 kwargs 透传）
            self._audit.log(
                "add",
                patient_id=patient_id,
                memory_id=memory_id,
                source_type=kwargs.get("source_type", ""),
                source_id=kwargs.get("source_id", ""),
            )
            return memory_id

        def hooked_query(query: str, top_k: int = 5, **kwargs) -> list[dict]:
            results = original_query(query, top_k=top_k, **kwargs)
            for r in results:
                pid = (r.get("metadata") or {}).get("patient_id", "")
                mid = r.get("memory_id", r.get("id", ""))
                self._audit.log("query", patient_id=pid, memory_id=mid)
            return results

        self._client.add = hooked_add  # type: ignore[method-assign,assignment]
        self._client.query = hooked_query  # type: ignore[method-assign,assignment]
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
        if graph is None:
            logger.warning("[Compliance] graph 未启用，无法执行删除")
            return report

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

        # V6: 清理向量/倒排/时空/缓存索引（否则删除后仍可被检索，删除权失效）
        # 1. 倒排索引 _index（keyword → memory_ids）：移除已删记忆的引用
        if hasattr(self._client, "_index"):
            for kw in list(self._client._index.keys()):
                self._client._index[kw] = {
                    mid for mid in self._client._index[kw] if mid not in to_delete
                }
                if not self._client._index[kw]:
                    del self._client._index[kw]
        # 2. FAISS 向量索引：标记重建（下次 query 自动重建，_id_faiss_map 清失效项）
        if hasattr(self._client, "_id_faiss_map"):
            for mid in to_delete:
                self._client._id_faiss_map.pop(mid, None)
            # 若 map 清空或大幅变动，强制重建（置 None 触发懒重建）
            if hasattr(self._client, "_faiss_index"):
                self._client._faiss_index = None
                logger.info("[Compliance] FAISS 索引标记重建（删除权保证）")
        # 3. 时空索引：清理已删节点
        if hasattr(self._client, "_spacetime") and self._client._spacetime is not None:
            try:
                st = self._client._spacetime
                for mid in to_delete:
                    # spacetime 内部 time_index 的 nodes
                    if hasattr(st, "time_index") and hasattr(st.time_index, "nodes"):
                        st.time_index.nodes.pop(mid, None)
            except Exception as e:
                logger.debug("[Compliance] 时空索引清理降级: %s", e)
        # 4. 查询缓存清空（避免返回已删记忆的缓存副本）
        if hasattr(self._client, "_query_cache"):
            self._client._query_cache.clear()

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
