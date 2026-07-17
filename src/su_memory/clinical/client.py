"""
client — 共享临床适配层

整合 Phase 1 + Phase 2 所有能力的统一入口。
其他医疗项目一行 import 即可用，零适配代码重复。

Example:
  >>> from su_memory.clinical import ClinicalMemoryClient
  >>> client = ClinicalMemoryClient(
  ...     embedding_backend="none",      # 医院内网
  ...     compliance_level="mask",       # PHI 自动脱敏
  ... )
  >>> client.add_patient_event("P001", "高蛋白方案1800kcal", "plan")
  >>> hits = client.recall("P001", "营养方案")
  >>> trend = client.get_lab_trend("P001", "白蛋白")
  >>> interactions = client.check_drug_interaction(["华法林"])
  >>> client.purge_patient("P001")      # 删除权
"""

from __future__ import annotations

import logging
from typing import Any

from su_memory.clinical.association_kb import MedicalAssociationKB
from su_memory.clinical.compliance import (
    AuditLogger,
    ComplianceManager,
    PurgeReport,
)
from su_memory.clinical.confidence import ConfidenceTracker
from su_memory.clinical.feedback_trainer import FeedbackTrainer
from su_memory.clinical.knowledge import (
    DrugInteraction,
    LabReference,
    MedicalKnowledgeBase,
)
from su_memory.clinical.patient_profile import (
    PatientMemorySpace,
    TrendResult,
)
from su_memory.clinical.pattern_miner import ClinicalPatternMiner
from su_memory.clinical.safety_gate import POLICY_MARK, SafetyGate

logger = logging.getLogger(__name__)


class ClinicalMemoryClient:
    """医疗智能体记忆引擎 — 统一临床适配层。

    整合：
      - P1-S1 医疗关联知识库（药物/营养/过敏/疾病关联注入）
      - P1-S2 记忆置信度增强（贝叶斯可靠性评估）
      - P1-S3 反馈训练飞轮（反馈→排序优化）
      - P1-S4 临床模式提炼（群体决策模式发现）
      - P2-S1 患者纵向记忆（检验趋势/诊疗轨迹）
      - P2-S2 领域知识层（药物交互/检验参考/过敏禁忌）
      - P2-S3 合规层（PHI脱敏/审计/删除权）

    Args:
        storage_path: 存储目录
        embedding_backend: 嵌入后端（none/ollama/sentence-transformers）
        enable_association: 启用医疗关联注入（默认 True）
        enable_confidence: 启用置信度重排序（默认 True）
        compliance_level: PHI 脱敏级别（mask/hash/remove/None=关闭）
        audit_log_path: 审计日志路径（None=仅内存）
    """

    def __init__(
        self,
        storage_path: str | None = None,
        embedding_backend: str = "none",
        enable_llm_energy: bool = False,
        enable_association: bool = True,
        enable_confidence: bool = True,
        compliance_level: str | None = "mask",
        audit_log_path: str | None = None,
        safety_screen: bool = True,
        safety_policy: str = POLICY_MARK,
    ):
        import os

        from su_memory.sdk.lite_pro import SuMemoryLitePro

        # 默认在 storage_path 下持久化置信度记录和审计日志
        if storage_path and not audit_log_path:
            audit_log_path = os.path.join(storage_path, "audit.jsonl")
        confidence_persist_path = (
            os.path.join(storage_path, "confidence.json") if storage_path else None
        )

        # ── 初始化核心引擎 ──
        # storage_path 可为 None（内存模式），SuMemoryLitePro 运行时支持
        self._engine = SuMemoryLitePro(
            storage_path=storage_path,  # type: ignore[arg-type]
            embedding_backend=embedding_backend,
            enable_llm_energy=enable_llm_energy,
            autosave=True,
        )

        # ── P2-S2 知识库（无依赖，最先初始化）──
        self._knowledge = MedicalKnowledgeBase()

        # ── C3 风险门控（召回后、返回前的安全校验）──
        self._safety_gate: SafetyGate | None = (
            SafetyGate(self._knowledge, policy=safety_policy)
            if safety_screen else None
        )

        # ── P1-S1 医疗关联注入 ──
        self._assoc_kb: MedicalAssociationKB | None = None
        if enable_association:
            self._assoc_kb = MedicalAssociationKB()
            self._assoc_kb.inject_hooks(self._engine)

        # ── P2-S1 患者记忆空间 ──
        self._patient_space = PatientMemorySpace(self._engine)

        # ── P1-S2 置信度追踪（带持久化）──
        self._confidence: ConfidenceTracker | None = None
        if enable_confidence:
            self._confidence = ConfidenceTracker(
                self._engine, persist_path=confidence_persist_path
            )
            self._confidence.inject_hooks(self._engine)

        # ── P1-S3 反馈训练器 ──
        self._trainer: FeedbackTrainer | None = None
        if self._confidence:
            self._trainer = FeedbackTrainer(self._confidence)

        # ── P1-S4 模式提炼器 ──
        self._miner = ClinicalPatternMiner(self._engine, self._assoc_kb)

        # ── P2-S3 合规层（最后注入，确保脱敏在最外层）──
        self._compliance: ComplianceManager | None = None
        if compliance_level:
            self._compliance = ComplianceManager(
                self._engine,
                phi_level=compliance_level,
                audit_log_path=audit_log_path,
            )
            self._compliance.inject_hooks()

        logger.info(
            "[ClinicalClient] 初始化完成: assoc=%s conf=%s compliance=%s",
            enable_association, enable_confidence, compliance_level,
        )

    # ── 写入 ──────────────────────────────────────────────

    def add_patient_event(
        self,
        patient_id: str,
        content: str,
        event_type: str = "",
        metadata: dict | None = None,
        source_type: str = "unknown",
        source_id: str = "",
        source_confidence: float = 1.0,
    ) -> str | None:
        """写入患者事件（带来源溯源）。

        Args:
            source_type: 来源类型 order|lab_report|patient|ai_inferred|imported
            source_id: 原始记录 ID（病历号/对话ID/FHIR Resource ID）
            source_confidence: 来源可信度 [0,1]（医嘱1.0/患者自述0.6/AI推断0.4）
        """
        full_meta = {"patient_id": patient_id}
        if event_type:
            full_meta["event_type"] = event_type
        if metadata:
            full_meta.update(metadata)
        try:
            return self._engine.add(
                content,
                metadata=full_meta,
                source_type=source_type,
                source_id=source_id,
                source_confidence=source_confidence,
            )
        except Exception as e:
            logger.error("[ClinicalClient] add 异常: %s", e)
            return None

    def add_lab_value(
        self,
        patient_id: str,
        lab_name: str,
        value: float,
        unit: str = "",
        reference_range: str = "",
    ) -> str | None:
        """写入结构化检验值。"""
        try:
            return self._patient_space.add_lab_value(
                patient_id, lab_name, value, unit, reference_range
            )
        except Exception as e:
            logger.error("[ClinicalClient] add_lab_value 异常: %s", e)
            return None

    # ── 检索 ──────────────────────────────────────────────

    def recall(
        self,
        patient_id: str,
        query: str,
        top_k: int = 5,
        *,
        max_fetch: int = 500,
    ) -> list[dict[str, Any]]:
        """按患者 ID 召回相关记忆（带患者隔离）。

        采用分页倍增拉取策略：多患者共享引擎时，避免 top_k*3 固定
        窗口把稀疏患者的记忆漏掉。fetch_size 从 max(top_k*3, 15) 起，
        若该窗口内匹配目标患者的记忆不足 top_k，则 fetch_size *= 2 继续拉，
        直到凑够 top_k 或触达 max_fetch 上限。
        """
        try:
            fetch_size = max(top_k * 3, 15)
            collected: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            while fetch_size <= max_fetch:
                raw = self._engine.query(query, top_k=fetch_size)
                for r in raw:
                    rid = str(r.get("memory_id") or r.get("id") or id(r))
                    if rid in seen_ids:
                        continue
                    meta = r.get("metadata") or {}
                    if meta.get("patient_id") == patient_id:
                        collected.append(r)
                        seen_ids.add(rid)
                if len(collected) >= top_k:
                    break
                fetch_size *= 2
            result = collected[:top_k]
            # C3: 风险门控——召回后、返回前校验禁忌（零禁忌泄露）
            if self._safety_gate is not None:
                try:
                    result = self._safety_gate.screen(
                        result,
                        patient_allergies=self._patient_allergies(patient_id),
                    )
                except Exception as e:
                    logger.debug("[ClinicalClient] 风险门控降级: %s", e)
            return result
        except Exception as e:
            logger.error("[ClinicalClient] recall 异常: %s", e)
            return []

    def _patient_allergies(self, patient_id: str) -> list[str]:
        """从患者记忆提取已知过敏原（供风险门控增强过敏检测）。

        扫描 event_type=allergy 的记忆，提取过敏原。
        这是辅助方法，不做因果推断，只做已有信息的汇总。
        """
        allergies: list[str] = []
        try:
            graph = getattr(self._engine, "_graph", None)
            if graph is None:
                return allergies
            for _mid, node in getattr(graph, "_nodes", {}).items():
                meta = node.metadata or {}
                if meta.get("patient_id") != patient_id:
                    continue
                if meta.get("event_type") != "allergy":
                    continue
                allergen = meta.get("allergen") or meta.get("allergy")
                if allergen and allergen not in allergies:
                    allergies.append(allergen)
        except Exception as e:
            logger.debug("[ClinicalClient] 过敏原提取降级: %s", e)
        return allergies

    # ── 纵向记忆 ──────────────────────────────────────────

    def get_lab_trend(
        self, patient_id: str, lab_name: str
    ) -> TrendResult:
        """获取检验值趋势。"""
        return self._patient_space.get_lab_trend(patient_id, lab_name)

    def find_abnormal_labs(self, patient_id: str) -> list[dict]:
        """筛查异常检验值。"""
        return self._patient_space.find_abnormal_labs(patient_id)

    def get_care_trajectory(self, patient_id: str, limit: int = 50) -> list[dict]:
        """获取诊疗轨迹。"""
        return self._patient_space.get_care_trajectory(patient_id, limit)

    # ── 知识查询 ──────────────────────────────────────────

    def check_drug_interaction(self, drug_list: list[str]) -> list[DrugInteraction]:
        """查询药物-营养交互。"""
        return self._knowledge.check_drug_interaction(drug_list)

    def get_lab_reference(self, lab_name: str) -> LabReference | None:
        """获取检验参考值。"""
        return self._knowledge.get_lab_reference(lab_name)

    # ── 反馈训练 ──────────────────────────────────────────

    def train_from_feedback(
        self, memory_id: str, rating: int, action: str = "accept"
    ) -> float | None:
        """从反馈训练检索排序。"""
        if self._trainer:
            return self._trainer.train_from_feedback(memory_id, rating, action)
        return None

    # ── 模式提炼 ──────────────────────────────────────────

    def mine_patterns(self, min_support: int = 3) -> list:
        """提炼临床决策模式。"""
        return self._miner.mine_patterns(min_support=min_support)

    # ── 合规 ──────────────────────────────────────────────

    def purge_patient(self, patient_id: str) -> PurgeReport | None:
        """删除患者所有记忆（删除权）。"""
        if self._compliance:
            return self._compliance.purge_patient(patient_id)
        return None

    @property
    def audit(self) -> AuditLogger | None:
        """审计日志器。"""
        return self._compliance.audit if self._compliance else None

    # ── 健康检查 ──────────────────────────────────────────

    def health_check(self) -> dict[str, Any]:
        """统一健康检查。"""
        result = self._engine.health_check()
        result["modules"] = {
            "association": self._assoc_kb is not None,
            "confidence": self._confidence is not None,
            "feedback_trainer": self._trainer is not None,
            "compliance": self._compliance is not None,
            "knowledge": True,
        }
        return result

    def close(self) -> None:
        """关闭客户端，持久化置信度记录等状态。"""
        if self._confidence:
            self._confidence.save()
        if hasattr(self._engine, "close"):
            self._engine.close()
        elif hasattr(self._engine, "flush"):
            self._engine.flush()
        logger.info("[ClinicalClient] 已关闭并持久化状态")
