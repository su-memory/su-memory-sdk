"""
su_memory.clinical — 医疗智能体记忆引擎临床层

将医疗领域知识注入 su-memory 检索引擎，增强医疗场景的语义检索关联性。

⚠️ 项目区隔：本层仅增强检索关联，不做因果推断。
因果推断（do-calculus/反事实）由 MCI World Model 负责。

模块：
  - association_kb: 医疗关联知识库（药物-营养/缺乏-症状/过敏-禁忌/疾病-营养）
"""

from su_memory.clinical.association_kb import (
    AssociationRule,
    AssociationType,
    MedicalAssociationKB,
)

__all__ = [
    "AssociationRule",
    "AssociationType",
    "MedicalAssociationKB",
]
