"""
extractor — 临床记忆抽取层（C2: 记忆信噪比提升）

入库前把长原文压缩为结构化要点 + 原文引用，提升信噪比、降低 token 成本。

两层抽取策略：
  1. 规则抽取（默认，零依赖）：正则 + 字典提取药名/剂量/检验值/诊断/过敏
  2. LLM 抽取（opt-in）：有 LLM 时用，无则降级规则

⚠️ 项目区隔：抽取是「信息压缩」，不是「医学推理」或「因果推断」。
   只做事实提取与归一化，不做诊断推理（归 MCI World Model）。
   事实保真由规则 + 原文引用保证。

Example:
  >>> from su_memory.clinical.extractor import ClinicalMemoryExtractor
  >>> ext = ClinicalMemoryExtractor()
  >>> fact = ext.extract("患者服用华法林5mg每日一次，白蛋白32g/L偏低")
  >>> print(fact.summary)        # 结构化摘要
  >>> print(fact.entities)       # 提取的实体
  >>> print(fact.original)       # 原文引用
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """抽取出的单个实体"""
    entity_type: str        # drug|dose|lab_value|diagnosis|allergy|nutrition
    name: str               # 实体名
    value: str = ""         # 值（剂量/检验值）
    unit: str = ""          # 单位


@dataclass
class ExtractedFact:
    """抽取结果"""
    summary: str                                    # 结构化摘要
    entities: list[ExtractedEntity] = field(default_factory=list)
    original: str = ""                              # 原文引用
    compression_ratio: float = 1.0                  # 压缩比（原文字数/摘要字数）
    confidence: float = 1.0                         # 抽取置信度
    method: str = "rule"                            # rule|llm


# 常见药物清单（与 knowledge.py 种子对齐 + 扩展）
_DRUG_NAMES = [
    "华法林", "二甲双胍", "甲氨蝶呤", "地高辛", "卡托普利",
    "呋塞米", "环丙沙星", "左旋甲状腺素", "胰岛素", "阿司匹林",
    "氯吡格雷", "氨氯地平", "美托洛尔", "奥美拉唑", "头孢曲松",
]

# 常见检验项目
_LAB_NAMES = [
    "白蛋白", "前白蛋白", "血红蛋白", "血糖", "肌酐", "钾", "钠",
    "转铁蛋白", "C反应蛋白", "BMI", "总蛋白", "尿酸", "胆固醇",
]

# 常见诊断/状况
_DIAGNOSIS_KEYWORDS = [
    "营养不良", "糖尿病", "高血压", "贫血", "低蛋白血症",
    "骨质疏松", "脱水", "肥胖", "消瘦", "水肿",
]


class ClinicalMemoryExtractor:
    """临床记忆抽取器。

    Args:
        use_llm: 是否启用 LLM 抽取（需配置 LLM 后端，默认关闭走规则）
    """

    def __init__(self, use_llm: bool = False):
        self._use_llm = use_llm
        self._drug_pattern = re.compile(
            r"(" + "|".join(_DRUG_NAMES) + r")"
            r"(?:\s*(\d+(?:\.\d+)?)\s*(mg|mg/d|g|ml|μg|ug|单位))?"
        )
        self._lab_pattern = re.compile(
            r"(" + "|".join(_LAB_NAMES) + r")"
            r"\s*[:：]?\s*(\d+(?:\.\d+)?)\s*"
            r"(g/L|mg/L|mmol/L|μmol/L|kg/m²|%|mg/dL|U/L)?"
        )

    def extract(self, content: str) -> ExtractedFact:
        """抽取临床要点。

        Args:
            content: 原始记忆文本

        Returns:
            ExtractedFact 含摘要、实体、原文引用、压缩比
        """
        if not content or not content.strip():
            return ExtractedFact(summary="", original=content or "")

        if self._use_llm:
            result = self._extract_with_llm(content)
            if result is not None:
                return result
            # LLM 失败降级规则
            logger.debug("[Extractor] LLM 抽取降级到规则")

        return self._extract_with_rules(content)

    def _extract_with_rules(self, content: str) -> ExtractedFact:
        """规则抽取（零依赖，默认路径）。"""
        entities: list[ExtractedEntity] = []

        # 1. 药物 + 剂量
        for m in self._drug_pattern.finditer(content):
            drug_name = m.group(1)
            dose = m.group(2) or ""
            unit = m.group(3) or ""
            entities.append(ExtractedEntity(
                entity_type="drug",
                name=drug_name,
                value=dose,
                unit=unit,
            ))

        # 2. 检验值
        for m in self._lab_pattern.finditer(content):
            lab_name = m.group(1)
            value = m.group(2) or ""
            unit = m.group(3) or ""
            entities.append(ExtractedEntity(
                entity_type="lab_value",
                name=lab_name,
                value=value,
                unit=unit,
            ))

        # 3. 诊断关键词
        for diag in _DIAGNOSIS_KEYWORDS:
            if diag in content:
                entities.append(ExtractedEntity(
                    entity_type="diagnosis", name=diag
                ))

        # 4. 过敏（双向匹配："花生过敏" 或 "过敏:花生" 或 "对花生过敏"）
        allergy_patterns = [
            r"([^\s,，。；:：对]{1,6})过敏",       # 花生过敏 / 对花生过敏
            r"过敏[：: ]*([^\s,，。；;]+)",         # 过敏:花生
        ]
        found_allergens: set[str] = set()
        for pat in allergy_patterns:
            for m in re.finditer(pat, content):
                allergen = m.group(1).strip()
                if allergen and allergen not in ("对", "的", "了"):
                    found_allergens.add(allergen)
        for allergen in found_allergens:
            entities.append(ExtractedEntity(
                entity_type="allergy", name=allergen
            ))

        # 5. 构建摘要
        summary_parts: list[str] = []
        drug_entities = [e for e in entities if e.entity_type == "drug"]
        lab_entities = [e for e in entities if e.entity_type == "lab_value"]
        diag_entities = [e for e in entities if e.entity_type == "diagnosis"]
        allergy_entities = [e for e in entities if e.entity_type == "allergy"]

        if drug_entities:
            drug_str = "; ".join(
                f"{e.name} {e.value}{e.unit}".strip()
                for e in drug_entities
            )
            summary_parts.append(f"药物: {drug_str}")
        if lab_entities:
            lab_str = "; ".join(
                f"{e.name}={e.value}{e.unit}".strip()
                for e in lab_entities
            )
            summary_parts.append(f"检验: {lab_str}")
        if diag_entities:
            summary_parts.append(
                "诊断: " + "; ".join(e.name for e in diag_entities)
            )
        if allergy_entities:
            summary_parts.append(
                "过敏: " + "; ".join(e.name for e in allergy_entities)
            )

        summary = " | ".join(summary_parts) if summary_parts else content

        # 压缩比（避免除零）
        orig_len = len(content)
        summary_len = max(len(summary), 1)
        ratio = orig_len / summary_len if summary_len else 1.0

        return ExtractedFact(
            summary=summary,
            entities=entities,
            original=content,
            compression_ratio=round(ratio, 2),
            confidence=0.9 if entities else 0.5,  # 有实体提取则高置信
            method="rule",
        )

    def _extract_with_llm(self, content: str) -> ExtractedFact | None:
        """LLM 抽取（opt-in，需配置后端）。

        当前预留接口，未配置 LLM 时返回 None 降级规则。
        """
        if not self._use_llm:
            return None
        try:
            # 尝试导入 LLM（与 lite_pro 的能量推断 LLM 复用）
            import os
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                return None
            # 实际 LLM 调用预留——保持架构开放，当前返回 None 走规则降级
            # 生产环境可在此接入 DeepSeek/其他 LLM 做结构化抽取
            return None
        except Exception as e:
            logger.debug("[Extractor] LLM 抽取异常: %s", e)
            return None
