"""
safety_gate — 检索结果风险门控（C3: 医疗级安全关键）

在 recall() 返回前，对每条召回记忆做风险校验：
  1. 从记忆 content 提取药名（复用 MedicalKnowledgeBase 的药物清单）
  2. 调 check_drug_interaction 查药物-营养交互
  3. 调 check_allergy 查过敏冲突
  4. 给每条记忆打 risk_flags + risk_level (safe|caution|contraindicated)
  5. 默认策略：contraindicated 标记并附告警（可配置为拦截）

⚠️ 项目区隔：风险门控只做「禁忌标记/拦截」，不做「因果风险预测」。
   因果推断（do-calculus/反事实）由 MCI World Model 负责。
   本模块只回答「这条记忆是否触及已知禁忌」，不回答「会导致什么后果」。

Example:
  >>> from su_memory.clinical import MedicalKnowledgeBase
  >>> from su_memory.clinical.safety_gate import SafetyGate
  >>> gate = SafetyGate(MedicalKnowledgeBase())
  >>> screened = gate.screen(results, patient_allergies=["花生"])
  >>> for r in screened:
  ...     print(r["risk_level"], r["risk_flags"])
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# 风险等级
RISK_SAFE = "safe"
RISK_CAUTION = "caution"
RISK_CONTRAINDICATED = "contraindicated"

# 门控策略
POLICY_MARK = "mark"            # 标记但不拦截（默认，保留信息给医生判断）
POLICY_BLOCK = "block"          # 拦截 contraindicated（高风险场景）


class SafetyGate:
    """检索结果风险门控——召回后、返回前的一道安全校验。

    Args:
        knowledge: MedicalKnowledgeBase 实例（提供药物交互/过敏查询）
        policy: 门控策略 mark|block
    """

    def __init__(self, knowledge: Any, policy: str = POLICY_MARK):
        self._kb = knowledge
        self._policy = policy
        # 缓存药物清单（避免每次扫全量种子）
        self._drug_names: list[str] = self._extract_drug_names()
        # 缓存过敏原清单
        self._allergens: list[str] = self._extract_allergens()

    @staticmethod
    def _normalize_text(text: str) -> str:
        """V2: 文本归一化——去空格/制表符，便于对抗子串绕过。

        "华 法 林" → "华法林"，防止用空格拆药名绕过禁忌检测。
        """
        if not text:
            return ""
        # 去除所有空白字符（含全角空格、零宽字符）
        import re
        return re.sub(r"[\s　​‌‍]+", "", text)

    def _extract_drug_names(self) -> list[str]:
        """从知识库提取所有药名（用于 content 扫描）。"""
        try:
            interactions = self._kb._drug_interactions
            names = list({di.drug_name for di in interactions})
            return names
        except Exception as e:
            logger.debug("药名提取降级: %s", e)
            return []

    def _extract_allergens(self) -> list[str]:
        """从知识库提取所有过敏原。"""
        try:
            return list(self._kb._allergies.keys())
        except Exception as e:
            logger.debug("过敏原提取降级: %s", e)
            return []

    def screen(
        self,
        results: list[dict[str, Any]],
        patient_allergies: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """对检索结果逐一做风险校验。

        Args:
            results: recall 返回的记忆列表（每条含 content）
            patient_allergies: 患者已知过敏原（增强过敏冲突检测）

        Returns:
            门控后的结果列表。每条新增字段：
              - risk_level: safe|caution|contraindicated
              - risk_flags: list[str]（命中的风险描述）
              - risk_interactions: list[dict]（命中的药物交互详情）

            若 policy=block，contraindicated 的记忆被移除。
        """
        if not results:
            return results

        patient_allergies = patient_allergies or []
        screened: list[dict[str, Any]] = []

        for r in results:
            content = r.get("content", "") or ""
            flags: list[str] = []
            interactions_hit: list[dict] = []
            level = RISK_SAFE

            # 1. 药物-营养交互检测（V2: 归一化后匹配，防空格绕过）
            norm_content = self._normalize_text(content)
            drugs_in_content = [
                d for d in self._drug_names if d in norm_content
            ]
            if drugs_in_content:
                try:
                    interactions = self._kb.check_drug_interaction(drugs_in_content)
                    for inter in interactions:
                        if inter.severity == "major":
                            interactions_hit.append({
                                "drug": inter.drug_name,
                                "nutrient": inter.nutrient,
                                "severity": inter.severity,
                                "advice": inter.clinical_advice,
                            })
                            flags.append(
                                f"重大交互: {inter.drug_name} × {inter.nutrient}"
                            )
                            if level != RISK_CONTRAINDICATED:
                                level = RISK_CONTRAINDICATED
                        elif inter.severity == "moderate":
                            interactions_hit.append({
                                "drug": inter.drug_name,
                                "nutrient": inter.nutrient,
                                "severity": inter.severity,
                                "advice": inter.clinical_advice,
                            })
                            flags.append(
                                f"中度交互: {inter.drug_name} × {inter.nutrient}"
                            )
                            if level == RISK_SAFE:
                                level = RISK_CAUTION
                except Exception as e:
                    logger.debug("药物交互校验降级: %s", e)

            # 2. 过敏冲突检测（患者已知过敏原 vs 记忆内容）
            for allergen in patient_allergies:
                entry = None
                try:
                    entry = self._kb.check_allergy(allergen)
                except Exception:
                    entry = None
                if entry:
                    # 记忆里是否提到禁忌物质（V2: 归一化匹配）
                    for substance in entry.contraindicated_substances:
                        if substance in norm_content:
                            flags.append(
                                f"过敏禁忌: 患者({allergen})忌 {substance}"
                            )
                            level = RISK_CONTRAINDICATED
                            break

            # 3. 写入风险标记
            r["risk_level"] = level
            r["risk_flags"] = flags
            r["risk_interactions"] = interactions_hit

            # 4. 策略执行
            # V10: block 策略下——contraindicated 整条拦截；caution 剥除临床建议
            # （用药剂量/方案文本不应直接喂给下游 LLM，但记忆本身对医生可见）
            if self._policy == POLICY_BLOCK and level == RISK_CONTRAINDICATED:
                logger.info(
                    "[SafetyGate] 拦截禁忌记忆: %s (flags=%s)",
                    r.get("memory_id", ""), flags,
                )
                continue
            if self._policy == POLICY_BLOCK and level == RISK_CAUTION:
                # 剥除临床建议字段，保留交互元数据（drug/nutrient/severity）
                for inter in interactions_hit:
                    inter.pop("advice", None)
                r["risk_interactions"] = interactions_hit
                logger.debug(
                    "[SafetyGate] caution 级剥除临床建议: %s",
                    r.get("memory_id", ""),
                )
            screened.append(r)

        blocked = len(results) - len(screened)
        if blocked > 0:
            logger.info("[SafetyGate] 拦截 %d 条禁忌记忆", blocked)

        return screened

    def stats(self) -> dict[str, int | str]:
        """门控器状态。"""
        return {
            "drug_names_tracked": len(self._drug_names),
            "allergens_tracked": len(self._allergens),
            "policy": self._policy,
        }
