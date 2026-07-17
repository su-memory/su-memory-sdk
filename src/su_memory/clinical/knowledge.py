"""
knowledge — 医疗领域知识层

提供药物-营养交互查询、检验参考值、过敏原-制剂禁忌映射。
与 P1-S1 的 association_kb 互补：association_kb 负责注入检索关联边，
本模块提供独立的知识查询接口。

⚠️ 项目区隔：知识查询服务，不做因果推断（后者归 World Model）。

Example:
  >>> from su_memory.clinical import MedicalKnowledgeBase
  >>> kb = MedicalKnowledgeBase()
  >>> interactions = kb.check_drug_interaction(["华法林", "二甲双胍"])
  >>> ref = kb.get_lab_reference("白蛋白")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════


@dataclass(eq=False)
class DrugInteraction:
    """药物-营养交互记录"""
    drug_name: str
    nutrient: str
    interaction_type: str   # antagonism/absorption/metabolism/excretion/synergy
    severity: str           # major/moderate/minor
    mechanism: str = ""
    clinical_advice: str = ""
    dietary_adjustment: str = ""
    evidence_level: str = "B"


@dataclass(eq=False)
class LabReference:
    """检验参考值"""
    name: str
    unit: str
    low: float
    high: float
    critical_low: float | None = None
    critical_high: float | None = None
    abnormal_notes: str = ""


@dataclass(eq=False)
class AllergyEntry:
    """过敏原-禁忌条目"""
    allergen: str
    contraindicated_substances: list[str]
    severity: str = "major"
    alternative: str = ""


# ═══════════════════════════════════════════════════════════════════
# 种子数据
# ═══════════════════════════════════════════════════════════════════


def _seed_drug_interactions() -> list[DrugInteraction]:
    return [
        DrugInteraction("华法林", "维生素K", "antagonism", "major",
            "华法林拮抗维生素K依赖的凝血因子",
            "限制深色蔬菜(菠菜/西兰花)，保持维K摄入稳定",
            "每日维K摄入≤120μg", "A"),
        DrugInteraction("甲氨蝶呤", "叶酸", "antagonism", "major",
            "MTX抑制二氢叶酸还原酶",
            "补充叶酸，减少黏膜炎风险",
            "MTX后24h补充叶酸1mg/d", "A"),
        DrugInteraction("二甲双胍", "维生素B12", "absorption", "moderate",
            "影响回肠末端B12吸收",
            "长期使用监测B12水平",
            "定期检测B12，必要时肌注补充", "B"),
        DrugInteraction("卡托普利", "钾", "excretion", "major",
            "ACE抑制剂减少醛固酮→减少钾排泄",
            "避免高钾食物(香蕉/橙子/土豆)和代盐",
            "监测血钾", "A"),
        DrugInteraction("环丙沙星", "钙", "absorption", "moderate",
            "喹诺酮与多价阳离子螯合",
            "服药前后2h避免乳制品/钙片/铁剂",
            "错开服用时间", "B"),
        DrugInteraction("地高辛", "膳食纤维", "absorption", "moderate",
            "高纤维影响地高辛吸收",
            "保持膳食纤维摄入稳定，勿突然增减",
            "稳定纤维摄入", "B"),
        DrugInteraction("左旋甲状腺素", "钙", "absorption", "moderate",
            "钙与甲状腺素结合影响吸收",
            "空腹服用，与钙/铁间隔4h",
            "晨起空腹服药", "B"),
        DrugInteraction("呋塞米", "钾", "excretion", "major",
            "袢利尿剂增加钾/镁排泄",
            "补充富钾食物，监测电解质",
            "监测血钾血镁", "A"),
    ]


def _seed_lab_references() -> list[LabReference]:
    return [
        LabReference("白蛋白", "g/L", 35, 55, 30, None,
            "低白蛋白提示营养不良/肝病/肾病综合征"),
        LabReference("前白蛋白", "mg/L", 200, 400, 100, None,
            "半衰期短，反映近期营养状况"),
        LabReference("血红蛋白", "g/L", 120, 160, 70, None,
            "贫血分级：<90中度 <70重度（男）"),
        LabReference("钾", "mmol/L", 3.5, 5.5, 3.0, 6.0,
            "高钾>6.0危及生命，低钾<3.0需紧急补钾"),
        LabReference("钠", "mmol/L", 135, 145, 125, 155,
            "低钠<135需限水，高钠>145需补水"),
        LabReference("血糖", "mmol/L", 3.9, 6.1, 2.8, 16.7,
            "空腹血糖；>7.0糖尿病，<3.9低血糖"),
        LabReference("肌酐", "μmol/L", 44, 133, None, None,
            "反映肾功能；eGFR更准确"),
        LabReference("C反应蛋白", "mg/L", 0, 10, None, 100,
            "炎症标志；>100提示严重感染/炎症"),
        LabReference("转铁蛋白", "g/L", 2.0, 3.6, 1.5, None,
            "营养+铁代谢指标"),
        LabReference("BMI", "kg/m²", 18.5, 24.0, 17.0, 32.0,
            "<18.5偏瘦 <17严重营养不良 >28肥胖"),
    ]


def _seed_allergies() -> list[AllergyEntry]:
    return [
        AllergyEntry("花生", ["花生蛋白", "花生油", "花生酱", "arachis_oil"],
            "major", "改用其他植物油/蛋白来源"),
        AllergyEntry("牛奶", ["乳糖", "乳清蛋白", "酪蛋白", "whey"],
            "moderate", "选用无乳糖配方或植物蛋白"),
        AllergyEntry("鸡蛋", ["卵白蛋白", "蛋清", "ovalbumin", "lysozyme"],
            "moderate", "选用无蛋配方"),
        AllergyEntry("麸质", ["小麦蛋白", "大麦", "黑麦", "gluten", "麦胶"],
            "moderate", "选用无麸质配方"),
        AllergyEntry("大豆", ["大豆蛋白", "豆粕", "soy_protein", "大豆卵磷脂"],
            "moderate", "选用其他植物蛋白"),
        AllergyEntry("鱼", ["鱼精蛋白", "鱼油", "omega3_fish"],
            "moderate", "改用亚麻籽油(植物omega3)"),
    ]


# ═══════════════════════════════════════════════════════════════════
# 医疗知识库
# ═══════════════════════════════════════════════════════════════════


def _drug_interaction_from_dict(d: dict) -> DrugInteraction:
    """从 dict 构造 DrugInteraction。"""
    return DrugInteraction(
        drug_name=d.get("drug_name", ""),
        nutrient=d.get("nutrient", ""),
        interaction_type=d.get("interaction_type", "antagonism"),
        severity=d.get("severity", "moderate"),
        mechanism=d.get("mechanism", ""),
        clinical_advice=d.get("clinical_advice", ""),
        dietary_adjustment=d.get("dietary_adjustment", ""),
        evidence_level=d.get("evidence_level", "B"),
    )


def _lab_reference_from_dict(d: dict) -> LabReference:
    """从 dict 构造 LabReference。"""
    cl = d.get("critical_low")
    ch = d.get("critical_high")
    return LabReference(
        name=d.get("name", ""),
        unit=d.get("unit", ""),
        low=float(d.get("low", 0)),
        high=float(d.get("high", 0)),
        critical_low=float(cl) if cl is not None else None,
        critical_high=float(ch) if ch is not None else None,
        abnormal_notes=d.get("abnormal_notes", d.get("notes", "")),
    )


def _allergy_from_dict(d: dict) -> AllergyEntry:
    """从 dict 构造 AllergyEntry。"""
    return AllergyEntry(
        allergen=d.get("allergen", ""),
        contraindicated_substances=list(d.get("contraindicated_substances", [])),
        severity=d.get("severity", "major"),
        alternative=d.get("alternative", ""),
    )


class MedicalKnowledgeBase:
    """医疗领域知识库 — 药物交互/检验参考/过敏禁忌。

    用法：
        kb = MedicalKnowledgeBase()
        kb.check_drug_interaction(["华法林"])
        kb.get_lab_reference("白蛋白")
        kb.check_allergy("花生")
    """

    def __init__(self):
        self._drug_interactions = _seed_drug_interactions()
        self._lab_references: dict[str, LabReference] = {
            lr.name: lr for lr in _seed_lab_references()
        }
        self._allergies: dict[str, AllergyEntry] = {
            ae.allergen: ae for ae in _seed_allergies()
        }

    @classmethod
    def load_from_file(cls, path: str) -> MedicalKnowledgeBase:
        """从外部 JSON 文件加载知识库数据。

        JSON 格式（三类数据，键均可缺省，缺省则用种子数据填充）：
            {
              "drug_interactions": [ {drug_name, nutrient, interaction_type, ...} ],
              "lab_references":    [ {name, unit, low, high, ...} ],
              "allergies":         [ {allergen, contraindicated_substances, ...} ]
            }

        Example JSON (药物交互):
            {
              "drug_interactions": [
                {
                  "drug_name": "华法林",
                  "nutrient": "维生素K",
                  "interaction_type": "antagonism",
                  "severity": "major",
                  "mechanism": "竞争性抑制维生素K环氧化物还原酶",
                  "clinical_advice": "监测INR，稳定摄入维生素K"
                }
              ]
            }
        """
        import json
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        kb = cls()  # 先用种子数据初始化

        if "drug_interactions" in data:
            kb._drug_interactions = [
                _drug_interaction_from_dict(d)
                for d in data["drug_interactions"]
            ]
        if "lab_references" in data:
            kb._lab_references = {
                lr.name: lr
                for lr in (_lab_reference_from_dict(d) for d in data["lab_references"])
            }
        if "allergies" in data:
            kb._allergies = {
                ae.allergen: ae
                for ae in (_allergy_from_dict(d) for d in data["allergies"])
            }
        return kb

    def add_from_file(self, path: str) -> dict[str, int]:
        """从 JSON 文件追加知识（不替换种子数据），返回各类追加数量。"""
        import json
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        counts = {"drug_interactions": 0, "lab_references": 0, "allergies": 0}
        for d in data.get("drug_interactions", []):
            self._drug_interactions.append(_drug_interaction_from_dict(d))
            counts["drug_interactions"] += 1
        for d in data.get("lab_references", []):
            lr = _lab_reference_from_dict(d)
            self._lab_references[lr.name] = lr
            counts["lab_references"] += 1
        for d in data.get("allergies", []):
            ae = _allergy_from_dict(d)
            self._allergies[ae.allergen] = ae
            counts["allergies"] += 1
        return counts

    # ── 药物-营养交互 ──────────────────────────────────────

    def check_drug_interaction(
        self, drug_list: list[str]
    ) -> list[DrugInteraction]:
        """查询药物列表的所有营养交互。

        Args:
            drug_list: 药物名称列表

        Returns:
            匹配的交互记录列表
        """
        drug_lower = [d.lower() for d in drug_list]
        results: list[DrugInteraction] = []

        for interaction in self._drug_interactions:
            for drug in drug_lower:
                if drug in interaction.drug_name.lower():
                    results.append(interaction)
                    break

        return results

    def get_contraindicated_nutrients(
        self, drug_list: list[str]
    ) -> dict[str, list[str]]:
        """获取药物列表的所有禁忌营养素。

        Returns:
            {drug_name: [nutrient1, nutrient2, ...], ...}
        """
        interactions = self.check_drug_interaction(drug_list)
        result: dict[str, list[str]] = {}
        for ia in interactions:
            result.setdefault(ia.drug_name, []).append(ia.nutrient)
        return result

    # ── 检验参考值 ────────────────────────────────────────

    def get_lab_reference(self, lab_name: str) -> LabReference | None:
        """获取检验项目参考值"""
        return self._lab_references.get(lab_name)

    def is_abnormal(self, lab_name: str, value: float) -> bool:
        """判断检验值是否异常"""
        ref = self._lab_references.get(lab_name)
        if ref is None:
            return False
        return value < ref.low or value > ref.high

    def is_critical(self, lab_name: str, value: float) -> bool:
        """判断检验值是否危急值"""
        ref = self._lab_references.get(lab_name)
        if ref is None:
            return False
        if ref.critical_low is not None and value <= ref.critical_low:
            return True
        if ref.critical_high is not None and value >= ref.critical_high:
            return True
        return False

    def find_abnormal_from_dict(
        self, labs: dict[str, float]
    ) -> list[dict[str, Any]]:
        """从检验值字典中找出异常项。

        Args:
            labs: {lab_name: value}

        Returns:
            [{"name", "value", "reference", "abnormal", "critical"}, ...]
        """
        results: list[dict[str, Any]] = []
        for name, value in labs.items():
            ref = self._lab_references.get(name)
            if ref is None:
                continue
            abnormal = self.is_abnormal(name, value)
            critical = self.is_critical(name, value)
            if abnormal or critical:
                results.append({
                    "name": name,
                    "value": value,
                    "unit": ref.unit,
                    "reference": f"{ref.low}-{ref.high}",
                    "abnormal": abnormal,
                    "critical": critical,
                    "notes": ref.abnormal_notes,
                })
        return results

    # ── 过敏-禁忌 ─────────────────────────────────────────

    def check_allergy(self, allergen: str) -> AllergyEntry | None:
        """查询过敏原"""
        return self._allergies.get(allergen)

    def check_substance_allergy(
        self, substances: list[str]
    ) -> list[AllergyEntry]:
        """检查物质列表中是否有已知过敏原。

        Args:
            substances: 待检查物质列表

        Returns:
            匹配的过敏条目列表
        """
        results: list[AllergyEntry] = []
        sub_lower = [s.lower() for s in substances]

        for entry in self._allergies.values():
            for contra in entry.contraindicated_substances:
                if any(c in s for c in [contra.lower()] for s in sub_lower):
                    results.append(entry)
                    break

        return results

    # ── 统计 ──────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        """知识库统计"""
        return {
            "drug_interactions": len(self._drug_interactions),
            "lab_references": len(self._lab_references),
            "allergies": len(self._allergies),
        }
