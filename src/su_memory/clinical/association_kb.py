"""
medical association_kb — 医疗关联知识库

将医疗领域知识（药物-营养交互、营养缺乏-临床表现、过敏-禁忌）注入 su-memory
检索关联图，增强语义检索的关联性。

⚠️ 项目区隔声明：
  本模块注入的是**检索关联边**（"含华法林的记忆"与"含维生素K的记忆"关联更强），
  不是因果推断。因果推断（do-calculus/反事实）由 MCI World Model 负责。

工作原理：
  1. 预定义医疗关联三元组（source_pattern → relation → target_pattern）
  2. add() 写入新记忆时，扫描内容是否命中关联模式
  3. 命中则在两条记忆间自动创建 explicit 级别的关联边（confidence=0.90）
  4. query_multihop 时，关联边让相关记忆在多跳检索中互相可达

Example:
  >>> from su_memory.clinical.association_kb import MedicalAssociationKB
  >>> kb = MedicalAssociationKB()
  >>> kb.inject_hooks(memory_client)  # 注入 add() 钩子
  >>> memory_client.add("患者服用华法林抗凝")
  >>> memory_client.add("建议限制深色蔬菜摄入，因含维生素K")
  >>> # 两条记忆自动建立关联边，query_multihop 可互相可达
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from su_memory.sdk.lite_pro import SuMemoryLitePro

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 关联类型枚举
# ═══════════════════════════════════════════════════════════════════


class AssociationType(str, Enum):
    """医疗关联类型（用于检索关联图，非因果推断）"""

    DRUG_NUTRIENT = "drug_nutrient"          # 药物-营养素交互
    DEFICIENCY_SYMPTOM = "deficiency_symptom"  # 营养缺乏-临床表现
    ALLERGY_CONTRAINDICATION = "allergy_contra"  # 过敏-禁忌
    DISEASE_NUTRITION = "disease_nutrition"   # 疾病-营养方案
    LAB_NUTRITION = "lab_nutrition"           # 检验值-营养干预


# ═══════════════════════════════════════════════════════════════════
# 关联规则数据结构
# ═══════════════════════════════════════════════════════════════════


@dataclass(eq=False)
class AssociationRule:
    """一条医疗关联规则。

    当记忆内容命中 source_patterns 且另一条记忆命中 target_patterns 时，
    在两条记忆间创建关联边。

    Attributes:
        rule_id: 规则唯一标识
        assoc_type: 关联类型
        source_patterns: 源端关键词列表（任一命中即匹配）
        target_patterns: 目标端关键词列表（任一命中即匹配）
        relation_desc: 关联描述（用于 explain_multihop 输出）
        confidence: 关联置信度 [0,1]（注入为 explicit 级边）
        bidirectional: 是否双向关联
    """

    rule_id: str
    assoc_type: AssociationType
    source_patterns: list[str]
    target_patterns: list[str]
    relation_desc: str
    confidence: float = 0.90
    bidirectional: bool = True


# ═══════════════════════════════════════════════════════════════════
# 种子关联规则（临床验证过的医疗知识）
# ═══════════════════════════════════════════════════════════════════


def _seed_rules() -> list[AssociationRule]:
    """生成种子关联规则集。

    来源：营养系统 models/knowledge.py 的 DrugNutrientInteraction 表 +
    临床营养学教材（ESPEN/ASPEN 指南）。
    """
    rules: list[AssociationRule] = []

    # ── 药物-营养交互（8 条核心规则）──────────────────────
    drug_nutrient_pairs = [
        ("warfarin", ["华法林", "warfarin", "香豆素"],
         ["维生素K", "vitamin_k", "深色蔬菜", "菠菜", "西兰花"],
         "华法林与维生素K拮抗，需限制深色蔬菜摄入"),
        ("methotrexate", ["甲氨蝶呤", "methotrexate", "MTX"],
         ["叶酸", "folate", "folic_acid"],
         "甲氨蝶呤干扰叶酸代谢，需补充叶酸"),
        ("metformin", ["二甲双胍", "metformin"],
         ["维生素B12", "vitamin_b12", "钴胺素"],
         "二甲双胍影响B12吸收，长期使用需监测B12"),
        ("captopril", ["卡托普利", "captopril", "ACE抑制剂", "ACEI"],
         ["钾", "potassium", "高钾"],
         "ACE抑制剂减少钾排泄，需避免高钾饮食"),
        ("ciprofloxacin", ["环丙沙星", "ciprofloxacin", "喹诺酮"],
         ["钙", "calcium", "乳制品", "铁", "iron", "锌", "zinc"],
         "喹诺酮类与钙/铁/锌结合影响吸收，服药时避开"),
        ("digoxin", ["地高辛", "digoxin"],
         ["膳食纤维", "fiber", "纤维素"],
         "膳食纤维影响地高辛吸收，需错开服用"),
        ("levothyroxine", ["左旋甲状腺素", "levothyroxine", "甲状腺素"],
         ["钙", "calcium", "铁", "iron", "大豆", "黄豆"],
         "甲状腺素与钙/铁/大豆结合影响吸收，空腹服用"),
        ("diuretic", ["利尿剂", "diuretic", "呋塞米", "furosemide"],
         ["钾", "potassium", "镁", "magnesium"],
         "利尿剂增加钾镁排泄，需补充富钾镁食物"),
    ]
    for rid, src, tgt, desc in drug_nutrient_pairs:
        rules.append(AssociationRule(
            rule_id=f"dn_{rid}",
            assoc_type=AssociationType.DRUG_NUTRIENT,
            source_patterns=src,
            target_patterns=tgt,
            relation_desc=desc,
        ))

    # ── 营养缺乏-临床表现（6 条）────────────────────────
    deficiency_pairs = [
        ("protein", ["低蛋白", "蛋白质缺乏", "hypoalbuminemia", "白蛋白低", "白蛋白偏低"],
         ["水肿", "edema", "营养不良", "愈合延迟", "感染"],
        "蛋白质缺乏导致低蛋白血症、水肿和愈合延迟"),
        ("iron", ["缺铁", "iron_deficiency", "铁缺乏"],
         ["贫血", "anemia", "乏力", "面色苍白"],
         "缺铁性贫血表现为乏力、面色苍白"),
        ("vitd", ["维生素D缺乏", "vitamin_d_deficiency"],
         ["骨质疏松", "佝偻病", "骨折", "osteoporosis"],
         "维生素D缺乏导致钙吸收障碍和骨质疏松"),
        ("vitc", ["维生素C缺乏", "scurvy", "坏血病"],
         ["出血", "牙龈出血", "瘀斑", "伤口不愈"],
         "维生素C缺乏导致出血倾向和伤口愈合不良"),
        ("b1", ["维生素B1缺乏", "thiamine_deficiency", "脚气病"],
         ["心衰", "神经病变", "Wernicke", "再喂养综合征"],
         "维生素B1缺乏导致神经病变和心衰，再喂养时需先补充"),
        ("zinc", ["锌缺乏", "zinc_deficiency"],
         ["味觉障碍", "伤口不愈", "免疫低下", "脱发"],
         "锌缺乏影响味觉、伤口愈合和免疫功能"),
    ]
    for rid, src, tgt, desc in deficiency_pairs:
        rules.append(AssociationRule(
            rule_id=f"ds_{rid}",
            assoc_type=AssociationType.DEFICIENCY_SYMPTOM,
            source_patterns=src,
            target_patterns=tgt,
            relation_desc=desc,
        ))

    # ── 过敏-禁忌（4 条）─────────────────────────────────
    allergy_pairs = [
        ("peanut", ["花生过敏", "peanut_allergy", "花生"],
         ["花生蛋白", "花生油", "peanut_protein", "坚果"],
         "花生过敏者禁用含花生蛋白/花生油的制剂"),
        ("lactose", ["乳糖不耐受", "lactose_intolerance"],
         ["乳糖", "牛奶", "乳清蛋白", "lactose"],
         "乳糖不耐受者避免含乳糖制剂，可选无乳糖配方"),
        ("gluten", ["麸质过敏", "乳糜泻", "celiac", "gluten"],
         ["小麦", "大麦", "麸质", "wheat", "gluten"],
         "麸质过敏/乳糜泻患者避免含麸质食物"),
        ("egg", ["鸡蛋过敏", "egg_allergy", "蛋清过敏"],
         ["卵白蛋白", "鸡蛋", "egg_protein", "ovalbumin"],
         "鸡蛋过敏者避免含卵白蛋白的制剂"),
    ]
    for rid, src, tgt, desc in allergy_pairs:
        rules.append(AssociationRule(
            rule_id=f"ac_{rid}",
            assoc_type=AssociationType.ALLERGY_CONTRAINDICATION,
            source_patterns=src,
            target_patterns=tgt,
            relation_desc=desc,
            confidence=0.95,  # 过敏-禁忌置信度更高
        ))

    # ── 疾病-营养方案（4 条）─────────────────────────────
    disease_pairs = [
        ("ckd", ["慢性肾病", "CKD", "肾功能不全", "chronic_kidney"],
         ["低蛋白饮食", "low_protein", "优质蛋白", "限钾", "限磷"],
         "慢性肾病需低蛋白饮食，限制钾磷摄入"),
        ("diabetes", ["糖尿病", "diabetes", "高血糖", "diabetic"],
         ["低GI", "low_gi", "控糖", "低碳水", "糖尿病饮食"],
         "糖尿病需低GI饮食，控制碳水摄入"),
        ("cirrhosis", ["肝硬化", "cirrhosis", "肝病"],
         ["高蛋白", "支链氨基酸", "BCAA", "限钠", "低盐"],
         "肝硬化需高蛋白（含BCAA）、限钠饮食"),
        ("copd", ["慢阻肺", "COPD", "慢性阻塞性肺病"],
         ["高热量", "高脂肪", "低碳水", "低糖"],
         "COPD患者呼吸商高，适合高脂低碳饮食"),
    ]
    for rid, src, tgt, desc in disease_pairs:
        rules.append(AssociationRule(
            rule_id=f"din_{rid}",
            assoc_type=AssociationType.DISEASE_NUTRITION,
            source_patterns=src,
            target_patterns=tgt,
            relation_desc=desc,
        ))

    return rules


# ═══════════════════════════════════════════════════════════════════
# 医疗关联知识库主类
# ═══════════════════════════════════════════════════════════════════


class MedicalAssociationKB:
    """医疗关联知识库 — 注入 su-memory 检索关联图。

    用法：
        kb = MedicalAssociationKB()
        kb.inject_hooks(memory_client)
        # 之后每次 add() 自动扫描医疗关联并创建关联边

    或手动查询：
        rules = kb.match("患者服用华法林")
        # 返回命中的关联规则
    """

    def __init__(self, custom_rules: list[AssociationRule] | None = None):
        self._rules = custom_rules or _seed_rules()
        self._injected_client: SuMemoryLitePro | None = None
        # 缓存：content_hash → 命中的规则列表（避免重复扫描）
        self._match_cache: dict[int, list[tuple[AssociationRule, str]]] = {}

    @property
    def rules(self) -> list[AssociationRule]:
        """全部关联规则"""
        return list(self._rules)

    def add_rule(self, rule: AssociationRule) -> None:
        """添加自定义关联规则"""
        self._rules.append(rule)
        self._match_cache.clear()

    def match(self, content: str) -> list[tuple[AssociationRule, str]]:
        """扫描内容，返回命中的 (规则, 命中端) 列表。

        Args:
            content: 记忆内容文本

        Returns:
            [(rule, "source"|"target"), ...] 命中的规则及命中端
        """
        results: list[tuple[AssociationRule, str]] = []
        content_lower = content.lower()

        for rule in self._rules:
            source_hit = any(p.lower() in content_lower for p in rule.source_patterns)
            target_hit = any(p.lower() in content_lower for p in rule.target_patterns)

            if source_hit:
                results.append((rule, "source"))
            if target_hit:
                results.append((rule, "target"))

        return results

    def find_cross_links(
        self,
        new_content: str,
        existing_contents: list[tuple[str, str]],
    ) -> list[tuple[str, str, AssociationRule]]:
        """找出新内容与已有内容之间的关联。

        Args:
            new_content: 新写入的记忆内容
            existing_contents: [(memory_id, content), ...] 已有记忆

        Returns:
            [(source_id, target_id, rule), ...] 应创建的关联边
        """
        new_matches = self.match(new_content)
        if not new_matches:
            return []

        # 新内容命中的 source/target 端
        new_sources = {r for r, end in new_matches if end == "source"}
        new_targets = {r for r, end in new_matches if end == "target"}

        links: list[tuple[str, str, AssociationRule]] = []

        for mem_id, exist_content in existing_contents:
            exist_matches = self.match(exist_content)
            if not exist_matches:
                continue

            exist_sources = {r for r, end in exist_matches if end == "source"}
            exist_targets = {r for r, end in exist_matches if end == "target"}

            # 匹配规则：一端是 source，另一端是 target（同一规则）
            for rule in new_sources:
                if rule in exist_targets:
                    links.append((mem_id, "PENDING", rule))
            for rule in new_targets:
                if rule in exist_sources:
                    links.append(("PENDING", mem_id, rule))
                    # source 在已有记忆中，target 在新记忆中

        return links

    def inject_hooks(self, client: SuMemoryLitePro) -> None:
        """注入 add() 后置钩子，自动扫描医疗关联。

        在 client.add() 完成后，自动扫描新记忆与已有记忆之间的医疗关联，
        命中则调用 MemoryGraph.add_edge() 创建 explicit 级关联边。

        Args:
            client: SuMemoryLitePro 实例
        """
        self._injected_client = client
        original_add = client.add

        def hooked_add(content: str, metadata: dict = None, **kwargs) -> str:
            # 先执行原始 add
            memory_id = original_add(content, metadata=metadata, **kwargs)

            # 扫描医疗关联并创建边
            try:
                self._scan_and_link(memory_id, content)
            except Exception as e:
                logger.debug("医疗关联扫描降级（非阻塞）: %s", e)

            return memory_id

        client.add = hooked_add
        logger.info(
            "[MedicalKB] 已注入医疗关联钩子，规则数=%d", len(self._rules)
        )

    def _scan_and_link(self, new_memory_id: str, new_content: str) -> int:
        """扫描新记忆与已有记忆，创建医疗关联边。

        Returns:
            创建的关联边数量
        """
        if self._injected_client is None:
            return 0

        client = self._injected_client
        new_matches = self.match(new_content)
        if not new_matches:
            return 0

        new_sources = {r: end for r, end in new_matches if end == "source"}
        new_targets = {r: end for r, end in new_matches if end == "target"}

        if not new_sources and not new_targets:
            return 0

        linked_count = 0

        # 遍历已有记忆，寻找匹配
        for mem_id, node in client._graph._nodes.items():
            if mem_id == new_memory_id:
                continue

            exist_content = node.content
            exist_matches = self.match(exist_content)
            exist_sources = {r for r, end in exist_matches if end == "source"}
            exist_targets = {r for r, end in exist_matches if end == "target"}

            # 新内容是 source 端，已有内容是 target 端
            for rule in new_sources:
                if rule in exist_targets:
                    self._create_link(
                        client, new_memory_id, mem_id, rule
                    )
                    linked_count += 1

            # 新内容是 target 端，已有内容是 source 端
            for rule in new_targets:
                if rule in exist_sources:
                    self._create_link(
                        client, mem_id, new_memory_id, rule
                    )
                    linked_count += 1

        if linked_count > 0:
            logger.info(
                "[MedicalKB] 记忆 %s 创建了 %d 条医疗关联边",
                new_memory_id[:12], linked_count,
            )

        return linked_count

    def _create_link(
        self,
        client: SuMemoryLitePro,
        source_id: str,
        target_id: str,
        rule: AssociationRule,
    ) -> None:
        """在两条记忆间创建医疗关联边。"""
        if hasattr(client, "_graph") and hasattr(client._graph, "add_edge"):
            client._graph.add_edge(
                parent_id=source_id,
                child_id=target_id,
                causal_type="sequence",
                confidence=rule.confidence,
                evidence_type="explicit",
            )

    def explain_association(
        self, source_content: str, target_content: str
    ) -> str | None:
        """解释两条记忆为什么关联（供 explain_multihop 输出）。

        Returns:
            关联描述文本，无关联返回 None
        """
        source_matches = self.match(source_content)
        target_matches = self.match(target_content)

        source_sources = {r for r, end in source_matches if end == "source"}
        target_targets = {r for r, end in target_matches if end == "target"}

        for rule in source_sources:
            if rule in target_targets:
                return rule.relation_desc

        return None
