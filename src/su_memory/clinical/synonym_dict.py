"""
synonym_dict — 医学术语同义词典（C1: 医疗同义召回）

在医院内网（无向量服务）场景下，纯关键词检索对跨语言/缩写/近义词
召回率极低（实测 4%）。本词典在 query 侧扩展同义词，让倒排索引
能命中跨语言/缩写/近义的记忆。

⚠️ 项目区隔：同义词典是「检索增强」，不是「语义理解」。
   真正的语义理解由向量模型负责；本词典是向量不可用时的兜底。

支持 load_from_file 扩展（医院可维护自己的术语对照表）。

Example:
  >>> from su_memory.clinical.synonym_dict import MedicalSynonymDict
  >>> syn = MedicalSynonymDict()
  >>> syn.expand_query("华法林")  # → ["华法林", "warfarin"]
  >>> syn.load_from_file("hospital_terms.json")  # 扩展
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def _seed_synonyms() -> dict[str, list[str]]:
    """种子同义词典——双向映射会自动生成。"""
    return {
        # 药物：通用名 ↔ 英文/商品名
        "华法林": ["warfarin"],
        "二甲双胍": ["metformin"],
        "甲氨蝶呤": ["methotrexate", "mtx"],
        "地高辛": ["digoxin"],
        "卡托普利": ["captopril"],
        "呋塞米": ["furosemide", "速尿"],
        "环丙沙星": ["ciprofloxacin"],
        "左旋甲状腺素": ["levothyroxine"],
        # 检验项目：中文 ↔ 英文/缩写
        "白蛋白": ["albumin", "alb"],
        "血红蛋白": ["hemoglobin", "hb", "hgb"],
        "血糖": ["glucose", "glu"],
        "肌酐": ["creatinine", "cr"],
        "前白蛋白": ["prealbumin", "pa"],
        "转铁蛋白": ["transferrin", "tf"],
        "C反应蛋白": ["CRP", "c-reactive protein"],
        "体重指数": ["BMI", "body mass index"],
        "钾": ["potassium", "k+"],
        "钠": ["sodium", "na+"],
        # 临床概念近义
        "禁忌症": ["过敏", "不耐受", "contraindication"],
        "营养不良": ["营养缺乏", "malnutrition"],
        "高血压": ["hypertension", "htn"],
        "糖尿病": ["diabetes", "dm"],
        "贫血": ["anemia"],
        "低蛋白血症": ["hypoproteinemia"],
        # 营养素
        "维生素B12": ["cobalamin", "b12"],
        "叶酸": ["folic acid", "folate"],
        "膳食纤维": ["dietary fiber"],
        "卡路里": ["热量", "calories"],
        # 症状
        "头晕": ["眩晕", "dizziness"],
        "恶心": ["呕吐", "nausea"],
        "腹泻": ["diarrhea"],
        "便秘": ["constipation"],
        "水肿": ["浮肿", "edema"],
        "消瘦": ["体重下降", "weight loss"],
        # 诊疗
        "医嘱": ["处方", "prescription"],
        "营养方案": ["饮食方案", "diet plan"],
        "肠内营养": ["enteral nutrition", "en"],
        "肠外营养": ["parenteral nutrition", "pn"],
        "随访": ["复诊", "follow up"],
        "出院": ["discharge"],
    }


class MedicalSynonymDict:
    """医学术语同义词典。

    提供双向映射：
      - 正向：术语 → 同义词列表（query 扩展用）
      - 反向：同义词 → 原术语列表（query 扩展用）
    """

    def __init__(self, custom: dict[str, list[str]] | None = None):
        self._forward: dict[str, set[str]] = {}
        self._load(_seed_synonyms())
        if custom:
            self._load(custom)

    def _load(self, mapping: dict[str, list[str]]) -> None:
        """加载一组同义词映射（自动生成双向）。"""
        for term, synonyms in mapping.items():
            group = set([term] + list(synonyms))
            for g in group:
                self._forward.setdefault(g, set()).update(group)

    def load_from_file(self, path: str) -> int:
        """从 JSON 文件扩展同义词典。

        JSON 格式：{"术语": ["同义词1", "同义词2"], ...}
        返回加载的条目数。
        """
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self._load(data)
            logger.info("[SynonymDict] 从 %s 加载了 %d 组同义词", path, len(data))
            return len(data)
        except Exception as e:
            logger.warning("[SynonymDict] 加载 %s 失败: %s", path, e)
            return 0

    def expand_query(self, query: str) -> list[str]:
        """扩展查询——返回 query 中包含的所有同义词（去重）。

        用于 query 侧扩展：把扩展后的词拼入 query，让倒排索引命中。

        Args:
            query: 原始查询文本

        Returns:
            扩展后的同义词列表（含原词）
        """
        expanded: list[str] = [query]
        seen = {query}
        for term, group in self._forward.items():
            if term in query:
                for syn in group:
                    if syn not in seen:
                        expanded.append(syn)
                        seen.add(syn)
        return expanded

    def expand_terms(self, terms: list[str]) -> list[str]:
        """扩展术语列表（用于 add 侧索引增强）。

        Args:
            terms: 原始术语列表（如分词结果）

        Returns:
            扩展后的术语列表（含同义词）
        """
        result = list(terms)
        seen = set(terms)
        for term in terms:
            if term in self._forward:
                for syn in self._forward[term]:
                    if syn not in seen:
                        result.append(syn)
                        seen.add(syn)
        return result

    def get_reverse_map(self) -> dict[str, list[str]]:
        """反向映射：同义词 → 该同义词的所有等价词（含自身）。

        用于 query 侧按词扩展。
        """
        return {k: sorted(v) for k, v in self._forward.items()}

    def stats(self) -> dict[str, int]:
        """词典统计。"""
        return {
            "terms": len(self._forward),
            "groups": len({frozenset(v) for v in self._forward.values()}),
        }
