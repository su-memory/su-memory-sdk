"""
八卦模块 — 万物类象，记忆的语义分类基础

八卦：乾坤兑离震巽坎艮
对应方位：西北西 南 东 东南 北 东北 西南

每卦三重属性：
- 五行归属（能量类型）
- 方位（空间语义）
- 性情（描述记忆的性质）
"""

from enum import Enum
from typing import Dict, List, Set, Tuple
import math


class Bagua(Enum):
    """后天八卦（主流应用体系）"""
    QIAN  = ("乾", "☰", "金", "西北", "刚健", "天")
    DUI   = ("兑", "☱", "金", "西",  "喜悦", "泽")
    LI    = ("离", "☲", "火", "南",  "光明", "火")
    ZHEN  = ("震", "☳", "木", "东",  "震动", "雷")
    XUN   = ("巽", "☴", "木", "东南","入",   "风")
    KAN   = ("坎", "☵", "水", "北",  "陷",   "水")
    GEN   = ("艮", "☶", "土", "东北","止",   "山")
    KUN   = ("坤", "☷", "土", "西南","柔顺", "地")

    def __init__(self, name_zh: str, symbol: str, wuxing: str, direction: str, nature: str, category: str):
        self.name_zh = name_zh
        self.symbol = symbol
        self.wuxing = wuxing      # 五行归属
        self.direction = direction  # 方位
        self.nature = nature       # 性情/性质
        self.category = category    # 类别

    @classmethod
    def from_trigram(cls, trigram: str) -> 'Bagua':
        """从卦名、符号或方位获取卦象"""
        for gua in cls:
            if trigram in (gua.name_zh, gua.symbol, gua.direction, gua.wuxing):
                return gua
        raise ValueError(f"未知八卦标识: {trigram}")

    def get_associations(self) -> Dict[str, str]:
        """获取该卦的全部关联信息"""
        return {
            "卦名": self.name_zh,
            "符号": self.symbol,
            "五行": self.wuxing,
            "方位": self.direction,
            "性情": self.nature,
            "类别": self.category,
        }


# 八卦类象映射表（说卦传 + 延伸）
BAGUA_ASSOCIATIONS: Dict[str, List[str]] = {
    "乾": ["天", "父", "首", "马", "西北", "金", "健", "君", "玉", "寒"],
    "兑": ["泽", "少女", "口", "羊", "西", "金", "悦", "妾", "雨", "决"],
    "离": ["火", "中女", "目", "雉", "南", "火", "丽", "文明", "日", "电"],
    "震": ["雷", "长男", "足", "龙", "东", "木", "动", "帝", "竹", "乍"],
    "巽": ["风", "长女", "股", "鸡", "东南","木", "入", "商", "高", "长"],
    "坎": ["水", "中男", "耳", "豕", "北", "水", "陷", "雨", "川", "月"],
    "艮": ["山", "少男", "手", "狗", "东北","土", "止", "径路", "小石", "门"],
    "坤": ["地", "母", "腹", "牛", "西南","土", "顺", "文", "布", "囊"],
}

# 八卦→五行→方向的能量映射
WUXING_TO_BAGUA: Dict[str, List[Bagua]] = {
    "金": [Bagua.QIAN, Bagua.DUI],
    "火": [Bagua.LI],
    "木": [Bagua.ZHEN, Bagua.XUN],
    "水": [Bagua.KAN],
    "土": [Bagua.GEN, Bagua.KUN],
}

# 记忆类型→八卦的最简映射（用于自动分类）
MEMORY_TYPE_TO_BAGUA: Dict[str, Bagua] = {
    # 事实类 → 乾（刚健、确定）
    "fact": Bagua.QIAN,
    # 偏好类 → 兑（喜悦、满足）
    "preference": Bagua.DUI,
    # 事件类 → 震（震动、变化）
    "event": Bagua.ZHEN,
    # 关系类 → 巽（入、连接）
    "relationship": Bagua.XUN,
    # 知识类 → 离（光明、文明）
    "knowledge": Bagua.LI,
    # 危险/问题类 → 坎（陷、危机）
    "danger": Bagua.KAN,
    # 目标类 → 艮（止、定目标）
    "goal": Bagua.GEN,
    # 基础/背景类 → 坤（顺、承载）
    "background": Bagua.KUN,
}

# 八卦 anchor 文本（基于说卦传 + 类象，用于语义匹配）
BAGUA_ANCHORS: Dict[str, str] = {
    "乾": "权威确定规则系统领导决策执行命令",
    "兑": "喜悦满足偏好选择快乐奖励交流愉悦",
    "离": "知识理解文化教育研究光明文明智慧",
    "震": "变化行动事件突然动态开始紧急运动",
    "巽": "关系连接网络沟通传递渗透影响交流",
    "坎": "困难风险危险问题压力挑战危机陷阱",
    "艮": "目标界限停止稳定边界坚持方向定位",
    "坤": "背景基础环境承载包容状态条件支撑",
}

# 关键词映射（用于 fallback 评分）
KEYWORDS_TO_BAGUA: Dict[str, List[str]] = {
    "乾": ["确定", "绝对", "必须", "规则", "系统", "父亲", "领导", "首要"],
    "兑": ["喜欢", "满意", "开心", "偏好", "选择", "快乐", "奖励"],
    "离": ["知道", "理解", "相信", "认为", "知识", "文化", "教育", "研究"],
    "震": ["发生", "突然", "变化", "事件", "动态", "行动", "开始", "紧急"],
    "巽": ["关系", "连接", "属于", "朋友", "家人", "团队", "网络", "联系"],
    "坎": ["问题", "危险", "困难", "失败", "风险", "疾病", "压力", "担忧"],
    "艮": ["目标", "计划", "停止", "坚持", "边界", "目的", "未来", "方向"],
    "坤": ["背景", "基础", "环境", "条件", "事实", "情况", "状态", "当前"],
}


# ========================
# 延迟加载模型缓存
# ========================
_anchor_embeddings = None


def _softmax(values):
    """数值稳定的 softmax"""
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def _cosine_sim(a, b):
    """cosine similarity between two vectors"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


def _get_anchor_embeddings(model):
    """获取八卦 anchor 文本的 embeddings（缓存）"""
    global _anchor_embeddings
    if _anchor_embeddings is not None:
        return _anchor_embeddings
    names = list(BAGUA_ANCHORS.keys())
    texts = [BAGUA_ANCHORS[n] for n in names]
    vecs = model.encode(texts).tolist()
    _anchor_embeddings = {names[i]: vecs[i] for i in range(len(names))}
    return _anchor_embeddings


def _keyword_scores(content: str) -> Dict[str, float]:
    """基于关键词的八卦评分"""
    content_lower = content.lower()
    scores = {}
    for gua_name, kws in KEYWORDS_TO_BAGUA.items():
        s = 0
        for kw in kws:
            if kw in content_lower:
                s += 1
        scores[gua_name] = float(s)
    return scores


def infer_bagua_soft(content: str, metadata: Dict = None) -> Dict[str, float]:
    """
    返回所有八卦的概率分布（和为1.0）

    如果有 sentence-transformers 模型：
    1. 计算输入文本与 8 个 anchor 的 cosine similarity
    2. softmax 归一化得到概率分布
    3. 与关键词评分做加权融合（语义 0.7 + 关键词 0.3）

    如果没有模型，fallback 到关键词评分后 softmax
    """
    gua_names = list(BAGUA_ANCHORS.keys())

    # 关键词评分
    kw_scores = _keyword_scores(content)

    # 元数据辅助
    if metadata:
        mem_type = metadata.get("type", "")
        if mem_type in MEMORY_TYPE_TO_BAGUA:
            target = MEMORY_TYPE_TO_BAGUA[mem_type].name_zh
            kw_scores[target] = kw_scores.get(target, 0) + 2.0

    # 尝试语义方式
    model = None
    try:
        from su_core._sys.encoders import _get_st_model
        model = _get_st_model()
    except Exception:
        pass

    if model is not None:
        # 语义方式
        anchor_embs = _get_anchor_embeddings(model)
        content_vec = model.encode(content).tolist()

        sem_sims = []
        for gn in gua_names:
            sim = _cosine_sim(content_vec, anchor_embs[gn])
            sem_sims.append(sim)

        # softmax 归一化语义相似度
        sem_probs = _softmax(sem_sims)

        # 关键词 softmax（加一个小常数避免全零）
        kw_vals = [kw_scores.get(gn, 0) + 0.1 for gn in gua_names]
        kw_probs = _softmax(kw_vals)

        # 加权融合：语义 0.7 + 关键词 0.3
        fused = [sem_probs[i] * 0.7 + kw_probs[i] * 0.3 for i in range(8)]
        total = sum(fused)
        return {gua_names[i]: fused[i] / total for i in range(8)}
    else:
        # fallback: 关键词 softmax
        kw_vals = [kw_scores.get(gn, 0) + 0.1 for gn in gua_names]
        kw_probs = _softmax(kw_vals)
        return {gua_names[i]: kw_probs[i] for i in range(8)}


def infer_bagua_from_content(content: str, metadata: Dict = None) -> Bagua:
    """
    从记忆内容自动推断八卦归属

    使用 infer_bagua_soft() 取 argmax
    """
    probs = infer_bagua_soft(content, metadata)
    best_name = max(probs, key=probs.get)

    # 名称 → Bagua enum 映射
    name_to_bagua = {b.name_zh: b for b in Bagua}
    return name_to_bagua.get(best_name, Bagua.KUN)


def get_bagua_relations(b1: Bagua, b2: Bagua) -> str:
    """
    获取两卦之间的关系

    Returns: "同象" | "相生" | "相克" | "无关"
    """
    if b1.wuxing == b2.wuxing:
        return "同象"

    # 五行生克判断
    try:
        from su_core._sys._c2 import WUXING_SHENG, WUXING_KE, Wuxing

        wx_map = {"金": Wuxing.JIN, "木": Wuxing.MU, "水": Wuxing.SHUI, "火": Wuxing.HUO, "土": Wuxing.TU}
        w1 = wx_map.get(b1.wuxing)
        w2 = wx_map.get(b2.wuxing)

        if w1 and w2:
            # 相生判断：w1生w2 或 w2生w1
            if WUXING_SHENG.get(w1) == w2 or WUXING_SHENG.get(w2) == w1:
                return "相生"
            # 相克判断：w1克w2 或 w2克w1
            if WUXING_KE.get(w1) == w2 or WUXING_KE.get(w2) == w1:
                return "相克"
    except Exception:
        pass

    return "无关"
