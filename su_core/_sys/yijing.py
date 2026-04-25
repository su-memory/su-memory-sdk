"""
周易模块 — 动态变机法则体系

对应易学四位一体中的"周易为变机"层
以六十四卦、三百八十四爻为载体
揭示万物从本源到变化的完整规律

核心：三易法则（不易/变易/简易）
      本卦/互卦/变卦体系
      世爻/应爻/动爻系统
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ============================================================
# 八卦基础
# ============================================================

class HexagramType(Enum):
    """卦象类型"""
    QIAN = 0   # 乾
    DUI = 1    # 兑
    LI = 2     # 离
    ZHEN = 3   # 震
    XUN = 4    # 巽
    KAN = 5    # 坎
    GEN = 6    # 艮
    KUN = 7    # 坤
    
    @property
    def name_zh(self) -> str:
        return ["乾", "兑", "离", "震", "巽", "坎", "艮", "坤"][self.value]
    
    @property
    def symbol(self) -> str:
        return ["☰", "☱", "☲", "☳", "☴", "☵", "☶", "☷"][self.value]
    
    @property
    def wuxing(self) -> str:
        return ["金", "金", "火", "木", "木", "水", "土", "土"][self.value]
    
    @property
    def direction(self) -> str:
        return ["西北", "西", "南", "东", "东南", "北", "东北", "西南"][self.value]
    
    @property
    def nature(self) -> str:
        return ["刚健", "喜悦", "光明", "震动", "入", "陷", "止", "柔顺"][self.value]
    
    @property
    def sheng(self) -> 'HexagramType':
        """相生"""
        map = {0:2, 2:5, 5:7, 7:3, 3:0, 1:0, 4:3, 6:5}
        return HexagramType(map.get(self.value, 0))
    
    @property
    def ke(self) -> 'HexagramType':
        """相克"""
        map = {0:4, 2:1, 5:0, 3:7, 7:5, 4:2, 6:0, 1:6}
        return HexagramType(map.get(self.value, 0))


# 先天八卦（本体定位）
XIAN_TIAN_BAGUA = {
    HexagramType.QIAN: {"position": "南", "yin_yang": "阳"},
    HexagramType.KUN: {"position": "北", "yin_yang": "阴"},
    HexagramType.LI: {"position": "东", "yin_yang": "阴"},
    HexagramType.KAN: {"position": "西", "yin_yang": "阳"},
    HexagramType.ZHEN: {"position": "东北", "yin_yang": "阳"},
    HexagramType.XUN: {"position": "西南", "yin_yang": "阴"},
    HexagramType.GEN: {"position": "西北", "yin_yang": "阳"},
    HexagramType.DUI: {"position": "东南", "yin_yang": "阴"},
}

# 后天八卦（时空应用）
HOU_TIAN_BAGUA = {
    HexagramType.ZHEN: {"position": "正东", "season": "春", "month": "2-3"},
    HexagramType.XUN: {"position": "东南", "season": "春末", "month": "3-4"},
    HexagramType.LI: {"position": "正南", "season": "夏", "month": "5-6"},
    HexagramType.DUI: {"position": "正西", "season": "秋", "month": "8-9"},
    HexagramType.QIAN: {"position": "西北", "season": "秋末", "month": "9-10"},
    HexagramType.GEN: {"position": "东北", "season": "冬春", "month": "12-1"},
    HexagramType.KUN: {"position": "西南", "season": "夏末", "month": "6-7"},
    HexagramType.KAN: {"position": "正北", "season": "冬", "month": "11-12"},
}


# ============================================================
# 六十四卦表（简化版）
# ============================================================

HEXAGRAM_NAMES = [
    "乾", "坤", "屯", "蒙", "需", "讼", "师", "比",
    "小畜", "履", "泰", "否", "同人", "大有", "谦", "豫",
    "随", "蛊", "临", "观", "噬嗑", "贲", "剥", "复",
    "无妄", "大畜", "颐", "大过", "坎", "离", "咸", "恒",
    "遁", "大壮", "晋", "明夷", "家人", "睽", "蹇", "解",
    "损", "益", "夬", "姤", "萃", "升", "困", "井",
    "革", "鼎", "震", "艮", "渐", "归妹", "丰", "旅",
    "巽", "兑", "涣", "节", "中孚", "小过", "既济", "未济"
]

# 完整64卦上下卦映射（从 encoders.py 导入）
from .encoders import HEXAGRAM_TRIGRAMS_BELOW, HEXAGRAM_TRIGRAMS_ABOVE

def _build_hexagram_trigrams():
    """构建完整的64卦上下卦映射"""
    result = []
    for i in range(64):
        upper = HexagramType(HEXAGRAM_TRIGRAMS_ABOVE[i])
        lower = HexagramType(HEXAGRAM_TRIGRAMS_BELOW[i])
        result.append((upper, lower))
    return result

HEXAGRAM_TRIGRAMS = _build_hexagram_trigrams()


# ============================================================
# 卦爻系统
# ============================================================

@dataclass
class YaoPosition:
    """爻位信息"""
    index: int
    name: str
    yin_yang: str
    dignity: str
    position_nature: str


class Hexagram:
    """六十四卦对象"""
    
    def __init__(self, number: int, upper: HexagramType, lower: HexagramType, 
                 gua_name: str = ""):
        self.number = number
        self.upper = upper
        self.lower = lower
        self.name = gua_name or HEXAGRAM_NAMES[number] if number < 64 else "未知"
        self.wuxing = upper.wuxing  # 上卦五行
        
    @property
    def gua_xiang(self) -> str:
        return f"{self.lower.name_zh}{self.upper.name_zh}"
    
    @property
    def hexagram_type(self) -> str:
        """卦的属性分类"""
        if self.upper == HexagramType.QIAN or self.lower == HexagramType.QIAN:
            return "乾天系"
        if self.upper == HexagramType.KUN or self.lower == HexagramType.KUN:
            return "坤地系"
        return "其他"
    
    def get_yin_yang_string(self) -> str:
        """获取卦的阴阳爻列（从下往上）"""
        base = self.number
        result = ""
        for i in range(6):
            result += "阳" if (base + i) % 2 == 0 else "阴"
        return result
    
    def get_base_info(self) -> dict:
        return {
            "number": self.number,
            "name": self.name,
            "卦象": self.gua_xiang,
            "upper": self.upper.name_zh,
            "lower": self.lower.name_zh,
            "wuxing": self.wuxing,
            "upper_wuxing": self.upper.wuxing,
            "lower_wuxing": self.lower.wuxing,
            "type": self.hexagram_type,
            "先天方位": XIAN_TIAN_BAGUA.get(self.lower, {}).get("position", ""),
            "后天方位": HOU_TIAN_BAGUA.get(self.lower, {}).get("position", ""),
        }


# 快速创建卦
def create_hexagram(upper_idx: int, lower_idx: int) -> Hexagram:
    """通过上下卦索引创建卦"""
    upper = HexagramType(upper_idx % 8)
    lower = HexagramType(lower_idx % 8)
    number = upper_idx * 8 + lower_idx
    return Hexagram(number, upper, lower)


# 京房纳甲（卦配天干）
def get_jianggong(hexagram: Hexagram) -> str:
    """获取卦的纳甲天干"""
    jia_gong = {
        HexagramType.QIAN: "甲", HexagramType.DUI: "丁",
        HexagramType.LI: "己", HexagramType.ZHEN: "庚",
        HexagramType.XUN: "辛", HexagramType.KAN: "戊",
        HexagramType.GEN: "丙", HexagramType.KUN: "癸",
    }
    return jia_gong.get(hexagram.upper, "甲")


@dataclass
class TrigramInfo:
    """互卦信息 — 事物内在发展"""
    upper: HexagramType
    lower: HexagramType
    name: str


# ============================================================
# 三易法则编码
# ============================================================

class YiJingRule:
    """三易法则 — 不易/变易/简易"""
    
    @staticmethod
    def bu_yi() -> str:
        """不易 — 恒定规律"""
        return "五行生克、阴阳匹配、层级映射永恒不变"
    
    @staticmethod
    def bian_yi(wuxing_state: str, yao_moving: bool) -> str:
        """变易 — 动态变化"""
        if yao_moving:
            return f"动爻触发，{wuxing_state}气场转变"
        return f"{wuxing_state}气机流转中"
    
    @staticmethod
    def jian_yi(core_wuxing: str) -> str:
        """简易 — 以简驭繁"""
        return f"抓住{core_wuxing}核心气机，把握本质"


@dataclass  
class MemoryYiJing:
    """记忆的周易标注"""
    hexagram: Hexagram
    shi_yao: int
    ying_yao: int
    dong_yao: List[int]
    ben_gua: Hexagram
    hu_gua: Optional[Hexagram]
    bian_gua: Optional[Hexagram]
    
    def get_trend(self) -> str:
        """获取趋势"""
        if self.hu_gua and self.bian_gua:
            return f"{self.ben_gua.name}→{self.hu_gua.name}→{self.bian_gua.name}"
        return self.ben_gua.name
    
    def get_jixing(self) -> str:
        """获取气机形态"""
        return f"{self.hexagram.wuxing}气{'旺' if self.dong_yao else '静'}"


# ============================================================
# 世爻/应爻计算
# ============================================================

def _trigram_to_bits(t: HexagramType) -> List[int]:
    """八卦→三位二进制（先天八卦序）"""
    bits_map = {
        HexagramType.QIAN: [1,1,1],
        HexagramType.DUI:  [0,1,1],
        HexagramType.LI:   [1,0,1],
        HexagramType.ZHEN: [0,0,1],
        HexagramType.XUN:  [1,1,0],
        HexagramType.KAN:  [0,1,0],
        HexagramType.GEN:  [1,0,0],
        HexagramType.KUN:  [0,0,0],
    }
    return bits_map.get(t, [0,0,0])


def compute_shi_ying(upper: HexagramType, lower: HexagramType) -> Tuple[int, int]:
    """
    计算世爻和应爻位置
    
    京房八宫规则（简化版）：
    - 上下卦相同（八纯卦）→ 世爻=6, 应爻=3
    - 上下卦不同 → 根据二进制差异确定世爻位
    - 世应相距3位
    
    Returns:
        (shi_yao_position, ying_yao_position)  # 1-6
    """
    if upper == lower:
        return (6, 3)
    
    upper_bits = _trigram_to_bits(upper)
    lower_bits = _trigram_to_bits(lower)
    
    # 从下往上找第一个不同的爻
    diff_pos = 1
    for i in range(3):
        if upper_bits[i] != lower_bits[i]:
            diff_pos = i + 1
            break
    
    shi = max(1, min(6, diff_pos))
    ying = shi + 3 if shi <= 3 else shi - 3
    return (shi, ying)


def predict_dong_yao(hexagram_index: int, query_context: str = "") -> List[int]:
    """
    预测动爻位置
    
    规则：
    1. 世爻为动爻（核心变动）
    2. 阴阳交界处为动爻（上下卦连接处，即3-4爻）
    3. 如果有特定语义（如"变化""突然"），增加初爻为动爻
    
    Returns:
        动爻位置列表 [1-6]
    """
    if hexagram_index < 0 or hexagram_index >= 64:
        return [1]
    
    upper, lower = HEXAGRAM_TRIGRAMS[hexagram_index]
    shi, _ = compute_shi_ying(upper, lower)
    
    dong_yao = [shi]
    
    # 上下卦连接处（3-4爻）
    if shi not in (3, 4):
        dong_yao.append(3)
    
    # 语义触发
    change_keywords = ["变化", "突然", "转变", "突破", "剧烈", "急", "新"]
    if query_context and any(kw in query_context for kw in change_keywords):
        if 1 not in dong_yao:
            dong_yao.append(1)
    
    return sorted(set(dong_yao))


def _get_yao_line(hexagram_index: int, yao_pos: int) -> int:
    """获取卦象某爻的阴阳值（0=阴, 1=阳）"""
    if hexagram_index < 0 or hexagram_index >= 64:
        return 0
    upper, lower = HEXAGRAM_TRIGRAMS[hexagram_index]
    upper_bits = _trigram_to_bits(upper)
    lower_bits = _trigram_to_bits(lower)
    # 爻位 1-3 对应下卦, 4-6 对应上卦
    all_bits = lower_bits + upper_bits  # [1,2,3,4,5,6]
    if 1 <= yao_pos <= 6:
        return all_bits[yao_pos - 1]
    return 0


def _bits_to_trigram(bits: List[int]) -> HexagramType:
    """三位二进制→八卦"""
    bits_map = {
        (1,1,1): HexagramType.QIAN,
        (0,1,1): HexagramType.DUI,
        (1,0,1): HexagramType.LI,
        (0,0,1): HexagramType.ZHEN,
        (1,1,0): HexagramType.XUN,
        (0,1,0): HexagramType.KAN,
        (1,0,0): HexagramType.GEN,
        (0,0,0): HexagramType.KUN,
    }
    return bits_map.get(tuple(bits), HexagramType.KUN)


def _find_hexagram_by_trigrams(upper: HexagramType, lower: HexagramType) -> int:
    """根据上下卦找64卦索引"""
    for i in range(64):
        if HEXAGRAM_TRIGRAMS[i] == (upper, lower):
            return i
    return 0


def _compute_hu_gua_trigrams(hexagram_index: int) -> Tuple[HexagramType, HexagramType]:
    """计算互卦的上下卦（取2-3-4爻为下卦，3-4-5爻为上卦）"""
    upper, lower = HEXAGRAM_TRIGRAMS[hexagram_index]
    upper_bits = _trigram_to_bits(upper)
    lower_bits = _trigram_to_bits(lower)
    all_bits = lower_bits + upper_bits  # [yao1, yao2, yao3, yao4, yao5, yao6]
    
    hu_lower_bits = [all_bits[1], all_bits[2], all_bits[3]]  # 2-3-4
    hu_upper_bits = [all_bits[2], all_bits[3], all_bits[4]]  # 3-4-5
    
    hu_lower = _bits_to_trigram(hu_lower_bits)
    hu_upper = _bits_to_trigram(hu_upper_bits)
    return (hu_upper, hu_lower)


def _compute_bian_gua(hexagram_index: int, dong_yao: List[int]) -> int:
    """计算变卦（动爻阴阳互换后的新卦）"""
    if not dong_yao:
        return hexagram_index
    
    upper, lower = HEXAGRAM_TRIGRAMS[hexagram_index]
    upper_bits = _trigram_to_bits(upper)
    lower_bits = _trigram_to_bits(lower)
    all_bits = lower_bits + upper_bits
    
    for pos in dong_yao:
        if 1 <= pos <= 6:
            all_bits[pos - 1] = 1 - all_bits[pos - 1]
    
    new_lower = _bits_to_trigram(all_bits[0:3])
    new_upper = _bits_to_trigram(all_bits[3:6])
    return _find_hexagram_by_trigrams(new_upper, new_lower)


# ============================================================
# 周易三层推理引擎
# ============================================================

class YiJingInference:
    """
    周易三层推理引擎
    
    本卦（当前状态）→ 互卦（内在发展）→ 变卦（最终趋势）
    """
    
    def __init__(self):
        pass
    
    def create_memory_yijing(self, hexagram_index: int, content: str = "") -> MemoryYiJing:
        """
        为一条记忆创建完整的周易标注
        """
        idx = hexagram_index % 64
        upper, lower = HEXAGRAM_TRIGRAMS[idx]
        ben_gua = Hexagram(idx, upper, lower)
        
        # 世爻/应爻
        shi, ying = compute_shi_ying(upper, lower)
        
        # 动爻
        dong = predict_dong_yao(idx, content)
        
        # 互卦
        hu_upper, hu_lower = _compute_hu_gua_trigrams(idx)
        hu_idx = _find_hexagram_by_trigrams(hu_upper, hu_lower)
        hu_gua = Hexagram(hu_idx, hu_upper, hu_lower)
        
        # 变卦
        bian_idx = _compute_bian_gua(idx, dong)
        bian_upper, bian_lower = HEXAGRAM_TRIGRAMS[bian_idx]
        bian_gua = Hexagram(bian_idx, bian_upper, bian_lower)
        
        return MemoryYiJing(
            hexagram=ben_gua,
            shi_yao=shi,
            ying_yao=ying,
            dong_yao=dong,
            ben_gua=ben_gua,
            hu_gua=hu_gua,
            bian_gua=bian_gua,
        )
    
    def three_layer_retrieve(self, query_index: int,
                              candidate_indices: List[int],
                              top_k: int = 8) -> List[Dict]:
        """
        三层推理检索
        
        1. 本卦层：直接匹配候选的本卦
        2. 互卦层：匹配候选的互卦 = 查询的本卦（内在关联）
        3. 变卦层：匹配候选的变卦 = 查询的本卦（趋势关联）
        
        每层权重：本卦 0.5, 互卦 0.3, 变卦 0.2
        """
        query_idx = query_index % 64
        query_my = self.create_memory_yijing(query_idx)
        
        results = []
        for cand_idx in candidate_indices:
            cidx = cand_idx % 64
            cand_my = self.create_memory_yijing(cidx)
            
            # 本卦层：直接匹配
            ben_score = 1.0 if cidx == query_idx else 0.0
            # 部分匹配：共享上卦或下卦
            if ben_score == 0:
                q_upper, q_lower = HEXAGRAM_TRIGRAMS[query_idx]
                c_upper, c_lower = HEXAGRAM_TRIGRAMS[cidx]
                if q_upper == c_upper:
                    ben_score = 0.4
                elif q_lower == c_lower:
                    ben_score = 0.4
                # 五行同
                elif query_my.hexagram.wuxing == cand_my.hexagram.wuxing:
                    ben_score = 0.2
            
            # 互卦层：候选的互卦 = 查询的本卦（或反向）
            hu_score = 0.0
            if cand_my.hu_gua and cand_my.hu_gua.number == query_idx:
                hu_score = 1.0
            elif query_my.hu_gua and query_my.hu_gua.number == cidx:
                hu_score = 0.8
            elif cand_my.hu_gua and query_my.hu_gua and cand_my.hu_gua.number == query_my.hu_gua.number:
                hu_score = 0.5
            
            # 变卦层：候选的变卦 = 查询的本卦（或反向）
            bian_score = 0.0
            if cand_my.bian_gua and cand_my.bian_gua.number == query_idx:
                bian_score = 1.0
            elif query_my.bian_gua and query_my.bian_gua.number == cidx:
                bian_score = 0.8
            elif cand_my.bian_gua and query_my.bian_gua and cand_my.bian_gua.number == query_my.bian_gua.number:
                bian_score = 0.5
            
            total = 0.5 * ben_score + 0.3 * hu_score + 0.2 * bian_score
            
            trend = cand_my.ben_gua.name
            if cand_my.hu_gua:
                trend += f"→{cand_my.hu_gua.name}"
            if cand_my.bian_gua:
                trend += f"→{cand_my.bian_gua.name}"
            
            results.append({
                "index": cidx,
                "score": round(total, 4),
                "layer_scores": {
                    "ben_gua": round(ben_score, 4),
                    "hu_gua": round(hu_score, 4),
                    "bian_gua": round(bian_score, 4),
                },
                "trend": trend,
            })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def get_trend_analysis(self, memory_yijing: MemoryYiJing) -> Dict:
        """
        获取记忆的趋势分析
        """
        my = memory_yijing
        current = {"name": my.ben_gua.name, "wuxing": my.ben_gua.wuxing,
                    "type": my.ben_gua.hexagram_type}
        
        internal = None
        if my.hu_gua:
            internal = {"name": my.hu_gua.name, "wuxing": my.hu_gua.wuxing,
                        "type": my.hu_gua.hexagram_type}
        
        future = None
        if my.bian_gua:
            future = {"name": my.bian_gua.name, "wuxing": my.bian_gua.wuxing,
                       "type": my.bian_gua.hexagram_type}
        
        jixing = my.get_jixing()
        
        # 世爻/应爻关系建议
        if my.shi_yao == 6:
            recommendation = "八纯卦，气机极旺，适合作为核心记忆"
        elif my.shi_yao <= 3:
            recommendation = "世爻在下卦，适合作为基础性记忆"
        else:
            recommendation = "世爻在上卦，适合作为应用性记忆"
        
        return {
            "current": current,
            "internal": internal,
            "future": future,
            "jixing": jixing,
            "recommendation": recommendation,
            "trend_path": my.get_trend(),
        }
