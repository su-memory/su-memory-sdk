"""
天干地支模块 — 时空量化符号系统

对应易学四位一体中的"干支为时空"层
将五行能量在时间和空间维度精确量化

十天干：甲乙丙丁戊己庚辛壬癸
十二地支：子丑寅卯辰巳午未申酉戌亥
六十甲子：天干地支循环组合
"""

from enum import Enum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ============================================================
# 天干系统 (天之阳气)
# ============================================================

class Tiangan(Enum):
    """十天干 — 阳气趋势符号"""
    JIA_YANG = 0   # 甲木，阳
    YI_YIN = 1     # 乙木，阴
    BING_YANG = 2  # 丙火，阳
    DING_YIN = 3   # 丁火，阴
    WU_YANG = 4    # 戊土，阳
    JI_YIN = 5     # 己土，阴
    GENG_YANG = 6  # 庚金，阳
    XIN_YIN = 7    # 辛金，阴
    REN_YANG = 8   # 壬水，阳
    GUI_YIN = 9    # 癸水，阴
    
    @property
    def element(self) -> str:
        elements = ["木", "木", "火", "火", "土", "土", "金", "金", "水", "水"]
        return elements[self.value]
    
    @property
    def name(self) -> str:
        names = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
        return names[self.value]
    
    @property
    def yin_yang(self) -> str:
        return "阳" if self.value % 2 == 0 else "阴"
    
    @property
    def nature(self) -> str:
        natures = ["生发", "柔顺", "炎上", "柔和", "中和", "柔润", "刚健", "收敛", "流动", "滋润"]
        return natures[self.value]


# 天干五合 (阴阳交感，气之合化)
TIANGAN_HE = {
    Tiangan.JIA_YANG: Tiangan.JI_YIN,    # 甲己合土
    Tiangan.YI_YIN: Tiangan.GENG_YANG,   # 乙庚合金
    Tiangan.BING_YANG: Tiangan.XIN_YIN,  # 丙辛合水
    Tiangan.DING_YIN: Tiangan.REN_YANG,  # 丁壬合木
    Tiangan.WU_YANG: Tiangan.GUI_YIN,    # 戊癸合火
}

# 天干相冲 (阴阳对立，气之对抗)
TIANGAN_CHONG = {
    Tiangan.JIA_YANG: Tiangan.GENG_YANG,  # 甲庚冲
    Tiangan.YI_YIN: Tiangan.XIN_YIN,      # 乙辛冲
    Tiangan.BING_YANG: Tiangan.REN_YANG,  # 丙壬冲
    Tiangan.DING_YIN: Tiangan.GUI_YIN,    # 丁癸冲
    Tiangan.WU_YANG: Tiangan.JI_YIN,      # 戊己冲
}


# ============================================================
# 地支系统 (地之阴气)
# ============================================================

class Dizhi(Enum):
    """十二地支 — 空间旺衰符号"""
    ZI_YANG = 0    # 子水，阳
    CHOU_YIN = 1   # 丑土，阴
    YIN_YANG = 2   # 寅木，阳
    MAO_YIN = 3    # 卯木，阴
    CHEN_YANG = 4  # 辰土，阳
    SI_YIN = 5     # 巳火，阴
    WU_YANG = 6    # 午火，阳
    WEI_YIN = 7    # 未土，阴
    SHEN_YANG = 8  # 申金，阳
    YOU_YIN = 9    # 酉金，阴
    XU_YANG = 10   # 戌土，阳
    HAI_YIN = 11   # 亥水，阴
    
    @property
    def element(self) -> str:
        elements = ["水", "土", "木", "木", "土", "火", "火", "土", "金", "金", "土", "水"]
        return elements[self.value]
    
    @property
    def name(self) -> str:
        names = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]
        return names[self.value]
    
    @property
    def yin_yang(self) -> str:
        return "阳" if self.value % 2 == 0 else "阴"
    
    @property
    def hidden_stems(self) -> List[Tiangan]:
        """地支藏干 — 地气深层内涵"""
        hidden_map = {
            0: [Tiangan.GUI_YIN],                           # 子：癸
            1: [Tiangan.JI_YIN, Tiangan.GENG_YANG, Tiangan.REN_YANG],  # 丑：己庚壬
            2: [Tiangan.JIA_YANG, Tiangan.BING_YANG, Tiangan.WU_YANG], # 寅：甲丙戊
            3: [Tiangan.YI_YIN],                             # 卯：乙
            4: [Tiangan.WU_YANG, Tiangan.YI_YIN, Tiangan.GUI_YIN],     # 辰：戊乙癸
            5: [Tiangan.BING_YANG, Tiangan.GENG_YANG, Tiangan.WU_YANG],# 巳：丙庚戊
            6: [Tiangan.DING_YIN, Tiangan.JI_YIN],          # 午：丁己
            7: [Tiangan.JI_YIN, Tiangan.YI_YIN, Tiangan.DING_YIN],     # 未：己乙丁
            8: [Tiangan.GENG_YANG, Tiangan.REN_YANG, Tiangan.WU_YANG],  # 申：庚壬戊
            9: [Tiangan.XIN_YIN],                           # 酉：辛
            10: [Tiangan.WU_YANG, Tiangan.XIN_YIN, Tiangan.DING_YIN],  # 戌：戊辛丁
            11: [Tiangan.REN_YANG, Tiangan.JIA_YANG],       # 亥：壬甲
        }
        return hidden_map.get(self.value, [])


# 地支六合
DIZHI_HE = {
    Dizhi.ZI_YANG: Dizhi.CHOU_YIN,     # 子丑合土
    Dizhi.YIN_YANG: Dizhi.WEI_YIN,     # 寅亥合木
    Dizhi.MAO_YIN: Dizhi.XU_YANG,      # 卯戌合火
    Dizhi.SI_YIN: Dizhi.SHEN_YANG,     # 巳申合水
    Dizhi.WU_YANG: Dizhi.YOU_YIN,      # 午未合土
    Dizhi.CHEN_YANG: Dizhi.SI_YIN,     # 辰酉合金
}

# 地支三合
DIZHI_SANHE = {
    # 寅午戌合火
    (Dizhi.YIN_YANG, Dizhi.WU_YANG, Dizhi.XU_YANG): "火局",
    # 申子辰合水
    (Dizhi.SHEN_YANG, Dizhi.ZI_YANG, Dizhi.CHEN_YANG): "水局",
    # 亥卯未合木
    (Dizhi.HAI_YIN, Dizhi.MAO_YIN, Dizhi.WEI_YIN): "木局",
    # 巳酉丑合金
    (Dizhi.SI_YIN, Dizhi.YOU_YIN, Dizhi.CHOU_YIN): "金局",
}

# 地支六冲
DIZHI_CHONG = {
    Dizhi.ZI_YANG: Dizhi.WU_YANG,      # 子午冲
    Dizhi.CHOU_YIN: Dizhi.WEI_YIN,     # 丑未冲
    Dizhi.YIN_YANG: Dizhi.SHEN_YANG,   # 寅申冲
    Dizhi.MAO_YIN: Dizhi.YOU_YIN,      # 卯酉冲
    Dizhi.CHEN_YANG: Dizhi.XU_YANG,    # 辰戌冲
    Dizhi.SI_YIN: Dizhi.HAI_YIN,       # 巳亥冲
}

# 地支三刑
DIZHI_XING = {
    "寅巳申": "无礼之刑",
    "子卯": "无恩之刑", 
    "丑戌未": "恃势之刑",
    "辰辰": "自刑",
    "午午": "自刑",
    "酉酉": "自刑",
    "亥亥": "自刑",
}


# ============================================================
# 六十甲子
# ============================================================

class Jiazi:
    """六十甲子 — 时空统一循环"""
    
    _instance = None
    _cycle: List[Tuple[Tiangan, Dizhi]] = []
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._build_cycle()
        return cls._instance
    
    def _build_cycle(self):
        """构建六十甲子循环"""
        tg_list = list(Tiangan)
        dz_list = list(Dizhi)
        for i in range(60):
            tg = tg_list[i % 10]
            dz = dz_list[i % 12]
            self._cycle.append((tg, dz))
    
    def get(self, index: int) -> Tuple[Tiangan, Dizhi]:
        """获取第n个甲子（0-59循环）"""
        return self._cycle[index % 60]
    
    def get_name(self, index: int) -> str:
        """获取第n个甲子名称"""
        tg, dz = self.get(index)
        return f"{tg.name}{dz.name}"
    
    def get_wuxing(self, index: int) -> str:
        """获取第n个甲子的五行"""
        tg, _ = self.get(index)
        return tg.element


# 便捷函数
def get_jiagan(index: int) -> str:
    """获取第n个天干名称"""
    return list(Tiangan)[index % 10].name

def get_dizhi(index: int) -> str:
    """获取第n个地支名称"""
    return list(Dizhi)[index % 12].name

def get_jiazi(index: int) -> str:
    """获取第n个甲子名称"""
    return Jiazi().get_name(index)


@dataclass
class GanzhiInfo:
    """干支信息 — 用于记忆的时空标注"""
    tiangan: Tiangan
    dizhi: Dizhi
    jiazi_index: int  # 0-59
    
    @property
    def element(self) -> str:
        return self.tiangan.element
    
    @property
    def wuxing(self) -> str:
        return self.element
    
    @property
    def name(self) -> str:
        return f"{self.tiangan.name}{self.dizhi.name}"
    
    @property
    def dayun(self) -> str:
        """大运 — 人生阶段"""
        elements = self.tiangan.element
        # 简化版大运
        return elements


def create_ganzhi(tiangan_idx: int, dizhi_idx: int) -> GanzhiInfo:
    """创建干支信息"""
    jiazi_idx = (tiangan_idx % 10) * 6 + (dizhi_idx % 12)
    return GanzhiInfo(
        tiangan=Tiangan(tiangan_idx % 10),
        dizhi=Dizhi(dizhi_idx % 12),
        jiazi_index=jiazi_idx % 60
    )