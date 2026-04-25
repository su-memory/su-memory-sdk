"""五行模块 — 能量状态与动力学（五态旺衰 + 乘侮机制）"""
from enum import Enum
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass


class Wuxing(Enum):
    """五行能量状态"""
    MU  = 0
    HUO = 1
    TU  = 2
    JIN = 3
    SHUI= 4

    @property
    def element(self) -> str:
        return ("木", "火", "土", "金", "水")[self.value]

    @property
    def nature(self) -> str:
        return ("生长", "温热", "承载", "收敛", "滋润")[self.value]

    @property
    def movement(self) -> str:
        return ("升发", "光明", "生化", "肃杀", "向下")[self.value]

    @property
    def direction(self) -> str:
        return ("东", "南", "中", "西", "北")[self.value]

    @property
    def season(self) -> str:
        return ("春", "夏", "四季末", "秋", "冬")[self.value]


# 相生顺序：键生值
WUXING_SHENG: Dict[Wuxing, Wuxing] = {
    Wuxing.MU: Wuxing.HUO,
    Wuxing.HUO: Wuxing.TU,
    Wuxing.TU: Wuxing.JIN,
    Wuxing.JIN: Wuxing.SHUI,
    Wuxing.SHUI: Wuxing.MU,
}

# 相克顺序：键克值
WUXING_KE: Dict[Wuxing, Wuxing] = {
    Wuxing.MU: Wuxing.TU,
    Wuxing.TU: Wuxing.SHUI,
    Wuxing.SHUI: Wuxing.HUO,
    Wuxing.HUO: Wuxing.JIN,
    Wuxing.JIN: Wuxing.MU,
}


# ========================
# 五行旺衰休囚死 完整五态
# ========================

WUXING_STATE_MULTIPLIERS = {
    "旺": 2.0,
    "相": 1.3,
    "休": 1.0,
    "囚": 0.5,
    "死": 0.3,
}


def _get_sheng_parent(target: Wuxing) -> Wuxing:
    """找到生 target 的五行"""
    for k, v in WUXING_SHENG.items():
        if v == target:
            return k
    return target


def get_wuxing_state(target: Wuxing, current_season: Wuxing) -> Tuple[str, float]:
    """
    获取五行在当前月令下的旺衰状态

    Args:
        target: 目标五行
        current_season: 当前月令所属五行（春→木, 夏→火, 四季末→土, 秋→金, 冬→水）

    Returns:
        (状态名称, 强度倍数) 如 ("旺", 2.0)
    """
    if target == current_season:
        return "旺", 2.0
    if WUXING_SHENG.get(current_season) == target:
        return "相", 1.3
    if _get_sheng_parent(current_season) == target:
        return "休", 1.0
    if WUXING_KE.get(target) == current_season:
        return "囚", 0.5
    if WUXING_KE.get(current_season) == target:
        return "死", 0.3
    return "休", 1.0


def check_cheng_wu(attacker: Wuxing, defender: Wuxing,
                   attacker_intensity: float, defender_intensity: float) -> str:
    """
    检查五行乘侮关系
    Returns: "normal" | "cheng" | "wu"
    """
    if WUXING_KE.get(attacker) != defender:
        return "normal"
    if defender_intensity <= 0:
        return "cheng"
    if attacker_intensity / defender_intensity > 2.0:
        return "cheng"
    if attacker_intensity <= 0:
        return "wu"
    if defender_intensity / attacker_intensity > 2.0:
        return "wu"
    return "normal"


def wuxing_similarity(w1: Wuxing, w2: Wuxing) -> float:
    """
    计算两个五行之间的相似度（0.0~1.0）
    - 同行: 1.0
    - 相生: 0.7
    - 无关: 0.3
    - 相克: 0.1
    """
    if w1 == w2:
        return 1.0
    if WUXING_SHENG.get(w1) == w2 or WUXING_SHENG.get(w2) == w1:
        return 0.7
    if WUXING_KE.get(w1) == w2 or WUXING_KE.get(w2) == w1:
        return 0.1
    return 0.3


@dataclass
class WuxingState:
    wuxing: Wuxing
    intensity: float = 1.0
    status: str = "平衡"

    def get_effective_intensity(self, environment: 'WuxingState' = None) -> float:
        if environment is None:
            return self.intensity
        state_name, multiplier = get_wuxing_state(self.wuxing, environment.wuxing)
        self.status = state_name
        return self.intensity * multiplier


class WuxingEnergyNetwork:
    def __init__(self):
        self.memory_states: Dict[str, WuxingState] = {}

    def register_memory(self, memory_id: str, wuxing: Wuxing) -> None:
        self.memory_states[memory_id] = WuxingState(wuxing=wuxing)

    def propagate_energy(self, source_id: str, delta: float) -> None:
        source = self.memory_states.get(source_id)
        if not source:
            return
        target_wuxing = WUXING_SHENG.get(source.wuxing)
        if not target_wuxing:
            return
        for mem_id, state in self.memory_states.items():
            if state.wuxing == target_wuxing:
                state.intensity += delta

    def get_dominant_wuxing(self) -> Wuxing:
        if not self.memory_states:
            return Wuxing.TU
        counts: Dict[Wuxing, float] = {}
        for state in self.memory_states.values():
            w = state.wuxing
            counts[w] = counts.get(w, 0) + state.intensity
        return max(counts, key=counts.get)


def wuxing_from_bagua(bagua_name: str) -> Wuxing:
    BAGUA_WUXING_MAP = {
        "乾": "金", "兑": "金",
        "离": "火", "震": "木", "巽": "木",
        "坎": "水", "艮": "土", "坤": "土",
    }
    wx_name = BAGUA_WUXING_MAP.get(bagua_name, "土")
    for w in Wuxing:
        if w.element == wx_name:
            return w
    return Wuxing.TU
