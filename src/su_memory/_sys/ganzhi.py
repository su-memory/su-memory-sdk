"""
TimeCode Module - Spatio-temporal Quantification System

Corresponds to the "TimeCode for Spatio-temporal" layer in the Four-in-One System
Precisely quantifies energy patterns across time and space dimensions

Ten TimeStems: Jia Yi Bing Ding Wu Ji Geng Xin Ren Gui
Twelve TimeBranches: Zi Chou Yin Mao Chen Si Wu Wei Shen You Xu Hai
Sixty TimeCycles: TimeStem-TimeBranch cyclic combinations
"""

from enum import Enum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ============================================================
# TimeStem System (Yang energy in celestial order)
# ============================================================

class TimeStem(Enum):
    """Ten Heavenly Stems - Yang energy tendency symbols"""
    JIA_YANG = 0   # wood, yang
    YI_YIN = 1     # wood, yin
    BING_YANG = 2  # fire, yang
    DING_YIN = 3   # fire, yin
    WU_YANG = 4    # earth, yang
    JI_YIN = 5     # earth, yin
    GENG_YANG = 6  # metal, yang
    XIN_YIN = 7    # metal, yin
    REN_YANG = 8   # water, yang
    GUI_YIN = 9    # water, yin
    
    @property
    def energy_type(self) -> str:
        """Get the energy type for this stem"""
        energy_types = ["wood", "wood", "fire", "fire", "earth", "earth", "metal", "metal", "water", "water"]
        return energy_types[self.value]
    
    @property
    def name(self) -> str:
        """Get the stem name"""
        names = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
        return names[self.value]
    
    @property
    def polarity(self) -> str:
        """Get polarity: yang or yin"""
        return "yang" if self.value % 2 == 0 else "yin"
    
    @property
    def nature(self) -> str:
        """Get the nature descriptor"""
        natures = ["生发", "柔顺", "炎上", "柔和", "中和", "柔润", "刚健", "收敛", "流动", "滋润"]
        return natures[self.value]


# TimeStem Combinatorial Harmony (yin-yang interaction, energy fusion)
STEM_HE = {
    TimeStem.JIA_YANG: TimeStem.JI_YIN,    # Jia-Ji = earth
    TimeStem.YI_YIN: TimeStem.GENG_YANG,   # Yi-Geng = metal
    TimeStem.BING_YANG: TimeStem.XIN_YIN,  # Bing-Xin = water
    TimeStem.DING_YIN: TimeStem.REN_YANG,   # Ding-Ren = wood
    TimeStem.WU_YANG: TimeStem.GUI_YIN,    # Wu-Gui = fire
}

# TimeStem Oppositional Conflict (yin-yang opposition, energy confrontation)
STEM_CHONG = {
    TimeStem.JIA_YANG: TimeStem.GENG_YANG,  # Jia-Geng
    TimeStem.YI_YIN: TimeStem.XIN_YIN,       # Yi-Xin
    TimeStem.BING_YANG: TimeStem.REN_YANG,   # Bing-Ren
    TimeStem.DING_YIN: TimeStem.GUI_YIN,     # Ding-Gui
    TimeStem.WU_YANG: TimeStem.JI_YIN,        # Wu-Ji
}


# ============================================================
# TimeBranch System (Yin energy in terrestrial order)
# ============================================================

class TimeBranch(Enum):
    """Twelve Earthly Branches - Spatial strength_state symbols"""
    ZI_YANG = 0    # water, yang
    CHOU_YIN = 1   # earth, yin
    YIN_YANG = 2   # wood, yang
    MAO_YIN = 3    # wood, yin
    CHEN_YANG = 4  # earth, yang
    SI_YIN = 5     # fire, yin
    WU_YANG = 6    # fire, yang
    WEI_YIN = 7    # earth, yin
    SHEN_YANG = 8  # metal, yang
    YOU_YIN = 9    # metal, yin
    XU_YANG = 10   # earth, yang
    HAI_YIN = 11   # water, yin
    
    @property
    def energy_type(self) -> str:
        """Get the energy type for this branch"""
        energy_types = ["water", "earth", "wood", "wood", "earth", "fire", "fire", "earth", "metal", "metal", "earth", "water"]
        return energy_types[self.value]
    
    @property
    def name(self) -> str:
        """Get the branch name"""
        names = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]
        return names[self.value]
    
    @property
    def polarity(self) -> str:
        """Get polarity: yang or yin"""
        return "yang" if self.value % 2 == 0 else "yin"
    
    @property
    def hidden_stems(self) -> List[TimeStem]:
        """Hidden stems - deep terrestrial energy content"""
        hidden_map = {
            0: [TimeStem.GUI_YIN],                            # Zi: Gui
            1: [TimeStem.JI_YIN, TimeStem.GENG_YANG, TimeStem.REN_YANG],  # Chou: Ji, Geng, Ren
            2: [TimeStem.JIA_YANG, TimeStem.BING_YANG, TimeStem.WU_YANG], # Yin: Jia, Bing, Wu
            3: [TimeStem.YI_YIN],                             # Mao: Yi
            4: [TimeStem.WU_YANG, TimeStem.YI_YIN, TimeStem.GUI_YIN],    # Chen: Wu, Yi, Gui
            5: [TimeStem.BING_YANG, TimeStem.GENG_YANG, TimeStem.WU_YANG],# Si: Bing, Geng, Wu
            6: [TimeStem.DING_YIN, TimeStem.JI_YIN],          # Wu: Ding, Ji
            7: [TimeStem.JI_YIN, TimeStem.YI_YIN, TimeStem.DING_YIN],    # Wei: Ji, Yi, Ding
            8: [TimeStem.GENG_YANG, TimeStem.REN_YANG, TimeStem.WU_YANG], # Shen: Geng, Ren, Wu
            9: [TimeStem.XIN_YIN],                            # You: Xin
            10: [TimeStem.WU_YANG, TimeStem.XIN_YIN, TimeStem.DING_YIN],  # Xu: Wu, Xin, Ding
            11: [TimeStem.REN_YANG, TimeStem.JIA_YANG],       # Hai: Ren, Jia
        }
        return hidden_map.get(self.value, [])


# TimeBranch Binary Harmony
BRANCH_HE = {
    TimeBranch.ZI_YANG: TimeBranch.CHOU_YIN,     # Zi-Chou = earth
    TimeBranch.YIN_YANG: TimeBranch.WEI_YIN,     # Yin-Hai = wood
    TimeBranch.MAO_YIN: TimeBranch.XU_YANG,      # Mao-Xu = fire
    TimeBranch.SI_YIN: TimeBranch.SHEN_YANG,     # Si-Shen = water
    TimeBranch.WU_YANG: TimeBranch.YOU_YIN,       # Wu-You = earth
    TimeBranch.CHEN_YANG: TimeBranch.SI_YIN,     # Chen-Si = metal
}

# TimeBranch Triple Conjunction
BRANCH_SANHE = {
    # Yin-Wu-Xu = fire pattern
    (TimeBranch.YIN_YANG, TimeBranch.WU_YANG, TimeBranch.XU_YANG): "fire_pattern",
    # Shen-Zi-Chen = water pattern
    (TimeBranch.SHEN_YANG, TimeBranch.ZI_YANG, TimeBranch.CHEN_YANG): "water_pattern",
    # Hai-Mao-Wei = wood pattern
    (TimeBranch.HAI_YIN, TimeBranch.MAO_YIN, TimeBranch.WEI_YIN): "wood_pattern",
    # Si-You-Chou = metal pattern
    (TimeBranch.SI_YIN, TimeBranch.YOU_YIN, TimeBranch.CHOU_YIN): "metal_pattern",
}

# TimeBranch Six Conflicts
BRANCH_CHONG = {
    TimeBranch.ZI_YANG: TimeBranch.WU_YANG,      # Zi-Wu
    TimeBranch.CHOU_YIN: TimeBranch.WEI_YIN,      # Chou-Wei
    TimeBranch.YIN_YANG: TimeBranch.SHEN_YANG,    # Yin-Shen
    TimeBranch.MAO_YIN: TimeBranch.YOU_YIN,       # Mao-You
    TimeBranch.CHEN_YANG: TimeBranch.XU_YANG,     # Chen-Xu
    TimeBranch.SI_YIN: TimeBranch.HAI_YIN,        # Si-Hai
}

# TimeBranch Triple Punishments
BRANCH_XING = {
    "寅巳申": "impolite_punishment",
    "子卯": "ungrateful_punishment",
    "丑戌未": "arrogant_punishment",
    "辰辰": "self_punishment",
    "午午": "self_punishment",
    "酉酉": "self_punishment",
    "亥亥": "self_punishment",
}


# ============================================================
# TimeCycle System (Sixty Cyclic Combinations)
# ============================================================

class TimeCycle:
    """Sixty TimeCycles - Unified spatio-temporal cycle"""
    
    _instance = None
    _cycle: List[Tuple[TimeStem, TimeBranch]] = []
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._build_cycle()
        return cls._instance
    
    def _build_cycle(self):
        """Build the 60-cycle sequence"""
        stem_list = list(TimeStem)
        branch_list = list(TimeBranch)
        for i in range(60):
            stem = stem_list[i % 10]
            branch = branch_list[i % 12]
            self._cycle.append((stem, branch))
    
    def get(self, index: int) -> Tuple[TimeStem, TimeBranch]:
        """Get the nth cycle element (0-59 cyclic)"""
        return self._cycle[index % 60]
    
    def get_name(self, index: int) -> str:
        """Get the name of the nth cycle element"""
        stem, branch = self.get(index)
        return f"{stem.name}{branch.name}"
    
    def get_energy_type(self, index: int) -> str:
        """Get the energy type of the nth cycle element"""
        stem, _ = self.get(index)
        return stem.energy_type


# Convenience functions
def get_stem(index: int) -> str:
    """Get the name of the nth TimeStem"""
    return list(TimeStem)[index % 10].name

def get_branch(index: int) -> str:
    """Get the name of the nth TimeBranch"""
    return list(TimeBranch)[index % 12].name

def get_cycle(index: int) -> str:
    """Get the name of the nth TimeCycle"""
    return TimeCycle().get_name(index)


@dataclass
class TimeCodeInfo:
    """TimeCode information - spatio-temporal annotation for memory"""
    time_stem: TimeStem
    time_branch: TimeBranch
    cycle_index: int  # 0-59
    
    @property
    def energy_type(self) -> str:
        """Get the energy type"""
        return self.time_stem.energy_type
    
    @property
    def name(self) -> str:
        """Get the combined name"""
        return f"{self.time_stem.name}{self.time_branch.name}"
    
    @property
    def polarity(self) -> str:
        """Get the polarity from time stem"""
        return self.time_stem.polarity
    
    @property
    def life_cycle(self) -> str:
        """Life cycle phase indicator"""
        energy = self.time_stem.energy_type
        # Simplified life cycle mapping
        return energy


def create_time_code(stem_idx: int, branch_idx: int) -> TimeCodeInfo:
    """Create a TimeCodeInfo instance"""
    cycle_idx = (stem_idx % 10) * 6 + (branch_idx % 12)
    return TimeCodeInfo(
        time_stem=TimeStem(stem_idx % 10),
        time_branch=TimeBranch(branch_idx % 12),
        cycle_index=cycle_idx % 60
    )
