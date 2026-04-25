"""
术语安全映射配置

本文件提供敏感术语的安全替换映射，确保代码公开时不会泄露底层架构信息。
所有原始术语已被替换为现代技术词汇，功能逻辑保持不变。

术语映射原则:
- 八卦/八经卦 → SemanticCategory (语义分类)
- 五行 → EnergyType (能量类型)
- 旺相 → StrengthState (强度状态)
- 干支 → TimeCode (时序编码)
- 天干地支 → TimeStem/TimeBranch (时序干支)
"""

# ============================================================
# 语义分类映射 (替代八卦)
# ============================================================

# 语义分类枚举值 - 对应原八卦
SEMANTIC_CATEGORY = {
    "CAT_CREATIVE": 0,      # 乾 - 刚健/主动
    "CAT_LAKE": 1,          # 兑 - 喜悦/交换
    "CAT_LIGHT": 2,         # 离 - 明亮/连接
    "CAT_THUNDER": 3,       # 震 - 震动/触发
    "CAT_WIND": 4,          # 巽 - 进入/扩散
    "CAT_ABYSS": 5,         # 坎 - 陷入/风险
    "CAT_MOUNTAIN": 6,      # 艮 - 停止/阻碍
    "CAT_RECEPTIVE": 7,     # 坤 - 柔顺/承载
}

# 语义分类名称
SEMANTIC_CATEGORY_NAMES = {
    0: "creative",    # 乾
    1: "lake",        # 兑
    2: "light",       # 离
    3: "thunder",     # 震
    4: "wind",        # 巽
    5: "abyss",       # 坎
    6: "mountain",    # 艮
    7: "receptive",   # 坤
}

# 语义分类符号 (Unicode符号已移除，改用文字)
SEMANTIC_CATEGORY_SYMBOLS = {
    0: "[C]",   # creative
    1: "[L]",   # lake
    2: "[I]",   # light
    3: "[T]",   # thunder
    4: "[W]",   # wind
    5: "[A]",   # abyss
    6: "[M]",   # mountain
    7: "[R]",   # receptive
}

# 语义分类属性
SEMANTIC_CATEGORY_PROPERTIES = {
    0: {"direction": "nw", "season": "fall", "nature": "active"},
    1: {"direction": "w", "season": "fall", "nature": "joyful"},
    2: {"direction": "s", "season": "summer", "nature": "bright"},
    3: {"direction": "e", "season": "spring", "nature": "dynamic"},
    4: {"direction": "se", "season": "spring", "nature": "penetrating"},
    5: {"direction": "n", "season": "winter", "nature": "hidden"},
    6: {"direction": "ne", "season": "winter", "nature": "steady"},
    7: {"direction": "sw", "season": "summer", "nature": "receptive"},
}


# ============================================================
# 能量元素映射 (替代五行)
# ============================================================

# 能量类型枚举
ENERGY_TYPE = {
    "ELEM_WOOD": "wood",      # 木
    "ELEM_FIRE": "fire",      # 火
    "ELEM_EARTH": "earth",    # 土
    "ELEM_METAL": "metal",    # 金
    "ELEM_WATER": "water",    # 水
}

# 能量类型常量
ELEM_WOOD = "wood"
ELEM_FIRE = "fire"
ELEM_EARTH = "earth"
ELEM_METAL = "metal"
ELEM_WATER = "water"

# 能量增强关系 (替代相生)
ENERGY_ENHANCE = {
    "wood": "fire",     # 木生火
    "fire": "earth",    # 火生土
    "earth": "metal",    # 土生金
    "metal": "water",    # 金生水
    "water": "wood",     # 水生木
}

# 能量抑制关系 (替代相克)
ENERGY_SUPPRESS = {
    "wood": "earth",    # 木克土
    "earth": "water",   # 土克水
    "water": "fire",    # 水克火
    "fire": "metal",    # 火克金
    "metal": "wood",     # 金克木
}


# ============================================================
# 能量状态映射 (替代旺相)
# ============================================================

# 强度状态枚举 (替代旺相休囚死)
STRENGTH_STATE = {
    "STATE_STRONG": "strong",        # 旺
    "STATE_BALANCED": "balanced",     # 相
    "STATE_RESTED": "rested",        # 休
    "STATE_RESTRAINED": "restrained",# 囚
    "STATE_DECLINED": "declined"      # 死
}

# 月份对应的能量状态 (替代月份旺相)
MONTH_ENERGY_STATE = {
    1: "water", 2: "wood", 3: "wood", 4: "fire", 5: "fire",
    6: "earth", 7: "earth", 8: "metal", 9: "metal", 10: "water", 11: "water", 12: "wood"
}

# 能量状态强度映射
STRENGTH_MULTIPLIER = {
    "strong": 1.2,      # 旺
    "balanced": 1.1,    # 相
    "rested": 1.0,      # 休
    "restrained": 0.9,  # 囚
    "declined": 0.8     # 死
}


# ============================================================
# 时序编码映射 (替代干支)
# ============================================================

# 时序干 (替代天干)
TIME_STEMS = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
TIME_STEM_NAMES = ["jia", "yi", "bing", "ding", "wu", "ji", "geng", "xin", "ren", "gui"]

# 时序支 (替代地支)
TIME_BRANCHES = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]
TIME_BRANCH_NAMES = ["zi", "chou", "yin", "mao", "chen", "si", "wu", "wei", "shen", "you", "xu", "hai"]

# 时序支对应的能量类型
TIME_BRANCH_ENERGY = {
    "子": "water", "丑": "earth", "寅": "wood", "卯": "wood",
    "辰": "earth", "巳": "fire", "午": "fire", "未": "earth",
    "申": "metal", "酉": "metal", "戌": "earth", "亥": "water"
}

# 时序周期 (替代甲子)
TIME_CYCLE_LENGTH = 60  # 六十甲子周期


# ============================================================
# 兼容性别名 (向后兼容)
# ============================================================

# 向后兼容的别名 - 避免现有代码报错
# 这些将在后续版本中移除
import warnings

def _deprecation_warning(old_name, new_name):
    warnings.warn(
        f"{old_name} 已弃用，请使用 {new_name}",
        DeprecationWarning,
        stacklevel=3
    )

# 八卦相关别名
BAGUA_ALIAS = SEMANTIC_CATEGORY.copy()
BAGUA_NAMES = SEMANTIC_CATEGORY_NAMES.copy()

# 五行相关别名
WUXING = ENERGY_TYPE.copy()
WUXING_SHENG = ENERGY_ENHANCE.copy()
WUXING_KE = ENERGY_SUPPRESS.copy()

# 旺相相关别名
WANGXIANG = STRENGTH_STATE.copy()
MONTH_WANG = MONTH_ENERGY_STATE.copy()
