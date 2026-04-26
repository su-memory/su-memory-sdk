"""
术语安全映射配置

本文件提供敏感术语的安全替换映射，确保代码公开时不会泄露底层架构信息。
所有原始术语已被替换为现代技术词汇，功能逻辑保持不变。

术语映射原则:
- Trigram Patterns/八经卦 → SemanticCategory (语义分类)
- Energy System → EnergyType (能量类型)
- 旺相 → StrengthState (强度状态)
- 干支 → TimeCode (时序编码)
- Heavenly StemsEarthly Branches → TimeStem/TimeBranch (时序干支)
"""

# ============================================================
# 语义分类映射 (替代Trigram Patterns)
# ============================================================

# 语义分类枚举值 - 对应原Trigram Patterns
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
# 能量元素映射 (替代Energy System)
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
# Energy System完整属性映射 (Extended Energy Attributes)
# ============================================================

# Energy-Season mapping: which seasons each energy type governs
ENERGY_SEASON = {
    "wood": ["spring", "early_summer"],
    "fire": ["summer", "late_summer"],
    "earth": ["late_summer", "mid_autumn"],
    "metal": ["autumn", "early_winter"],
    "water": ["winter", "early_spring"],
}

# Energy-Direction mapping: cardinal and intercardinal directions
ENERGY_DIRECTION = {
    "wood": ["east", "southeast"],
    "fire": ["south", "southeast"],
    "earth": ["center", "northeast", "southwest"],
    "metal": ["west", "northwest"],
    "water": ["north", "northeast"],
}

# Energy-Color mapping: associated colors for each energy type
ENERGY_COLOR = {
    "wood": ["green", "blue_green"],
    "fire": ["red", "orange"],
    "earth": ["yellow", "brown"],
    "metal": ["white", "silver"],
    "water": ["black", "blue"],
}

# Energy-Organ mapping: traditional Chinese medicine organ associations
ENERGY_ORGAN = {
    "wood": "liver",
    "fire": "heart",
    "earth": "spleen",
    "metal": "lung",
    "water": "kidney",
}

# Energy-Taste mapping: five tastes associated with each energy type
ENERGY_TASTE = {
    "wood": "sour",
    "fire": "bitter",
    "earth": "sweet",
    "metal": "pungent",
    "water": "salty",
}

# Energy-Emotion mapping: emotional states related to each energy type
ENERGY_EMOTION = {
    "wood": "anger",
    "fire": "joy",
    "earth": "thought",
    "metal": "grief",
    "water": "fear",
}

# Energy-Industry mapping: industries associated with each energy type
ENERGY_INDUSTRY = {
    "wood": ["forestry", "paper", "publishing", "education"],
    "fire": ["energy", "light", "electronics", "it"],
    "earth": ["construction", "real_estate", "farming", "mining"],
    "metal": ["metalwork", "finance", "government", "law"],
    "water": ["transport", "trade", "shipping", "consulting"],
}

# ============================================================
# 语义分类完整属性映射 (Extended Semantic Category Attributes)
# ============================================================

# Prior trigram directions (先天方位): original bagua positions
PRIOR_TRIGRAM_DIRECTION = {
    0: "south",      # 乾 - creative direction
    1: "north",      # 坤 - receptive direction
    2: "northeast",  # 震 - thunder direction
    3: "northeast",  # 巽 - wind direction (note: 2&3 both ne in prior)
    4: "west",       # 坎 - abyss direction
    5: "east",       # 离 - light direction
    6: "northeast",  # 艮 - mountain direction
    7: "southeast",  # 兑 - lake direction
}

# Post trigram directions (后天方位): later bagua positions from 9-palace
POST_TRIGRAM_DIRECTION = {
    0: "northwest",  # 乾 - creative in post system
    1: "southwest",  # 坤 - receptive in post system
    2: "east",       # 震 - thunder in post system
    3: "southeast",  # 巽 - wind in post system
    4: "north",      # 坎 - abyss in post system
    5: "south",      # 离 - light in post system
    6: "northeast",  # 艮 - mountain in post system
    7: "west",       # 兑 - lake in post system
}

# Trigram energy mapping: which energy type each trigram belongs to
TRIGRAM_ENERGY_MAP = {
    0: "metal",   # 乾 - creative (metal)
    1: "earth",   # 坤 - receptive (earth)
    2: "wood",    # 震 - thunder (wood)
    3: "wood",    # 巽 - wind (wood)
    4: "water",   # 坎 - abyss (water)
    5: "fire",    # 离 - light (fire)
    6: "earth",   # 艮 - mountain (earth)
    7: "metal",   # 兑 - lake (metal)
}

# Trigram body mapping: body parts associated with each trigram
TRIGRAM_BODY_MAP = {
    0: ["head", "brain"],           # 乾 - creative: head and brain
    1: ["abdomen", "digestive"],    # 坤 - receptive: abdomen and digestive
    2: ["feet", "nerves"],          # 震 - thunder: feet and nerves
    3: ["thighs", "respiratory"],   # 巽 - wind: thighs and respiratory
    4: ["ears", "reproductive"],     # 坎 - abyss: ears and reproductive
    5: ["eyes", "cardiovascular"],   # 离 - light: eyes and cardiovascular
    6: ["hands", "digestive"],      # 艮 - mountain: hands and digestive
    7: ["mouth", "respiratory"],    # 兑 - lake: mouth and respiratory
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

# 时序干 (替代Heavenly Stems)
TIME_STEMS = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
TIME_STEM_NAMES = ["jia", "yi", "bing", "ding", "wu", "ji", "geng", "xin", "ren", "gui"]

# 时序支 (替代Earthly Branches)
TIME_BRANCHES = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]
TIME_BRANCH_NAMES = ["zi", "chou", "yin", "mao", "chen", "si", "wu", "wei", "shen", "you", "xu", "hai"]

# 时序支对应的能量类型
TIME_BRANCH_ENERGY = {
    "子": "water", "丑": "earth", "寅": "wood", "卯": "wood",
    "辰": "earth", "巳": "fire", "午": "fire", "未": "earth",
    "申": "metal", "酉": "metal", "戌": "earth", "亥": "water"
}

# 时序周期 (替代甲子)
TIME_CYCLE_LENGTH = 60  # Sixty Cycle周期

# ============================================================
# Heavenly StemsEarthly Branches关系映射 (Stem-Branch Relations)
# ============================================================

# Stem-He map (Heavenly Stems五合): harmonious combinations of yang-yin stems
# Key: stem index (0-9), Value: paired stem index
STEM_HE_MAP = {
    0: 5,   # 甲-己
    1: 6,   # 乙-庚
    2: 7,   # 丙-辛
    3: 8,   # 丁-壬
    4: 9,   # 戊-癸
}

# Stem-Chong map (Heavenly Stems相冲): conflicting stem combinations
# Key: stem index (0-9), Value: opposing stem index
STEM_CHONG_MAP = {
    0: 6,   # 甲-庚
    1: 7,   # 乙-辛
    2: 8,   # 丙-壬
    3: 9,   # 丁-癸
    4: 5,   # 戊-己
}

# Branch-He map (Earthly Branches六合): harmonious branch combinations
# Key: branch index (0-11), Value: paired branch index
BRANCH_HE_MAP = {
    0: 1,   # 子-丑
    2: 11,  # 寅-亥 (FIXED: was 7, should be 11)
    3: 10,  # 卯-戌
    4: 9,   # 辰-酉
    5: 8,   # 巳-申
    6: 7,   # 午-未
}

# Branch-Chong map (Earthly Branches六冲): conflicting branch combinations
# Key: branch index (0-11), Value: opposing branch index
BRANCH_CHONG_MAP = {
    0: 6,   # 子-午
    1: 7,   # 丑-未
    2: 8,   # 寅-申
    3: 9,   # 卯-酉
    4: 10,  # 辰-戌
    5: 11,  # 巳-亥
}

# Branch-Sanhe map (Earthly Branches三合局): three-branch combined patterns
# Key: frozenset of branch indices, Value: resulting energy type
BRANCH_SANHE_MAP = {
    frozenset([8, 0, 4]): "water",     # 申子辰 - water formation
    frozenset([11, 3, 7]): "wood",     # 亥卯未 - wood formation
    frozenset([2, 6, 10]): "fire",     # 寅午戌 - fire formation
    frozenset([5, 9, 1]): "metal",     # 巳酉丑 - metal formation
}

# Branch-Hidden-Stem map (Earthly Branches藏干): hidden stems within each branch
# Key: branch index (0-11), Value: list of stem indices
BRANCH_HIDDEN_STEM_MAP = {
    0: [8],                    # 子: 癸
    1: [5, 6, 8],             # 丑: 己庚癸
    2: [0, 2, 4],             # 寅: 甲丙戊
    3: [1],                    # 卯: 乙
    4: [4, 1, 8],             # 辰: 戊乙癸
    5: [2, 6, 4],             # 巳: 丙庚戊
    6: [3, 5],                # 午: 丁己
    7: [5, 1, 3],             # 未: 己乙丁
    8: [6, 8, 4],             # 申: 庚壬戊
    9: [7],                    # 酉: 辛
    10: [4, 7, 3],            # 戌: 戊辛丁
    11: [8, 0],               # 亥: 壬甲
}


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

# Trigram Patterns相关别名
BAGUA_ALIAS = SEMANTIC_CATEGORY.copy()
BAGUA_NAMES = SEMANTIC_CATEGORY_NAMES.copy()

# Energy System相关别名
WUXING = ENERGY_TYPE.copy()
WUXING_SHENG = ENERGY_ENHANCE.copy()
WUXING_KE = ENERGY_SUPPRESS.copy()

# 旺相相关别名
WANGXIANG = STRENGTH_STATE.copy()
MONTH_WANG = MONTH_ENERGY_STATE.copy()
