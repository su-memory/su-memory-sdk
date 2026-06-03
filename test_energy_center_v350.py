"""
su-memory v3.5.0 能量中心全面测试

覆盖：
1. 三重导出链完整性 (_sys → world_model → SDK)
2. 新增 25 个符号功能测试
3. 天/地/人/三才合一 集成联动
4. 边界条件与异常处理
"""

import sys
import traceback

# ── 0. 环境准备 ──────────────────────────────────────────
sys.path.insert(0, 'src')

passed = 0
failed = 0
errors = []

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        msg = f"  ✗ {name} FAILED" + (f" — {detail}" if detail else "")
        print(msg)
        errors.append(msg)

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════
# 第1部分：三重导出链完整性
# ══════════════════════════════════════════════════════════════
section("1. 三重导出链完整性验证")

# 所有应导出符号（来自 _sys/__init__.py __all__）
EXPECTED_SYMBOLS = [
    # 基础类型层 — 枚举系统 (12)
    "YinYang", "ThreePowers", "FourSymbols", "Season",
    "TimeStem", "TimeBranch", "BranchRelation",
    "TrigramType", "TrigramRelation",
    "EnergyEnumType", "EnergyEnumRelation",
    "StrengthState", "EnergyPattern",
    # 基础类型层 — 数据字典 (18)
    "SEMANTIC_CATEGORY", "SEMANTIC_CATEGORY_NAMES",
    "TERMS_ENERGY_ENHANCE", "TERMS_ENERGY_SUPPRESS",
    "STRENGTH_STATE", "MONTH_ENERGY_STATE", "STRENGTH_MULTIPLIER",
    "TIME_STEMS", "TIME_BRANCHES", "TIME_BRANCH_ENERGY",
    "STEM_HE_MAP", "STEM_CHONG_MAP",
    "BRANCH_HE_MAP", "BRANCH_CHONG_MAP",
    "BRANCH_SANHE_MAP", "BRANCH_HIDDEN_STEM_MAP",
    "TRIGRAM_ENERGY_MAP", "TRIGRAM_BODY_MAP",
    # 基础类型层 — 语义/能量分类 (13)
    "SemanticCategory", "MEMORY_TYPE_TO_CATEGORY",
    "CATEGORY_ANCHORS", "KEYWORDS_TO_CATEGORY", "ENERGY_TO_CATEGORY",
    "C2EnergyType", "C2EnergyState", "EnergyNetwork",
    "ENERGY_ENHANCE_MAP", "ENERGY_SUPPRESS_MAP", "STATE_STRENGTH_MAP",
    "get_seasonal_energy_state", "check_state_interaction",
    "energy_similarity", "energy_from_category",
    # 天层 — 时空建模 (21)
    "TianGan", "DiZhi",
    "TemporalCore", "StemBranchCode", "create_stem_branch", "get_cycle_name",
    "TemporalSystem", "TemporalInfo", "DynamicPriority",
    "TimeCycle", "TimeCodeInfo", "create_time_code",
    "STEM_HE", "STEM_CHONG",
    "BRANCH_HE", "BRANCH_SANHE", "BRANCH_CHONG", "BRANCH_XING",
    "get_stem", "get_branch", "get_cycle",
    # 地层 — 卦象空间 (3)
    "TrigramCore", "TaijiMapper", "PatternInference",
    # 人层 — 能量与因果 (14)
    "EnergyCore", "EnergyStateInfo",
    "EnergyBalanceResult", "EnergyFlow",
    "EnergyBus", "EnergyNode", "EnergyChannel", "EnergySignal",
    "EnergyLayer", "PropagationConfig",
    "create_energy_bus", "create_complete_energy_network",
    "CategoryCausalEngine", "EnergyMemoryNode",
    # 能量关系 (20)
    "analyze_balance", "calculate_link_weight",
    "analyze_relation", "get_affinity_score",
    "surface_entities", "find_reverse_causal_chain",
    "is_enhancing", "is_suppressing",
    "get_enhanced_energy", "get_cycle_sequence",
    "get_enhance_relation", "get_suppress_relation",
    "get_suppress_chain",
    "get_enhancing_energy", "get_suppressed_energy", "get_suppressing_energy",
    "RelationType", "EnergyRelation", "EnergyRelationsType",
    "MemoryNodeEnergy", "RELATION_STRENGTH", "FOUR_SYMBOLS_TO_ENERGY",
    # 三才合一 (3)
    "UnifiedInfoUnit", "UnifiedInfoFactory", "create_unified_unit",
    # 元认知 (2)
    "CognitiveGap", "KnowledgeAging",
    # 检索融合 (1)
    "MultiViewRetriever",
]

# 1a. _sys 层
print("\n[1a] _sys/__init__.py 导出链")
from su_memory._sys import __all__ as sys_all
sys_all_set = set(sys_all)
for sym in EXPECTED_SYMBOLS:
    check(f"_sys.{sym}", sym in sys_all_set)

# 1b. world_model 层
print("\n[1b] world_model.py 导出链")
from su_memory.world_model import __all__ as wm_all
wm_all_set = set(wm_all)
# 内部别名（仅 _sys 使用，world_model 不导出）
_WM_EXCLUDE = {"TERMS_ENERGY_ENHANCE", "TERMS_ENERGY_SUPPRESS",
               "STRENGTH_STATE", "C2EnergyType", "C2EnergyState"}
for sym in EXPECTED_SYMBOLS:
    if sym in _WM_EXCLUDE:
        continue  # world_model 有意不导出这些内部别名
    check(f"world_model.{sym}", sym in wm_all_set)

# 1c. SDK 层
print("\n[1c] SDK __init__.py 导出链")
from su_memory import __all__ as sdk_all
sdk_all_set = set(sdk_all)
# SDK 有自己的 __all__ 格式，不包含某些内部别名
# 检查能量中心专属符号
sdk_energy_syms = [
    # 天层
    "TianGan", "DiZhi",
    "TemporalCore", "StemBranchCode", "create_stem_branch", "get_cycle_name",
    "TemporalSystem", "TemporalInfo", "DynamicPriority",
    "TimeCycle", "TimeCodeInfo", "create_time_code",
    # 地层
    "TrigramCore", "TaijiMapper", "PatternInference",
    # 人层
    "EnergyCore", "EnergyState",
    "EnergyBalanceResult", "EnergyFlow",
    "EnergyBus", "EnergyNode", "EnergyChannel", "EnergySignal",
    "PropagationConfig", "create_energy_bus", "create_complete_energy_network",
    "CategoryCausalEngine",
    # 能量关系
    "analyze_balance", "calculate_link_weight",
    "analyze_relation", "get_affinity_score",
    "surface_entities", "find_reverse_causal_chain",
    "is_enhancing", "is_suppressing",
    "get_enhanced_energy", "get_cycle_sequence",
    "get_enhance_relation", "get_suppress_relation", "get_suppress_chain",
    "get_enhancing_energy", "get_suppressed_energy", "get_suppressing_energy",
    "RelationType", "EnergyRelation",
    "MemoryNodeEnergy", "RELATION_STRENGTH", "FOUR_SYMBOLS_TO_ENERGY",
    # 三才合一
    "UnifiedInfoUnit", "UnifiedInfoFactory", "create_unified_unit",
    # 基础类型
    "YinYang", "ThreePowers", "FourSymbols", "Season",
    "TimeStem", "TimeBranch", "BranchRelation",
    "TrigramType", "TrigramRelation",
    "StrengthState", "EnergyPattern",
    "SemanticCategory", "MEMORY_TYPE_TO_CATEGORY",
    "EnergyNetwork", "ENERGY_ENHANCE_MAP", "ENERGY_SUPPRESS_MAP",
    "STEM_HE", "STEM_CHONG",
    "BRANCH_HE", "BRANCH_SANHE", "BRANCH_CHONG", "BRANCH_XING",
    "get_stem", "get_branch", "get_cycle",
    "get_energy_state", "check_state_interaction",
    "energy_similarity", "energy_from_category",
]
for sym in sdk_energy_syms:
    check(f"SDK.{sym}", sym in sdk_all_set)

# 1d. 三重链符号一致性（SDK 能实际 import）
print("\n[1d] SDK 实际 import 验证（懒加载触发）")
import su_memory
for sym in sdk_energy_syms:
    try:
        obj = getattr(su_memory, sym)
        check(f"SDK.getattr({sym})", obj is not None, f"type={type(obj).__name__}")
    except Exception as e:
        check(f"SDK.getattr({sym})", False, str(e))


# ══════════════════════════════════════════════════════════════
# 第2部分：新增25个符号功能测试
# ══════════════════════════════════════════════════════════════
section("2. 新增 25 个符号功能测试")

from su_memory._sys import (
    TianGan, DiZhi,
    EnergyStateInfo,
    EnergyChannel, EnergySignal,
    get_enhance_relation, get_suppress_relation, get_suppress_chain,
    get_enhancing_energy, get_suppressed_energy, get_suppressing_energy,
    MemoryNodeEnergy, RELATION_STRENGTH,
    STEM_HE, STEM_CHONG, BRANCH_HE, BRANCH_SANHE, BRANCH_CHONG, BRANCH_XING,
    get_stem, get_branch, get_cycle,
    get_seasonal_energy_state, check_state_interaction, energy_similarity, energy_from_category,
    EnergyCore, EnergyBus, EnergyLayer, EnergyNode,
    TemporalSystem,
)

# ── 2.1 TianGan / DiZhi ──
print("\n[2.1] TianGan / DiZhi Time Code data classes")
check("TianGan.NAMES[0]=='甲'", TianGan.NAMES[0] == '甲')
check("TianGan.NAMES[9]=='癸'", TianGan.NAMES[9] == '癸')
check("TianGan.CATEGORY[0]=='wood'", TianGan.CATEGORY[0] == 'wood')
check("TianGan.YIN_YANG[0]==1 (阳)", TianGan.YIN_YANG[0] == 1)
check("DiZhi.NAMES[0]=='子'", DiZhi.NAMES[0] == '子')
check("DiZhi.NAMES[11]=='亥'", DiZhi.NAMES[11] == '亥')
check("DiZhi.LIUHE[0]==1 (子丑合)", DiZhi.LIUHE[0] == 1)
check("DiZhi.SANHE 共4组", len(DiZhi.SANHE) == 4)

# ── 2.2 EnergyStateInfo ──
print("\n[2.2] EnergyStateInfo 能量状态数据类")
ec = EnergyCore()
state = ec.get_energy_state('wood', 2)  # 寅月木旺
check("state 是 EnergyStateInfo 类型", type(state).__name__ == 'EnergyState')
check("state.energy_type", state.energy_type is not None)
check("state.strength", state.strength is not None)
check("state.intensity>0", state.intensity > 0)
check("state.is_enhanced (寅月木旺)", state.is_enhanced == True)

# ── 2.3 EnergyChannel / EnergySignal ──
print("\n[2.3] EnergyChannel / EnergySignal 能量传播通道")
from su_memory._sys._energy_relations import RelationType as RT
ch = EnergyChannel(
    channel_id="ch_test",
    source_id="node_a",
    target_id="node_b",
    relation_type=RT.ENHANCE,
    base_weight=1.0
)
check("EnergyChannel.effective_weight (enhance)", abs(ch.effective_weight - 1.2) < 0.01)

sig = EnergySignal(
    signal_id="sig_1",
    source_node="a",
    target_node="b",
    energy_type="wood",
    intensity=0.5,
    timestamp=1234567890.0,
    layer=EnergyLayer.FIVE_ELEMENTS,
    ttl=3
)
check("EnergySignal.signal_id", sig.signal_id == "sig_1")
check("EnergySignal.intensity", sig.intensity == 0.5)
check("EnergySignal.layer", sig.layer == EnergyLayer.FIVE_ELEMENTS)

# ── 2.4 能量关系原子函数 ──
print("\n[2.4] 能量关系原子函数")
check("get_enhance_relation(wood,fire)", get_enhance_relation("wood", "fire") == True)
check("get_enhance_relation(fire,wood)", get_enhance_relation("fire", "wood") == False)
check("get_suppress_relation(wood,earth)", get_suppress_relation("wood", "earth") == True)
check("get_suppress_relation(water,fire)", get_suppress_relation("water", "fire") == True)
check("get_enhancing_energy(fire)", get_enhancing_energy("fire") == "wood")
check("get_suppressed_energy(wood)", get_suppressed_energy("wood") == "earth")
check("get_suppressing_energy(earth)", get_suppressing_energy("earth") == "wood")
chain = get_suppress_chain("wood", 5)
check("get_suppress_chain(wood,5) len", len(chain) == 5)
check("get_suppress_chain(wood,5) sequence",
      chain == ["wood", "earth", "water", "fire", "metal"])

# ── 2.5 MemoryNodeEnergy / RELATION_STRENGTH ──
print("\n[2.5] MemoryNodeEnergy / RELATION_STRENGTH")
mne = MemoryNodeEnergy(node_id="m1", energy_type="wood", intensity=0.8,
                       stem_idx=2, branch_idx=3)
check("MemoryNodeEnergy.node_id", mne.node_id == "m1")
check("MemoryNodeEnergy.stem_idx", mne.stem_idx == 2)
check("RELATION_STRENGTH keys", len(RELATION_STRENGTH) == 6)
check("RELATION_STRENGTH[ENHANCE]", abs(RELATION_STRENGTH[RT.ENHANCE] - 1.2) < 0.01)

# ── 2.6 时空量化符号 ──
print("\n[2.6] 时空量化符号 (STEM/Branch 关系)")
check("STEM_HE keys", len(STEM_HE) == 5)
check("STEM_CHONG keys", len(STEM_CHONG) == 5)
check("BRANCH_HE keys", len(BRANCH_HE) == 6)
check("BRANCH_SANHE entries", len(BRANCH_SANHE) == 4)
check("BRANCH_CHONG keys", len(BRANCH_CHONG) == 6)
check("BRANCH_XING entries", len(BRANCH_XING) == 7)
check("get_stem(0)=='甲'", get_stem(0) == '甲')
check("get_stem(9)=='癸'", get_stem(9) == '癸')
check("get_branch(0)=='子'", get_branch(0) == '子')
check("get_cycle(0)", len(get_cycle(0)) == 2)  # "甲子"

# ── 2.7 C2 能量工具函数 ──
print("\n[2.7] C2 能量工具函数")
from su_memory._sys._c2 import EnergyType as C2ET
state_name, multiplier = get_seasonal_energy_state(C2ET.WOOD, C2ET.WOOD)
check("get_seasonal_energy_state(W,W)→strong", state_name == "strong")
check("get_seasonal_energy_state(W,W)→2.0", multiplier == 2.0)
check("energy_similarity(W,F)", abs(energy_similarity(C2ET.WOOD, C2ET.FIRE) - 0.7) < 0.01)
check("energy_similarity(W,E)", abs(energy_similarity(C2ET.WOOD, C2ET.EARTH) - 0.1) < 0.01)
interaction = check_state_interaction(C2ET.WOOD, C2ET.EARTH, 3.0, 1.0)
check("check_state_interaction(overwhelming)", interaction == "overwhelming")
result = energy_from_category("thunder")
check("energy_from_category(thunder)", result == C2ET.EARTH)


# ══════════════════════════════════════════════════════════════
# 第3部分：天/地/人/三才合一 集成联动
# ══════════════════════════════════════════════════════════════
section("3. 天/地/人/三才合一 集成联动")

from su_memory._sys import (
    TemporalCore, StemBranchCode,
    TrigramCore,
    EnergyCore, EnergyBus, EnergyNode, EnergyLayer,
    UnifiedInfoFactory,
)
from su_memory._sys._energy_relations import analyze_relation, RelationType

# ── 3.1 天层：时间编码 → 能量分析 ──
print("\n[3.1] 天层→人层：时间编码到能量分析")
tc = TemporalCore()
code = tc.create_code(stem_idx=0, branch_idx=2)  # 甲寅
check("TemporalCore.create_code(0,2)", code is not None)
check("code.name", code.name is not None)
check("code.cycle_index", code.cycle_index is not None)

# 获取 stem/branch
check("code.stem", code.stem is not None)
check("code.branch", code.branch is not None)

# 用 EnergyCore 分析当月令下能量状态
ec2 = EnergyCore()
energy_state = ec2.get_energy_state('wood', code.branch.value % 12)
check("天→人: 能量状态类型", type(energy_state).__name__ == 'EnergyState')
check("天→人: 有有效强度", energy_state.intensity > 0)

# ── 3.2 地层：卦象 → 能量总线 ──
print("\n[3.2] 地层→人层：卦象到能量传播")
from su_memory._sys._enums import TrigramType

trigram_core = TrigramCore()
info = trigram_core.get_trigram_info(TrigramType.QIAN)
check("TrigramCore.get_trigram_info(QIAN)", info is not None)

bus = EnergyBus()
wood_node = EnergyNode("wuxing_wood", "wood", EnergyLayer.FIVE_ELEMENTS)
fire_node = EnergyNode("wuxing_fire", "fire", EnergyLayer.FIVE_ELEMENTS)
bus.add_node(wood_node)
bus.add_node(fire_node)
bus.connect("wuxing_wood", "wuxing_fire", RelationType.ENHANCE)

state_before = bus.get_node_state("wuxing_fire")
signals = bus.propagate_energy("wuxing_wood", 0.5)
state_after = bus.get_node_state("wuxing_fire")
check("EnergyBus.propagate 产生信号", len(signals) > 0)
check("地→人: 传播后目标强度增加", state_after["intensity"] > state_before["intensity"])

# ── 3.3 三才合一 ──
print("\n[3.3] 三才合一：UnifiedInfoFactory")
factory = UnifiedInfoFactory()
try:
    unit = factory.create_from_temporal_code(0, 0)  # 甲子
    check("UnifiedInfoFactory.create_from_temporal_code(0,0)", unit is not None)
    check("甲子: cyclic_code == 0", unit.cyclic_code == 0)
    check("甲子: energy_type == 4 (trust/water)", unit.energy_type == 4)
except Exception as e:
    check(f"UnifiedInfoFactory", False, str(e))

# Test invalid Yin-Yang pairing (should raise ValueError)
try:
    factory.create_from_temporal_code(4, 5)  # 戊巳: yang stem + yin branch
    check("非法干支配对应抛 ValueError", False, "未抛异常")
except ValueError as e:
    check(f"非法干支配对正确抛出 ValueError", True, str(e)[:60])

# ── 3.4 能量关系集成：生克→亲和度→实体浮现→溯源 ──
print("\n[3.4] 能量关系全链路：生克→亲和度→surface→reverse_causal")
from su_memory._sys import (
    get_affinity_score, surface_entities, find_reverse_causal_chain,
    calculate_link_weight
)
affinity = get_affinity_score("wood", "fire")
check("get_affinity_score(wood,fire) > 1.0", affinity > 1.0)

surfaced = surface_entities("fire")
check("surface_entities(fire) 返回结果", len(surfaced) > 0)

chains = find_reverse_causal_chain("fire", depth=2)
check("find_reverse_causal_chain(fire,2) 有链", len(chains) > 0)

weight = calculate_link_weight("wood", "fire", 1.0)
check("calculate_link_weight(wood,fire) > 1.0", weight > 1.0)


# ══════════════════════════════════════════════════════════════
# 第4部分：边界条件与异常处理
# ══════════════════════════════════════════════════════════════
section("4. 边界条件与异常处理")

# ── 4.1 空值/null 安全 ──
print("\n[4.1] 空值/null 安全")
try:
    result = analyze_relation("", "fire")
    check("analyze_relation('',fire) 不崩溃", True)
except Exception as e:
    check("analyze_relation('',fire) 不崩溃", False, str(e))

try:
    result = get_enhance_relation("", "")
    check("get_enhance_relation('','') 不崩溃", True)
except Exception as e:
    check("get_enhance_relation('','') 不崩溃", False, str(e))

# ── 4.2 无效输入 ──
print("\n[4.2] 无效输入处理")
try:
    ec3 = EnergyCore()
    ec3.get_energy_state('wood', 99)
    check("EnergyCore.get_energy_state(month=99) 应抛异常", False)
except ValueError:
    check("EnergyCore.get_energy_state(month=99) 抛ValueError", True)

try:
    surface_entities("")
    check("surface_entities('') 应抛异常", False)
except ValueError:
    check("surface_entities('') 抛ValueError", True)

# ── 4.3 循环引用 ──
print("\n[4.3] 循环引用防护")
# 能量传播中的 visited 集合防止死循环
bus2 = EnergyBus()
a = EnergyNode("loop_a", "wood", EnergyLayer.FIVE_ELEMENTS)
b = EnergyNode("loop_b", "fire", EnergyLayer.FIVE_ELEMENTS)
c = EnergyNode("loop_c", "earth", EnergyLayer.FIVE_ELEMENTS)
bus2.add_node(a); bus2.add_node(b); bus2.add_node(c)
bus2.connect("loop_a", "loop_b", RelationType.ENHANCE)
bus2.connect("loop_b", "loop_c", RelationType.ENHANCE)
bus2.connect("loop_c", "loop_a", RelationType.ENHANCE)  # 循环！

try:
    signals = bus2.propagate_energy("loop_a", 1.0, max_hops=10)
    check("循环传播不崩溃", True)
    check("循环传播产生有限信号", len(signals) < 50)  # visited 防无限
except Exception as e:
    check("循环传播不崩溃", False, str(e))

# ── 4.4 能量强度边界 ──
print("\n[4.4] 能量强度边界")
node_test = EnergyNode("test", "wood", EnergyLayer.FIVE_ELEMENTS, intensity=5.0)
check("intensity 被 clamp 到 max_intensity(2.0)", abs(node_test.intensity - 2.0) < 0.01)
node_test.adjust_intensity(-10.0)
check("intensity 最低不低于 0", node_test.intensity >= 0.0)

# ── 4.5 calculate_link_weight 边界 ──
print("\n[4.5] calculate_link_weight 边界")
w_neutral = calculate_link_weight("wood", "metal", 1.0)
check("无关元素 link_weight 正常", 0 < w_neutral <= 1.5)

# ── 4.6 时空编码边界 ──
print("\n[4.6] 时空编码边界")
try:
    from su_memory._sys import create_time_code
    tci = create_time_code(0, 0)
    check("create_time_code(0,0) 成功", tci is not None)
    tci2 = create_time_code(9, 11)
    check("create_time_code(9,11) 成功", tci2 is not None)
    # 超出范围的也应该循环
    tci3 = create_time_code(15, 15)
    check("create_time_code(15,15) 循环取模", tci3 is not None)
except Exception as e:
    check("create_time_code 边界", False, str(e))


# ══════════════════════════════════════════════════════════════
# 第5部分：Consistency Check — world_model vs _sys
# ══════════════════════════════════════════════════════════════
section("5. world_model 与 _sys 符号一致性")

from su_memory._sys import __all__ as sys_all_list
from su_memory.world_model import __all__ as wm_all_list

# World model 有意不导出的内部别名（仅 _sys 内部使用）
WM_EXCLUDED = {"TERMS_ENERGY_ENHANCE", "TERMS_ENERGY_SUPPRESS",
               "STRENGTH_STATE", "C2EnergyType", "C2EnergyState"}

sys_energy = {s for s in sys_all_list if s in set(EXPECTED_SYMBOLS)}
wm_energy = set(wm_all_list)

missing_in_wm = (sys_energy - wm_energy) - WM_EXCLUDED
extra_in_wm = wm_energy - sys_energy

if missing_in_wm:
    print(f"\n⚠️  world_model 缺少 {len(missing_in_wm)} 个符号:")
    for s in sorted(missing_in_wm):
        print(f"   - {s}")

if extra_in_wm:
    print(f"\n⚠️  world_model 多余 {len(extra_in_wm)} 个符号:")
    for s in sorted(extra_in_wm):
        print(f"   + {s}")

if not missing_in_wm and not extra_in_wm:
    print("✓ world_model 与 _sys 导出完全一致")

# 验证 world_model 和 _sys 导入同一对象
print("\n[5.1] 对象身份一致性 (is check)")
from su_memory._sys import TianGan as sys_TianGan
from su_memory.world_model import TianGan as wm_TianGan
check("_sys.TianGan is world_model.TianGan", sys_TianGan is wm_TianGan)

from su_memory._sys import EnergyCore as sys_EnergyCore
from su_memory.world_model import EnergyCore as wm_EnergyCore
check("_sys.EnergyCore is world_model.EnergyCore", sys_EnergyCore is wm_EnergyCore)


# ══════════════════════════════════════════════════════════════
# 结果汇总
# ══════════════════════════════════════════════════════════════
section("测试结果汇总")
total = passed + failed
print(f"\n  ✓ 通过: {passed}")
print(f"  ✗ 失败: {failed}")
print(f"  总计: {total}")
print(f"  通过率: {passed / total * 100:.1f}%")

if errors:
    print(f"\n失败详情:")
    for e in errors:
        print(f"  {e}")

print()
if failed == 0:
    print("🎉 所有测试通过！能量中心 v3.5.0 API 完整释放验证成功。")
else:
    print(f"⚠️  {failed} 个测试失败，需要修复。")
    sys.exit(1)
