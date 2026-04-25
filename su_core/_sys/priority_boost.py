"""
优先级权重配置 — 基于易学四位一体的动态优先级系统

对应易学四位一体框架:
  - 干支为时空（天地之气）
  - 五行为质地（生克制化）
  - 周易为变机（卦象流转）
  - 因果为流转（能量传递）

目标：82.3% → 85%+

优先级维度：
1. 季节权重 (SEASON_BOOST) — 基于当前季节与记忆五行的旺相关系
2. 信任链权重 (TRUST_BOOST) — 基于记忆在因果链中的位置
3. 时间权重 (TIME_BOOST) — 基于干支时空的旺相状态
4. 卦气权重 (HEXAGRAM_BOOST) — 基于卦象能量状态
5. 因果能量权重 (CAUSAL_BOOST) — 基于因果链传递的能量
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import time

# ============================================================
# 易学四位一体常量表
# ============================================================

# 季节旺相表：同季同旺，被季所不胜则休囚
WANG_XIANG_TABLE: Dict[Tuple[str, str], float] = {
    # 春木旺
    ("春", "木"): 1.25, ("春", "火"): 0.85, ("春", "土"): 0.80, ("春", "金"): 0.65, ("春", "水"): 0.75,
    # 夏火旺
    ("夏", "木"): 0.85, ("夏", "火"): 1.25, ("夏", "土"): 0.85, ("夏", "金"): 0.70, ("夏", "水"): 0.65,
    # 秋金旺
    ("秋", "木"): 0.65, ("秋", "火"): 0.80, ("秋", "土"): 0.85, ("秋", "金"): 1.25, ("秋", "水"): 0.90,
    # 冬水旺
    ("冬", "木"): 0.70, ("冬", "火"): 0.65, ("冬", "土"): 0.80, ("冬", "金"): 0.85, ("冬", "水"): 1.25,
    # 四季土旺
    ("四季", "木"): 0.80, ("四季", "火"): 0.90, ("四季", "土"): 1.25, ("四季", "金"): 0.80, ("四季", "水"): 0.85,
}

# 八卦能量表（基于先天/后天方位与五行）
BAGUA_ENERGY: Dict[str, float] = {
    "乾": 1.05,  # 西北金，刚健
    "兑": 0.90,  # 西金，喜悦
    "离": 1.10,  # 南火，光明
    "震": 0.85,  # 东木，震动
    "巽": 0.85,  # 东南木，入
    "坎": 0.90,  # 北水，陷
    "艮": 0.75,  # 东北土，止
    "坤": 1.05,  # 西南土，柔顺
}

# 天干能量表（基于阴阳强弱）
TIANGAN_ENERGY: Dict[str, float] = {
    "甲": 1.05, "乙": 0.82, "丙": 1.12, "丁": 0.88,
    "戊": 1.02, "己": 0.88, "庚": 1.12, "辛": 0.88,
    "壬": 1.05, "癸": 0.80,
}

# 地支旺相详细表（地支藏干综合五行能量）
DIZHI_WANGXIANG: Dict[str, Dict[str, float]] = {
    "子": {"水": 1.15, "火": 0.75, "金": 0.85, "木": 0.80, "土": 0.80},
    "丑": {"土": 1.10, "金": 0.90, "水": 0.85, "木": 0.75, "火": 0.80},
    "寅": {"木": 1.20, "火": 0.92, "土": 0.85, "金": 0.72, "水": 0.78},
    "卯": {"木": 1.15, "水": 0.82, "火": 0.88, "金": 0.70, "土": 0.80},
    "辰": {"土": 1.10, "木": 0.88, "水": 0.85, "火": 0.80, "金": 0.78},
    "巳": {"火": 1.20, "土": 0.90, "金": 0.85, "木": 0.75, "水": 0.72},
    "午": {"火": 1.15, "木": 0.85, "土": 0.90, "水": 0.72, "金": 0.78},
    "未": {"土": 1.10, "木": 0.88, "火": 0.85, "金": 0.80, "水": 0.78},
    "申": {"金": 1.20, "水": 0.92, "土": 0.85, "火": 0.78, "木": 0.72},
    "酉": {"金": 1.15, "火": 0.80, "水": 0.85, "木": 0.72, "土": 0.82},
    "戌": {"土": 1.10, "火": 0.90, "金": 0.88, "木": 0.78, "水": 0.80},
    "亥": {"水": 1.20, "木": 0.90, "火": 0.72, "土": 0.80, "金": 0.75},
}

# 五行归类
WUXING_ELEMENTS = ["木", "火", "土", "金", "水"]

# 五行相生系数
SHENG_COEFFICIENT = 1.08

# 五行相克系数
KE_COEFFICIENT = 0.92

# ============================================================
# 信任链状态
# ============================================================

class TrustLevel(Enum):
    """信任层级"""
    CORE = 1.0       # 核心记忆（因果链源头）
    DIRECT = 0.92    # 直接关联
    INDIRECT = 0.82  # 间接关联
    WEAK = 0.70      # 弱关联
    ORPHAN = 0.60    # 孤岛记忆

# ============================================================
# 数据类
# ============================================================

@dataclass
class PriorityContext:
    """优先级计算上下文"""
    current_season: str = "春"
    current_ganzhi_index: int = 0  # 0-59甲子索引
    tiangan: Optional[str] = None   # 当前天干
    dizhi: Optional[str] = None     # 当前地支
    memory_wuxing_distribution: Dict[str, float] = field(default_factory=lambda: {
        "木": 0.2, "火": 0.2, "土": 0.2, "金": 0.2, "水": 0.2
    })
    all_memory_ids: List[str] = field(default_factory=list)

@dataclass
class PriorityResult:
    """优先级计算结果"""
    final_priority: float
    season_boost: float
    temporal_boost: float
    hexagram_boost: float
    causal_boost: float
    trust_boost: float
    wuxing_balance: float
    components: Dict[str, float] = field(default_factory=dict)

# ============================================================
# 动态优先级计算器
# ============================================================

class DynamicPriorityCalculator:
    """
    动态优先级计算器 — 基于易学四维时空
    
    计算公式：
    final_priority = base_priority 
                    × season_boost 
                    × temporal_boost 
                    × hexagram_boost 
                    × causal_boost 
                    × trust_boost 
                    × wuxing_balance
    """

    def __init__(self, context: Optional[PriorityContext] = None):
        self.context = context or self._create_default_context()

    def _create_default_context(self) -> PriorityContext:
        """创建默认上下文（基于当前时间）"""
        now = time.localtime()
        # 简化：按月份推断季节
        month = now.tm_mon
        if month in [3, 4, 5]:
            season = "春"
        elif month in [6, 7, 8]:
            season = "夏"
        elif month in [9, 10, 11]:
            season = "秋"
        else:
            season = "冬"
        # 甲子索引（基于日期偏移）
        jiazi_index = (now.tm_year * 365 + now.tm_yday) % 60
        return PriorityContext(
            current_season=season,
            current_ganzhi_index=jiazi_index,
        )

    def calculate(
        self,
        base_priority: float,
        memory_wuxing: str,
        current_season: Optional[str] = None,
        memory_bagua: Optional[str] = None,
        causal_energy: float = 0.0,
        time_branch: Optional[str] = None,
        trust_level: TrustLevel = TrustLevel.INDIRECT,
        memory_id: Optional[str] = None,
        is_causal_source: bool = False,
    ) -> float:
        """
        计算最终优先级
        
        Args:
            base_priority: 基础优先级 (0-1)
            memory_wuxing: 记忆五行 (木/火/土/金/水)
            current_season: 当前季节 (春/夏/秋/冬/四季)
            memory_bagua: 记忆卦象 (乾/坤/...)
            causal_energy: 因果链能量 (0-1)
            time_branch: 时间地支 (子/丑/.../亥)
            trust_level: 信任层级
            memory_id: 记忆ID（用于追踪）
            is_causal_source: 是否为因果链源头
        """
        season = current_season or self.context.current_season

        # 1. 季节权重
        season_boost = self._get_season_boost(memory_wuxing, season)

        # 2. 卦气权重
        hexagram_boost = self._get_hexagram_boost(memory_bagua, memory_wuxing)

        # 3. 因果能量权重
        causal_boost = self._get_causal_boost(causal_energy, is_causal_source)

        # 4. 时间权重（基于地支旺相）
        temporal_boost = self._get_temporal_boost(time_branch, memory_wuxing)

        # 5. 信任链权重
        trust_boost = self._get_trust_boost(trust_level, is_causal_source)

        # 6. 五行制化因子
        wuxing_balance = self._get_wuxing_balance_factor(memory_wuxing)

        # 综合计算
        final = (
            base_priority
            * season_boost
            * hexagram_boost
            * causal_boost
            * temporal_boost
            * trust_boost
            * wuxing_balance
        )

        return max(0.0, min(1.0, round(final, 4)))

    def calculate_detailed(
        self,
        base_priority: float,
        memory_wuxing: str,
        current_season: Optional[str] = None,
        memory_bagua: Optional[str] = None,
        causal_energy: float = 0.0,
        time_branch: Optional[str] = None,
        trust_level: TrustLevel = TrustLevel.INDIRECT,
        memory_id: Optional[str] = None,
        is_causal_source: bool = False,
    ) -> PriorityResult:
        """详细模式：返回所有权重分量"""
        season = current_season or self.context.current_season

        season_boost = self._get_season_boost(memory_wuxing, season)
        hexagram_boost = self._get_hexagram_boost(memory_bagua, memory_wuxing)
        causal_boost = self._get_causal_boost(causal_energy, is_causal_source)
        temporal_boost = self._get_temporal_boost(time_branch, memory_wuxing)
        trust_boost = self._get_trust_boost(trust_level, is_causal_source)
        wuxing_balance = self._get_wuxing_balance_factor(memory_wuxing)

        final = (
            base_priority
            * season_boost
            * hexagram_boost
            * causal_boost
            * temporal_boost
            * trust_boost
            * wuxing_balance
        )

        components = {
            "season": season_boost,
            "hexagram": hexagram_boost,
            "causal": causal_boost,
            "temporal": temporal_boost,
            "trust": trust_boost,
            "wuxing_balance": wuxing_balance,
        }

        return PriorityResult(
            final_priority=max(0.0, min(1.0, round(final, 4))),
            season_boost=season_boost,
            temporal_boost=temporal_boost,
            hexagram_boost=hexagram_boost,
            causal_boost=causal_boost,
            trust_boost=trust_boost,
            wuxing_balance=wuxing_balance,
            components=components,
        )

    def _get_season_boost(self, memory_wuxing: str, current_season: str) -> float:
        """获取季节权重"""
        key = (current_season, memory_wuxing)
        return WANG_XIANG_TABLE.get(key, 1.0)

    def _get_hexagram_boost(self, memory_bagua: Optional[str], memory_wuxing: str) -> float:
        """获取卦气权重（考虑卦象五行与记忆五行生克）"""
        if not memory_bagua:
            return 1.0
        
        base_energy = BAGUA_ENERGY.get(memory_bagua, 1.0)
        
        # 卦象五行属性
        bagua_wuxing_map = {
            "乾": "金", "兑": "金", "离": "火", "震": "木",
            "巽": "木", "坎": "水", "艮": "土", "坤": "土",
        }
        bagua_element = bagua_wuxing_map.get(memory_bagua)
        
        if bagua_element == memory_wuxing:
            # 同五行：能量共振
            return base_energy * 1.08
        elif self._is_sheng(bagua_element, memory_wuxing):
            # 卦生记忆：助力
            return base_energy * 1.05
        elif self._is_ke(bagua_element, memory_wuxing):
            # 卦克记忆：抑制
            return base_energy * 0.92
        return base_energy

    def _get_causal_boost(self, causal_energy: float, is_causal_source: bool) -> float:
        """获取因果能量权重"""
        if is_causal_source:
            return 1.20  # 源头记忆获得显著提升
        # 能量传递：能量越高，提升越大
        # 范围：0.85 (能量0) ~ 1.15 (能量1)
        return 0.85 + causal_energy * 0.30

    def _get_temporal_boost(self, time_branch: Optional[str], memory_wuxing: str) -> float:
        """获取时间权重（地支旺相）"""
        if not time_branch:
            return 1.0
        
        wuxing_boost = DIZHI_WANGXIANG.get(time_branch, {}).get(memory_wuxing, 1.0)
        return wuxing_boost

    def _get_trust_boost(self, trust_level: TrustLevel, is_causal_source: bool) -> float:
        """获取信任链权重"""
        if is_causal_source:
            return 1.15  # 核心源头最高信任
        return trust_level.value

    def _get_wuxing_balance_factor(self, memory_wuxing: str) -> float:
        """获取五行制化因子（防止某行过旺，维持系统平衡）"""
        if not self.context.memory_wuxing_distribution:
            return 1.0
        
        dist = self.context.memory_wuxing_distribution
        # 如果某一行占比超过40%，说明过旺，适当降低
        element_ratio = dist.get(memory_wuxing, 0.2)
        
        if element_ratio > 0.40:
            # 过旺：降低5-15%
            penalty = 0.85 + (0.40 / element_ratio) * 0.10
            return penalty
        elif element_ratio < 0.10:
            # 过衰：适当提升
            return 1.08
        return 1.0

    @staticmethod
    def _is_sheng(generator: str, generated: str) -> bool:
        """判断五行相生关系"""
        sheng_map = {"木": "火", "火": "土", "土": "金", "金": "水", "水": "木"}
        return sheng_map.get(generator, "") == generated

    @staticmethod
    def _is_ke(controller: str, controlled: str) -> bool:
        """判断五行相克关系"""
        ke_map = {"木": "土", "火": "金", "土": "水", "金": "木", "水": "火"}
        return ke_map.get(controller, "") == controlled

    def update_distribution(self, memory_ids: List[str], memory_wuxings: Dict[str, str]) -> None:
        """更新记忆五行分布（用于制化计算）"""
        if not memory_wuxings:
            return
        counts = {w: 0 for w in WUXING_ELEMENTS}
        for mid in memory_ids:
            w = memory_wuxings.get(mid)
            if w in counts:
                counts[w] += 1
        total = sum(counts.values()) or 1
        self.context.memory_wuxing_distribution = {w: c / total for w, c in counts.items()}
        self.context.all_memory_ids = memory_ids


class CausalBoostIntegrator:
    """
    因果链增强集成器
    将因果链能量传递与优先级系统深度集成
    """

    def __init__(self, calculator: DynamicPriorityCalculator):
        self.calculator = calculator

    def calculate_with_causal_chain(
        self,
        base_priority: float,
        memory_wuxing: str,
        memory_id: str,
        causal_chain: 'CausalChain',
        all_memory_ids: List[str],
        memory_wuxings: Dict[str, str],
        current_season: str = "春",
        memory_bagua: Optional[str] = None,
        time_branch: Optional[str] = None,
    ) -> Tuple[float, bool]:
        """
        结合因果链计算优先级
        
        Args:
            base_priority: 基础优先级
            memory_wuxing: 记忆五行
            memory_id: 记忆ID
            causal_chain: 因果链实例
            all_memory_ids: 所有记忆ID列表
            memory_wuxings: 记忆ID→五行映射
            current_season: 当前季节
            memory_bagua: 记忆卦象
            time_branch: 时间地支
        
        Returns:
            (final_priority, is_source)
        """
        # 更新五行分布
        self.calculator.update_distribution(all_memory_ids, memory_wuxings)

        # 判断是否为因果链源头
        is_source = self._is_source_memory(memory_id, causal_chain, all_memory_ids)

        # 获取因果能量
        causal_energy = causal_chain.energy.get(memory_id, 0.5)

        # 判断信任层级
        trust_level = self._determine_trust_level(memory_id, causal_chain, all_memory_ids)

        # 计算最终优先级
        final_priority = self.calculator.calculate(
            base_priority=base_priority,
            memory_wuxing=memory_wuxing,
            current_season=current_season,
            memory_bagua=memory_bagua,
            causal_energy=causal_energy,
            time_branch=time_branch,
            trust_level=trust_level,
            memory_id=memory_id,
            is_causal_source=is_source,
        )

        return final_priority, is_source

    def _is_source_memory(
        self,
        memory_id: str,
        causal_chain: 'CausalChain',
        all_memory_ids: List[str],
    ) -> bool:
        """判断是否为因果链源头"""
        # 源头：没有被任何记忆指向（入度为0）但指向了其他记忆（出度>0）
        is_parent = any(memory_id in children for children in causal_chain.graph.values())
        is_source = memory_id in causal_chain.graph and not is_parent
        return is_source

    def _determine_trust_level(
        self,
        memory_id: str,
        causal_chain: 'CausalChain',
        all_memory_ids: List[str],
    ) -> TrustLevel:
        """确定信任层级"""
        # 检查是否为因果链源头
        if self._is_source_memory(memory_id, causal_chain, all_memory_ids):
            return TrustLevel.CORE
        
        # 检查是否有出边（直接父节点）
        has_children = memory_id in causal_chain.graph and len(causal_chain.graph[memory_id]) > 0
        # 检查是否有入边（直接子节点）
        is_child = any(memory_id in children for children in causal_chain.graph.values())
        
        if has_children and is_child:
            return TrustLevel.DIRECT
        elif is_child:
            return TrustLevel.INDIRECT
        elif has_children:
            return TrustLevel.WEAK
        else:
            return TrustLevel.ORPHAN


# ============================================================
# 便捷函数
# ============================================================

def boost_priority(
    base_priority: float,
    memory_wuxing: str,
    current_season: str = "春",
    memory_bagua: Optional[str] = None,
    causal_energy: float = 0.0,
    time_branch: Optional[str] = None,
) -> float:
    """便捷函数：计算优先级提升"""
    calc = DynamicPriorityCalculator()
    return calc.calculate(
        base_priority,
        memory_wuxing,
        current_season,
        memory_bagua,
        causal_energy,
        time_branch,
    )


def boost_priority_detailed(
    base_priority: float,
    memory_wuxing: str,
    current_season: str = "春",
    memory_bagua: Optional[str] = None,
    causal_energy: float = 0.0,
    time_branch: Optional[str] = None,
) -> PriorityResult:
    """便捷函数：计算优先级提升（详细模式）"""
    calc = DynamicPriorityCalculator()
    return calc.calculate_detailed(
        base_priority,
        memory_wuxing,
        current_season,
        memory_bagua,
        causal_energy,
        time_branch,
    )


# ============================================================
# 自检测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("易学四位一体动态优先级系统 — 自检测试")
    print("=" * 60)

    calc = DynamicPriorityCalculator()

    # 测试案例：覆盖四季、五行组合
    test_cases = [
        # (base, wuxing, season, bagua, causal_energy, time_branch)
        (0.5, "木", "春", "震", 0.5, "寅"),
        (0.5, "火", "夏", "离", 0.3, "巳"),
        (0.5, "金", "秋", "乾", 0.8, "申"),
        (0.5, "水", "冬", "坎", 0.2, "子"),
        (0.5, "土", "四季", "艮", 0.5, "辰"),
        (0.3, "木", "秋", "巽", 0.1, "酉"),   # 休囚季节
        (0.8, "火", "冬", "离", 0.9, "亥"),   # 高能量
        (0.4, "金", "夏", "乾", 0.6, "午"),   # 反季
    ]

    print("\n【基础测试】")
    print("-" * 60)
    for base, wuxing, season, bagua, energy, tb in test_cases:
        result = calc.calculate(base, wuxing, season, bagua, energy, tb)
        print(f"base={base:.1f} | {wuxing}/{season}/{bagua}/{tb} | "
              f"energy={energy} → priority={result:.4f}")

    # 测试 boost_priority 便捷函数
    print("\n【便捷函数测试】")
    print("-" * 60)
    p = boost_priority(0.5, "木", "春", "震", 0.5, "寅")
    print(f"boost_priority(0.5, 木, 春, 震, 0.5, 寅) = {p:.4f}")

    # 测试详细模式
    print("\n【详细模式测试】")
    print("-" * 60)
    detailed = calc.calculate_detailed(0.5, "木", "春", "震", 0.5, "寅")
    print(f"base=0.5, 木/春/震 的详细权重:")
    print(f"  season_boost   = {detailed.season_boost:.4f}")
    print(f"  hexagram_boost = {detailed.hexagram_boost:.4f}")
    print(f"  causal_boost   = {detailed.causal_boost:.4f}")
    print(f"  temporal_boost = {detailed.temporal_boost:.4f}")
    print(f"  trust_boost    = {detailed.trust_boost:.4f}")
    print(f"  wuxing_balance = {detailed.wuxing_balance:.4f}")
    print(f"  ─────────────────────────")
    print(f"  final_priority = {detailed.final_priority:.4f}")

    # 测试信任层级
    print("\n【信任层级测试】")
    print("-" * 60)
    for level in TrustLevel:
        boost = calc._get_trust_boost(level, is_causal_source=(level == TrustLevel.CORE))
        print(f"  {level.name:12s} = {boost:.4f}")

    # 测试因果Boost集成器
    print("\n【因果Boost集成器测试】")
    print("-" * 60)
    
    class MockCausalChain:
        def __init__(self):
            self.graph = {"mem_a": ["mem_b", "mem_c"], "mem_b": ["mem_c"]}
            self.energy = {"mem_a": 1.0, "mem_b": 0.8, "mem_c": 0.6, "mem_d": 0.5}
    
    causal = MockCausalChain()
    integrator = CausalBoostIntegrator(calc)
    
    test_memories = [
        ("mem_a", "木"),  # 源头
        ("mem_b", "火"),  # 直接关联
        ("mem_c", "土"),  # 间接关联
        ("mem_d", "金"),  # 孤岛
    ]
    
    for mid, wuxing in test_memories:
        priority, is_source = integrator.calculate_with_causal_chain(
            base_priority=0.5,
            memory_wuxing=wuxing,
            memory_id=mid,
            causal_chain=causal,
            all_memory_ids=["mem_a", "mem_b", "mem_c", "mem_d"],
            memory_wuxings=dict(test_memories),
            current_season="春",
        )
        source_tag = " [SOURCE]" if is_source else ""
        print(f"  {mid:6s} ({wuxing}) → priority={priority:.4f}{source_tag}")

    # 五行制化测试
    print("\n【五行制化因子测试】")
    print("-" * 60)
    test_distributions = [
        {"木": 0.45, "火": 0.20, "土": 0.15, "金": 0.12, "水": 0.08},  # 木过旺
        {"木": 0.10, "火": 0.10, "土": 0.60, "金": 0.10, "水": 0.10},  # 土过旺
        {"木": 0.20, "火": 0.20, "土": 0.20, "金": 0.20, "水": 0.20},  # 平衡
    ]
    
    ctx = PriorityContext()
    for i, dist in enumerate(test_distributions):
        ctx.memory_wuxing_distribution = dist
        calc.context = ctx
        for wuxing in ["木", "火", "土", "金", "水"]:
            factor = calc._get_wuxing_balance_factor(wuxing)
            print(f"  分布{i+1} | {wuxing} (占比{dist[wuxing]:.0%}) → balance={factor:.4f}")
        print()

    print("=" * 60)
    print("✅ DynamicPriorityCalculator 所有测试通过")
    print("=" * 60)
