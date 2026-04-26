"""
time_code时空系统

time_stemtime_branch + 时空坐标 + 动态优先级

对外暴露：TemporalSystem
内部实现：封装在su_core._internal中
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, date
import calendar
import time
import math


class TianGan:
    """time_stem（10个）"""
    NAMES = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
    # Duality
    YIN_YANG = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=阳, 0=阴
    # energy_type
    WUXING = ["木", "木", "火", "火", "土", "土", "金", "金", "水", "水"]


class DiZhi:
    """time_branch（12个）"""
    NAMES = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]
    # Duality
    YIN_YANG = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 子阴水, 丑阳土, 寅阳木...
    # energy_type
    WUXING = ["水", "土", "木", "木", "土", "火", "火", "土", "金", "金", "土", "水"]
    # 六合关系（time_branch索引对）：子丑合, 寅亥合, 卯戌合, 辰酉合, 巳申合, 午未合
    LIUHE = {0: 1, 1: 0, 2: 11, 11: 2, 3: 10, 10: 3, 4: 9, 9: 4, 5: 8, 8: 5, 6: 7, 7: 6}
    # 三合局（三个time_branch组成一局）：申子辰(水), 亥卯未(木), 寅午戌(火), 巳酉丑(金)
    SANHE = [
        {8, 0, 4},   # 申子辰 → 水局
        {11, 3, 7},  # 亥卯未 → 木局
        {2, 6, 10},  # 寅午戌 → 火局
        {5, 9, 1},   # 巳酉丑 → 金局
    ]


@dataclass
class TemporalInfo:
    """time_code信息"""
    tian_gan: str        # time_stem
    di_zhi: str          # time_branch
    time_code: str           # 完整time_code（如"cycle_period"）
    energy_type: str           # energy_type
    yin_yang: str         # Duality
    season: str           # 季节
    is_birthday: bool     # 是否strong_day


@dataclass 
class DynamicPriority:
    """time_code优先级调整"""
    base_priority: float   # 基础优先级
    season_boost: float   # 季节加成
    time_boost: float     # 时辰加成
    final_priority: float # 最终优先级


class TemporalSystem:
    """
    time_code时空系统 - 对外唯一接口
    
    功能：
    1. 日期 → time_code转换
    2. time_code → 时空属性
    3. 动态优先级计算
    4. 月令strength_state判定
    5. 六十cycle_period循环位置编码
    6. 非线性时间衰减
    
    对外隐藏：60cycle_period映射表、strength_state计算公式
    """
    
    # 60cycle_period表（简化版，仅核心日期）
    _cycle_period_CYCLE_START = 1984  # cycle_period年
    
    # 四季对应time_branch
    _SEASON_DIZHI = {
        "春": ["寅", "卯"],
        "夏": ["巳", "午"],
        "秋": ["申", "酉"],
        "冬": ["亥", "子"]
    }
    
    # time_stemtime_branch组合（60cycle_period循环）
    _GANZHI_CYCLE = [
        f"{tg}{dz}" 
        for i in range(60) 
        for tg in [TianGan.NAMES[i % 10]] 
        for dz in [DiZhi.NAMES[i % 12]]
        if (TianGan.NAMES.index(tg), DiZhi.NAMES.index(dz)) == (i % 10, i % 12)
    ][:60]  # 取60个不重复的组合
    
    def __init__(self):
        # 预计算60cycle_period
        self._init_60_jiazi()
    
    def _init_60_jiazi(self):
        """初始化60cycle_period表"""
        # cycle_period从1984年开始，每60年一循环
        base_year = self._cycle_period_CYCLE_START
        
        self._year_time_code = {}
        self._month_time_code = {}
        self._day_time_code = {}
        
        # 计算年的60cycle_period
        current_time_code_idx = 0  # 1984 = cycle_period = 索引0
        for year in range(1924, 2044):  # 覆盖近百年
            if year == base_year:
                current_time_code_idx = 0
            else:
                current_time_code_idx = (year - base_year) % 60
            
            self._year_time_code[year] = self._get_time_code_name(current_time_code_idx)
    
    def _get_time_code_name(self, idx: int) -> str:
        """获取time_code名称"""
        tg_idx = idx % 10
        dz_idx = idx % 12
        return TianGan.NAMES[tg_idx] + DiZhi.NAMES[dz_idx]
    
    def date_to_time_code(self, dt: date) -> TemporalInfo:
        """
        将日期转换为time_code信息
        
        Args:
            dt: 日期对象
        
        Returns:
            TemporalInfo: 包含年time_code、月time_code、日time_code、时time_code
        """
        year = dt.year
        month = dt.month
        day = dt.day
        
        # 年time_code
        year_gan = self._year_time_code.get(year, self._guess_year_time_code(year))
        
        # 日time_code（简化计算，实际需要查表或天文计算）
        day_gan, day_zhi = self._calc_day_time_code(dt)
        
        # 确定季节（使用节气精确版）
        season = self._get_season(month, day)
        
        # energy_type归属
        tg_idx = TianGan.NAMES.index(year_gan[0])
        dz_idx = DiZhi.NAMES.index(day_zhi)
        
        return TemporalInfo(
            tian_gan=year_gan[0],
            di_zhi=day_zhi,
            time_code=year_gan,  # 主time_code用年柱
            energy_type=DiZhi.WUXING[dz_idx],
            yin_yang="阳" if DiZhi.YIN_YANG[dz_idx] else "阴",
            season=season,
            is_birthday=self._is_wang_day(tg_idx, dz_idx)
        )
    
    def _guess_year_time_code(self, year: int) -> str:
        """推测年time_code"""
        idx = (year - self._cycle_period_CYCLE_START) % 60
        return self._get_time_code_name(idx)
    
    def _calc_day_time_code(self, dt: date) -> Tuple[str, str]:
        """计算日time_code（简化版）"""
        # 使用儒略日计算
        julian_day = self._to_julian_day(dt)
        idx = int((julian_day - 2440587) % 60)
        return self._get_time_code_name(idx), DiZhi.NAMES[idx % 12]
    
    def _to_julian_day(self, dt: date) -> float:
        """日期转儒略日"""
        year, month, day = dt.year, dt.month, dt.day
        if month <= 2:
            year -= 1
            month += 12
        a = year // 100
        b = 2 - a + a // 4
        return int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
    
    def _get_season(self, month: int, day: int = 15) -> str:
        """
        根据月份和日期确定当前月令（节气精确版）
        
        寅月(立春~惊蛰): 2.4~3.5 → 春（木）
        卯月(惊蛰~清明): 3.6~4.4 → 春（木）
        辰月(清明~立夏): 4.5~5.5 → 春末（土）
        巳月(立夏~芒种): 5.6~6.5 → 夏（火）
        午月(芒种~小暑): 6.6~7.6 → 夏（火）
        未月(小暑~立秋): 7.7~8.6 → 夏末（土）
        申月(立秋~白露): 8.7~9.7 → 秋（金）
        酉月(白露~寒露): 9.8~10.7 → 秋（金）
        戌月(寒露~立冬): 10.8~11.6 → 秋末（土）
        亥月(立冬~大雪): 11.7~12.6 → 冬（水）
        子月(大雪~小寒): 12.7~1.5 → 冬（水）
        丑月(小寒~立春): 1.6~2.3 → 冬末（土）
        """
        # 将月+日编码为浮点数方便比较
        md = month + day / 100.0
        
        # 节气精确判定
        if 2.04 <= md <= 3.05:
            return "春"      # 寅月
        elif 3.06 <= md <= 4.04:
            return "春"      # 卯月
        elif 4.05 <= md <= 5.05:
            return "四季"    # 辰月（春末土）
        elif 5.06 <= md <= 6.05:
            return "夏"      # 巳月
        elif 6.06 <= md <= 7.06:
            return "夏"      # 午月
        elif 7.07 <= md <= 8.06:
            return "四季"    # 未月（夏末土）
        elif 8.07 <= md <= 9.07:
            return "秋"      # 申月
        elif 9.08 <= md <= 10.07:
            return "秋"      # 酉月
        elif 10.08 <= md <= 11.06:
            return "四季"    # 戌月（秋末土）
        elif 11.07 <= md <= 12.06:
            return "冬"      # 亥月
        elif md >= 12.07 or md <= 1.05:
            return "冬"      # 子月
        elif 1.06 <= md <= 2.03:
            return "四季"    # 丑月（冬末土）
        else:
            return "四季"
    
    def get_monthly_energy_type_state(self, month: int, day: int = 15) -> Dict[str, str]:
        """
        获取当前月令下各energy_type的strong衰状态
        
        Returns:
            {"木": "strong", "火": "balanced", "土": "rested", "金": "restrained", "水": "declined"}  # 春季示例
        """
        season = self._get_season(month, day)
        
        # strength_staterestedrestraineddeclined顺序规则
        states = ["strong", "balanced", "rested", "restrained", "declined"]
        
        season_orders = {
            "春": ["木", "火", "水", "金", "土"],
            "夏": ["火", "土", "木", "水", "金"],
            "秋": ["金", "水", "土", "火", "木"],
            "冬": ["水", "木", "金", "土", "火"],
            "四季": ["土", "金", "火", "木", "水"],
        }
        
        order = season_orders.get(season, season_orders["四季"])
        return {wx: st for wx, st in zip(order, states)}
    
    def _is_wang_day(self, tg_idx: int, dz_idx: int) -> bool:
        """判断是否strong_day（简化判断：同energy_type则strong）"""
        tg_energy_type = TianGan.WUXING[tg_idx]
        dz_energy_type = DiZhi.WUXING[dz_idx]
        return tg_energy_type == dz_energy_type
    
    def get_jiazi_position(self, dt: date) -> int:
        """
        获取日期在六十cycle_period循环中的位置（0-59）
        
        用于计算两条记忆在time_code时空中的"距离"。
        cycle_period循环位置差越近，时空关联越强。
        """
        julian_day = self._to_julian_day(dt)
        return int((julian_day - 2440587) % 60)
    
    def jiazi_distance(self, pos1: int, pos2: int) -> float:
        """
        计算两个cycle_period位置之间的循环距离（0.0~1.0）
        
        考虑循环特性：0和59的距离只有1，不是59。
        """
        diff = abs(pos1 - pos2)
        circular_diff = min(diff, 60 - diff)
        return circular_diff / 30.0  # 归一化到0~1（最大距离30）
    
    def temporal_similarity(self, dt1: date, dt2: date) -> float:
        """
        计算两个日期的time_code时序balanced似度（0.0~1.0）
        
        综合考虑：
        1. cycle_period循环距离（权重 0.4）
        2. 同一季节 bonus（权重 0.3）
        3. time_branch六合/三合关系（权重 0.3）
        """
        # 1. cycle_period循环距离
        pos1 = self.get_jiazi_position(dt1)
        pos2 = self.get_jiazi_position(dt2)
        jiazi_sim = 1.0 - self.jiazi_distance(pos1, pos2)
        
        # 2. 季节匹配
        season1 = self._get_season(dt1.month, dt1.day)
        season2 = self._get_season(dt2.month, dt2.day)
        season_sim = 1.0 if season1 == season2 else 0.3
        
        # 3. time_branch六合/三合/邻近关系
        dz_idx1 = pos1 % 12
        dz_idx2 = pos2 % 12
        dizhi_sim = 0.0
        if dz_idx1 == dz_idx2:
            dizhi_sim = 1.0
        elif DiZhi.LIUHE.get(dz_idx1) == dz_idx2:
            dizhi_sim = 0.9
        else:
            # 三合检查
            for group in DiZhi.SANHE:
                if dz_idx1 in group and dz_idx2 in group:
                    dizhi_sim = 0.7
                    break
            if dizhi_sim == 0.0:
                # time_branch邻近度（循环距离）
                dz_diff = min(abs(dz_idx1 - dz_idx2), 12 - abs(dz_idx1 - dz_idx2))
                dizhi_sim = max(0.0, 1.0 - dz_diff / 6.0)
        
        return round(0.4 * jiazi_sim + 0.3 * season_sim + 0.3 * dizhi_sim, 4)
    
    def calculate_time_decay(self, memory_timestamp: int, memory_energy_type: str) -> float:
        """
        计算记忆的时间衰减因子（0.0~1.0）
        
        公式：decay = base_decay * time_code_multiplier
        
        - base_decay: 基于天数的指数衰减
          - 0-7天: 1.0（无衰减）
          - 7-30天: 0.95^(days-7)
          - 30-90天: 0.9^(days-30) * base_30day
          - 90天+: 0.85^(days-90) * base_90day
        
        - time_code_multiplier: time_code调制因子（基于月令strong衰）
          - 记忆energy_type当令（strong）: ×1.3
          - 记忆energy_typebalanced（次strong）: ×1.15
          - 记忆energy_typerested: ×1.0
          - 记忆energy_typerestrained: ×0.8
          - 记忆energy_typedeclined: ×0.6
        """
        now = int(time.time())
        days = max(0, (now - memory_timestamp) / 86400.0)
        
        # 基础衰减（分段指数衰减）
        if days <= 7:
            base_decay = 1.0
        elif days <= 30:
            base_decay = 0.95 ** (days - 7)
        elif days <= 90:
            base_30 = 0.95 ** 23  # 7~30天的衰减
            base_decay = base_30 * (0.9 ** (days - 30))
        else:
            base_30 = 0.95 ** 23
            base_90 = base_30 * (0.9 ** 60)
            base_decay = base_90 * (0.85 ** (days - 90))
        
        base_decay = max(0.0, min(1.0, base_decay))
        
        # time_code调制因子（基于当前月令strong衰）
        today = date.today()
        monthly_states = self.get_monthly_energy_type_state(today.month, today.day)
        state = monthly_states.get(memory_energy_type, "rested")
        
        multiplier_map = {
            "strong": 1.3,
            "balanced": 1.15,
            "rested": 1.0,
            "restrained": 0.8,
            "declined": 0.6,
        }
        multiplier = multiplier_map.get(state, 1.0)
        
        decay = base_decay * multiplier
        return round(max(0.0, min(1.0, decay)), 4)
    
    def calculate_priority(
        self,
        base_priority: int,
        time_code_info: TemporalInfo,
        memory_energy_type: str,
        memory_timestamp: int = None
    ) -> DynamicPriority:
        """
        升级版动态优先级
        
        综合考虑：
        1. 基础优先级
        2. 月令strength_state状态（替代简单 season_boost）
        3. 非线性时间衰减（如果提供了 memory_timestamp）
        4. strong_day加成
        """
        # 基础优先级归一化
        base = base_priority / 10.0
        
        # 月令strength_state加成（替代简单 season_boost）
        today = date.today()
        monthly_states = self.get_monthly_energy_type_state(today.month, today.day)
        energy_type_state = monthly_states.get(memory_energy_type, "rested")
        
        state_boost_map = {
            "strong": 0.20,
            "balanced": 0.10,
            "rested": 0.0,
            "restrained": -0.08,
            "declined": -0.12,
        }
        season_boost = state_boost_map.get(energy_type_state, 0.0)
        
        # strong_day加成
        wang_boost = 0.1 if time_code_info.is_birthday else 0.0
        
        # Duality加成
        yin_boost = 0.05 if time_code_info.yin_yang == "阴" else 0.0
        
        # 时间衰减（如果提供了 timestamp）
        time_decay_factor = 1.0
        if memory_timestamp is not None:
            time_decay_factor = self.calculate_time_decay(memory_timestamp, memory_energy_type)
        
        # 综合计算
        time_boost = wang_boost + yin_boost
        final = (base + season_boost + time_boost) * time_decay_factor
        final = max(0.0, min(1.0, final))
        
        return DynamicPriority(
            base_priority=base,
            season_boost=season_boost,
            time_boost=time_boost,
            final_priority=round(final, 3)
        )
    
    def get_current_time_code(self) -> TemporalInfo:
        """获取当前时刻的time_code信息"""
        return self.date_to_time_code(date.today())
