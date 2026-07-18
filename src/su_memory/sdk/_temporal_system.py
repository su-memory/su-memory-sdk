"""
_temporal_system — 时序编码与衰减系统（lite_pro.py 拆分）

TemporalSystem: 时间编码（五行能量映射）+ recency 衰减打分。
从 lite_pro.py 拆分，对外通过 lite_pro.py 再导出保持兼容。
"""
from __future__ import annotations

import math
import time
from typing import Any


class TemporalSystem:
    """
    Temporal encoding and time-aware recall system
    Enhanced temporal dimension retrieval capability
    Supports energy strength state calculation and time decay weighting
    """

    # Time stems (pinyin)
    TIME_STEMS = ["jia", "yi", "bing", "ding", "wu", "ji", "geng", "xin", "ren", "gui"]
    # Time branches (pinyin)
    TIME_BRANCHES = ["zi", "chou", "yin", "mao", "chen", "si", "wu", "wei", "shen", "you", "xu", "hai"]

    # Energy types mapped to branches
    BRANCH_ENERGY = {
        "zi": "water", "chou": "earth", "yin": "wood", "mao": "wood",
        "chen": "earth", "si": "fire", "wu": "fire", "wei": "earth",
        "shen": "metal", "you": "metal", "xu": "earth", "hai": "water"
    }

    # Energy enhancement cycle
    ENERGY_ENHANCE = {"wood": "fire", "fire": "earth", "earth": "metal", "metal": "water", "water": "wood"}

    # Energy suppression cycle
    ENERGY_SUPPRESS = {"wood": "earth", "earth": "water", "water": "fire", "fire": "metal", "metal": "wood"}

    # Strength state mapping: strong/thriving/resting/restrained/dormant
    STRENGTH_MAP = {
        "wood": {"strong": "yin mao", "thriving": "hai zi", "resting": "si wu", "restrained": "shen you", "dormant": "chen xu chou wei"},
        "fire": {"strong": "si wu", "thriving": "yin mao", "resting": "shen you", "restrained": "hai zi", "dormant": "chen xu chou wei"},
        "earth": {"strong": "chen xu chou wei", "thriving": "si wu", "resting": "hai zi", "restrained": "yin mao", "dormant": "shen you"},
        "metal": {"strong": "shen you", "thriving": "chen xu chou wei", "resting": "yin mao", "restrained": "si wu", "dormant": "hai zi"},
        "water": {"strong": "hai zi", "thriving": "shen you", "resting": "chen xu chou wei", "restrained": "si wu", "dormant": "yin mao"}
    }

    # Month to dominant energy mapping
    MONTH_ENERGY = {
        1: "water", 2: "wood", 3: "wood", 4: "fire", 5: "fire",
        6: "earth", 7: "earth", 8: "metal", 9: "metal", 10: "water", 11: "water", 12: "wood"
    }

    def get_time_code(self, timestamp: int = None) -> dict[str, Any]:
        """
        Get current time stem and branch encoding

        Args:
            timestamp: Unix timestamp (seconds)

        Returns:
            Dictionary containing time_stem, time_branch, energy_type, year_code
        """
        ts = timestamp or int(time.time())

        # Calculate year (based on Unix timestamp)
        year = 1970 + (ts // 31556926)  # approximate year

        # Jiazi year cycle calculation (60-year cycle)
        jiazi_year = (year - 1984) % 60

        time_stem_idx = jiazi_year % 10
        time_branch_idx = jiazi_year % 12

        time_stem = self.TIME_STEMS[time_stem_idx]
        time_branch = self.TIME_BRANCHES[time_branch_idx]

        return {
            "time_stem": time_stem,
            "time_branch": time_branch,
            "energy_type": self.BRANCH_ENERGY[time_branch],
            "year_code": f"{time_stem}{time_branch}"
        }

    def get_strength_state(self, energy_type: str, month: int = None) -> str:
        """
        Get energy strength state for current time period

        Args:
            energy_type: Energy type (wood/fire/earth/metal/water)
            month: Month (1-12), defaults to current month

        Returns:
            Strength state: strong/thriving/resting/restrained/dormant
        """
        if month is None:
            import datetime
            month = datetime.datetime.now().month

        current_energy = self.MONTH_ENERGY.get(month, "earth")

        if energy_type == current_energy:
            return "strong"  # Same element dominant
        elif self.ENERGY_ENHANCE.get(current_energy) == energy_type:
            return "thriving"  # Enhanced by current
        elif self.ENERGY_SUPPRESS.get(current_energy) == energy_type:
            return "dormant"  # Suppressed by current
        else:
            return "resting"  # Neutral

    def infer_energy_from_content(self, content: str) -> str:
        """
        Infer energy type from memory content

        Args:
            content: Memory content text

        Returns:
            Energy type classification (wood/fire/earth/metal/water)
        """
        energy_keywords = {
            "wood": ["生长", "发展", "树木", "森林", "绿色", "东方", "春季",
                   "肝", "筋", "希望", "创造", "开始", "健康"],
            "fire": ["热情", "炎热", "红色", "南方", "夏季", "心",
                   "血液", "高温", "活力", "能量", "动力", "激情"],
            "earth": ["稳定", "黄色", "中央", "四季", "脾", "消化",
                   "土地", "基础", "踏实", "信任", "稳定", "持续"],
            "metal": ["收敛", "白色", "西方", "秋季", "肺", "呼吸",
                   "金属", "价值", "收获", "总结", "结束", "财"],
            "water": ["流动", "蓝色", "北方", "冬季", "肾", "泌尿",
                   "智慧", "灵活", "变化", "适应", "学习", "思考"]
        }

        scores = dict.fromkeys(energy_keywords, 0)
        for e, kws in energy_keywords.items():
            for kw in kws:
                if kw in content:
                    scores[e] += 1

        return max(scores, key=scores.get) if max(scores.values()) > 0 else "earth"

    def calculate_recency_score(
        self,
        memory_timestamp: int,
        memory_energy_type: str = "earth",
        current_timestamp: int = None
    ) -> float:
        """
        Calculate recency/decay score considering energy strength state

        Args:
            memory_timestamp: Memory creation timestamp
            memory_energy_type: Memory energy type
            current_timestamp: Current timestamp

        Returns:
            Recency score (0-1)
        """
        ts = current_timestamp or int(time.time())
        # V15/V16: 时间合法性防御——负数/零/未来时间归一化
        # 负数或零（黑洞记忆）回退到当前时间（衰减最小，不霸占也不消失）；
        # 未来时间 clamp 到当前（防止 recency>1.0 永久霸占召回）。
        if not memory_timestamp or memory_timestamp < 0:
            memory_timestamp = ts
        elif memory_timestamp > ts:
            memory_timestamp = ts
        days = (ts - memory_timestamp) / 86400

        # Exponential decay（days>=0 保证 decay<=1.0，不再被未来时间抬高）
        decay = math.exp(-0.02 * days)

        # Get current energy state
        time_code = self.get_time_code(ts)
        current_energy = time_code["energy_type"]

        # Energy enhancement/suppression effects
        if self.ENERGY_ENHANCE.get(current_energy) == memory_energy_type:
            # Memory energy enhances current energy - strengthen decay
            decay *= 1.3
        elif self.ENERGY_SUPPRESS.get(current_energy) == memory_energy_type:
            # Memory energy is suppressed by current energy - weaken decay
            decay *= 0.7
        elif memory_energy_type == current_energy:
            # Same energy type - moderate strengthen
            decay *= 1.1

        # Apply strength state modifier
        strength_state = self.get_strength_state(memory_energy_type)
        if strength_state == "strong":
            decay *= 1.2
        elif strength_state == "thriving":
            decay *= 1.1
        elif strength_state == "dormant":
            decay *= 0.8

        # Short-term memory bonus (within 30 days)
        if days < 1:
            decay *= 1.2
        elif days < 7:
            decay *= 1.1
        elif days < 30:
            decay *= 1.0
        else:
            decay *= 0.9

        return max(0.1, min(1.0, decay))

    def get_temporal_context(self, timestamp: int = None) -> dict[str, Any]:
        """
        Get temporal context for a given timestamp

        Args:
            timestamp: Unix timestamp

        Returns:
            Temporal context information
        """
        ts = timestamp or int(time.time())

        import datetime
        dt = datetime.datetime.fromtimestamp(ts)

        time_code = self.get_time_code(ts)

        return {
            "datetime": dt.isoformat(),
            "timestamp": ts,
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "weekday": dt.strftime("%A"),
            "time_code": time_code,
            "strength_state": self.get_strength_state(time_code["energy_type"], dt.month)
        }


