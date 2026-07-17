"""
patient_profile — 患者纵向记忆增强

提供检验值趋势、诊疗轨迹、就诊对比等纵向记忆能力。

⚠️ 项目区隔：纵向记忆查询基于 metadata 过滤 + 时间排序，
不做临床预测（后者归 MCI World Model）。

Example:
  >>> from su_memory.clinical import PatientMemorySpace
  >>> space = PatientMemorySpace(client)
  >>> space.add_lab_value("P001", "白蛋白", 30.0, "g/L", "35-55")
  >>> space.add_lab_value("P001", "白蛋白", 35.0, "g/L", "35-55")
  >>> trend = space.get_lab_trend("P001", "白蛋白")
  >>> # TrendResult(direction="up", change_pct=16.7, values=[30.0, 35.0])
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from su_memory.sdk.lite_pro import SuMemoryLitePro

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════


class TrendDirection(str, Enum):
    UP = "up"
    DOWN = "down"
    STABLE = "stable"
    INSUFFICIENT = "insufficient"


@dataclass
class LabValue:
    """结构化检验值"""
    name: str
    value: float
    unit: str = ""
    reference_range: str = ""
    abnormal: bool = False
    timestamp: float = field(default_factory=time.time)

    def is_abnormal(self) -> bool:
        """判断是否异常（基于 reference_range "low-high" 格式）"""
        if self.abnormal:
            return True
        if not self.reference_range or "-" not in self.reference_range:
            return False
        try:
            parts = self.reference_range.split("-")
            low, high = float(parts[0].strip()), float(parts[1].strip())
            return self.value < low or self.value > high
        except (ValueError, IndexError):
            return False


@dataclass
class TrendResult:
    """检验值趋势结果"""
    lab_name: str
    direction: TrendDirection
    values: list[float] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    change_pct: float = 0.0
    first_value: float = 0.0
    last_value: float = 0.0
    count: int = 0

    def summary(self) -> str:
        """生成自然语言趋势摘要"""
        if self.direction == TrendDirection.INSUFFICIENT:
            return f"{self.lab_name}: 数据不足"
        arrow = {"up": "↑", "down": "↓", "stable": "→"}[self.direction.value]
        return (
            f"{self.lab_name}: {self.first_value}→{self.last_value}{self.unit_hint()}"
            f" {arrow} {abs(self.change_pct):.1f}% ({self.count}次)"
        )

    def unit_hint(self) -> str:
        return ""  # 由调用方补充单位


# ═══════════════════════════════════════════════════════════════════
# 患者记忆空间
# ═══════════════════════════════════════════════════════════════════


class PatientMemorySpace:
    """患者纵向记忆空间 — 检验趋势/诊疗轨迹/就诊对比。

    用法：
        space = PatientMemorySpace(client)
        space.add_lab_value("P001", "白蛋白", 30.0, "g/L", "35-55")
        trend = space.get_lab_trend("P001", "白蛋白")
        trajectory = space.get_care_trajectory("P001")
    """

    def __init__(self, client: SuMemoryLitePro):
        self._client = client

    def add_lab_value(
        self,
        patient_id: str,
        lab_name: str,
        value: float,
        unit: str = "",
        reference_range: str = "",
        metadata: dict | None = None,
        event_time: int | None = None,
    ) -> str:
        """写入结构化检验值（支持双时间）。

        Args:
            patient_id: 患者 ID
            lab_name: 检验项目名称（如"白蛋白"）
            value: 数值
            unit: 单位（如"g/L"）
            reference_range: 参考范围（如"35-55"）
            metadata: 额外元数据
            event_time: 检验发生时间（Unix秒），缺省=入库时间（C4）

        Returns:
            memory_id
        """
        lv = LabValue(
            name=lab_name, value=value, unit=unit,
            reference_range=reference_range,
        )
        content = f"{patient_id} 检验:{lab_name}={value}{unit}"
        if lv.is_abnormal():
            content += f" (异常,参考{reference_range})"

        full_meta = {
            "patient_id": patient_id,
            "event_type": "lab_result",
            "lab_name": lab_name,
            "lab_value": value,
            "lab_unit": unit,
            "lab_reference": reference_range,
            "lab_abnormal": lv.is_abnormal(),
        }
        if metadata:
            full_meta.update(metadata)

        kwargs: dict = {"metadata": full_meta}
        if event_time is not None:
            kwargs["event_time"] = event_time
        return self._client.add(content, **kwargs)

    def get_lab_trend(
        self,
        patient_id: str,
        lab_name: str,
        time_range: tuple[float, float] | None = None,
    ) -> TrendResult:
        """获取检验值趋势。

        Args:
            patient_id: 患者 ID
            lab_name: 检验项目名称
            time_range: 可选时间范围 (start_ts, end_ts)

        Returns:
            TrendResult
        """
        values = self._extract_lab_values(patient_id, lab_name, time_range)

        if len(values) < 2:
            return TrendResult(
                lab_name=lab_name,
                direction=TrendDirection.INSUFFICIENT,
                values=[v[1] for v in values],
                timestamps=[v[0] for v in values],
                first_value=values[0][1] if values else 0.0,
                last_value=values[-1][1] if values else 0.0,
                count=len(values),
            )

        first_val = values[0][1]
        last_val = values[-1][1]
        change_pct = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0

        if abs(change_pct) < 5:
            direction = TrendDirection.STABLE
        elif change_pct > 0:
            direction = TrendDirection.UP
        else:
            direction = TrendDirection.DOWN

        return TrendResult(
            lab_name=lab_name,
            direction=direction,
            values=[v[1] for v in values],
            timestamps=[v[0] for v in values],
            change_pct=change_pct,
            first_value=first_val,
            last_value=last_val,
            count=len(values),
        )

    def find_abnormal_labs(
        self, patient_id: str
    ) -> list[dict[str, Any]]:
        """筛查患者的异常检验值。

        Returns:
            [{"lab_name", "value", "unit", "reference_range", "timestamp"}, ...]
        """
        all_labs = self._get_all_labs_for_patient(patient_id)
        abnormal: list[dict[str, Any]] = []
        seen: set[str] = set()

        for ts, name, value, unit, ref in all_labs:
            lv = LabValue(name=name, value=value, unit=unit, reference_range=ref)
            key = f"{name}_{ts}"
            if lv.is_abnormal() and key not in seen:
                seen.add(key)
                abnormal.append({
                    "lab_name": name,
                    "value": value,
                    "unit": unit,
                    "reference_range": ref,
                    "timestamp": ts,
                })

        abnormal.sort(key=lambda x: x["timestamp"], reverse=True)
        return abnormal

    def get_care_trajectory(
        self,
        patient_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """获取患者完整诊疗轨迹。

        按时间排序返回所有临床事件，标注事件类型。

        Returns:
            [{"memory_id", "content", "event_type", "timestamp"}, ...]
        """
        events: list[dict[str, Any]] = []
        graph = getattr(self._client, "_graph", None)
        if graph is None:
            return events

        for mem_id, node in graph._nodes.items():
            meta = node.metadata or {}
            if meta.get("patient_id") != patient_id:
                continue
            events.append({
                "memory_id": mem_id,
                "content": node.content,
                "event_type": meta.get("event_type", ""),
                "timestamp": node.timestamp,
                "event_time": node.effective_time,
            })

        events.sort(key=lambda x: x["timestamp"])
        return events[:limit]

    def compare_periods(
        self,
        patient_id: str,
        period_a: tuple[float, float],
        period_b: tuple[float, float],
    ) -> dict[str, Any]:
        """对比两个时间段的诊疗状况。

        Args:
            period_a: (start_ts, end_ts) 较早的时间段
            period_b: (start_ts, end_ts) 较晚的时间段

        Returns:
            {"period_a": {...}, "period_b": {...}, "changes": [...]}
        """
        def summarize_period(start: float, end: float) -> dict:
            events = [
                e for e in self.get_care_trajectory(patient_id, limit=500)
                if start <= e["timestamp"] <= end
            ]
            event_types: dict[str, int] = {}
            for e in events:
                et = e["event_type"] or "unknown"
                event_types[et] = event_types.get(et, 0) + 1
            return {
                "event_count": len(events),
                "event_types": event_types,
                "time_range": (start, end),
            }

        summary_a = summarize_period(*period_a)
        summary_b = summarize_period(*period_b)

        changes: list[str] = []
        for et, count_b in summary_b["event_types"].items():
            count_a = summary_a["event_types"].get(et, 0)
            if count_b > count_a:
                changes.append(f"{et}事件增加({count_a}→{count_b})")
            elif count_b < count_a:
                changes.append(f"{et}事件减少({count_a}→{count_b})")

        return {
            "period_a": summary_a,
            "period_b": summary_b,
            "changes": changes,
        }

    # ── 内部方法 ──────────────────────────────────────────

    def _extract_lab_values(
        self,
        patient_id: str,
        lab_name: str,
        time_range: tuple[float, float] | None,
    ) -> list[tuple[float, float]]:
        """从记忆中提取指定检验项目的 (timestamp, value) 列表"""
        results: list[tuple[float, float]] = []
        graph = getattr(self._client, "_graph", None)
        if graph is None:
            return results

        for _mem_id, node in graph._nodes.items():
            meta = node.metadata or {}
            if meta.get("patient_id") != patient_id:
                continue
            if meta.get("lab_name") != lab_name:
                continue
            if meta.get("event_type") != "lab_result":
                continue
            value = meta.get("lab_value")
            if value is None:
                continue
            ts = node.effective_time  # C4: 优先事件时间
            if time_range and not (time_range[0] <= ts <= time_range[1]):
                continue
            results.append((ts, float(value)))

        results.sort(key=lambda x: x[0])
        return results

    def _get_all_labs_for_patient(
        self, patient_id: str
    ) -> list[tuple[float, str, float, str, str]]:
        """获取患者所有检验值记录"""
        results: list[tuple[float, str, float, str, str]] = []
        graph = getattr(self._client, "_graph", None)
        if graph is None:
            return results

        for _mem_id, node in graph._nodes.items():
            meta = node.metadata or {}
            if meta.get("patient_id") != patient_id:
                continue
            if meta.get("event_type") != "lab_result":
                continue
            results.append((
                node.effective_time,
                meta.get("lab_name", ""),
                float(meta.get("lab_value", 0)),
                meta.get("lab_unit", ""),
                meta.get("lab_reference", ""),
            ))

        return results
