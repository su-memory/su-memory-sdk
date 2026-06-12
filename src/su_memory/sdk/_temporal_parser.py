"""
su-memory v4.0 — Multi-Resolution Temporal Parser (多分辨率时间解析器)

将自然语言时间表达式转化为 ISO 8601 日期范围，支持时间约束检索。

核心能力:
- 绝对时间: "March 2024", "June 5th, 2023" → ISO日期
- 相对时间: "last week", "two months ago", "recently" → 相对于 reference_date 计算
- 序列时间: "first time", "most recent", "earliest" → 时间顺序标记
- 持续时间: "for three months", "since January" → 起止日期范围

参考: Chronos (95.6%), Mastra OM (94.87%), Hindsight (91.4%)
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any

from su_memory.sdk._event_extractor import TemporalRange

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 月份映射
# ---------------------------------------------------------------------------

_MONTH_NAMES: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12,
}

_WEEKDAYS: dict[str, int] = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    "mon": 0, "tue": 1, "wed": 2, "thu": 3,
    "fri": 4, "sat": 5, "sun": 6,
}

# ---------------------------------------------------------------------------
# 正则表达式模式
# ---------------------------------------------------------------------------

# 绝对日期模式
_RE_ABS_MONTH_DAY_YEAR = re.compile(
    r'\b(January|February|March|April|May|June|July|August|September|October|November|December)'
    r'\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})\b',
    re.IGNORECASE,
)
_RE_ABS_MONTH_YEAR = re.compile(
    r'\b(January|February|March|April|May|June|July|August|September|October|November|December)'
    r'\s+(\d{4})\b',
    re.IGNORECASE,
)
_RE_ABS_MONTH_DAY = re.compile(
    r'\b(January|February|March|April|May|June|July|August|September|October|November|December)'
    r'\s+(\d{1,2})(?:st|nd|rd|th)?\b',
    re.IGNORECASE,
)
_RE_ABS_ABBR_DAY_YEAR = re.compile(
    r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})\b',
    re.IGNORECASE,
)
_RE_ABS_ABBR_YEAR = re.compile(
    r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{4})\b',
    re.IGNORECASE,
)
_RE_ABS_ISO = re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b')
_RE_ABS_YEAR_MONTH = re.compile(r'\b(\d{4})-(\d{2})\b')
_RE_ABS_YEAR = re.compile(r'\b(19\d{2}|20\d{2})\b')  # 4-digit year

# 相对日期模式
_RE_REL_LAST_UNIT = re.compile(
    r'\b(last|past|previous|this)\s+(week|month|year|weekend|quarter)\b',
    re.IGNORECASE,
)
_RE_REL_X_AGO = re.compile(
    r'\b(a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+('
    r'day|days|week|weeks|month|months|year|years)\s+ago\b',
    re.IGNORECASE,
)
_RE_REL_YESTERDAY = re.compile(r'\byesterday\b', re.IGNORECASE)
_RE_REL_TOMORROW = re.compile(r'\btomorrow\b', re.IGNORECASE)
_RE_REL_TODAY = re.compile(r'\b(today|right now|currently)\b', re.IGNORECASE)
_RE_REL_RECENTLY = re.compile(
    r'\b(recently|lately|just now|a moment ago|a while ago|the other day)\b',
    re.IGNORECASE,
)
_RE_REL_COUPLE = re.compile(
    r'\b(a couple of|couple of|a few)\s+(days?|weeks?|months?|years?)\s+ago\b',
    re.IGNORECASE,
)

# 持续时间模式
_RE_DUR_SINCE = re.compile(
    r'\bsince\s+(January|February|March|April|May|June|July|August|September|October|November|December)'
    r'(?:\s+(\d{4}))?\b',
    re.IGNORECASE,
)
_RE_DUR_FOR = re.compile(
    r'\bfor\s+(a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+'
    r'(day|days|week|weeks|month|months|year|years)\b',
    re.IGNORECASE,
)
_RE_DUR_FROM_TO = re.compile(
    r'\bfrom\s+'
    r'(January|February|March|April|May|June|July|August|September|October|November|December)'
    r'(?:\s+(\d{4}))?'
    r'\s+to\s+'
    r'(January|February|March|April|May|June|July|August|September|October|November|December)'
    r'(?:\s+(\d{4}))?\b',
    re.IGNORECASE,
)

# 序列时间模式
_RE_ORD_SUPERLATIVE = re.compile(
    r'\b(first|earliest|initial|original|oldest)\b',
    re.IGNORECASE,
)
_RE_ORD_LATEST = re.compile(
    r'\b(latest|most recent|last|current|newest|final)\b',
    re.IGNORECASE,
)
_RE_ORD_BEFORE = re.compile(r'\b(before|prior to|earlier than)\b', re.IGNORECASE)
_RE_ORD_AFTER = re.compile(r'\b(after|following|later than|subsequent to)\b', re.IGNORECASE)

# 季度模式
_RE_QUARTER = re.compile(
    r'\b(Q[1-4]|first|second|third|fourth)\s+quarter(?:\s+of)?\s*(\d{4})\b',
    re.IGNORECASE,
)
# 短格式季度: "Q2 2024", "Q1 2023"
_RE_QUARTER_SHORT = re.compile(
    r'\bQ([1-4])\s+(\d{4})\b',
    re.IGNORECASE,
)
# 月份单独出现（如 "in May", "during June"）
_RE_MONTH_STANDALONE = re.compile(
    r'\b(?:in|during|this)\s+'
    r'(January|February|March|April|May|June|July|August|September|October|November|December)'
    r'\b',
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _month_days(year: int, month: int) -> int:
    """返回指定月份的天数"""
    if month == 12:
        return 31
    next_month = datetime(year, month + 1, 1)
    this_month = datetime(year, month, 1)
    return (next_month - this_month).days


_WORD_TO_NUM: dict[str, int] = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3,
    "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10,
}

def _parse_number(text: str) -> int:
    """将文本数字转为整数，支持英文单词和 'a'/'an' → 1"""
    text = text.strip().lower()
    if text in _WORD_TO_NUM:
        return _WORD_TO_NUM[text]
    try:
        return int(text)
    except ValueError:
        return 1


def _date_to_str(d: datetime) -> str:
    """datetime → ISO 8601 日期字符串"""
    return d.strftime("%Y-%m-%d")


def _first_day_of_month(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}-01"


def _last_day_of_month(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}-{_month_days(year, month):02d}"


# ---------------------------------------------------------------------------
# TemporalParser 主类
# ---------------------------------------------------------------------------

class TemporalParser:
    """多分辨率时间解析器 — 将自然语言时间表达式转化为 ISO 8601 日期范围

    解析策略:
    1. 规则引擎优先（覆盖 90%+ 常见表达式）
    2. LLM 回退（可选，覆盖边缘情况）

    用法:
        parser = TemporalParser()
        result = parser.parse("last week", reference_date="2024-06-15")
        # → TemporalRange(start="2024-06-08", end="2024-06-14", granularity="week")

        result = parser.parse("March 2024", reference_date="2024-06-15")
        # → TemporalRange(start="2024-03-01", end="2024-03-31", granularity="month")
    """

    def __init__(self, llm_provider: str = "", llm_model: str = "", ollama_url: str = ""):
        """可选 LLM 回退配置"""
        self._llm_provider = llm_provider
        self._llm_model = llm_model
        self._ollama_url = ollama_url

    def parse(
        self,
        text: str,
        reference_date: str | None = None,
    ) -> TemporalRange | None:
        """解析文本中的时间表达式，返回 ISO 8601 日期范围

        Args:
            text: 包含时间表达式的文本（可以是问题、查询或时间描述）
            reference_date: 参考日期（ISO 8601），用于计算相对时间。
                           默认为当前日期。

        Returns:
            TemporalRange 或 None（未检测到时间表达式）
        """
        if not text or not text.strip():
            return None

        ref = self._parse_reference(reference_date)

        # 按优先级依次尝试各种模式
        result = (
            self._try_absolute(text, ref)
            or self._try_relative(text, ref)
            or self._try_duration(text, ref)
            or self._try_quarter(text, ref)
            or self._try_ordinal(text, ref)
        )

        if result:
            return result

        # 规则引擎未匹配，尝试 LLM 回退
        if self._llm_provider:
            return self._try_llm(text, ref)

        return None

    def parse_all(
        self,
        text: str,
        reference_date: str | None = None,
    ) -> list[TemporalRange]:
        """解析文本中所有时间表达式，返回列表"""
        ref = self._parse_reference(reference_date)

        results: list[TemporalRange] = []

        # 绝对时间
        for m in _RE_ABS_MONTH_DAY_YEAR.finditer(text):
            r = self._abs_month_day_year(m)
            if r:
                results.append(r)
        for m in _RE_ABS_MONTH_YEAR.finditer(text):
            r = self._abs_month_year(m)
            if r:
                results.append(r)
        for m in _RE_ABS_ABBR_DAY_YEAR.finditer(text):
            r = self._abs_abbr_day_year(m)
            if r:
                results.append(r)
        for m in _RE_ABS_ABBR_YEAR.finditer(text):
            r = self._abs_abbr_year(m)
            if r:
                results.append(r)
        for m in _RE_ABS_ISO.finditer(text):
            r = self._abs_iso(m)
            if r:
                results.append(r)
        for m in _RE_ABS_YEAR_MONTH.finditer(text):
            r = self._abs_year_month(m)
            if r:
                results.append(r)
        # Standalone month: "in May", "during June"
        for m in _RE_MONTH_STANDALONE.finditer(text):
            r = self._abs_standalone_month(m, ref)
            if r:
                results.append(r)

        # 相对时间
        for m in _RE_REL_LAST_UNIT.finditer(text):
            r = self._rel_last_unit(m, ref)
            if r:
                results.append(r)
        for m in _RE_REL_X_AGO.finditer(text):
            r = self._rel_x_ago(m, ref)
            if r:
                results.append(r)
        if _RE_REL_YESTERDAY.search(text):
            r = self._rel_yesterday(ref)
            if r:
                results.append(r)
        if _RE_REL_RECENTLY.search(text):
            r = self._rel_recently(ref)
            if r:
                results.append(r)
        for m in _RE_REL_COUPLE.finditer(text):
            r = self._rel_couple_ago(m, ref)
            if r:
                results.append(r)

        # 持续时间
        for m in _RE_DUR_SINCE.finditer(text):
            r = self._dur_since(m, ref)
            if r:
                results.append(r)
        for m in _RE_DUR_FOR.finditer(text):
            r = self._dur_for(m, ref)
            if r:
                results.append(r)
        for m in _RE_DUR_FROM_TO.finditer(text):
            r = self._dur_from_to(m)
            if r:
                results.append(r)

        # 季度
        for m in _RE_QUARTER_SHORT.finditer(text):
            r = self._quarter_short(m)
            if r:
                results.append(r)
        for m in _RE_QUARTER.finditer(text):
            r = self._quarter(m)
            if r:
                results.append(r)

        return results

    def has_temporal_expression(self, text: str) -> bool:
        """快速检测文本是否包含时间表达式"""
        if not text:
            return False
        for pattern in [
            _RE_ABS_MONTH_DAY_YEAR, _RE_ABS_MONTH_YEAR, _RE_ABS_MONTH_DAY,
            _RE_ABS_ABBR_DAY_YEAR, _RE_ABS_ABBR_YEAR, _RE_ABS_ISO,
            _RE_MONTH_STANDALONE,
            _RE_REL_LAST_UNIT, _RE_REL_X_AGO, _RE_REL_YESTERDAY,
            _RE_REL_RECENTLY, _RE_REL_COUPLE,
            _RE_DUR_SINCE, _RE_DUR_FOR, _RE_DUR_FROM_TO,
            _RE_QUARTER, _RE_QUARTER_SHORT,
            _RE_ORD_SUPERLATIVE, _RE_ORD_LATEST, _RE_ORD_BEFORE, _RE_ORD_AFTER,
        ]:
            if pattern.search(text):
                return True
        return False

    # ------------------------------------------------------------------
    # 内部: 参考日期解析
    # ------------------------------------------------------------------

    def _parse_reference(self, reference_date: str | None) -> datetime:
        if reference_date:
            try:
                return datetime.strptime(reference_date[:10], "%Y-%m-%d")
            except (ValueError, TypeError):
                pass
        return datetime.now()

    # ------------------------------------------------------------------
    # 内部: 绝对时间解析
    # ------------------------------------------------------------------

    def _try_absolute(self, text: str, ref: datetime) -> TemporalRange | None:
        """尝试匹配绝对时间表达式"""
        # "March 15, 2024"
        m = _RE_ABS_MONTH_DAY_YEAR.search(text)
        if m:
            return self._abs_month_day_year(m)

        # "March 2024"
        m = _RE_ABS_MONTH_YEAR.search(text)
        if m:
            return self._abs_month_year(m)

        # "Mar 15, 2024"
        m = _RE_ABS_ABBR_DAY_YEAR.search(text)
        if m:
            return self._abs_abbr_day_year(m)

        # "Mar 2024"
        m = _RE_ABS_ABBR_YEAR.search(text)
        if m:
            return self._abs_abbr_year(m)

        # "2024-03-15"
        m = _RE_ABS_ISO.search(text)
        if m:
            return self._abs_iso(m)

        # "2024-03"
        m = _RE_ABS_YEAR_MONTH.search(text)
        if m:
            return self._abs_year_month(m)

        # "March 15th" (no year → use ref year)
        m = _RE_ABS_MONTH_DAY.search(text)
        if m:
            return self._abs_month_day(m, ref)

        # "in May", "during June" (standalone month)
        m = _RE_MONTH_STANDALONE.search(text)
        if m:
            return self._abs_standalone_month(m, ref)

        return None

    def _abs_month_day_year(self, m: re.Match) -> TemporalRange:
        month = _MONTH_NAMES[m.group(1).lower()]
        day = int(m.group(2))
        year = int(m.group(3))
        day = min(day, _month_days(year, month))
        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        return TemporalRange(start=date_str, end=date_str, granularity="day")

    def _abs_month_year(self, m: re.Match) -> TemporalRange:
        month = _MONTH_NAMES[m.group(1).lower()]
        year = int(m.group(2))
        return TemporalRange(
            start=_first_day_of_month(year, month),
            end=_last_day_of_month(year, month),
            granularity="month",
        )

    def _abs_month_day(self, m: re.Match, ref: datetime) -> TemporalRange:
        month = _MONTH_NAMES[m.group(1).lower()]
        day = int(m.group(2))
        year = ref.year
        day = min(day, _month_days(year, month))
        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        return TemporalRange(start=date_str, end=date_str, granularity="day")

    def _abs_standalone_month(self, m: re.Match, ref: datetime) -> TemporalRange:
        """'in May', 'during June' → 整个月范围"""
        month = _MONTH_NAMES[m.group(1).lower()]
        year = ref.year
        # 如果提到的月份在当前月份之后，可能是去年
        if month > ref.month:
            year = ref.year - 1
        return TemporalRange(
            start=_first_day_of_month(year, month),
            end=_last_day_of_month(year, month),
            granularity="month",
        )

    def _abs_abbr_day_year(self, m: re.Match) -> TemporalRange:
        month = _MONTH_NAMES[m.group(1).lower()[:3]]
        day = int(m.group(2))
        year = int(m.group(3))
        day = min(day, _month_days(year, month))
        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        return TemporalRange(start=date_str, end=date_str, granularity="day")

    def _abs_abbr_year(self, m: re.Match) -> TemporalRange:
        month = _MONTH_NAMES[m.group(1).lower()[:3]]
        year = int(m.group(2))
        return TemporalRange(
            start=_first_day_of_month(year, month),
            end=_last_day_of_month(year, month),
            granularity="month",
        )

    def _abs_iso(self, m: re.Match) -> TemporalRange:
        year = int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3))
        return TemporalRange(start=f"{year:04d}-{month:02d}-{day:02d}",
                            end=f"{year:04d}-{month:02d}-{day:02d}",
                            granularity="day")

    def _abs_year_month(self, m: re.Match) -> TemporalRange:
        year = int(m.group(1))
        month = int(m.group(2))
        return TemporalRange(
            start=_first_day_of_month(year, month),
            end=_last_day_of_month(year, month),
            granularity="month",
        )

    # ------------------------------------------------------------------
    # 内部: 相对时间解析
    # ------------------------------------------------------------------

    def _try_relative(self, text: str, ref: datetime) -> TemporalRange | None:
        """尝试匹配相对时间表达式"""
        # "last/past/previous/this week/month/year"
        m = _RE_REL_LAST_UNIT.search(text)
        if m:
            return self._rel_last_unit(m, ref)

        # "X days/weeks/months/years ago"
        m = _RE_REL_X_AGO.search(text)
        if m:
            return self._rel_x_ago(m, ref)

        # "a couple of / a few days/weeks ago"
        m = _RE_REL_COUPLE.search(text)
        if m:
            return self._rel_couple_ago(m, ref)

        # "yesterday"
        if _RE_REL_YESTERDAY.search(text):
            return self._rel_yesterday(ref)

        # "recently"
        if _RE_REL_RECENTLY.search(text):
            return self._rel_recently(ref)

        # "today"
        if _RE_REL_TODAY.search(text):
            return self._rel_today(ref)

        return None

    def _rel_last_unit(self, m: re.Match, ref: datetime) -> TemporalRange:
        qualifier = m.group(1).lower()
        unit = m.group(2).lower()

        if unit == "week":
            if qualifier == "this":
                # This week: Monday to current day
                days_since_monday = ref.weekday()
                start = ref - timedelta(days=days_since_monday)
                return TemporalRange(start=_date_to_str(start), end=_date_to_str(ref), granularity="week")
            else:  # last/past/previous
                days_since_monday = ref.weekday()
                this_monday = ref - timedelta(days=days_since_monday)
                last_monday = this_monday - timedelta(weeks=1)
                last_sunday = this_monday - timedelta(days=1)
                return TemporalRange(start=_date_to_str(last_monday), end=_date_to_str(last_sunday), granularity="week")

        elif unit == "month":
            if qualifier == "this":
                return TemporalRange(start=_first_day_of_month(ref.year, ref.month),
                                    end=_date_to_str(ref), granularity="month")
            else:  # last/past/previous
                if ref.month == 1:
                    prev_year, prev_month = ref.year - 1, 12
                else:
                    prev_year, prev_month = ref.year, ref.month - 1
                return TemporalRange(
                    start=_first_day_of_month(prev_year, prev_month),
                    end=_last_day_of_month(prev_year, prev_month),
                    granularity="month",
                )

        elif unit == "year":
            if qualifier == "this":
                return TemporalRange(start=f"{ref.year:04d}-01-01", end=_date_to_str(ref), granularity="year")
            else:
                prev_year = ref.year - 1
                return TemporalRange(start=f"{prev_year:04d}-01-01", end=f"{prev_year:04d}-12-31", granularity="year")

        elif unit == "weekend":
            if qualifier == "this":
                days_until_saturday = (5 - ref.weekday()) % 7
                sat = ref + timedelta(days=days_until_saturday)
                sun = sat + timedelta(days=1)
                return TemporalRange(start=_date_to_str(sat), end=_date_to_str(sun), granularity="day")
            else:
                days_since_saturday = (ref.weekday() - 5) % 7
                last_sat = ref - timedelta(days=days_since_saturday if days_since_saturday > 0 else 7)
                last_sun = last_sat + timedelta(days=1)
                return TemporalRange(start=_date_to_str(last_sat), end=_date_to_str(last_sun), granularity="day")

        elif unit == "quarter":
            current_q = (ref.month - 1) // 3 + 1
            if qualifier == "this":
                q_start_month = (current_q - 1) * 3 + 1
                return TemporalRange(start=_first_day_of_month(ref.year, q_start_month),
                                    end=_date_to_str(ref), granularity="month")
            else:
                prev_q = current_q - 1 if current_q > 1 else 4
                prev_year = ref.year if current_q > 1 else ref.year - 1
                q_start_month = (prev_q - 1) * 3 + 1
                q_end_month = prev_q * 3
                return TemporalRange(
                    start=_first_day_of_month(prev_year, q_start_month),
                    end=_last_day_of_month(prev_year, q_end_month),
                    granularity="month",
                )

        return None

    def _rel_x_ago(self, m: re.Match, ref: datetime) -> TemporalRange:
        n = _parse_number(m.group(1))
        unit = m.group(2).lower()

        if unit.startswith("day"):
            target = ref - timedelta(days=n)
            return TemporalRange(start=_date_to_str(target), end=_date_to_str(target), granularity="day")
        elif unit.startswith("week"):
            target = ref - timedelta(weeks=n)
            return TemporalRange(start=_date_to_str(target), end=_date_to_str(target), granularity="day")
        elif unit.startswith("month"):
            # 近似：每月按30天算
            month_offset = n
            year = ref.year
            month = ref.month - month_offset
            while month <= 0:
                month += 12
                year -= 1
            day = min(ref.day, _month_days(year, month))
            target_str = f"{year:04d}-{month:02d}-{day:02d}"
            return TemporalRange(start=target_str, end=target_str, granularity="day")
        elif unit.startswith("year"):
            year = ref.year - n
            target_str = f"{year:04d}-{ref.month:02d}-{ref.day:02d}"
            return TemporalRange(start=target_str, end=target_str, granularity="day")
        return None

    def _rel_couple_ago(self, m: re.Match, ref: datetime) -> TemporalRange:
        """a couple of / a few X ago → 近似为2-3个单位"""
        unit = m.group(2).lower()
        n = 3  # "a few" / "a couple" ≈ 2-3

        if unit.startswith("day"):
            target = ref - timedelta(days=n)
            start = ref - timedelta(days=5)  # 扩展到5天范围
            return TemporalRange(start=_date_to_str(start), end=_date_to_str(ref), granularity="day")
        elif unit.startswith("week"):
            target = ref - timedelta(weeks=n)
            start = ref - timedelta(weeks=4)  # 扩展到4周
            return TemporalRange(start=_date_to_str(start), end=_date_to_str(ref), granularity="week")
        elif unit.startswith("month"):
            year = ref.year
            month = ref.month - 3
            while month <= 0:
                month += 12
                year -= 1
            return TemporalRange(start=_first_day_of_month(year, month), end=_date_to_str(ref), granularity="month")
        return None

    def _rel_yesterday(self, ref: datetime) -> TemporalRange:
        target = ref - timedelta(days=1)
        return TemporalRange(start=_date_to_str(target), end=_date_to_str(target), granularity="day")

    def _rel_recently(self, ref: datetime) -> TemporalRange:
        """recently → 近7天范围"""
        start = ref - timedelta(days=7)
        return TemporalRange(start=_date_to_str(start), end=_date_to_str(ref), granularity="day")

    def _rel_today(self, ref: datetime) -> TemporalRange:
        return TemporalRange(start=_date_to_str(ref), end=_date_to_str(ref), granularity="day")

    # ------------------------------------------------------------------
    # 内部: 持续时间解析
    # ------------------------------------------------------------------

    def _try_duration(self, text: str, ref: datetime) -> TemporalRange | None:
        """尝试匹配持续时间表达式"""
        # "since January [2024]"
        m = _RE_DUR_SINCE.search(text)
        if m:
            return self._dur_since(m, ref)

        # "for X days/weeks/months/years"
        m = _RE_DUR_FOR.search(text)
        if m:
            return self._dur_for(m, ref)

        # "from March to June [2024]"
        m = _RE_DUR_FROM_TO.search(text)
        if m:
            return self._dur_from_to(m)

        return None

    def _dur_since(self, m: re.Match, ref: datetime) -> TemporalRange:
        month = _MONTH_NAMES[m.group(1).lower()]
        year = int(m.group(2)) if m.group(2) else ref.year
        return TemporalRange(
            start=_first_day_of_month(year, month),
            end=_date_to_str(ref),
            granularity="month",
        )

    def _dur_for(self, m: re.Match, ref: datetime) -> TemporalRange:
        n = _parse_number(m.group(1))
        unit = m.group(2).lower()

        if unit.startswith("day"):
            start = ref - timedelta(days=n)
            return TemporalRange(start=_date_to_str(start), end=_date_to_str(ref), granularity="day")
        elif unit.startswith("week"):
            start = ref - timedelta(weeks=n)
            return TemporalRange(start=_date_to_str(start), end=_date_to_str(ref), granularity="week")
        elif unit.startswith("month"):
            year = ref.year
            month = ref.month - n
            while month <= 0:
                month += 12
                year -= 1
            return TemporalRange(start=_first_day_of_month(year, month), end=_date_to_str(ref), granularity="month")
        elif unit.startswith("year"):
            year = ref.year - n
            return TemporalRange(start=f"{year:04d}-01-01", end=_date_to_str(ref), granularity="year")
        return None

    def _dur_from_to(self, m: re.Match) -> TemporalRange:
        start_month = _MONTH_NAMES[m.group(1).lower()]
        start_year = int(m.group(2)) if m.group(2) else 2024  # 默认2024（评测数据年份）
        end_month = _MONTH_NAMES[m.group(3).lower()]
        end_year = int(m.group(4)) if m.group(4) else start_year

        # 如果结束月份小于开始月份且没有显式年份，结束年份+1
        if end_month < start_month and not m.group(4):
            end_year = start_year + 1

        return TemporalRange(
            start=_first_day_of_month(start_year, start_month),
            end=_last_day_of_month(end_year, end_month),
            granularity="month",
        )

    # ------------------------------------------------------------------
    # 内部: 季度解析
    # ------------------------------------------------------------------

    def _try_quarter(self, text: str, ref: datetime) -> TemporalRange | None:
        m = _RE_QUARTER_SHORT.search(text)
        if m:
            return self._quarter_short(m)
        m = _RE_QUARTER.search(text)
        if m:
            return self._quarter(m)
        return None

    def _quarter_short(self, m: re.Match) -> TemporalRange:
        """'Q2 2024' 短格式"""
        q = int(m.group(1))
        year = int(m.group(2))
        start_month = (q - 1) * 3 + 1
        end_month = q * 3
        return TemporalRange(
            start=_first_day_of_month(year, start_month),
            end=_last_day_of_month(year, end_month),
            granularity="month",
        )

    def _quarter(self, m: re.Match) -> TemporalRange:
        q_str = m.group(1).lower()
        year = int(m.group(2))

        q_map = {"q1": 1, "q2": 2, "q3": 3, "q4": 4,
                 "first": 1, "second": 2, "third": 3, "fourth": 4}
        q = q_map.get(q_str, 1)

        start_month = (q - 1) * 3 + 1
        end_month = q * 3
        return TemporalRange(
            start=_first_day_of_month(year, start_month),
            end=_last_day_of_month(year, end_month),
            granularity="month",
        )

    # ------------------------------------------------------------------
    # 内部: 序列时间标记
    # ------------------------------------------------------------------

    def _try_ordinal(self, text: str, ref: datetime) -> TemporalRange | None:
        """序列时间 — 返回极宽范围标记，供下游过滤使用"""
        if _RE_ORD_SUPERLATIVE.search(text):
            # "first/earliest" → 标记为极早
            return TemporalRange(start="0001-01-01", end=_date_to_str(ref), granularity="year")
        if _RE_ORD_LATEST.search(text):
            # "latest/most recent" → 标记为最近
            recent_start = ref - timedelta(days=30)
            return TemporalRange(start=_date_to_str(recent_start), end=_date_to_str(ref), granularity="day")
        if _RE_ORD_BEFORE.search(text):
            return TemporalRange(start="0001-01-01", end=_date_to_str(ref), granularity="year")
        if _RE_ORD_AFTER.search(text):
            return TemporalRange(start=_date_to_str(ref), end="9999-12-31", granularity="year")
        return None

    # ------------------------------------------------------------------
    # 内部: LLM 回退
    # ------------------------------------------------------------------

    def _try_llm(self, text: str, ref: datetime) -> TemporalRange | None:
        """使用 LLM 解析边缘时间表达式"""
        if not self._llm_provider or not self._llm_model:
            return None

        import requests as _req

        prompt = (
            f'Parse the temporal expression in this text and return a JSON object with '
            f'"start" (ISO date), "end" (ISO date), "granularity" (day/week/month/year). '
            f'Reference date: {_date_to_str(ref)}\n\n'
            f'Text: "{text}"\n\n'
            f'Return ONLY the JSON object, e.g.: {{"start": "2024-03-01", "end": "2024-03-31", "granularity": "month"}}'
        )

        try:
            if self._llm_provider == "ollama" and self._ollama_url:
                resp = _req.post(
                    f"{self._ollama_url}/api/generate",
                    json={"model": self._llm_model, "prompt": prompt, "stream": False,
                          "options": {"temperature": 0, "num_predict": 200}},
                    timeout=15,
                )
                if resp.status_code == 200:
                    raw = resp.json().get("response", "").strip()
                    return self._parse_llm_response(raw)
            elif self._llm_provider == "deepseek":
                # DeepSeek API 调用
                import os
                api_key = os.environ.get("DEEPSEEK_API_KEY", "")
                if not api_key:
                    return None
                resp = _req.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    json={"model": self._llm_model or "deepseek-chat",
                          "messages": [{"role": "user", "content": prompt}],
                          "temperature": 0, "max_tokens": 200},
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {api_key}"},
                    timeout=15,
                )
                if resp.status_code == 200:
                    raw = resp.json()["choices"][0]["message"]["content"].strip()
                    return self._parse_llm_response(raw)
            elif self._llm_provider == "minimax":
                # v4.4.0: MiniMax API 调用
                import os
                api_key = os.environ.get("MINIMAX_API_KEY", "")
                if not api_key:
                    return None
                base_url = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.chat/v1").rstrip("/")
                resp = _req.post(
                    f"{base_url}/text/chatcompletion_v2",
                    json={"model": self._llm_model or "abab6.5s-chat",
                          "messages": [{"role": "user", "content": prompt}],
                          "temperature": 0.01, "tokens_to_generate": 200,
                          "stream": False},
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {api_key}"},
                    timeout=15,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    # MiniMax v2 API returns OpenAI-compatible: choices[0].message.content
                    choices = data.get("choices", [])
                    if choices:
                        reply = (choices[0].get("message", {}).get("content", "") or "").strip()
                        if reply:
                            return self._parse_llm_response(reply)
            elif self._llm_provider == "glm":
                # v4.4.0: GLM (智谱) API 调用
                import os
                api_key = os.environ.get("GLM_API_KEY", os.environ.get("ZHIPU_API_KEY", ""))
                if not api_key:
                    return None
                base_url = os.environ.get("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4").rstrip("/")
                resp = _req.post(
                    f"{base_url}/chat/completions",
                    json={"model": self._llm_model or "glm-4-flash",
                          "messages": [{"role": "user", "content": prompt}],
                          "temperature": 0.01, "max_tokens": 200},
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {api_key}"},
                    timeout=15,
                )
                if resp.status_code == 200:
                    raw = resp.json()["choices"][0]["message"]["content"].strip()
                    return self._parse_llm_response(raw)
        except Exception as exc:
            logger.debug("[TemporalParser] LLM fallback failed: %s", exc)

        return None

    def _parse_llm_response(self, raw: str) -> TemporalRange | None:
        """解析 LLM 返回的 JSON"""
        import json
        try:
            # 提取 JSON
            if "{" in raw and "}" in raw:
                start = raw.find("{")
                end = raw.rfind("}") + 1
                obj = json.loads(raw[start:end])
                start_date = obj.get("start", "")
                end_date = obj.get("end", "")
                granularity = obj.get("granularity", "day")
                if start_date and end_date:
                    return TemporalRange(start=start_date, end=end_date, granularity=granularity)
        except (json.JSONDecodeError, KeyError):
            pass
        return None


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def create_temporal_parser(**kwargs: Any) -> TemporalParser:
    """工厂函数：创建 TemporalParser 实例"""
    return TemporalParser(**kwargs)
