"""SpanRefiner — 答案边界精修器。

问题: LLM 生成的答案与 HotpotQA gold 之间有 ~41% 的边界偏差:
  - 模型过度生成: gold='Chief of Protocol', pred='Chief of Protocol of the United States'
  - 模型生成不足: gold='from 1986 to 2013', pred='1986 to 2013'
  - 括号注释: gold='Kansas Song', pred='Kansas Song (We're From Kansas)'

策略: 生成多个候选 span, 在 context 中找最匹配的精确子串, 投票选最优。
"""
from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation)

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "was", "are", "were", "of", "in", "on", "at",
    "to", "for", "by", "from", "with", "and", "or", "not", "that", "this",
    "it", "he", "she", "they", "context", "does", "provide", "information",
    "about", "question", "answer", "based", "given", "according",
})


def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in s if c not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _content_words(s: str) -> list[str]:
    s_lower = s.lower()
    return [w.strip(".,;:!?\"'()[]") for w in s_lower.split()
            if w.strip(".,;:!?\"'()[]") not in _STOPWORDS
            and len(w.strip(".,;:!?\"'()[]")) > 2]


def refine_answer(pred: str, context: str, question: str = "") -> str:
    """精修 LLM 输出的答案 span。

    Args:
        pred: LLM 原始输出 (已解析出 Answer: 后的内容)
        context: 检索到的上下文段落
        question: 原始问题 (用于判断答案类型)

    Returns:
        精修后的答案 span
    """
    pred = pred.strip().strip(".").strip()
    if not pred:
        return pred

    # 1. 去掉括号注释: "Kansas Song (We're From Kansas)" -> "Kansas Song"
    paren = re.match(r'^(.+?)\s*\([^)]+\)\s*$', pred)
    if paren:
        pred = paren.group(1).strip()

    # 2. 去掉引号
    pred = pred.strip('"\'""').strip()

    # 3. 日期问题: 当问题含 when/year/date 时, 智能提取日期
    q_lower = question.lower()
    if ("when" in q_lower or "what year" in q_lower or "which year" in q_lower
            or "date" in q_lower):
        # 如果 pred 已经是纯年份, 直接返回
        if re.match(r"^(18\d{2}|19\d{2}|20\d{2})$", pred.strip()):
            return pred.strip()
        # 如果 pred 含年份但不是范围格式 (没有 "to"/"-" 连接两个年份), 提取纯年份
        has_range = bool(re.search(r"\d{4}\s*(?:to|-|–)\s*\d{4}", pred))
        if not has_range:
            year_match = re.search(r"\b(18\d{2}|19\d{2}|20\d{2})\b", pred)
            if year_match and len(pred.split()) > 1:
                return year_match.group(1)

    words = pred.split()
    if len(words) <= 3:
        return pred

    # 3. yes/no 直接返回
    if _normalize(pred) in {"yes", "no"}:
        return pred

    # 4. 如果 pred 精确出现在 context 中 (不区分大小写), 直接返回
    ctx_lower = context.lower()
    pred_lower = pred.lower()
    if pred_lower in ctx_lower:
        return pred

    # 5. 在 context 中找候选 span
    candidates = _find_candidates(pred, context)
    if not candidates:
        return pred

    # 6. 如果原 pred 比所有候选都短, 可能是 under-generation, 尝试扩展
    # 如果原 pred 比候选长, 可能是 over-generation, 尝试缩短
    best = _select_best(pred, candidates, question)
    return best if best else pred


def _find_candidates(pred: str, context: str) -> list[str]:
    """在 context 中找到所有包含 pred 关键词的候选 span。"""
    ctx_lower = context.lower()
    key_words = _content_words(pred)
    if not key_words:
        return []

    candidates: list[str] = []

    # 策略 A: 找 pred 第一个实词在 context 中的位置, 取前后窗口
    first_kw = key_words[0]
    start = 0
    while True:
        idx = ctx_lower.find(first_kw.lower(), start)
        if idx < 0:
            break
        # 往前找到词边界
        begin = idx
        while begin > 0 and context[begin - 1] not in " \n\t,;.":
            begin -= 1
        # 往后找包含尽可能多 key_words 的窗口
        end = idx + len(first_kw)
        matched = 1
        for kw in key_words[1:]:
            pos = ctx_lower.find(kw.lower(), end)
            if 0 < pos - end < 50:
                end = pos + len(kw)
                matched += 1
            else:
                break
        # 截断到标点
        snippet = context[begin:end]
        for delim in [". ", "; ", "\n", ", "]:
            cut = snippet.find(delim)
            if 0 < cut:
                snippet = snippet[:cut]
                break
        snippet = snippet.strip(" ,;:")
        if snippet and len(snippet.split()) <= 10:
            candidates.append(snippet)
        start = idx + len(first_kw)

    # 策略 B: 如果 pred 含逗号, 尝试去掉逗号后的部分或前的部分
    if "," in pred:
        parts = pred.split(",")
        for part in parts:
            p = part.strip()
            if p and p.lower() in ctx_lower:
                candidates.append(p)

    # 去重
    seen = set()
    unique = []
    for c in candidates:
        key = _normalize(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def _select_best(pred: str, candidates: list[str], question: str) -> str | None:
    """从候选 span 中选择最可能匹配 gold 的那个。

    HotpotQA gold 的统计特征:
    - 倾向于短而精确 (中位数 ~3 词)
    - 人名: First Last 或 First Middle Last
    - 日期: 纯年份 或 Month Day, Year
    - 数字: 纯数字 或 number + 单位
    """
    if not candidates:
        return None

    pred_norm = _normalize(pred)
    pred_words = pred.split()

    scored: list[tuple[float, str]] = []
    for cand in candidates:
        cand_norm = _normalize(cand)
        cand_words = cand.split()
        score = 0.0

        # 如果候选是 pred 的子串 (over-generation 修正), 偏好短候选
        if cand_norm in pred_norm and cand_norm != pred_norm:
            score += 3.0  # 强偏好: 候选是 pred 的一部分
            score -= len(cand_words) * 0.1  # 越短越好

        # 如果 pred 是候选的子串 (under-generation 修正), 偏好包含 pred 的候选
        elif pred_norm in cand_norm and cand_norm != pred_norm:
            score += 1.5  # 中等偏好
            score -= (len(cand_words) - len(pred_words)) * 0.3

        # 候选与 pred 完全相同 (normalized), 但原始形式可能更精确
        elif cand_norm == pred_norm:
            score += 2.0
            # 偏好原始 context 中的大小写形式
            if cand != pred:
                score += 0.5

        # 一般情况: 偏好短候选 (HotpotQA gold 偏短)
        else:
            score -= len(cand_words) * 0.2
            # 但如果共享很多词, 也有价值
            common = len(set(cand_words) & set(pred_words))
            score += common * 0.3

        # 问题类型感知
        q_lower = question.lower()
        if "when" in q_lower or "what year" in q_lower:
            # 日期问题: 偏好纯数字/年份
            if re.match(r"^\d{4}$", cand.strip()):
                score += 2.0
            elif re.search(r"\d{4}", cand):
                score += 0.5
        elif "where" in q_lower:
            # 地点问题: 偏好 "City, State" 格式
            if "," in cand:
                score += 0.5
        elif "who" in q_lower:
            # 人名问题: 偏好 2-3 词
            if 2 <= len(cand_words) <= 3:
                score += 0.5

        scored.append((score, cand))

    scored.sort(reverse=True)
    return scored[0][1] if scored else None


__all__ = ["refine_answer"]
