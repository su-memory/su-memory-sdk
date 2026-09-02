"""
_span_refiner 答案边界精修器测试

纯函数覆盖: 归一化、实词提取、括号/引号清理、时间类问题年份提取、
候选 span 查找与投票选择、refine_answer 端到端行为。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from su_memory.sdk._span_refiner import (
    _content_words,
    _find_candidates,
    _is_time_question,
    _normalize,
    _select_best,
    _strip_paren_comment,
    _year_only_for_time_answer,
    refine_answer,
)

CTX = ("Chief of Protocol of the United States oversees protocol matters "
       "in Washington DC since 1994 and retired in 2013")


class TestHelpers:
    def test_normalize(self):
        assert _normalize("The Kansas Song, We're (From) Kansas!") == \
            "kansas song were from kansas"

    def test_content_words_filters_stopwords(self):
        words = _content_words("the Chief of Protocol of the United States")
        assert words == ["chief", "protocol", "united", "states"]

    def test_strip_paren_comment(self):
        assert _strip_paren_comment("Kansas Song (We're From Kansas)") == "Kansas Song"
        assert _strip_paren_comment("plain answer") == "plain answer"

    def test_is_time_question(self):
        assert _is_time_question("When did he serve?") is True
        assert _is_time_question("what year was she born") is True
        assert _is_time_question("which year did they move") is True
        assert _is_time_question("What is the date?") is True
        assert _is_time_question("who is the president") is False
        assert _is_time_question("") is False

    def test_year_only_for_time_answer(self):
        assert _year_only_for_time_answer("1986") == "1986"
        assert _year_only_for_time_answer(" in 1994 he joined ") == "1994"
        assert _year_only_for_time_answer("1986 to 2013") is None  # 范围保留
        assert _year_only_for_time_answer("no year here") is None
        assert _year_only_for_time_answer("") is None


class TestRefineAnswer:
    def test_empty_pred(self):
        assert refine_answer("", CTX, "") == ""

    def test_paren_comment_stripped(self):
        assert refine_answer("Kansas Song (We're From Kansas)", CTX, "") == "Kansas Song"

    def test_quotes_stripped(self):
        assert refine_answer('"Chief of Protocol"', CTX, "") == "Chief of Protocol"

    def test_short_answer_kept(self):
        assert refine_answer("Chief of Protocol", CTX, "") == "Chief of Protocol"

    def test_yes_no_kept(self):
        assert refine_answer("yes", CTX, "Did he agree?") == "yes"
        assert refine_answer("no", CTX, "Did he agree?") == "no"

    def test_exact_context_match_kept(self):
        pred = "Chief of Protocol of the United States oversees"
        assert refine_answer(pred, CTX, "") == pred

    def test_time_question_extracts_year(self):
        assert refine_answer("from 1994 he served as Chief of Protocol",
                             CTX, "When did he serve?") == "1994"
        assert refine_answer("in 1994 he joined the office",
                             CTX, "what year did he join?") == "1994"

    def test_time_question_keeps_year_range(self):
        pred = "1986 to 2013 he held the office"
        out = refine_answer(pred, CTX, "When did he hold office?")
        assert "1986" in out  # 范围不精简, 走后续精修

    def test_not_in_context_keeps_pred(self):
        pred = "a completely unrelated long answer phrase here"
        assert refine_answer(pred, "无关上下文", "") == pred

    def test_find_candidates_returns_window(self):
        cands = _find_candidates("Chief of Protocol of the United States", CTX)
        assert cands
        assert cands[0] == "Chief of Protocol of the United States"

    def test_find_candidates_no_keywords(self):
        assert _find_candidates("the of and", CTX) == []

    def test_select_best_prefers_context_candidate(self):
        pred = "Chief of Protocol of the United States oversees"
        cands = _find_candidates(pred, CTX)
        best = _select_best(pred, cands, "")
        assert best is None or best in CTX
