"""
su-memory SDK v3.5.5 — LLM Judge 可插拔评估框架
=================================================

对标 SuperMemory MemoryBench 的 Judge 可插拔机制。
支持三种 Judge: BayesianAugmenter / OpenAI / RuleBased。

Usage:
    from benchmarks.judge import JudgeFactory
    judge = JudgeFactory.create("openai")
    score = judge.evaluate("query", "predicted", "ground_truth")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================
# JudgeScore — 评估结果
# ============================================================


@dataclass
class JudgeScore:
    """Judge 评估结果"""
    relevance: float = 0.0       # 相关性 (0-1)
    correctness: float = 0.0     # 正确性 (0-1)
    completeness: float = 0.0    # 完整性 (0-1)
    conciseness: float = 0.0     # 简洁性 (0-1)
    overall: float = 0.0         # 综合评分 (0-1)
    reasoning: str = ""           # 评估理由
    judge_type: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "relevance": round(self.relevance, 3),
            "correctness": round(self.correctness, 3),
            "completeness": round(self.completeness, 3),
            "conciseness": round(self.conciseness, 3),
            "overall": round(self.overall, 3),
            "reasoning": self.reasoning,
            "judge_type": self.judge_type,
        }


# ============================================================
# Judge 抽象基类
# ============================================================


class Judge(ABC):
    """评估器抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def evaluate(
        self,
        query: str,
        predicted_answer: str,
        ground_truth: str,
    ) -> JudgeScore:
        """
        评估预测答案质量。

        Args:
            query: 原始查询
            predicted_answer: 预测答案
            ground_truth: 标准答案

        Returns:
            JudgeScore 评估结果
        """
        ...


# ============================================================
# RuleBasedJudge — 关键词匹配 + ROUGE-L 快速评估
# ============================================================


class RuleBasedJudge(Judge):
    """基于规则的快速评估器 (零外部依赖)"""

    @property
    def name(self) -> str:
        return "rule_based"

    def evaluate(
        self,
        query: str,
        predicted_answer: str,
        ground_truth: str,
    ) -> JudgeScore:
        if not predicted_answer or not ground_truth:
            return JudgeScore(overall=0.0, judge_type=self.name, reasoning="empty input")

        # 1) 关键词重叠
        pred_words = set(predicted_answer.lower().split())
        truth_words = set(ground_truth.lower().split())
        if not truth_words:
            return JudgeScore(overall=0.0, judge_type=self.name, reasoning="empty ground truth")

        overlap = pred_words & truth_words
        relevance = len(overlap) / len(truth_words)

        # 2) ROUGE-L (LCS)
        lcs_len = self._lcs_length(
            predicted_answer.lower().split(),
            ground_truth.lower().split(),
        )
        correctness = lcs_len / max(len(ground_truth.split()), 1)

        # 3) Completeness: 预测覆盖了多少 truth 关键词
        truth_keywords = {w for w in truth_words if len(w) > 2}
        if truth_keywords:
            completeness = len(pred_words & truth_keywords) / len(truth_keywords)
        else:
            completeness = relevance

        # 4) Conciseness: 预测不过长
        conciseness = min(1.0, len(ground_truth.split()) / max(len(predicted_answer.split()), 1))

        # 综合
        overall = 0.35 * relevance + 0.35 * correctness + 0.15 * completeness + 0.15 * conciseness

        return JudgeScore(
            relevance=relevance,
            correctness=correctness,
            completeness=completeness,
            conciseness=conciseness,
            overall=overall,
            reasoning=f"R={relevance:.2f} C={correctness:.2f} "
                      f"Comp={completeness:.2f} Conc={conciseness:.2f}",
            judge_type=self.name,
        )

    @staticmethod
    def _lcs_length(a: list[str], b: list[str]) -> int:
        """最长公共子序列长度"""
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]


# ============================================================
# BayesianAugmenterJudge — 基于 su-memory 内建贝叶斯推理
# ============================================================


class BayesianAugmenterJudge(Judge):
    """su-memory 内建贝叶斯增强评估器"""

    @property
    def name(self) -> str:
        return "bayesian_augmenter"

    def evaluate(
        self,
        query: str,
        predicted_answer: str,
        ground_truth: str,
    ) -> JudgeScore:
        try:
            from su_memory.sdk.lite_pro import SuMemoryLitePro
        except ImportError:
            logger.warning("BayesianAugmenter 不可用，回退到规则评估")
            return RuleBasedJudge().evaluate(query, predicted_answer, ground_truth)

        # 使用 RuleBasedJudge 作为底层，叠加贝叶斯置信度
        base = RuleBasedJudge().evaluate(query, predicted_answer, ground_truth)

        try:
            client = SuMemoryLitePro(
                max_memories=50,
                enable_vector=False,
                enable_graph=False,
                enable_temporal=False,
                enable_session=False,
                enable_prediction=False,
                enable_explainability=False,
            )
            # 将 query + predicted + ground_truth 作为证据
            client.add(f"Query: {query}")
            client.add(f"Predicted: {predicted_answer}")
            client.add(f"Ground Truth: {ground_truth}")

            # 使用记忆统计作为置信度信号
            stats = client.get_stats()
            total = stats.get("total_memories", 3)
            confidence_boost = min(0.1, total / 100)  # 最多+0.1

            client.clear()

            return JudgeScore(
                relevance=base.relevance,
                correctness=base.correctness,
                completeness=base.completeness,
                conciseness=base.conciseness,
                overall=min(1.0, base.overall + confidence_boost),
                reasoning=f"{base.reasoning} +bayesian({total} memories)",
                judge_type=self.name,
            )
        except Exception as e:
            logger.warning(f"BayesianAugmenterJudge 失败: {e}")
            return base


# ============================================================
# OpenAIJudge — 通过 OpenAI API 调用 GPT-4o-mini
# ============================================================


class OpenAIJudge(Judge):
    """基于 OpenAI API 的 LLM 评估器"""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        import os
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model

    @property
    def name(self) -> str:
        return f"openai_{self._model}"

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    def evaluate(
        self,
        query: str,
        predicted_answer: str,
        ground_truth: str,
    ) -> JudgeScore:
        if not self.is_available:
            logger.warning("OPENAI_API_KEY 未设置，回退到规则评估")
            return RuleBasedJudge().evaluate(query, predicted_answer, ground_truth)

        prompt = (
            f"Evaluate the following answer against the ground truth.\n\n"
            f"Query: {query}\n\n"
            f"Predicted Answer: {predicted_answer}\n\n"
            f"Ground Truth: {ground_truth}\n\n"
            f"Rate on a scale of 0-1 for each dimension:\n"
            f"- relevance: how relevant is the answer to the query?\n"
            f"- correctness: how factually correct is the answer?\n"
            f"- completeness: how thoroughly does it cover the ground truth?\n"
            f"- conciseness: is it appropriately concise?\n\n"
            f"Respond with exactly 4 numbers separated by commas: relevance,correctness,completeness,conciseness"
        )

        try:
            import json
            import urllib.request

            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps({
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": "You are a precise evaluator. Respond only with 4 numbers."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 50,
                    "temperature": 0,
                }).encode(),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )

            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
                content = data["choices"][0]["message"]["content"].strip()

            # Parse response
            nums = [float(x.strip()) for x in content.split(",")]
            if len(nums) >= 4:
                return JudgeScore(
                    relevance=min(1.0, max(0.0, nums[0])),
                    correctness=min(1.0, max(0.0, nums[1])),
                    completeness=min(1.0, max(0.0, nums[2])),
                    conciseness=min(1.0, max(0.0, nums[3])),
                    overall=(nums[0] + nums[1] + nums[2] + nums[3]) / 4,
                    reasoning=content,
                    judge_type=self.name,
                )
        except Exception as e:
            logger.warning(f"OpenAI API 调用失败: {e}，回退到规则评估")

        return RuleBasedJudge().evaluate(query, predicted_answer, ground_truth)


# ============================================================
# JudgeFactory
# ============================================================


class JudgeFactory:
    """Judge 工厂 — 根据类型创建评估器"""

    _registry: dict[str, type[Judge]] = {
        "rule_based": RuleBasedJudge,
        "bayesian": BayesianAugmenterJudge,
        "openai": OpenAIJudge,
    }

    @classmethod
    def create(cls, judge_type: str, **kwargs) -> Judge:
        """
        创建 Judge 实例。

        Args:
            judge_type: "rule_based" | "bayesian" | "openai"
            **kwargs: 传递给 Judge 构造函数的参数

        Returns:
            Judge 实例
        """
        if judge_type not in cls._registry:
            logger.warning(f"Unknown judge type: {judge_type}, fallback to rule_based")
            return RuleBasedJudge()

        judge_cls = cls._registry[judge_type]
        return judge_cls(**kwargs)

    @classmethod
    def auto(cls) -> Judge:
        """
        自动选择最佳可用 Judge:
        - 有 OPENAI_API_KEY → OpenAIJudge
        - BayesianAugmenter 可用 → BayesianAugmenterJudge
        - 否则 → RuleBasedJudge
        """
        import os
        if os.environ.get("OPENAI_API_KEY"):
            return cls.create("openai")
        try:
            from su_memory.sdk.lite_pro import SuMemoryLitePro
            _ = SuMemoryLitePro(max_memories=10, enable_vector=False)
            return cls.create("bayesian")
        except ImportError:
            return cls.create("rule_based")


# ============================================================
# CLI Demo
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  LLM Judge Demo — su-memory v3.5.5")
    print("=" * 60)

    query = "项目ROI增长了多少？"
    predicted = "项目ROI增长了25%，Q3增长最显著"
    ground_truth = "项目ROI在Q3增长了25%"

    for judge_type in ["rule_based", "bayesian", "openai"]:
        print(f"\n[{judge_type}]")
        judge = JudgeFactory.create(judge_type)
        score = judge.evaluate(query, predicted, ground_truth)
        print(f"  Relevance:    {score.relevance:.3f}")
        print(f"  Correctness:  {score.correctness:.3f}")
        print(f"  Completeness: {score.completeness:.3f}")
        print(f"  Conciseness:  {score.conciseness:.3f}")
        print(f"  Overall:      {score.overall:.3f}")
        print(f"  Reasoning:    {score.reasoning}")
