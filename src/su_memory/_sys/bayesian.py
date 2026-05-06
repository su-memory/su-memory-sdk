"""
贝叶斯推理核心引擎

提供完整的概率推断能力：
1. 先验概率建模（Beta共轭先验）
2. 似然函数计算
3. 后验概率更新（贝叶斯定理）
4. 不确定性量化（方差、置信区间）
5. 假设检验与模型比较

对外暴露：BayesianEngine
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import math
import time
import json


# ============================================================
# 数据结构
# ============================================================

@dataclass
class BetaDistribution:
    """
    Beta分布 — 信念的共轭先验

    用于建模二项分布的概率参数 p 的不确定性。

    Beta(α, β):
    - α: 正面证据（成功次数 + 先验）
    - β: 负面证据（失败次数 + 先验）
    - 均值: α / (α + β)
    - 方差: αβ / ((α+β)²(α+β+1))
    - 众数: (α-1)/(α+β-2)  (α,β > 1)
    """
    alpha: float = 1.0   # 伪计数：正面证据
    beta: float = 1.0    # 伪计数：负面证据

    @property
    def mean(self) -> float:
        """后验期望概率 E[p]"""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """后验方差 Var[p]"""
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total * total * (total + 1))

    @property
    def std(self) -> float:
        """后验标准差"""
        return math.sqrt(self.variance)

    @property
    def mode(self) -> float:
        """后验众数（最大后验估计 MAP）"""
        if self.alpha > 1 and self.beta > 1:
            return (self.alpha - 1) / (self.alpha + self.beta - 2)
        elif self.alpha >= 1 and self.beta < 1:
            return 1.0
        elif self.alpha < 1 and self.beta >= 1:
            return 0.0
        else:
            return 0.5

    @property
    def effective_sample_size(self) -> float:
        """有效样本量 α+β — 反映证据强度"""
        return self.alpha + self.beta

    @property
    def precision(self) -> float:
        """精度 1/Var[p]"""
        var = self.variance
        return 1.0 / var if var > 0 else float('inf')

    def credible_interval(self, probability: float = 0.95) -> Tuple[float, float]:
        """
        最高后验密度区间 (HDPI) — 使用正态近似

        Args:
            probability: 置信水平 (默认 95%)

        Returns:
            (lower, upper) 区间边界
        """
        z = {
            0.90: 1.645, 0.95: 1.96, 0.99: 2.576
        }.get(probability, 1.96)

        m = self.mean
        s = self.std
        lower = max(0.0, m - z * s)
        upper = min(1.0, m + z * s)
        return (lower, upper)

    def to_dict(self) -> Dict:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "mean": self.mean,
            "std": self.std,
            "ci_95": list(self.credible_interval(0.95)),
            "n_eff": self.effective_sample_size
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'BetaDistribution':
        return cls(alpha=d["alpha"], beta=d["beta"])

    @classmethod
    def uniform(cls) -> 'BetaDistribution':
        """无信息先验 Beta(1,1) — 均匀分布"""
        return cls(alpha=1.0, beta=1.0)

    @classmethod
    def weak_informative(cls, prior_belief: float = 0.5, strength: float = 2.0) -> 'BetaDistribution':
        """
        弱信息先验 — 以 prior_belief 为中心，strength 控制先验强度

        Args:
            prior_belief: 先验期望 (0.0-1.0)
            strength: 先验强度 (等价于伪样本量)
        """
        alpha = prior_belief * strength
        beta = (1 - prior_belief) * strength
        return cls(alpha=alpha, beta=beta)

    @classmethod
    def jeffreys(cls) -> 'BetaDistribution':
        """Jeffreys 非信息先验 Beta(0.5, 0.5)"""
        return cls(alpha=0.5, beta=0.5)


@dataclass
class BayesianBelief:
    """
    单个信念的贝叶斯状态

    包含：
    - 信念ID和内容摘要
    - Beta分布（概率信念表示）
    - 证据历史记录
    - 时序信息
    - 阶段推断
    """
    belief_id: str
    content_summary: str = ""
    prior: BetaDistribution = field(default_factory=BetaDistribution.uniform)
    posterior: BetaDistribution = field(default_factory=BetaDistribution.uniform)

    # 证据计数
    positive_evidence: int = 0
    negative_evidence: int = 0

    # 时序
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    evidence_history: List[Dict] = field(default_factory=list)

    # 元信息
    category: str = "general"
    tags: List[str] = field(default_factory=list)

    def update_evidence(
        self,
        success: bool,
        weight: float = 1.0,
        source: str = "unknown",
        note: str = ""
    ):
        """
        更新证据（应用贝叶斯更新）

        Args:
            success: True=正面证据, False=负面证据
            weight: 证据权重 (0.0-1.0)
            source: 证据来源
            note: 备注
        """
        now = time.time()

        if success:
            self.positive_evidence += 1
            self.posterior.alpha += weight
        else:
            self.negative_evidence += 1
            self.posterior.beta += weight

        self.last_updated = now
        self.evidence_history.append({
            "time": now,
            "success": success,
            "weight": weight,
            "source": source,
            "note": note,
            "posterior_mean": self.posterior.mean
        })

    def get_confidence(self) -> float:
        """获取当前信念置信度（后验期望）"""
        return self.posterior.mean

    def get_uncertainty(self) -> float:
        """获取不确定性（后验标准差）"""
        return self.posterior.std

    def get_evidence_strength(self) -> float:
        """获取证据强度（有效样本量）"""
        return self.posterior.effective_sample_size

    def get_stage(self) -> str:
        """
        基于贝叶斯推断的阶段判定

        阶段逻辑：
        - cognition:  证据不足，高不确定性 (n_eff < 3)
        - confirm:    中等证据，信念形成 (3 ≤ n_eff < 10)
        - reinforce:  强证据，高置信度 (n_eff ≥ 10, mean ≥ 0.7)
        - shake:      置信度下降 (mean < 0.5, 曾有较强证据)
        - reshape:    需要重建 (mean < 0.3)
        - decay:      长期未更新但仍有中等置信度
        """
        mean = self.posterior.mean
        n_eff = self.posterior.effective_sample_size
        std = self.posterior.std

        # 重塑：置信度极低
        if mean < 0.3 and n_eff >= 5:
            return "reshape"

        # 动摇：置信度跌破 0.5 且有一定证据积累
        if mean < 0.5 and n_eff >= 3:
            return "shake"

        # 衰减：长期未更新 (30天+) 但仍有一些证据
        days_since = (time.time() - self.last_updated) / (24 * 3600)
        if days_since > 30 and n_eff >= 3:
            return "decay"

        # 认知：证据不足
        if n_eff < 3:
            return "cognition"

        # 确认：中等证据
        if n_eff < 10:
            return "confirm"

        # 强化：强证据 + 高置信度
        if mean >= 0.7:
            return "reinforce"

        return "confirm"

    def to_dict(self) -> Dict:
        return {
            "belief_id": self.belief_id,
            "content_summary": self.content_summary,
            "posterior": self.posterior.to_dict(),
            "confidence": self.get_confidence(),
            "uncertainty": self.get_uncertainty(),
            "evidence_strength": self.get_evidence_strength(),
            "stage": self.get_stage(),
            "positive_evidence": self.positive_evidence,
            "negative_evidence": self.negative_evidence,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "category": self.category,
            "tags": self.tags
        }


# ============================================================
# 似然函数
# ============================================================

class LikelihoodFunctions:
    """
    似然函数库

    提供多种似然模型：
    1. Bernoulli: P(evidence | p) = p^success * (1-p)^failure
    2. Binomial: P(k | n, p) = C(n,k) * p^k * (1-p)^(n-k)
    3. Gaussian (with known variance)
    """

    @staticmethod
    def bernoulli_log_likelihood(successes: int, failures: int, p: float) -> float:
        """
        Bernoulli 对数似然

        log P(successes, failures | p) = s*log(p) + f*log(1-p)

        Args:
            successes: 成功次数
            failures: 失败次数
            p: 成功概率

        Returns:
            对数似然值
        """
        if p <= 0 or p >= 1:
            return float('-inf')
        return successes * math.log(p) + failures * math.log(1 - p)

    @staticmethod
    def binomial_log_likelihood(k: int, n: int, p: float) -> float:
        """
        Binomial 对数似然

        Args:
            k: 成功次数
            n: 总试验数
            p: 成功概率
        """
        if p <= 0 or p >= 1 or k > n:
            return float('-inf')
        log_comb = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        return log_comb + k * math.log(p) + (n - k) * math.log(1 - p)

    @staticmethod
    def gaussian_log_likelihood(x: float, mu: float, sigma: float) -> float:
        """
        高斯对数似然

        Args:
            x: 观测值
            mu: 均值
            sigma: 标准差
        """
        return -0.5 * math.log(2 * math.pi * sigma * sigma) - \
               (x - mu) * (x - mu) / (2 * sigma * sigma)

    @staticmethod
    def weighted_likelihood(
        evidence_list: List,
        hypothesis_mean: float
    ) -> float:
        """
        加权似然函数

        每条证据有独立权重，合并为总体似然:
        P(E | H) = Σ w_i * P(e_i | H)

        Args:
            evidence_list: [{"success": bool, "weight": float}, ...] 或 EvidenceRecord 对象列表
            hypothesis_mean: 假设的概率值

        Returns:
            加权对数似然
        """
        total_log_like = 0.0
        for ev in evidence_list:
            # 兼容 dict 和 EvidenceRecord 对象
            if hasattr(ev, 'calibrated_weight'):
                # EvidenceRecord 对象
                w = ev.calibrated_weight
                success = ev.is_positive
            elif hasattr(ev, 'get'):
                # dict
                w = ev.get("weight", 1.0)
                success = ev.get("success", True)
            else:
                continue

            if success:
                total_log_like += w * math.log(max(hypothesis_mean, 1e-10))
            else:
                total_log_like += w * math.log(max(1 - hypothesis_mean, 1e-10))
        return total_log_like


# ============================================================
# 贝叶斯引擎
# ============================================================

class BayesianEngine:
    """
    贝叶斯推理引擎 - 对外唯一接口

    功能：
    1. 信念注册与管理
    2. 证据驱动的后验更新
    3. 不确定性量化
    4. 贝叶斯假设检验
    5. 多信念比较与排序
    6. 先验配置管理
    """

    def __init__(
        self,
        default_prior_type: str = "uniform",
        default_prior_strength: float = 2.0
    ):
        """
        Args:
            default_prior_type: 默认先验类型
                - "uniform": Beta(1,1) 均匀先验
                - "jeffreys": Beta(0.5,0.5) Jeffreys先验
                - "weak": 弱信息先验 Beta(strength*p, strength*(1-p))
            default_prior_strength: 弱信息先验的强度
        """
        self._beliefs: Dict[str, BayesianBelief] = {}
        self._default_prior_type = default_prior_type
        self._default_prior_strength = default_prior_strength

        # 全局统计
        self._total_evidence = 0
        self._update_count = 0

    # ---- 信念管理 ----

    def register_belief(
        self,
        belief_id: str,
        content_summary: str = "",
        prior_belief: float = None,
        prior_strength: float = None,
        category: str = "general",
        tags: List[str] = None
    ) -> BayesianBelief:
        """
        注册新信念

        Args:
            belief_id: 信念唯一标识
            content_summary: 内容摘要
            prior_belief: 先验期望 (None=使用默认先验)
            prior_strength: 先验强度
            category: 类别
            tags: 标签

        Returns:
            BayesianBelief 对象
        """
        if belief_id in self._beliefs:
            return self._beliefs[belief_id]

        # 确定先验
        if self._default_prior_type == "uniform":
            prior = BetaDistribution.uniform()
        elif self._default_prior_type == "jeffreys":
            prior = BetaDistribution.jeffreys()
        elif prior_belief is not None:
            strength = prior_strength or self._default_prior_strength
            prior = BetaDistribution.weak_informative(prior_belief, strength)
        else:
            prior = BetaDistribution.uniform()

        belief = BayesianBelief(
            belief_id=belief_id,
            content_summary=content_summary,
            prior=prior,
            posterior=BetaDistribution(alpha=prior.alpha, beta=prior.beta),
            category=category,
            tags=tags or []
        )
        self._beliefs[belief_id] = belief
        return belief

    def get_belief(self, belief_id: str) -> Optional[BayesianBelief]:
        """获取信念"""
        return self._beliefs.get(belief_id)

    def get_or_create(self, belief_id: str, **kwargs) -> BayesianBelief:
        """获取或创建信念"""
        if belief_id in self._beliefs:
            return self._beliefs[belief_id]
        return self.register_belief(belief_id, **kwargs)

    # ---- 证据更新 ----

    def observe(
        self,
        belief_id: str,
        success: bool,
        weight: float = 1.0,
        source: str = "unknown",
        note: str = ""
    ) -> BayesianBelief:
        """
        观测证据并更新后验

        应用贝叶斯定理:
        P(θ | E) ∝ P(E | θ) × P(θ)

        其中 Beta 分布是共轭先验，更新规则:
        - 正面证据: α' = α + w
        - 负面证据: β' = β + w

        Args:
            belief_id: 信念ID
            success: True=正面证据, False=负面证据
            weight: 证据权重 (0.0-1.0, >1.0表示高可靠性来源)
            source: 证据来源
            note: 备注

        Returns:
            更新后的 BayesianBelief
        """
        belief = self.get_or_create(belief_id)
        belief.update_evidence(success=success, weight=weight, source=source, note=note)
        self._total_evidence += 1
        self._update_count += 1
        return belief

    def batch_observe(
        self,
        observations: List[Dict]
    ) -> List[BayesianBelief]:
        """
        批量观测证据

        Args:
            observations: [
                {"belief_id": str, "success": bool, "weight": float, ...},
                ...
            ]

        Returns:
            更新后的信念列表
        """
        updated = []
        for obs in observations:
            belief = self.observe(
                belief_id=obs["belief_id"],
                success=obs["success"],
                weight=obs.get("weight", 1.0),
                source=obs.get("source", "batch"),
                note=obs.get("note", "")
            )
            updated.append(belief)
        return updated

    # ---- 推理查询 ----

    def query_confidence(self, belief_id: str) -> Optional[float]:
        """查询信念置信度"""
        belief = self._beliefs.get(belief_id)
        return belief.get_confidence() if belief else None

    def query_uncertainty(self, belief_id: str) -> Optional[float]:
        """查询信念不确定性"""
        belief = self._beliefs.get(belief_id)
        return belief.get_uncertainty() if belief else None

    def query_credible_interval(
        self,
        belief_id: str,
        probability: float = 0.95
    ) -> Optional[Tuple[float, float]]:
        """查询信念的置信区间"""
        belief = self._beliefs.get(belief_id)
        return belief.posterior.credible_interval(probability) if belief else None

    def compare_beliefs(
        self,
        belief_id_a: str,
        belief_id_b: str
    ) -> Optional[Dict]:
        """
        比较两个信念

        使用贝叶斯因子:
        BF = P(D | H_A) / P(D | H_B)

        以及后验优势比:
        Posterior Odds = Prior Odds × BF

        Returns:
            {
                "belief_a": mean_a,
                "belief_b": mean_b,
                "difference": mean_a - mean_b,
                "bayes_factor": BF_AB,
                "posterior_odds": posterior_odds,
                "superior": "a"/"b"/"tie"
            }
        """
        belief_a = self._beliefs.get(belief_id_a)
        belief_b = self._beliefs.get(belief_id_b)
        if not belief_a or not belief_b:
            return None

        mean_a = belief_a.posterior.mean
        mean_b = belief_b.posterior.mean

        # 贝叶斯因子（近似）: 使用后验优势比 / 先验优势比
        prior_odds_ab = (belief_a.prior.mean / max(1 - belief_a.prior.mean, 1e-10)) / \
                        (belief_b.prior.mean / max(1 - belief_b.prior.mean, 1e-10))
        posterior_odds_ab = (mean_a / max(1 - mean_a, 1e-10)) / \
                            (mean_b / max(1 - mean_b, 1e-10))
        bayes_factor = posterior_odds_ab / max(prior_odds_ab, 1e-10)

        # 判定优劣
        if abs(mean_a - mean_b) < 0.05:
            superior = "tie"
        elif mean_a > mean_b:
            superior = "a"
        else:
            superior = "b"

        return {
            "belief_a": mean_a,
            "belief_b": mean_b,
            "difference": mean_a - mean_b,
            "bayes_factor": bayes_factor,
            "posterior_odds": posterior_odds_ab,
            "superior": superior
        }

    def hypothesis_test(
        self,
        belief_id: str,
        null_value: float = 0.5,
        threshold: float = 0.05
    ) -> Dict:
        """
        贝叶斯假设检验

        H0: p = null_value
        H1: p ≠ null_value

        使用贝叶斯因子和 credible interval 做决策

        Args:
            belief_id: 信念ID
            null_value: 零假设值
            threshold: 判定阈值

        Returns:
            {
                "null_value": float,
                "posterior_mean": float,
                "credible_interval": (lower, upper),
                "reject_null": bool,
                "bayes_factor": float,
                "conclusion": str
            }
        """
        belief = self._beliefs.get(belief_id)
        if not belief:
            return {"error": f"Belief '{belief_id}' not found"}

        mean = belief.posterior.mean
        ci = belief.posterior.credible_interval(0.95)

        # 判断是否拒绝零假设（null_value 不在 95% CI 内）
        reject = not (ci[0] <= null_value <= ci[1])

        # 贝叶斯因子近似: P(D|H1)/P(D|H0)
        # 使用 Savage-Dickey density ratio
        likelihood_h1 = LikelihoodFunctions.bernoulli_log_likelihood(
            belief.positive_evidence, belief.negative_evidence, mean
        )
        likelihood_h0 = LikelihoodFunctions.bernoulli_log_likelihood(
            belief.positive_evidence, belief.negative_evidence, null_value
        )
        log_bf = likelihood_h1 - likelihood_h0
        bf = math.exp(log_bf) if log_bf > -700 else 0.0

        return {
            "null_value": null_value,
            "posterior_mean": mean,
            "credible_interval": list(ci),
            "reject_null": reject,
            "bayes_factor": bf,
            "conclusion": (
                f"拒绝零假设 (p ≠ {null_value})" if reject
                else f"未能拒绝零假设 (p = {null_value})"
            )
        }

    # ---- 批量查询 ----

    def get_top_beliefs(
        self,
        n: int = 10,
        category: str = None,
        min_evidence: float = 2.0
    ) -> List[Dict]:
        """
        获取置信度最高的信念（按后验期望排序）

        Args:
            n: 返回数量
            category: 类别过滤
            min_evidence: 最小证据量

        Returns:
            排序后的信念摘要列表
        """
        candidates = []
        for belief in self._beliefs.values():
            if category and belief.category != category:
                continue
            if belief.posterior.effective_sample_size < min_evidence:
                continue
            candidates.append(belief.to_dict())

        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        return candidates[:n]

    def get_uncertain_beliefs(self, n: int = 10) -> List[Dict]:
        """获取不确定性最高的信念（需要更多证据）"""
        candidates = [b.to_dict() for b in self._beliefs.values()]
        candidates.sort(key=lambda x: x["uncertainty"], reverse=True)
        return candidates[:n]

    def get_stage_distribution(self) -> Dict[str, int]:
        """获取信念阶段分布统计"""
        distribution = defaultdict(int)
        for belief in self._beliefs.values():
            distribution[belief.get_stage()] += 1
        return dict(distribution)

    def get_statistics(self) -> Dict:
        """获取引擎统计信息"""
        dist = self.get_stage_distribution()
        total = len(self._beliefs)

        confidences = [b.get_confidence() for b in self._beliefs.values()]
        uncertainties = [b.get_uncertainty() for b in self._beliefs.values()]

        return {
            "total_beliefs": total,
            "total_evidence": self._total_evidence,
            "update_count": self._update_count,
            "stage_distribution": dist,
            "mean_confidence": sum(confidences) / max(len(confidences), 1),
            "mean_uncertainty": sum(uncertainties) / max(len(uncertainties), 1),
            "high_confidence_count": sum(1 for c in confidences if c >= 0.8),
            "low_confidence_count": sum(1 for c in confidences if c < 0.3),
        }

    # ---- 持久化 ----

    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            "beliefs": {bid: b.to_dict() for bid, b in self._beliefs.items()},
            "total_evidence": self._total_evidence,
            "update_count": self._update_count,
            "config": {
                "default_prior_type": self._default_prior_type,
                "default_prior_strength": self._default_prior_strength
            }
        }

    def to_json(self) -> str:
        """序列化为 JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: Dict) -> 'BayesianEngine':
        """从字典恢复"""
        engine = cls(
            default_prior_type=d.get("config", {}).get("default_prior_type", "uniform"),
            default_prior_strength=d.get("config", {}).get("default_prior_strength", 2.0)
        )
        for bid, bd in d.get("beliefs", {}).items():
            belief = BayesianBelief(
                belief_id=bid,
                content_summary=bd.get("content_summary", ""),
                posterior=BetaDistribution(
                    alpha=bd["posterior"]["alpha"],
                    beta=bd["posterior"]["beta"]
                ),
                positive_evidence=bd.get("positive_evidence", 0),
                negative_evidence=bd.get("negative_evidence", 0),
                category=bd.get("category", "general"),
                tags=bd.get("tags", [])
            )
            engine._beliefs[bid] = belief
        engine._total_evidence = d.get("total_evidence", 0)
        engine._update_count = d.get("update_count", 0)
        return engine

    @classmethod
    def from_json(cls, json_str: str) -> 'BayesianEngine':
        """从 JSON 字符串恢复"""
        return cls.from_dict(json.loads(json_str))

    def reset(self):
        """重置所有信念"""
        self._beliefs.clear()
        self._total_evidence = 0
        self._update_count = 0
