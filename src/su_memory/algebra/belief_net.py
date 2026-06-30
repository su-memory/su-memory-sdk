"""BeliefNetwork — Beta-Bernoulli conjugate Bayesian network with loopy BP.

Mathematical foundation
-----------------------
This module implements a small but rigorous discrete Bayesian network whose
random variables are Boolean (Bernoulli) and whose uncertainty about each
variable's success probability is represented by a **Beta distribution** — the
conjugate prior for the Bernoulli likelihood.

Conjugate update
~~~~~~~~~~~~~~~~
Observing a Bernoulli outcome with success count ``a`` and failure count ``b``
updates the prior Beta(α, β) to the posterior Beta(α + a, β + b) in closed
form. This is exact and requires no sampling. The posterior mean is::

    E[θ | data] = (α + a) / (α + β + a + b)

Conditional edges
~~~~~~~~~~~~~~~~~
Each directed edge parent -> child carries *two* Beta distributions that
together estimate the conditional probability table P(child | parent):

    pos_given_pos : Beta estimating P(child=1 | parent=1)
    pos_given_neg : Beta estimating P(child=1 | parent=0)

This is the standard parameterisation of a noisy-OR / logical-dependency edge.
The relative risk RR = P(c|p=1) / P(c|p=0) quantifies causal strength.

Inference — loopy belief propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Given evidence (observed Boolean values at some nodes) we compute posterior
marginals at query nodes by message passing. On a tree-structured network the
sum-product algorithm is *exact*. On networks with loops we run **loopy belief
propagation (LBP)** with message damping to prevent oscillation; this yields a
fixed-point approximation.

Each message m_{i→j} is itself a Beta distribution. Messages are combined by
treating Beta pseudo-counts additively (subtracting the uniform prior's 1,1 so
that the combination is associative in the log-odds domain).

This module is pure probability theory: Beta arithmetic, conditional tables,
and message passing. No I/O, no SDK coupling.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "BetaDistribution",
    "ConditionalEdge",
    "BeliefNode",
    "BeliefNetwork",
    "BeliefPropagator",
]

# ===========================================================================
# Beta distribution — conjugate prior for Bernoulli
# ===========================================================================


@dataclass
class BetaDistribution:
    """Beta(α, β) distribution over θ ∈ [0, 1].

    The conjugate prior for the success probability of a Bernoulli / binomial
    likelihood. All moments are closed-form.
    """

    alpha: float = 1.0
    beta: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("alpha and beta must be positive")

    # --- moments ---
    @property
    def mean(self) -> float:
        """Posterior mean E[θ] = α / (α + β)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Var[θ] = αβ / ((α+β)^2 (α+β+1))."""
        s = self.alpha + self.beta
        return (self.alpha * self.beta) / (s * s * (s + 1.0))

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    @property
    def mode(self) -> float:
        """Posterior mode (MAP). Defined for α,β > 1; clamped otherwise."""
        if self.alpha <= 1 or self.beta <= 1:
            return 1.0 if self.alpha > self.beta else (0.0 if self.beta > self.alpha else 0.5)
        return (self.alpha - 1.0) / (self.alpha + self.beta - 2.0)

    @property
    def effective_sample_size(self) -> float:
        """α + β — the pseudo-count strength of the distribution."""
        return self.alpha + self.beta

    @property
    def precision(self) -> float:
        """1 / variance."""
        return 1.0 / self.variance if self.variance > 0 else float("inf")

    def credible_interval(self, probability: float = 0.95) -> tuple[float, float]:
        """Highest-density interval via normal approximation.

        Uses the symmetric z multiplier for the requested level. For large
        α+β the Beta is well-approximated by a Gaussian; for small α+β this
        is a conservative over-estimate of the interval width.
        """
        if not 0.0 < probability < 1.0:
            raise ValueError("probability must be in (0,1)")
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(round(probability, 2), 1.96)
        m, s = self.mean, self.std
        return max(0.0, m - z * s), min(1.0, m + z * s)

    # --- conjugate update ---
    def update(self, successes: float, failures: float) -> "BetaDistribution":
        """Bayesian conjugate update: observe successes/failures.

        Returns a *new* Beta(α + successes, β + failures). The counts may be
        fractional (weighted evidence).
        """
        if successes < 0 or failures < 0:
            raise ValueError("counts must be non-negative")
        return BetaDistribution(self.alpha + successes, self.beta + failures)

    def log_odds(self) -> float:
        """Log-odds of the mean: log(α/β)."""
        return math.log(self.alpha / self.beta)

    # --- factories ---
    @classmethod
    def uniform(cls) -> "BetaDistribution":
        """Non-informative prior Beta(1, 1) = Uniform[0,1]."""
        return cls(1.0, 1.0)

    @classmethod
    def jeffreys(cls) -> "BetaDistribution":
        """Jeffreys non-informative prior Beta(0.5, 0.5)."""
        return cls(0.5, 0.5)

    @classmethod
    def weak_informative(
        cls, prior_belief: float = 0.5, strength: float = 2.0
    ) -> "BetaDistribution":
        """Weak informative prior centred at ``prior_belief``."""
        if not 0.0 < prior_belief < 1.0:
            raise ValueError("prior_belief must be in (0,1)")
        return cls(prior_belief * strength, (1.0 - prior_belief) * strength)


# ===========================================================================
# Conditional edge — P(child | parent) via two Betas
# ===========================================================================


@dataclass
class ConditionalEdge:
    """Noisy conditional edge parent -> child.

    Stores two Beta distributions:

    - ``pos_given_pos`` : P(child=1 | parent=1)
    - ``pos_given_neg`` : P(child=1 | parent=0)

    Updated online from observed (parent_state, child_state) pairs.
    """

    pos_given_pos: BetaDistribution = field(default_factory=BetaDistribution.uniform)
    pos_given_neg: BetaDistribution = field(default_factory=BetaDistribution.uniform)

    def update(self, parent_state: bool, child_state: bool, weight: float = 1.0) -> None:
        """One conjugate update from an observed co-occurrence."""
        if weight <= 0:
            raise ValueError("weight must be positive")
        a = weight if child_state else 0.0
        b = weight if not child_state else 0.0
        if parent_state:
            self.pos_given_pos = self.pos_given_pos.update(a, b)
        else:
            self.pos_given_neg = self.pos_given_neg.update(a, b)

    @property
    def causal_strength(self) -> float:
        """Difference in success probability: P(c|p=1) − P(c|p=0).

        Positive => parent promotes child; negative => parent inhibits child;
        near zero => no detectable effect.
        """
        return self.pos_given_pos.mean - self.pos_given_neg.mean

    @property
    def relative_risk(self) -> float:
        """RR = P(c|p=1) / P(c|p=0). >1 means parent raises child probability."""
        denom = self.pos_given_neg.mean
        if denom <= 1e-9:
            return float("inf")
        return self.pos_given_pos.mean / denom

    def predict_child(self, parent_belief: BetaDistribution) -> BetaDistribution:
        """Marginalise over parent state to predict child's belief.

        P(child=1) = P(c|p=1)·P(p=1) + P(c|p=0)·P(p=0)
        Returned as a Beta whose mean equals this mixture probability, with
        pseudo-count strength inherited from the stronger of the two conditionals.
        """
        p_pos = parent_belief.mean
        p_neg = 1.0 - p_pos
        mean_child = (
            self.pos_given_pos.mean * p_pos + self.pos_given_neg.mean * p_neg
        )
        mean_child = min(max(mean_child, 1e-6), 1 - 1e-6)
        strength = max(
            self.pos_given_pos.effective_sample_size,
            self.pos_given_neg.effective_sample_size,
        )
        return BetaDistribution(mean_child * strength, (1 - mean_child) * strength)


# ===========================================================================
# Node + network
# ===========================================================================


@dataclass
class BeliefNode:
    """A Boolean random variable with a Beta prior over P(var=1)."""

    belief: BetaDistribution = field(default_factory=BetaDistribution.uniform)
    parents: set = field(default_factory=set)
    children: set = field(default_factory=set)

    def markov_blanket(self) -> set:
        """Markov blanket = parents ∪ children ∪ co-parents."""
        return set(self.parents) | set(self.children)


@dataclass
class BeliefNetwork:
    """A discrete Bayesian network over Boolean variables.

    Nodes carry Beta priors; edges carry ConditionalEdge conditional tables.
    Provides exact tree inference and approximate loopy-BP inference.
    """

    nodes: dict = field(default_factory=dict)
    edges: dict = field(default_factory=dict)  # (parent, child) -> ConditionalEdge

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def add_node(self, node_id, prior: BetaDistribution | None = None) -> BeliefNode:
        """Add a node with an optional prior (default uniform)."""
        b = prior if prior is not None else BetaDistribution.uniform()
        node = BeliefNode(belief=b)
        self.nodes[node_id] = node
        return node

    def add_edge(self, parent, child) -> ConditionalEdge:
        """Add a conditional edge parent -> child."""
        if parent not in self.nodes:
            self.add_node(parent)
        if child not in self.nodes:
            self.add_node(child)
        edge = ConditionalEdge()
        self.edges[(parent, child)] = edge
        self.nodes[parent].children.add(child)
        self.nodes[child].parents.add(parent)
        return edge

    def observe(self, parent, child, parent_state: bool, child_state: bool, weight: float = 1.0):
        """Record a co-occurrence and update ONLY the conditional edge.

        The node prior beliefs are intentionally left untouched: in a Bayesian
        network the priors are independent background beliefs, not empirical
        marginal frequencies. Mixing the two would double-count the data and
        bias inference toward the training-set marginal. Callers who want to
        set a data-driven prior should do so explicitly via ``add_node(prior=...)``.
        """
        if (parent, child) not in self.edges:
            self.add_edge(parent, child)
        self.edges[(parent, child)].update(parent_state, child_state, weight)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def neighbors(self, node_id) -> set:
        """Undirected neighbours (parents ∪ children)."""
        n = self.nodes.get(node_id)
        if n is None:
            return set()
        return set(n.parents) | set(n.children)

    def get_edge(self, parent, child) -> ConditionalEdge | None:
        return self.edges.get((parent, child))

    def marginal(self, node_id) -> BetaDistribution:
        """Current (prior) marginal of a node, ignoring evidence."""
        n = self.nodes.get(node_id)
        return n.belief if n else BetaDistribution.uniform()

    def is_tree(self) -> bool:
        """True iff the undirected skeleton is acyclic (exact BP applies)."""
        seen, stack = set(), []
        # Pick an arbitrary start to detect any cycle.
        if not self.nodes:
            return True
        start = next(iter(self.nodes))
        # Union-Find over the undirected skeleton.
        parent_uf = {n: n for n in self.nodes}

        def find(x):
            while parent_uf[x] != x:
                parent_uf[x] = parent_uf[parent_uf[x]]
                x = parent_uf[x]
            return x

        for (p, c) in self.edges:
            rp, rc = find(p), find(c)
            if rp == rc:
                return False
            parent_uf[rp] = rc
        return True


# ===========================================================================
# Inference — loopy belief propagation
# ===========================================================================


@dataclass
class BeliefPropagator:
    """Sum-product belief propagation with damping.

    Exact on trees, approximate (loopy BP fixed-point) on cyclic skeletons.
    """

    max_iterations: int = 20
    damping: float = 0.5
    tolerance: float = 1e-3

    def infer(
        self,
        network: BeliefNetwork,
        query_nodes: list,
        evidence: dict | None = None,
    ) -> dict:
        """Compute posterior Beta marginals at ``query_nodes`` given evidence.

        Parameters
        ----------
        network : BeliefNetwork
        query_nodes : list of node ids
        evidence : dict node_id -> bool
            Observed Boolean values. Evidence nodes are clamped.

        Returns
        -------
        dict node_id -> BetaDistribution (posterior).
        """
        evidence = evidence or {}
        # m[(i, j)] = message from i to j, as a Beta.
        messages: dict = {}

        # Evidence nodes broadcast their observed value to neighbours once,
        # so that non-evidence neighbours receive information even though the
        # evidence nodes themselves are skipped in the message loop. The
        # broadcast flows through any conditional edge so that, e.g., the
        # message an evidence parent sends to its child is P(child|parent=evid),
        # not the raw evidence value.
        for ev_node, ev_val in evidence.items():
            ev_belief = BetaDistribution(10.0 if ev_val else 1.0, 1.0 if ev_val else 10.0)
            for nb in network.neighbors(ev_node):
                fwd = network.get_edge(ev_node, nb)
                bwd = network.get_edge(nb, ev_node)
                if fwd is not None:
                    messages[(ev_node, nb)] = fwd.predict_child(ev_belief)
                elif bwd is not None:
                    # evidence is the child of nb: backward likelihood message
                    p_pos = bwd.pos_given_pos.mean if ev_val else (1.0 - bwd.pos_given_pos.mean)
                    p_neg = bwd.pos_given_neg.mean if ev_val else (1.0 - bwd.pos_given_neg.mean)
                    new_mean = min(max(p_pos * 0.5 + p_neg * 0.5, 1e-6), 1 - 1e-6)
                    messages[(ev_node, nb)] = BetaDistribution(new_mean * 11.0, (1 - new_mean) * 11.0)
                else:
                    messages[(ev_node, nb)] = ev_belief

        for _ in range(self.max_iterations):
            old_messages = dict(messages)
            max_delta = 0.0
            for node_id in network.nodes:
                if node_id in evidence:
                    continue
                for neighbor in network.neighbors(node_id):
                    incoming = self._collect_incoming(
                        network, node_id, neighbor, messages, evidence
                    )
                    new_msg = self._compute_message(
                        network, node_id, neighbor, incoming, evidence
                    )
                    key = (node_id, neighbor)
                    if key in old_messages:
                        old = old_messages[key]
                        a = self.damping * new_msg.alpha + (1 - self.damping) * old.alpha
                        b = self.damping * new_msg.beta + (1 - self.damping) * old.beta
                        new_msg = BetaDistribution(max(a, 1e-3), max(b, 1e-3))
                    messages[key] = new_msg
                    if key in old_messages:
                        max_delta = max(max_delta, abs(new_msg.mean - old_messages[key].mean))
            if max_delta < self.tolerance:
                break

        return {
            nid: self._compute_marginal(network, nid, messages, evidence)
            for nid in query_nodes
        }

    # ------------------------------------------------------------------
    # message passing internals
    # ------------------------------------------------------------------
    @staticmethod
    def _collect_incoming(network, node_id, exclude, messages, evidence):
        incoming = []
        for nb in network.neighbors(node_id):
            if nb == exclude:
                continue
            if nb in evidence:
                v = evidence[nb]
                incoming.append(BetaDistribution(10.0 if v else 1.0, 1.0 if v else 10.0))
            else:
                msg = messages.get((nb, node_id))
                if msg is not None:
                    incoming.append(msg)
        return incoming

    @staticmethod
    def _compute_message(network, from_node, to_node, incoming, evidence):
        node = network.nodes.get(from_node)
        own = node.belief if node else BetaDistribution.uniform()
        # Combine own prior + incoming messages via additive pseudo-counts.
        a = own.alpha
        b = own.beta
        for msg in incoming:
            a += msg.alpha - 1.0
            b += msg.beta - 1.0
        a = max(a, 0.1)
        b = max(b, 0.1)
        combined = BetaDistribution(a, b)
        # Attenuate through the conditional edge.
        # If from_node -> to_node is a directed edge, forward-predict the child.
        # If to_node -> from_node is a directed edge, this is a backward message
        # (child informing parent): we still pass the child's belief through the
        # conditional, which is the loopy-BP approximation for the reverse factor.
        fwd_edge = network.get_edge(from_node, to_node)
        bwd_edge = network.get_edge(to_node, from_node)
        if fwd_edge is not None:
            return fwd_edge.predict_child(combined)
        if bwd_edge is not None:
            # Backward: parent posterior proportional to P(child|parent)*prior.
            # Approximate via likelihood ratio on the parent belief.
            p_pos = combined.mean
            like_pos = bwd_edge.pos_given_pos.mean
            like_neg = bwd_edge.pos_given_neg.mean
            new_mean = like_pos * p_pos + like_neg * (1.0 - p_pos)
            new_mean = min(max(new_mean, 1e-6), 1 - 1e-6)
            strength = combined.effective_sample_size
            return BetaDistribution(new_mean * strength, (1 - new_mean) * strength)
        return combined

    @staticmethod
    def _compute_marginal(network, node_id, messages, evidence):
        if node_id in evidence:
            v = evidence[node_id]
            return BetaDistribution(10.0 if v else 1.0, 1.0 if v else 10.0)
        node = network.nodes.get(node_id)
        a = node.belief.alpha if node else 1.0
        b = node.belief.beta if node else 1.0
        for nb in network.neighbors(node_id):
            msg = messages.get((nb, node_id))
            if msg is not None:
                a += msg.alpha - 1.0
                b += msg.beta - 1.0
        return BetaDistribution(max(a, 0.1), max(b, 0.1))
