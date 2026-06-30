"""Unit tests for su_memory.algebra.belief_net — Beta-Bernoulli + loopy BP."""
import math
import numpy as np
import pytest

from su_memory.algebra.belief_net import (
    BetaDistribution,
    BeliefNetwork,
    BeliefPropagator,
    ConditionalEdge,
)


class TestBetaDistribution:
    def test_mean_uniform(self):
        assert BetaDistribution.uniform().mean == 0.5

    def test_mean_general(self):
        assert math.isclose(BetaDistribution(8, 4).mean, 8 / 12)

    def test_variance_formula(self):
        b = BetaDistribution(2, 2)
        # var = 4 / (16 * 5) = 0.05
        assert math.isclose(b.variance, 0.05)

    def test_variance_positive(self):
        b = BetaDistribution(3, 5)
        assert b.variance > 0

    def test_std_is_sqrt_var(self):
        b = BetaDistribution(5, 3)
        assert math.isclose(b.std, math.sqrt(b.variance))

    def test_effective_sample_size(self):
        assert BetaDistribution(3, 7).effective_sample_size == 10

    def test_conjugate_update(self):
        b = BetaDistribution(1, 1).update(7, 3)
        assert math.isclose(b.alpha, 8) and math.isclose(b.beta, 4)
        assert math.isclose(b.mean, 8 / 12)

    def test_update_weighted(self):
        b = BetaDistribution(1, 1).update(2.5, 1.5)
        assert math.isclose(b.alpha, 3.5) and math.isclose(b.beta, 2.5)

    def test_update_rejects_negative(self):
        with pytest.raises(ValueError):
            BetaDistribution(1, 1).update(-1, 0)

    def test_credible_interval_contains_mean(self):
        b = BetaDistribution(10, 10)
        lo, hi = b.credible_interval(0.95)
        assert lo < b.mean < hi

    def test_credible_interval_clamped(self):
        b = BetaDistribution(1, 1)
        lo, hi = b.credible_interval(0.95)
        assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0

    def test_jeffreys_prior(self):
        j = BetaDistribution.jeffreys()
        assert j.alpha == 0.5 and j.beta == 0.5
        assert j.mean == 0.5

    def test_weak_informative_centred(self):
        w = BetaDistribution.weak_informative(0.7, strength=10)
        assert math.isclose(w.mean, 0.7)

    def test_log_odds(self):
        b = BetaDistribution(3, 1)
        assert math.isclose(b.log_odds(), math.log(3))

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError):
            BetaDistribution(0, 1)
        with pytest.raises(ValueError):
            BetaDistribution(1, -1)


class TestConditionalEdge:
    def test_default_uniform(self):
        e = ConditionalEdge()
        assert e.pos_given_pos.mean == 0.5
        assert e.pos_given_neg.mean == 0.5

    def test_update_learns_conditional(self):
        e = ConditionalEdge()
        for _ in range(9):
            e.update(True, True)
        for _ in range(1):
            e.update(True, False)
        # uniform prior Beta(1,1) + 9 successes + 1 failure = Beta(10,2)
        assert math.isclose(e.pos_given_pos.alpha, 10.0)
        assert math.isclose(e.pos_given_pos.beta, 2.0)
        assert math.isclose(e.pos_given_pos.mean, 10.0 / 12.0, abs_tol=1e-9)

    def test_causal_strength_zero_at_uniform(self):
        e = ConditionalEdge()
        assert e.causal_strength == 0.0

    def test_predict_child_mixes(self):
        e = ConditionalEdge()
        e.pos_given_pos = BetaDistribution(9, 1)  # P(c|p=1)=0.9
        e.pos_given_neg = BetaDistribution(1, 9)  # P(c|p=0)=0.1
        parent_belief = BetaDistribution(8, 2)  # P(p=1)=0.8
        pred = e.predict_child(parent_belief)
        expected = 0.9 * 0.8 + 0.1 * 0.2
        assert math.isclose(pred.mean, expected, abs_tol=1e-6)


class TestBeliefNetwork:
    def test_add_node_default_prior(self):
        bn = BeliefNetwork()
        bn.add_node("x")
        assert bn.nodes["x"].belief.mean == 0.5

    def test_add_edge_links_nodes(self):
        bn = BeliefNetwork()
        bn.add_edge("A", "B")
        assert "B" in bn.nodes["A"].children
        assert "A" in bn.nodes["B"].parents

    def test_observe_updates_edge_only(self):
        bn = BeliefNetwork()
        bn.add_node("A")
        bn.add_node("B")
        bn.observe("A", "B", True, True)
        assert bn.nodes["B"].belief.mean == 0.5  # prior untouched
        edge = bn.get_edge("A", "B")
        assert edge.pos_given_pos.alpha > 1.0  # learned

    def test_neighbors_union(self):
        bn = BeliefNetwork()
        bn.add_edge("A", "B")
        bn.add_edge("C", "B")
        assert bn.neighbors("B") == {"A", "C"}

    def test_is_tree_simple(self):
        bn = BeliefNetwork()
        bn.add_edge("A", "B")
        bn.add_edge("B", "C")
        assert bn.is_tree()

    def test_is_tree_false_with_cycle(self):
        bn = BeliefNetwork()
        bn.add_edge("A", "B")
        bn.add_edge("B", "C")
        bn.add_edge("C", "A")
        assert not bn.is_tree()


class TestBeliefPropagation:
    @pytest.fixture
    def rain_net(self):
        bn = BeliefNetwork()
        bn.add_node("Rain")
        bn.add_node("Wet")
        bn.add_edge("Rain", "Wet")
        for _ in range(20):
            bn.observe("Rain", "Wet", True, True)
        for _ in range(5):
            bn.observe("Rain", "Wet", True, False)
        for _ in range(20):
            bn.observe("Rain", "Wet", False, False)
        for _ in range(5):
            bn.observe("Rain", "Wet", False, True)
        return bn

    def test_positive_evidence_raises_posterior(self, rain_net):
        bp = BeliefPropagator(max_iterations=50)
        post = bp.infer(rain_net, ["Wet"], evidence={"Rain": True})
        # P(Wet|Rain=True) should be high (conditional table ~0.78).
        assert post["Wet"].mean > 0.6

    def test_negative_evidence_lowers_posterior(self, rain_net):
        bp = BeliefPropagator(max_iterations=50)
        post = bp.infer(rain_net, ["Wet"], evidence={"Rain": False})
        assert post["Wet"].mean < 0.4

    def test_posterior_ordering(self, rain_net):
        bp = BeliefPropagator(max_iterations=50)
        hi = bp.infer(rain_net, ["Wet"], evidence={"Rain": True})["Wet"].mean
        lo = bp.infer(rain_net, ["Wet"], evidence={"Rain": False})["Wet"].mean
        assert hi > lo

    def test_evidence_clamped(self, rain_net):
        bp = BeliefPropagator(max_iterations=20)
        post = bp.infer(rain_net, ["Rain"], evidence={"Rain": True})
        assert post["Rain"].mean > 0.9  # clamped to evidence
