"""Unit tests for su_memory.algebra.causal_graph — CausalDAG + BFS propagation."""
import numpy as np
import pytest

from su_memory.algebra.causal_graph import CausalDAG


@pytest.fixture
def chain():
    g = CausalDAG()
    for p, c in [("A", "B"), ("B", "C"), ("C", "D")]:
        g.add_edge(p, c, weight=0.5)
    return g


class TestConstruction:
    def test_add_node_idempotent(self):
        g = CausalDAG()
        g.add_node("x")
        g.add_node("x")
        assert "x" in g.nodes

    def test_add_edge_weight_range(self):
        g = CausalDAG()
        # weight must be strictly positive; amplification (>1) is allowed
        with pytest.raises(ValueError):
            g.add_edge("a", "b", weight=0)
        with pytest.raises(ValueError):
            g.add_edge("a", "b", weight=-0.5)
        # amplification edge is valid
        g.add_edge("a", "b", weight=1.5)
        assert g.edge_weight("a", "b") == 1.5

    def test_remove_node_cleans_edges(self, chain):
        chain.remove_node("B")
        assert "B" not in chain.nodes
        assert chain.children("A") == []
        assert chain.parents("C") == []


class TestQueries:
    def test_children(self, chain):
        assert chain.children("A") == ["B"]
        assert chain.children("D") == []

    def test_parents(self, chain):
        assert chain.parents("B") == ["A"]
        assert chain.parents("A") == []

    def test_edge_weight(self, chain):
        assert chain.edge_weight("A", "B") == 0.5
        assert chain.edge_weight("B", "A") == 0.0

    def test_reach(self, chain):
        assert chain.reach("A") == {"B", "C", "D"}
        assert chain.reach("D") == set()


class TestTopology:
    def test_is_dag_for_chain(self, chain):
        assert chain.is_dag()

    def test_topological_order(self, chain):
        order = chain.topological_order()
        assert order.index("A") < order.index("B") < order.index("C") < order.index("D")

    def test_cycle_not_dag(self):
        g = CausalDAG()
        g.add_edge("X", "Y", 0.5)
        g.add_edge("Y", "X", 0.5)
        assert not g.is_dag()
        with pytest.raises(ValueError):
            g.topological_order()


class TestPropagation:
    def test_geometric_attenuation(self, chain):
        eff = chain.propagate("A", delta=1.0)
        assert np.isclose(eff["A"], 1.0)
        assert np.isclose(eff["B"], 0.5)
        assert np.isclose(eff["C"], 0.25)
        assert np.isclose(eff["D"], 0.125)

    def test_unreachable_absent(self, chain):
        eff = chain.propagate("D")
        assert eff == {"D": 1.0}  # no children

    def test_missing_source(self, chain):
        assert chain.propagate("Z") == {}

    def test_negative_delta_rejected(self, chain):
        with pytest.raises(ValueError):
            chain.propagate("A", delta=-1.0)

    def test_multi_source_summation(self, chain):
        eff = chain.propagate_multi({"A": 1.0, "C": 0.5})
        assert np.isclose(eff["C"], 0.75)  # 0.25 from A + 0.5 own
        assert np.isclose(eff["D"], 0.375)  # 0.75 * 0.5

    def test_each_node_visited_once(self):
        # Diamond: A->B, A->C, B->D, C->D. D visited once (first arrival).
        g = CausalDAG()
        g.add_edge("A", "B", 1.0)
        g.add_edge("A", "C", 1.0)
        g.add_edge("B", "D", 0.5)
        g.add_edge("C", "D", 0.9)
        eff = g.propagate("A", 1.0)
        # D's effect is determined by whichever parent is dequeued first.
        assert "D" in eff
        assert eff["D"] in (0.5, 0.9)


class TestMatrixView:
    def test_adjacency_matrix(self, chain):
        W, order = chain.adjacency_matrix()
        assert W.shape == (4, 4)
        assert order == ["A", "B", "C", "D"]
        assert W[0, 1] == 0.5

    def test_propagation_vector(self, chain):
        v, order = chain.propagation_vector("A")
        assert v.shape == (4,)
        assert np.allclose(v, [1.0, 0.5, 0.25, 0.125])
