"""
L2-L4: Knowledge Distillation, Memory Routing, Self-Reflection tests
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestKnowledgeDistillation:
    """Layer 2: distill patterns from grouped memories."""

    @pytest.mark.slow
    def test_distill_patterns_by_energy(self):
        """Grouped memories should yield common patterns."""
        from su_memory.sdk.lite_pro import SuMemoryLitePro

        pro = SuMemoryLitePro(
            enable_vector=False, enable_graph=False,
            enable_temporal=False, enable_session=False,
            enable_prediction=False, enable_explainability=False,
            storage_path="/tmp/l2_test_distill"
        )

        # Use Chinese keywords, 2+ per cluster
        pro.add("春天树木生长绿色东方")
        pro.add("生长发展东方绿色森林")
        pro.add("夏季炎热红色南方高温")
        pro.add("热情高温红色夏天活力")
        pro.add("稳定黄色中央土地基础")
        pro.add("消化四季土地中央稳定")

        assert hasattr(pro, 'distill_patterns')
        report = pro.distill_patterns()

        assert "patterns" in report
        assert "cluster_count" in report
        assert report["total_memories"] >= 6  # may include persisted data

        pro.clear()

    @pytest.mark.slow
    def test_extract_rules_from_clusters(self):
        """Rule extraction should produce actionable insights."""
        from su_memory.sdk.lite_pro import SuMemoryLitePro

        pro = SuMemoryLitePro(
            enable_vector=False, enable_graph=False,
            enable_temporal=False, enable_session=False,
            enable_prediction=False, enable_explainability=False,
            storage_path="/tmp/l2_test_rules"
        )

        # Add multiple wood-related memories to form a pattern
        for i in range(5):
            pro.add(f"春天生长东方绿色树木{i}")
        for i in range(3):
            pro.add(f"夏季炎热南方高温热情{i}")

        assert hasattr(pro, 'extract_rules')
        rules = pro.extract_rules(min_cluster_size=2)

        assert isinstance(rules, list)
        assert len(rules) > 0, "Should extract at least one rule"
        assert "energy" in rules[0] or "pattern" in rules[0]

        pro.clear()


class TestMemoryRouting:
    """Layer 3: intelligent memory routing."""

    @pytest.mark.slow
    def test_route_memory_suggests_destination(self):
        """New memory routing should suggest energy cluster."""
        from su_memory.sdk.lite_pro import SuMemoryLitePro

        pro = SuMemoryLitePro(
            enable_vector=False, enable_graph=False,
            enable_temporal=False, enable_session=False,
            enable_prediction=False, enable_explainability=False,
            storage_path="/tmp/l3_test_route"
        )

        # Build an existing memory ecosystem with Chinese keywords
        pro.add("春天东方绿色生长树木森林")

        assert hasattr(pro, 'route_memory')

        # New memory with Chinese wood keywords
        route = pro.route_memory("东方绿色树木种植春天")
        assert "energy" in route
        assert "affinity_score" in route

        pro.clear()

    @pytest.mark.slow
    def test_memory_importance_scores(self):
        """Memories should have importance scores based on usage."""
        from su_memory.sdk.lite_pro import SuMemoryLitePro

        pro = SuMemoryLitePro(
            enable_vector=False, enable_graph=False,
            enable_temporal=False, enable_session=False,
            enable_prediction=False, enable_explainability=False,
            storage_path="/tmp/l3_test_importance"
        )

        mid = pro.add("Critical project deadline next week")
        # Query it multiple times to increase importance
        for _ in range(5):
            pro.query("deadline project", top_k=3)

        assert hasattr(pro, 'get_importance_scores')
        scores = pro.get_importance_scores()

        assert mid in scores
        assert scores[mid] > 0, f"Queried memory should have importance > 0"

        pro.clear()


class TestSelfReflection:
    """Layer 4: periodic self-reflection and optimization."""

    @pytest.mark.slow
    def test_reflect_and_optimize(self):
        """Reflection agent should audit memory quality."""
        from su_memory.sdk.lite_pro import SuMemoryLitePro

        pro = SuMemoryLitePro(
            enable_vector=False, enable_graph=False,
            enable_temporal=False, enable_session=False,
            enable_prediction=False, enable_explainability=False,
            storage_path="/tmp/l4_test_reflect"
        )

        # Add varied memories with matching keywords
        pro.add("春天生长绿色东方")
        pro.add("夏天炎热红色高温")
        pro.add("稳定土地黄色中央")
        pro.add("秋天白色西方金属")

        assert hasattr(pro, 'reflect_and_optimize')
        report = pro.reflect_and_optimize()

        assert "health_score" in report
        assert "suggestions" in report
        assert "memory_count" in report
        assert report["memory_count"] == 4

        pro.clear()

    @pytest.mark.slow
    def test_evolution_pipeline_runs(self):
        """Full evolution pipeline: distill → route → reflect."""
        from su_memory.sdk.lite_pro import SuMemoryLitePro

        pro = SuMemoryLitePro(
            enable_vector=False, enable_graph=False,
            enable_temporal=False, enable_session=False,
            enable_prediction=False, enable_explainability=False,
            storage_path="/tmp/l4_test_evolve"
        )

        # Populate with diverse memories using Chinese keywords
        contents = [
            "春天东方绿色树木生长",
            "夏天南方红色高温热情",
            "中央黄色稳定土地基础",
            "秋天西方白色收敛金属",
            "冬天北方蓝色流动智慧",
        ]
        for c in contents:
            pro.add(c)

        assert hasattr(pro, 'evolution_pipeline')

        # Run the full pipeline
        result = pro.evolution_pipeline()

        assert result["success"], f"Pipeline failed: {result}"
        assert "distilled_patterns" in result
        assert "routing_suggestions" in result
        assert "reflection_report" in result

        pro.clear()
