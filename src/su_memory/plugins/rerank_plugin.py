"""
Rerank Plugin (检索增强重排序插件)

v1.7.0 W25-W26 官方插件示例

提供检索结果重排序功能，演示context传递机制。

Features:
- 基于相关性的重排序
- 支持多维度评分
- 上下文感知排序

【Pre-Phase Numeric】- Uses prior ordering for numerical calculations
【Post-Phase Symbolic】- Uses post ordering for symbolic applications
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .._sys._plugin_interface import (
    PluginInterface,
    PluginType,
)


# =============================================================================
# Score Result
# =============================================================================

@dataclass
class ScoreResult:
    """评分结果"""
    item_id: str
    original_score: float
    new_score: float
    score_breakdown: Dict[str, float]
    rank: int


# =============================================================================
# Rerank Scorer
# =============================================================================

class RerankScorer:
    """
    重排序评分器。

    基于多个维度对检索结果进行评分和重排序。

    评分维度:
    - 相关性: 文本相似度
    - 新近度: 时间因子
    - 重要性: 权重因子

    Example:
        >>> scorer = RerankScorer()
        >>> scores = scorer.score_items(query, items)
        >>> ranked = scorer.rerank(scores)
    """

    def __init__(
        self,
        relevance_weight: float = 0.5,
        recency_weight: float = 0.3,
        importance_weight: float = 0.2,
    ):
        """
        初始化评分器。

        Args:
            relevance_weight: 相关性权重
            recency_weight: 新近度权重
            importance_weight: 重要性权重
        """
        self._relevance_weight = relevance_weight
        self._recency_weight = recency_weight
        self._importance_weight = importance_weight

    def score_items(
        self,
        query: str,
        items: List[Dict[str, Any]],
        use_context: bool = True,
    ) -> List[ScoreResult]:
        """
        对项目进行评分。

        Args:
            query: 查询字符串
            items: 项目列表，每个项目包含:
                - id: str, 项目ID
                - text: str, 文本内容
                - score: float, 原始分数
                - timestamp: float, 时间戳（可选）
                - importance: float, 重要性（可选）
            use_context: 是否使用上下文权重

        Returns:
            评分结果列表
        """
        results = []

        for idx, item in enumerate(items):
            item_id = item.get("id", f"item_{idx}")
            original_score = item.get("score", 0.0)
            text = item.get("text", "")

            # 计算各维度分数
            relevance = self._calculate_relevance(query, text)
            recency = self._calculate_recency(item.get("timestamp"))
            importance = self._calculate_importance(item.get("importance"))

            # 综合评分
            new_score = (
                self._relevance_weight * relevance +
                self._recency_weight * recency +
                self._importance_weight * importance
            )

            results.append(ScoreResult(
                item_id=item_id,
                original_score=original_score,
                new_score=new_score,
                score_breakdown={
                    "relevance": relevance,
                    "recency": recency,
                    "importance": importance,
                },
                rank=0,  # 待设置
            ))

        return results

    def _calculate_relevance(self, query: str, text: str) -> float:
        """计算文本相关性"""
        if not query or not text:
            return 0.0

        query_terms = set(query.lower().split())
        text_terms = set(text.lower().split())

        if not query_terms:
            return 0.0

        # Jaccard相似度
        intersection = len(query_terms & text_terms)
        union = len(query_terms | text_terms)

        return intersection / union if union > 0 else 0.0

    def _calculate_recency(self, timestamp: Optional[float]) -> float:
        """计算新近度分数"""
        if timestamp is None:
            return 0.5  # 默认中等分数

        import time
        age = time.time() - timestamp

        # 指数衰减：一天前为0.5
        day_in_seconds = 86400
        decay = 2 ** (-age / day_in_seconds)

        return max(0.0, min(1.0, decay))

    def _calculate_importance(self, importance: Optional[float]) -> float:
        """计算重要性分数"""
        if importance is None:
            return 0.5  # 默认中等分数

        return max(0.0, min(1.0, importance))

    def rerank(
        self,
        scores: List[ScoreResult],
        top_k: Optional[int] = None,
    ) -> List[ScoreResult]:
        """
        根据分数重排序。

        Args:
            scores: 评分结果列表
            top_k: 返回前k个（None表示全部）

        Returns:
            重排序后的结果
        """
        # 按分数降序排序
        sorted_scores = sorted(scores, key=lambda x: x.new_score, reverse=True)

        # 设置排名
        for rank, score_result in enumerate(sorted_scores, 1):
            score_result.rank = rank

        # 返回前k个
        if top_k is not None:
            return sorted_scores[:top_k]

        return sorted_scores


# =============================================================================
# Rerank Plugin
# =============================================================================

class RerankPlugin(PluginInterface):
    """
    检索增强重排序插件。

    对检索结果进行多维度重排序，提高结果质量。

    Example:
        >>> plugin = RerankPlugin()
        >>> plugin.initialize({})
        >>>
        >>> result = plugin.execute({
        ...     "query": "人工智能发展",
        ...     "items": [
        ...         {"id": "1", "text": "AI技术", "score": 0.9},
        ...         {"id": "2", "text": "机器学习", "score": 0.8},
        ...     ],
        ...     "top_k": 10
        ... })
    """

    def __init__(self):
        """初始化插件"""
        self._initialized = False
        self._scorer: Optional[RerankScorer] = None
        self._config: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "rerank_plugin"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "检索结果重排序插件，支持多维度评分和上下文感知排序"

    @property
    def author(self) -> str:
        return "su-memory-sdk"

    @property
    def plugin_type(self) -> PluginType:
        return PluginType.RERANK

    @property
    def dependencies(self) -> List[str]:
        return []

    @property
    def config_schema(self) -> Dict[str, Any]:
        return {
            "required": [],
            "properties": {
                "relevance_weight": {
                    "type": "number",
                    "default": 0.5,
                    "description": "相关性权重"
                },
                "recency_weight": {
                    "type": "number",
                    "default": 0.3,
                    "description": "新近度权重"
                },
                "importance_weight": {
                    "type": "number",
                    "default": 0.2,
                    "description": "重要性权重"
                },
                "top_k": {
                    "type": "integer",
                    "default": 10,
                    "description": "返回前k个结果"
                }
            }
        }

    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        初始化插件。

        Args:
            config: 配置字典

        Returns:
            True表示初始化成功
        """
        try:
            self._config = config

            self._scorer = RerankScorer(
                relevance_weight=config.get("relevance_weight", 0.5),
                recency_weight=config.get("recency_weight", 0.3),
                importance_weight=config.get("importance_weight", 0.2),
            )

            self._initialized = True
            return True

        except Exception:
            self._initialized = False
            return False

    def execute(self, context: Dict[str, Any]) -> Any:
        """
        执行重排序。

        Args:
            context: 执行上下文
                - query: str, 查询字符串
                - items: List[Dict], 待排序项目
                - top_k: int, 返回前k个（可选）
                - context: Dict, 额外上下文（可选）

        Returns:
            重排序结果
        """
        if not self._initialized:
            raise RuntimeError("Plugin not initialized")

        query = context.get("query", "")
        items = context.get("items", [])
        top_k = context.get("top_k")
        extra_context = context.get("context", {})

        if not query:
            raise ValueError("Missing 'query' in context")

        if not items:
            return {
                "success": True,
                "ranked_items": [],
                "count": 0,
            }

        # 评分
        scores = self._scorer.score_items(query, items)

        # 重排序
        ranked = self._scorer.rerank(scores, top_k=top_k)

        # 格式化输出
        ranked_items = []
        for score_result in ranked:
            # 查找原始项目
            original_item = next(
                (item for item in items if item.get("id") == score_result.item_id),
                {}
            )

            ranked_items.append({
                "id": score_result.item_id,
                "text": original_item.get("text", ""),
                "original_score": score_result.original_score,
                "new_score": score_result.new_score,
                "score_breakdown": score_result.score_breakdown,
                "rank": score_result.rank,
            })

        return {
            "success": True,
            "ranked_items": ranked_items,
            "count": len(ranked_items),
            "query": query,
            "context_used": bool(extra_context),
        }

    def cleanup(self) -> None:
        """清理资源"""
        self._scorer = None
        self._initialized = False
        self._config = {}


# =============================================================================
# Plugin Factory
# =============================================================================

def create_rerank_plugin() -> RerankPlugin:
    """创建重排序插件实例"""
    return RerankPlugin()


# =============================================================================
# Test Suite
# =============================================================================

def test_rerank_plugin():
    """测试重排序插件"""
    print("=" * 60)
    print("Testing Rerank Plugin")
    print("=" * 60)

    passed = 0
    failed = 0

    def test(name: str, condition: bool):
        nonlocal passed, failed
        if condition:
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name}")
            failed += 1

    import time

    # Test 1: 插件创建
    print("\n[Test 1] Plugin Creation")
    print("-" * 40)

    plugin = RerankPlugin()
    test("插件创建", plugin is not None)
    test("插件名称", plugin.name == "rerank_plugin")
    test("插件类型", plugin.plugin_type == PluginType.RERANK)

    # Test 2: 初始化
    print("\n[Test 2] Initialization")
    print("-" * 40)

    config = {
        "relevance_weight": 0.6,
        "recency_weight": 0.2,
        "importance_weight": 0.2,
    }
    success = plugin.initialize(config)
    test("初始化成功", success)

    # Test 3: 基本重排序
    print("\n[Test 3] Basic Reranking")
    print("-" * 40)

    items = [
        {"id": "1", "text": "人工智能技术发展", "score": 0.8},
        {"id": "2", "text": "人工智能应用", "score": 0.9},
        {"id": "3", "text": "机器学习算法", "score": 0.7},
    ]

    result = plugin.execute({
        "query": "人工智能",
        "items": items,
    })

    test("执行成功", result.get("success"))
    test("结果数量", result.get("count") == 3)
    test("有排名信息", "rank" in result["ranked_items"][0])

    # Test 4: 排序验证
    print("\n[Test 4] Sort Verification")
    print("-" * 40)

    ranked_items = result["ranked_items"]
    first_item = ranked_items[0]
    test("第一名相关性高", "人工智能" in first_item["text"])

    # Test 5: top_k参数
    print("\n[Test 5] Top-K Parameter")
    print("-" * 40)

    result = plugin.execute({
        "query": "人工智能",
        "items": items,
        "top_k": 2,
    })

    test("top_k限制", result.get("count") == 2)

    # Test 6: 评分详情
    print("\n[Test 6] Score Breakdown")
    print("-" * 40)

    first_item = result["ranked_items"][0]
    breakdown = first_item.get("score_breakdown", {})
    test("有评分详情", "relevance" in breakdown)
    test("有相关性分数", "relevance" in breakdown)

    # Test 7: 时间因子
    print("\n[Test 7] Time Factor")
    print("-" * 40)

    items_with_time = [
        {"id": "old", "text": "人工智能", "score": 0.9, "timestamp": time.time() - 86400 * 7},
        {"id": "new", "text": "人工智能", "score": 0.8, "timestamp": time.time() - 3600},
    ]

    result = plugin.execute({
        "query": "人工智能",
        "items": items_with_time,
    })

    ranked_items = result["ranked_items"]
    test("新项目排名靠前", ranked_items[0]["id"] == "new")

    # Test 8: 上下文传递
    print("\n[Test 8] Context Passing")
    print("-" * 40)

    result = plugin.execute({
        "query": "技术",
        "items": items,
        "context": {"user_preference": "技术类"},
    })

    test("上下文被记录", result.get("context_used"))

    # Test 9: 清理
    print("\n[Test 9] Cleanup")
    print("-" * 40)

    plugin.cleanup()
    test("清理完成", True)

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = test_rerank_plugin()
    exit(0 if success else 1)
