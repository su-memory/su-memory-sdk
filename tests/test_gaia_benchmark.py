"""
GAIA 风格基准测试 — 贝叶斯增强 vs 原始系统

模拟 GAIA benchmark 的三级难度评估框架，量化 BayesianAugmenter
对系统查询、预测、推理能力的准确度提升。

测试设计：
  Level 1: 事实检索准确度（单步查询）
  Level 2: 多跳推理准确度（组合查询+因果推断）
  Level 3: 复杂推理与规划（多约束决策）

每个 Level 包含 N 道测试题，对比原始输出与贝叶斯增强输出的准确性。

Usage:
    PYTHONPATH=src python tests/test_gaia_benchmark.py
"""

import sys
import os
import json
import time
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from su_memory.sdk import SuMemoryLitePro
from su_memory.sdk.bayesian_augmenter import BayesianAugmenter, EnhancedOutput


# ============================================================
# GAIA 风格测试题设计
# ============================================================

@dataclass
class GAIATestCase:
    """GAIA 风格测试用例"""
    id: str
    level: int  # 1, 2, or 3
    category: str  # "factual", "temporal", "causal", "multi-hop", "planning"
    description: str
    setup_queries: List[str] = field(default_factory=list)  # 需要先注入的背景知识
    test_query: str = ""
    ground_truth_keywords: List[str] = field(default_factory=list)  # 正确答案应包含的关键词
    ground_truth_exclude: List[str] = field(default_factory=list)   # 正确答案不应包含的关键词
    max_score: float = 1.0


# 背景知识注入
BACKGROUND_KNOWLEDGE = [
    "项目ROI在2024年Q3达到25.3%，比Q2提升了8.2个百分点",
    "团队使用Python 3.11进行开发，主要框架是FastAPI和React",
    "数据库采用PostgreSQL 15，支持JSONB和全文搜索",
    "系统在2024年12月完成了微服务架构升级，从单体应用拆分为12个独立服务",
    "因为引入了Redis缓存层，API响应时间从平均320ms降低到45ms",
    "用户满意度调查显示，2024年NPS得分从35提升到72",
    "机器学习模型使用PyTorch 2.0训练，在A100 GPU上训练了48小时",
    "模型在测试集上达到94.7%的准确率，比上一个版本提升了2.3%",
    "如果日志量超过100GB/天，系统会自动触发告警并扩容ES集群",
    "当并发请求超过10000 QPS时，负载均衡器会自动启动新的Pod",
    "市场价格波动超过5%时，风险控制模块会暂停交易并通知管理员",
    "用户流失率达到10%时，自动触发客户挽留策略",
    "系统上线后3个月内，共处理了500万次API请求",
    "故障恢复时间从平均15分钟缩短到3分钟，可用性达到99.97%",
    "通过代码审查和自动化测试，缺陷率降低了40%",
]


# ============================================================
# Level 1: 事实检索 (GAIA Level 1 — 单步信息提取)
# ============================================================

LEVEL1_TESTS = [
    GAIATestCase(
        id="L1-001",
        level=1,
        category="factual",
        description="单一事实提取：项目的投资回报率",
        test_query="项目ROI是多少？",
        ground_truth_keywords=["25", "Q3", "8.2"],
        ground_truth_exclude=["35", "下降"],
        max_score=1.0,
    ),
    GAIATestCase(
        id="L1-002",
        level=1,
        category="factual",
        description="数值精确回忆：API响应时间优化",
        test_query="API响应时间优化了多少？",
        ground_truth_keywords=["320", "45"],
        ground_truth_exclude=["变慢", "增加"],
        max_score=1.0,
    ),
    GAIATestCase(
        id="L1-003",
        level=1,
        category="factual",
        description="多事实并行提取：技术栈与架构",
        test_query="系统使用什么技术栈？微服务数量是多少？",
        ground_truth_keywords=["Python", "FastAPI", "React", "PostgreSQL", "12", "个"],
        ground_truth_exclude=["Django", "Vue"],
        max_score=1.5,
    ),
    GAIATestCase(
        id="L1-004",
        level=1,
        category="factual",
        description="数值对比：NPS得分变化",
        test_query="NPS得分变化情况？",
        ground_truth_keywords=["35", "72"],
        ground_truth_exclude=["下降", "降低"],
        max_score=1.0,
    ),
    GAIATestCase(
        id="L1-005",
        level=1,
        category="factual",
        description="精确日期回忆：微服务架构升级时间",
        test_query="微服务架构什么时候完成的升级？",
        ground_truth_keywords=["2024", "12"],
        ground_truth_exclude=["2023", "2025"],
        max_score=1.0,
    ),
]

# ============================================================
# Level 2: 多跳推理 (GAIA Level 2 — 组合+因果)
# ============================================================

LEVEL2_TESTS = [
    GAIATestCase(
        id="L2-001",
        level=2,
        category="multi-hop",
        description="多跳推理：优化效果因果链",
        test_query="引入Redis缓存后，系统性能改善体现在哪些方面？具体数据是什么？",
        ground_truth_keywords=["响应时间", "320", "45", "Redis", "缓存"],
        ground_truth_exclude=[],
        max_score=2.0,
    ),
    GAIATestCase(
        id="L2-002",
        level=2,
        category="causal",
        description="因果预测：日志告警触发条件",
        test_query="什么情况下会触发ES集群扩容？后果是什么？",
        ground_truth_keywords=["日志量", "100GB", "告警", "扩容"],
        ground_truth_exclude=["手动", "人工"],
        max_score=2.0,
    ),
    GAIATestCase(
        id="L2-003",
        level=2,
        category="multi-hop",
        description="数据综合：模型性能全景",
        test_query="机器学习模型的训练情况和性能如何？",
        ground_truth_keywords=["PyTorch", "A100", "48小时", "94.7", "2.3"],
        ground_truth_exclude=["TensorFlow"],
        max_score=2.0,
    ),
    GAIATestCase(
        id="L2-004",
        level=2,
        category="causal",
        description="条件推理：并发触发机制",
        test_query="当并发请求超过10000 QPS时会发生什么？",
        ground_truth_keywords=["负载均衡", "Pod", "启动"],
        ground_truth_exclude=["崩溃", "宕机"],
        max_score=1.5,
    ),
    GAIATestCase(
        id="L2-005",
        level=2,
        category="temporal",
        description="时序推理：三个月运营数据",
        test_query="系统上线后前三个月的表现如何？",
        ground_truth_keywords=["500万", "API请求", "三个月"],
        ground_truth_exclude=[],
        max_score=1.5,
    ),
    GAIATestCase(
        id="L2-006",
        level=2,
        category="causal",
        description="阈值触发：风险管理",
        test_query="市场价格波动超过阈值时系统的反应是什么？",
        ground_truth_keywords=["5%", "暂停交易", "通知管理员"],
        ground_truth_exclude=[],
        max_score=1.5,
    ),
]

# ============================================================
# Level 3: 复杂推理 (GAIA Level 3 — 多约束规划)
# ============================================================

LEVEL3_TESTS = [
    GAIATestCase(
        id="L3-001",
        level=3,
        category="planning",
        description="综合诊断：系统优化路线图推断",
        test_query="基于已有信息，系统做了哪些优化？效果如何？还有哪些可以继续优化的地方？",
        ground_truth_keywords=["Redis", "微服务", "代码审查", "自动化测试", "性能提升"],
        ground_truth_exclude=[],
        max_score=3.0,
    ),
    GAIATestCase(
        id="L3-002",
        level=3,
        category="planning",
        description="风险评估：识别系统薄弱环节",
        test_query="分析系统目前的监控和自愈能力，指出可能的风险点",
        ground_truth_keywords=["日志", "并发", "告警", "扩容", "风险"],
        ground_truth_exclude=[],
        max_score=3.0,
    ),
    GAIATestCase(
        id="L3-003",
        level=3,
        category="planning",
        description="效果归因：质量改进链路",
        test_query="从缺陷率降低回溯，分析是什么措施导致了质量改善？",
        ground_truth_keywords=["代码审查", "自动化测试", "40%", "缺陷"],
        ground_truth_exclude=[],
        max_score=2.5,
    ),
    GAIATestCase(
        id="L3-004",
        level=3,
        category="planning",
        description="趋势推断：用户满意度提升路径",
        test_query="用户满意度大幅提升的可能原因有哪些？能否量化说明？",
        ground_truth_keywords=["NPS", "35", "72", "响应时间", "性能"],
        ground_truth_exclude=[],
        max_score=2.5,
    ),
]

ALL_TESTS = LEVEL1_TESTS + LEVEL2_TESTS + LEVEL3_TESTS


# ============================================================
# 评分引擎
# ============================================================

class GAIAScorer:
    """GAIA 风格评分引擎 — 基于关键词匹配的准确度量化"""

    @staticmethod
    def score_response(response_text: str, test_case: GAIATestCase) -> Tuple[float, Dict]:
        """
        对单个回答打分

        Returns:
            (normalized_score, details_dict)
        """
        text_lower = response_text.lower() if response_text else ""

        # 关键词命中
        hits = []
        for kw in test_case.ground_truth_keywords:
            kw_lower = kw.lower()
            if kw_lower in text_lower:
                hits.append(kw)

        # 排除词检查（不应出现的词出现了则扣分）
        penalties = []
        for ex in test_case.ground_truth_exclude:
            ex_lower = ex.lower()
            if ex_lower in text_lower:
                penalties.append(ex)

        # 计算原始得分
        hit_ratio = len(hits) / len(test_case.ground_truth_keywords) if test_case.ground_truth_keywords else 0
        penalty_ratio = len(penalties) / max(len(test_case.ground_truth_exclude), 1)
        raw_score = hit_ratio - (penalty_ratio * 0.5)  # 每个排除词扣 0.5 分权重

        # 归一化到 max_score
        normalized = max(0, raw_score) * test_case.max_score

        return normalized, {
            "hits": hits,
            "total_keywords": len(test_case.ground_truth_keywords),
            "penalties": penalties,
            "penalty_exclude": len(test_case.ground_truth_exclude),
            "hit_ratio": round(hit_ratio, 3),
            "raw_score": round(raw_score, 3),
            "normalized_score": round(normalized, 3),
            "max_score": test_case.max_score,
        }

    @staticmethod
    def score_from_enhanced_output(output: EnhancedOutput, test_case: GAIATestCase) -> Dict:
        """
        从 BayesianAugmenter 的 EnhancedOutput 中提取两路回答并分别打分
        """
        # 原始路径回答
        original_text = ""
        if output.original and isinstance(output.original, dict):
            results = output.original.get("results", [])
            if results:
                for r in results[:3]:
                    original_text += str(r.get("content", r)) + " "

        # 贝叶斯路径回答
        bayesian_text = ""
        if output.bayesian and isinstance(output.bayesian, dict):
            results = output.bayesian.get("results", [])
            if results:
                for r in results[:3]:
                    bayesian_text += str(r.get("content", r)) + " "

        # 对比信息
        comparisons = []
        if output.comparisons:
            for c in output.comparisons:
                if hasattr(c, 'to_dict'):
                    comparisons.append(c.to_dict())
                elif isinstance(c, dict):
                    comparisons.append(c)

        original_score, original_details = GAIAScorer.score_response(original_text, test_case)
        bayesian_score, bayesian_details = GAIAScorer.score_response(bayesian_text, test_case)

        improvement = bayesian_score - original_score
        improvement_pct = (improvement / max(original_score, 0.001)) * 100

        return {
            "test_id": test_case.id,
            "level": test_case.level,
            "category": test_case.category,
            "description": test_case.description,
            "original_score": round(original_score, 3),
            "bayesian_score": round(bayesian_score, 3),
            "improvement": round(improvement, 3),
            "improvement_pct": round(improvement_pct, 1),
            "original_details": original_details,
            "bayesian_details": bayesian_details,
            "comparisons": comparisons[:3],  # top 3 comparisons
        }


# ============================================================
# GAIA 基准测试执行器
# ============================================================

class GAIA_BenchmarkRunner:
    """GAIA 风格基准测试运行器"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.client: Optional[SuMemoryLitePro] = None
        self.augmenter: Optional[BayesianAugmenter] = None
        self._all_results: List[Dict] = []

    def setup(self) -> bool:
        """初始化系统并注入背景知识"""
        try:
            self.client = SuMemoryLitePro(
                enable_vector=False,
                enable_graph=False,   # 跳过 JGraphT（需 Java）
                enable_temporal=False, # 跳过 Ollama API 调用
                enable_prediction=False,
            )
            self.augmenter = BayesianAugmenter(
                self.client,
                enable_network=True,
                enable_predictor=True,
                enable_auto_sync=True,
                prior_type="weak",
                verbose=False,
            )

            # 注入背景知识
            for i, text in enumerate(BACKGROUND_KNOWLEDGE):
                self.augmenter._client.add(f"knowledge_{i}", text, {"source": "gaia_benchmark"})

            time.sleep(0.5)  # 等待索引完成
            return True
        except Exception as e:
            print(f"Setup failed: {e}")
            return False

    def run_level(self, tests: List[GAIATestCase], level_name: str) -> Dict:
        """运行一个级别的所有测试"""
        level_results = []
        total_original = 0.0
        total_bayesian = 0.0

        for tc in tests:
            try:
                output = self.augmenter.query(tc.test_query, top_k=5)
                result = GAIAScorer.score_from_enhanced_output(output, tc)
                level_results.append(result)
                total_original += result["original_score"]
                total_bayesian += result["bayesian_score"]

                if self.verbose:
                    print(f"  {tc.id}: orig={result['original_score']:.2f}, "
                          f"bayes={result['bayesian_score']:.2f}, "
                          f"Δ={result['improvement']:+.2f}")
            except Exception as e:
                if self.verbose:
                    print(f"  {tc.id}: ERROR - {e}")

        n = len(tests)
        avg_original = total_original / n if n > 0 else 0
        avg_bayesian = total_bayesian / n if n > 0 else 0

        return {
            "level": level_name,
            "num_tests": n,
            "avg_original_score": round(avg_original, 3),
            "avg_bayesian_score": round(avg_bayesian, 3),
            "avg_improvement": round(avg_bayesian - avg_original, 3),
            "improvement_pct": round(((avg_bayesian - avg_original) / max(avg_original, 0.001)) * 100, 1),
            "results": level_results,
        }

    def run_full_benchmark(self) -> Dict:
        """运行完整 GAIA 基准测试"""
        print("\n" + "=" * 70)
        print("  GAIA 风格基准测试 — 贝叶斯增强 vs 原始系统")
        print("=" * 70)

        if not self.setup():
            return {"status": "error", "message": "Setup failed"}

        print(f"\n  背景知识注入: {len(BACKGROUND_KNOWLEDGE)} 条")
        print(f"  总测试用例: {len(ALL_TESTS)} 个")
        print()

        all_level_data = {}
        grand_original = 0.0
        grand_bayesian = 0.0
        total_tests = 0

        for level_num, tests, level_name in [
            (1, LEVEL1_TESTS, "Level 1: 事实检索"),
            (2, LEVEL2_TESTS, "Level 2: 多跳推理"),
            (3, LEVEL3_TESTS, "Level 3: 复杂推理"),
        ]:
            print(f"  ▶ {level_name} ({len(tests)} tests)...")
            level_result = self.run_level(tests, f"Level {level_num}")
            all_level_data[f"level_{level_num}"] = level_result

            n = level_result["num_tests"]
            total_tests += n
            grand_original += level_result["avg_original_score"] * n
            grand_bayesian += level_result["avg_bayesian_score"] * n

            print(f"    原始平均分: {level_result['avg_original_score']:.3f}")
            print(f"    贝叶斯平均分: {level_result['avg_bayesian_score']:.3f}")
            print(f"    提升: {level_result['improvement_pct']:+.1f}%")
            print()

        # 总体统计
        grand_avg_original = grand_original / total_tests if total_tests > 0 else 0
        grand_avg_bayesian = grand_bayesian / total_tests if total_tests > 0 else 0
        grand_improvement = grand_avg_bayesian - grand_avg_original
        grand_improvement_pct = (grand_improvement / max(grand_avg_original, 0.001)) * 100

        # 反馈闭环：基于测试结果更新贝叶斯信念
        self._apply_benchmark_feedback(all_level_data)

        # 获取准确度报告
        accuracy_report = self.augmenter.get_accuracy_report()

        # 构建最终报告
        final_report = {
            "benchmark": "GAIA-style",
            "version": "1.0",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests": total_tests,
            "background_knowledge_injected": len(BACKGROUND_KNOWLEDGE),
            "summary": {
                "grand_avg_original": round(grand_avg_original, 3),
                "grand_avg_bayesian": round(grand_avg_bayesian, 3),
                "grand_improvement": round(grand_improvement, 3),
                "grand_improvement_pct": round(grand_improvement_pct, 1),
                "verdict": (
                    "✅ 贝叶斯增强显著提升准确度"
                    if grand_improvement_pct > 5
                    else "🟡 贝叶斯增强有边际提升" if grand_improvement_pct > 0
                    else "⚠️ 贝叶斯增强未见提升"
                ),
            },
            "level_breakdown": all_level_data,
            "accuracy_tracking": accuracy_report,
        }

        # 打印总结报告
        self._print_summary(final_report)

        self._all_results = [final_report]
        return final_report

    def _apply_benchmark_feedback(self, all_level_data: Dict):
        """将基准测试结果作为反馈更新贝叶斯信念"""
        for level_key, level_data in all_level_data.items():
            for result in level_data.get("results", []):
                test_id = result["test_id"]
                orig_score = result["original_score"]
                bayes_score = result["bayesian_score"]

                # 为 query 类预测提供反馈
                # Bayesian 得分高于原始 = 正确预测
                if bayes_score > orig_score:
                    self.augmenter.feedback(
                        query=test_id,
                        ground_truth_value=bayes_score,
                    )
                elif bayes_score < orig_score:
                    self.augmenter.feedback(
                        query=test_id,
                        ground_truth_value=orig_score,
                    )

    def _print_summary(self, report: Dict):
        """打印格式化的基准测试报告"""
        print("=" * 70)
        print("  📊 GAIA 基准测试结果总结")
        print("=" * 70)

        s = report["summary"]
        print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  总测试用例:    {report['total_tests']:>3d} 个                              │
  │  背景知识:      {report['background_knowledge_injected']:>3d} 条                              │
  │                                                     │
  │  原始系统平均分:   {s['grand_avg_original']:.3f}                          │
  │  贝叶斯增强平均分: {s['grand_avg_bayesian']:.3f}                          │
  │  绝对提升:        {s['grand_improvement']:+.3f}                          │
  │  相对提升:        {s['grand_improvement_pct']:+.1f}%                        │
  │                                                     │
  │  判定: {s['verdict']}              │
  └─────────────────────────────────────────────────────┘
""")

        # 逐级明细
        for level_key in ["level_1", "level_2", "level_3"]:
            ld = report["level_breakdown"].get(level_key, {})
            if ld:
                print(f"  [{ld.get('level', level_key)}] "
                      f"原始:{ld.get('avg_original_score', 0):.3f} → "
                      f"贝叶斯:{ld.get('avg_bayesian_score', 0):.3f} "
                      f"({ld.get('improvement_pct', 0):+.1f}%)")

        print()

    def save_report(self, filepath: str = None):
        """保存报告到 JSON 文件"""
        if not self._all_results:
            print("No results to save.")
            return

        if filepath is None:
            filepath = os.path.join(
                os.path.dirname(__file__),
                "..",
                "benchmarks",
                "gaia_benchmark_results.json",
            )

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._all_results[0], f, indent=2, ensure_ascii=False)

        print(f"  📁 报告已保存: {filepath}")


# ============================================================
# 快速验证函数
# ============================================================

def quick_validation():
    """快速验证 — 仅跑少量测试确认框架正确"""
    scorer = GAIAScorer()

    # 测试评分逻辑
    tc = LEVEL1_TESTS[0]
    good_answer = "项目ROI在2024年Q3达到25.3%，比Q2提升了8.2个百分点"
    bad_answer = "ROI下降了35%"

    good_score, good_detail = scorer.score_response(good_answer, tc)
    bad_score, bad_detail = scorer.score_response(bad_answer, tc)

    print("Quick Validation:")
    print(f"  Good answer score: {good_score:.3f} (hits: {good_detail['hits']})")
    print(f"  Bad answer score:  {bad_score:.3f} (penalties: {bad_detail['penalties']})")
    print(f"  Scorer logic: {'✅ OK' if good_score > bad_score else '❌ FAIL'}")

    return good_score > bad_score


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GAIA 风格贝叶斯增强基准测试")
    parser.add_argument("--quick", action="store_true", help="快速验证评分逻辑")
    parser.add_argument("--save", action="store_true", help="保存结果到 JSON")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    args = parser.parse_args()

    if args.quick:
        success = quick_validation()
        sys.exit(0 if success else 1)

    runner = GAIA_BenchmarkRunner(verbose=args.verbose)
    report = runner.run_full_benchmark()

    if args.save:
        runner.save_report()

    if report["summary"]["grand_improvement_pct"] > 0:
        print("✅ 贝叶斯增强在 GAIA 基准测试中展现正向提升。")
    else:
        print("⚠️ 贝叶斯增强未在本次基准测试中展现显著提升，可能需要更多反馈数据。")
