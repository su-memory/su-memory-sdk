"""
_explainability — 可解释性模块（lite_pro.py 拆分）

ExplainabilityModule: 召回结果的归因解释。依赖 MemoryGraph。
从 lite_pro.py 拆分，对外通过 lite_pro.py 再导出保持兼容。
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from su_memory.sdk._memory_graph import MemoryGraph


class ExplainabilityModule:
    """
    可解释性模块
    提供决策路径追溯和因果链可视化
    """

    def __init__(self, memory_graph: MemoryGraph = None):
        self._graph = memory_graph
        self._reasoning_trace: list[dict] = []

    def record_reasoning_step(self, step_type: str, content: str, metadata: dict = None):
        """
        记录推理步骤

        Args:
            step_type: 步骤类型 (perception/recall/reasoning/action)
            content: 步骤内容
            metadata: 额外元数据
        """
        self._reasoning_trace.append({
            "step_type": step_type,
            "content": content,
            "timestamp": int(time.time()),
            "metadata": metadata or {}
        })

    def explain_query(self, query: str, results: list[dict], memory_ids: list[str] = None) -> dict[str, Any]:
        """
        生成查询可解释性报告

        Args:
            query: 查询文本
            results: 查询结果
            memory_ids: 涉及的memory ID列表

        Returns:
            可解释性报告
        """
        report = {
            "query": query,
            "result_count": len(results),
            "reasoning_chain": [],
            "confidence_factors": [],
            "explanation": ""
        }

        # 构建推理链
        for i, result in enumerate(results[:5]):
            chain_item = {
                "rank": i + 1,
                "memory_id": result.get("memory_id"),
                "content_preview": result["content"][:50] + "..." if len(result["content"]) > 50 else result["content"],
                "score": result.get("score", 0),
                "factors": []
            }

            # 分析得分因素
            if result.get("score"):
                chain_item["factors"].append({
                    "factor": "语义相似度",
                    "contribution": f"{result['score']:.2%}"
                })

            if result.get("hops"):
                chain_item["factors"].append({
                    "factor": "多跳推理",
                    "contribution": f"{result['hops']}跳",
                    "path": result.get("path", [])
                })

            if result.get("causal_type"):
                chain_item["factors"].append({
                    "factor": "因果类型",
                    "contribution": result["causal_type"]
                })

            # 时空维度因素
            if result.get("time_decay"):
                chain_item["factors"].append({
                    "factor": "时间衰减",
                    "contribution": f"{result['time_decay']:.2%}"
                })

            if result.get("energy_boost"):
                chain_item["factors"].append({
                    "factor": "能量增强",
                    "contribution": f"{result['energy_boost']:.2f}x",
                    "energy_type": result.get("energy_type", "earth")
                })

            if result.get("energy_type"):
                chain_item["factors"].append({
                    "factor": "Energy System类型",
                    "contribution": result["energy_type"]
                })

            report["reasoning_chain"].append(chain_item)

        # 置信度因素
        if results:
            top_score = results[0].get("score", 0)
            report["confidence_factors"] = [
                {"factor": "语义匹配", "weight": 0.4, "value": f"{top_score:.2%}"},
                {"factor": "因果关联", "weight": 0.3, "value": "基于图谱推理"},
                {"factor": "时序相关性", "weight": 0.2, "value": "时效性已计算"},
                {"factor": "会话上下文", "weight": 0.1, "value": "会话已隔离"}
            ]

        # 生成自然语言解释
        report["explanation"] = self._generate_explanation(query, results, report)

        return report

    def _generate_explanation(self, query: str, results: list[dict], report: dict) -> str:
        """生成自然语言解释"""
        if not results:
            return f"未找到与'{query}'相关的记忆。"

        explanation = f"针对查询'{query}'，系统检索到{len(results)}条相关记忆。\n\n"

        # Top结果解释
        top = results[0]
        explanation += f"最相关记忆：{top['content'][:100]}...\n"
        explanation += f"相关度得分：{top.get('score', 0):.2%}\n\n"

        # 推理路径解释
        if top.get('hops', 0) > 0:
            path = top.get('path', [])
            explanation += f"推理路径：{' → '.join(path[:5])}\n"
            explanation += f"经过{top['hops']}跳推理找到此记忆\n\n"

        # 时空维度解释
        if top.get('time_decay') and top.get('time_decay') < 1.0:
            explanation += f"时间衰减：{top['time_decay']:.2%}（记忆时效性影响）\n"

        if top.get('energy_boost') and top.get('energy_boost') != 1.0:
            energy_type = top.get('energy_type', '土')
            boost = top['energy_boost']
            explanation += f"能量增强：{boost:.2f}x（Energy System类型：{energy_type}）\n"

        # 置信度说明
        explanation += "\n检索因素：\n"
        for factor in report.get("confidence_factors", []):
            explanation += f"  • {factor['factor']}（权重{factor['weight']:.0%}）：{factor['value']}\n"

        return explanation

    def explain_multihop(self, start_memory: str, end_memory: str, path: list[str]) -> dict[str, Any]:
        """
        解释多跳推理路径

        Args:
            start_memory: 起始记忆ID
            end_memory: 结束记忆ID
            path: 推理路径

        Returns:
            多跳解释报告
        """
        if not self._graph:
            return {"error": "MemoryGraph not available"}

        explanation = {
            "path": path,
            "hops": len(path) - 1,
            "edges": [],
            "total_confidence": 1.0
        }

        # 分析每条边
        for i in range(len(path) - 1):
            parent_id, child_id = path[i], path[i + 1]
            causal_type = self._graph.get_causal_type(parent_id, child_id)

            # 获取边上的节点内容
            parent_node = self._graph._nodes.get(parent_id)
            child_node = self._graph._nodes.get(child_id)

            edge_info = {
                "from": parent_id,
                "from_content": parent_node.content[:50] + "..." if parent_node else "",
                "to": child_id,
                "to_content": child_node.content[:50] + "..." if child_node else "",
                "causal_type": causal_type,
                "confidence": self._get_causal_confidence(causal_type)
            }

            explanation["edges"].append(edge_info)
            explanation["total_confidence"] *= edge_info["confidence"]

        # 生成自然语言解释
        explanation["narrative"] = self._generate_path_narrative(explanation)

        return explanation

    def _get_causal_confidence(self, causal_type: str) -> float:
        """获取因果类型置信度"""
        confidence_map = {
            "cause": 0.85,
            "condition": 0.80,
            "result": 0.75,
            "sequence": 0.60,
            "start": 1.0
        }
        return confidence_map.get(causal_type, 0.5)

    def _generate_path_narrative(self, explanation: dict) -> str:
        """生成路径叙事"""
        narrative = f"推理路径共{explanation['hops']}跳\n\n"

        for i, edge in enumerate(explanation["edges"]):
            causal_verb = {
                "cause": "导致",
                "condition": "条件触发",
                "result": "结果产生",
                "sequence": "随后发生"
            }.get(edge["causal_type"], "关联到")

            narrative += f"第{i + 1}跳：{edge['from_content']}\n"
            narrative += f"   {causal_verb} → {edge['to_content']}\n\n"

        narrative += f"综合置信度：{explanation['total_confidence']:.1%}\n"

        return narrative

    def visualize_reasoning_tree(self, query: str, results: list[dict]) -> dict[str, Any]:
        """
        生成推理树可视化数据

        Args:
            query: 查询文本
            results: 结果列表

        Returns:
            树形结构数据（可用于前端渲染）
        """
        tree = {
            "name": query,
            "type": "query",
            "children": []
        }

        for result in results[:5]:
            node = {
                "name": result["content"][:30] + "...",
                "type": "memory",
                "score": result.get("score", 0),
                "metadata": {
                    "memory_id": result.get("memory_id"),
                    "hops": result.get("hops", 0),
                    "causal_type": result.get("causal_type", "unknown")
                }
            }

            # 如果有路径，展开子节点
            if result.get("path") and len(result["path"]) > 1:
                node["children"] = [
                    {
                        "name": f"跳{i}: {pid[:20]}...",
                        "type": "hop",
                        "hop_index": i
                    }
                    for i, pid in enumerate(result["path"][1:], 1)
                ]

            tree["children"].append(node)


        return tree

    def get_reasoning_summary(self) -> dict[str, Any]:
        """
        获取推理过程摘要

        Returns:
            推理摘要统计
        """
        if not self._reasoning_trace:
            return {"total_steps": 0, "message": "暂无推理记录"}

        # 统计各类型步骤
        step_counts = defaultdict(int)
        for step in self._reasoning_trace:
            step_counts[step["step_type"]] += 1

        return {
            "total_steps": len(self._reasoning_trace),
            "step_distribution": dict(step_counts),
            "first_step": self._reasoning_trace[0] if self._reasoning_trace else None,
            "last_step": self._reasoning_trace[-1] if self._reasoning_trace else None
        }


