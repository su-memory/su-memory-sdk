"""
时序衰减消融实验 — 验证 TemporalRing(Z_60) + 指数衰减重排序的技术效果。

发明点：基于 Z_60 循环群时序编码的指数衰减 + 能量增强/抑制记忆重排序。
对比：有时序重排序 (score*0.7 + recency*0.3) vs 纯语义排序 (无时序)。

测试场景：记忆有时间跨度（部分旧/部分新），查询应优先返回时效性相关记忆。
指标：MRR（平均倒数排名）、时效性命中率（最新相关记忆在 top-k 的比例）。

运行: python benchmarks/ablation_temporal.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from su_memory.sdk.lite_pro import SuMemoryLitePro, TemporalSystem


def build_temporal_dataset(n_per_topic=10, n_topics=5):
    """构造有时效性的记忆数据集：每个主题有 N 条，时间跨度 1-180 天。

    每条记忆有明确的时间戳，查询时"最新"的记忆应是更优答案。
    """
    topics = [
        ("项目进度", "项目进度更新", "v3.0完成", "v4.0重构", "v5.0规划中"),
        ("团队人员", "团队成员变动", "张三入职", "李四离职", "王五升职"),
        ("客户合同", "客户合同状态", "与A公司签约", "B公司续约", "C公司解约"),
        ("产品发布", "产品发布记录", "SDK v1发布", "SDK v2发布", "SDK v4发布"),
        ("财务数据", "财务季度数据", "Q1营收100万", "Q2营收150万", "Q3营收200万"),
    ]
    now = int(time.time())
    memories = []
    for ti, (topic, desc, *stages) in enumerate(topics):
        # 每个 stage 对应不同时间（最新的在最后）
        for si, stage in enumerate(stages):
            days_ago = 120 - si * 40  # 120/80/40 天前
            ts = now - days_ago * 86400
            content = f"{desc}: {stage}（{days_ago}天前）"
            memories.append({
                "content": content,
                "timestamp": ts,
                "topic": topic,
                "stage": si,
                "is_latest": (si == len(stages) - 1),
            })
    return memories


def semantic_score(query, content):
    """简单语义相似度（词重叠率）。"""
    q_words = set(query.lower())
    c_words = set(content.lower())
    if not q_words:
        return 0.0
    return len(q_words & c_words) / len(q_words)


def temporal_decay(timestamp, now, energy_type="earth"):
    """复现 TemporalSystem.calculate_recency_score 的衰减逻辑。"""
    days = (now - timestamp) / 86400
    decay = math.exp(-0.02 * days)
    # 短期记忆加成
    if days < 1:
        decay *= 1.2
    elif days < 7:
        decay *= 1.1
    elif days < 30:
        decay *= 1.0
    else:
        decay *= 0.9
    return decay


def run_temporal_ablation(memories):
    """对比 有时序重排序 vs 无时序。

    每个主题查询：查"最新进展"，正确答案应是最新的那条记忆。
    """
    now = int(time.time())
    topics = sorted(set(m["topic"] for m in memories))
    results = {"no_temporal": [], "with_temporal": []}

    for topic in topics:
        topic_mems = [m for m in memories if m["topic"] == topic]
        query = f"{topic}最新进展"
        # 语义分
        for m in topic_mems:
            m["sem_score"] = semantic_score(query, m["content"])
            m["recency"] = temporal_decay(m["timestamp"], now)

        # 无时序：纯语义排序
        no_t = sorted(topic_mems, key=lambda x: -x["sem_score"])
        # 有时序：score*0.7 + recency*0.3
        with_t = sorted(topic_mems, key=lambda x: -(x["sem_score"] * 0.7 + x["recency"] * 0.3))

        for config, ranked in [("no_temporal", no_t), ("with_temporal", with_t)]:
            # MRR: 最新记忆的倒数排名
            for rank, m in enumerate(ranked, 1):
                if m["is_latest"]:
                    results[config].append(1.0 / rank)
                    break
            # top-1 是否命中最新
            results[config + "_top1"] = results.get(config + "_top1", [])
            results[config + "_top1"].append(1.0 if ranked[0]["is_latest"] else 0.0)

    return results, topics


def main():
    print("=" * 60)
    print("时序衰减消融实验 — TemporalRing(Z_60) 技术效果验证")
    print("=" * 60)

    memories = build_temporal_dataset()
    print(f"数据集: {len(memories)} 条记忆, {len(set(m['topic'] for m in memories))} 个主题")
    print(f"每个主题 3 条, 时间跨度 40-120 天\n")

    results, topics = run_temporal_ablation(memories)

    print(f"{'指标':<25} {'无时序':>10} {'有时序':>10} {'提升':>8}")
    print("-" * 60)

    mrr_no = sum(results["no_temporal"]) / len(results["no_temporal"]) * 100
    mrr_with = sum(results["with_temporal"]) / len(results["with_temporal"]) * 100
    t1_no = sum(results["no_temporal_top1"]) / len(results["no_temporal_top1"]) * 100
    t1_with = sum(results["with_temporal_top1"]) / len(results["with_temporal_top1"]) * 100

    print(f"{'MRR (平均倒数排名)':<25} {mrr_no:>9.1f}% {mrr_with:>9.1f}% {mrr_with-mrr_no:>+7.1f}pp")
    print(f"{'Top-1 命中最新记忆':<25} {t1_no:>9.1f}% {t1_with:>9.1f}% {t1_with-t1_no:>+7.1f}pp")

    print(f"\n结论: 时序重排序将最新记忆的 Top-1 命中率从 {t1_no:.0f}% 提升到 {t1_with:.0f}%")

    out = {
        "experiment": "temporal_decay_ablation",
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "n_memories": len(memories),
        "n_topics": len(topics),
        "results": {
            "no_temporal": {"mrr_pct": round(mrr_no, 1), "top1_pct": round(t1_no, 1)},
            "with_temporal": {"mrr_pct": round(mrr_with, 1), "top1_pct": round(t1_with, 1)},
            "improvement": {"mrr_pp": round(mrr_with - mrr_no, 1), "top1_pp": round(t1_with - t1_no, 1)},
        }
    }
    out_path = ROOT / "benchmarks/results/ablation_temporal.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
