"""
su-memory v3.5.0 — Reflection QA Synthesizer
=============================================
基于 MEMO (2605.15156v2) 的 Reflection QA 框架，
从 su-memory 记忆语料中合成因果 QA 对。

适配策略 (MEMO Table 9 消融实验指导):
- ✅ Step 1: Fact Extraction — 从记忆中提取实体和属性
- ❌ Step 2: Fact Consolidation — 跳过 (叙事文本中反而有害)
- ❌ Step 3: Fact Verification — 跳过 (同上)
- ✅ Step 4: Entity Surfacing — 为每个实体发现相关事实
- ✅ Step 5: Cross-document Synthesis — 跨文档因果合成

su-memory 增强:
- 能量分组: 用五行生克关系自动分块，控制 Step 5 复杂度
- BayesianCausal 校验: 合成的 QA 对经过后验量化筛选
- v3.6.0 本地训练就绪: training_data_report() 输出质量检查

本地优先设计 (v3.6.0 路线):
- max_pairs 默认 200 (精炼 > 海量)
- min_confidence 默认 0.4 (高门槛)
- training_data_report() → ready_for_training 布尔值
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ==========================================================================
# SynthesizedQAPair
# ==========================================================================

@dataclass
class SynthesizedQAPair:
    """合成的因果 QA 对"""
    cause_text: str
    effect_text: str
    cause_entity: str              # 原因实体
    effect_entity: str             # 结果实体
    reflection_depth: int          # 反射深度 (1=直连, 2=单跳, 3=双跳)
    energy_relation: str           # 能量关系 (enhance/suppress/same/neutral)
    confidence: float              # BayesianCausal 后验置信度
    source_memory_ids: list[str]   # 来源记忆 ID
    qa_pair_id: str                # 合成 QA 对唯一 ID


# ==========================================================================
# ReflectionSynthesizer
# ==========================================================================

class ReflectionSynthesizer:
    """
    MEMO-style Reflection QA 合成引擎。

    工作流:
    1. 接收记忆列表 → 分词 + 实体识别
    2. 能量类型推断 (wood/fire/earth/metal/water)
    3. 能量分组 → 块内 Step 1: Fact Extraction
    4. 跨能量组 Step 4: Entity Surfacing (发现果→因)
    5. 跨组 Step 5: Cross-document Synthesis (因果合成)
    6. BayesianCausal 后验筛选
    """

    # 实体识别模式 (中文)
    ENTITY_INDICATORS = [
        "上涨", "下降", "增长", "减少", "提升", "降低",
        "突破", "创", "宣布", "发布", "实施", "落地",
        "执行", "完成", "启动",
    ]

    # 因果指示词 (MEMO Step 1)
    CAUSAL_MARKERS = [
        "导致", "因为", "因此", "所以", "使得", "促使", "带动",
        "引起", "引发", "造成", "带来", "产生", "推动",
    ]

    # 数值提取模式
    NUMERIC_PATTERN = re.compile(
        r"([\d]+(?:\.[\d]+)?)\s*(%|倍|成|个百分点|点|万|亿|个)?"
    )
    # 中文数字
    CN_NUMERIC = {
        "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
        "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
        "百": 100, "千": 1000, "万": 10000, "亿": 100000000,
    }

    def __init__(
        self,
        energy_bus=None,
        bayesian=None,
        min_confidence: float = 0.4,
        max_pairs: int = 200,
        seed: int = 42,
    ):
        """
        Args:
            energy_bus: EnergyBus 实例 (用于能量类型推断和分组)
            bayesian: BayesianCausal 实例 (用于合成后筛选)
            min_confidence: 最低后验置信度 (v3.6.0 路线: 0.4)
            max_pairs: 最大合成 QA 对数 (v3.6.0 路线: 200, 精炼>海量)
            seed: 随机种子
        """
        self._energy_bus = energy_bus
        self._bayesian = bayesian
        self.min_confidence = min_confidence
        self.max_pairs = max_pairs
        self._rng = random.Random(seed)

    # ────────────────────────────────────────────────
    # Step 1: Fact Extraction
    # ────────────────────────────────────────────────

    def extract_facts(
        self, memories: list[dict]
    ) -> list[dict]:
        """
        从记忆列表中提取实体、数值、属性和因果指示词。
        """
        facts = []
        for mem in memories:
            content = mem.get("content", "")
            if not content:
                continue
            facts.append({
                "memory_id": mem.get("id", ""),
                "content": content,
                "entities": self._extract_entities(content),
                "numerics": self._extract_numerics(content),
                "causals": self._extract_causal_indicators(content),
                "energy_type": self._infer_energy_type_from_content(content),
            })
        return facts

    def _extract_entities(self, content: str) -> list[str]:
        """
        中文实体提取 (基于规则 + 关键词分割)。

        策略: 以因果指示词和动词为锚点，将内容拆分为多段，
        每段最长的名词短语作为实体。
        """
        entities: list[str] = []

        # 按指示词和动词切分
        segments = [content]
        for indicator in self.ENTITY_INDICATORS:
            new_segments = []
            for seg in segments:
                if indicator in seg:
                    parts = seg.split(indicator, 1)
                    new_segments.extend(p for p in parts if p.strip())
                else:
                    new_segments.append(seg)
            segments = new_segments

        # 每个片段取前 6 个字符作为实体名
        for seg in segments:
            seg = seg.strip()
            if seg and len(seg) >= 2:
                # 取片段中不含数字的纯文本部分
                clean = re.sub(r"[\d.%]+", "", seg)[:6]
                if clean and clean not in entities:
                    entities.append(clean)

        # 去重
        return entities

    def _extract_numerics(self, content: str) -> list[dict]:
        """
        提取数值变化模式。
        Returns:
            [{"value": float, "unit": str, "direction": "+"/"-"/"~"}, ...]
        """
        numerics: list[dict] = []

        # 阿拉伯数字 + 可选单位
        for m in self.NUMERIC_PATTERN.finditer(content):
            num = float(m.group(1))
            unit = m.group(2) or ""
            # 方向推断: 查看附近的动词
            start = max(0, m.start() - 6)
            prefix = content[start:m.start()]
            if any(w in prefix for w in ["上涨", "增长", "提升", "增", "升", "突破"]):
                direction = "+"
            elif any(w in prefix for w in ["下降", "减少", "降低", "降", "减"]):
                direction = "-"
            else:
                direction = "~"
            numerics.append({"value": num, "unit": unit, "direction": direction})

        # 中文数字
        for cn_word, cn_val in self.CN_NUMERIC.items():
            if cn_word in content:
                idx = content.index(cn_word)
                start = max(0, idx - 6)
                prefix = content[start:idx]
                if any(w in prefix for w in ["上涨", "增长", "提升"]):
                    direction = "+"
                elif any(w in prefix for w in ["下降", "减少", "降低"]):
                    direction = "-"
                else:
                    direction = "~"
                numerics.append({
                    "value": cn_val,
                    "unit": "cn",
                    "direction": direction,
                })

        return numerics

    def _extract_causal_indicators(self, content: str) -> list[str]:
        """提取因果指示词。"""
        return [m for m in self.CAUSAL_MARKERS if m in content]

    # ────────────────────────────────────────────────
    # Step 4: Entity Surfacing
    # ────────────────────────────────────────────────

    def surface_entities(
        self, facts: list[dict]
    ) -> dict[str, list[str]]:
        """
        为每个实体发现相关的跨文档事实 (果→因 反向查找)。

        对每个 effect 实体，在所有记忆中搜索可能的原因。
        使用因果指示词作为桥接信号。

        Returns:
            {entity_name: [related_memory_ids]}
        """
        entity_map: dict[str, list[str]] = defaultdict(list)

        # 收集所有提取的实体
        all_entities: dict[str, set[str]] = {}  # entity → {memory_id}
        for fact in facts:
            for entity in fact.get("entities", []):
                if entity not in all_entities:
                    all_entities[entity] = set()
                all_entities[entity].add(fact.get("memory_id", ""))

        # 因果指示词跨文档桥接
        for fact_a in facts:
            causals_a = set(fact_a.get("causals", []))
            entities_a = set(fact_a.get("entities", []))
            if not causals_a:
                continue

            for fact_b in facts:
                if fact_b.get("memory_id") == fact_a.get("memory_id"):
                    continue
                causals_b = set(fact_b.get("causals", []))
                # 如果 B 有因果指示词且实体不同于 A → B 可能影响 A
                if causals_b and entities_a != set(fact_b.get("entities", [])):
                    shared = causals_a & causals_b
                    if shared:
                        for entity in fact_a.get("entities", []):
                            if fact_b.get("memory_id") not in entity_map[entity]:
                                entity_map[entity].append(fact_b["memory_id"])

        return dict(entity_map)

    # ────────────────────────────────────────────────
    # Step 5: Cross-document Causal Synthesis
    # ────────────────────────────────────────────────

    def synthesize_causal_pairs(
        self, facts: list[dict]
    ) -> list[SynthesizedQAPair]:
        """
        跨文档因果合成。

        按能量分组 (五行)，组内做逐对合成，
        组间做因果链合成 (生克路径)。
        """
        if len(facts) < 2:
            return []

        pairs: list[SynthesizedQAPair] = []

        # 能量分组
        groups = self._group_by_energy(facts)

        # 组内合成 (同元素内的因果线索)
        for _etype, group_facts in groups.items():
            if len(group_facts) < 2:
                continue
            sampled = self._rng.sample(
                group_facts, min(20, len(group_facts))
            )
            for i in range(len(sampled)):
                for j in range(i + 1, len(sampled)):
                    if len(pairs) >= self.max_pairs:
                        return self._sort_by_confidence(pairs)
                    pair = self._try_synthesize(sampled[i], sampled[j])
                    if pair and pair.confidence >= self.min_confidence:
                        pairs.append(pair)

        # 组间合成 (生克关系驱动的因果链)
        for etype_a in groups:
            enhanced = self._get_enhanced_element(etype_a)
            if enhanced and enhanced in groups:
                sa = self._rng.sample(
                    groups[etype_a], min(10, len(groups[etype_a]))
                )
                sb = self._rng.sample(
                    groups[enhanced], min(10, len(groups[enhanced]))
                )
                for fa in sa:
                    for fb in sb:
                        if len(pairs) >= self.max_pairs:
                            return self._sort_by_confidence(pairs)
                        pair = self._try_synthesize_chain(fa, fb, "enhance")
                        if pair and pair.confidence >= self.min_confidence:
                            pairs.append(pair)

        return self._sort_by_confidence(pairs)

    def _try_synthesize(
        self, fact_a: dict, fact_b: dict
    ) -> SynthesizedQAPair | None:
        """
        尝试从两个事实合成因果 QA 对。

        合成判断:
        1. A 含因果指示词 ?→ B (A 是原因，B 是结果)
        2. A 和 B 共享实体 → 直接因果
        3. BayesianCausal 后验筛选
        """
        content_a = fact_a.get("content", "")
        content_b = fact_b.get("content", "")
        entities_a = fact_a.get("entities", [])
        entities_b = fact_b.get("entities", [])
        causals_a = set(fact_a.get("causals", []))

        # 条件 1: A 包含因果指示词 (A 很可能是原因)
        is_a_cause = len(causals_a) > 0

        if not is_a_cause:
            return None

        # 条件 2: 共享实体或因果指示词
        shared_entities = set(entities_a) & set(entities_b)

        # 置信度计算
        confidence = 0.5
        if shared_entities:
            confidence += 0.1 * len(shared_entities)
        confidence += 0.05 * len(causals_a)

        # BayesianCausal 后验筛选
        if self._bayesian:
            try:
                bf = self._bayesian.causal_hypothesis_test(content_a, content_b)
                posterior = bf / (bf + 1) if bf else 0
                confidence = (confidence + posterior) / 2
            except Exception as e:
                logger.warning(
                    "BayesianCausal hypothesis test failed in _try_synthesize: %s",
                    e,
                )
        else:
            # 无 BayesianCausal 时用因果关键词匹配度
            causals_b = set(fact_b.get("causals", []))
            shared_causals = causals_a & causals_b
            confidence += 0.05 * len(shared_causals)

        confidence = round(min(confidence, 0.95), 4)

        if confidence < self.min_confidence:
            return None

        # 判断能量关系
        etype_a = fact_a.get("energy_type", "unknown")
        etype_b = fact_b.get("energy_type", "unknown")
        energy_rel = self._classify_energy_relation(etype_a, etype_b)

        return SynthesizedQAPair(
            cause_text=content_a,
            effect_text=content_b,
            cause_entity=entities_a[0] if entities_a else "unknown",
            effect_entity=entities_b[0] if entities_b else "unknown",
            reflection_depth=2 if energy_rel == "enhance" else 1,
            energy_relation=energy_rel,
            confidence=confidence,
            source_memory_ids=[
                fact_a.get("memory_id", ""),
                fact_b.get("memory_id", ""),
            ],
            qa_pair_id=_hash_pair_id(content_a, content_b),
        )

    def _try_synthesize_chain(
        self, fact_a: dict, fact_b: dict, relation: str
    ) -> SynthesizedQAPair | None:
        """
        能量关系驱动的因果链合成。

        Args:
            relation: "enhance" (生) | "suppress" (克)
        """
        content_a = fact_a.get("content", "")
        content_b = fact_b.get("content", "")

        # 生克关系的置信度基准更高
        base_confidence = 0.6 if relation == "enhance" else 0.5

        # 检查是否有因果指示词
        causals_a = set(fact_a.get("causals", []))
        causals_b = set(fact_b.get("causals", []))
        if causals_a or causals_b:
            base_confidence += 0.1

        if self._bayesian:
            try:
                bf = self._bayesian.causal_hypothesis_test(content_a, content_b)
                posterior = bf / (bf + 1) if bf else 0
                base_confidence = (base_confidence + posterior) / 2
            except Exception as e:
                logger.warning(
                    "BayesianCausal hypothesis test failed in _try_synthesize_chain: %s",
                    e,
                )

        confidence = round(min(base_confidence, 0.95), 4)

        if confidence < self.min_confidence:
            return None

        return SynthesizedQAPair(
            cause_text=content_a,
            effect_text=content_b,
            cause_entity=fact_a.get("entities", ["unknown"])[0],
            effect_entity=fact_b.get("entities", ["unknown"])[0],
            reflection_depth=2,
            energy_relation=relation,
            confidence=confidence,
            source_memory_ids=[
                fact_a.get("memory_id", ""),
                fact_b.get("memory_id", ""),
            ],
            qa_pair_id=_hash_pair_id(content_a, content_b),
        )

    # ────────────────────────────────────────────────
    # 合成 QA → GaussianDAG Prior
    # ────────────────────────────────────────────────

    def to_prior_matrix(
        self,
        pairs: list[SynthesizedQAPair],
        n_memories: int,
    ) -> np.ndarray:
        """
        将合成的因果 QA 对转换为 GaussianDAG 的偏相关先验矩阵。

        P[i][j] = 因果置信度 (0=无先验, 1=强因果)

        Returns:
            np.ndarray, shape=(n_memories, n_memories)
        """
        prior = np.zeros((n_memories, n_memories), dtype=np.float32)

        for pair in pairs:
            if len(pair.source_memory_ids) < 2:
                continue
            cause_id = pair.source_memory_ids[0]
            effect_id = pair.source_memory_ids[1]
            # 使用哈希将 memory_id 映射到索引
            # (实际使用时由 GaussianDAG 负责对齐)
            ci = abs(hash(cause_id)) % n_memories
            ei = abs(hash(effect_id)) % n_memories
            prior[ci, ei] = max(prior[ci, ei], pair.confidence)

        return prior

    def run_pipeline(
        self, memories: list[dict]
    ) -> tuple[list[SynthesizedQAPair], np.ndarray]:
        """
        完整合成流水线。

        Returns:
            (pairs, prior_matrix)
        """
        facts = self.extract_facts(memories)
        pairs = self.synthesize_causal_pairs(facts)
        prior = self.to_prior_matrix(pairs, len(memories))
        return pairs, prior

    # ────────────────────────────────────────────────
    # v3.6.0 本地训练: 训练数据质量报告
    # ────────────────────────────────────────────────

    def training_data_report(
        self, pairs: list[SynthesizedQAPair]
    ) -> dict:
        """
        生成训练数据质量报告。

        Returns:
            {
                "total_pairs": int,
                "avg_confidence": float,
                "confidence_above_04": int,
                "energy_distribution": {"enhance": N, "suppress": M, ...},
                "reflection_depths": {1: N, 2: M, 3: K},
                "diversity_score": float,    # 五行覆盖均衡度
                "ready_for_training": bool,  # ≥ 3000 + confidence ≥ 0.4 + 五行均衡
            }
        """
        if not pairs:
            return {
                "total_pairs": 0,
                "avg_confidence": 0.0,
                "confidence_above_04": 0,
                "energy_distribution": {},
                "reflection_depths": {},
                "diversity_score": 0.0,
                "ready_for_training": False,
            }

        total = len(pairs)
        avg_conf = sum(p.confidence for p in pairs) / total
        above_threshold = sum(1 for p in pairs if p.confidence >= 0.4)

        # 能量分布
        energy_dist: dict[str, int] = defaultdict(int)
        for p in pairs:
            energy_dist[p.energy_relation] += 1

        # 反射深度分布
        depth_dist: dict[int, int] = defaultdict(int)
        for p in pairs:
            depth_dist[p.reflection_depth] += 1

        # 五行覆盖均衡度 (能量类型 5 种 × 每组至少 500 对才均衡)
        energy_types = {"enhance", "suppress", "same", "neutral", "reverse"}
        present_types = sum(1 for e in energy_types if energy_dist.get(e, 0) > 0)
        diversity_score = present_types / len(energy_types)

        # v3.6.0 本地训练就绪判断
        ready = (
            total >= 3000
            and avg_conf >= 0.4
            and diversity_score >= 0.6
            and above_threshold >= 2000
        )

        return {
            "total_pairs": total,
            "avg_confidence": round(avg_conf, 4),
            "confidence_above_04": above_threshold,
            "energy_distribution": dict(energy_dist),
            "reflection_depths": dict(depth_dist),
            "diversity_score": round(diversity_score, 3),
            "ready_for_training": ready,
        }

    # ────────────────────────────────────────────────
    # 内部工具方法
    # ────────────────────────────────────────────────

    def _group_by_energy(self, facts: list[dict]) -> dict[str, list[dict]]:
        """按能量类型分组。"""
        groups: dict[str, list[dict]] = defaultdict(list)
        for fact in facts:
            etype = fact.get("energy_type", "unknown")
            groups[etype].append(fact)
        return dict(groups)

    def _infer_energy_type_from_content(self, content: str) -> str:
        """
        从文本内容推断能量类型 (wood/fire/earth/metal/water)。

        基于关键词匹配:
        - wood (木): 生长、发展、扩展、春、东方
        - fire (火): 上涨、突破、创新、夏、南方
        - earth (土): 稳定、基础、支撑、季节交替期、中央
        - metal (金): 下跌、收缩、减少、秋、西方
        - water (水): 流动、变化、下降、冬、北方
        """
        content_lower = content.lower()

        keywords = {
            "wood": ["增长", "发展", "扩展", "创新", "生长", "研发", "突破"],
            "fire": ["上涨", "上升", "突破", "高温", "热量", "爆发", "峰值", "创"],
            "earth": ["稳定", "基础", "支撑", "维持", "平衡", "平台", "保障", "落地"],
            "metal": ["下降", "减少", "收缩", "下跌", "回落", "裁员", "削减", "缩减"],
            "water": ["流动", "变化", "波动", "下调", "下行", "融资", "资金", "流动"],
        }

        scores: dict[str, int] = defaultdict(int)
        for etype, kws in keywords.items():
            for kw in kws:
                if kw in content_lower:
                    scores[etype] += 1

        if scores:
            return max(scores, key=scores.get)
        return "unknown"

    def _get_enhanced_element(self, element: str) -> str | None:
        """获取 '生' 关系的目标元素。"""
        mapping = {
            "wood": "fire", "fire": "earth",
            "earth": "metal", "metal": "water",
            "water": "wood",
        }
        return mapping.get(element)

    def _get_suppressed_element(self, element: str) -> str | None:
        """获取 '克' 关系的目标元素。"""
        mapping = {
            "wood": "earth", "fire": "metal",
            "earth": "water", "metal": "wood",
            "water": "fire",
        }
        return mapping.get(element)

    def _classify_energy_relation(self, etype_a: str, etype_b: str) -> str:
        """分类两个能量类型之间的关系。"""
        if etype_a == etype_b:
            return "same"
        if self._get_enhanced_element(etype_a) == etype_b:
            return "enhance"
        if self._get_suppressed_element(etype_a) == etype_b:
            return "suppress"
        if self._get_enhanced_element(etype_b) == etype_a:
            return "reverse"
        return "neutral"

    @staticmethod
    def _sort_by_confidence(
        pairs: list[SynthesizedQAPair],
    ) -> list[SynthesizedQAPair]:
        return sorted(pairs, key=lambda p: p.confidence, reverse=True)


# ==========================================================================
# 工具函数
# ==========================================================================

def _hash_pair_id(cause: str, effect: str) -> str:
    """生成因果对的唯一哈希 ID。"""
    h = hashlib.sha256(f"{cause}:::{effect}".encode()).hexdigest()
    return f"qa_{h[:16]}"
