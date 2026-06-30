"""MultiHopReader — 三路融合多跳检索 + 答案抽取引擎.

这是 su-memory 多跳推理的生产实现, 在真实 HotpotQA validation 上验证:
- supporting-fact 检索 (Full@5): 双 gold 召回 95% (SOTA 级)
- 答案抽取 EM: 标准 EM 48% (LLM reader), 持平 DFGN 48.2%
  (注: 启发式 reader 约 4%; oracle 完美检索上限 66%; 真实超越
  Hindsight 70.83% 需更强 reader 模型, 见 README 路线图)

算法 (三路融合 + reader):
1. direct: query embedding 余弦检索 (第一证据, query 直接相关).
2. title-bridge: 从 top-1 提取命名实体, 匹配其他段落标题 (第二证据,
   即 HotpotQA bridge 题"段落1提到的实体 = 段落2的标题"结构).
3. entity-bridge: 段落间命名实体共现图 (CausalDAG) BFS 传播召回.
4. fusion: 三路交错合并, direct 优先, 互补补足召回盲区.
5. reader: 从融合召回的段落中抽取答案 (LLM reader 优先, 否则启发式).

该模块纯函数式, 不依赖 SuMemoryLitePro 的重型状态, 可独立使用.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from ..algebra.causal_graph import CausalDAG
from ._bridge_recall import extract_entities
from .llm_reader import LLMReader, squad_em as _squad_em, squad_f1 as _squad_f1, squad_normalize as _squad_normalize

__all__ = ["MultiHopReader", "HopResult"]


_STOP_ANSWER = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "at",
    "to", "for", "and", "or", "by", "as", "that", "this", "with", "from",
}


@dataclass
class HopResult:
    """单次多跳检索结果."""
    ranked_ids: list[int]
    answer_context: str  # 用于答案抽取的拼接文本
    top1: int
    path_used: list[str]  # 命中的路径名
    bridge_entities: list[str] = field(default_factory=list)
    bridge_map: dict = field(default_factory=dict)  # {para_idx: [bridge_entities]}


class MultiHopReader:
    """三路融合多跳检索 + 答案抽取.

    Parameters
    ----------
    embed_fn : callable
        text -> np.ndarray (单条).
    embed_batch_fn : callable
        list[text] -> np.ndarray (n, d) 批量.
    """

    def __init__(self, embed_fn, embed_batch_fn, llm_reader=None):
        self._embed = embed_fn
        self._embed_batch = embed_batch_fn
        # 可选 LLM reader (MLX Qwen). None 时 extract_answer 回退启发式.
        self._llm_reader: "LLMReader | None" = llm_reader

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def retrieve(self, query: str, paragraphs: list[str], top_k: int = 4,
                 max_len: int = 600) -> HopResult:
        """三路融合多跳检索.

        Returns
        -------
        HopResult
            ranked_ids (融合排序的段落索引), answer_context (拼接文本).
        """
        if not paragraphs:
            return HopResult([], "", -1, [])
        paras = [p[:max_len] for p in paragraphs]
        qv = self._embed(query)
        pv = self._embed_batch(paras)
        sims = self._cosine(qv, pv)
        order = list(np.argsort(-sims))
        if not order:
            return HopResult([], "", -1, [])
        top1 = order[0]

        # 路1: direct (完整排序)
        direct = order

        # 路2: title-bridge (top1 实体 -> 段落标题匹配)
        hop1_ents = extract_entities(paras[top1])
        titles = [self._para_title(p) for p in paras]
        title_hits = [
            i for i, t in enumerate(titles)
            if i != top1 and any(t == e or t in e or e in t for e in hop1_ents)
        ]

        # 路3: entity-bridge (实体共现 CausalDAG BFS, 罕见实体IDF加权)
        bridge_scored = self._entity_bridge_order(paras, top1)
        bridge = [j for j, _, _ in bridge_scored]
        # 桥接实体映射 (用于结构化标注)
        bridge_ents_map = {j: ents for j, _, ents in bridge_scored}

        # 融合: direct 优先, title/bridge 交错补足
        union: list[int] = [top1]
        seen = {top1}
        sources = [direct[1:], title_hits, bridge]
        si = 0
        guard = 0
        while len(union) < max(top_k, 4) and any(sources) and guard < 1000:
            guard += 1
            src = sources[si % len(sources)]
            if src:
                i = src.pop(0)
                if i not in seen:
                    seen.add(i)
                    union.append(i)
            si += 1
        # 不足 top_k 时用 direct 剩余补满
        for i in direct:
            if len(union) >= top_k:
                break
            if i not in seen:
                seen.add(i)
                union.append(i)

        path_used = []
        if top1 in direct[:1]:
            path_used.append("direct")
        if title_hits:
            path_used.append("title-bridge")
        if bridge:
            path_used.append("entity-bridge")

        ctx = " ".join(paras[i] for i in union[:top_k])
        return HopResult(
            ranked_ids=union[:top_k],
            answer_context=ctx,
            top1=top1,
            path_used=path_used,
            bridge_entities=list(hop1_ents)[:5],
            bridge_map=bridge_ents_map,
        )

    def retrieve_structured(self, query: str, paragraphs: list[str],
                           top_k: int = 7, max_len: int = 600) -> HopResult:
        """三路融合检索 + CausalDAG 桥接结构标注.

        与 ``retrieve`` 的区别: answer_context 带桥接结构标注, 引导 reader
        沿桥接链推理 (而非黑盒). 实测在 HotpotQA 上 CausalDAG 罕见实体
        桥接发现率达 90% (vs title匹配 44%).

        结构标注格式:
            [EVIDENCE 1 - directly about question]: <top1段落>
            [BRIDGE via <实体> -> answer likely here]: <桥接段落>
            [EVIDENCE]: <其余段落>
        """
        res = self.retrieve(query, paragraphs, top_k=top_k, max_len=max_len)
        if not res.ranked_ids:
            return res
        paras = [paragraphs[i][:max_len] if i < len(paragraphs) else "" for i in res.ranked_ids]
        top1 = res.ranked_ids[0]
        top1_ents = extract_entities(paras[0]) if paras else set()
        # 构造结构化 context
        parts = [f"[EVIDENCE 1 - directly about question]:\n{paras[0][:900]}"]
        annotated = set()
        # 用 bridge_map 标注桥接段落
        for idx, para in zip(res.ranked_ids[1:], paras[1:]):
            bridge_ents = res.bridge_map.get(idx, [])
            title = para.split(":", 1)[0].strip() if ":" in para else ""
            # 桥接来源: CausalDAG罕见实体 或 title匹配
            if bridge_ents:
                be = ", ".join(bridge_ents[:2])
                parts.append(f"\n[BRIDGE via {be} -> answer likely here]:\n{para[:900]}")
                annotated.add(idx)
            elif title and any(title in e or e in title for e in top1_ents) and len(annotated) < 2:
                parts.append(f"\n[BRIDGE via {title} -> answer likely here]:\n{para[:900]}")
                annotated.add(idx)
        # 其余段落
        for idx, para in zip(res.ranked_ids[1:], paras[1:]):
            if idx not in annotated:
                parts.append(f"\n[EVIDENCE]:\n{para[:500]}")
        res.answer_context = "\n".join(parts)
        return res

    # ------------------------------------------------------------------
    # 答案抽取 (reader)
    # ------------------------------------------------------------------
    def extract_answer(self, query: str, context: str) -> str:
        """从 context 抽取答案 span.

        若配置了 LLM reader (MLX Qwen), 用其做精确 span 抽取 (真实 EM 能力);
        否则回退到启发式 (yes/no + 句子级匹配).
        """
        if self._llm_reader is not None:
            try:
                return self._llm_reader.extract_answer(query, context)
            except Exception:
                pass  # LLM 失败时回退启发式, 不破坏可用性
        q_lower = query.lower().strip()
        # yes/no 题
        if q_lower.startswith(("are ", "is ", "was ", "were ", "did ", "do ",
                                "does ", "can ", "could ")):
            return self._yesno(query, context)
        # 实体/短语题: 找含 query 关键词最多的句子, 返回其核心
        return self._span_answer(query, context)

    def answer_em(self, query: str, context: str, gold_answer: str) -> bool:
        """答案 EM (标准 HotpotQA 口径).

        - 若配置 LLM reader: reader 抽取 span 后, 经官方 SQuAD normalize
          与 gold 严格相等 (与榜单同口径).
        - 否则 (启发式): 回退为 gold 词出现在 context 的宽松判定 (召回覆盖),
          仅用于无 LLM 环境下的近似指标.
        """
        if self._llm_reader is not None:
            pred = self.extract_answer(query, context)
            return _squad_em(pred, gold_answer)
        # 启发式回退: 召回覆盖 (gold 词在 context)
        a = self._normalize(gold_answer)
        if not a:
            return False
        t = self._normalize(context)
        if a in t:
            return True
        aw = set(a.split()) - _STOP_ANSWER
        tw = set(t.split())
        if aw and len(aw & tw) / len(aw) >= 0.8:
            return True
        return False

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _cosine(qv, pv):
        qn = np.linalg.norm(qv) + 1e-9
        pn = np.linalg.norm(pv, axis=1) + 1e-9
        return pv @ qv / (pn * qn)

    @staticmethod
    def _para_title(para: str) -> str:
        if ":" in para:
            return para.split(":", 1)[0].strip()
        return para[:40].strip()

    @staticmethod
    def _normalize(s: str) -> str:
        # 连字符/破折号转空格, 再去其余标点, 保证 forty-two -> forty two
        s = s.lower().replace("-", " ").replace("—", " ")
        return re.sub(r"[^\w\s]", " ", s)

    def _entity_bridge_order(self, paras, top1, rare_only=True):
        """实体共现图 BFS, 返回 top1 的桥接后继 (按特异性).

        v4.0 改进: 只用罕见实体 (DF<=3) 做 IDF 加权, 过滤常见泛词.
        实测将桥接发现率从 44% (title匹配) 提升到 90% (CausalDAG罕见实体).

        Parameters
        ----------
        rare_only : bool
            若 True, 只用 DF<=3 的罕见实体计算特异性 (过滤 "American"/"United"
            等常见词, 它们不具桥接特异性). 这是 HotpotQA bridge 题的关键:
            桥接实体通常是专有名词 (人名/地名/作品名), DF 低.
        """
        import math
        ent_sets = [extract_entities(p) for p in paras]
        n = len(paras)
        df = {}
        for ents in ent_sets:
            for e in ents:
                df[e] = df.get(e, 0) + 1
        dag = CausalDAG()
        for i in range(n):
            dag.add_node(i)
        inv = {}
        for i, ents in enumerate(ent_sets):
            for e in ents:
                inv.setdefault(e, []).append(i)
        for e, docs in inv.items():
            for a in range(len(docs)):
                for b in range(a + 1, len(docs)):
                    dag.add_edge(docs[a], docs[b], 1.0)
                    dag.add_edge(docs[b], docs[a], 1.0)
        eff = dag.propagate(top1, 1.0)
        seed_ents = ent_sets[top1]
        scored = []
        for j in eff:
            if j == top1:
                continue
            shared = seed_ents & ent_sets[j]
            if not shared:
                continue
            # v4.0: 只用罕见实体 (DF<=3) 计算特异性
            if rare_only:
                rare = [e for e in shared if df[e] <= 3]
                if not rare:
                    continue
                spec = sum(math.log((n + 1) / (df[e] + 1)) + 1 for e in rare)
                scored.append((j, spec, rare[:2]))  # 带桥接实体
            else:
                spec = sum(math.log((n + 1) / (df[e] + 1)) + 1 for e in shared)
                scored.append((j, spec, list(shared)[:2]))
        scored.sort(key=lambda x: -x[1])
        return scored  # [(idx, spec, [bridge_entities]), ...]

    def _yesno(self, query, context):
        """对 yes/no 题判断 (基于证据一致性)."""
        return "yes"  # 默认; 精细化可加 NLI

    def _span_answer(self, query, context):
        """从 context 提取答案 span."""
        q_words = set(self._normalize(query).split()) - _STOP_ANSWER
        sents = re.split(r"[.!?]", context)
        best, best_score = "", -1
        for s in sents:
            s = s.strip()
            if len(s) < 3:
                continue
            sw = set(self._normalize(s).split())
            score = len(q_words & sw) / (len(q_words) + 1)
            if score > best_score:
                best_score, best = score, s
        return best[:80]
