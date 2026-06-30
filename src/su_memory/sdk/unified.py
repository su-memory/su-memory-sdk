"""SuMemory — 统一语义记忆引擎 (单一产品线, 无 Lite/LitePro 区分).

v4.0 起, su-memory 取消免费/付费分级, 所有能力统一在 ``SuMemory`` 一个类:

- 向量语义检索 (FAISS HNSW + bge-m3, 原生 batch)
- TF-IDF 关键词检索 + 倒排索引
- 多跳推理 (三路融合 MultiHopReader + LLM reader 答案抽取)
- 因果推理 (BeliefNetwork + CausalDAG, algebra 层)
- 时空关联 (TemporalRing Z_60 + AffinityMatrix 5x5)
- 持久化 (SQLite 三级分层存储)

``SuMemoryLite`` / ``SuMemoryLitePro`` 保留为向后兼容别名, 内部均委托给
``SuMemory``, 不再有功能差异.
"""
from __future__ import annotations

from typing import Any

from .lite_pro import SuMemoryLitePro
from .multi_hop_reader import MultiHopReader

__all__ = ["SuMemory"]


class SuMemory(SuMemoryLitePro):
    """统一语义记忆引擎 (v4.0 单一产品线).

    继承 ``SuMemoryLitePro`` 的全部能力 (向量/图/因果/时空), 并内置
    ``MultiHopReader`` 提供多跳检索 + 答案抽取 (LLM reader 可选).

    Example:
        >>> from su_memory.sdk.unified import SuMemory
        >>> mem = SuMemory()
        >>> mem.add("暴雨导致水库水位暴涨")
        >>> res = mem.query_multihop("暴雨的后果", max_hops=2, top_k=3)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reader: MultiHopReader | None = None

    def _ensure_reader(self, use_llm: bool = True) -> MultiHopReader:
        """懒加载 MultiHopReader (需要 embedding 就绪).

        Parameters
        ----------
        use_llm : bool
            是否尝试加载本地 LLM reader (MLX Qwen) 做精确答案抽取.
            加载失败 (无模型/无 mlx_lm) 时静默回退启发式 reader, 不报错.
        """
        if self._reader is not None:
            return self._reader
        emb = self._ensure_embedding()

        def embed_fn(text: str):
            v = emb.encode(text)
            import numpy as np
            arr = np.asarray(v, dtype=np.float32)
            return arr.ravel() if arr.ndim > 1 else arr

        def embed_batch_fn(texts: list[str]):
            import numpy as np
            # 优先 native batch (sentence-transformers); 否则逐条
            try:
                arr = np.asarray(emb.encode(texts), dtype=np.float32)
                if arr.ndim == 2 and arr.shape[0] == len(texts):
                    return arr
            except Exception:
                pass
            return np.stack([embed_fn(t) for t in texts])

        llm_reader = None
        if use_llm:
            # 优先线上 API (能力远超本地 7B, DeepSeek 多跳强); 失败则本地 MLX; 再失败启发式
            try:
                from .api_reader import APIReader, probe_api
                if probe_api() is not None:
                    llm_reader = APIReader()
            except Exception:
                llm_reader = None
            if llm_reader is None:
                try:
                    from .llm_reader import LLMReader
                    llm_reader = LLMReader()
                except Exception:
                    llm_reader = None  # 无 LLM 时回退启发式
        self._reader = MultiHopReader(embed_fn, embed_batch_fn, llm_reader=llm_reader)
        return self._reader

    def query_multihop_reader(
        self,
        query: str,
        paragraphs: list[str] | None = None,
        top_k: int = 4,
    ) -> dict[str, Any]:
        """SOTA 级多跳检索 + 答案抽取 (三路融合 MultiHopReader).

        这是 ``query_multihop`` 的增强版, 用三路融合检索 (direct + title-bridge
        + entity-bridge) 召回证据段落, 并提供答案抽取. 在真实 HotpotQA 上
        答案 EM 取决于 reader 后端 (LLM reader 启用时为真实 EM 口径).

        Parameters
        ----------
        query : str
            问题.
        paragraphs : list[str], optional
            待检索段落. 默认用内存中全部记忆.
        top_k : int
            召回段落数.

        Returns
        -------
        dict
            {ranked_contents, answer_context, answer, top1, paths, bridge_entities}
        """
        reader = self._ensure_reader()
        if paragraphs is None:
            paragraphs = [n.content for n in self._memories]
        if not paragraphs:
            return {"ranked_contents": [], "answer": "", "paths": []}
        res = reader.retrieve_structured(query, paragraphs, top_k=top_k)
        answer = reader.extract_answer(query, res.answer_context)
        return {
            "ranked_contents": [paragraphs[i] for i in res.ranked_ids if i < len(paragraphs)],
            "answer_context": res.answer_context,
            "answer": answer,
            "top1": res.top1,
            "paths": res.path_used,
            "bridge_entities": res.bridge_entities,
        }

    def answer_question(self, query: str, paragraphs: list[str] | None = None,
                        gold_answer: str | None = None) -> dict[str, Any]:
        """问答: 多跳检索 + 答案抽取, 可选验证 EM.

        Parameters
        ----------
        gold_answer : str, optional
            若提供, 返回 em (是否命中).
        """
        result = self.query_multihop_reader(query, paragraphs)
        reader = self._ensure_reader()
        if gold_answer is not None:
            result["em"] = reader.answer_em(query, result["answer_context"], gold_answer)
        return result
