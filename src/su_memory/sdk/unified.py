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
import logging

logger = logging.getLogger(__name__)

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
            except Exception as e:
                logger.debug("降级处理: %s", e)
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

    # ============================================================
    # 因果推理纵深 (委托给 mci-world-model)
    # ============================================================
    # 设计原则 (方案 B): su-memory 是记忆引擎, 因果推理的*实现*属于
    # 专属的 mci-world-model 世界模型引擎。这里只保留*方法签名*,
    # 运行时优先委托 mci_world_model; 未安装时抛清晰 ImportError,
    # 绝不静默降级到假因果推理 (诚实优先)。
    #
    # 安装: pip install mci-world-model

    @staticmethod
    def _require_mci():
        """加载 mci-world-model; 缺失时抛带安装指引的 ImportError。"""
        try:
            import mci_world_model  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "因果推理能力由 mci-world-model 提供, 但未安装。\n"
                "安装: pip install mci-world-model\n"
                "或设置 PYTHONPATH 指向 mci-world-model/src"
            ) from e

    def discover_causal_structure(
        self,
        data: Any,
        var_names: list[str] | None = None,
        method: str = "auto",
    ) -> dict[str, Any]:
        """从观测数据学习因果结构 (七种发现算法, 委托 mci-world-model)。

        Parameters
        ----------
        data : np.ndarray | list[list[float]]
            观测矩阵, shape (n_samples, n_vars)。
        var_names : list[str], optional
            变量名。默认 ["x0", "x1", ...]。
        method : str
            "auto" / "pc" / "fci" / "ges" / "lingam" / "notears" / "golem" / "cam"。

        Returns
        -------
        dict
            {skeleton: {nodes, edges}, confidence, method_used, n_edges}

        Raises
        ------
        ImportError
            未安装 mci-world-model 时。
        """
        import numpy as np
        self._require_mci()
        X = np.asarray(data, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"data 必须是 2D 矩阵, 收到 shape={X.shape}")
        if var_names is None:
            var_names = [f"x{i}" for i in range(X.shape[1])]

        from mci_world_model.sdk import AutonomousLawDiscovererV2
        discoverer = AutonomousLawDiscovererV2()
        discoverer.discover_causal_structure(X, var_names)
        sk = discoverer.causal_structure
        return {
            "skeleton": {
                "nodes": list(sk.nodes),
                "edges": [
                    {
                        "cause": e[0] if isinstance(e, (tuple, list)) else e.cause,
                        "effect": e[1] if isinstance(e, (tuple, list)) else e.effect,
                    }
                    for e in sk.edges
                ],
            },
            "confidence": getattr(sk, "confidence", None),
            "method_used": "pc+symbolic",
            "n_edges": len(sk.edges),
        }

    def fit_sem(
        self,
        data: Any,
        var_names: list[str],
        edges: list[tuple[str, str]] | None = None,
        method: str = "auto",
    ) -> Any:
        """从数据拟合结构方程模型 (委托 mci-world-model)。

        Returns
        -------
        mci_world_model StructuralEquationModel

        Raises
        ------
        ImportError
            未安装 mci-world-model 时。
        """
        import numpy as np
        self._require_mci()
        from mci_world_model.sdk import CausalGraph, CounterfactualEngine
        X = np.asarray(data, dtype=np.float64)
        if edges is None:
            r = self.discover_causal_structure(X, var_names, method=method)
            edges = [(e["cause"], e["effect"]) for e in r["skeleton"]["edges"]]
        cg = CausalGraph(nodes=list(var_names), edges=[tuple(e) for e in edges])
        engine = CounterfactualEngine.from_causal_graph(cg)
        return engine.sem if engine else None

    def counterfactual_query(
        self,
        sem: Any,
        evidence: dict[str, float],
        interventions: dict[str, float],
        query_vars: list[str],
    ) -> dict[str, Any]:
        """Pearl 三步反事实推理 (委托 mci-world-model)。

        Parameters
        ----------
        sem : StructuralEquationModel
            由 fit_sem 构建。
        evidence : dict[str, float]
            观测事实。
        interventions : dict[str, float]
            do 操作的干预值。
        query_vars : list[str]
            要预测的变量 (取首个为 target)。

        Raises
        ------
        ImportError
            未安装 mci-world-model 时。
        """
        self._require_mci()
        from mci_world_model.sdk import CounterfactualEngine
        if len(query_vars) != 1:
            return {
                "error": "counterfactual_query 一次只预测一个 target (query_vars 取首个)",
                "evidence": evidence,
                "interventions": interventions,
            }
        target = query_vars[0]
        engine = CounterfactualEngine(sem, sem.node_names)
        result = engine.query(evidence, interventions, target)
        cf_val = getattr(result, "counterfactual_value", None)
        return {
            "counterfactual": {target: cf_val} if cf_val is not None else {},
            "evidence": evidence,
            "interventions": interventions,
            "pn": getattr(result, "pn", None),
            "ps": getattr(result, "ps", None),
            "pns": getattr(result, "pns", None),
        }
