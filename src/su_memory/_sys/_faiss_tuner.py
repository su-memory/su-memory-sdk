"""
su-memory FAISS 自动调参器

根据数据规模和硬件环境，自动选择最优 FAISS 索引参数：
- HNSW: M, efConstruction, efSearch
- IVF: nlist, nprobe
- 混合: nlist + HNSW M

使用方式:
    from su_memory._sys._faiss_tuner import FAISSAutoTuner
    tuner = FAISSAutoTuner(dims=768)
    index, params = tuner.build_index(n_vectors=10000)
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# 检测 FAISS
try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None
    np = None


@dataclass
class FAISSParams:
    """FAISS 索引参数"""
    index_type: str = "hnsw"       # "hnsw" | "ivf" | "auto"
    dims: int = 768
    # HNSW 参数
    hnsw_m: int = 32               # 每个节点的连接数 (16/32/64)
    hnsw_ef_construction: int = 200  # 构建时搜索范围
    hnsw_ef_search: int = 64       # 搜索时搜索范围
    # IVF 参数
    ivf_nlist: int = 100           # 聚类中心数
    ivf_nprobe: int = 10           # 搜索时探测的聚类数
    # 量化
    use_quantization: bool = False
    quantization_mode: str = "INT8"  # "INT8" | "FP16" | "FP32"


class FAISSAutoTuner:
    """FAISS 自动调参器

    算法:
    - HNSW 适合: N < 100K, 高召回 (>95%)
    - IVF  适合: N >= 10K, 高吞吐
    - 混合  适合: N >= 100K, 大容量 + 高召回
    """

    def __init__(self, dims: int = 768, preferred: str = "auto"):
        self.dims = dims
        self.preferred = preferred
        self._params = FAISSParams(dims=dims)

    def recommend(self, n_vectors: int = 1000) -> FAISSParams:
        """根据数据量推荐最佳参数"""
        p = FAISSParams(dims=self.dims)

        if self.preferred == "hnsw":
            p.index_type = "hnsw"
        elif self.preferred == "ivf":
            p.index_type = "ivf"
        else:
            # auto: 根据数据量选择
            if n_vectors >= 100000:
                p.index_type = "ivf_hnsw"  # 混合
            elif n_vectors >= 10000:
                p.index_type = "ivf"
            else:
                p.index_type = "hnsw"

        # HNSW 参数调优
        p.hnsw_m = self._recommend_m()
        p.hnsw_ef_construction = min(200, max(40, int(n_vectors * 0.1)))
        p.hnsw_ef_search = min(128, max(16, p.hnsw_ef_construction // 2))

        # IVF 参数调优
        p.ivf_nlist = self._recommend_nlist(n_vectors)
        p.ivf_nprobe = self._recommend_nprobe(p.ivf_nlist)

        # 量化：大向量集启用 INT8 量化
        if n_vectors >= 50000 and self.dims >= 512:
            p.use_quantization = True
            p.quantization_mode = "INT8"

        self._params = p
        logger.debug(
            f"[FAISSAutoTuner] N={n_vectors} dim={self.dims} → "
            f"type={p.index_type} M={p.hnsw_m} nlist={p.ivf_nlist}"
        )
        return p

    def build_index(
        self, n_vectors: int = 1000, train_vectors: Optional[np.ndarray] = None
    ) -> Tuple[Any, FAISSParams]:
        """构建并返回最优 FAISS 索引

        Args:
            n_vectors: 预期的向量数量
            train_vectors: 训练向量 (IVF 需要，HNSW 不需要)

        Returns:
            (faiss.Index, FAISSParams)
        """
        if not FAISS_AVAILABLE:
            return None, self._params

        p = self.recommend(n_vectors)

        try:
            if p.index_type == "hnsw":
                index = self._build_hnsw(p)
            elif p.index_type == "ivf_hnsw":
                index = self._build_ivf_hnsw(p, train_vectors, n_vectors)
            else:  # ivf
                index = self._build_ivf(p, train_vectors, n_vectors)

            return index, p
        except Exception as e:
            logger.warning(f"[FAISSAutoTuner] 构建失败 ({p.index_type}): {e}，回退到 HNSW")
            p.index_type = "hnsw"
            try:
                return self._build_hnsw(p), p
            except Exception:
                return None, p

    def _build_hnsw(self, p: FAISSParams):
        """构建 HNSW 索引"""
        index = faiss.IndexHNSWFlat(self.dims, p.hnsw_m)
        index.hnsw.efConstruction = p.hnsw_ef_construction
        index.hnsw.efSearch = p.hnsw_ef_search
        logger.info(
            f"[FAISSAutoTuner] HNSW: M={p.hnsw_m} "
            f"efConstruction={p.hnsw_ef_construction} efSearch={p.hnsw_ef_search}"
        )
        return index

    def _build_ivf(self, p: FAISSParams, train_vectors, n_vectors: int):
        """构建 IVF 索引"""
        quantizer = faiss.IndexFlatL2(self.dims)
        index = faiss.IndexIVFFlat(quantizer, self.dims, p.ivf_nlist)

        if train_vectors is not None and len(train_vectors) >= p.ivf_nlist:
            index.train(train_vectors)

        index.nprobe = p.ivf_nprobe
        logger.info(
            f"[FAISSAutoTuner] IVF: nlist={p.ivf_nlist} nprobe={p.ivf_nprobe}"
        )
        return index

    def _build_ivf_hnsw(self, p: FAISSParams, train_vectors, n_vectors: int):
        """构建 IVF+HNSW 混合索引 (大容量 + 高精度)"""
        quantizer = faiss.IndexHNSWFlat(self.dims, p.hnsw_m)
        index = faiss.IndexIVFFlat(quantizer, self.dims, p.ivf_nlist)

        if train_vectors is not None and len(train_vectors) >= p.ivf_nlist:
            index.train(train_vectors)

        index.nprobe = p.ivf_nprobe
        logger.info(
            f"[FAISSAutoTuner] IVF+HNSW: nlist={p.ivf_nlist} "
            f"nprobe={p.ivf_nprobe} M={p.hnsw_m}"
        )
        return index

    def _recommend_m(self) -> int:
        """根据维度推荐 HNSW M"""
        if self.dims <= 256:
            return 16
        elif self.dims <= 768:
            return 32
        else:
            return 64

    def _recommend_nlist(self, n_vectors: int) -> int:
        """根据向量数推荐 IVF nlist: 4 * sqrt(N)"""
        nlist = int(4 * math.sqrt(n_vectors))
        return max(4, min(4096, nlist))  # 4 <= nlist <= 4096

    def _recommend_nprobe(self, nlist: int) -> int:
        """根据 nlist 推荐 nprobe"""
        return max(1, min(64, int(math.sqrt(nlist))))

    @property
    def params(self) -> FAISSParams:
        return self._params
