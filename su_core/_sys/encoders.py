"""
64卦全息编码系统

六爻编码 + 互卦/综卦/错卦 + 全息检索接口
语义向量投影 → 四位一体多维空间

对外暴露：SemanticEncoder, EncoderCore
内部实现：完全封装，不对外透露
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import hashlib
import math


# ========================
# 64卦名称表（0-63）
# ========================

HEXAGRAM_NAMES = [
    "乾", "坤", "屯", "蒙", "需", "讼", "师", "比",
    "小畜", "履", "泰", "否", "同人", "大有", "谦", "豫",
    "随", "蛊", "临", "观", "噬嗑", "贲", "剥", "复",
    "无妄", "大畜", "颐", "大过", "坎", "离", "咸", "恒",
    "遁", "大壮", "晋", "明夷", "家人", "睽", "蹇", "解",
    "损", "益", "夬", "姤", "萃", "升", "困", "井",
    "革", "鼎", "震", "艮", "渐", "归妹", "丰", "旅",
    "巽", "兑", "涣", "节", "中孚", "小过", "既济", "未济"
]

# 上下卦查表（用于计算互卦/综卦/错卦）
HEXAGRAM_TRIGRAMS_BELOW = [
    0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 1, 1, 2, 3,
    4, 5, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 3, 3,
    4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 2, 2,
    2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7, 0, 0
]

HEXAGRAM_TRIGRAMS_ABOVE = [
    0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
    0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
    0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
    0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7
]

# 五行归属表
HEXAGRAM_WUXING = [
    "金", "土", "水", "水", "水", "火", "水", "水",
    "木", "金", "木", "土", "火", "火", "土", "土",
    "金", "土", "土", "金", "火", "火", "土", "土",
    "木", "金", "土", "金", "水", "火", "土", "木",
    "金", "金", "火", "火", "木", "火", "金", "水",
    "土", "木", "金", "金", "金", "木", "火", "水",
    "火", "火", "木", "土", "木", "木", "火", "火",
    "木", "金", "水", "水", "木", "火", "火", "水"
]

# 方位表
HEXAGRAM_DIRECTION = [
    "西北", "西南", "北", "北", "北", "南", "北", "北",
    "东南", "东南", "西北", "东南", "南", "南", "东北", "北",
    "东北", "西南", "东", "西南", "南", "南", "东北", "北",
    "东北", "西北", "东北", "西北", "北", "南", "南", "东",
    "西北", "东", "南", "西", "南", "东", "北", "东",
    "东北", "东南", "西", "西北", "西", "西南", "北", "东南",
    "西北", "东南", "东", "东北", "东南", "东北", "东", "南",
    "东南", "西", "南", "北", "东南", "南", "东", "北"
]

# ========================
# 先天八卦序相关常量
# ========================

# 八卦名称（先天八卦序）
BAGUA_NAMES = ["乾", "兑", "离", "震", "巽", "坎", "艮", "坤"]

# 八卦→五行映射（用于概率聚合）
BAGUA_TO_WUXING_MAP = {
    0: "金", 1: "金",   # 乾、兑
    2: "火",             # 离
    3: "木", 4: "木",   # 震、巽
    5: "水",             # 坎
    6: "土", 7: "土",   # 艮、坤
}

WUXING_NAMES = ["金", "木", "水", "火", "土"]


# ========================
# Ollama 本地语义编码器（离线优先）
# ========================
_st_model = None


class _OllamaEncoder:
    """包装 Ollama bge-m3 为 sentence-transformers 兼容接口"""

    def __init__(self):
        self.dim = 1024

    def encode(self, texts, **kwargs):
        """返回与 sentence-transformers 兼容的结果对象"""
        import urllib.request, json
        if isinstance(texts, str):
            texts = [texts]
        results = []
        for text in texts:
            req = urllib.request.Request(
                "http://localhost:11434/api/embeddings",
                data=json.dumps({"model": "bge-m3", "prompt": text}).encode(),
                headers={"Content-Type": "application/json"}
            )
            resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
            results.append(resp["embedding"])

        class _Result:
            def __init__(self, data):
                import numpy as np
                self.data = np.array(data)

            def tolist(self):
                arr = self.data
                if arr.ndim == 2:
                    arr = arr[0]
                return arr.tolist()  # already flat

        return _Result(results)


def _get_st_model():
    """加载多语言语义模型，优先 Ollama bge-m3（完全离线）"""
    global _st_model
    if _st_model is not None:
        return _st_model
    # 优先使用本地 Ollama bge-m3（完全离线）
    try:
        import urllib.request, json
        req = urllib.request.Request(
            "http://localhost:11434/api/embeddings",
            data=json.dumps({"model": "bge-m3", "prompt": "test"}).encode(),
            headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=5)
        _st_model = _OllamaEncoder()
        return _st_model
    except Exception:
        pass
    # Fallback: 尝试 HuggingFace paraphrase-multilingual-MiniLM-L12-v2
    try:
        from sentence_transformers import SentenceTransformer
        from huggingface_hub import try_to_load_from_cache
        path = try_to_load_from_cache(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "config.json")
        if path:
            _st_model = SentenceTransformer(os.path.dirname(path))
            return _st_model
        _st_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return _st_model
    except Exception:
        _st_model = None
        return None



def _softmax(values):
    """数值稳定的 softmax"""
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def _cosine_similarity(a, b):
    """计算两个向量的 cosine similarity"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1e-9
    norm_b = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (norm_a * norm_b)


def _cosine_similarity_dict(d1, d2):
    """计算两个字典形式概率分布的 cosine similarity"""
    keys = sorted(set(d1.keys()) | set(d2.keys()))
    v1 = [d1.get(k, 0.0) for k in keys]
    v2 = [d2.get(k, 0.0) for k in keys]
    return _cosine_similarity(v1, v2)


# ========================
# 公开接口：六十四卦系统
# ========================

@dataclass
class EncodingInfo:
    """卦象信息（对外暴露的最小信息单元）"""
    index: int              # 0-63
    name: str               # 卦名
    ben_gua: int            # 本卦索引
    hu_gua: int             # 互卦索引
    zong_gua: int           # 综卦索引
    cuo_gua: int            # 错卦索引
    wuxing: str             # 五行归属
    direction: str          # 方位
    # 新增语义扩展字段
    semantic_vector: Optional[List[float]] = None
    bagua_probs: Optional[Dict[str, float]] = None
    wuxing_scores: Optional[Dict[str, float]] = None

    @classmethod
    def from_index(cls, index):
        """从索引创建卦象信息"""
        return cls(
            index=index,
            name=HEXAGRAM_NAMES[index],
            ben_gua=index,
            hu_gua=_compute_hu_gua(index),
            zong_gua=_compute_zong_gua(index),
            cuo_gua=_compute_cuo_gua(index),
            wuxing=HEXAGRAM_WUXING[index],
            direction=HEXAGRAM_DIRECTION[index]
        )


# ========================
# 内部计算函数（不对外暴露）
# ========================

def _compute_hu_gua(index):
    """计算互卦（下卦的二三爻+上卦的一二爻）"""
    below = HEXAGRAM_TRIGRAMS_BELOW[index]
    above = HEXAGRAM_TRIGRAMS_ABOVE[index]
    new_below = (below >> 1) & 0x03
    new_above = (above & 0x01) | ((above >> 1) & 0x02)
    return _find_hexagram(new_below, new_above)


def _compute_zong_gua(index):
    """计算综卦（上下卦互换）"""
    below = HEXAGRAM_TRIGRAMS_BELOW[index]
    above = HEXAGRAM_TRIGRAMS_ABOVE[index]
    return _find_hexagram(above, below)


def _compute_cuo_gua(index):
    """计算错卦（阴阳全反）"""
    return 63 - index


def _find_hexagram(below, above):
    """根据上下卦找对应卦象索引"""
    for i in range(64):
        if HEXAGRAM_TRIGRAMS_BELOW[i] == below and HEXAGRAM_TRIGRAMS_ABOVE[i] == above:
            return i
    return 0


def _vector_to_bagua_probs(vector):
    """
    将语义向量投影到八卦概率分布
    - 输入：1024维 Ollama bge-m3 或 384维 sentence-transformers
    - 取前min(16,len(vector))维做均匀分桶 fold，mean 后 softmax 得到八卦概率
    - 16维 fold 成 8 卦：每 2 维一组，每组求 mean，再 softmax
    """
    k = min(16, len(vector))
    v = list(vector[:k])
    bagua_raw = [0.0] * 8
    for i in range(8):
        start = i * 2
        end = start + 2
        bucket = v[start:end] if end <= k else v[start:]
        bagua_raw[i] = sum(bucket) / len(bucket) if bucket else 0.0
    probs = _softmax(bagua_raw)
    return {BAGUA_NAMES[i]: probs[i] for i in range(8)}


def _bagua_probs_to_wuxing_scores(bagua_probs):
    """
    将八卦概率聚合到五行得分
    乾兑→金, 离→火, 震巽→木, 坎→水, 艮坤→土
    """
    wuxing_scores = {"金": 0.0, "木": 0.0, "水": 0.0, "火": 0.0, "土": 0.0}
    mapping = {
        "乾": "金", "兑": "金",
        "离": "火",
        "震": "木", "巽": "木",
        "坎": "水",
        "艮": "土", "坤": "土",
    }
    for gua_name, wx in mapping.items():
        wuxing_scores[wx] += bagua_probs.get(gua_name, 0.0)
    return wuxing_scores


def _bagua_probs_to_index(bagua_probs):
    """
    从八卦概率分布取 top-2 作为上下卦，映射到64卦索引
    概率最高 → 下卦，次高 → 上卦
    """
    sorted_bagua = sorted(bagua_probs.items(), key=lambda x: x[1], reverse=True)
    top1_name = sorted_bagua[0][0]  # 下卦
    top2_name = sorted_bagua[1][0]  # 上卦

    top1_idx = BAGUA_NAMES.index(top1_name)
    top2_idx = BAGUA_NAMES.index(top2_name)

    return _find_hexagram(top1_idx, top2_idx)


# ========================
# 对外接口类
# ========================

class SemanticEncoder:
    """
    六爻编码器 - 对外唯一接口

    功能：
    1. 文本 → 卦象编码（语义投影）
    2. 卦象信息查询
    3. 全息四卦计算

    对外隐藏：编码算法细节、映射表
    """

    def __init__(self):
        self._cache = {}
        self._model = None
        self._model_loaded = False

    def _ensure_model(self):
        """延迟加载模型"""
        if not self._model_loaded:
            self._model = _get_st_model()
            self._model_loaded = True
        return self._model

    def encode(self, content, memory_type="fact"):
        """
        将文本内容编码为卦象

        Args:
            content: 原始文本
            memory_type: 记忆类型（影响编码）

        Returns:
            EncodingInfo: 包含本卦/互卦/综卦/错卦的完整信息
        """
        model = self._ensure_model()

        if model is not None:
            return self._encode_semantic(content, memory_type, model)
        else:
            index = self._content_to_index_hash(content, memory_type)
            return EncodingInfo.from_index(index)

    def encode_with_vector(self, content, memory_type="fact"):
        """
        编码并返回原始语义向量

        Returns:
            (EncodingInfo, vector) 元组
        """
        model = self._ensure_model()

        if model is not None:
            result = model.encode(f"{content} [{memory_type}]")
            if hasattr(result, 'tolist'):
                vector = result.tolist()
            else:
                vector = result
            if isinstance(vector, list) and vector and isinstance(vector[0], list):
                vector = vector[0]
            info = self._build_info_from_vector(vector)
            info.semantic_vector = vector  # 确保存储完整向量
            return info, vector
        else:
            index = self._content_to_index_hash(content, memory_type)
            return EncodingInfo.from_index(index), None

    def _encode_semantic(self, content, memory_type, model):
        """使用 sentence-transformers 进行语义编码"""
        text = f"{content} [{memory_type}]"
        result = model.encode(text)
        # 确保 1D vector（sentence-transformers 返回 2D，Ollama 返回 1D）
        if hasattr(result, 'tolist'):
            vector = result.tolist()
        else:
            vector = result
        if isinstance(vector, list) and vector and isinstance(vector[0], list):
            vector = vector[0]
        info = self._build_info_from_vector(vector)
        info.semantic_vector = vector  # 存储完整向量供全息检索使用
        return info

    def _build_info_from_vector(self, vector):
        """从语义向量构建 EncodingInfo"""
        bagua_probs = _vector_to_bagua_probs(vector)
        wuxing_scores = _bagua_probs_to_wuxing_scores(bagua_probs)
        index = _bagua_probs_to_index(bagua_probs)

        info = EncodingInfo.from_index(index)
        info.semantic_vector = vector
        info.bagua_probs = bagua_probs
        info.wuxing_scores = wuxing_scores
        return info

    def _content_to_index_hash(self, content, memory_type):
        """内部：内容→索引映射（hash fallback）"""
        seed = f"{content}:{memory_type}"
        hash_val = int(hashlib.md5(seed.encode()).hexdigest(), 16)
        return hash_val % 64

    def batch_encode(self, contents):
        """
        批量编码

        Args:
            contents: [{"content": "...", "type": "fact"}]
        """
        return [self.encode(item["content"], item.get("type", "fact")) for item in contents]


class EncoderCore:
    """
    64卦全息系统 - 提供四卦视角查询

    对外接口，隐藏内部实现细节
    """

    def __init__(self):
        self.coder = SemanticEncoder()

    def get_holographic_views(self, hexagram_index):
        """
        获取某卦的全息四视图

        Returns:
            {"本卦": idx, "互卦": idx, "综卦": idx, "错卦": idx}
        """
        info = EncodingInfo.from_index(hexagram_index)
        return {
            "本卦": info.ben_gua,
            "互卦": info.hu_gua,
            "综卦": info.zong_gua,
            "错卦": info.cuo_gua
        }

    def retrieve_holographic(
        self,
        query_index,
        candidate_indices,
        top_k=8,
        query_info=None,
        candidate_infos=None,
        use_vector_sim=False
    ):
        """
        全息检索入口

        Args:
            use_vector_sim: True 时直接用 full-vector cosine 做精确语义排序
                            （每条记忆独立评分，不按卦象去重）
                            False 时使用卦象四视图融合评分（同一卦象只保留最高分）
        """
        """
        全息检索：在候选集合中找与查询卦象全息相关的记忆

        支持连续语义距离得分（当提供 bagua_probs 时）和结构匹配。

        Returns:
            [(index, score), ...] 按得分降序
        """
        query_views = self.get_holographic_views(query_index)
        query_bagua_vec = None
        query_wuxing_vec = None

        if query_info and query_info.bagua_probs:
            query_bagua_vec = query_info.bagua_probs
        if query_info and query_info.wuxing_scores:
            query_wuxing_vec = query_info.wuxing_scores

        scored = []
        seen = {}
        for cand in candidate_indices:
            cand_views = self.get_holographic_views(cand)

            cand_bagua_vec = None
            cand_wuxing_vec = None
            if candidate_infos and cand in candidate_infos:
                ci = candidate_infos[cand]
                if ci.bagua_probs:
                    cand_bagua_vec = ci.bagua_probs
                if ci.wuxing_scores:
                    cand_wuxing_vec = ci.wuxing_scores

            q_vec = query_info.semantic_vector if query_info else None
            c_vec = candidate_infos[cand].semantic_vector if (candidate_infos and cand in candidate_infos) else None

            if use_vector_sim and q_vec is not None and c_vec is not None:
                # 全量向量 cosine（最精确的语义检索，保留每个候选独立评分）
                score = _cosine_similarity(q_vec, c_vec)
            else:
                score = self._compute_holographic_score(
                    query_index, cand, query_views, cand_views,
                    query_bagua_vec, cand_bagua_vec,
                    query_wuxing_vec, cand_wuxing_vec,
                    q_vec, c_vec
                )

            if score > 0:
                if not use_vector_sim:
                    # 卦象索引去重：同卦保留最高分
                    if cand not in seen or score > seen[cand]:
                        seen[cand] = score
                else:
                    # 全量向量模式：每个候选独立评分
                    scored.append((cand, score))

        if not use_vector_sim:
            scored = list(seen.items())

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _compute_holographic_score(
        self,
        query_idx, cand_idx,
        query_views, cand_views,
        query_bagua, cand_bagua,
        query_wuxing, cand_wuxing,
        query_vector=None, cand_vector=None
    ):
        """
        计算全息匹配得分 — 三层融合

        Layer 1 (权重0.15): 64卦结构匹配（互卦/综卦/错卦加成）
        Layer 2 (权重0.35): 八卦概率分布 cosine similarity
        Layer 3 (权重0.50): 五行得分 cosine similarity

        当有 full semantic_vector 时，Layer 3 直接用向量 cosine 替代。
        """
        hu_bonus = 0.0
        if cand_views["互卦"] == query_views["本卦"]:
            hu_bonus = 0.15
        elif cand_views["本卦"] == query_views["互卦"]:
            hu_bonus = 0.10

        zong_bonus = 0.0
        if cand_views["综卦"] == query_views["本卦"]:
            zong_bonus = 0.10

        cuo_bonus = 0.0
        if cand_views["错卦"] == query_views["错卦"]:
            cuo_bonus = 0.05

        structure_score = hu_bonus + zong_bonus + cuo_bonus

        # ---- 连续化模式（有 bagua_probs）----
        if query_bagua and cand_bagua:
            bagua_sim = _cosine_similarity_dict(query_bagua, cand_bagua)

            # Full-vector cosine similarity (最高权重，最精确的语义检索)
            if query_vector and cand_vector:
                vec_sim = _cosine_similarity(query_vector, cand_vector)
                # 归一化到 [0,1]
                vec_sim_norm = (vec_sim + 1.0) / 2.0
                total = vec_sim_norm * 0.75 + bagua_sim * 0.15 + structure_score * 0.10
                return min(total, 1.0)

            wuxing_bonus = 0.0
            if query_wuxing and cand_wuxing:
                wuxing_sim = _cosine_similarity_dict(query_wuxing, cand_wuxing)
                wuxing_bonus = wuxing_sim * 0.35

            total = bagua_sim * 0.50 + wuxing_bonus + structure_score * 0.15
            return min(total, 1.0)

        # ---- 离散 fallback ----
        score = 0.0
        cand_info = EncodingInfo.from_index(cand_idx)
        query_info_fb = EncodingInfo.from_index(query_idx)

        if cand_idx == query_idx:
            score += 0.40
        q_below = HEXAGRAM_TRIGRAMS_BELOW[query_idx]
        q_above = HEXAGRAM_TRIGRAMS_ABOVE[query_idx]
        c_below = HEXAGRAM_TRIGRAMS_BELOW[cand_idx]
        c_above = HEXAGRAM_TRIGRAMS_ABOVE[cand_idx]
        if q_below == c_below:
            score += 0.10
        if q_above == c_above:
            score += 0.10
        if cand_info.wuxing == query_info_fb.wuxing:
            score += 0.08
        if cand_info.direction == query_info_fb.direction:
            score += 0.02

        return min(score, 1.0)
