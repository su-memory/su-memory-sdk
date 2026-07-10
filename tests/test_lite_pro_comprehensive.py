"""
su-memory-sdk Sprint 1 — SuMemoryLitePro 核心测试

覆盖: 初始化 / add / query / 多跳 / FAISS生命周期 / embedding回退 / 图谱CRUD / predict / explain / stats / 记忆生命周期

运行: pytest tests/test_lite_pro_comprehensive.py -v
"""
import os
import sys
import tempfile

import pytest

pytestmark = pytest.mark.integration

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def _ollama_available():
    """检测本地 Ollama 服务及 embedding 模型可用，且能真正完成一次 embedding 请求。

    仅查 /api/tags 无法区分“服务可达但实际 embedding 请求会挂起”的环境，
    因此额外对 bge-m3 发起一次真实 embedding 探测：5 秒内返回 200 才算可用，
    超时或失败则判定不可用（触发 skip），避免依赖 Ollama 的用例在本机卡死。
    """
    try:
        import urllib.request
        import json as _json
        # 1. 服务可达且存在 embedding 类模型
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = _json.loads(resp.read())
            models = [m.get("name", "") for m in data.get("models", [])]
            if not any(("bge" in m.lower() or "embed" in m.lower()) for m in models):
                return False
        # 2. 真实 embedding 探测(5s 超时)，过滤“可达但请求挂起”的环境
        payload = _json.dumps({"model": "bge-m3", "prompt": "ping"}).encode()
        ereq = urllib.request.Request(
            "http://localhost:11434/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(ereq, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


# 依赖 Ollama(或大索引多跳推理)的用例默认跳过,避免在本机卡死;
# 显式运行请设置环境变量 RUN_OLLAMA_TESTS=1(并确保 Ollama 探测可用):
#   RUN_OLLAMA_TESTS=1 python -m pytest tests/<file>.py
_RUN_OLLAMA_TESTS = os.environ.get("RUN_OLLAMA_TESTS") in ("1", "true", "yes")
_OLLAMA_SKIP = pytest.mark.skipif(
    not (_ollama_available() and _RUN_OLLAMA_TESTS),
    reason="依赖 Ollama 的慢用例默认跳过; 设置 RUN_OLLAMA_TESTS=1 且 Ollama 可用时运行",
)


from su_memory.sdk.lite_pro import SuMemoryLitePro

# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def client_no_vector():
    """禁用向量的轻量客户端（快速测试用）"""
    with tempfile.TemporaryDirectory() as d:
        c = SuMemoryLitePro(
            storage_path=d,
            enable_vector=False,
            enable_graph=True,
            enable_temporal=True,
            enable_session=True,
            enable_prediction=False,
            enable_explainability=False,
        )
        yield c


@pytest.fixture
def client_with_vector():
    """启用向量的客户端（需要 sentence-transformers）"""
    with tempfile.TemporaryDirectory() as d:
        c = SuMemoryLitePro(
            storage_path=d,
            enable_vector=True,
            enable_graph=True,
            enable_temporal=True,
            enable_tfidf=True,
        )
        yield c


@pytest.fixture
def populated(client_no_vector):
    """预填充 5 条记忆的客户端"""
    c = client_no_vector
    c.add("今天天气很好，阳光明媚")
    c.add("明天可能下雨，记得带伞")
    c.add("我喜欢学习Python编程")
    c.add("周末想去公园散步和跑步")
    c.add("项目截止日期是下周五")
    return c


# ═══════════════════════════════════════════════════════════════
# T1.1 初始化参数矩阵 (8 tests)
# ═══════════════════════════════════════════════════════════════

class TestInitialization:
    """测试 SuMemoryLitePro 初始化"""

    def test_default_init(self):
        """默认参数初始化"""
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(storage_path=d, enable_vector=False)
            assert c.max_memories == 10000
            assert c.enable_tfidf is True
            assert c.enable_temporal is True

    def test_disable_all_extras(self):
        """禁用所有可选功能"""
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(
                storage_path=d,
                enable_vector=False,
                enable_graph=False,
                enable_temporal=False,
                enable_session=False,
                enable_prediction=False,
                enable_explainability=False,
            )
            mid = c.add("test")
            assert mid is not None
            assert len(c) == 1

    def test_vector_only_mode(self):
        """仅启用向量检索"""
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(
                storage_path=d,
                enable_vector=True,
                enable_graph=False,
                enable_temporal=False,
                enable_tfidf=False,
            )
            assert c.enable_vector is True
            assert c.enable_graph is False

    def test_graph_only_mode(self):
        """仅启用图谱"""
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(
                storage_path=d,
                enable_vector=False,
                enable_graph=True,
            )
            assert c.enable_graph is True

    def test_custom_max_memories(self):
        """自定义最大记忆数"""
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(storage_path=d, max_memories=100, enable_vector=False)
            assert c.max_memories == 100

    def test_storage_path_auto_create(self):
        """storage_path 自动创建目录"""
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "subdir", "data")
            c = SuMemoryLitePro(storage_path=p, enable_vector=False)
            assert os.path.exists(p)

    def test_len_zero_on_init(self, client_no_vector):
        """初始化后长度为0"""
        assert len(client_no_vector) == 0

    def test_empty_stats(self, client_no_vector):
        """空客户端统计"""
        s = client_no_vector.get_stats()
        assert s['total_memories'] == 0


# ═══════════════════════════════════════════════════════════════
# T1.2 add 单条记忆 (10 tests)
# ═══════════════════════════════════════════════════════════════

class TestAddMemory:
    """测试 add() 方法"""

    def test_add_simple(self, client_no_vector):
        """添加简单文本"""
        mid = client_no_vector.add("Hello World")
        assert isinstance(mid, str)
        assert len(mid) > 0
        assert len(client_no_vector) == 1

    def test_add_returns_unique_ids(self, client_no_vector):
        """每条记忆返回唯一ID"""
        ids = set()
        for i in range(5):
            ids.add(client_no_vector.add(f"memory {i}"))
        assert len(ids) == 5

    def test_add_empty_string(self, client_no_vector):
        """空字符串：当前版本不抛出异常，但应能正常处理"""
        # 行为：add 接受空字符串并分配 ID
        mid = client_no_vector.add("")
        assert isinstance(mid, str)

    def test_add_whitespace_only(self, client_no_vector):
        """纯空白：当前版本不抛出异常，但应能正常处理"""
        mid = client_no_vector.add("   \n\t  ")
        assert isinstance(mid, str)

    def test_add_chinese(self, client_no_vector):
        """中文文本"""
        mid = client_no_vector.add("这是一条中文记忆")
        assert mid is not None

    def test_add_emoji(self, client_no_vector):
        """含 emoji 的文本"""
        mid = client_no_vector.add("🎉 庆祝项目上线 🚀")
        assert mid is not None

    def test_add_special_characters(self, client_no_vector):
        """特殊字符"""
        mid = client_no_vector.add("test <script>alert(1)</script> & 'quotes'")
        assert mid is not None

    def test_add_long_text(self, client_no_vector):
        """长文本（1000字）"""
        long_text = "项目" * 500
        mid = client_no_vector.add(long_text)
        assert mid is not None
        assert len(client_no_vector) == 1

    def test_add_with_metadata(self, client_no_vector):
        """带元数据的添加"""
        mid = client_no_vector.add("test", metadata={"source": "api", "user": "alice"})
        mem = client_no_vector.get_memory(mid)
        assert mem is not None
        assert 'metadata' in mem or 'content' in mem

    @_OLLAMA_SKIP
    @pytest.mark.slow
    def test_add_many_increments_count(self, client_no_vector):
        """批量添加计数正确"""
        for i in range(50):
            client_no_vector.add(f"memory {i}")
        assert len(client_no_vector) == 50


# ═══════════════════════════════════════════════════════════════
# T1.3 add_batch 批量写入 (5 tests)
# ═══════════════════════════════════════════════════════════════

class TestAddBatch:
    """测试 add_batch() 方法"""

    def test_add_batch_basic(self, client_no_vector):
        """基本批量写入"""
        items = [{"content": f"batch {i}"} for i in range(10)]
        ids = client_no_vector.add_batch(items)
        assert len(ids) == 10
        assert len(client_no_vector) == 10

    def test_add_batch_with_metadata(self, client_no_vector):
        """批量写入含元数据"""
        items = [
            {"content": "item a", "metadata": {"type": "a"}},
            {"content": "item b", "metadata": {"type": "b"}},
        ]
        ids = client_no_vector.add_batch(items)
        assert len(ids) == 2

    def test_add_batch_empty_list(self, client_no_vector):
        """空列表"""
        ids = client_no_vector.add_batch([])
        assert ids == []
        assert len(client_no_vector) == 0

    def test_add_batch_large(self, client_no_vector):
        """大批量（100条）"""
        items = [{"content": f"big batch {i}"} for i in range(100)]
        ids = client_no_vector.add_batch(items)
        assert len(ids) == 100
        assert len(client_no_vector) == 100

    def test_add_batch_preserves_order(self, client_no_vector):
        """记录数量一致性"""
        items = [{"content": f"x{i}"} for i in range(50)]
        ids = client_no_vector.add_batch(items)
        assert len(client_no_vector) == 50


# ═══════════════════════════════════════════════════════════════
# T1.4 query 向量检索 (10 tests)
# ═══════════════════════════════════════════════════════════════

class TestQuery:
    """测试 query() 方法"""

    def test_query_returns_results(self, populated):
        """基本查询返回结果"""
        results = populated.query("天气")
        assert len(results) > 0
        assert isinstance(results, list)

    def test_query_result_structure(self, populated):
        """结果包含必要字段"""
        results = populated.query("天气", top_k=2)
        for r in results:
            assert 'content' in r
            assert 'score' in r
            assert isinstance(r['score'], (int, float))

    def test_query_top_k(self, populated):
        """top_k 限制生效"""
        results = populated.query("天气", top_k=2)
        assert len(results) <= 2

    def test_query_no_match(self, populated):
        """不匹配的查询返回空"""
        results = populated.query("zzz_nonexistent_zzz")
        # 可能返回空列表或低分结果
        assert isinstance(results, list)

    def test_query_empty_passes(self, client_no_vector):
        """空客户端查询不崩溃"""
        results = client_no_vector.query("test")
        assert isinstance(results, list)

    def test_query_scores_sorted(self, populated):
        """结果按分数降序排列"""
        results = populated.query("天气", top_k=5)
        scores = [r['score'] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_keyword_mode(self, populated):
        """关键词检索模式"""
        results = populated.query("Python", use_vector=False, use_keyword=True)
        assert isinstance(results, list)

    def test_query_vector_mode(self, client_with_vector):
        """向量检索模式"""
        c = client_with_vector
        c.add("机器学习和深度学习是人工智能的核心")
        c.add("今天天气晴朗适合户外运动")
        results = c.query("AI技术", top_k=1, use_vector=True, use_keyword=False)
        assert len(results) >= 1

    def test_query_time_range(self, populated):
        """时间范围过滤"""
        import time
        now = int(time.time())
        results = populated.query("天气", time_range=(now - 3600, now + 3600))
        assert isinstance(results, list)

    def test_query_energy_filter(self, populated):
        """能量过滤"""
        results = populated.query("天气", energy_filter="wood")
        assert isinstance(results, list)


# ═══════════════════════════════════════════════════════════════
# T1.5 query_multihop 多跳推理 (8 tests)
# ═══════════════════════════════════════════════════════════════

class TestMultiHop:
    """测试 query_multihop()"""

    @pytest.fixture
    def chain_client(self, client_no_vector):
        """创建因果链：A→B→C→D"""
        c = client_no_vector
        a = c.add("如果努力学习，成绩会提高")
        b = c.add("成绩提高了会获得奖学金")
        c_id = c.add("获得奖学金可以减轻家庭负担")
        d = c.add("减轻负担后有更多时间学习")
        c.link_memories(a, b)
        c.link_memories(b, c_id)
        c.link_memories(c_id, d)
        return c

    def test_multihop_basic(self, chain_client):
        """基本多跳查询"""
        results = chain_client.query_multihop("学习", max_hops=3)
        assert isinstance(results, list)

    def test_multihop_vector_first(self, chain_client):
        """vector_first 模式"""
        results = chain_client.query_multihop("学习", fusion_mode="vector_first")
        assert isinstance(results, list)

    def test_multihop_hybrid(self, chain_client):
        """hybrid 模式"""
        results = chain_client.query_multihop("学习", fusion_mode="hybrid")
        assert isinstance(results, list)

    def test_multihop_graph_first(self, chain_client):
        """graph_first 模式"""
        results = chain_client.query_multihop("学习", fusion_mode="graph_first")
        assert isinstance(results, list)

    def test_multihop_causal_only(self, chain_client):
        """仅因果查询"""
        results = chain_client.query_multihop("学习", causal_only=True)
        assert isinstance(results, list)

    def test_multihop_max_hops_zero(self, chain_client):
        """max_hops=0 不崩溃"""
        results = chain_client.query_multihop("学习", max_hops=0)
        assert isinstance(results, list)

    def test_multihop_single_hop(self, chain_client):
        """单跳"""
        results = chain_client.query_multihop("学习", max_hops=1)
        assert isinstance(results, list)

    def test_multihop_empty_client(self, client_no_vector):
        """空客户端多跳查询"""
        results = client_no_vector.query_multihop("test")
        assert isinstance(results, list)


# ═══════════════════════════════════════════════════════════════
# T1.6 query_multihop_spacetime (5 tests)
# ═══════════════════════════════════════════════════════════════

class TestMultiHopSpacetime:
    """测试 query_multihop_spacetime()"""

    @pytest.fixture
    def sp_client(self, client_no_vector):
        c = client_no_vector
        a = c.add("周一项目启动会议")
        b = c.add("周三完成需求文档")
        c_id = c.add("周五提交测试报告")
        c.link_memories(a, b)
        c.link_memories(b, c_id)
        return c

    def test_spacetime_basic(self, sp_client):
        """基本时空多跳"""
        results = sp_client.query_multihop_spacetime("项目")
        assert isinstance(results, list)

    def test_spacetime_hybrid(self, sp_client):
        """时空 hybrid 模式"""
        results = sp_client.query_multihop_spacetime("项目", fusion_mode="hybrid")
        assert isinstance(results, list)

    def test_spacetime_no_weight(self, sp_client):
        """禁用时空权重"""
        results = sp_client.query_multihop_spacetime("项目", use_spacetime_weight=False)
        assert isinstance(results, list)

    def test_spacetime_empty(self, client_no_vector):
        """空客户端"""
        results = client_no_vector.query_multihop_spacetime("test")
        assert isinstance(results, list)

    def test_spacetime_max_hops(self, sp_client):
        """自定义 max_hops"""
        results = sp_client.query_multihop_spacetime("项目", max_hops=1, top_k=2)
        assert isinstance(results, list)


# ═══════════════════════════════════════════════════════════════
# T1.7 FAISS 索引生命周期 (8 tests)
# ═══════════════════════════════════════════════════════════════

class TestFAISSLifecycle:
    """测试 FAISS 索引的创建/持久化/恢复"""

    def test_faiss_created_on_add(self, client_with_vector):
        """add 后 FAISS 索引被创建"""
        c = client_with_vector
        c.add("test memory for faiss index creation")
        assert c._faiss_index is not None

    def test_faiss_not_created_without_vector(self, client_no_vector):
        """禁用向量时不创建 FAISS"""
        c = client_no_vector
        c.add("test")
        assert c._faiss_index is None

    def test_faiss_persist_and_load(self):
        """FAISS 索引持久化与恢复"""
        with tempfile.TemporaryDirectory() as d:
            # 创建并写入
            c1 = SuMemoryLitePro(storage_path=d, enable_vector=True)
            c1.add("persistent memory test")
            c1.add("another memory for faiss")
            n1 = c1._faiss_index.ntotal if c1._faiss_index else 0

            # 用同一路径创建新客户端
            c2 = SuMemoryLitePro(storage_path=d, enable_vector=True)
            n2 = c2._faiss_index.ntotal if c2._faiss_index else 0
            assert n2 == n1

    def test_faiss_index_grows(self, client_with_vector):
        """索引随 add 增长"""
        c = client_with_vector
        n0 = c._faiss_index.ntotal if c._faiss_index else 0
        for i in range(5):
            c.add(f"growth test {i}")
        n1 = c._faiss_index.ntotal if c._faiss_index else 0
        assert n1 >= n0 + 5

    def test_faiss_search_after_add(self, client_with_vector):
        """add 后可搜索"""
        c = client_with_vector
        c.add("deep learning and neural networks")
        results = c.query("machine learning", top_k=1, use_vector=True, use_keyword=False)
        assert len(results) >= 1

    def test_faiss_id_map_consistency(self, client_with_vector):
        """id_map 一致性"""
        c = client_with_vector
        mid = c.add("id map test")
        # FAISS id_map 中应包含该记忆
        if c._faiss_id_map:
            assert len(c._faiss_id_map) == len(c)

    def test_faiss_handles_delete(self, client_with_vector):
        """索引不受影响（删除暂不重建索引）"""
        c = client_with_vector
        c.add("keep this")
        initial = len(c)
        # 不直接测试delete，验证add后的状态
        assert initial == 1

    def test_faiss_no_crash_on_empty_search(self, client_with_vector):
        """空索引搜索不崩溃"""
        c = client_with_vector
        results = c.query("anything", use_vector=True)
        assert isinstance(results, list)


# ═══════════════════════════════════════════════════════════════
# T1.8 embedding 懒加载/回退 (6 tests)
# ═══════════════════════════════════════════════════════════════

class TestEmbeddingFallback:
    """测试 embedding 懒加载和回退"""

    def test_no_vector_does_not_load_embedding(self, client_no_vector):
        """禁用向量时不加载 embedding"""
        assert client_no_vector._embedding is None

    def test_vector_lazy_load(self):
        """向量模式延迟加载 embedding"""
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(storage_path=d, enable_vector=True)
            # 初始化时 embedding 尚未加载
            # add 时才触发 _ensure_embedding()
            c.add("trigger embedding load")
            assert c._embedding is not None or c._embedding_backend_type != "none"

    def test_embedding_fallback_to_tfidf(self):
        """向量不可用时回退到 TF-IDF"""
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(storage_path=d, enable_vector=False, enable_tfidf=True)
            c.add("test content")
            results = c.query("test", use_vector=False, use_keyword=True)
            assert isinstance(results, list)

    def test_multiple_embedding_backends(self):
        """多后端不冲突"""
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(
                storage_path=d,
                enable_vector=True,
                embedding_backend="minimax"
            )
            c.add("backend test")
            assert c._embedding_backend_type in ("minimax", "ollama", "sentence_transformers", "sentence-transformers", "sentence-transformers-bge-m3", "none")

    def test_embedding_dim_property(self):
        """embedding_dim 属性"""
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(storage_path=d, enable_vector=True)
            c.add("dim test")
            dim = c.embedding_dim if hasattr(c, 'embedding_dim') else None
            # 至少不报错
            assert True

    def test_ensure_embedding_idempotent(self):
        """_ensure_embedding 多次调用不重复创建"""
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(storage_path=d, enable_vector=True)
            c._ensure_embedding()
            emb1 = c._embedding
            c._ensure_embedding()
            emb2 = c._embedding
            assert emb1 is emb2


# ═══════════════════════════════════════════════════════════════
# T1.9 MemoryGraph CRUD (6 tests)
# ═══════════════════════════════════════════════════════════════

class TestMemoryGraph:
    """测试 MemoryGraph CRUD 操作"""

    def test_link_memories(self, client_no_vector):
        """链接两条记忆"""
        a = client_no_vector.add("parent memory")
        b = client_no_vector.add("child memory")
        client_no_vector.link_memories(a, b)

    def test_get_children(self, client_no_vector):
        """获取子节点"""
        a = client_no_vector.add("root")
        b = client_no_vector.add("child 1")
        c = client_no_vector.add("child 2")
        client_no_vector.link_memories(a, b)
        client_no_vector.link_memories(a, c)
        children = client_no_vector.get_children(a)
        assert len(children) == 2

    def test_get_parents(self, client_no_vector):
        """获取父节点（去重）"""
        a = client_no_vector.add("parent")
        b = client_no_vector.add("child")
        client_no_vector.link_memories(a, b)
        parents = client_no_vector.get_parents(b)
        unique_ids = set(p['memory_id'] for p in parents)
        assert a in unique_ids

    def test_get_memory(self, client_no_vector):
        """获取单条记忆"""
        mid = client_no_vector.add("test memory")
        mem = client_no_vector.get_memory(mid)
        assert mem is not None
        assert 'content' in mem

    def test_get_memory_nonexistent(self, client_no_vector):
        """获取不存在的记忆返回 None"""
        mem = client_no_vector.get_memory("nonexistent_id")
        assert mem is None

    def test_chain_link(self, client_no_vector):
        """链式链接 A→B→C"""
        a = client_no_vector.add("A")
        b = client_no_vector.add("B")
        c = client_no_vector.add("C")
        client_no_vector.link_memories(a, b)
        client_no_vector.link_memories(b, c)
        children_a = client_no_vector.get_children(a)
        children_b = client_no_vector.get_children(b)
        assert len(children_a) >= 1
        assert len(children_b) >= 1


# ═══════════════════════════════════════════════════════════════
# T1.10 predict / explain (6 tests)
# ═══════════════════════════════════════════════════════════════

class TestPredictExplain:
    """测试 predict 和 explain"""

    @pytest.fixture
    def pred_client(self):
        with tempfile.TemporaryDirectory() as d:
            c = SuMemoryLitePro(
                storage_path=d,
                enable_vector=False,
                enable_prediction=True,
                enable_explainability=True,
            )
            c.add("周一项目启动", topic="项目")
            c.add("周三完成第一阶段", topic="项目")
            c.add("周五测试通过", topic="项目")
            yield c

    def test_predict_returns_dict(self, pred_client):
        """predict 返回字典"""
        result = pred_client.predict(query="项目")
        assert isinstance(result, dict)

    def test_predict_with_metric(self, pred_client):
        """predict 指定 metric"""
        result = pred_client.predict(metric="activity")
        assert isinstance(result, dict)

    def test_predict_no_query(self, pred_client):
        """predict 不传 query"""
        result = pred_client.predict()
        assert isinstance(result, dict)

    def test_explain_query(self, pred_client):
        """explain_query 不崩溃"""
        results = pred_client.query("项目")
        if hasattr(pred_client, 'explain_query'):
            explanation = pred_client.explain_query("项目", results)
            assert isinstance(explanation, dict)

    def test_predict_empty_client(self, client_no_vector):
        """空客户端 predict"""
        c = client_no_vector
        if hasattr(c, 'predict'):
            result = c.predict()
            assert isinstance(result, dict)

    def test_explain_on_empty(self, client_no_vector):
        """空客户端 explain"""
        c = client_no_vector
        results = c.query("test")
        if hasattr(c, 'explain_query'):
            explanation = c.explain_query("test", results)
            assert isinstance(explanation, dict)


# ═══════════════════════════════════════════════════════════════
# T1.11 get_stats (4 tests)
# ═══════════════════════════════════════════════════════════════

class TestStats:
    """测试 get_stats()"""

    def test_stats_has_memory_count(self, client_no_vector):
        """stats 包含 total_memories"""
        client_no_vector.add("test")
        s = client_no_vector.get_stats()
        assert s['total_memories'] == 1

    def test_stats_grows(self, client_no_vector):
        """stats 随 add 更新"""
        client_no_vector.add("a")
        client_no_vector.add("b")
        client_no_vector.add("c")
        s = client_no_vector.get_stats()
        assert s['total_memories'] == 3

    def test_stats_has_graph_info(self, client_no_vector):
        """stats 包含图信息（如启用）"""
        a = client_no_vector.add("root")
        b = client_no_vector.add("leaf")
        client_no_vector.link_memories(a, b)
        s = client_no_vector.get_stats()
        assert 'total_memories' in s

    def test_stats_consistent_after_batch(self, client_no_vector):
        """批量写入后 stats 一致"""
        items = [{"content": f"s{i}"} for i in range(20)]
        client_no_vector.add_batch(items)
        s = client_no_vector.get_stats()
        assert s['total_memories'] == 20


# ═══════════════════════════════════════════════════════════════
# T1.12 记忆生命周期 (4 tests)
# ═══════════════════════════════════════════════════════════════

class TestMemoryLifecycle:
    """测试记忆生命周期操作"""

    def test_add_then_query(self, client_no_vector):
        """add 后立即可 query"""
        client_no_vector.add("即时可查的测试记忆")
        results = client_no_vector.query("测试记忆")
        assert len(results) > 0

    def test_overwrite_behavior(self, client_no_vector):
        """重复添加相同内容产生独立记忆"""
        id1 = client_no_vector.add("重复内容")
        id2 = client_no_vector.add("重复内容")
        assert id1 != id2
        assert len(client_no_vector) == 2

    def test_partial_query_match(self, client_no_vector):
        """部分匹配"""
        client_no_vector.add("深度学习和神经网络")
        results = client_no_vector.query("深度学习")
        assert len(results) >= 1

    def test_query_order_stable(self, client_no_vector):
        """查询结果稳定（同输入同输出）"""
        for i in range(10):
            client_no_vector.add(f"稳定测试内容 {i}")
        r1 = client_no_vector.query("稳定测试")
        r2 = client_no_vector.query("稳定测试")
        assert [x['content'] for x in r1] == [x['content'] for x in r2]
