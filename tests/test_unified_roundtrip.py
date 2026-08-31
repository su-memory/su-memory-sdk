"""SuMemory 统一引擎核心链路测试。

验证第一性承诺：add → 持久化 → 重载 → query 的数据一致性。
不依赖外部服务（无 Ollama 时走 TF-IDF/Hash 兜底后端，CI 可运行）。
"""

import tempfile

from su_memory import SuMemory


def _make_client(path, **kwargs):
    """构造统一引擎，关闭与核心链路无关的可选子系统以减少状态依赖。"""
    return SuMemory(
        storage_path=path,
        enable_graph=False,
        enable_temporal=False,
        enable_session=False,
        enable_prediction=False,
        **kwargs,
    )


def test_add_query_persist_roundtrip():
    """add → flush → 重载 → query 的记忆数与会话内容一致。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        client = _make_client(tmpdir)
        client.add("张总在周一会议上提到Q3目标增长25%")
        client.add("Q3目标增长主要由新功能上线驱动")
        client.flush()
        assert len(client) == 2
        client.close()

        reloaded = _make_client(tmpdir)
        assert len(reloaded) == 2, "重载后记忆数应一致"
        results = reloaded.query("Q3目标", top_k=2)
        assert isinstance(results, list) and len(results) > 0, "重载后查询应返回结果"
        contents = {r["content"] for r in results}
        assert "张总在周一会议上提到Q3目标增长25%" in contents
        reloaded.close()


def test_query_returns_required_fields():
    """query 结果应包含 memory_id/content/score/metadata 完整字段。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        client = _make_client(tmpdir)
        client.add("Transformer模型革新了自然语言处理", metadata={"topic": "nlp"})
        client.flush()
        results = client.query("Transformer 自然语言", top_k=1)
        assert results, "至少应返回 1 条结果"
        result = results[0]
        for field in ("memory_id", "content", "score", "metadata"):
            assert field in result, f"结果缺少字段 {field}"
        assert result["metadata"].get("topic") == "nlp"
        client.close()


def test_add_returns_unique_ids():
    """add 应返回非空且唯一的记忆 ID。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        client = _make_client(tmpdir)
        ids = [client.add(f"记忆内容 {i}") for i in range(5)]
        assert all(ids), "记忆 ID 不应为空"
        assert len(set(ids)) == 5, "记忆 ID 应唯一"
        client.close()
