"""
rag_engine 统一入口(facade)测试

覆盖: RAGType 枚举、字符串转换、两种 RAG 变体的工厂分发与错误路径。
不依赖真实向量库/外部服务(构造器均打桩)。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from su_memory.sdk.rag_engine import RAGType, create_rag


class TestRAGType:
    def test_enum_values(self):
        assert RAGType.VECTOR_GRAPH.value == "vector_graph"
        assert RAGType.SPATIAL.value == "spatial"

    def test_enum_from_string(self):
        assert RAGType("vector_graph") is RAGType.VECTOR_GRAPH
        assert RAGType("spatial") is RAGType.SPATIAL

    def test_enum_is_str_subclass(self):
        # 允许直接当字符串用(与旧版调用方兼容)
        assert isinstance(RAGType.VECTOR_GRAPH, str)


class TestCreateRag:
    def test_default_creates_vector_graph(self, monkeypatch):
        called = {}
        def fake(**kw):
            called.update(kw)
            return "vector-rag"
        monkeypatch.setattr(
            "su_memory.sdk.vector_graph_rag.create_vector_graph_rag", fake
        )
        out = create_rag(RAGType.VECTOR_GRAPH, dims=128)
        assert out == "vector-rag"
        assert called == {"dims": 128}

    def test_string_type_conversion(self, monkeypatch):
        def fake(**kw):
            return "ok"
        monkeypatch.setattr(
            "su_memory.sdk.vector_graph_rag.create_vector_graph_rag", fake
        )
        assert create_rag("vector_graph") == "ok"

    def test_spatial_type(self, monkeypatch):
        def fake(**kw):
            return "spatial-rag"
        monkeypatch.setattr(
            "su_memory.sdk.spatial_rag.create_spatial_rag", fake
        )
        assert create_rag(RAGType.SPATIAL, radius=3.0) == "spatial-rag"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            create_rag("no_such_rag")

    def test_vector_module_unavailable_reraises(self, monkeypatch):
        # 模拟 vector_graph_rag 依赖缺失: from-import 抛 ImportError
        monkeypatch.setitem(
            sys.modules, "su_memory.sdk.vector_graph_rag", None
        )
        with pytest.raises(ImportError):
            create_rag(RAGType.VECTOR_GRAPH)

    def test_spatial_module_unavailable_reraises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "su_memory.sdk.spatial_rag", None)
        with pytest.raises(ImportError):
            create_rag(RAGType.SPATIAL)
