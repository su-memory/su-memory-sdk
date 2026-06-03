"""
v3.5.2 Embedding 后端单元测试

测试所有新增后端的接口合规性 (mock-based，不需要外部服务)
"""
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from su_memory.sdk.embedding import (
    CohereEmbedV3,
    DeepSeekEmbedding,
    EmbeddingBackend,
    EmbeddingManager,
    GoogleGeminiEmbedding,
    HashFallbackEmbedding,
    HuggingFaceTEIEmbedding,
    LlamaCppEmbedding,
    VoyageAIEmbedding,
)

# ─── 基类测试 ─────────────────────────────────────────────────────

class TestEmbeddingBackendBase:
    """基类 encode_batch 默认实现"""

    def test_encode_batch_default_impl(self):
        """默认 encode_batch 逐条调用 encode"""

        class DummyBackend(EmbeddingBackend):
            def encode(self, text: str) -> list[float]:
                return [float(ord(c)) for c in text[:3]]

        backend = DummyBackend()
        results = backend.encode_batch(["ab", "cd"])
        assert len(results) == 2
        assert results[0] == [float(ord("a")), float(ord("b"))]
        assert results[1] == [float(ord("c")), float(ord("d"))]


# ─── LlamaCppEmbedding ────────────────────────────────────────────

class TestLlamaCppEmbedding:
    """LlamaCpp 后端测试"""

    def test_init_default_model_path(self):
        backend = LlamaCppEmbedding()
        assert "bge-m3-q4_k_m.gguf" in backend.model_path

    def test_init_env_override(self):
        with patch.dict(os.environ, {"SU_MEMORY_GGUF_MODEL_PATH": "/tmp/test.gguf"}):
            backend = LlamaCppEmbedding()
            assert backend.model_path == "/tmp/test.gguf"

    def test_dims_default(self):
        backend = LlamaCppEmbedding()
        assert backend.dims == 1024

    def test_encode_calls_model(self):
        backend = LlamaCppEmbedding(model_path="/tmp/fake.gguf")
        mock_model = MagicMock()
        mock_model.embed.return_value = [[0.1, 0.2, 0.3]]
        backend._model = mock_model

        result = backend.encode("hello")
        assert result == [0.1, 0.2, 0.3]
        mock_model.embed.assert_called_once_with("hello")

    def test_encode_batch_native(self):
        backend = LlamaCppEmbedding(model_path="/tmp/fake.gguf")
        mock_model = MagicMock()
        mock_model.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
        backend._model = mock_model

        results = backend.encode_batch(["a", "b"])
        assert results == [[0.1, 0.2], [0.3, 0.4]]
        mock_model.embed.assert_called_once_with(["a", "b"])

    def test_encode_batch_empty(self):
        backend = LlamaCppEmbedding(model_path="/tmp/fake.gguf")
        assert backend.encode_batch([]) == []

    def test_import_error(self):
        backend = LlamaCppEmbedding(model_path="/tmp/fake.gguf")
        with patch.dict("sys.modules", {"llama_cpp": None}):
            with pytest.raises(ImportError, match="llama-cpp-python"):
                backend._get_model()

    def test_file_not_found(self):
        backend = LlamaCppEmbedding(model_path="/nonexistent/model.gguf")
        mock_llama_mod = MagicMock()
        import sys
        with patch.dict(sys.modules, {"llama_cpp": mock_llama_mod}):
            with pytest.raises(FileNotFoundError, match="GGUF"):
                backend._get_model()


# ─── DeepSeekEmbedding ────────────────────────────────────────────

class TestDeepSeekEmbedding:
    """DeepSeek 双模后端测试"""

    def test_cloud_mode_with_api_key(self):
        backend = DeepSeekEmbedding(api_key="sk-test")
        assert backend._resolve_mode() == "cloud"

    def test_local_mode_fallback(self, tmp_path):
        gguf_file = tmp_path / "deepseek-test.gguf"
        gguf_file.write_text("fake")
        env_patch = {"DEEPSEEK_API_KEY": ""}
        with patch.dict(os.environ, env_patch, clear=False):
            # Remove the key entirely if present
            os.environ.pop("DEEPSEEK_API_KEY", None)
            with patch(
                "su_memory.sdk.embedding.LlamaCppEmbedding.DEFAULT_MODEL_DIR",
                str(tmp_path),
            ):
                backend = DeepSeekEmbedding(api_key="")
                mode = backend._resolve_mode()
                assert mode == "local"
                assert backend._local_backend is not None

    def test_cloud_encode(self):
        backend = DeepSeekEmbedding(api_key="sk-test")
        response_data = {
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = backend.encode("hello")
            assert result == [0.1, 0.2, 0.3]

    def test_no_key_no_local_uses_fallback(self):
        backend = DeepSeekEmbedding(api_key="")
        backend._mode = "cloud"
        result = backend.encode("test")
        # Should use HashFallbackEmbedding
        assert isinstance(result, list)
        assert len(result) == 256  # HashFallback default dims


# ─── VoyageAIEmbedding ────────────────────────────────────────────

class TestVoyageAIEmbedding:
    """Voyage AI 后端测试"""

    def test_init_defaults(self):
        backend = VoyageAIEmbedding(api_key="test-key")
        assert backend.model == "voyage-3-large"
        assert backend.input_type == "document"

    def test_encode_for_query(self):
        backend = VoyageAIEmbedding(api_key="test-key")
        response_data = {
            "data": [{"embedding": [0.5, 0.6], "index": 0}]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            result = backend.encode_for_query("query text")
            assert result == [0.5, 0.6]
            # Verify input_type=query was sent
            call_args = mock_open.call_args
            req = call_args[0][0]
            body = json.loads(req.data)
            assert body["input_type"] == "query"

    def test_no_api_key_fallback(self):
        backend = VoyageAIEmbedding(api_key="")
        result = backend.encode("test")
        assert isinstance(result, list)
        assert len(result) == 256

    def test_encode_batch_splits_128(self):
        backend = VoyageAIEmbedding(api_key="test-key")
        # 生成 200 条文本
        texts = [f"text_{i}" for i in range(200)]

        def make_response_data(n):
            return {
                "data": [{"embedding": [0.1] * 4, "index": i} for i in range(n)]
            }

        # First call: 128, Second call: 72
        call_count = [0]

        def side_effect(req, timeout=60):
            body = json.loads(req.data)
            n = len(body["input"])
            call_count[0] += 1
            resp = MagicMock()
            resp.read.return_value = json.dumps(make_response_data(n)).encode()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("urllib.request.urlopen", side_effect=side_effect):
            results = backend.encode_batch(texts)
            assert len(results) == 200
            assert call_count[0] == 2  # Split into 2 batches


# ─── HuggingFaceTEIEmbedding ─────────────────────────────────────

class TestHuggingFaceTEIEmbedding:
    """HuggingFace TEI 后端测试"""

    def test_init_default_url(self):
        backend = HuggingFaceTEIEmbedding()
        assert backend.base_url == "http://localhost:8080"

    def test_encode_native_batch(self):
        backend = HuggingFaceTEIEmbedding()
        response_data = [[0.1, 0.2], [0.3, 0.4]]
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            results = backend.encode_batch(["a", "b"])
            assert results == [[0.1, 0.2], [0.3, 0.4]]

    def test_env_override(self):
        with patch.dict(os.environ, {"HF_TEI_URL": "http://custom:9090"}):
            backend = HuggingFaceTEIEmbedding()
            assert backend.base_url == "http://custom:9090"


# ─── CohereEmbedV3 ───────────────────────────────────────────────

class TestCohereEmbedV3:
    """Cohere Embed v3 后端测试"""

    def test_init_defaults(self):
        backend = CohereEmbedV3(api_key="test")
        assert backend.model == "embed-multilingual-v3.0"
        assert backend.input_type == "search_document"

    def test_encode_search_query(self):
        backend = CohereEmbedV3(api_key="test")
        response_data = {
            "embeddings": {"float": [[0.7, 0.8, 0.9]]}
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            result = backend.encode_for_query("search text")
            assert result == [0.7, 0.8, 0.9]
            req = mock_open.call_args[0][0]
            body = json.loads(req.data)
            assert body["input_type"] == "search_query"

    def test_encode_batch_splits_96(self):
        backend = CohereEmbedV3(api_key="test")
        texts = [f"t{i}" for i in range(100)]

        call_count = [0]

        def side_effect(req, timeout=60):
            body = json.loads(req.data)
            n = len(body["texts"])
            call_count[0] += 1
            resp_data = {"embeddings": {"float": [[0.1] * 4] * n}}
            resp = MagicMock()
            resp.read.return_value = json.dumps(resp_data).encode()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("urllib.request.urlopen", side_effect=side_effect):
            results = backend.encode_batch(texts)
            assert len(results) == 100
            assert call_count[0] == 2  # 96 + 4


# ─── GoogleGeminiEmbedding ────────────────────────────────────────

class TestGoogleGeminiEmbedding:
    """Google Gemini 后端测试"""

    def test_init_defaults(self):
        backend = GoogleGeminiEmbedding(api_key="test")
        assert backend.model == "text-embedding-004"
        assert backend.dims == 768

    def test_encode(self):
        backend = GoogleGeminiEmbedding(api_key="test-key")
        response_data = {"embedding": {"values": [0.1] * 768}}
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = backend.encode("test text")
            assert len(result) == 768
            assert result[0] == 0.1

    def test_no_api_key_fallback(self):
        backend = GoogleGeminiEmbedding(api_key="")
        result = backend.encode("test")
        assert isinstance(result, list)
        assert len(result) == 256


# ─── EmbeddingManager 扩展测试 ────────────────────────────────────

class TestEmbeddingManagerV351:
    """EmbeddingManager v3.5.2 扩展功能"""

    def test_supported_backends_expanded(self):
        assert "llama_cpp" in EmbeddingManager.SUPPORTED_BACKENDS
        assert "deepseek" in EmbeddingManager.SUPPORTED_BACKENDS
        assert "voyage" in EmbeddingManager.SUPPORTED_BACKENDS
        assert "hf_tei" in EmbeddingManager.SUPPORTED_BACKENDS
        assert "cohere_v3" in EmbeddingManager.SUPPORTED_BACKENDS
        assert "google" in EmbeddingManager.SUPPORTED_BACKENDS
        assert "onnx" in EmbeddingManager.SUPPORTED_BACKENDS

    def test_init_llama_cpp_backend(self):
        """测试初始化 llama_cpp 后端"""
        mgr = EmbeddingManager(backend="llama_cpp")
        assert mgr.backend_name == "llama_cpp"
        assert isinstance(mgr._backend, LlamaCppEmbedding)

    def test_init_deepseek_backend(self):
        mgr = EmbeddingManager(backend="deepseek")
        assert mgr.backend_name == "deepseek"
        assert isinstance(mgr._backend, DeepSeekEmbedding)

    def test_init_voyage_backend(self):
        mgr = EmbeddingManager(backend="voyage")
        assert mgr.backend_name == "voyage"
        assert isinstance(mgr._backend, VoyageAIEmbedding)

    def test_encode_batch_method(self):
        """EmbeddingManager.encode_batch 委托到后端"""
        mgr = EmbeddingManager(backend="ollama")
        # Mock the backend
        mock_backend = MagicMock()
        mock_backend.encode_batch.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mgr._backend = mock_backend

        results = mgr.encode_batch(["a", "b"])
        assert results == [[0.1, 0.2], [0.3, 0.4]]
        mock_backend.encode_batch.assert_called_once_with(["a", "b"])

    def test_auto_detect_priority(self):
        """_auto_detect 优先级列表包含新后端"""
        mgr = EmbeddingManager.__new__(EmbeddingManager)
        mgr._backend = None
        mgr._backend_info = None
        mgr.backend_name = "auto"

        # 全部不可用时 fallback 到 hash
        with patch.object(EmbeddingManager, "_test_backend", return_value=False):
            mgr._auto_detect()
            assert mgr.backend_name == "hash_fallback"


# ─── _sys/embedder.py 工厂测试 ────────────────────────────────────

class TestCreateEmbedderFactory:
    """create_embedder 工厂 v3.5.2 扩展"""

    def test_create_llama_cpp(self):
        from su_memory._sys.embedder import LlamaCppEmbedder, create_embedder
        embedder = create_embedder("llama_cpp")
        assert isinstance(embedder, LlamaCppEmbedder)

    def test_create_deepseek(self):
        from su_memory._sys.embedder import DeepSeekEmbedder, create_embedder
        embedder = create_embedder("deepseek")
        assert isinstance(embedder, DeepSeekEmbedder)

    def test_create_voyage(self):
        from su_memory._sys.embedder import VoyageAIEmbedder, create_embedder
        embedder = create_embedder("voyage")
        assert isinstance(embedder, VoyageAIEmbedder)

    def test_create_hf_tei(self):
        from su_memory._sys.embedder import HuggingFaceTEIEmbedder, create_embedder
        embedder = create_embedder("hf_tei")
        assert isinstance(embedder, HuggingFaceTEIEmbedder)

    def test_create_cohere_v3(self):
        from su_memory._sys.embedder import CohereV3Embedder, create_embedder
        embedder = create_embedder("cohere_v3")
        assert isinstance(embedder, CohereV3Embedder)

    def test_create_google(self):
        from su_memory._sys.embedder import GoogleGeminiEmbedder, create_embedder
        embedder = create_embedder("google")
        assert isinstance(embedder, GoogleGeminiEmbedder)

    def test_unknown_backend_raises(self):
        from su_memory._sys.embedder import create_embedder
        with pytest.raises(ValueError, match="Unknown embedder backend"):
            create_embedder("nonexistent")


# ─── _sys/_async_embedder.py 测试 ─────────────────────────────────

class TestAsyncEmbeddingFactoryV351:
    """AsyncEmbeddingFactory v3.5.2 扩展"""

    def test_providers_expanded(self):
        from su_memory._sys._async_embedder import AsyncEmbeddingFactory
        providers = AsyncEmbeddingFactory._providers
        assert "llama_cpp" in providers
        assert "deepseek" in providers
        assert "voyage" in providers
        assert "hf_tei" in providers
        assert "cohere_v3" in providers
        assert "google" in providers

    def test_list_providers(self):
        from su_memory._sys._async_embedder import AsyncEmbeddingFactory
        all_providers = AsyncEmbeddingFactory.list_providers()
        assert "llama_cpp" in all_providers
        assert "deepseek" in all_providers
        assert "tfidf" in all_providers

    @pytest.mark.asyncio
    async def test_auto_detect_fallback_to_tfidf(self):
        from su_memory._sys._async_embedder import (
            AsyncEmbeddingFactory,
        )
        # All backends unavailable → should fall back to TF-IDF
        provider = await AsyncEmbeddingFactory.auto_detect()
        # In test environment nothing is available, so tfidf or similar
        assert provider is not None


# ─── HashFallbackEmbedding 测试 ───────────────────────────────────

class TestHashFallbackEmbedding:
    """Hash fallback 保持稳定"""

    def test_deterministic(self):
        backend = HashFallbackEmbedding()
        r1 = backend.encode("hello")
        r2 = backend.encode("hello")
        assert r1 == r2

    def test_normalized(self):
        backend = HashFallbackEmbedding()
        vec = backend.encode("test text here")
        norm = sum(v * v for v in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-6

    def test_custom_dims(self):
        backend = HashFallbackEmbedding(dims=128)
        vec = backend.encode("test")
        assert len(vec) == 128
