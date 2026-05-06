"""
LLMAdapter 测试

测试 llm_adapter/openai_compat.py 的 LLMAdapter 类。
不实际调用 LLM，只测试配置、初始化和降级行为。
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'llm_adapter'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestLLMAdapterInit:
    """LLMAdapter 初始化测试"""

    def test_import(self):
        """测试模块可导入"""
        from llm_adapter.openai_compat import LLMAdapter
        assert LLMAdapter is not None

    @patch.dict(os.environ, {}, clear=True)
    @patch('llm_adapter.openai_compat.OpenAI')
    @patch('llm_adapter.openai_compat.httpx')
    def test_default_config(self, mock_httpx, mock_openai):
        """测试默认配置（环境变量全空）"""
        # 需要设置环境变量默认值
        from llm_adapter.openai_compat import LLMAdapter
        adapter = LLMAdapter()
        assert adapter.provider == "ollama"
        assert adapter.base_url == "http://localhost:11434/v1"
        assert adapter.api_key == "ollama"
        assert adapter.default_model == "qwen2.5:7b"

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "openai",
        "LLM_BASE_URL": "https://api.openai.com/v1",
        "LLM_API_KEY": "sk-test123",
        "LLM_MODEL": "gpt-4"
    })
    @patch('llm_adapter.openai_compat.OpenAI')
    @patch('llm_adapter.openai_compat.httpx')
    def test_custom_config(self, mock_httpx, mock_openai):
        """测试自定义环境变量配置"""
        from llm_adapter.openai_compat import LLMAdapter
        adapter = LLMAdapter()
        assert adapter.provider == "openai"
        assert adapter.base_url == "https://api.openai.com/v1"
        assert adapter.api_key == "sk-test123"
        assert adapter.default_model == "gpt-4"

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "vllm",
        "LLM_BASE_URL": "http://localhost:8000/v1"
    })
    @patch('llm_adapter.openai_compat.OpenAI')
    @patch('llm_adapter.openai_compat.httpx')
    def test_vllm_config(self, mock_httpx, mock_openai):
        """测试 vLLM 配置"""
        from llm_adapter.openai_compat import LLMAdapter
        adapter = LLMAdapter()
        assert adapter.provider == "vllm"
        assert adapter.base_url == "http://localhost:8000/v1"

    @patch.dict(os.environ, {}, clear=True)
    @patch('llm_adapter.openai_compat.OpenAI')
    @patch('llm_adapter.openai_compat.httpx')
    def test_openai_client_created(self, mock_httpx, mock_openai):
        """测试 OpenAI 客户端已创建"""
        from llm_adapter.openai_compat import LLMAdapter
        mock_client_instance = MagicMock()
        mock_openai.return_value = mock_client_instance

        adapter = LLMAdapter()
        mock_openai.assert_called_once()
        assert adapter.client == mock_client_instance

    @patch.dict(os.environ, {}, clear=True)
    @patch('llm_adapter.openai_compat.OpenAI')
    @patch('llm_adapter.openai_compat.httpx.Client')
    def test_httpx_timeout(self, mock_httpx_client, mock_openai):
        """测试 HTTP 超时设置"""
        from llm_adapter.openai_compat import LLMAdapter
        mock_httpx_instance = MagicMock()
        mock_httpx_client.return_value = mock_httpx_instance

        LLMAdapter()
        mock_httpx_client.assert_called_once_with(timeout=120.0)

    @patch.dict(os.environ, {}, clear=True)
    @patch('llm_adapter.openai_compat.OpenAI')
    @patch('llm_adapter.openai_compat.httpx')
    def test_openai_base_url_passed(self, mock_httpx, mock_openai):
        """测试 OpenAI 客户端接收 base_url"""
        from llm_adapter.openai_compat import LLMAdapter
        LLMAdapter()
        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["base_url"] == "http://localhost:11434/v1"
        assert call_kwargs["api_key"] == "ollama"


class TestLLMAdapterAttributes:
    """LLMAdapter 属性测试"""

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "qwen",
        "LLM_BASE_URL": "http://qwen.local/v1",
        "LLM_API_KEY": "qwen-key",
        "LLM_MODEL": "qwen-max"
    })
    @patch('llm_adapter.openai_compat.OpenAI')
    @patch('llm_adapter.openai_compat.httpx')
    def test_client_not_none(self, mock_httpx, mock_openai):
        """测试 client 属性不为 None"""
        from llm_adapter.openai_compat import LLMAdapter
        adapter = LLMAdapter()
        assert adapter.client is not None

    @patch.dict(os.environ, {}, clear=True)
    @patch('llm_adapter.openai_compat.OpenAI')
    @patch('llm_adapter.openai_compat.httpx')
    def test_client_is_openai_instance(self, mock_httpx, mock_openai):
        """测试 client 是 OpenAI 实例"""
        from llm_adapter.openai_compat import LLMAdapter
        mock_instance = MagicMock()
        mock_openai.return_value = mock_instance

        adapter = LLMAdapter()
        assert adapter.client == mock_instance


class TestLLMAdapterEmbed:
    """LLMAdapter embed() 方法测试"""

    @patch.dict(os.environ, {}, clear=True)
    @patch('llm_adapter.openai_compat.OpenAI')
    @patch('llm_adapter.openai_compat.httpx')
    def test_embed_raises_not_implemented(self, mock_httpx, mock_openai):
        """测试 embed 方法抛 NotImplementedError"""
        import asyncio
        from llm_adapter.openai_compat import LLMAdapter
        adapter = LLMAdapter()

        async def run():
            with pytest.raises(NotImplementedError,
                               match="sentence-transformers"):
                await adapter.embed(["测试文本"])
        asyncio.run(run())

    @patch.dict(os.environ, {}, clear=True)
    @patch('llm_adapter.openai_compat.OpenAI')
    @patch('llm_adapter.openai_compat.httpx')
    def test_embed_multiple_texts(self, mock_httpx, mock_openai):
        """测试 embed 多文本也抛错"""
        import asyncio
        from llm_adapter.openai_compat import LLMAdapter
        adapter = LLMAdapter()

        async def run():
            with pytest.raises(NotImplementedError):
                await adapter.embed(["文本1", "文本2", "文本3"])
        asyncio.run(run())

    @patch.dict(os.environ, {}, clear=True)
    @patch('llm_adapter.openai_compat.OpenAI')
    @patch('llm_adapter.openai_compat.httpx')
    def test_embed_custom_model(self, mock_httpx, mock_openai):
        """测试 embed 自定义模型也抛错"""
        import asyncio
        from llm_adapter.openai_compat import LLMAdapter
        adapter = LLMAdapter()

        async def run():
            with pytest.raises(NotImplementedError):
                await adapter.embed(["测试"], model="bge-m3")
        asyncio.run(run())


class TestLLMAdapterChatSignature:
    """LLMAdapter chat() 方法签名测试（不实际调用）"""

    @patch.dict(os.environ, {}, clear=True)
    @patch('llm_adapter.openai_compat.OpenAI')
    @patch('llm_adapter.openai_compat.httpx')
    def test_chat_exists(self, mock_httpx, mock_openai):
        """测试 chat 方法存在"""
        from llm_adapter.openai_compat import LLMAdapter
        adapter = LLMAdapter()
        assert hasattr(adapter, 'chat')
        assert callable(adapter.chat)

    @patch.dict(os.environ, {}, clear=True)
    @patch('llm_adapter.openai_compat.OpenAI')
    @patch('llm_adapter.openai_compat.httpx')
    def test_chat_is_async(self, mock_httpx, mock_openai):
        """测试 chat 是异步方法"""
        import asyncio
        from llm_adapter.openai_compat import LLMAdapter
        adapter = LLMAdapter()
        assert asyncio.iscoroutinefunction(adapter.chat)

    @patch.dict(os.environ, {}, clear=True)
    @patch('llm_adapter.openai_compat.OpenAI')
    @patch('llm_adapter.openai_compat.httpx')
    def test_chat_uses_default_model(self, mock_httpx, mock_openai):
        """测试 chat 使用默认模型"""
        import asyncio
        from llm_adapter.openai_compat import LLMAdapter

        mock_response = MagicMock()
        mock_response.id = "resp-1"
        mock_response.model = "qwen2.5:7b"
        mock_choice = MagicMock()
        mock_choice.index = 0
        mock_choice.message.role = "assistant"
        mock_choice.message.content = "你好！"
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        adapter = LLMAdapter()

        async def run():
            result = await adapter.chat(
                messages=[{"role": "user", "content": "你好"}]
            )
            # 验证调用了默认模型
            create_call = mock_client.chat.completions.create
            assert create_call.call_args[1]["model"] == "qwen2.5:7b"

        asyncio.run(run())

    @patch.dict(os.environ, {}, clear=True)
    @patch('llm_adapter.openai_compat.OpenAI')
    @patch('llm_adapter.openai_compat.httpx')
    def test_chat_custom_model(self, mock_httpx, mock_openai):
        """测试 chat 使用自定义模型"""
        import asyncio
        from llm_adapter.openai_compat import LLMAdapter

        mock_response = MagicMock()
        mock_response.id = "resp-2"
        mock_response.model = "llama3"
        mock_choice = MagicMock()
        mock_choice.index = 0
        mock_choice.message.role = "assistant"
        mock_choice.message.content = "回答"
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_response.usage.total_tokens = 8

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        adapter = LLMAdapter()

        async def run():
            result = await adapter.chat(
                messages=[{"role": "user", "content": "你好"}],
                model="llama3"
            )
            create_call = mock_client.chat.completions.create
            assert create_call.call_args[1]["model"] == "llama3"

        asyncio.run(run())

    @patch.dict(os.environ, {}, clear=True)
    @patch('llm_adapter.openai_compat.OpenAI')
    @patch('llm_adapter.openai_compat.httpx')
    def test_chat_temperature_and_max_tokens(self, mock_httpx, mock_openai):
        """测试 chat 传递 temperature 和 max_tokens"""
        import asyncio
        from llm_adapter.openai_compat import LLMAdapter

        mock_response = MagicMock()
        mock_response.id = "resp-3"
        mock_response.model = "test-model"
        mock_choice = MagicMock()
        mock_choice.index = 0
        mock_choice.message.role = "assistant"
        mock_choice.message.content = "OK"
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 1
        mock_response.usage.completion_tokens = 1
        mock_response.usage.total_tokens = 2

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        adapter = LLMAdapter()

        async def run():
            result = await adapter.chat(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.2,
                max_tokens=100
            )
            create_call = mock_client.chat.completions.create
            assert create_call.call_args[1]["temperature"] == 0.2
            assert create_call.call_args[1]["max_tokens"] == 100

        asyncio.run(run())

    @patch.dict(os.environ, {}, clear=True)
    @patch('llm_adapter.openai_compat.OpenAI')
    @patch('llm_adapter.openai_compat.httpx')
    def test_chat_response_structure(self, mock_httpx, mock_openai):
        """测试 chat 响应结构"""
        import asyncio
        from llm_adapter.openai_compat import LLMAdapter

        mock_response = MagicMock()
        mock_response.id = "resp-struct"
        mock_response.model = "qwen2.5:7b"
        mock_choice = MagicMock()
        mock_choice.index = 0
        mock_choice.message.role = "assistant"
        mock_choice.message.content = "结构化回复"
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 200
        mock_response.usage.total_tokens = 300

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        adapter = LLMAdapter()

        async def run():
            result = await adapter.chat(
                messages=[{"role": "user", "content": "hello"}]
            )
            assert result["id"] == "resp-struct"
            assert result["model"] == "qwen2.5:7b"
            assert len(result["choices"]) == 1
            assert result["choices"][0]["index"] == 0
            assert result["choices"][0]["message"]["role"] == "assistant"
            assert result["choices"][0]["message"]["content"] == "结构化回复"
            assert result["choices"][0]["finish_reason"] == "stop"
            assert result["usage"]["prompt_tokens"] == 100
            assert result["usage"]["completion_tokens"] == 200
            assert result["usage"]["total_tokens"] == 300

        asyncio.run(run())

    @patch.dict(os.environ, {}, clear=True)
    @patch('llm_adapter.openai_compat.OpenAI')
    @patch('llm_adapter.openai_compat.httpx')
    def test_chat_error_propagation(self, mock_httpx, mock_openai):
        """测试 chat 错误向上传播"""
        import asyncio
        from llm_adapter.openai_compat import LLMAdapter

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("连接失败")
        mock_openai.return_value = mock_client

        adapter = LLMAdapter()

        async def run():
            with pytest.raises(RuntimeError, match="连接失败"):
                await adapter.chat(
                    messages=[{"role": "user", "content": "test"}]
                )
        asyncio.run(run())

    @patch.dict(os.environ, {}, clear=True)
    @patch('llm_adapter.openai_compat.OpenAI')
    @patch('llm_adapter.openai_compat.httpx')
    def test_chat_multiple_messages(self, mock_httpx, mock_openai):
        """测试 chat 多轮对话"""
        import asyncio
        from llm_adapter.openai_compat import LLMAdapter

        mock_response = MagicMock()
        mock_response.id = "resp-multi"
        mock_response.model = "test"
        mock_choice = MagicMock()
        mock_choice.index = 0
        mock_choice.message.role = "assistant"
        mock_choice.message.content = "回复"
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 30

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        adapter = LLMAdapter()

        async def run():
            messages = [
                {"role": "system", "content": "你是助手"},
                {"role": "user", "content": "问题1"},
                {"role": "assistant", "content": "回答1"},
                {"role": "user", "content": "问题2"},
            ]
            result = await adapter.chat(messages=messages)
            create_call = mock_client.chat.completions.create
            assert create_call.call_args[1]["messages"] == messages

        asyncio.run(run())

    @patch.dict(os.environ, {}, clear=True)
    @patch('llm_adapter.openai_compat.OpenAI')
    @patch('llm_adapter.openai_compat.httpx')
    def test_chat_extra_kwargs_passed(self, mock_httpx, mock_openai):
        """测试 chat 额外参数传递"""
        import asyncio
        from llm_adapter.openai_compat import LLMAdapter

        mock_response = MagicMock()
        mock_response.id = "resp-extra"
        mock_response.model = "test"
        mock_choice = MagicMock()
        mock_choice.index = 0
        mock_choice.message.role = "assistant"
        mock_choice.message.content = "OK"
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 1
        mock_response.usage.completion_tokens = 1
        mock_response.usage.total_tokens = 2

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        adapter = LLMAdapter()

        async def run():
            result = await adapter.chat(
                messages=[{"role": "user", "content": "test"}],
                top_p=0.9,
                stop=["END"],
                seed=42
            )
            create_call = mock_client.chat.completions.create
            assert create_call.call_args[1]["top_p"] == 0.9
            assert create_call.call_args[1]["stop"] == ["END"]
            assert create_call.call_args[1]["seed"] == 42

        asyncio.run(run())


class TestLLMAdapterMultipleChoices:
    """LLMAdapter 多选项响应测试"""

    @patch.dict(os.environ, {}, clear=True)
    @patch('llm_adapter.openai_compat.OpenAI')
    @patch('llm_adapter.openai_compat.httpx')
    def test_chat_multiple_choices(self, mock_httpx, mock_openai):
        """测试多选项响应（n>1）"""
        import asyncio
        from llm_adapter.openai_compat import LLMAdapter

        mock_response = MagicMock()
        mock_response.id = "resp-multi-choice"
        mock_response.model = "test"

        choice1 = MagicMock()
        choice1.index = 0
        choice1.message.role = "assistant"
        choice1.message.content = "选项A"
        choice1.finish_reason = "stop"

        choice2 = MagicMock()
        choice2.index = 1
        choice2.message.role = "assistant"
        choice2.message.content = "选项B"
        choice2.finish_reason = "length"

        mock_response.choices = [choice1, choice2]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        adapter = LLMAdapter()

        async def run():
            result = await adapter.chat(
                messages=[{"role": "user", "content": "test"}],
                n=2
            )
            assert len(result["choices"]) == 2
            assert result["choices"][0]["message"]["content"] == "选项A"
            assert result["choices"][1]["message"]["content"] == "选项B"
            assert result["choices"][0]["finish_reason"] == "stop"
            assert result["choices"][1]["finish_reason"] == "length"

        asyncio.run(run())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
