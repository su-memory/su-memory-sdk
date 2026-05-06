"""
LangChain 集成适配器测试

测试 su_memory.integrations.langchain 模块：
- SuMemoryRetriever
- SuMemoryRetrieverConfig
- SuMemoryLoader
- SuMemoryTool
- SuMemoryMemory
- create_rag_chain
- create_conversational_chain

由于 LangChain 可能未安装，测试侧重：
- 配置数据类
- 优雅降级行为
- Mock 客户端下的核心逻辑
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestSuMemoryRetrieverConfig:
    """SuMemoryRetrieverConfig 配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        from su_memory.integrations.langchain import SuMemoryRetrieverConfig
        config = SuMemoryRetrieverConfig()
        assert config.search_type == "similarity"
        assert config.top_k == 5
        assert config.threshold == 0.5
        assert config.filter is None
        assert config.fetch_k == 20
        assert config.lambda_mult == 0.5

    def test_custom_config(self):
        """测试自定义配置"""
        from su_memory.integrations.langchain import SuMemoryRetrieverConfig
        config = SuMemoryRetrieverConfig(
            search_type="mmr",
            top_k=10,
            threshold=0.7,
            filter={"type": "conversation"},
            fetch_k=50,
            lambda_mult=0.8
        )
        assert config.search_type == "mmr"
        assert config.top_k == 10
        assert config.threshold == 0.7
        assert config.filter == {"type": "conversation"}
        assert config.fetch_k == 50
        assert config.lambda_mult == 0.8

    def test_is_dataclass(self):
        """测试是dataclass"""
        from su_memory.integrations.langchain import SuMemoryRetrieverConfig
        from dataclasses import is_dataclass
        assert is_dataclass(SuMemoryRetrieverConfig)

    def test_config_equality(self):
        """测试相等性"""
        from su_memory.integrations.langchain import SuMemoryRetrieverConfig
        config1 = SuMemoryRetrieverConfig(top_k=5, threshold=0.5)
        config2 = SuMemoryRetrieverConfig(top_k=5, threshold=0.5)
        config3 = SuMemoryRetrieverConfig(top_k=10, threshold=0.5)
        assert config1 == config2
        assert config1 != config3


class TestSuMemoryRetriever:
    """SuMemoryRetriever 检索器测试"""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.query.return_value = [
            {"content": "记忆1", "memory_id": "m1", "score": 0.9,
             "metadata": {"tag": "important"}},
            {"content": "记忆2", "id": "m2", "score": 0.7, "metadata": {}},
            {"content": "记忆3", "memory_id": "m3", "score": 0.3, "metadata": {}},
        ]
        return client

    def test_init(self, mock_client):
        """测试初始化"""
        import su_memory.integrations.langchain as mod
        mod.LANGCHAIN_AVAILABLE = True
        try:
            from su_memory.integrations.langchain import SuMemoryRetriever
            retriever = SuMemoryRetriever(mock_client)
            assert retriever._client == mock_client
            assert retriever._config is not None
            assert retriever._config.top_k == 5
        finally:
            mod.LANGCHAIN_AVAILABLE = False

    def test_init_custom_config(self, mock_client):
        """测试自定义配置初始化"""
        import su_memory.integrations.langchain as mod
        mod.LANGCHAIN_AVAILABLE = True
        try:
            from su_memory.integrations.langchain import (
                SuMemoryRetriever, SuMemoryRetrieverConfig)
            config = SuMemoryRetrieverConfig(top_k=10, threshold=0.7)
            retriever = SuMemoryRetriever(mock_client, config=config)
            assert retriever._config.top_k == 10
            assert retriever._config.threshold == 0.7
        finally:
            mod.LANGCHAIN_AVAILABLE = False

    def test_raises_without_langchain(self):
        """测试无 LangChain 时抛错误"""
        from su_memory.integrations.langchain import SuMemoryRetriever
        mock_client = MagicMock()
        with pytest.raises(ImportError, match="LangChain"):
            SuMemoryRetriever(mock_client)

    def test_invoke_calls_query(self, mock_client):
        """测试 invoke 调用查询并返回文档（阈值0.5过滤低分）"""
        import su_memory.integrations.langchain as mod
        mod.LANGCHAIN_AVAILABLE = True
        try:
            from su_memory.integrations.langchain import SuMemoryRetriever
            original_doc = mod.Document
            mock_doc = MagicMock()
            mock_doc.side_effect = lambda page_content, metadata: MagicMock(
                page_content=page_content, metadata=metadata)
            mod.Document = mock_doc

            retriever = SuMemoryRetriever(mock_client)
            docs = retriever.invoke("测试查询")

            mock_client.query.assert_called_once_with("测试查询", top_k=5)
            # 默认阈值0.5: 分数0.3的被过滤，只剩2条
            assert len(docs) == 2
            mod.Document = original_doc
        finally:
            mod.LANGCHAIN_AVAILABLE = False

    def test_invoke_filters_by_threshold(self, mock_client):
        """测试阈值过滤"""
        import su_memory.integrations.langchain as mod
        mod.LANGCHAIN_AVAILABLE = True
        try:
            from su_memory.integrations.langchain import (
                SuMemoryRetriever, SuMemoryRetrieverConfig)
            original_doc = mod.Document
            mock_doc = MagicMock()
            mock_doc.side_effect = lambda page_content, metadata: MagicMock(
                page_content=page_content, metadata=metadata)
            mod.Document = mock_doc

            config = SuMemoryRetrieverConfig(top_k=5, threshold=0.5)
            retriever = SuMemoryRetriever(mock_client, config=config)
            docs = retriever.invoke("测试")

            assert len(docs) == 2
            mod.Document = original_doc
        finally:
            mod.LANGCHAIN_AVAILABLE = False

    def test_invoke_empty_results(self, mock_client):
        """测试空结果"""
        import su_memory.integrations.langchain as mod
        mod.LANGCHAIN_AVAILABLE = True
        try:
            from su_memory.integrations.langchain import SuMemoryRetriever
            mock_client.query.return_value = []

            retriever = SuMemoryRetriever(mock_client)
            docs = retriever.invoke("无匹配")
            assert docs == []
        finally:
            mod.LANGCHAIN_AVAILABLE = False

    def test_get_relevant_documents_similarity(self, mock_client):
        """测试 get_relevant_documents similarity 类型"""
        import su_memory.integrations.langchain as mod
        mod.LANGCHAIN_AVAILABLE = True
        try:
            from su_memory.integrations.langchain import SuMemoryRetriever
            original_doc = mod.Document
            mock_doc = MagicMock()
            mock_doc.side_effect = lambda page_content, metadata: MagicMock(
                page_content=page_content, metadata=metadata)
            mod.Document = mock_doc

            retriever = SuMemoryRetriever(mock_client)
            docs = retriever.get_relevant_documents("查询")
            # 默认阈值0.5: 分数0.3的被过滤
            assert len(docs) == 2
            mod.Document = original_doc
        finally:
            mod.LANGCHAIN_AVAILABLE = False

    def test_get_relevant_documents_mmr(self, mock_client):
        """测试 get_relevant_documents mmr 类型"""
        import su_memory.integrations.langchain as mod
        mod.LANGCHAIN_AVAILABLE = True
        try:
            from su_memory.integrations.langchain import (
                SuMemoryRetriever, SuMemoryRetrieverConfig)
            original_doc = mod.Document
            mock_doc = MagicMock()
            mock_doc.side_effect = lambda page_content, metadata: MagicMock(
                page_content=page_content, metadata=metadata)
            mod.Document = mock_doc

            config = SuMemoryRetrieverConfig(search_type="mmr")
            retriever = SuMemoryRetriever(mock_client, config=config)
            docs = retriever.get_relevant_documents("查询")
            # 阈值0.5过滤
            assert len(docs) == 2
            mod.Document = original_doc
        finally:
            mod.LANGCHAIN_AVAILABLE = False

    def test_search_multihop(self, mock_client):
        """测试多跳搜索"""
        import su_memory.integrations.langchain as mod
        mod.LANGCHAIN_AVAILABLE = True
        try:
            from su_memory.integrations.langchain import SuMemoryRetriever
            original_doc = mod.Document
            mock_doc = MagicMock()
            mock_doc.side_effect = lambda page_content, metadata: MagicMock(
                page_content=page_content, metadata=metadata)
            mod.Document = mock_doc

            mock_client.query_multihop.return_value = [
                {"content": "多跳记忆", "id": "mh1", "score": 0.9,
                 "hops": 2, "path": ["a", "b", "c"],
                 "causal_type": "causes", "metadata": {}}
            ]

            retriever = SuMemoryRetriever(mock_client)
            docs = retriever._search_multihop("多跳查询", 5)

            mock_client.query_multihop.assert_called_once_with("多跳查询", max_hops=3)
            assert len(docs) == 1
            mod.Document = original_doc
        finally:
            mod.LANGCHAIN_AVAILABLE = False

    def test_search_multihop_fallback(self, mock_client):
        """测试多跳搜索回退到相似度搜索"""
        import su_memory.integrations.langchain as mod
        mod.LANGCHAIN_AVAILABLE = True
        try:
            from su_memory.integrations.langchain import SuMemoryRetriever
            original_doc = mod.Document
            mock_doc = MagicMock()
            mock_doc.side_effect = lambda page_content, metadata: MagicMock(
                page_content=page_content, metadata=metadata)
            mod.Document = mock_doc

            del mock_client.query_multihop

            retriever = SuMemoryRetriever(mock_client)
            docs = retriever._search_multihop("查询", 5)

            mock_client.query.assert_called()
            # 阈值0.5过滤低分
            assert len(docs) == 2
            mod.Document = original_doc
        finally:
            mod.LANGCHAIN_AVAILABLE = False


class TestSuMemoryLoader:
    """SuMemoryLoader 加载器测试"""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.get_all_memories.return_value = [
            {"content": "记忆A", "id": "a1", "timestamp": 1000,
             "metadata": {"session_id": "s1"}},
            {"content": "记忆B", "id": "a2", "timestamp": 2000,
             "metadata": {"session_id": "s1"}},
            {"content": "记忆C", "id": "a3", "timestamp": 3000,
             "metadata": {"session_id": "s2"}},
        ]
        return client

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    def test_init(self, mock_client):
        """测试初始化"""
        from su_memory.integrations.langchain import SuMemoryLoader
        loader = SuMemoryLoader(mock_client)
        assert loader._client == mock_client
        assert loader._session_id is None

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    def test_init_with_session(self, mock_client):
        """测试带会话ID初始化"""
        from su_memory.integrations.langchain import SuMemoryLoader
        loader = SuMemoryLoader(mock_client, session_id="s1")
        assert loader._session_id == "s1"

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', False)
    def test_raises_without_langchain(self):
        """测试无 LangChain 时抛错误"""
        from su_memory.integrations.langchain import SuMemoryLoader
        mock_client = MagicMock()
        with pytest.raises(ImportError, match="LangChain"):
            SuMemoryLoader(mock_client)

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    @patch('su_memory.integrations.langchain.Document')
    def test_load_all(self, mock_doc_class, mock_client):
        """测试加载所有记忆"""
        from su_memory.integrations.langchain import SuMemoryLoader
        mock_doc_class.side_effect = lambda page_content, metadata: MagicMock(
            page_content=page_content, metadata=metadata)

        loader = SuMemoryLoader(mock_client)
        docs = loader.load()

        mock_client.get_all_memories.assert_called_once()
        assert len(docs) == 3

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    @patch('su_memory.integrations.langchain.Document')
    def test_load_with_session_filter(self, mock_doc_class, mock_client):
        """测试会话过滤加载"""
        from su_memory.integrations.langchain import SuMemoryLoader
        mock_doc_class.side_effect = lambda page_content, metadata: MagicMock(
            page_content=page_content, metadata=metadata)

        loader = SuMemoryLoader(mock_client, session_id="s2")
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].page_content == "记忆C"

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    def test_load_no_get_all_memories(self):
        """测试无方法时返回空"""
        from su_memory.integrations.langchain import SuMemoryLoader
        client = MagicMock()
        del client.get_all_memories

        loader = SuMemoryLoader(client)
        docs = loader.load()
        assert docs == []

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    @patch('su_memory.integrations.langchain.Document')
    def test_lazy_load(self, mock_doc_class, mock_client):
        """测试懒加载"""
        from su_memory.integrations.langchain import SuMemoryLoader
        mock_doc_class.side_effect = lambda page_content, metadata: MagicMock(
            page_content=page_content, metadata=metadata)

        loader = SuMemoryLoader(mock_client)
        docs = list(loader.lazy_load())

        assert len(docs) == 3

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    @patch('su_memory.integrations.langchain.Document')
    def test_lazy_load_with_session(self, mock_doc_class, mock_client):
        """测试懒加载带会话过滤"""
        from su_memory.integrations.langchain import SuMemoryLoader
        mock_doc_class.side_effect = lambda page_content, metadata: MagicMock(
            page_content=page_content, metadata=metadata)

        loader = SuMemoryLoader(mock_client, session_id="s1")
        docs = list(loader.lazy_load())

        assert len(docs) == 2

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    def test_lazy_load_no_method(self):
        """测试懒加载无方法"""
        from su_memory.integrations.langchain import SuMemoryLoader
        client = MagicMock()
        del client.get_all_memories

        loader = SuMemoryLoader(client)
        docs = list(loader.lazy_load())
        assert docs == []


class TestSuMemoryTool:
    """SuMemoryTool 工具测试"""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.query.return_value = [
            {"content": "相关记忆1", "score": 0.95},
            {"content": "相关记忆2", "score": 0.80},
        ]
        return client

    def test_init_default(self, mock_client):
        """测试默认初始化"""
        import su_memory.integrations.langchain as mod
        mod.LANGCHAIN_AVAILABLE = True
        try:
            from su_memory.integrations.langchain import SuMemoryTool
            tool = SuMemoryTool(mock_client)
            assert tool._client == mock_client
            assert tool._name == "memory_search"
            assert "记忆" in tool._description
        finally:
            mod.LANGCHAIN_AVAILABLE = False

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    def test_init_custom_name(self, mock_client):
        """测试自定义名称和描述"""
        from su_memory.integrations.langchain import SuMemoryTool
        tool = SuMemoryTool(
            mock_client,
            name="custom_tool",
            description="自定义搜索工具"
        )
        assert tool.name == "custom_tool"
        assert tool.description == "自定义搜索工具"

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', False)
    def test_raises_without_langchain(self):
        """测试无 LangChain 抛错"""
        from su_memory.integrations.langchain import SuMemoryTool
        mock_client = MagicMock()
        with pytest.raises(ImportError, match="LangChain"):
            SuMemoryTool(mock_client)

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    def test_name_property(self, mock_client):
        """测试 name 属性"""
        from su_memory.integrations.langchain import SuMemoryTool
        tool = SuMemoryTool(mock_client, name="test_tool")
        assert tool.name == "test_tool"

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    def test_description_property(self, mock_client):
        """测试 description 属性"""
        from su_memory.integrations.langchain import SuMemoryTool
        tool = SuMemoryTool(mock_client, description="测试描述")
        assert tool.description == "测试描述"

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    def test_invoke_with_results(self, mock_client):
        """测试 invoke 返回搜索结果"""
        from su_memory.integrations.langchain import SuMemoryTool
        tool = SuMemoryTool(mock_client)

        result = tool.invoke("搜索关键词")

        mock_client.query.assert_called_once_with("搜索关键词", top_k=5)
        assert "找到 2 条相关记忆" in result
        assert "相关记忆1" in result
        assert "相关记忆2" in result

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    def test_invoke_empty_query(self, mock_client):
        """测试空查询"""
        from su_memory.integrations.langchain import SuMemoryTool
        tool = SuMemoryTool(mock_client)

        result = tool.invoke("  ")
        assert "请提供搜索关键词" in result
        mock_client.query.assert_not_called()

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    def test_invoke_no_results(self, mock_client):
        """测试无结果"""
        from su_memory.integrations.langchain import SuMemoryTool
        mock_client.query.return_value = []
        tool = SuMemoryTool(mock_client)

        result = tool.invoke("不存在的关键词")
        assert "没有找到" in result

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    def test_invoke_query_error(self, mock_client):
        """测试查询异常"""
        from su_memory.integrations.langchain import SuMemoryTool
        mock_client.query.side_effect = RuntimeError("数据库错误")
        tool = SuMemoryTool(mock_client)

        result = tool.invoke("查询")
        assert "搜索失败" in result
        assert "数据库错误" in result

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    def test_run_delegates_to_invoke(self, mock_client):
        """测试 run 委托给 invoke"""
        from su_memory.integrations.langchain import SuMemoryTool
        tool = SuMemoryTool(mock_client)

        result = tool.run("测试")
        assert "找到 2 条相关记忆" in result

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    def test_async_run_delegates_to_invoke(self, mock_client):
        """测试 async_run 委托给 invoke"""
        from su_memory.integrations.langchain import SuMemoryTool
        tool = SuMemoryTool(mock_client)

        result = tool.async_run("测试")
        assert "找到 2 条相关记忆" in result

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    def test_invoke_score_formatting(self, mock_client):
        """测试分数格式化"""
        from su_memory.integrations.langchain import SuMemoryTool
        tool = SuMemoryTool(mock_client)

        result = tool.invoke("查询")
        assert "[0.95]" in result
        assert "[0.80]" in result


class TestSuMemoryMemory:
    """SuMemoryMemory 记忆组件测试"""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.query.return_value = [
            {"content": "历史记忆1"},
            {"content": "历史记忆2"},
        ]
        # Mock hasattr for clear_session
        client.clear_session = MagicMock()
        return client

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    def test_init_default(self, mock_client):
        """测试默认初始化"""
        from su_memory.integrations.langchain import SuMemoryMemory
        memory = SuMemoryMemory(mock_client)
        assert memory._client == mock_client
        assert memory._session_id is None
        assert memory._return_messages is False
        assert memory._input_key == "input"
        assert memory._output_key == "output"

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    def test_init_custom(self, mock_client):
        """测试自定义初始化"""
        from su_memory.integrations.langchain import SuMemoryMemory
        memory = SuMemoryMemory(
            mock_client,
            session_id="s1",
            return_messages=True,
            input_key="query",
            output_key="answer"
        )
        assert memory._session_id == "s1"
        assert memory._return_messages is True
        assert memory._input_key == "query"
        assert memory._output_key == "answer"

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', False)
    def test_raises_without_langchain(self):
        """测试无 LangChain 抛错"""
        from su_memory.integrations.langchain import SuMemoryMemory
        mock_client = MagicMock()
        with pytest.raises(ImportError, match="LangChain"):
            SuMemoryMemory(mock_client)

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    def test_load_memory_variables_with_query(self, mock_client):
        """测试有查询时加载记忆"""
        from su_memory.integrations.langchain import SuMemoryMemory
        memory = SuMemoryMemory(mock_client)

        result = memory.load_memory_variables({"input": "查询内容"})

        mock_client.query.assert_called_once_with("查询内容", top_k=5)
        assert "history" in result
        assert "历史记忆1" in result["history"]

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    def test_load_memory_variables_empty_input(self, mock_client):
        """测试空输入时返回空"""
        from su_memory.integrations.langchain import SuMemoryMemory
        memory = SuMemoryMemory(mock_client)

        result = memory.load_memory_variables({})
        assert result == {"history": ""}
        mock_client.query.assert_not_called()

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    def test_load_memory_variables_none_input(self, mock_client):
        """测试 None 输入时的处理"""
        from su_memory.integrations.langchain import SuMemoryMemory
        memory = SuMemoryMemory(mock_client)

        result = memory.load_memory_variables(inputs=None)
        assert result == {"history": ""}

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    def test_load_memory_variables_return_messages(self, mock_client):
        """测试 return_messages 模式"""
        from su_memory.integrations.langchain import SuMemoryMemory
        memory = SuMemoryMemory(mock_client, return_messages=True)

        result = memory.load_memory_variables({"input": "查询"})

        assert "history" in result
        assert isinstance(result["history"], list)
        assert len(result["history"]) == 2
        assert result["history"][0]["type"] == "human"

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    def test_save_context(self, mock_client):
        """测试保存上下文"""
        from su_memory.integrations.langchain import SuMemoryMemory
        memory = SuMemoryMemory(mock_client)

        memory.save_context(
            inputs={"input": "用户问题"},
            outputs={"output": "助手回答"}
        )

        assert mock_client.add.call_count == 2

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    def test_save_context_with_custom_keys(self, mock_client):
        """测试自定义键的保存"""
        from su_memory.integrations.langchain import SuMemoryMemory
        memory = SuMemoryMemory(
            mock_client,
            input_key="query",
            output_key="response"
        )

        memory.save_context(
            inputs={"query": "用户问题"},
            outputs={"response": "助手回答"}
        )

        assert mock_client.add.call_count == 2
        # 检查第一次 add 的参数
        first_call = mock_client.add.call_args_list[0]
        assert "用户: 用户问题" in first_call[1]["content"]

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    def test_save_context_empty(self, mock_client):
        """测试空内容不保存"""
        from su_memory.integrations.langchain import SuMemoryMemory
        memory = SuMemoryMemory(mock_client)

        memory.save_context(
            inputs={"input": ""},
            outputs={"output": ""}
        )

        mock_client.add.assert_not_called()

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    def test_clear_with_session(self, mock_client):
        """测试清除会话记忆"""
        from su_memory.integrations.langchain import SuMemoryMemory
        memory = SuMemoryMemory(mock_client, session_id="s1")

        memory.clear()
        mock_client.clear_session.assert_called_once_with("s1")

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    def test_clear_no_session(self, mock_client):
        """测试无会话时不调用"""
        from su_memory.integrations.langchain import SuMemoryMemory
        memory = SuMemoryMemory(mock_client)

        memory.clear()
        # session_id 为 None，不调用 clear_session
        mock_client.clear_session.assert_not_called()

    @patch('su_memory.integrations.langchain.LANGCHAIN_AVAILABLE', True)
    def test_clear_no_method(self, mock_client):
        """测试无 clear_session 方法时不崩溃"""
        from su_memory.integrations.langchain import SuMemoryMemory
        del mock_client.clear_session

        memory = SuMemoryMemory(mock_client, session_id="s1")
        # 不应崩溃
        memory.clear()


class TestFactoryFunctions:
    """工厂函数测试"""

    @pytest.fixture
    def mock_client(self):
        return MagicMock()

    @pytest.fixture
    def mock_llm(self):
        return MagicMock()

    def test_create_rag_chain_raises_without_langchain(self, mock_client, mock_llm):
        """测试无 LangChain 时 create_rag_chain 抛错"""
        from su_memory.integrations.langchain import create_rag_chain
        with pytest.raises(ImportError, match="LangChain"):
            create_rag_chain(mock_client, mock_llm)

    def test_create_conversational_chain_raises_without_langchain(self,
                                                                  mock_client, mock_llm):
        """测试无 LangChain 时 create_conversational_chain 抛错"""
        from su_memory.integrations.langchain import create_conversational_chain
        with pytest.raises(ImportError, match="LangChain"):
            create_conversational_chain(mock_client, mock_llm)

    def test_create_rag_chain_success(self, mock_client, mock_llm):
        """测试 create_rag_chain 成功创建"""
        import su_memory.integrations.langchain as mod
        mod.LANGCHAIN_AVAILABLE = True
        # Mock langchain.chains.RetrievalQA
        mock_retrieval_qa = MagicMock()
        mock_chain = MagicMock()
        mock_retrieval_qa.from_chain_type.return_value = mock_chain
        mock_chains = MagicMock()
        mock_chains.RetrievalQA = mock_retrieval_qa
        with patch.dict('sys.modules', {'langchain': MagicMock(),
                                         'langchain.chains': mock_chains}):
            from su_memory.integrations.langchain import create_rag_chain
            result = create_rag_chain(
                mock_client, mock_llm,
                chain_type="stuff",
                search_type="similarity",
                top_k=5
            )
            assert result == mock_chain
        mod.LANGCHAIN_AVAILABLE = False

    def test_create_conversational_chain_success(self, mock_client, mock_llm):
        """测试 create_conversational_chain 成功创建"""
        import su_memory.integrations.langchain as mod
        mod.LANGCHAIN_AVAILABLE = True
        mock_conv = MagicMock()
        mock_chain = MagicMock()
        mock_conv.return_value = mock_chain
        mock_chains = MagicMock()
        mock_chains.ConversationChain = mock_conv
        mock_prompts = MagicMock()
        mock_prompts.PromptTemplate = MagicMock()
        with patch.dict('sys.modules', {
            'langchain': MagicMock(),
            'langchain.chains': mock_chains,
            'langchain.prompts': mock_prompts,
        }):
            from su_memory.integrations.langchain import create_conversational_chain
            result = create_conversational_chain(mock_client, mock_llm)
            assert result == mock_chain
        mod.LANGCHAIN_AVAILABLE = False

    def test_create_conversational_chain_with_system_prompt(
            self, mock_client, mock_llm):
        """测试带系统提示的 create_conversational_chain"""
        import su_memory.integrations.langchain as mod
        mod.LANGCHAIN_AVAILABLE = True
        mock_conv = MagicMock()
        mock_chain = MagicMock()
        mock_conv.return_value = mock_chain
        mock_chains = MagicMock()
        mock_chains.ConversationChain = mock_conv
        mock_prompts = MagicMock()
        mock_prompts.PromptTemplate = MagicMock()
        with patch.dict('sys.modules', {
            'langchain': MagicMock(),
            'langchain.chains': mock_chains,
            'langchain.prompts': mock_prompts,
        }):
            from su_memory.integrations.langchain import create_conversational_chain
            result = create_conversational_chain(
                mock_client, mock_llm,
                system_prompt="你是专业营养师"
            )
            assert result == mock_chain
        mod.LANGCHAIN_AVAILABLE = False


class TestModuleExports:
    """模块导出完整性测试"""

    def test_all_exports_exist(self):
        """测试 __all__ 导出存在"""
        from su_memory.integrations import langchain as module
        for name in module.__all__:
            assert hasattr(module, name), f"导出 {name} 不存在"

    def test_langchain_available_flag(self):
        """测试 LANGCHAIN_AVAILABLE 标志"""
        from su_memory.integrations.langchain import LANGCHAIN_AVAILABLE
        assert isinstance(LANGCHAIN_AVAILABLE, bool)

    def test_export_count(self):
        """测试导出数量"""
        from su_memory.integrations.langchain import __all__
        assert len(__all__) == 8
        expected = [
            "SuMemoryRetriever",
            "SuMemoryRetrieverConfig",
            "SuMemoryLoader",
            "SuMemoryTool",
            "SuMemoryMemory",
            "create_rag_chain",
            "create_conversational_chain",
            "LANGCHAIN_AVAILABLE",
        ]
        for name in expected:
            assert name in __all__, f"应导出 {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
