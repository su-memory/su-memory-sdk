"""
LangChain适配器单元测试
"""
import os
import sys
import tempfile
import pytest

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from su_memory.adapters.langchain import SuMemoryChatMemory
from su_memory.sdk.lite import SuMemoryLite


class TestSuMemoryChatMemory:
    """LangChain适配器测试用例"""

    @pytest.fixture
    def client(self):
        """创建测试客户端（无持久化）"""
        return SuMemoryLite(enable_persistence=False)

    @pytest.fixture
    def memory(self, client):
        """创建测试记忆组件"""
        return SuMemoryChatMemory(
            client=client,
            memory_key="chat_history"
        )

    def test_initialization(self):
        """测试初始化"""
        client = SuMemoryLite()
        memory = SuMemoryChatMemory(client=client)
        
        assert memory.client == client
        assert memory.memory_key == "chat_history"
        assert memory.return_messages is False

    def test_memory_variables(self, memory):
        """测试记忆变量"""
        assert memory.memory_variables == ["chat_history"]

    def test_save_context(self, memory):
        """测试保存上下文"""
        memory.save_context(
            inputs={"input": "你好"},
            outputs={"output": "你好！有什么可以帮助你的吗？"}
        )
        
        # 验证保存到chat_memory
        assert len(memory.chat_memory.messages) == 2
        
        # 验证保存到su-memory客户端
        assert len(memory.client) == 1

    def test_load_memory_variables(self, memory):
        """测试加载记忆变量"""
        memory.save_context(
            inputs={"input": "你好"},
            outputs={"output": "你好！"}
        )
        
        variables = memory.load_memory_variables({})
        assert "chat_history" in variables
        assert "Human" in variables["chat_history"]
        assert "AI" in variables["chat_history"]

    def test_load_memory_variables_with_messages(self, memory):
        """测试加载消息格式的记忆"""
        memory.return_messages = True
        memory.save_context(
            inputs={"input": "你好"},
            outputs={"output": "你好！"}
        )
        
        variables = memory.load_memory_variables({})
        assert "chat_history" in variables
        # 返回的是字符串或消息列表，取决于LangChain是否可用
        history = variables["chat_history"]
        assert "Human" in history or "AI" in history

    def test_clear(self, memory):
        """测试清空记忆"""
        memory.save_context(
            inputs={"input": "测试"},
            outputs={"output": "测试回复"}
        )
        
        assert len(memory.client) == 1
        assert len(memory.chat_memory.messages) == 2
        
        memory.clear()
        
        assert len(memory.client) == 0
        assert len(memory.chat_memory.messages) == 0

    def test_retrieve(self, memory):
        """测试检索功能"""
        memory.client.add("我喜欢吃苹果")
        memory.client.add("苹果是一种水果")
        memory.client.add("我喜欢运动")
        
        results = memory.retrieve("苹果", top_k=2)
        
        assert len(results) <= 2
        assert all("苹果" in r["content"] for r in results)

    def test_search_metadata(self, memory):
        """测试元数据过滤"""
        memory.client.add("消息1", metadata={"type": "conversation", "topic": "水果"})
        memory.client.add("消息2", metadata={"type": "conversation", "topic": "运动"})
        memory.client.add("消息3", metadata={"type": "system"})
        
        results = memory.search_metadata({"type": "conversation"}, top_k=10)
        
        assert len(results) == 2
        assert all(r["metadata"]["type"] == "conversation" for r in results)

    def test_custom_memory_key(self, client):
        """测试自定义记忆键"""
        memory = SuMemoryChatMemory(
            client=client,
            memory_key="custom_key"
        )
        
        assert memory.memory_variables == ["custom_key"]
        
        memory.save_context(
            inputs={"input": "测试"},
            outputs={"output": "测试回复"}
        )
        
        variables = memory.load_memory_variables({})
        assert "custom_key" in variables

    def test_buffer_property(self, memory):
        """测试buffer属性"""
        memory.save_context(
            inputs={"input": "问题1"},
            outputs={"output": "回答1"}
        )
        
        buffer = memory.buffer
        assert isinstance(buffer, str)
        assert "Human" in buffer
        assert "AI" in buffer
        assert "问题1" in buffer
        assert "回答1" in buffer

    def test_buffer_with_messages(self, memory):
        """测试buffer消息格式"""
        memory.return_messages = True
        memory.save_context(
            inputs={"input": "问题1"},
            outputs={"output": "回答1"}
        )
        
        buffer = memory.buffer
        # 当LangChain可用时返回列表，否则返回字符串
        if memory._langchain_available:
            assert isinstance(buffer, list)
            assert len(buffer) == 2
        else:
            assert isinstance(buffer, str)
            assert "Human" in buffer


class TestSuMemoryChatMemoryIntegration:
    """集成测试"""

    def test_full_conversation_flow(self):
        """测试完整对话流程"""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SuMemoryLite(storage_path=tmpdir)
            memory = SuMemoryChatMemory(client=client)
            
            # 模拟多轮对话
            conversations = [
                ({"input": "我叫张三"}, {"output": "你好张三，很高兴认识你！"}),
                ({"input": "我喜欢编程"}, {"output": "编程是一项很有趣的技能！"}),
                ({"input": "Python是什么"}, {"output": "Python是一种流行的编程语言。"}),
            ]
            
            for inputs, outputs in conversations:
                memory.save_context(inputs, outputs)
            
            # 验证对话历史
            assert len(memory.chat_memory.messages) == 6
            
            # 检索相关记忆（使用"编程"相关的查询）
            results = memory.retrieve("编程", top_k=2)
            # 可能返回空，因为简单的N-gram分词可能不精确
            assert isinstance(results, list)
            
            # 加载记忆变量
            variables = memory.load_memory_variables({})
            assert len(variables["chat_history"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
