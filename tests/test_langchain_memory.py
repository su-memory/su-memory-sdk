"""
LangChain Agent 语义记忆组件测试 — P3-S1 验证
"""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("SU_MEMORY_SKIP_ENV_CHECK", "1")
os.environ.setdefault("MEMORY_EMBEDDING_BACKEND", "none")


class TestSemanticAgentMemory:
    """语义记忆组件测试"""

    def test_save_and_load(self, tmp_path):
        """保存对话上下文后可语义召回"""
        from su_memory.clinical import ClinicalMemoryClient, SemanticAgentMemory

        client = ClinicalMemoryClient(
            storage_path=str(tmp_path / "lc_test"),
            embedding_backend="none",
        )
        memory = SemanticAgentMemory(patient_id="P001", client=client)

        # 保存一轮对话
        memory.save_context(
            {"input": "患者白蛋白偏低"},
            {"output": "建议高蛋白饮食，每日1.2g/kg"},
        )

        # 加载记忆 — 语义召回
        result = memory.load_memory_variables({"input": "白蛋白营养方案"})
        assert "chat_history" in result
        assert len(result["chat_history"]) > 10  # 有内容

    def test_patient_isolation(self, tmp_path):
        """不同患者的记忆隔离"""
        from su_memory.clinical import ClinicalMemoryClient, SemanticAgentMemory

        client = ClinicalMemoryClient(
            storage_path=str(tmp_path / "iso_lc"),
            embedding_backend="none",
        )
        mem_a = SemanticAgentMemory(patient_id="P001", client=client)
        mem_b = SemanticAgentMemory(patient_id="P002", client=client)

        mem_a.save_context({"input": "P001专属问题"}, {"output": "P001专属回答"})
        mem_b.save_context({"input": "P002专属问题"}, {"output": "P002专属回答"})

        result_a = mem_a.load_memory_variables({"input": "问题"})
        result_b = mem_b.load_memory_variables({"input": "问题"})

        assert "P001" in result_a["chat_history"]
        assert "P002" not in result_a["chat_history"]

    def test_memory_variables(self, tmp_path):
        """memory_variables 属性正确"""
        from su_memory.clinical import SemanticAgentMemory
        memory = SemanticAgentMemory(patient_id="P001")
        assert "chat_history" in memory.memory_variables

    def test_lab_summary_in_context(self, tmp_path):
        """加载记忆时附带检验趋势"""
        from su_memory.clinical import ClinicalMemoryClient, SemanticAgentMemory

        client = ClinicalMemoryClient(
            storage_path=str(tmp_path / "lab_lc"),
            embedding_backend="none",
        )
        # 写入异常检验值
        client.add_lab_value("P001", "白蛋白", 28.0, "g/L", "35-55")

        memory = SemanticAgentMemory(
            patient_id="P001", client=client, include_lab_summary=True
        )
        result = memory.load_memory_variables({"input": "test"})
        # 应包含检验趋势摘要
        assert "白蛋白" in result["chat_history"] or len(result["chat_history"]) > 0

    def test_lazy_client_init(self):
        """不传 client 时惰性初始化"""
        from su_memory.clinical import SemanticAgentMemory
        memory = SemanticAgentMemory(patient_id="P001")
        # 调用 _ensure_client 应不报错
        client = memory._ensure_client()
        assert client is not None

    def test_clear_no_error(self, tmp_path):
        """clear 不报错"""
        from su_memory.clinical import SemanticAgentMemory
        memory = SemanticAgentMemory(patient_id="P001")
        memory.clear()  # 不应抛异常
