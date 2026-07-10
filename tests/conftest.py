"""
pytest 配置文件
"""
import os
import sys

# 添加 src 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# P0-1 后：LLM 能量推断默认关闭（enable_llm_energy 默认 False）。
# 如需在测试中验证真实 LLM 能量推断，设置 SU_MEMORY_LLM_ENERGY=1
# （SuMemoryLitePro 构造时会读取该变量启用 LLM 路径）。
# 保留 NO_LLM 变量的清理以兼容旧调用方。
os.environ.pop("SU_MEMORY_NO_LLM_ENERGY", None)


def pytest_configure(config):
    """注册自定义 marker，避免 pytest warning。"""
    config.addinivalue_line(
        "markers", "integration: 依赖外部服务（Ollama/DeepSeek/Qdrant/Redis/PG），CI 默认跳过"
    )
    config.addinivalue_line(
        "markers", "slow: 运行时间较长的测试"
    )
