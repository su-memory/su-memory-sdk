"""
pytest 配置文件
"""
import os
import sys

# 添加 src 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# 测试环境默认关闭 SuMemoryLitePro 的「逐条 LLM 能量推断」(每次 add ~0.8s 本地 LLM 调用)，
# 避免批量/性能测试被逐条 LLM 调用拖慢（50 条 ~42s）。
# 如需验证真实 LLM 能量推断，设置 SU_MEMORY_LLM_ENERGY=1（会清除此变量，恢复默认 LLM 行为）。
if os.environ.get("SU_MEMORY_LLM_ENERGY", ""):
    os.environ.pop("SU_MEMORY_NO_LLM_ENERGY", None)
else:
    os.environ.setdefault("SU_MEMORY_NO_LLM_ENERGY", "1")
