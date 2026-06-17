"""
pytest 配置文件 — su-memory SDK v3.5.5
======================================
提供全局 fixtures、markers 配置、路径设置、状态清理。
"""
import gc
import os
import sys
import tempfile
from pathlib import Path

import pytest

# 添加 src 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ============================================================
# 全局状态清理 (v3.5.5-p0)
# ============================================================

@pytest.fixture(autouse=True)
def _clean_global_state():
    """
    每个测试前后清理 SuMemoryLitePro 全局状态。

    防止测试间状态泄漏导致:
    - LifecycleManager 的 total_memories 膨胀
    - SOTA benchmark 的记忆数污染
    - FAISS 索引文件残留
    """
    # pre-test: 强制垃圾回收
    gc.collect()
    yield
    # post-test: 清理可能泄露的临时文件
    gc.collect()


# ============================================================
# 全局 Fixtures
# ============================================================

@pytest.fixture
def temp_dir():
    """临时目录 (自动清理)"""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def lite_client():
    """SuMemoryLitePro 测试客户端 (100条容量)"""
    from su_memory.sdk.lite_pro import SuMemoryLitePro
    return SuMemoryLitePro(max_memories=100, enable_graph=False)


@pytest.fixture
def lite_pro_client():
    """SuMemoryLitePro 测试客户端 (500条容量, 全特性)"""
    from su_memory.sdk.lite_pro import SuMemoryLitePro
    return SuMemoryLitePro(max_memories=500, enable_graph=True, enable_prediction=True)


@pytest.fixture
def sample_chinese_texts():
    """标准中文测试文本集"""
    return [
        "项目ROI增长了25%，其中Q3增长最为显著",
        "由于产品新功能上线，客户满意度提升至92%",
        "团队完成了3个核心模块的重构，代码质量评分从B提升至A",
        "市场调研显示竞品推出类似功能，需要关注竞争动态",
        "用户反馈批量导入功能存在性能问题，需要优化",
        "服务器在高峰时段CPU使用率达到85%，需要扩容",
        "A/B测试显示新UI设计转化率提升15%",
        "安全审计发现2个中危漏洞，已修复",
        "Q4预算获批，可以开始招聘3名新工程师",
        "数据迁移计划从PostgreSQL 14升级到16，预计耗时2周",
    ]
