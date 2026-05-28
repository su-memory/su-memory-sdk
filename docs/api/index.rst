su-memory SDK 文档
====================

**让你的 AI 拥有记忆** — 本地优先的语义记忆引擎，LLM能量推断、多跳推理、FAISS向量检索。

.. toctree::
   :maxdepth: 2
   :caption: 核心模块

   su_memory.core
   su_memory.sdk
   su_memory.exceptions

.. toctree::
   :maxdepth: 2
   :caption: 内部系统

   su_memory._sys

.. toctree::
   :maxdepth: 2
   :caption: 基础设施

   su_memory.storage
   su_memory.embeddings
   su_memory.plugins
   su_memory.cli
   su_memory.integrations

.. toctree::
   :maxdepth: 1
   :caption: 其他

   changelog_link


快速开始
--------

.. code-block:: python

    from su_memory import SuMemory

    client = SuMemory()
    client.add("张总在周一会议上提到Q3目标增长25%")
    results = client.query("Q3目标")  # 秒级返回，带推理路径


版本信息
--------

当前版本：**v2.6.0**

.. list-table::
   :header-rows: 1

   * - 版本
     - 主要特性
     - 日期
   * - v2.6.0
     - 统一异常体系、降级矩阵、性能优化、FAISS 自动调参
     - 2026-04
   * - v2.5.0
     - AGI Continual Learning Loop、能量推断系统
     - 2026-05
   * - v2.0.1
     - 记忆生命周期管理、REST API
     - 2026-05
