su\_memory.sdk
================

SDK 核心模块 — 客户端、配置、推理引擎。

客户端
------

.. automodule:: su_memory.sdk
   :members:
   :undoc-members:

.. autoclass:: su_memory.sdk.lite.SuMemoryLite
   :members: add, query, add_batch, clear, get_all
   :undoc-members:
   :show-inheritance:

.. autoclass:: su_memory.sdk.lite_pro.SuMemoryLitePro
   :members: add, query, query_multihop, query_multihop_spacetime, predict, explain_query, forget, decay, summarize, link_memories
   :undoc-members:
   :show-inheritance:

多跳推理引擎
-----------

.. automodule:: su_memory.sdk.vector_graph_rag
   :members:
   :undoc-members:

时空索引
--------

.. automodule:: su_memory.sdk.spacetime_index
   :members:
   :undoc-members:

.. automodule:: su_memory.sdk.spacetime_multihop
   :members:
   :undoc-members:

多模态
------

.. automodule:: su_memory.sdk.multimodal
   :members:
   :undoc-members:

三维世界模型
-----------

.. automodule:: su_memory.sdk.spatial_rag
   :members:
   :undoc-members:

配置
----

.. automodule:: su_memory.sdk.config
   :members:
   :undoc-members:
