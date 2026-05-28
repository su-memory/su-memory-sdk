su\_memory.exceptions
==========================

统一异常体系 — ErrorCode 枚举 + SuMemoryError 基类。

.. automodule:: su_memory.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

错误码分类
----------

.. list-table:: ErrorCode 枚举 (42 个)
   :header-rows: 1

   * - 分类
     - 代码范围
     - 数量
   * - FAISS 向量索引
     - FAISS\_E001-E005
     - 5
   * - 嵌入服务
     - EMB\_E001-E005
     - 5
   * - 存储
     - STO\_E001-E004
     - 4
   * - 查询
     - QRY\_E001-E003, QRY\_W001-W003
     - 6
   * - 图谱
     - GPH\_E001-E003
     - 3
   * - 并发
     - CON\_E001-E002
     - 2
   * - 配置
     - CFG\_E001-E003
     - 3
   * - 时序
     - TMP\_E001-E002
     - 2
   * - 会话
     - SES\_E001-E002
     - 2
   * - 插件
     - PLG\_E001-E003
     - 3
   * - 数据迁移
     - MIG\_E001-E002
     - 2
   * - 记忆管理
     - MEM\_E001-E003 (含1W)
     - 3
   * - 预测
     - PRD\_E001-E002
     - 2
