"""
su-memory SDK 订单管理与 License 发放模块

支付通过支付宝企业支付完成，由 Pipedream 工作流处理：
1. 前端调用 Pipedream HTTP 端点创建支付宝订单
2. 用户完成支付宝支付
3. 支付宝异步通知 → Pipedream
4. Pipedream 生成 License Key 并通过 Gmail 发送

此模块提供：
- 本地订单管理（创建、查询、取消、退款）
- License Key 本地生成
- License 文件导出

套餐配置：
| 套餐 | 价格 | 支付方式 |
|------|------|----------|
| Community | ¥0 | 免费 |
| Starter | ¥29.9/月 | 支付宝 |
| Pro | ¥99.9/月 | 支付宝 |
| Enterprise | ¥399/月 | 支付宝 |
| On-Premise | ¥9999 | 支付宝 |

Example:
    >>> from su_memory.payment import OrderService
    >>>
    >>> service = OrderService()
    >>>
    >>> # 创建本地订单
    >>> result = service.create_order("pro", buyer_email="user@example.com")
    >>> print(result["order_id"])
    >>> print(result["pipedream_create_url"])  # 前端调用此 URL
    >>>
    >>> # 查询订单
    >>> order = service.get_order("SM-xxx")
    >>> print(order.status if order else "订单不存在")
    >>>
    >>> # 生成 License 文件
    >>> license = service.generate_license_file("SM-xxx")
    >>> if license:
    >>>     print(f"License: {license['license_key']}")
"""

from su_memory.payment.order_service import OrderService, OrderStatus, PlanType

__all__ = [
    "OrderService",
    "OrderStatus",
    "PlanType",
]
