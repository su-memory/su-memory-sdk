"""
订单管理服务

负责订单的创建、查询、退款、持久化，以及 License Key 的生成。
订单数据通过 JSON 文件进行持久化存储。

订单号格式: SM-{timestamp}-{random_hex}
License Key 格式: SM-{plan_prefix}-{timestamp}-{random_hex}

架构说明：
- 此模块仅负责本地订单记录管理，不包含支付网关集成代码
- 支付由 Pipedream 工作流处理：前端调用 Pipedream 创建支付宝订单，
  支付宝异步通知直接发送到 Pipedream，Pipedream 生成 License Key 并发送邮件
- SDK 不持有任何支付密钥，所有支付签名在 Pipedream 服务端完成
"""

import json
import os
import random
import string
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List

# 套餐价格映射（Community 免费，Starter 及以上付费）
PLAN_PRICES: Dict[str, float] = {
    "community": 0.0,
    "starter": 29.9,
    "pro": 99.9,
    "enterprise": 399.0,
    "on_premise": 9999.0,
}

# 套餐显示名称
PLAN_NAMES: Dict[str, str] = {
    "community": "社区版 Community",
    "starter": "入门版 Starter",
    "pro": "专业版 Pro",
    "enterprise": "企业版 Enterprise",
    "on_premise": "私有部署版 On-Premise",
}

# 默认订单存储目录
DEFAULT_ORDER_DIR = Path.home() / ".su-memory" / "orders"

# Pipedream 工作流端点（用于创建支付宝订单 + 接收异步通知）
# 可在运行时通过环境变量 PIPEDREAM_BASE_URL 覆盖
DEFAULT_PIPEDREAM_BASE_URL = os.environ.get(
    "SU_MEMORY_PIPEDREAM_URL",
    "https://eoyjjsu9jrea1nh.m.pipedream.net"
)


class OrderStatus(str, Enum):
    """订单状态"""
    PENDING = "pending"       # 待支付
    PAID = "paid"             # 已支付
    CANCELLED = "cancelled"   # 已取消
    REFUNDED = "refunded"     # 已退款
    EXPIRED = "expired"       # 已过期
    CLOSED = "closed"         # 已关闭


class PlanType(str, Enum):
    """套餐类型"""
    COMMUNITY = "community"
    STARTER = "starter"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    ON_PREMISE = "on_premise"


@dataclass
class Order:
    """订单数据模型

    Attributes:
        order_id: 商户订单号 (out_trade_no)，格式 SM-{timestamp}-{random}
        plan_type: 套餐类型
        amount: 支付金额 (元)
        status: 订单状态
        license_key: 生成的 License Key (支付成功后由 Pipedream 生成并邮件发送)
        buyer_email: 买家邮箱
        trade_no: 支付宝交易号
        created_at: 创建时间 (ISO格式)
        updated_at: 更新时间 (ISO格式)
        paid_at: 支付时间 (ISO格式)
        refund_amount: 退款金额
        refund_reason: 退款原因
        body: 订单描述
        metadata: 额外元数据
    """
    order_id: str
    plan_type: str
    amount: float
    status: str = OrderStatus.PENDING.value
    license_key: Optional[str] = None
    buyer_email: Optional[str] = None
    trade_no: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    paid_at: Optional[str] = None
    refund_amount: Optional[float] = None
    refund_reason: Optional[str] = None
    body: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Order":
        """从字典创建"""
        return cls(**data)


class OrderService:
    """订单管理服务

    管理订单的完整生命周期，包括创建、查询、退款和 License 生成。
    订单数据持久化到 JSON 文件。

    注意：此服务不处理支付。
    支付流程通过 Pipedream 工作流完成：
    1. 前端调用 Pipedream HTTP 端点创建支付宝订单
    2. 用户完成支付宝支付
    3. 支付宝异步通知 → Pipedream
    4. Pipedream 生成 License Key 并通过 Gmail 发送

    Example:
        >>> service = OrderService()
        >>>
        >>> # 创建本地订单记录
        >>> result = service.create_order("pro", buyer_email="user@example.com")
        >>> print(result["order_id"])
        >>> print(result["pipedream_create_url"])  # 前端调用此 URL 创建支付宝订单
        >>>
        >>> # 查询订单
        >>> order = service.get_order("SM-xxx")
        >>> print(order.status if order else "订单不存在")
        >>>
        >>> # 列出所有订单
        >>> orders = service.list_orders(status="paid")
    """

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        pipedream_base_url: Optional[str] = None,
    ):
        """
        Args:
            storage_dir: 订单存储目录，默认为 ~/.su-memory/orders/
            pipedream_base_url: Pipedream 工作流基础 URL，默认从环境变量读取
        """
        self._storage_dir = Path(storage_dir) if storage_dir else DEFAULT_ORDER_DIR
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._pipedream_base_url = (
            pipedream_base_url or DEFAULT_PIPEDREAM_BASE_URL
        ).rstrip("/")

    @staticmethod
    def _generate_order_id() -> str:
        """生成唯一订单号: SM-{timestamp_ms}-{6位随机hex}"""
        timestamp_ms = int(time.time() * 1000)
        random_hex = "".join(
            random.choices(string.hexdigits.lower(), k=6)
        )
        return f"SM-{timestamp_ms}-{random_hex}"

    @staticmethod
    def _generate_license_key(plan_type: str, order_id: str) -> str:
        """生成 License Key

        格式: SM-{plan_prefix}-{timestamp}-{8位随机hex}

        Args:
            plan_type: 套餐类型
            order_id: 订单号

        Returns:
            License Key 字符串
        """
        plan_prefix_map = {
            "community": "COM",
            "starter": "STD",
            "pro": "PRO",
            "enterprise": "ENT",
            "on_premise": "ONP",
        }
        prefix = plan_prefix_map.get(plan_type, "UNK")
        timestamp = int(time.time())
        random_hex = "".join(
            random.choices(string.hexdigits.upper(), k=8)
        )
        return f"SM-{prefix}-{timestamp:x}-{random_hex}"

    def _order_file_path(self, order_id: str) -> Path:
        """获取订单文件路径"""
        # 对文件名进行简单清理，防止路径遍历
        safe_id = "".join(c for c in order_id if c.isalnum() or c in "-_")
        return self._storage_dir / f"{safe_id}.json"

    def _save_order(self, order: Order) -> None:
        """保存订单到文件"""
        with self._lock:
            filepath = self._order_file_path(order.order_id)
            order.updated_at = datetime.now(timezone.utc).isoformat()
            filepath.write_text(
                json.dumps(order.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def _load_order(self, order_id: str) -> Optional[Order]:
        """从文件加载订单"""
        filepath = self._order_file_path(order_id)
        if not filepath.exists():
            return None
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            return Order.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # 数据损坏，记录日志但不抛出异常
            import logging
            logging.getLogger(__name__).warning(
                "订单文件加载失败: %s, error=%s", order_id, str(e)
            )
            return None

    def create_order(
        self,
        plan_type: str,
        buyer_email: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """创建本地订单记录

        注意：此方法仅创建本地订单记录。
        实际支付需要前端调用 Pipedream 端点创建支付宝订单。
        返回结果中包含 pipedream_create_url 供前端使用。

        Args:
            plan_type: 套餐类型 (community/starter/pro/enterprise/on_premise)
            buyer_email: 买家邮箱
            metadata: 额外元数据

        Returns:
            包含 order_id、pipedream_create_url 等字段的字典

        Raises:
            ValueError: 无效的套餐类型
        """
        if plan_type not in PLAN_PRICES:
            raise ValueError(
                f"无效的套餐类型: {plan_type}。"
                f"有效值: {list(PLAN_PRICES.keys())}"
            )

        amount = PLAN_PRICES[plan_type]
        order_id = self._generate_order_id()
        plan_name = PLAN_NAMES.get(plan_type, plan_type)
        body = f"su-memory SDK {plan_name} - 订单号: {order_id}"

        # 创建本地订单
        order = Order(
            order_id=order_id,
            plan_type=plan_type,
            amount=amount,
            status=OrderStatus.PENDING.value,
            buyer_email=buyer_email,
            body=body,
            metadata=metadata or {},
        )
        self._save_order(order)

        result: Dict[str, Any] = {
            "order_id": order_id,
            "plan_type": plan_type,
            "amount": amount,
            "status": OrderStatus.PENDING.value,
        }

        # 付费套餐：提供 Pipedream 创建支付订单的 URL
        if amount > 0:
            result["pipedream_create_url"] = (
                f"{self._pipedream_base_url}/create-order"
            )
        else:
            result["pipedream_create_url"] = None

        return result

    def get_order(self, order_id: str) -> Optional[Order]:
        """查询订单

        Args:
            order_id: 订单号

        Returns:
            订单对象，不存在则返回 None
        """
        return self._load_order(order_id)

    def query_order_status(self, order_id: str) -> Dict[str, Any]:
        """查询订单状态

        仅从本地存储读取订单状态。
        支付状态由 Pipedream 支付宝异步通知更新。

        Args:
            order_id: 订单号

        Returns:
            订单状态信息字典
        """
        order = self._load_order(order_id)
        if not order:
            return {"error": "订单不存在", "order_id": order_id}

        return {
            "order": {
                "order_id": order.order_id,
                "plan": order.plan_type,
                "amount": order.amount,
                "status": order.status,
                "trade_no": order.trade_no,
                "license_key": order.license_key,
                "created_at": order.created_at,
                "paid_at": order.paid_at,
            }
        }

    def mark_order_paid(
        self,
        order_id: str,
        trade_no: Optional[str] = None,
        buyer_email: Optional[str] = None,
    ) -> Order:
        """标记订单为已支付 (由 Pipedream 支付宝异步通知触发)

        注意：License Key 由 Pipedream 生成并通过邮件发送给用户。
        此方法仅更新本地订单状态，并生成本地 License Key 副本。

        Args:
            order_id: 订单号
            trade_no: 支付宝交易号
            buyer_email: 买家邮箱

        Returns:
            更新后的订单

        Raises:
            ValueError: 订单不存在
        """
        order = self._load_order(order_id)
        if not order:
            raise ValueError(f"订单不存在: {order_id}")

        # 幂等处理：已经是终态则直接返回
        if order.status in (
            OrderStatus.PAID.value,
            OrderStatus.REFUNDED.value,
        ):
            return order

        order.status = OrderStatus.PAID.value
        order.trade_no = trade_no
        order.paid_at = datetime.now(timezone.utc).isoformat()
        if buyer_email:
            order.buyer_email = buyer_email

        # 生成 License Key
        if not order.license_key:
            order.license_key = self._generate_license_key(
                order.plan_type, order.order_id
            )

        self._save_order(order)
        return order

    def refund_order(
        self,
        order_id: str,
        refund_amount: Optional[float] = None,
        reason: str = "用户申请退款",
    ) -> Dict[str, Any]:
        """退款

        注意：退款需要在支付宝商家后台手动操作。
        此方法仅更新本地订单状态。

        Args:
            order_id: 订单号
            refund_amount: 退款金额 (默认全额退款)
            reason: 退款原因

        Returns:
            退款结果

        Raises:
            ValueError: 订单不存在或状态不允许退款
        """
        order = self._load_order(order_id)
        if not order:
            raise ValueError(f"订单不存在: {order_id}")

        if order.status != OrderStatus.PAID.value:
            raise ValueError(
                f"订单状态不允许退款: "
                f"当前状态={order.status}, 需要状态={OrderStatus.PAID.value}"
            )

        # 默认全额退款
        effective_amount = refund_amount if refund_amount is not None else order.amount

        # 更新本地订单状态
        order.status = OrderStatus.REFUNDED.value
        order.refund_amount = effective_amount
        order.refund_reason = reason
        self._save_order(order)

        return {
            "success": True,
            "order_id": order_id,
            "refund_amount": effective_amount,
            "note": "退款需在支付宝商家后台手动处理，此处仅更新本地记录",
        }

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """取消订单

        将本地订单标记为已取消。

        Args:
            order_id: 订单号

        Returns:
            取消结果
        """
        order = self._load_order(order_id)
        if not order:
            raise ValueError(f"订单不存在: {order_id}")

        if order.status != OrderStatus.PENDING.value:
            raise ValueError(
                f"只能取消待支付订单，当前状态: {order.status}"
            )

        order.status = OrderStatus.CANCELLED.value
        self._save_order(order)

        return {
            "success": True,
            "order_id": order_id,
            "status": OrderStatus.CANCELLED.value,
        }

    def list_orders(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """列出订单

        Args:
            status: 按状态过滤 (可选)
            limit: 最大返回数量
            offset: 偏移量

        Returns:
            订单列表
        """
        orders = []
        try:
            for filepath in sorted(
                self._storage_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                try:
                    data = json.loads(filepath.read_text(encoding="utf-8"))
                    order = Order.from_dict(data)

                    if status and order.status != status:
                        continue

                    orders.append({
                        "order_id": order.order_id,
                        "plan_type": order.plan_type,
                        "amount": order.amount,
                        "status": order.status,
                        "license_key": order.license_key,
                        "created_at": order.created_at,
                        "paid_at": order.paid_at,
                    })
                except Exception:
                    continue
        except Exception:
            pass

        return orders[offset:offset + limit]

    def generate_license_file(self, order_id: str) -> Optional[Dict[str, Any]]:
        """为已支付订单生成 License 文件

        生成的 License 文件可用于 su-memory SDK 离线激活。

        Args:
            order_id: 订单号

        Returns:
            License 数据字典，订单未支付则返回 None
        """
        order = self._load_order(order_id)
        if not order or order.status != OrderStatus.PAID.value:
            return None

        if not order.license_key:
            order.license_key = self._generate_license_key(
                order.plan_type, order.order_id
            )
            self._save_order(order)

        # 计算到期时间 (按年)
        from datetime import timedelta
        issued_at = datetime.now(timezone.utc)
        if order.plan_type == PlanType.ON_PREMISE.value:
            # 永久许可
            expires_at = "never"
        else:
            expires_at = (issued_at + timedelta(days=365)).isoformat()

        # 获取套餐对应的功能集
        from su_memory.licensing import CAPACITY_PACKAGES
        pkg = CAPACITY_PACKAGES.get(order.plan_type, CAPACITY_PACKAGES["community"])

        license_data = {
            "version": "1.0",
            "license_key": order.license_key,
            "license_type": order.plan_type,
            "capacity": pkg.memories if pkg.memories > 0 else None,
            "issued_to": order.buyer_email or "",
            "issued_at": issued_at.isoformat(),
            "expires": expires_at,
            "features": {f: True for f in pkg.features},
            "order_id": order.order_id,
        }

        return license_data
