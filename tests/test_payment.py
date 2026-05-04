"""
su-memory SDK 支付模块测试

测试支付宝企业支付（Pipedream 工作流）模式下的订单管理和 License 生成功能。
"""

import json
import os
import tempfile
from pathlib import Path
import pytest

from su_memory.payment.order_service import OrderService, OrderStatus, PlanType


class TestOrderService:
    """OrderService 核心功能测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试使用独立的临时目录"""
        self.temp_dir = tempfile.mkdtemp()
        self.service = OrderService(storage_dir=self.temp_dir)
        yield
        # 清理
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_order_community_free(self):
        """Community 是免费套餐，返回 pipedream_create_url=None"""
        result = self.service.create_order("community", buyer_email="test@example.com")
        assert result["plan_type"] == "community"
        assert result["amount"] == 0.0
        assert result["status"] == "pending"
        assert result["pipedream_create_url"] is None

    def test_create_order_starter_paid(self):
        """Starter 是付费套餐，返回 Pipedream 创建订单 URL"""
        result = self.service.create_order("starter", buyer_email="test@example.com")
        assert result["plan_type"] == "starter"
        assert result["amount"] == 29.9
        assert result["status"] == "pending"
        assert "/create-order" in result["pipedream_create_url"]

    def test_create_order_pro_plan(self):
        """Pro 套餐返回 Pipedream 创建订单 URL"""
        result = self.service.create_order("pro", buyer_email="test@example.com")
        assert result["plan_type"] == "pro"
        assert result["amount"] == 99.9
        assert result["status"] == "pending"
        assert "/create-order" in result["pipedream_create_url"]

    def test_create_order_enterprise_plan(self):
        """Enterprise 套餐返回 Pipedream 创建订单 URL"""
        result = self.service.create_order("enterprise", buyer_email="test@example.com")
        assert result["plan_type"] == "enterprise"
        assert result["amount"] == 399.0
        assert "create-order" in result["pipedream_create_url"]

    def test_create_order_on_premise_plan(self):
        """On-Premise 套餐返回 Pipedream 创建订单 URL"""
        result = self.service.create_order("on_premise", buyer_email="test@example.com")
        assert result["plan_type"] == "on_premise"
        assert result["amount"] == 9999.0
        assert "create-order" in result["pipedream_create_url"]

    def test_create_order_invalid_plan(self):
        """无效套餐抛出 ValueError"""
        with pytest.raises(ValueError) as exc_info:
            self.service.create_order("invalid_plan")
        assert "无效的套餐类型" in str(exc_info.value)

    def test_get_order(self):
        """查询订单"""
        result = self.service.create_order("pro")
        order = self.service.get_order(result["order_id"])
        assert order is not None
        assert order.plan_type == "pro"
        assert order.amount == 99.9

    def test_get_order_not_found(self):
        """查询不存在的订单返回 None"""
        order = self.service.get_order("nonexistent-id")
        assert order is None

    def test_query_order_status(self):
        """查询订单状态"""
        result = self.service.create_order("enterprise")
        status = self.service.query_order_status(result["order_id"])
        assert "order" in status
        assert status["order"]["plan"] == "enterprise"
        assert status["order"]["status"] == "pending"

    def test_query_order_status_not_found(self):
        """查询不存在的订单"""
        status = self.service.query_order_status("nonexistent")
        assert "error" in status
        assert "订单不存在" in status["error"]

    def test_mark_order_paid(self):
        """标记订单为已支付"""
        result = self.service.create_order("pro")
        order = self.service.mark_order_paid(
            result["order_id"],
            trade_no="ALIPAY-202605021234567890",
            buyer_email="paid@example.com"
        )
        assert order.status == OrderStatus.PAID.value
        assert order.license_key is not None
        assert order.buyer_email == "paid@example.com"

    def test_mark_order_paid_idempotent(self):
        """幂等性：重复标记已支付订单"""
        result = self.service.create_order("pro")
        order1 = self.service.mark_order_paid(result["order_id"], trade_no="ALIPAY-1")
        order2 = self.service.mark_order_paid(result["order_id"], trade_no="ALIPAY-2")
        assert order1.license_key == order2.license_key

    def test_cancel_order(self):
        """取消订单"""
        result = self.service.create_order("pro")
        cancel_result = self.service.cancel_order(result["order_id"])
        assert cancel_result["success"] is True
        assert cancel_result["status"] == "cancelled"

        order = self.service.get_order(result["order_id"])
        assert order.status == "cancelled"

    def test_cancel_order_not_pending(self):
        """只能取消待支付订单"""
        result = self.service.create_order("pro")
        self.service.mark_order_paid(result["order_id"])
        with pytest.raises(ValueError) as exc_info:
            self.service.cancel_order(result["order_id"])
        assert "只能取消待支付订单" in str(exc_info.value)

    def test_refund_order(self):
        """退款（仅更新本地状态）"""
        result = self.service.create_order("enterprise")
        self.service.mark_order_paid(result["order_id"])
        refund_result = self.service.refund_order(
            result["order_id"],
            reason="用户申请退款"
        )
        assert refund_result["success"] is True
        assert refund_result["refund_amount"] == 399.0
        assert "支付宝" in refund_result["note"]

    def test_refund_non_paid_order(self):
        """未支付订单不能退款"""
        result = self.service.create_order("pro")
        with pytest.raises(ValueError) as exc_info:
            self.service.refund_order(result["order_id"])
        assert "不允许退款" in str(exc_info.value)

    def test_list_orders(self):
        """列出订单"""
        self.service.create_order("pro")
        self.service.create_order("enterprise")
        orders = self.service.list_orders()
        assert len(orders) == 2

    def test_list_orders_filter_by_status(self):
        """按状态过滤订单"""
        result = self.service.create_order("pro")
        self.service.mark_order_paid(result["order_id"])
        self.service.create_order("enterprise")

        paid_orders = self.service.list_orders(status="paid")
        pending_orders = self.service.list_orders(status="pending")
        assert len(paid_orders) == 1
        assert len(pending_orders) == 1

    def test_license_key_generation(self):
        """License Key 格式正确"""
        result = self.service.create_order("pro")
        self.service.mark_order_paid(result["order_id"])
        order = self.service.get_order(result["order_id"])

        assert order.license_key is not None
        assert order.license_key.startswith("SM-PRO-")
        parts = order.license_key.split("-")
        assert len(parts) == 4

    def test_generate_license_file(self):
        """生成 License 文件"""
        result = self.service.create_order("pro")
        self.service.mark_order_paid(result["order_id"])

        license_data = self.service.generate_license_file(result["order_id"])
        assert license_data is not None
        assert "license_key" in license_data
        assert "license_type" in license_data
        assert license_data["license_type"] == "pro"

    def test_generate_license_file_not_paid(self):
        """未支付订单不能生成 License"""
        result = self.service.create_order("pro")
        license_data = self.service.generate_license_file(result["order_id"])
        assert license_data is None

    def test_generate_license_file_on_premise_permanent(self):
        """On-Premise 永久 License"""
        result = self.service.create_order("on_premise")
        self.service.mark_order_paid(result["order_id"])

        license_data = self.service.generate_license_file(result["order_id"])
        assert license_data["expires"] == "never"

    def test_order_persistence(self):
        """订单持久化到文件"""
        result = self.service.create_order("enterprise")
        order_id = result["order_id"]

        # 创建新的 service 实例，应该能加载到之前的订单
        new_service = OrderService(storage_dir=self.temp_dir)
        order = new_service.get_order(order_id)
        assert order is not None
        assert order.plan_type == "enterprise"

    def test_pipedream_base_url_config(self):
        """Pipedream URL 可自定义配置"""
        custom_url = "https://custom-pipedream.example.com"
        service = OrderService(
            storage_dir=self.temp_dir,
            pipedream_base_url=custom_url
        )
        result = service.create_order("pro")
        assert result["pipedream_create_url"].startswith(custom_url)


class TestOrderStatus:
    """OrderStatus 枚举测试"""

    def test_order_status_values(self):
        """OrderStatus 枚举值正确"""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.PAID.value == "paid"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REFUNDED.value == "refunded"
        assert OrderStatus.EXPIRED.value == "expired"
        assert OrderStatus.CLOSED.value == "closed"


class TestPlanType:
    """PlanType 枚举测试"""

    def test_plan_type_values(self):
        """PlanType 枚举值正确"""
        assert PlanType.COMMUNITY.value == "community"
        assert PlanType.STARTER.value == "starter"
        assert PlanType.PRO.value == "pro"
        assert PlanType.ENTERPRISE.value == "enterprise"
        assert PlanType.ON_PREMISE.value == "on_premise"
