#!/usr/bin/env python3
"""
su-memory-sdk Python API Server
@version 1.7.0

提供RESTful API接口供多语言SDK调用
支持: TypeScript, JavaScript, Rust等

Usage:
    python python_api_server.py
    python python_api_server.py --port 8080 --host 0.0.0.0
"""

import argparse
import json
import os
import sys
import uuid
import logging
from datetime import datetime
from typing import Any, Optional
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
_logger = logging.getLogger("api_server")

# 支付模块 - 可选导入
_payment_available = False
_AlipayConfig = None
_AlipayClient = None
_OrderService = None
_CallbackHandler = None
_CallbackResult = None

try:
    from su_memory.payment.alipay_config import AlipayConfig as _AlipayConfig
    from su_memory.payment.alipay_client import AlipayClient as _AlipayClient
    from su_memory.payment.order_service import OrderService as _OrderService
    from su_memory.payment.callback_handler import (
        CallbackHandler as _CallbackHandler,
        CallbackResult as _CallbackResult,
    )
    _payment_available = True
    _logger.info("支付模块加载成功")
except ImportError as e:
    _logger.warning("支付模块未加载: %s。支付端点不可用。", e)
except Exception as e:
    _logger.warning("支付模块初始化失败: %s", e)


@dataclass
class MemoryItem:
    """记忆条目"""
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    timestamp: int = field(default_factory=lambda: int(datetime.now().timestamp() * 1000))
    embedding: list = field(default_factory=list)


class MemoryStore:
    """内存存储"""
    
    def __init__(self):
        self._memories: dict[str, MemoryItem] = {}
        self._next_id = 1
        self._lock = threading.Lock()
    
    def add(self, content: str, metadata: Optional[dict] = None) -> str:
        """添加记忆"""
        with self._lock:
            memory_id = f"mem_{self._next_id}"
            self._next_id += 1
            
            memory = MemoryItem(
                id=memory_id,
                content=content,
                metadata=metadata or {},
                timestamp=int(datetime.now().timestamp() * 1000)
            )
            self._memories[memory_id] = memory
            return memory_id
    
    def add_batch(self, items: list[dict]) -> list[str]:
        """批量添加"""
        memory_ids = []
        for item in items:
            content = item.get('content', '')
            metadata = item.get('metadata')
            memory_id = self.add(content, metadata)
            memory_ids.append(memory_id)
        return memory_ids
    
    def query(self, query: str, top_k: int = 10) -> list[dict]:
        """查询记忆 (简单实现)"""
        with self._lock:
            results = []
            query_words = query.lower().split()
            
            for memory in self._memories.values():
                # 简单匹配: 内容包含关键词 + 元数据匹配
                score = 0.0
                content_lower = memory.content.lower()
                
                # 完全包含查询
                if query.lower() in content_lower:
                    score = 1.0
                else:
                    # 关键词匹配
                    for word in query_words:
                        if len(word) > 1 and word in content_lower:
                            score = max(score, 0.5)
                
                # 元数据关键词匹配
                if score == 0 and memory.metadata:
                    for value in memory.metadata.values():
                        if isinstance(value, str) and query.lower() in value.lower():
                            score = max(score, 0.3)
                
                if score > 0:
                    results.append({
                        "id": memory.id,
                        "content": memory.content,
                        "score": score,
                        "metadata": memory.metadata
                    })
            
            # 按分数排序
            results.sort(key=lambda x: -x['score'])
            return results[:top_k]
    
    def search(self, query: str, filters: Optional[dict] = None) -> list[dict]:
        """搜索记忆 (支持过滤器)"""
        with self._lock:
            results = []
            query_words = query.lower().split()
            
            for memory in self._memories.values():
                # 检查过滤器
                if filters:
                    if filters.get('category') and memory.metadata.get('category') != filters['category']:
                        continue
                    if filters.get('energyType') and memory.metadata.get('energyType') != filters['energyType']:
                        continue
                    if filters.get('timeRange'):
                        ts = memory.timestamp
                        if not (filters['timeRange']['start'] <= ts <= filters['timeRange']['end']):
                            continue
                    if filters.get('custom'):
                        for key, value in filters['custom'].items():
                            if memory.metadata.get(key) != value:
                                continue
                
                # 匹配评分
                score = 0.0
                content_lower = memory.content.lower()
                
                if query.lower() in content_lower:
                    score = 1.0
                else:
                    for word in query_words:
                        if len(word) > 1 and word in content_lower:
                            score = max(score, 0.5)
                
                if score > 0:
                    results.append({
                        "id": memory.id,
                        "content": memory.content,
                        "score": score,
                        "metadata": memory.metadata
                    })
            
            results.sort(key=lambda x: -x['score'])
            return results
    
    def get(self, memory_id: str) -> Optional[dict]:
        """获取单条记忆"""
        with self._lock:
            memory = self._memories.get(memory_id)
            if memory:
                return {
                    "id": memory.id,
                    "content": memory.content,
                    "metadata": memory.metadata,
                    "timestamp": memory.timestamp
                }
            return None
    
    def delete(self, memory_id: str) -> bool:
        """删除记忆"""
        with self._lock:
            if memory_id in self._memories:
                del self._memories[memory_id]
                return True
            return False
    
    def delete_batch(self, memory_ids: list[str]) -> int:
        """批量删除"""
        with self._lock:
            count = 0
            for memory_id in memory_ids:
                if memory_id in self._memories:
                    del self._memories[memory_id]
                    count += 1
            return count
    
    def clear(self) -> None:
        """清空所有记忆"""
        with self._lock:
            self._memories.clear()
            self._next_id = 1
    
    def get_stats(self) -> dict:
        """获取统计"""
        with self._lock:
            total = len(self._memories)
            category_dist: dict[str, int] = {}
            energy_dist: dict[str, int] = {}
            
            for memory in self._memories.values():
                cat = memory.metadata.get('category', 'unknown')
                energy = memory.metadata.get('energyType', 'earth')
                category_dist[cat] = category_dist.get(cat, 0) + 1
                energy_dist[energy] = energy_dist.get(energy, 0) + 1
            
            return {
                "count": total,
                "categoryDistribution": category_dist,
                "energyDistribution": energy_dist
            }
    
    def __len__(self) -> int:
        return len(self._memories)


class APIHandler(BaseHTTPRequestHandler):
    """API请求处理器"""
    
    store: MemoryStore = None  # 类属性，由主程序设置
    payment_config = None      # 支付配置
    payment_client = None      # 支付宝客户端
    payment_service = None     # 订单服务
    payment_callback = None    # 回调处理器
    
    def _send_json(self, data: Any, status: int = 200) -> None:
        """发送JSON响应"""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def _send_error(self, status: int, message: str) -> None:
        """发送错误响应"""
        self._send_json({"error": message, "code": f"HTTP_{status}"}, status)
    
    def _parse_body(self) -> Optional[dict]:
        """解析请求体"""
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length > 0:
            body = self.rfile.read(content_length)
            return json.loads(body.decode('utf-8'))
        return None
    
    def _handle_payment_callback(self) -> None:
        """处理支付宝异步通知回调
        
        支付宝通过 POST application/x-www-form-urlencoded 发送通知，
        需要特殊解析（非 JSON）。
        """
        if not _payment_available or not self.payment_callback:
            self._send_error(503, "支付服务不可用")
            return
        
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length <= 0:
            self._send_error(400, "Missing request body")
            return
        
        raw_body = self.rfile.read(content_length).decode('utf-8')
        
        # 支付宝通知是 URL-encoded 表单格式
        parsed_params = parse_qs(raw_body)
        # parse_qs 返回 {key: [value]}，取出第一个值
        params = {k: v[0] if isinstance(v, list) else v for k, v in parsed_params.items()}
        
        _logger.info("收到支付宝异步通知: out_trade_no=%s", params.get("out_trade_no", "unknown"))
        
        success, message, result_type = self.payment_callback.handle(params)
        
        # 支付宝要求返回纯文本 "success" 或 "fail"
        if success:
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.end_headers()
            self.wfile.write(b"success")
        else:
            self.send_response(400)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.end_headers()
            self.wfile.write(b"fail")
    
    def do_OPTIONS(self) -> None:
        """处理CORS预检请求"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
    
    def do_GET(self) -> None:
        """处理GET请求"""
        parsed = urlparse(self.path)
        path = parsed.path
        
        # 健康检查
        if path == '/api/health':
            self._send_json({"status": "ok", "version": "1.7.0"})
            return
        
        # 获取统计
        if path == '/api/memories/stats':
            stats = self.store.get_stats()
            self._send_json({"stats": stats})
            return
        
        # 获取单条记忆
        if path.startswith('/api/memories/'):
            memory_id = path[len('/api/memories/'):]
            memory = self.store.get(memory_id)
            if memory:
                self._send_json({"memory": memory})
            else:
                self._send_error(404, f"Memory not found: {memory_id}")
            return
        
        # === 支付端点 ===
        
        # 支付服务健康检查
        if path == '/api/payment/health':
            if not _payment_available or not self.payment_client:
                self._send_json({
                    "status": "unavailable",
                    "message": "支付模块未启用。请配置 ALIPAY_APP_ID 等环境变量。",
                })
                return
            health = self.payment_client.health_check()
            self._send_json(health)
            return
        
        # 查询订单状态
        if path.startswith('/api/payment/order/'):
            order_id = path[len('/api/payment/order/'):]
            if not _payment_available or not self.payment_service:
                self._send_error(503, "支付服务不可用")
                return
            if not order_id:
                self._send_error(400, "缺少订单号")
                return
            result = self.payment_service.query_order_status(order_id)
            if "error" in result:
                self._send_error(404, result["error"])
                return
            self._send_json(result)
            return
        
        self._send_error(404, "Not found")
    
    def do_POST(self) -> None:
        """处理POST请求"""
        parsed = urlparse(self.path)
        path = parsed.path
        
        # 支付宝异步通知回调 - 需要特殊处理（URL-encoded 表单）
        if path == '/api/payment/callback':
            self._handle_payment_callback()
            return
        
        body = self._parse_body()
        
        if body is None:
            self._send_error(400, "Missing request body")
            return
        
        # 添加记忆
        if path == '/api/memories/add':
            content = body.get('content')
            metadata = body.get('metadata')
            if not content:
                self._send_error(400, "content is required")
                return
            memory_id = self.store.add(content, metadata)
            self._send_json({"memoryId": memory_id}, 201)
            return
        
        # 批量添加
        if path == '/api/memories/add_batch':
            items = body.get('items', [])
            if not items:
                self._send_error(400, "items is required")
                return
            memory_ids = self.store.add_batch(items)
            self._send_json({"memoryIds": memory_ids}, 201)
            return
        
        # 查询
        if path == '/api/memories/query':
            query = body.get('query', '')
            top_k = body.get('top_k', 10)
            results = self.store.query(query, top_k)
            self._send_json({"results": results})
            return
        
        # 搜索
        if path == '/api/memories/search':
            query = body.get('query', '')
            filters = body.get('filters')
            results = self.store.search(query, filters)
            self._send_json({"results": results})
            return
        
        # 批量删除
        if path == '/api/memories/delete_batch':
            memory_ids = body.get('memoryIds', [])
            count = self.store.delete_batch(memory_ids)
            self._send_json({"deletedCount": count})
            return
        
        # === 支付端点 ===
        
        # 创建支付订单
        if path == '/api/payment/order':
            if not _payment_available or not self.payment_service:
                self._send_error(503, "支付服务不可用")
                return
            
            plan = body.get('plan', '').strip()
            return_url = body.get('return_url', '').strip() or None
            buyer_email = body.get('email', '').strip() or None
            metadata = body.get('metadata')
            
            if not plan:
                self._send_error(400, "缺少 plan 参数。有效值: starter, pro, enterprise, on_premise")
                return
            
            try:
                result = self.payment_service.create_order(
                    plan_type=plan,
                    return_url=return_url,
                    buyer_email=buyer_email,
                    metadata=metadata,
                )
                self._send_json(result, 201)
            except ValueError as e:
                self._send_error(400, str(e))
            except RuntimeError as e:
                self._send_error(500, str(e))
            except Exception as e:
                _logger.exception("创建订单失败")
                self._send_error(500, f"创建订单失败: {str(e)}")
            return
        
        # 退款
        if path == '/api/payment/refund':
            if not _payment_available or not self.payment_service:
                self._send_error(503, "支付服务不可用")
                return
            
            order_id = body.get('order_id', '').strip()
            reason = body.get('reason', '用户申请退款').strip()
            refund_amount = body.get('refund_amount')
            
            if not order_id:
                self._send_error(400, "缺少 order_id 参数")
                return
            
            try:
                result = self.payment_service.refund_order(
                    order_id=order_id,
                    refund_amount=refund_amount,
                    reason=reason,
                )
                if result.get("success"):
                    self._send_json(result)
                else:
                    self._send_error(400, result.get("error", "退款失败"))
            except ValueError as e:
                self._send_error(400, str(e))
            except Exception as e:
                _logger.exception("退款失败")
                self._send_error(500, f"退款失败: {str(e)}")
            return
        
        self._send_error(404, "Not found")
    
    def do_DELETE(self) -> None:
        """处理DELETE请求"""
        parsed = urlparse(self.path)
        path = parsed.path
        
        # 清空所有
        if path == '/api/memories/clear':
            self.store.clear()
            self._send_json({"success": True})
            return
        
        # 删除单条
        if path.startswith('/api/memories/'):
            memory_id = path[len('/api/memories/'):]
            success = self.store.delete(memory_id)
            if success:
                self._send_json({"success": True, "memoryId": memory_id})
            else:
                self._send_error(404, f"Memory not found: {memory_id}")
            return
        
        self._send_error(404, "Not found")
    
    def log_message(self, format: str, *args) -> None:
        """日志格式"""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {args[0]}")


def create_server(host: str = '0.0.0.0', port: int = 8080) -> HTTPServer:
    """创建API服务器"""
    APIHandler.store = MemoryStore()
    
    # 初始化支付模块（如果可用）
    if _payment_available:
        try:
            config = _AlipayConfig.from_env()
            client = _AlipayClient(config)
            service = _OrderService(client)
            callback = _CallbackHandler(client, service)
            
            APIHandler.payment_config = config
            APIHandler.payment_client = client
            APIHandler.payment_service = service
            APIHandler.payment_callback = callback
            
            _logger.info(
                "支付模块已初始化: %s, 沙箱=%s",
                config.safe_repr(),
                config.debug,
            )
        except ValueError as e:
            _logger.warning("支付模块配置缺失，支付端点不可用: %s", e)
        except Exception as e:
            _logger.warning("支付模块初始化失败: %s", e)
    
    server = HTTPServer((host, port), APIHandler)
    return server


def main():
    parser = argparse.ArgumentParser(description='su-memory SDK API Server v1.7.0')
    parser.add_argument('--host', default='0.0.0.0', help='Host地址 (默认: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8080, help='Port端口 (默认: 8080)')
    args = parser.parse_args()
    
    server = create_server(args.host, args.port)
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           su-memory SDK API Server v1.7.0                    ║
╠══════════════════════════════════════════════════════════════╣
║  状态:     ✅ 运行中                                         ║
║  地址:     http://{args.host}:{args.port}                          ║
║  端点:                                                     ║
║    POST   /api/memories/add        - 添加记忆                ║
║    POST   /api/memories/query     - 查询记忆                ║
║    POST   /api/memories/search    - 搜索记忆                ║
║    DELETE /api/memories/{{id}}      - 删除记忆                ║
║    GET    /api/memories/stats     - 获取统计                ║
║    DELETE /api/memories/clear    - 清空记忆                ║
║    GET    /api/health            - 健康检查                ║
╠══════════════════════════════════════════════════════════════╣
║  支付:     {'✅' if APIHandler.payment_service else '⚠️  未配置'}                                         ║
║    POST   /api/payment/order      - 创建支付订单            ║
║    GET    /api/payment/order/{{id}} - 查询订单              ║
║    POST   /api/payment/callback   - 支付宝异步通知          ║
║    POST   /api/payment/refund     - 退款                    ║
║    GET    /api/payment/health     - 支付健康检查            ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭服务器...")
        server.shutdown()


if __name__ == '__main__':
    main()
