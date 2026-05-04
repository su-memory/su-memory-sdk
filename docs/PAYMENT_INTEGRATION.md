# su-memory SDK 支付宝支付接入文档

> **版本**: v1.7.0 | **更新日期**: 2026-05-03  
> **技术栈**: 支付宝电脑网站支付 (`alipay.trade.page.pay`) | Python 3.10+ | RSA2 签名

---

## 目录

1. [概述](#1-概述)
2. [架构设计](#2-架构设计)
3. [环境配置](#3-环境配置)
4. [快速开始](#4-快速开始)
5. [API 参考](#5-api-参考)
6. [套餐与价格](#6-套餐与价格)
7. [安全规范](#7-安全规范)
8. [测试指南](#8-测试指南)
9. [常见问题](#9-常见问题)

---

## 1. 概述

### 1.1 支付系统简介

su-memory SDK 集成了支付宝电脑网站支付能力，实现从订单创建到 License Key 自动发放的完整支付闭环。

**核心流程：**

```
用户选择套餐 → 创建订单 → 跳转支付宝 → 完成支付 → 异步通知 → 生成 License Key
```

**关键特性：**

| 特性 | 说明 |
|------|------|
| 支付方式 | 支付宝电脑网站支付 (`alipay.trade.page.pay`) |
| 签名算法 | RSA2 (SHA256) |
| 回调机制 | 异步通知 + 同步跳转 |
| 幂等处理 | 基于 `notify_id` 的重复通知检测 |
| License 发放 | 支付成功后自动生成 `SM-{PREFIX}-{timestamp}-{random}` 格式密钥 |
| 环境隔离 | 沙箱/正式环境一键切换 |

### 1.2 技术栈与依赖

| 组件 | 版本要求 | 用途 |
|------|---------|------|
| Python | ≥ 3.10 | 运行环境 |
| alipay-sdk-python | latest | 支付宝开放平台 SDK |
| JSON 文件存储 | — | 订单持久化 (无需数据库) |

### 1.3 系统模块

```
src/su_memory/payment/
├── alipay_config.py        # 支付宝配置管理 (从环境变量读取)
├── alipay_client.py        # 支付宝 API 客户端封装
├── order_service.py        # 订单创建、查询、退款、License 生成
└── callback_handler.py     # 异步通知验签与处理
```

---

## 2. 架构设计

### 2.1 支付流程图

```
┌──────────┐    ①POST /api/payment/order     ┌──────────────┐
│          │ ─────────────────────────────────> │              │
│  前端/   │ <───────────────────────────────── │  API Server  │
│  客户端  │     ②返回 payment_url (HTML)      │              │
│          │                                   └──────┬───────┘
│          │                                          │
│          │  ③渲染 HTML，跳转支付宝收银台          │
│  ┌───────▼───────┐                                  │
│  │  支付宝收银台  │                                  │
│  │  (用户付款)    │                                  │
│  └───────┬───────┘                                  │
│          │                                          │
│          │  ④用户完成支付                           │
│          │                                          │
│    ┌─────▼──────┐   ⑤POST /api/payment/callback    │
│    │  支付宝服务  │ ──────────────────────────────────> │
│    │  器         │ <────────────────────────────────── │
│    └────────────┘   ⑥返回 "success" / "fail"        │
│                                                      │
│                                        ┌─────────────▼──────┐
│                                        │  CallbackHandler    │
│                                        │  ① 验签             │
│                                        │  ② 幂等检查         │
│                                        │  ③ 金额校验         │
│                                        │  ④ 更新订单状态     │
│                                        │  ⑤ 生成 License Key │
│                                        └────────────────────┘
│
│  ⑦用户在 return_url 页面查看订单结果和 License Key
│
│  ⑧GET /api/payment/order/{id} 主动查询订单状态
```

### 2.2 组件职责说明

| 组件 | 类名 | 职责 |
|------|------|------|
| 配置管理 | `AlipayConfig` | 从环境变量读取 APPID、密钥、网关等配置；支持密钥字符串/文件路径双模式 |
| API 客户端 | `AlipayClient` | 封装支付宝核心接口：`page_pay`(下单)、`query`(查询)、`refund`(退款)、`close`(关闭)、`verify_signature`(验签) |
| 订单服务 | `OrderService` | 订单生命周期管理：创建、查询、退款、取消；License Key 生成与 License 文件导出 |
| 回调处理 | `CallbackHandler` | 异步通知的完整处理流程：验签 → 幂等检查 → 金额校验 → 状态更新 → License 发放 |
| API 路由 | `APIHandler` (python_api_server.py) | HTTP 端点路由，将请求分发到对应服务 |

### 2.3 订单状态机

```
                    ┌─────────┐
                    │ PENDING │  待支付 (初始状态)
                    └────┬────┘
                         │
              ┌──────────┼──────────┐
              │          │          │
              ▼          ▼          ▼
        ┌─────────┐ ┌─────────┐ ┌─────────┐
        │  PAID   │ │CANCELLED│ │ CLOSED  │
        │ 已支付   │ │ 已取消   │ │ 已关闭   │
        └────┬────┘ └─────────┘ └─────────┘
             │
             ▼
        ┌─────────┐
        │REFUNDED │  已退款
        └─────────┘
```

### 2.4 数据存储

订单数据以 JSON 文件形式持久化到本地文件系统，默认目录：

| 操作系统 | 默认路径 |
|---------|---------|
| Linux / macOS | `~/.su-memory/orders/` |
| Windows | `C:\Users\{用户名}\.su-memory\orders\` |

每个订单单独存储为一个 `{order_id}.json` 文件。已处理的通知记录存储在 `~/.su-memory/payment_notifications/`。

---

## 3. 环境配置

### 3.1 支付宝开放平台注册

> **前置条件**：已注册企业/个体工商户支付宝账号。

**步骤：**

1. 访问 [支付宝开放平台](https://open.alipay.com/)
2. 登录后进入「控制台」→「应用列表」
3. 创建应用，选择「网页/移动应用」
4. 在「产品绑定」中开通 **电脑网站支付** 功能
5. 记录 **APPID**（应用唯一标识）

### 3.2 配置密钥

#### 3.2.1 生成 RSA2 密钥对

```bash
# 生成 2048 位 RSA 私钥
openssl genrsa -out private_key.pem 2048

# 从私钥导出公钥
openssl rsa -in private_key.pem -pubout -out public_key.pem
```

#### 3.2.2 上传公钥到支付宝

1. 在应用详情页 →「开发设置」→「接口加签方式」
2. 将 `public_key.pem` 的内容上传
3. 支付宝会返回 **支付宝公钥**，保存为 `alipay_public_key.pem`

> ⚠️ **重要**：切勿混淆「应用公钥」和「支付宝公钥」。应用公钥 = 你生成的公钥（上传到支付宝），支付宝公钥 = 支付宝返回的公钥（用于验签）。

### 3.3 环境变量列表

| 环境变量 | 必填 | 说明 | 示例 |
|---------|------|------|------|
| `ALIPAY_APP_ID` | ✅ | 支付宝应用 ID | `2021000000000001` |
| `ALIPAY_PRIVATE_KEY` | 二选一 | 应用私钥内容 (PEM 字符串) | `-----BEGIN RSA PRIVATE KEY-----\n...` |
| `ALIPAY_PRIVATE_KEY_PATH` | 二选一 | 应用私钥文件路径 | `/etc/secrets/private_key.pem` |
| `ALIPAY_PUBLIC_KEY` | 二选一 | 支付宝公钥内容 (PEM 字符串) | `-----BEGIN PUBLIC KEY-----\n...` |
| `ALIPAY_PUBLIC_KEY_PATH` | 二选一 | 支付宝公钥文件路径 | `/etc/secrets/alipay_public_key.pem` |
| `ALIPAY_NOTIFY_URL` | ⚠️ 推荐 | 异步通知回调 URL | `https://your-domain.com/api/payment/callback` |
| `ALIPAY_RETURN_URL` | 可选 | 同步跳转 URL | `https://your-domain.com/payment/return` |
| `ALIPAY_SANDBOX` | 可选 | 沙箱模式开关 (默认 `true`) | `true` / `false` |
| `ALIPAY_GATEWAY` | 可选 | 自定义网关地址 (覆盖默认) | `https://...` |

**密钥读取优先级：**

```
环境变量字符串 (ALIPAY_PRIVATE_KEY) > 文件路径 (ALIPAY_PRIVATE_KEY_PATH)
```

> 💡 **容器部署建议**：使用环境变量字符串方式传入密钥，或通过 Kubernetes Secret / Docker Secret 挂载文件后使用文件路径方式。

### 3.4 沙箱 vs 正式环境

| 环境 | 网关地址 | `ALIPAY_SANDBOX` |
|------|---------|-------------------|
| 沙箱 | `https://openapi-sandbox.dl.alipaydev.com/gateway.do` | `true` (默认) |
| 正式 | `https://openapi.alipay.com/gateway.do` | `false` |

> ⚠️ 沙箱环境使用独立的 APPID 和密钥，正式上线前务必切换到正式环境配置。

---

## 4. 快速开始

### 4.1 安装依赖

```bash
# 安装 alipay-sdk-python
pip install alipay-sdk-python

# 验证安装
python -c "from alipay import AliPay; print('OK')"
```

### 4.2 配置环境变量

```bash
# 创建 .env 文件 (不要提交到版本控制)
cat > .env << 'EOF'
# 支付宝应用 ID
ALIPAY_APP_ID=2021000000000001

# 应用私钥 (PEM 格式)
ALIPAY_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA...
-----END RSA PRIVATE KEY-----"

# 支付宝公钥
ALIPAY_PUBLIC_KEY="-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG...
-----END PUBLIC KEY-----"

# 异步通知地址 (需要使用公网可达 URL)
ALIPAY_NOTIFY_URL=https://your-domain.com/api/payment/callback

# 同步跳转地址
ALIPAY_RETURN_URL=https://your-domain.com/payment/return

# 沙箱模式 (测试阶段保持 true)
ALIPAY_SANDBOX=true
EOF

# 导出环境变量
export $(cat .env | xargs)
```

### 4.3 启动服务

```bash
# 启动 API 服务器
python python_api_server.py --host 0.0.0.0 --port 8080
```

预期输出：

```
╔══════════════════════════════════════════════════════════════╗
║           su-memory SDK API Server v1.7.0                    ║
╠══════════════════════════════════════════════════════════════╣
║  支付:     ✅                                              ║
║    POST   /api/payment/order      - 创建支付订单            ║
║    GET    /api/payment/order/{id} - 查询订单              ║
║    POST   /api/payment/callback   - 支付宝异步通知          ║
║    POST   /api/payment/refund     - 退款                    ║
║    GET    /api/payment/health     - 支付健康检查            ║
╚══════════════════════════════════════════════════════════════╝
```

> ⚠️ 如果支付模块显示「未配置」，请检查环境变量是否正确设置且密钥格式正确。

### 4.4 验证配置

```bash
# 健康检查
curl http://localhost:8080/api/payment/health
```

预期响应：

```json
{
  "status": "ok",
  "gateway": "https://openapi-sandbox.dl.alipaydev.com/gateway.do",
  "sandbox": true,
  "client_initialized": true
}
```

---

## 5. API 参考

### 5.1 支付服务健康检查

```http
GET /api/payment/health
```

**响应示例 (已配置)：**

```json
{
  "status": "ok",
  "gateway": "https://openapi-sandbox.dl.alipaydev.com/gateway.do",
  "sandbox": true,
  "client_initialized": true
}
```

**响应示例 (未配置)：**

```json
{
  "status": "unavailable",
  "message": "支付模块未启用。请配置 ALIPAY_APP_ID 等环境变量。"
}
```

---

### 5.2 创建支付订单

```http
POST /api/payment/order
Content-Type: application/json
```

**请求参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `plan` | string | ✅ | 套餐类型：`starter` / `pro` / `enterprise` / `on_premise` |
| `return_url` | string | 否 | 支付完成后同步跳转地址 |
| `email` | string | 否 | 买家邮箱 (用于 License 发放) |
| `metadata` | object | 否 | 额外元数据 (自由格式) |

**请求示例：**

```bash
curl -X POST http://localhost:8080/api/payment/order \
  -H "Content-Type: application/json" \
  -d '{
    "plan": "pro",
    "email": "user@example.com",
    "return_url": "https://your-domain.com/payment/return",
    "metadata": {
      "source": "web",
      "campaign": "spring-sale"
    }
  }'
```

**成功响应 (201)：**

```json
{
  "order_id": "SM-1746123456789-a1b2c3",
  "payment_url": "<!DOCTYPE html><html>...支付宝支付表单...</html>",
  "plan_type": "pro",
  "amount": 99.9,
  "status": "pending"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `order_id` | string | 商户订单号，格式 `SM-{timestamp_ms}-{6位随机hex}` |
| `payment_url` | string | 支付宝支付页面 HTML，可直接在浏览器中渲染 |
| `plan_type` | string | 套餐类型 |
| `amount` | float | 支付金额 (元) |
| `status` | string | 订单状态 (`pending`) |

**前端集成示例：**

```javascript
// 创建订单后，将 payment_url 渲染到 iframe 或新窗口
const response = await fetch('/api/payment/order', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ plan: 'pro', email: 'user@example.com' })
});

const { payment_url, order_id } = await response.json();

// 方式一：新窗口打开
const payWindow = window.open('', '_blank');
payWindow.document.write(payment_url);
payWindow.document.close();

// 方式二：监听支付完成
// 通过 return_url 参数或轮询订单状态
```

**错误响应：**

| 状态码 | 说明 |
|-------|------|
| 400 | `plan` 参数无效或为免费套餐 (`community`) |
| 500 | 支付宝下单失败 (检查密钥配置) |
| 503 | 支付模块未启用 (环境变量未配置) |

---

### 5.3 查询订单状态

```http
GET /api/payment/order/{order_id}
```

**路径参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `order_id` | string | ✅ | 商户订单号 |

**请求示例：**

```bash
curl http://localhost:8080/api/payment/order/SM-1746123456789-a1b2c3
```

**成功响应 (200) - 已支付：**

```json
{
  "order_id": "SM-1746123456789-a1b2c3",
  "plan_type": "pro",
  "amount": 99.9,
  "status": "paid",
  "trade_no": "2026050322001000000000000000",
  "license_key": "SM-PRO-67f4a1b2-C3D4E5F6",
  "created_at": "2026-05-03T10:30:00.000000+00:00",
  "paid_at": "2026-05-03T10:32:15.000000+00:00"
}
```

**成功响应 (200) - 待支付：**

```json
{
  "order_id": "SM-1746123456789-a1b2c3",
  "plan_type": "pro",
  "amount": 99.9,
  "status": "pending",
  "trade_no": null,
  "license_key": null,
  "created_at": "2026-05-03T10:30:00.000000+00:00",
  "paid_at": null
}
```

**响应字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `order_id` | string | 商户订单号 |
| `plan_type` | string | 套餐类型 |
| `amount` | float | 支付金额 (元) |
| `status` | string | 订单状态：`pending` / `paid` / `refunded` / `cancelled` / `closed` |
| `trade_no` | string\|null | 支付宝交易号 (支付成功后返回) |
| `license_key` | string\|null | License Key (支付成功后生成) |
| `created_at` | string | 订单创建时间 (ISO 8601) |
| `paid_at` | string\|null | 支付完成时间 (ISO 8601) |

> 💡 **轮询建议**：对于 `pending` 状态订单，此接口会主动向支付宝发起同步查询以获取最新支付状态。建议每 3-5 秒轮询一次，最多轮询 60 次。

**状态码：**

| 状态码 | 说明 |
|-------|------|
| 200 | 成功 |
| 404 | 订单不存在 |
| 503 | 支付服务不可用 |

---

### 5.4 支付宝异步通知回调

```http
POST /api/payment/callback
Content-Type: application/x-www-form-urlencoded
```

> ⚠️ **注意**：此端点由支付宝服务端调用，非用户直接访问。请求体为 URL-encoded 表单格式，非 JSON。

**支付宝发送的参数 (部分关键字段)：**

| 参数 | 说明 |
|------|------|
| `out_trade_no` | 商户订单号 |
| `trade_no` | 支付宝交易号 |
| `trade_status` | 交易状态：`TRADE_SUCCESS` / `TRADE_FINISHED` |
| `total_amount` | 交易金额 |
| `buyer_logon_id` | 买家支付宝账号 |
| `notify_id` | 通知校验 ID (用于幂等) |
| `notify_time` | 通知时间 |
| `sign` | RSA2 签名 |
| `sign_type` | 签名类型 |

**回调处理流程：**

```
支付宝 POST → 验签 → 幂等检查 (notify_id) → 交易状态检查
  → 金额校验 (订单金额 ± 0.01) → 更新订单状态 → 生成 License Key
  → 记录已处理 → 返回 "success"
```

**响应格式：**

| 处理结果 | HTTP 状态码 | 响应内容 |
|---------|-----------|---------|
| 成功 | 200 | `success` (纯文本) |
| 失败 (验签/金额不匹配等) | 400 | `fail` (纯文本) |

> ⚠️ 支付宝要求返回纯文本 `success` 或 `fail`，而非 JSON。如果返回 `fail`，支付宝会按递增间隔重试通知（最多重试 4 次）。

**幂等保障：**

系统通过 `notify_id` 进行去重。每次成功处理通知后，会在 `~/.su-memory/payment_notifications/{notify_id}.json` 创建标记文件，重复通知直接返回 `success`，避免重复处理。

---

### 5.5 退款

```http
POST /api/payment/refund
Content-Type: application/json
```

**请求参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `order_id` | string | ✅ | 商户订单号 |
| `refund_amount` | float | 否 | 退款金额 (默认全额退款) |
| `reason` | string | 否 | 退款原因 (默认 "用户申请退款") |

**请求示例：**

```bash
curl -X POST http://localhost:8080/api/payment/refund \
  -H "Content-Type: application/json" \
  -d '{
    "order_id": "SM-1746123456789-a1b2c3",
    "reason": "用户误购，7天内申请退款"
  }'
```

**成功响应 (200)：**

```json
{
  "success": true,
  "order_id": "SM-1746123456789-a1b2c3",
  "refund_amount": 99.9,
  "alipay_result": {
    "code": "10000",
    "msg": "Success",
    "fund_change": "Y"
  }
}
```

**部分退款示例：**

```bash
curl -X POST http://localhost:8080/api/payment/refund \
  -H "Content-Type: application/json" \
  -d '{
    "order_id": "SM-1746123456789-a1b2c3",
    "refund_amount": 50.0,
    "reason": "部分退款"
  }'
```

**失败响应：**

```json
{
  "success": false,
  "order_id": "SM-1746123456789-a1b2c3",
  "error": "订单状态不允许退款: 当前状态=pending, 需要状态=paid"
}
```

**状态码：**

| 状态码 | 说明 |
|-------|------|
| 200 | 退款成功 / 退款失败 (检查 `success` 字段) |
| 400 | 参数错误或订单状态不允许退款 |
| 500 | 支付宝退款接口调用失败 |
| 503 | 支付服务不可用 |

**退款前置条件：**

- 订单状态必须为 `paid`
- 订单必须包含支付宝交易号 (`trade_no`)
- 退款金额 ≤ 订单金额
- on_premise 套餐不支持退款（一次性买断）

---

## 6. 套餐与价格

### 6.1 套餐概览

| 套餐 | plan 参数 | 价格 | 记忆容量 | 会话数 | API 调用 | 核心特性 |
|------|----------|------|---------|--------|---------|---------|
| 社区版 | `community` | ¥0 (免费) | 10,000 | 100 | 10,000/月 | 基础查询、TF-IDF |
| 入门版 | `starter` | ¥29.9/月 | 50,000 | 500 | 50,000/月 | + 向量检索 |
| 专业版 | `pro` | ¥99.9/月 | 200,000 | 无限制 | 无限制 | + 多跳推理、因果推理、时序预测 |
| 企业版 | `enterprise` | ¥399/月 | 无限制 | 无限制 | 无限制 | 全部功能、API 接口、多租户、SSO |
| 私有部署版 | `on_premise` | ¥9,999/永久 | 无限制 | 无限制 | 无限制 | 全部功能 + 源码 + 原厂支持 |

> **注意**：`community` 套餐免费，无需创建支付订单。

### 6.2 功能详细对比

| 功能 | Community | Starter | Pro | Enterprise | On-Premise |
|------|-----------|---------|-----|------------|------------|
| 基础查询 | ✅ | ✅ | ✅ | ✅ | ✅ |
| TF-IDF 检索 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 向量检索 | ❌ | ✅ | ✅ | ✅ | ✅ |
| 多跳推理 | ❌ | ❌ | ✅ | ✅ | ✅ |
| 因果推理 | ❌ | ❌ | ✅ | ✅ | ✅ |
| 时序预测 | ❌ | ❌ | ✅ | ✅ | ✅ |
| 可解释性 | ❌ | ❌ | ✅ | ✅ | ✅ |
| 多会话 | ❌ | ❌ | ❌ | ✅ | ✅ |
| REST API | ❌ | ❌ | ❌ | ✅ | ✅ |
| 多租户 | ❌ | ❌ | ❌ | ✅ | ✅ |
| SSO 集成 | ❌ | ❌ | ❌ | ✅ | ✅ |
| 源码交付 | ✅ | ❌ | ❌ | ❌ | ✅ |
| 技术支持 | 社区 | 邮件 | 邮件 | 专属 | 原厂一年 |

### 6.3 套餐升级路径

```
Community (免费)
  ├─→ Starter  (¥29.9/月)  适合个人轻度使用
  ├─→ Pro      (¥99.9/月)  适合专业用户
  └─→ Enterprise (¥399/月)  适合企业团队

On-Premise (¥9,999/永久)    适合大型企业/私有部署
```

### 6.4 License Key 格式

支付成功后自动生成的 License Key 遵循以下格式：

```
SM-{PREFIX}-{timestamp_hex}-{8位随机hex}
```

| 套餐 | 前缀 | 示例 |
|------|------|------|
| Community | `COM` | `SM-COM-67f4a1b2-A1B2C3D4` |
| Starter | `STD` | `SM-STD-67f4a1b2-E5F6G7H8` |
| Pro | `PRO` | `SM-PRO-67f4a1b2-C3D4E5F6` |
| Enterprise | `ENT` | `SM-ENT-67f4a1b2-G7H8I9J0` |
| On-Premise | `ONP` | `SM-ONP-67f4a1b2-K1L2M3N4` |

### 6.5 订单号格式

```
SM-{timestamp_ms}-{6位随机hex}

示例: SM-1746123456789-a1b2c3
```

---

## 7. 安全规范

### 7.1 私钥管理

> ⚠️ **绝对禁止**：将私钥硬编码在代码中、提交到版本控制系统、打印到日志。

**推荐实践：**

| 环境 | 建议方案 |
|------|---------|
| 开发/测试 | 使用沙箱环境的独立密钥对 |
| 生产-Docker | 通过 Docker Secret 或环境变量注入 |
| 生产-Kubernetes | 使用 Kubernetes Secret 挂载为文件 |
| 生产-传统部署 | 密钥文件权限设为 `600`，存储于受保护目录 |

**Kubernetes Secret 示例：**

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: alipay-secrets
type: Opaque
stringData:
  ALIPAY_APP_ID: "2021000000000001"
  ALIPAY_PRIVATE_KEY: |
    -----BEGIN RSA PRIVATE KEY-----
    MIIEpAIBAAKCAQEA...
    -----END RSA PRIVATE KEY-----
  ALIPAY_PUBLIC_KEY: |
    -----BEGIN PUBLIC KEY-----
    MIIBIjANBgkqhkiG...
    -----END PUBLIC KEY-----
```

```yaml
# 在 Deployment 中引用
envFrom:
  - secretRef:
      name: alipay-secrets
```

### 7.2 验签流程

系统使用 RSA2 (SHA256) 算法对支付宝异步通知进行签名验证：

```
1. 接收 POST 参数 (含 sign、sign_type)
2. 提取签名: sign = params["sign"]
3. 构建验签数据: 排除 sign 和 sign_type 的所有参数
4. 调用 alipay.verify(data, signature)
5. 验签通过 → 继续处理
6. 验签失败 → 返回 "fail"，记录告警日志
```

验签代码位于 `alipay_client.py` 的 `verify_signature` 方法，底层调用 `alipay-sdk-python` 的 `verify` 方法。

### 7.3 敏感信息保护

| 措施 | 实现 |
|------|------|
| 配置安全打印 | `AlipayConfig.safe_repr()` 仅显示 APPID 前4位 + `***`，不输出密钥内容 |
| 日志脱敏 | License Key 仅输出前 16 位 + `...` |
| 文件权限 | 订单文件由系统默认权限控制，建议部署时设置 `umask 077` |
| 传输安全 | 支付宝网关全部使用 HTTPS |

### 7.4 金额校验

异步通知处理时，系统会校验支付宝回调金额与本地订单金额是否一致：

```python
# 允许 0.01 元的微小误差
if abs(order.amount - context.total_amount) > 0.01:
    return (False, "金额不匹配", CallbackResult.AMOUNT_MISMATCH)
```

### 7.5 幂等性保障

| 层面 | 机制 |
|------|------|
| 通知幂等 | 基于 `notify_id` 的去重标记文件 |
| 订单状态幂等 | `mark_order_paid` 方法检查终态后直接返回 |
| 退款幂等 | 支付宝接口支持 `out_request_no` 参数去重 |

---

## 8. 测试指南

### 8.1 沙箱环境配置

支付宝沙箱环境提供独立于生产环境的完整测试能力。

**获取沙箱账号：**

1. 登录 [支付宝开放平台](https://open.alipay.com/)
2. 进入「控制台」→「沙箱环境」
3. 获取沙箱 APPID、沙箱网关地址
4. 配置沙箱密钥对 (可与正式环境不同)
5. 获取沙箱买家账号和密码

**沙箱配置示例：**

```bash
export ALIPAY_APP_ID=2021000000000001          # 沙箱 APPID
export ALIPAY_SANDBOX=true                      # 启用沙箱
export ALIPAY_GATEWAY=https://openapi-sandbox.dl.alipaydev.com/gateway.do
# 密钥同理设置
```

### 8.2 本地测试流程

#### 步骤 1：启动服务

```bash
python python_api_server.py --port 8080
```

确认日志显示 `支付模块已初始化: ... sandbox=True`。

#### 步骤 2：创建测试订单

```bash
curl -X POST http://localhost:8080/api/payment/order \
  -H "Content-Type: application/json" \
  -d '{"plan": "pro", "email": "test@example.com"}'
```

记录返回的 `order_id`，例如 `SM-1746123456789-a1b2c3`。

#### 步骤 3：完成沙箱支付

1. 将响应中的 `payment_url` 保存为 HTML 文件并在浏览器打开
2. 使用沙箱买家账号登录
3. 输入沙箱支付密码完成支付
4. 支付成功后页面将跳转到 `return_url`

#### 步骤 4：查询订单状态

```bash
curl http://localhost:8080/api/payment/order/SM-1746123456789-a1b2c3
```

预期返回 `"status": "paid"` 和 `"license_key": "SM-PRO-..."`。

### 8.3 异步通知测试 (内网穿透)

由于支付宝异步通知需要公网可达的 URL，本地开发时需要使用内网穿透工具：

```bash
# 使用 ngrok
ngrok http 8080

# 获取公网 URL，例如: https://abc123.ngrok.io
# 设置环境变量
export ALIPAY_NOTIFY_URL=https://abc123.ngrok.io/api/payment/callback
```

重启服务后，沙箱支付完成时支付宝会向该 URL 发送异步通知。

### 8.4 退款测试

```bash
# 全额退款
curl -X POST http://localhost:8080/api/payment/refund \
  -H "Content-Type: application/json" \
  -d '{"order_id": "SM-1746123456789-a1b2c3", "reason": "测试退款"}'

# 查询确认
curl http://localhost:8080/api/payment/order/SM-1746123456789-a1b2c3
# 预期状态: "refunded"
```

### 8.5 切换正式环境

```bash
# 1. 确认已完成沙箱测试
# 2. 更新环境变量
export ALIPAY_APP_ID=正式环境APPID
export ALIPAY_SANDBOX=false
# 密钥替换为正式环境密钥
export ALIPAY_PRIVATE_KEY="正式环境私钥"
export ALIPAY_PUBLIC_KEY="正式环境支付宝公钥"
export ALIPAY_NOTIFY_URL=https://your-production-domain.com/api/payment/callback
export ALIPAY_RETURN_URL=https://your-production-domain.com/payment/return

# 3. 重启服务
python python_api_server.py --port 8080
```

---

## 9. 常见问题

### Q1: 支付模块未加载，提示 "支付模块未加载"

**A:** 支付宝 SDK 未安装或环境变量未配置。

```bash
# 检查依赖
pip list | grep alipay

# 安装
pip install alipay-sdk-python

# 检查环境变量
echo $ALIPAY_APP_ID        # 应有值
echo $ALIPAY_SANDBOX       # 默认 true
```

### Q2: 提示 "ALIPAY_APP_ID 未设置"

**A:** 使用 `AlipayConfig.from_env()` 时，必须设置 `ALIPAY_APP_ID` 环境变量。可以通过 `.env` 文件或直接 `export` 设置。

### Q3: 支付页面返回后订单仍是 "pending"

**A:** 可能原因及排查：

1. **异步通知未到达**：检查 `ALIPAY_NOTIFY_URL` 是否公网可达，服务器日志是否有回调记录
2. **验签失败**：检查支付宝公钥是否正确（注意区分「应用公钥」和「支付宝公钥」）
3. **主动查询**：调用 `GET /api/payment/order/{id}` 会触发支付宝同步查询

### Q4: 验签失败怎么排查？

**A:** 按以下顺序检查：

1. 确认使用的密钥对和应用绑定的密钥一致
2. 确认配置的是「支付宝公钥」而非「应用公钥」
3. 检查密钥格式是否为标准 PEM 格式（含 `-----BEGIN/END-----` 标记）
4. 如密钥通过 Base64 编码传入，检查编码是否正确
5. 沙箱和正式环境使用不同的密钥对

### Q5: 如何处理退款？

**A:** 通过 `POST /api/payment/refund` 接口。前置条件：
- 订单状态为 `paid`
- on_premise 套餐不支持退款

如需人工处理退款，可在支付宝商家后台操作。

### Q6: License Key 在哪里获取？

**A:** 支付成功后，License Key 会通过以下方式获取：
- 查询订单接口 `GET /api/payment/order/{id}` → `license_key` 字段
- `return_url` 页面可解析订单号后查询

License Key 也可用于 SDK 离线激活，格式为 `SM-{PREFIX}-{timestamp}-{random}`。

### Q7: 沙箱支付金额有限制吗？

**A:** 沙箱环境金额无实际限制，但建议使用与实际套餐一致的价格进行测试。沙箱买家账号余额充足，不会产生真实扣款。

### Q8: 正式上线前需要做什么？

**A:** 检查清单：

- [ ] 将 `ALIPAY_SANDBOX` 设为 `false`
- [ ] 替换为正式环境 APPID 和密钥
- [ ] 确保 `ALIPAY_NOTIFY_URL` 为生产环境可访问的公网 HTTPS 地址
- [ ] 在支付宝开放平台提交应用审核并上线
- [ ] 确认电脑网站支付产品已签约生效
- [ ] 至少完成一笔小额真实支付测试

### Q9: 订单数据存储在哪里？

**A:** 订单以 JSON 文件存储，默认路径为 `~/.su-memory/orders/{order_id}.json`。已处理的通知记录在 `~/.su-memory/payment_notifications/`。

### Q10: 如何查看订单列表？

**A:** `OrderService.list_orders()` 方法支持按状态过滤和分页。当前版本尚未暴露 HTTP 端点，可通过 Python API 调用：

```python
from su_memory.payment.order_service import OrderService

service.list_orders(status="paid", limit=20)
```

---

## 附录

### A. 参考资料

| 资源 | 链接 |
|------|------|
| 支付宝开放平台 | https://open.alipay.com/ |
| 电脑网站支付文档 | https://opendocs.alipay.com/open/270/105898 |
| alipay-sdk-python | https://github.com/fzlee/alipay |
| su-memory SDK | https://github.com/su-memory/su-memory-sdk |

### B. 术语表

| 术语 | 说明 |
|------|------|
| APPID | 支付宝应用唯一标识 |
| out_trade_no | 商户订单号 (本系统格式: `SM-{timestamp_ms}-{random}`) |
| trade_no | 支付宝交易号 (支付成功后支付宝返回) |
| notify_id | 支付宝通知校验 ID (用于幂等去重) |
| page_pay | 电脑网站支付接口 |
| RSA2 | RSA-SHA256 签名算法 |
| License Key | 支付成功后生成的激活密钥 |

### C. 联系方式

| 渠道 | 信息 |
|------|------|
| 技术支持邮箱 | sandysu737@gmail.com |
| 项目地址 | https://github.com/su-memory/su-memory-sdk |

---

**文档版本**: v1.7.0 | **更新日期**: 2026-05-03
