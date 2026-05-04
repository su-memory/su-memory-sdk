# su-memory SDK 支付自动化配置指南

> 基于 Pipedream 的支付宝企业支付自动处理方案

---

## 📋 概述

本方案实现以下自动化流程：

```
用户选择套餐 → Pipedream 创建支付宝订单 → 用户支付宝付款
    → 支付宝异步通知 Pipedream → 验证签名 + 生成 License Key → Gmail 自动发送 ✅
```

**架构特点**：
- su-memory SDK 零服务器：所有服务端逻辑在 Pipedream 完成
- 密钥安全：支付宝私钥仅存储在 Pipedream 环境变量中
- 直接到账：用户支付资金直接进入企业支付宝账户
- 全自动授权：从支付到发 License Key 无需人工介入

---

## 🎯 前置准备

### 1. 必备工具

| 工具 | 用途 | 费用 |
|------|------|------|
| 支付宝企业账户 | 接收付款 | 0.6% 手续费 |
| Pipedream 账号 | 自动化处理 | 免费额度 |
| Gmail 邮箱 | 发送 License Key | 免费 |

### 2. 支付宝企业应用配置

| 配置项 | 值 |
|--------|-----|
| APP_ID | 2021006151644209 |
| 商户PID | 2088580297860296 |
| 企业名称 | 健源启晟（深圳）医疗科技有限公司 |
| 支付宝网关 | https://openapi.alipay.com/gateway.do |
| 加签方式 | RSA(SHA256) - 公钥模式 |

### 3. 注册 Pipedream

1. 访问 https://pipedream.com
2. 使用 GitHub 或 Google 账号登录
3. 进入 Dashboard

---

## 🔧 步骤一：创建 Pipedream 工作流

### 1.1 新建 Workflow

1. 点击 **Workflows** → **New Workflow**
2. 添加触发器：**HTTP / Webhook**
3. 记下生成的 URL，例如：`https://eoyjjsu9jrea1nh.m.pipedream.net`

### 1.2 配置环境变量

在 Workflow Settings → Environment Variables 中添加：

```
ALIPAY_APP_ID=2021006151644209
ALIPAY_PRIVATE_KEY=<应用私钥 PEM 格式>
ALIPAY_PUBLIC_KEY=<支付宝公钥 PEM 格式>
ALIPAY_GATEWAY=https://openapi.alipay.com/gateway.do
ALIPAY_NOTIFY_URL=https://YOUR_WORKFLOW.m.pipedream.net/alipay-notify
LICENSE_SECRET=<至少 32 字符的随机密钥>
GMAIL_USER=sandysu737@gmail.com
```

### 1.3 添加路由分发步骤

添加 **Code** 步骤（Node.js），参考 `deploy/pipedream-alipay-workflow.js` 中的 Step 1 代码。

### 1.4 添加创建订单步骤

添加 **Code** 步骤（Node.js），参考 `deploy/pipedream-alipay-workflow.js` 中的 Step 2 代码。
此步骤负责：
- 接收前端的订单请求（order_id, plan_type, amount, buyer_email）
- 使用支付宝 RSA2 签名构建支付页面 URL
- 返回支付宝支付页面地址

### 1.5 添加异步通知处理步骤

添加 **Code** 步骤（Node.js），参考 `deploy/pipedream-alipay-workflow.js` 中的 Step 3 代码。
此步骤负责：
- 接收支付宝异步通知
- RSA2 验证签名
- 检查交易状态
- 生成 License Key
- 构建 License 数据

### 1.6 添加 Gmail 发送步骤

添加 **Gmail** → **Send Email** 步骤，参考 `deploy/pipedream-alipay-workflow.js` 中的 Step 4 邮件模板。

---

## 📧 步骤二：配置支付宝异步通知

### 2.1 设置应用网关

在支付宝开放平台 → 应用详情 → 开发设置中：
- **应用网关**: `https://YOUR_WORKFLOW.m.pipedream.net`
- **授权回调地址**: `https://su-memory.ai`

### 2.2 验证回调

支付宝会发送 GET 请求验证 notify URL 可达性，Pipedream 工作流的 Step 3 已处理此验证。

---

## 🧪 步骤三：测试工作流

### 3.1 测试创建订单

使用 curl 测试创建订单接口：

```bash
curl -X POST https://YOUR_WORKFLOW.m.pipedream.net/create-order \
  -H "Content-Type: application/json" \
  -d '{
    "order_id": "SM-test-001",
    "plan_type": "starter",
    "amount": 29.9,
    "buyer_email": "test@example.com"
  }'
```

预期返回包含 `payment_url` 的 JSON。

### 3.2 模拟异步通知

```bash
curl -X POST https://YOUR_WORKFLOW.m.pipedream.net/alipay-notify \
  -d 'sign=test&trade_status=TRADE_SUCCESS&out_trade_no=SM-test-001&...'
```

### 3.3 检查 Gmail 发送

确认授权邮件已发送到指定邮箱。

---

## 🔐 步骤四：安全配置

### 4.1 密钥安全

支付宝私钥仅存储在 Pipedream 环境变量中，SDK 代码不包含任何密钥信息。
所有 API 签名在 Pipedream 服务端完成。

### 4.2 License 签名验证

在 Pipedream 工作流中使用 HMAC-SHA256 对 License Key 签名：

```javascript
const crypto = require('crypto');

function hmacSign(licenseKey, secret) {
  return crypto
    .createHmac('sha256', secret)
    .update(licenseKey)
    .digest('hex')
    .slice(0, 16);
}
```

SDK 端可验证此签名确保 License 未被篡改。

---

## 📱 步骤五：配置邮件转发（可选）

如果希望所有邮件都集中到 Gmail：

### 5.1 设置邮箱转发

在支付宝/微信商户后台设置：
```
收款邮箱 → 转发到 sandysu737@gmail.com
```

### 5.2 过滤规则

在 Gmail 中创建过滤器：
```
搜索: from:(alipay.com OR wechat.com)
操作: 标记星标 + 转发到 Pipedream
```

---

## 📊 监控与日志

### Pipedream Dashboard

- **Executions**: 查看所有执行记录
- **Logs**: 查看详细日志
- **Alerts**: 设置失败告警

### 推荐告警配置

```
触发条件: 连续 3 次执行失败
通知方式: Email + Slack
```

---

## 🔄 维护指南

### 定期检查

- [ ] 每周检查 Pipedream Workflow 执行日志
- [ ] 确认支付宝异步通知正常接收
- [ ] 确认 Gmail 发送成功率
- [ ] 定期更新 LICENSE_SECRET

### 价格调整

如需调整价格，修改以下两处：
1. su-memory SDK: `src/su_memory/payment/order_service.py` 中的 `PLAN_PRICES`
2. 前端: `frontend/payment.html` 中的 `PLANS` 配置

---

## ❓ 常见问题

### Q: Pipedream 免费额度够用吗？

A: 免费额度：每月 10,000 次执行，足够小型项目使用。

### Q: 如何处理退款？

A: 在支付宝商家后台手动操作退款。Pipedream 不需要额外处理。

### Q: 支付宝异步通知延迟怎么办？

A: 支付宝会在通知失败时重试（间隔递增），Pipedream 会自动接收重试通知。

### Q: License Key 邮件没收到怎么办？

A: 检查 Pipedream 执行日志确认是否处理成功，或联系 sandysu737@gmail.com 手动补发。

---

## 📞 联系支持

如需帮助，请联系：sandysu737@gmail.com

---

**文档版本**: v1.3.0  
**更新日期**: 2026-04-25
