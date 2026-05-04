# 🚀 su-memory SDK 支付宝支付 Pipedream 部署指南

> 在现有 LemonSqueezy 工作流基础上修改，5 分钟完成部署

---

## 前置条件

- ✅ RSA2 密钥已生成：`keys/alipay_private_key.pem`、`keys/alipay_public_key.pem`
- ✅ 支付宝企业账户 APP_ID：`2021006151644209`
- ✅ Gmail 已授权：sandysu737@gmail.com

---

## 步骤 0：登录 Pipedream

1. 打开 https://pipedream.com
2. 用 Google 账号 sandysu737@gmail.com 登录
3. 进入 **Workflows**，找到现有的 su-memory 工作流（名称可能含 "su-memory"、"lemon" 等）
4. 点击进入编辑

---

## 步骤 1：修改 HTTP 触发器

保留现有 HTTP 触发器（不用改），记下 URL：
```
https://xxxxx.m.pipedream.net
```

---

## 步骤 2：修改/添加路由分发步骤

如果已有 Code 步骤，修改其代码；如果没有，点击 **+** → **Code (Node.js)** 添加。

**步骤名**: `route`

**代码** (直接全选替换):
```javascript
async function routeRequest(steps) {
  const { method, path, body, query } = steps.trigger.event;

  if (method === 'POST' && path === '/create-order') {
    return { action: 'create_order', data: body };
  }
  if (method === 'POST' && path === '/alipay-notify') {
    return { action: 'alipay_notify', data: body };
  }
  if (method === 'GET' && path === '/alipay-notify') {
    return { action: 'ping', data: query };
  }
  return { action: 'not_found', path, method };
}
```

---

## 步骤 3：添加创建支付宝订单步骤

点击 **+** → **Code (Node.js)**。

**步骤名**: `create_order`

**代码**:
```javascript
const crypto = require('crypto');

async function createAlipayOrder(steps) {
  const routeResult = steps.route.$return_value;
  if (routeResult.action !== 'create_order') { return null; }

  const { order_id, plan_type, amount, buyer_email } = routeResult.data;
  if (!order_id || !plan_type || !amount) {
    return { status: 400, body: { error: '缺少必填参数' } };
  }

  const planNames = {
    community: '社区版', starter: '入门版 Starter',
    pro: '专业版 Pro', enterprise: '企业版 Enterprise',
    on_premise: '私有部署版 On-Premise',
  };

  const bizContent = JSON.stringify({
    out_trade_no: order_id,
    product_code: 'FAST_INSTANT_TRADE_PAY',
    total_amount: Number(amount).toFixed(2),
    subject: 'su-memory SDK ' + (planNames[plan_type] || plan_type),
    body: 'su-memory SDK ' + (planNames[plan_type] || plan_type),
    passback_params: encodeURIComponent(JSON.stringify({
      plan_type: plan_type,
      buyer_email: buyer_email || ''
    })),
  });

  const params = {
    app_id: process.env.ALIPAY_APP_ID,
    method: 'alipay.trade.page.pay',
    charset: 'utf-8',
    sign_type: 'RSA2',
    timestamp: formatTimestamp(new Date()),
    version: '1.0',
    notify_url: process.env.ALIPAY_NOTIFY_URL,
    return_url: '',
    biz_content: bizContent,
  };

  params.sign = rsaSign(params, process.env.ALIPAY_PRIVATE_KEY);

  const gateway = process.env.ALIPAY_GATEWAY || 'https://openapi.alipay.com/gateway.do';
  const qs = Object.entries(params)
    .map(([k, v]) => k + '=' + encodeURIComponent(v))
    .join('&');

  return {
    status: 200,
    body: {
      success: true,
      order_id: order_id,
      payment_url: gateway + '?' + qs,
      message: '请跳转至支付宝完成支付',
    },
  };
}

function formatTimestamp(d) {
  const p = n => String(n).padStart(2, '0');
  return d.getFullYear() + '-' + p(d.getMonth() + 1) + '-' + p(d.getDate())
    + ' ' + p(d.getHours()) + ':' + p(d.getMinutes()) + ':' + p(d.getSeconds());
}

function rsaSign(params, privateKeyPem) {
  const keys = Object.keys(params)
    .filter(k => k !== 'sign' && params[k] !== undefined && params[k] !== '')
    .sort();
  const signStr = keys.map(k => k + '=' + params[k]).join('&');
  const sign = crypto.createSign('RSA-SHA256');
  sign.update(signStr);
  sign.end();
  return sign.sign(privateKeyPem, 'base64');
}
```

---

## 步骤 4：修改通知处理步骤

找到现有的 License Key 生成步骤，**全选替换**为以下代码。

**步骤名**: `process_notify`

**代码**:
```javascript
const crypto = require('crypto');

async function processAlipayNotify(steps) {
  const routeResult = steps.route.$return_value;

  if (routeResult.action === 'ping') {
    return { status: 200, body: 'success' };
  }
  if (routeResult.action !== 'alipay_notify') { return null; }

  const notifyData = routeResult.data;

  // 1. 验证签名
  if (!verifySign(notifyData, process.env.ALIPAY_PUBLIC_KEY)) {
    console.error('签名验证失败');
    return { status: 400, body: 'fail' };
  }

  // 2. 检查交易状态
  const ts = notifyData.trade_status;
  if (ts !== 'TRADE_SUCCESS' && ts !== 'TRADE_FINISHED') {
    return { status: 200, body: 'success' };
  }

  // 3. 提取订单信息
  const outTradeNo = notifyData.out_trade_no;
  const tradeNo = notifyData.trade_no;
  const totalAmount = parseFloat(notifyData.total_amount || '0');
  const buyerLogonId = notifyData.buyer_logon_id || '';

  let planType = 'unknown', buyerEmail = '';
  try {
    const pb = JSON.parse(decodeURIComponent(notifyData.passback_params || '{}'));
    planType = pb.plan_type || 'unknown';
    buyerEmail = pb.buyer_email || '';
  } catch (e) {}

  // 4. 生成 License Key
  const licenseKey = generateLicenseKey(planType);

  // 5. 构建 License 数据
  const pkg = getCapacityConfig(planType);
  const now = new Date();
  const expires = planType === 'on_premise'
    ? 'never'
    : new Date(now.getFullYear() + 1, now.getMonth(), now.getDate())
        .toISOString().split('T')[0];

  const features = {};
  if (pkg.features[0] === '*') {
    features.all = true;
  } else {
    pkg.features.forEach(f => { features[f] = true; });
  }

  const sig = process.env.LICENSE_SECRET
    ? crypto.createHmac('sha256', process.env.LICENSE_SECRET)
        .update(licenseKey).digest('hex').slice(0, 16)
    : '';

  const licenseData = {
    version: '1.0',
    license_key: licenseKey,
    license_type: planType,
    capacity: pkg.memories,
    issued_to: buyerEmail || buyerLogonId,
    issued_at: now.toISOString(),
    expires: expires,
    features: features,
    order_id: outTradeNo,
    trade_no: tradeNo,
    amount: totalAmount,
    signature: sig,
  };

  console.log('PAYMENT_COMPLETE', JSON.stringify(licenseData));
  return { status: 200, body: 'success', license: licenseData };
}

function verifySign(data, pubKey) {
  try {
    const sign = data.sign;
    if (!sign) return false;
    const keys = Object.keys(data)
      .filter(k => k !== 'sign' && k !== 'sign_type'
        && data[k] !== undefined && data[k] !== '')
      .sort();
    const str = keys.map(k => k + '=' + data[k]).join('&');
    const v = crypto.createVerify('RSA-SHA256');
    v.update(str);
    v.end();
    return v.verify(pubKey, sign, 'base64');
  } catch (e) { return false; }
}

function generateLicenseKey(planType) {
  const prefixMap = {
    community: 'COM', starter: 'STD', pro: 'PRO',
    enterprise: 'ENT', on_premise: 'ONP',
  };
  const prefix = prefixMap[planType] || 'UNK';
  const ts = Date.now().toString(16).toUpperCase();
  let rnd = '';
  for (let i = 0; i < 8; i++) {
    rnd += Math.floor(Math.random() * 16).toString(16);
  }
  return 'SM-' + prefix + '-' + ts + '-' + rnd.toUpperCase();
}

function getCapacityConfig(planType) {
  const map = {
    community:  { memories: 1000,   features: ['basic_query', 'tfidf', 'session_basic'] },
    starter:    { memories: 50000,  features: ['basic_query', 'tfidf', 'session_basic', 'vector_search'] },
    pro:        { memories: 200000, features: ['basic_query', 'tfidf', 'session_basic', 'vector_search', 'multihop', 'causal_inference', 'temporal', 'prediction'] },
    enterprise: { memories: -1,     features: ['*'] },
    on_premise: { memories: -1,     features: ['*'] },
  };
  return map[planType] || map.community;
}
```

---

## 步骤 5：保留/修改 Gmail 发送步骤

现有的 Gmail Send Email 步骤保留，修改以下字段：

| 字段 | 值 |
|------|-----|
| **To** | `{{steps.process_notify.license.issued_to}}` |
| **Subject** | `su-memory SDK 授权码 — {{steps.process_notify.license.license_key}}` |

**邮件 HTML 内容**:
```html
<h2>🎉 感谢您的购买！</h2>
<p>我们已收到您的支付宝付款，以下是您的授权信息：</p>

<table style="border-collapse:collapse;width:100%;max-width:500px;font-family:sans-serif;">
<tr><td style="padding:10px;border:1px solid #ddd;background:#fafafa;"><strong>授权码</strong></td>
<td style="padding:10px;border:1px solid #ddd;font-family:monospace;">{{steps.process_notify.license.license_key}}</td></tr>

<tr><td style="padding:10px;border:1px solid #ddd;background:#fafafa;"><strong>版本类型</strong></td>
<td style="padding:10px;border:1px solid #ddd;">{{steps.process_notify.license.license_type}}</td></tr>

<tr><td style="padding:10px;border:1px solid #ddd;background:#fafafa;"><strong>记忆容量</strong></td>
<td style="padding:10px;border:1px solid #ddd;">{{steps.process_notify.license.capacity === -1 ? "无限制" : steps.process_notify.license.capacity + " 条"}}</td></tr>

<tr><td style="padding:10px;border:1px solid #ddd;background:#fafafa;"><strong>订单号</strong></td>
<td style="padding:10px;border:1px solid #ddd;font-family:monospace;">{{steps.process_notify.license.order_id}}</td></tr>

<tr><td style="padding:10px;border:1px solid #ddd;background:#fafafa;"><strong>支付宝交易号</strong></td>
<td style="padding:10px;border:1px solid #ddd;font-family:monospace;">{{steps.process_notify.license.trade_no}}</td></tr>

<tr><td style="padding:10px;border:1px solid #ddd;background:#fafafa;"><strong>支付金额</strong></td>
<td style="padding:10px;border:1px solid #ddd;">¥{{steps.process_notify.license.amount}}</td></tr>

<tr><td style="padding:10px;border:1px solid #ddd;background:#fafafa;"><strong>到期时间</strong></td>
<td style="padding:10px;border:1px solid #ddd;">{{steps.process_notify.license.expires === "never" ? "永久有效" : steps.process_notify.license.expires}}</td></tr>
</table>

<h3>📋 使用方法</h3>
<ol>
<li>设置环境变量:
<pre>export SU_MEMORY_LICENSE_KEY={{steps.process_notify.license.license_key}}</pre></li>
<li>或保存为 <code>~/.su-memory/license.json</code></li>
<li>重新启动应用，授权自动生效</li>
</ol>

<pre style="background:#f5f5f5;padding:15px;border-radius:5px;overflow-x:auto;font-size:12px;">
{{JSON.stringify(steps.process_notify.license, null, 2)}}
</pre>

<p>如有问题请联系 sandysu737@gmail.com<br>su-memory SDK 团队</p>
```

---

## 步骤 6：配置环境变量

在 Workflow Settings → **Environment Variables** 中添加/修改：

| 变量名 | 值 |
|--------|-----|
| `ALIPAY_APP_ID` | `2021006151644209` |
| `ALIPAY_PRIVATE_KEY` | 复制粘贴 `keys/alipay_private_key.pem` 的完整内容 |
| `ALIPAY_PUBLIC_KEY` | 复制粘贴 `keys/alipay_public_key.pem` 的完整内容 |
| `ALIPAY_GATEWAY` | `https://openapi.alipay.com/gateway.do` |
| `ALIPAY_NOTIFY_URL` | `{你的Pipedream URL}/alipay-notify` |
| `LICENSE_SECRET` | `74c10cd72148d1e61d5146a03a9eabe91e35332f764054e364b93d37f3d435ce` |
| `GMAIL_USER` | `sandysu737@gmail.com` |

⚠️ 删除旧的 LemonSqueezy 相关环境变量（如果有的话）:
- `LEMONSQUEEZY_SIGNING_SECRET`
- `LEMONSQUEEZY_STORE`
- `LEMONSQUEEZY_API_KEY`

---

## 步骤 7：部署

1. 点击右上角 **Deploy** 按钮
2. 确认部署成功
3. 记下 HTTP 触发器 URL

---

## 步骤 8：测试

### 测试 1：创建订单
```bash
curl -X POST https://YOUR_URL.m.pipedream.net/create-order \
  -H "Content-Type: application/json" \
  -d '{"order_id":"SM-test-001","plan_type":"starter","amount":29.9,"buyer_email":"test@example.com"}'
```

预期返回包含 `payment_url` 的 JSON。

### 测试 2：URL 可达性
```bash
curl https://YOUR_URL.m.pipedream.net/alipay-notify
```

预期返回 `success`。

### 测试 3：更新前端 URL
部署成功后，将 `frontend/payment.html` 第 683 行的 `PIPEDREAM_BASE_URL` 替换为实际工作流 URL。

---

## 步骤 9：支付宝平台配置

1. 登录 [支付宝开放平台](https://open.alipay.com)
2. 进入应用详情 → 开发设置
3. 上传新的应用公钥（`keys/alipay_public_key.pem` 的 base64 格式，去头尾）
4. 设置应用网关地址为 Pipedream URL
5. 保存配置

---

**部署完成后，完整的支付流程即可运行！** 🎉
