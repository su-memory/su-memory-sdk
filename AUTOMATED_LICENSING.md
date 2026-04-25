# su-memory SDK 支付自动化配置指南

> 基于 Pipedream 的邮件自动处理方案

---

## 📋 概述

本方案实现以下自动化流程：

```
用户扫码支付 → 发送订单截图到邮箱 → Pipedream自动处理 → 自动回复授权码 ✅
```

**优势**：
- 零服务器成本
- 无需运维
- 快速上线
- 可扩展

---

## 🎯 前置准备

### 1. 必备工具

| 工具 | 用途 | 费用 |
|------|------|------|
| Gmail 邮箱 | 接收订单邮件 | 免费 |
| Pipedream 账号 | 自动化处理 | 免费额度 |

### 2. 注册 Pipedream

1. 访问 https://pipedream.com
2. 使用 GitHub 或 Google 账号登录
3. 进入 Dashboard

---

## 📧 步骤一：配置 Gmail 邮件源

### 1.1 创建新的 Source

1. 点击 **Sources** → **New Source**
2. 选择 **Gmail** → **New email matching search**
3. 配置搜索条件：

```
Search query: to:sandysu737@gmail.com subject:(订单 OR 支付 OR 购买 OR licensing)
```

### 1.2 配置触发条件

```
✅ 启用后保留触发器
✅ 发送测试事件（发送一封测试邮件）
```

### 1.3 获取 Webhook URL

创建后会生成一个 Webhook URL，格式如下：
```
https://eolpnhmegp.execute-api.us-east-1.amazonaws.com/dev/spaces/xxxxx/../../../gmail/emails/match
```

**记录此URL，后续步骤会用到**

---

## 🔧 步骤二：创建处理 Workflow

### 2.1 新建 Workflow

1. 点击 **Workflows** → **New Workflow**
2. 添加触发器：**Email (Built-in)**
3. 连接你的 Gmail 账号

### 2.2 配置处理步骤

点击 **+** 添加以下步骤：

---

### 步骤 2.2.1：解析邮件内容

添加 **Code** 步骤（Node.js）：

```javascript
export default defineComponent({
  async run({ steps, $ }) {
    const email = steps.trigger.event;
    
    // 提取邮件内容
    const subject = email.headers?.subject || "";
    const from = email.from?.email || "";
    const body = email.text || email.html || "";
    
    // 解析订单金额
    const amountMatch = body.match(/¥(\d+)|(\d+)元|价格[：:]?\s*(\d+)/i);
    const amount = amountMatch 
      ? parseInt(amountMatch[1] || amountMatch[2] || amountMatch[3]) 
      : 0;
    
    // 解析订单类型
    let licenseType = "unknown";
    let capacity = 1000;
    let price = 0;
    
    if (amount >= 9999) {
      licenseType = "on_premise";
      capacity = null; // 无限制
      price = 9999;
    } else if (amount >= 399) {
      licenseType = "enterprise";
      capacity = 100000;
      price = 399;
    } else if (amount >= 99) {
      licenseType = "pro";
      capacity = 10000;
      price = 99;
    } else if (amount >= 9) {
      licenseType = "capacity_pack_1k";
      capacity = 1000;
      price = 9;
    } else if (amount >= 69) {
      licenseType = "capacity_pack_10k";
      capacity = 10000;
      price = 69;
    } else if (amount >= 499) {
      licenseType = "capacity_pack_100k";
      capacity = 100000;
      price = 499;
    }
    
    // 生成授权码
    const timestamp = Date.now().toString(36).toUpperCase();
    const random = Math.random().toString(36).substring(2, 6).toUpperCase();
    const licenseKey = `SM-${licenseType.toUpperCase().replace(/_/g, '')}-${timestamp}-${random}`;
    
    // 返回解析结果
    return {
      email: from,
      subject,
      amount,
      licenseType,
      capacity,
      price,
      licenseKey
    };
  }
})
```

---

### 步骤 2.2.2：验证订单（可选）

添加 **Code** 步骤：

```javascript
export default defineComponent({
  async run({ steps, $ }) {
    const order = steps.parse_email.$return_value;
    
    // 验证金额是否匹配已知价格
    const validPrices = [9, 69, 99, 399, 499, 9999];
    
    if (!validPrices.includes(order.price)) {
      return {
        valid: false,
        message: "未识别的订单金额"
      };
    }
    
    return {
      valid: true,
      message: "订单验证通过"
    };
  }
})
```

---

### 步骤 2.2.3：生成授权文件内容

添加 **Code** 步骤：

```javascript
export default defineComponent({
  async run({ steps, $ }) {
    const order = steps.parse_email.$return_value;
    
    // 计算过期时间（按月订阅）
    const now = new Date();
    let expires;
    
    if (order.licenseType === "on_premise") {
      // 永久授权
      expires = "2099-12-31";
    } else {
      // 按月订阅，默认一年
      const nextYear = new Date(now);
      nextYear.setFullYear(nextYear.getFullYear() + 1);
      expires = nextYear.toISOString().split('T')[0];
    }
    
    // 生成授权文件
    const licenseFile = {
      version: "1.0",
      license_type: order.licenseType,
      capacity: order.capacity,
      license_key: order.licenseKey,
      issued_to: order.email,
      issued_at: now.toISOString(),
      expires: expires,
      features: {
        vector_search: true,
        causal_reasoning: order.licenseType !== "community",
        temporal_prediction: order.licenseType !== "community",
        explainability: true,
        multi_session: order.licenseType === "pro" || order.licenseType === "enterprise" || order.licenseType === "on_premise",
        api_access: order.licenseType === "enterprise" || order.licenseType === "on_premise"
      }
    };
    
    return {
      licenseFile,
      licenseKey: order.licenseKey
    };
  }
})
```

---

### 步骤 2.2.4：发送回复邮件

添加 **Gmail** → **Send Email** 步骤：

**配置**：
```
To: {{steps.parse_email.$return_value.email}}
Subject: su-memory SDK 授权码 - {{steps.generate_license.$return_value.licenseKey}}
```

**邮件内容**（选择 HTML 模式）：

```html
<h2>🎉 感谢您的购买！</h2>

<p>您好，</p>

<p>我们已收到您的订单，以下是您的授权信息：</p>

<table style="border-collapse: collapse; width: 100%; max-width: 500px;">
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><strong>授权码</strong></td>
    <td style="padding: 10px; border: 1px solid #ddd; font-family: monospace;">{{steps.generate_license.$return_value.licenseKey}}</td>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><strong>版本类型</strong></td>
    <td style="padding: 10px; border: 1px solid #ddd;">{{steps.parse_email.$return_value.licenseType}}</td>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><strong>容量</strong></td>
    <td style="padding: 10px; border: 1px solid #ddd;">{{steps.generate_license.$return_value.capacity || "无限制"}}</td>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><strong>到期时间</strong></td>
    <td style="padding: 10px; border: 1px solid #ddd;">{{steps.generate_license.$return_value.licenseFile.expires}}</td>
  </tr>
</table>

<h3>📋 使用方法</h3>

<ol>
  <li>下载附件中的 <code>license.json</code> 文件</li>
  <li>将文件放置到以下目录之一：
    <ul>
      <li><strong>Linux/Mac</strong>: <code>~/.su-memory/license.json</code></li>
      <li><strong>Windows</strong>: <code>C:\Users\你的用户名\.su-memory\license.json</code></li>
    </ul>
  </li>
  <li>创建目录（如不存在）:
    <pre>mkdir -p ~/.su-memory</pre>
  </li>
  <li>重新启动您的应用，授权自动生效</li>
</ol>

<h3>📁 授权文件内容</h3>

<pre style="background: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto;">
{{JSON.stringify(steps.generate_license.$return_value.licenseFile, null, 2)}}
</pre>

<h3>❓ 遇到问题？</h3>

<p>如有任何问题，请回复此邮件或联系：<a href="mailto:sandysu737@gmail.com">sandysu737@gmail.com</a></p>

<p>感谢您的支持！</p>

<p>---<br>
su-memory SDK 团队<br>
<a href="https://github.com/su-memory/su-memory-sdk">GitHub</a></p>
```

---

### 步骤 2.2.5：保存授权文件到云存储（可选）

如果需要保存授权记录，添加 **Amazon S3** 或 **Google Cloud Storage** 步骤：

```javascript
// 保存到 S3
await $.s3.putObject({
  Bucket: "su-memory-licenses",
  Key: `${order.licenseKey}.json`,
  Body: JSON.stringify(licenseFile),
  ContentType: "application/json"
});
```

---

## 🧪 步骤三：测试 Workflow

### 3.1 发送测试邮件

使用测试邮箱发送一封邮件到 `sandysu737@gmail.com`：

```
主题: 购买 su-memory Pro 版本

正文:
金额: ¥99
```

### 3.2 检查 Workflow 执行

1. 在 Pipedream Dashboard 查看 Workflow 执行日志
2. 确认每个步骤都正确执行
3. 检查是否收到自动回复邮件

---

## 🔐 步骤四：安全配置

### 4.1 添加授权签名验证

在 SDK 中添加签名验证，防止伪造授权：

```python
import hashlib
import hmac
import json

class LicenseValidator:
    SECRET_KEY = "your-secret-key-here"  # 从环境变量读取
    
    @staticmethod
    def verify_signature(license_data: dict) -> bool:
        """验证授权文件签名"""
        signature = license_data.pop("signature", None)
        if not signature:
            return False
        
        # 重新计算签名
        content = json.dumps(license_data, sort_keys=True)
        expected = hmac.new(
            LicenseValidator.SECRET_KEY.encode(),
            content.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected)
    
    @staticmethod
    def generate_license(email: str, license_type: str, capacity: int) -> dict:
        """生成授权文件（仅服务端使用）"""
        license_data = {
            "email": email,
            "license_type": license_type,
            "capacity": capacity,
            "issued_at": datetime.now().isoformat()
        }
        
        # 添加签名
        content = json.dumps(license_data, sort_keys=True)
        license_data["signature"] = hmac.new(
            LicenseValidator.SECRET_KEY.encode(),
            content.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return license_data
```

### 4.2 环境变量配置

在 Pipedream 中配置敏感信息：

1. Workflow Settings → Environment Variables
2. 添加：
   - `GMAIL_USER`: 你的 Gmail 地址
   - `GMAIL_APP_PASSWORD`: Gmail 应用密码
   - `LICENSE_SECRET_KEY`: 授权签名密钥

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

- [ ] 每周检查 Workflow 执行日志
- [ ] 确认授权文件格式正确
- [ ] 更新价格映射表

### 价格调整

如需调整价格，修改步骤 2.2.1 中的价格映射：

```javascript
const priceMap = {
  9: { type: "capacity_1k", capacity: 1000 },
  69: { type: "capacity_10k", capacity: 10000 },
  99: { type: "pro", capacity: 10000 },
  399: { type: "enterprise", capacity: 100000 },
  499: { type: "capacity_100k", capacity: 100000 },
  9999: { type: "on_premise", capacity: null }
};
```

---

## ❓ 常见问题

### Q: Pipedream 免费额度够用吗？

A: 免费额度：每月 10,000 次执行，足够小型项目使用。

### Q: 如何处理退款？

A: 在 Pipedream 中手动禁用对应授权码的 Workflow 步骤，或发送禁用邮件。

### Q: 邮件回复延迟怎么办？

A: Pipedream 免费版可能有 1-2 分钟延迟，如需实时处理可升级到付费版。

---

## 📞 联系支持

如需帮助，请联系：sandysu737@gmail.com

---

**文档版本**: v1.3.0  
**更新日期**: 2026-04-25
