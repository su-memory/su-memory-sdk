/**
 * su-memory SDK 支付宝支付 Pipedream 工作流 — 参考代码
 *
 * ═══════════════════════════════════════════════════════════════
 * 架构说明
 * ═══════════════════════════════════════════════════════════════
 *
 * - su-memory SDK 是纯本地 SDK，无生产服务器
 * - 所有支付服务端逻辑部署在 Pipedream 上
 * - 支付宝私钥仅存储在 Pipedream 环境变量中，SDK 不持有任何密钥
 * - 支付宝异步通知直接发送到 Pipedream
 *
 * ═══════════════════════════════════════════════════════════════
 * 环境变量（Pipedream Workflow Settings → Environment Variables）
 * ═══════════════════════════════════════════════════════════════
 *
 * ALIPAY_APP_ID          = 2021006151644209
 * ALIPAY_PRIVATE_KEY     = 应用私钥（完整 PEM 格式，用于签名）
 * ALIPAY_PUBLIC_KEY      = 支付宝公钥（完整 PEM 格式，用于验签）
 * ALIPAY_GATEWAY         = https://openapi.alipay.com/gateway.do
 * ALIPAY_NOTIFY_URL      = https://YOUR_PIPEDREAM_WORKFLOW.m.pipedream.net/alipay-notify
 * LICENSE_SECRET         = License Key HMAC 签名密钥（至少 32 字符随机字符串）
 * GMAIL_USER             = sandysu737@gmail.com
 *
 * ═══════════════════════════════════════════════════════════════
 * Pipedream 工作流步骤结构
 * ═══════════════════════════════════════════════════════════════
 *
 * Step 0: HTTP Trigger  — 自动生成 URL，同时处理 /create-order 和 /alipay-notify
 * Step 1: 路由分发      — Code 步骤，解析请求路径分发到不同处理逻辑
 * Step 2: 创建订单      — Code 步骤，调用支付宝 page pay API
 * Step 3: 处理通知      — Code 步骤，验证异步通知 + 生成 License Key
 * Step 4: 发送邮件      — Gmail 步骤，发送 License Key 给用户
 *
 * 支付宝配置参考：
 * - APP_ID: 2021006151644209
 * - 商户PID: 2088580297860296
 * - 企业名称: 健源启晟（深圳）医疗科技有限公司
 * - 加签方式: RSA(SHA256) 公钥模式
 *
 * ═══════════════════════════════════════════════════════════════
 * Step 0: HTTP 触发器
 * ═══════════════════════════════════════════════════════════════
 *
 * 在 Pipedream 中点击 "HTTP / Webhook" 添加触发器。
 * 触发器会自动生成一个 URL，例如:
 *   https://eoyjjsu9jrea1nh.m.pipedream.net
 *
 * 将此 URL 填入上方 ALIPAY_NOTIFY_URL 环境变量，
 * 并在支付宝开放平台设置应用网关地址为此 URL。
 *
 * ═══════════════════════════════════════════════════════════════
 * Step 1: 路由分发 (Code 步骤 — Node.js)
 * ═══════════════════════════════════════════════════════════════
 *
 * 将以下代码粘贴到第一个 Code 步骤中：
 */

// --- Step 1 代码开始 ---
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

  // 非预期请求，返回 404
  return { action: 'not_found', path, method };
}
// --- Step 1 代码结束 ---


/**
 * ═══════════════════════════════════════════════════════════════
 * Step 2: 创建支付宝订单 (Code 步骤 — Node.js)
 * ═══════════════════════════════════════════════════════════════
 *
 * 仅当 Step 1 返回 action === 'create_order' 时执行
 * 使用支付宝 alipay.trade.page.pay 接口生成支付页面 URL
 * 将以下代码粘贴到第二个 Code 步骤中：
 */

// --- Step 2 代码开始 ---
const crypto = require('crypto');

async function createAlipayOrder(steps) {
  const routeResult = steps.route.$return_value;
  if (routeResult.action !== 'create_order') {
    return null; // 跳过此步骤
  }

  const { order_id, plan_type, amount, buyer_email } = routeResult.data;

  // 参数验证
  if (!order_id || !plan_type || !amount) {
    return {
      status: 400,
      body: { error: '缺少必填参数: order_id, plan_type, amount' },
    };
  }

  // ── 构建支付宝业务参数 ──
  const planNames = {
    community: '社区版 Community', starter: '入门版 Starter',
    pro: '专业版 Pro', enterprise: '企业版 Enterprise',
    on_premise: '私有部署版 On-Premise',
  };

  const bizContent = JSON.stringify({
    out_trade_no: order_id,
    product_code: 'FAST_INSTANT_TRADE_PAY',
    total_amount: Number(amount).toFixed(2),
    subject: `su-memory SDK ${planNames[plan_type] || plan_type}`,
    body: `su-memory SDK ${planNames[plan_type] || plan_type} - 订单号: ${order_id}`,
    passback_params: encodeURIComponent(
      JSON.stringify({ plan_type, buyer_email: buyer_email || '' })
    ),
  });

  // ── 构建支付宝公共参数 ──
  const alipayParams = {
    app_id: process.env.ALIPAY_APP_ID,
    method: 'alipay.trade.page.pay',
    charset: 'utf-8',
    sign_type: 'RSA2',
    timestamp: formatAlipayTimestamp(new Date()),
    version: '1.0',
    notify_url: process.env.ALIPAY_NOTIFY_URL,
    return_url: '',
    biz_content: bizContent,
  };

  // ── RSA2 签名 ──
  alipayParams.sign = rsa256Sign(alipayParams, process.env.ALIPAY_PRIVATE_KEY);

  // ── 构建支付页面 URL ──
  const gateway = process.env.ALIPAY_GATEWAY || 'https://openapi.alipay.com/gateway.do';
  const queryString = Object.entries(alipayParams)
    .map(([k, v]) => `${k}=${encodeURIComponent(v)}`)
    .join('&');
  const paymentUrl = `${gateway}?${queryString}`;

  return {
    status: 200,
    body: {
      success: true,
      order_id,
      payment_url: paymentUrl,
      message: '请跳转至支付宝完成支付',
    },
  };
}
// --- Step 2 代码结束 ---


/**
 * ═══════════════════════════════════════════════════════════════
 * Step 3: 处理支付宝异步通知 (Code 步骤 — Node.js)
 * ═══════════════════════════════════════════════════════════════
 *
 * 仅当 Step 1 返回 action === 'alipay_notify' 时执行
 * 验证签名、生成 License Key、构建 License 数据
 * 将以下代码粘贴到第三个 Code 步骤中：
 */

// --- Step 3 代码开始 ---
async function processAlipayNotify(steps) {
  const routeResult = steps.route.$return_value;

  // 支付宝 URL 可达性验证（GET 请求）
  if (routeResult.action === 'ping') {
    return { status: 200, body: 'success' };
  }

  if (routeResult.action !== 'alipay_notify') {
    return null;
  }

  const notifyData = routeResult.data;

  // ── 1. 验证签名 ──
  const signOk = rsa256Verify(notifyData, process.env.ALIPAY_PUBLIC_KEY);
  if (!signOk) {
    console.error('支付宝异步通知签名验证失败');
    return { status: 400, body: 'fail' };
  }

  // ── 2. 检查交易状态 ──
  const tradeStatus = notifyData.trade_status;
  if (tradeStatus !== 'TRADE_SUCCESS' && tradeStatus !== 'TRADE_FINISHED') {
    console.log(`交易状态未完成: ${tradeStatus}`);
    return { status: 200, body: 'success' };
  }

  // ── 3. 提取订单信息 ──
  const outTradeNo = notifyData.out_trade_no;
  const tradeNo = notifyData.trade_no;
  const totalAmount = parseFloat(notifyData.total_amount || '0');
  const buyerLogonId = notifyData.buyer_logon_id || '';

  let planType = 'unknown';
  let buyerEmail = '';
  try {
    const passback = JSON.parse(decodeURIComponent(notifyData.passback_params || '{}'));
    planType = passback.plan_type || 'unknown';
    buyerEmail = passback.buyer_email || '';
  } catch (e) {
    console.error('解析 passback_params 失败:', e.message);
  }

  // ── 4. 生成 License Key ──
  const licenseKey = generateLicenseKey(planType, outTradeNo);

  // ── 5. 构建 License 数据 ──
  const pkg = getCapacityConfig(planType);
  const now = new Date();
  const expires = planType === 'on_premise'
    ? 'never'
    : new Date(now.getFullYear() + 1, now.getMonth(), now.getDate())
        .toISOString().split('T')[0];

  const features = {};
  if (pkg.features[0] === '*') {
    features['all'] = true;
  } else {
    pkg.features.forEach(f => { features[f] = true; });
  }

  const licenseData = {
    version: '1.0',
    license_key: licenseKey,
    license_type: planType,
    capacity: pkg.memories,
    issued_to: buyerEmail || buyerLogonId,
    issued_at: now.toISOString(),
    expires,
    features,
    order_id: outTradeNo,
    trade_no: tradeNo,
    amount: totalAmount,
    signature: hmacSign(licenseKey, process.env.LICENSE_SECRET),
  };

  // ── 6. 记录日志 ──
  console.log('✅ 支付处理完成:', JSON.stringify({
    order_id: outTradeNo,
    trade_no: tradeNo,
    plan: planType,
    license_key: licenseKey,
    email: buyerEmail || buyerLogonId,
  }, null, 2));

  // 返回 success + license 数据（供 Step 4 Gmail 步骤使用）
  return {
    status: 200,
    body: 'success',
    license: licenseData,
  };
}
// --- Step 3 代码结束 ---


/**
 * ═══════════════════════════════════════════════════════════════
 * Step 4: 发送邮件 (Gmail — Send Email 步骤)
 * ═══════════════════════════════════════════════════════════════
 *
 * 类型: Gmail → Send Email
 * 连接已授权的 Gmail 账户 (sandysu737@gmail.com)
 *
 * To: {{steps.process_notify.license.issued_to}}
 * Subject: su-memory SDK 授权码 — {{steps.process_notify.license.license_key}}
 *
 * 使用以下 HTML 模板作为邮件内容：
 */

/*
<!DOCTYPE html>
<html>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 560px; margin: 0 auto; color: #333;">
  <div style="background: linear-gradient(135deg, #6366f1, #8b5cf6); padding: 28px 24px; border-radius: 12px 12px 0 0; text-align: center;">
    <h1 style="color: #fff; margin: 0; font-size: 22px;">🎉 感谢您的购买！</h1>
    <p style="color: rgba(255,255,255,0.85); margin: 6px 0 0; font-size: 14px;">su-memory SDK 授权已就绪</p>
  </div>

  <div style="background: #fff; padding: 24px; border: 1px solid #e5e7eb; border-top: none; border-radius: 0 0 12px 12px;">

    <p style="margin: 0 0 16px; color: #555;">您好，</p>
    <p style="margin: 0 0 20px; color: #555;">我们已收到您的支付宝付款，以下是您的授权信息：</p>

    <table style="width: 100%; border-collapse: collapse; margin-bottom: 24px;">
      <tr>
        <td style="padding: 10px 12px; border: 1px solid #e5e7eb; background: #f9fafb; font-weight: 600; width: 120px;">授权码</td>
        <td style="padding: 10px 12px; border: 1px solid #e5e7eb; font-family: 'Courier New', monospace; font-size: 13px; word-break: break-all;">
          {{steps.process_notify.license.license_key}}
        </td>
      </tr>
      <tr>
        <td style="padding: 10px 12px; border: 1px solid #e5e7eb; background: #f9fafb; font-weight: 600;">版本类型</td>
        <td style="padding: 10px 12px; border: 1px solid #e5e7eb;">
          {{steps.process_notify.license.license_type}}
        </td>
      </tr>
      <tr>
        <td style="padding: 10px 12px; border: 1px solid #e5e7eb; background: #f9fafb; font-weight: 600;">记忆容量</td>
        <td style="padding: 10px 12px; border: 1px solid #e5e7eb;">
          {{steps.process_notify.license.capacity === -1 ? "无限制" : steps.process_notify.license.capacity + " 条"}}
        </td>
      </tr>
      <tr>
        <td style="padding: 10px 12px; border: 1px solid #e5e7eb; background: #f9fafb; font-weight: 600;">订单号</td>
        <td style="padding: 10px 12px; border: 1px solid #e5e7eb; font-family: monospace; font-size: 12px;">
          {{steps.process_notify.license.order_id}}
        </td>
      </tr>
      <tr>
        <td style="padding: 10px 12px; border: 1px solid #e5e7eb; background: #f9fafb; font-weight: 600;">支付宝交易号</td>
        <td style="padding: 10px 12px; border: 1px solid #e5e7eb; font-family: monospace; font-size: 12px;">
          {{steps.process_notify.license.trade_no}}
        </td>
      </tr>
      <tr>
        <td style="padding: 10px 12px; border: 1px solid #e5e7eb; background: #f9fafb; font-weight: 600;">支付金额</td>
        <td style="padding: 10px 12px; border: 1px solid #e5e7eb;">¥{{steps.process_notify.license.amount}}</td>
      </tr>
      <tr>
        <td style="padding: 10px 12px; border: 1px solid #e5e7eb; background: #f9fafb; font-weight: 600;">到期时间</td>
        <td style="padding: 10px 12px; border: 1px solid #e5e7eb;">
          {{steps.process_notify.license.expires === "never" ? "永久有效" : steps.process_notify.license.expires}}
        </td>
      </tr>
    </table>

    <h3 style="margin: 0 0 12px; font-size: 16px;">📋 使用方法</h3>
    <ol style="margin: 0 0 20px; padding-left: 20px; color: #555; line-height: 1.8;">
      <li>复制上方<strong>授权码</strong></li>
      <li>设置环境变量:
        <pre style="background: #f3f4f6; padding: 8px 12px; border-radius: 6px; font-size: 12px; margin: 6px 0;">export SU_MEMORY_LICENSE_KEY={{steps.process_notify.license.license_key}}</pre>
      </li>
      <li>或将下方 JSON 保存为 <code>~/.su-memory/license.json</code></li>
      <li>重新启动应用，授权自动生效</li>
    </ol>

    <h3 style="margin: 0 0 8px; font-size: 16px;">📁 license.json</h3>
    <pre style="background: #f3f4f6; padding: 14px; border-radius: 6px; overflow-x: auto; font-size: 11px; line-height: 1.5;">{{JSON.stringify(steps.process_notify.license, null, 2)}}</pre>

    <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 20px 0;">

    <p style="color: #888; font-size: 13px;">
      如有任何问题，请回复此邮件或联系 <a href="mailto:sandysu737@gmail.com" style="color: #6366f1;">sandysu737@gmail.com</a>
    </p>
    <p style="color: #888; font-size: 13px; margin-bottom: 0;">感谢您的支持！</p>
  </div>

  <div style="text-align: center; padding: 16px; color: #9ca3af; font-size: 12px;">
    <strong>su-memory SDK 团队</strong><br>
    <a href="https://su-memory.ai" style="color: #9ca3af;">su-memory.ai</a> |
    <a href="https://github.com/su-memory/su-memory-sdk" style="color: #9ca3af;">GitHub</a>
  </div>
</body>
</html>
*/


// ═══════════════════════════════════════════════════════════════
// 共享辅助函数
// ═══════════════════════════════════════════════════════════════
//
// 以下函数需要在 Step 2 和 Step 3 中使用。
// 在 Pipedream 中，可以将它们放在每个 Code 步骤的最顶部，
// 或者创建 Shared Code 步骤供其他步骤引用。

/**
 * 格式化支付宝时间戳
 */
function formatAlipayTimestamp(date) {
  const pad = (n) => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

/**
 * 支付宝 RSA2 签名（SHA256WithRSA）
 */
function rsa256Sign(params, privateKeyPem) {
  const sortedKeys = Object.keys(params)
    .filter(k => k !== 'sign' && params[k] !== undefined && params[k] !== '')
    .sort();
  const signStr = sortedKeys.map(k => `${k}=${params[k]}`).join('&');
  const sign = crypto.createSign('RSA-SHA256');
  sign.update(signStr);
  sign.end();
  return sign.sign(privateKeyPem, 'base64');
}

/**
 * 验证支付宝异步通知 RSA2 签名
 */
function rsa256Verify(notifyData, alipayPublicKeyPem) {
  try {
    const sign = notifyData.sign;
    if (!sign) return false;
    const sortedKeys = Object.keys(notifyData)
      .filter(k => k !== 'sign' && k !== 'sign_type' && notifyData[k] !== undefined && notifyData[k] !== '')
      .sort();
    const signStr = sortedKeys.map(k => `${k}=${notifyData[k]}`).join('&');
    const verify = crypto.createVerify('RSA-SHA256');
    verify.update(signStr);
    verify.end();
    return verify.verify(alipayPublicKeyPem, sign, 'base64');
  } catch (e) {
    console.error('验签异常:', e.message);
    return false;
  }
}

/**
 * 生成 License Key
 * 格式: SM-{plan_prefix}-{timestamp_hex}-{8位hex}
 */
function generateLicenseKey(planType, orderId) {
  const prefixMap = {
    community: 'COM', starter: 'STD', pro: 'PRO',
    enterprise: 'ENT', on_premise: 'ONP',
  };
  const prefix = prefixMap[planType] || 'UNK';
  const timestamp = Date.now().toString(16).toUpperCase();
  const random = [...Array(8)]
    .map(() => Math.floor(Math.random() * 16).toString(16))
    .join('').toUpperCase();
  return `SM-${prefix}-${timestamp}-${random}`;
}

/**
 * HMAC-SHA256 签名 License Key
 */
function hmacSign(licenseKey, secret) {
  if (!secret) return '';
  return crypto
    .createHmac('sha256', secret)
    .update(licenseKey)
    .digest('hex')
    .slice(0, 16);
}

/**
 * 获取套餐容量配置
 */
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
