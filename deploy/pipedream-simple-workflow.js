/**
 * su-memory SDK 支付宝支付 Pipedream 工作流 — 简化版
 * 
 * 将所有逻辑放在一个 Code 步骤中，避免复杂的 Switch 配置
 * 
 * ═══════════════════════════════════════════════════════════════
 * 环境变量（Pipedream Workflow Settings → Environment Variables）
 * ═══════════════════════════════════════════════════════════════
 * 
 * ALIPAY_APP_ID          = 2021006151644209
 * ALIPAY_PRIVATE_KEY     = 应用私钥（完整 PEM 格式）
 * ALIPAY_PUBLIC_KEY      = 支付宝公钥（完整 PEM 格式）
 * ALIPAY_GATEWAY         = https://openapi.alipay.com/gateway.do
 * ALIPAY_NOTIFY_URL      = https://eo91ihemgrxrlsy.m.pipedream.net/alipay-notify
 * LICENSE_SECRET         = License Key HMAC 签名密钥（至少 32 字符）
 * GMAIL_USER             = sandysu737@gmail.com
 * 
 * ═══════════════════════════════════════════════════════════════
 * 工作流步骤结构
 * ═══════════════════════════════════════════════════════════════
 * 
 * Step 0: HTTP Trigger  — 自动生成 URL
 * Step 1: Payment Handler — 统一的支付处理逻辑（所有路由都在这里）
 * Step 2: Send Email     — Gmail 步骤，发送 License Key
 * 
 * ═══════════════════════════════════════════════════════════════
 */

import crypto from 'crypto';

/**
 * 辅助函数 — 放在 methods 中供 run() 通过 this 调用
 */
const methods = {
  formatAlipayTimestamp(date) {
    const pad = (n) => String(n).padStart(2, '0');
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
  },

  rsa256Sign(params, privateKeyPem) {
    const sortedKeys = Object.keys(params)
      .filter(k => k !== 'sign' && params[k] !== undefined && params[k] !== '')
      .sort();
    const signStr = sortedKeys.map(k => `${k}=${params[k]}`).join('&');
    const sign = crypto.createSign('RSA-SHA256');
    sign.update(signStr);
    sign.end();
    return sign.sign(privateKeyPem, 'base64');
  },

  rsa256Verify(notifyData, alipayPublicKeyPem) {
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
  },

  generateLicenseKey(planType) {
    const prefixMap = { community: 'COM', starter: 'STD', pro: 'PRO', enterprise: 'ENT', on_premise: 'ONP' };
    const prefix = prefixMap[planType] || 'UNK';
    const ts = Date.now().toString(16).toUpperCase();
    const rnd = [...Array(8)].map(() => Math.floor(Math.random() * 16).toString(16)).join('').toUpperCase();
    return `SM-${prefix}-${ts}-${rnd}`;
  },

  hmacSign(licenseKey, secret) {
    if (!secret) return '';
    return crypto.createHmac('sha256', secret).update(licenseKey).digest('hex').slice(0, 16);
  },

  getCapacityConfig(planType) {
    const map = {
      community:  { memories: 1000,   features: ['basic_query', 'tfidf', 'session_basic'] },
      starter:    { memories: 50000,  features: ['basic_query', 'tfidf', 'session_basic', 'vector_search'] },
      pro:        { memories: 200000, features: ['basic_query', 'tfidf', 'session_basic', 'vector_search', 'multihop', 'causal_inference', 'temporal', 'prediction'] },
      enterprise: { memories: -1,     features: ['*'] },
      on_premise: { memories: -1,     features: ['*'] },
    };
    return map[planType] || map.community;
  }
};

// ═══════════════════════════════════════════════════════════════
// 主逻辑
// ═══════════════════════════════════════════════════════════════

export default defineComponent({
  async run({ steps, $ }) {
    const { method, path, body } = steps.trigger.event;
    
    console.log(`收到请求: ${method} ${path}`);
    
    // ═══════════════════════════════════════════════════════════════
    // 路由：POST /create-order — 创建支付宝订单
    // ═══════════════════════════════════════════════════════════════
    if (method === 'POST' && path === '/create-order') {
      const { order_id, plan_type, amount, buyer_email } = body || {};
      
      if (!order_id || !plan_type || !amount) {
        return await $.respond({ status: 400, body: JSON.stringify({ error: '缺少必填参数: order_id, plan_type, amount' }) });
      }
      
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
      
      const alipayParams = {
        app_id: process.env.ALIPAY_APP_ID,
        method: 'alipay.trade.page.pay',
        charset: 'utf-8',
        sign_type: 'RSA2',
        timestamp: this.formatAlipayTimestamp(new Date()),
        version: '1.0',
        notify_url: process.env.ALIPAY_NOTIFY_URL,
        biz_content: bizContent,
      };
      
      alipayParams.sign = this.rsa256Sign(alipayParams, process.env.ALIPAY_PRIVATE_KEY);
      
      const gateway = process.env.ALIPAY_GATEWAY || 'https://openapi.alipay.com/gateway.do';
      const queryString = Object.entries(alipayParams)
        .map(([k, v]) => `${k}=${encodeURIComponent(v)}`)
        .join('&');
      const paymentUrl = `${gateway}?${queryString}`;
      
      console.log('生成支付链接成功:', order_id);
      
        return await $.respond({ status: 200, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ success: true, order_id, payment_url: paymentUrl, message: '请跳转至支付宝完成支付' }) });
    }
    
    // ═══════════════════════════════════════════════════════════════
    // 路由：GET /alipay-notify — 支付宝网关验证（应用网关检查）
    // ═══════════════════════════════════════════════════════════════
    if (method === 'GET' && path === '/alipay-notify') {
      console.log('支付宝网关验证请求');
      return await $.respond({ status: 200, body: 'success' });
    }
    
    // ═══════════════════════════════════════════════════════════════
    // 路由：POST /alipay-notify — 处理支付宝异步通知
    // ═══════════════════════════════════════════════════════════════
    if (method === 'POST' && path === '/alipay-notify') {
      const notifyData = body || {};
      
      console.log('收到支付宝异步通知:', JSON.stringify(notifyData));
      
      // 验证签名
      const signOk = this.rsa256Verify(notifyData, process.env.ALIPAY_PUBLIC_KEY);
      if (!signOk) {
        console.error('签名验证失败');
        return await $.respond({ status: 400, body: 'fail' });
      }
      
      // 检查交易状态
      const tradeStatus = notifyData.trade_status;
      if (tradeStatus !== 'TRADE_SUCCESS' && tradeStatus !== 'TRADE_FINISHED') {
        console.log(`交易未完成: ${tradeStatus}`);
        return await $.respond({ status: 200, body: 'success' });
      }
      
      // 提取订单信息
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
      
      // 生成 License Key
      const licenseKey = this.generateLicenseKey(planType, outTradeNo);
      
      // 构建 License 数据
      const pkg = this.getCapacityConfig(planType);
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
        signature: this.hmacSign(licenseKey, process.env.LICENSE_SECRET),
      };
      
      // 记录日志
      console.log('✅ 支付处理完成:', JSON.stringify({
        order_id: outTradeNo,
        trade_no: tradeNo,
        plan: planType,
        license_key: licenseKey,
        email: buyerEmail || buyerLogonId,
      }, null, 2));
      
      // 导出 License 数据供 Gmail 步骤使用
      $.export('license_data', licenseData);
      
      // 返回 success 给支付宝
      return await $.respond({ status: 200, body: 'success' });
    }
    
    // ═══════════════════════════════════════════════════════════════
    // 默认：未找到路由
    // ═══════════════════════════════════════════════════════════════
    console.log(`未处理的请求: ${method} ${path}`);
    return await $.respond({ status: 404, body: JSON.stringify({ error: 'Not found' }) });
  },
  methods
});
