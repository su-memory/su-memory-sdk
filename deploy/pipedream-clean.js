import crypto from 'crypto';

const methods = {
  formatAlipayTimestamp(date) {
    const y = date.getFullYear();
    const m = String(date.getMonth() + 1).padStart(2, '0');
    const d = String(date.getDate()).padStart(2, '0');
    const h = String(date.getHours()).padStart(2, '0');
    const min = String(date.getMinutes()).padStart(2, '0');
    const s = String(date.getSeconds()).padStart(2, '0');
    return `${y}-${m}-${d} ${h}:${min}:${s}`;
  },

  rsa256Sign(params, privateKeyPem) {
    const sign = crypto.createSign('RSA-SHA256');
    const sortedKeys = Object.keys(params).sort();
    const signStr = sortedKeys.map(k => `${k}=${params[k]}`).join('&');
    sign.update(signStr, 'utf8');
    return sign.sign(privateKeyPem, 'base64');
  },

  rsa256Verify(notifyData, alipayPublicKeyPem) {
    const verify = crypto.createVerify('RSA-SHA256');
    const sortedKeys = Object.keys(notifyData).filter(k => k !== 'sign' && k !== 'sign_type').sort();
    const signStr = sortedKeys.map(k => `${k}=${notifyData[k]}`).join('&');
    verify.update(signStr, 'utf8');
    return verify.verify(alipayPublicKeyPem, notifyData.sign, 'base64');
  },

  generateLicenseKey(planType) {
    const prefix = 'SM';
    const ts = Date.now().toString(36).toUpperCase();
    const rand = Math.random().toString(36).substring(2, 8).toUpperCase();
    const plan = planType === 'pro' ? 'PR' : planType === 'team' ? 'TM' : 'ST';
    return `${prefix}-${plan}-${ts}-${rand}`;
  },

  hmacSign(licenseKey, secret) {
    return crypto.createHmac('sha256', secret).update(licenseKey).digest('hex').substring(0, 8).toUpperCase();
  },

  getCapacityConfig(planType) {
    const configs = {
      starter: { memory_limit: 10000, agents: 1, features: ['basic_memory', 'single_agent'] },
      pro: { memory_limit: 100000, agents: 5, features: ['advanced_memory', 'multi_agent', 'spatial', 'temporal'] },
      team: { memory_limit: 500000, agents: 20, features: ['advanced_memory', 'multi_agent', 'spatial', 'temporal', 'priority_support'] }
    };
    return configs[planType] || configs.starter;
  }
};

export default defineComponent({
  async run({ steps, $ }) {
    const { method, path, body } = steps.trigger.event;
    const pathClean = path ? path.split('?')[0] : '/';

    if (method === 'POST' && pathClean === '/create-order') {
      try {
        const payload = typeof body === 'string' ? JSON.parse(body) : body;
        const { order_id, plan_type, amount, buyer_email } = payload;

        const bizContent = {
          out_trade_no: order_id || `SM-${Date.now()}`,
          product_code: 'FAST_INSTANT_TRADE_PAY',
          total_amount: String(amount || 29.9),
          subject: `su-memory SDK ${plan_type || 'starter'} License`,
          body: `${plan_type || 'starter'} plan for ${buyer_email || 'user'}`,
          timeout_express: '15m'
        };

        const alipayParams = {
          app_id: process.env.ALIPAY_APP_ID,
          method: 'alipay.trade.page.pay',
          charset: 'utf-8',
          sign_type: 'RSA2',
          timestamp: this.formatAlipayTimestamp(new Date()),
          version: '1.0',
          notify_url: process.env.ALIPAY_NOTIFY_URL,
          biz_content: JSON.stringify(bizContent)
        };

        const gateway = process.env.ALIPAY_GATEWAY || 'https://openapi.alipay.com/gateway.do';
        const sign = this.rsa256Sign(alipayParams, process.env.ALIPAY_PRIVATE_KEY);
        const params = new URLSearchParams(alipayParams);
        params.append('sign', sign);
        const paymentUrl = `${gateway}?${params.toString()}`;

        return await $.respond({
          status: 200,
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ success: true, order_id: bizContent.out_trade_no, payment_url: paymentUrl })
        });
      } catch (err) {
        console.error('create-order error:', err.message);
        return await $.respond({ status: 500, body: JSON.stringify({ error: err.message }) });
      }
    }

    if (method === 'GET' && pathClean === '/alipay-notify') {
      return await $.respond({ status: 200, body: 'success' });
    }

    if (method === 'POST' && pathClean === '/alipay-notify') {
      try {
        const params = new URLSearchParams(typeof body === 'string' ? body : '');
        const notifyData = {};
        for (const [k, v] of params) { notifyData[k] = v; }

        const signOk = this.rsa256Verify(notifyData, process.env.ALIPAY_PUBLIC_KEY);
        if (!signOk) {
          console.error('签名验证失败');
          return await $.respond({ status: 400, body: 'fail' });
        }

        const tradeStatus = notifyData.trade_status;
        if (tradeStatus !== 'TRADE_SUCCESS' && tradeStatus !== 'TRADE_FINISHED') {
          return await $.respond({ status: 200, body: 'received' });
        }

        const planType = (notifyData.body || '').split(' ')[0] || 'starter';
        const licenseKey = this.generateLicenseKey(planType);
        const hmac = this.hmacSign(licenseKey, process.env.LICENSE_SECRET);
        const capacity = this.getCapacityConfig(planType);
        const buyerEmail = notifyData.buyer_email || notifyData.buyer_logon_id || '';

        const licenseData = {
          license_key: licenseKey,
          activation_code: `${licenseKey}-${hmac}`,
          plan_type: planType,
          buyer_email: buyerEmail,
          trade_no: notifyData.trade_no,
          total_amount: notifyData.total_amount,
          gmt_payment: notifyData.gmt_payment,
          capacity: capacity,
          issued_at: new Date().toISOString()
        };

        $.export('license_data', licenseData);
        return await $.respond({ status: 200, body: 'success' });
      } catch (err) {
        console.error('notify error:', err.message);
        return await $.respond({ status: 500, body: JSON.stringify({ error: err.message }) });
      }
    }

    return await $.respond({ status: 404, body: JSON.stringify({ error: 'Not found' }) });
  },
  methods
});
