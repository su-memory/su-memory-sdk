#!/usr/bin/env python3
"""
自动修改 Pipedream 工作流：su-memory SDK Payment
添加路由、创建支付宝订单、处理异步通知、Gmail 发送的步骤

使用方式: python3 deploy/update_pipedream.py

要求:
- pip install playwright && python3 -m playwright install chromium
- Google 已登录 sandysu737@gmail.com
"""

import time
from playwright.sync_api import sync_playwright

PIPEDREAM_WF_URL = "https://pipedream.com/workflows/su-memory-sdk-payment-p_8rCrPR"

# === Step 1: 路由分发 ===
ROUTE_CODE = """
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
""".strip()

# === Step 2: 创建支付宝订单 ===
CREATE_ORDER_CODE = """
const crypto = require('crypto');

function formatAlipayTimestamp(date) {
  const pad = (n) => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

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

async function createAlipayOrder(steps) {
  const routeResult = steps.route.$return_value;
  if (routeResult.action !== 'create_order') {
    return null;
  }

  const { order_id, plan_type, amount, buyer_email } = routeResult.data;

  if (!order_id || !plan_type || !amount) {
    return {
      status: 400,
      body: { error: '缺少必填参数: order_id, plan_type, amount' },
    };
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
    timestamp: formatAlipayTimestamp(new Date()),
    version: '1.0',
    notify_url: process.env.ALIPAY_NOTIFY_URL,
    return_url: '',
    biz_content: bizContent,
  };

  alipayParams.sign = rsa256Sign(alipayParams, process.env.ALIPAY_PRIVATE_KEY);

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
""".strip()

# === Step 3: 处理异步通知 ===
NOTIFY_CODE = """
const crypto = require('crypto');

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

function hmacSign(licenseKey, secret) {
  if (!secret) return '';
  return crypto
    .createHmac('sha256', secret)
    .update(licenseKey)
    .digest('hex')
    .slice(0, 16);
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

async function processAlipayNotify(steps) {
  const routeResult = steps.route.$return_value;

  if (routeResult.action === 'ping') {
    return { status: 200, body: 'success' };
  }

  if (routeResult.action !== 'alipay_notify') {
    return null;
  }

  const notifyData = routeResult.data;

  const signOk = rsa256Verify(notifyData, process.env.ALIPAY_PUBLIC_KEY);
  if (!signOk) {
    console.error('支付宝异步通知签名验证失败');
    return { status: 400, body: 'fail' };
  }

  const tradeStatus = notifyData.trade_status;
  if (tradeStatus !== 'TRADE_SUCCESS' && tradeStatus !== 'TRADE_FINISHED') {
    console.log(`交易状态未完成: ${tradeStatus}`);
    return { status: 200, body: 'success' };
  }

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

  const licenseKey = generateLicenseKey(planType, outTradeNo);

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

  console.log('Payment processed:', JSON.stringify({
    order_id: outTradeNo,
    trade_no: tradeNo,
    plan: planType,
    license_key: licenseKey,
    email: buyerEmail || buyerLogonId,
  }, null, 2));

  return {
    status: 200,
    body: 'success',
    license: licenseData,
  };
}
""".strip()


def main():
    with sync_playwright() as p:
        # Use persistent context to reuse Google login session
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        print("=" * 60)
        print("Pipedream Workflow 修改工具")
        print("=" * 60)
        print()
        print("请确保 Google 账号 sandysu737@gmail.com 已登录")
        print()

        # Navigate to Pipedream
        page.goto("https://pipedream.com")
        page.wait_for_load_state("networkidle")
        print("✅ Pipedream 页面加载完成")
        time.sleep(2)

        # Navigate to the workflow
        page.goto(PIPEDREAM_WF_URL)
        page.wait_for_load_state("networkidle")
        print("✅ 工作流页面加载完成")
        time.sleep(3)

        # Check if we need to login
        if "login" in page.url or "signin" in page.url:
            print("⚠️  需要登录，请在浏览器中完成 Google 登录")
            print("   登录后按 Enter 继续...")
            input()
            page.goto(PIPEDREAM_WF_URL)
            page.wait_for_load_state("networkidle")
            time.sleep(3)

        # Take initial screenshot
        page.screenshot(path="pipedream_before.png")
        print("📸 当前状态截图: pipedream_before.png")

        # === Check current steps ===
        print("\n当前步骤:")
        steps_before = page.locator('[data-testid="step"]').count()
        print(f"  步骤数: {steps_before}")

        # Try to find and click the "Add step" button (the + button after trigger)
        add_step_selectors = [
            'button:has-text("Add")',
            '[data-testid="add-step"]',
            'button[aria-label="Add step"]',
            '.step-add button',
            'text=Add step',
        ]

        # === Add Step 1: route ===
        print("\n--- 添加 Step 1: route ---")
        add_success = False
        for selector in add_step_selectors:
            try:
                btn = page.locator(selector).first
                if btn.is_visible(timeout=2000):
                    btn.click()
                    add_success = True
                    print(f"  点击了: {selector}")
                    break
            except:
                continue

        if not add_success:
            print("  ⚠️  未找到添加步骤按钮，尝试点击画布上的 + 节点")
            page.click("text=step-add-4")

        time.sleep(2)

        # Select Code (Node.js)
        try:
            page.click("text=Code")
            time.sleep(1)
            page.click("text=Node.js")
        except:
            page.click("text=Run Node.js code")
        time.sleep(2)

        # Rename step to "route"
        try:
            page.click('[placeholder="Step name"]')
            page.fill('[placeholder="Step name"]', "route")
        except:
            pass

        # Clear existing code and paste
        try:
            code_editor = page.locator('.CodeMirror').first
            code_editor.click()
            page.keyboard.press("Meta+a")
            page.keyboard.press("Backspace")
            page.keyboard.type(ROUTE_CODE, delay=10)
        except:
            # Fallback: use textarea
            textarea = page.locator('textarea').first
            textarea.fill(ROUTE_CODE)

        # Click Test or Save
        try:
            page.click("text=Save")
        except:
            page.click("text=Deploy")

        print("  ✅ route 步骤已添加")
        time.sleep(3)

        # === Add Step 2: create_alipay_order ===
        print("\n--- 添加 Step 2: create_alipay_order ---")
        # Click add step after route
        try:
            page.locator('button:has-text("Add")').last.click()
        except:
            page.click("text=step-add")
        time.sleep(2)

        try:
            page.click("text=Code")
            time.sleep(1)
            page.click("text=Node.js")
        except:
            page.click("text=Run Node.js code")
        time.sleep(2)

        # Rename
        try:
            page.locator('[placeholder="Step name"]').click()
            page.locator('[placeholder="Step name"]').fill("create_alipay_order")
        except:
            pass

        # Paste code
        try:
            code_editor = page.locator('.CodeMirror').last
            code_editor.click()
            page.keyboard.press("Meta+a")
            page.keyboard.press("Backspace")
            page.keyboard.type(CREATE_ORDER_CODE, delay=10)
        except:
            textarea = page.locator('textarea').last
            textarea.fill(CREATE_ORDER_CODE)

        try:
            page.click("text=Save")
        except:
            page.click("text=Deploy")

        print("  ✅ create_alipay_order 步骤已添加")
        time.sleep(3)

        # === Add Step 3: process_alipay_notify ===
        print("\n--- 添加 Step 3: process_alipay_notify ---")
        try:
            page.locator('button:has-text("Add")').last.click()
        except:
            page.click("text=step-add")
        time.sleep(2)

        try:
            page.click("text=Code")
            time.sleep(1)
            page.click("text=Node.js")
        except:
            page.click("text=Run Node.js code")
        time.sleep(2)

        try:
            page.locator('[placeholder="Step name"]').click()
            page.locator('[placeholder="Step name"]').fill("process_alipay_notify")
        except:
            pass

        try:
            code_editor = page.locator('.CodeMirror').last
            code_editor.click()
            page.keyboard.press("Meta+a")
            page.keyboard.press("Backspace")
            page.keyboard.type(NOTIFY_CODE, delay=10)
        except:
            textarea = page.locator('textarea').last
            textarea.fill(NOTIFY_CODE)

        try:
            page.click("text=Save")
        except:
            page.click("text=Deploy")

        print("  ✅ process_alipay_notify 步骤已添加")
        time.sleep(3)

        # === Deploy ===
        print("\n--- 部署工作流 ---")
        try:
            page.click("text=Deploy")
            print("  ✅ 部署按钮已点击")
        except:
            print("  ⚠️  请手动点击 Deploy")

        time.sleep(3)
        page.screenshot(path="pipedream_after.png")
        print("\n📸 最终状态截图: pipedream_after.png")
        print("\n🎉 脚本执行完成！请在浏览器中检查工作流状态。")
        print("   按 Enter 关闭浏览器...")
        input()
        browser.close()


if __name__ == "__main__":
    main()
