#!/usr/bin/env python3
"""
简化的 Pipedream 工作流修改脚本

使用方式:
1. 先确保浏览器中已登录 Pipedream (sandysu737@gmail.com)
2. 然后运行: python3 deploy/interactive_pipedream.py

脚本会自动:
- 打开 Pipedream 并等待你登录
- 进入 su-memory SDK Payment 工作流
- 添加 route / create_alipay_order / process_alipay_notify 三个步骤
"""

import time
from playwright.sync_api import sync_playwright

WF_EDITOR_URL = "https://pipedream.com/workflows/su-memory-sdk-payment-p_8rCrPR/edit"

ROUTE_CODE = """async function routeRequest(steps) {
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
}"""

CREATE_ORDER_CODE = """const crypto = require('crypto');

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
  if (routeResult.action !== 'create_order') return null;

  const { order_id, plan_type, amount, buyer_email } = routeResult.data;
  if (!order_id || !plan_type || !amount) {
    return { status: 400, body: { error: '缺少必填参数' } };
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
    body: `su-memory SDK - 订单号: ${order_id}`,
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
    biz_content: bizContent,
  };

  alipayParams.sign = rsa256Sign(alipayParams, process.env.ALIPAY_PRIVATE_KEY);

  const gateway = process.env.ALIPAY_GATEWAY || 'https://openapi.alipay.com/gateway.do';
  const queryString = Object.entries(alipayParams)
    .map(([k, v]) => `${k}=${encodeURIComponent(v)}`)
    .join('&');

  return {
    status: 200,
    body: {
      success: true,
      order_id,
      payment_url: `${gateway}?${queryString}`,
      message: '请跳转至支付宝完成支付',
    },
  };
}"""

NOTIFY_CODE = """const crypto = require('crypto');

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
  } catch (e) { return false; }
}

function generateLicenseKey(planType) {
  const prefixMap = { community: 'COM', starter: 'STD', pro: 'PRO', enterprise: 'ENT', on_premise: 'ONP' };
  const prefix = prefixMap[planType] || 'UNK';
  const ts = Date.now().toString(16).toUpperCase();
  const rand = [...Array(8)].map(() => Math.floor(Math.random() * 16).toString(16)).join('').toUpperCase();
  return `SM-${prefix}-${ts}-${rand}`;
}

function hmacSign(licenseKey, secret) {
  if (!secret) return '';
  return crypto.createHmac('sha256', secret).update(licenseKey).digest('hex').slice(0, 16);
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
  if (routeResult.action === 'ping') return { status: 200, body: 'success' };
  if (routeResult.action !== 'alipay_notify') return null;

  const notifyData = routeResult.data;
  if (!rsa256Verify(notifyData, process.env.ALIPAY_PUBLIC_KEY)) {
    return { status: 400, body: 'fail' };
  }

  const tradeStatus = notifyData.trade_status;
  if (tradeStatus !== 'TRADE_SUCCESS' && tradeStatus !== 'TRADE_FINISHED') {
    return { status: 200, body: 'success' };
  }

  const outTradeNo = notifyData.out_trade_no;
  let planType = 'unknown', buyerEmail = '';
  try {
    const passback = JSON.parse(decodeURIComponent(notifyData.passback_params || '{}'));
    planType = passback.plan_type || 'unknown';
    buyerEmail = passback.buyer_email || '';
  } catch (e) {}

  const pkg = getCapacityConfig(planType);
  const now = new Date();
  const expires = planType === 'on_premise' ? 'never'
    : new Date(now.getFullYear() + 1, now.getMonth(), now.getDate()).toISOString().split('T')[0];

  const features = {};
  if (pkg.features[0] === '*') features['all'] = true;
  else pkg.features.forEach(f => { features[f] = true; });

  const licenseKey = generateLicenseKey(planType);
  const licenseData = {
    version: '1.0',
    license_key: licenseKey,
    license_type: planType,
    capacity: pkg.memories,
    issued_to: buyerEmail || notifyData.buyer_logon_id || '',
    issued_at: now.toISOString(),
    expires,
    features,
    order_id: outTradeNo,
    trade_no: notifyData.trade_no,
    amount: parseFloat(notifyData.total_amount || '0'),
    signature: hmacSign(licenseKey, process.env.LICENSE_SECRET),
  };

  console.log('Payment processed:', JSON.stringify({
    order_id: outTradeNo,
    plan: planType,
    license_key: licenseKey,
    email: licenseData.issued_to,
  }, null, 2));

  return { status: 200, body: 'success', license: licenseData };
}"""


def add_code_step(page, step_name, code, step_num):
    """点击 + 添加 Node.js Code 步骤"""
    print(f"\n  [Step {step_num}] 添加 {step_name}...")

    # 找到添加步骤的按钮
    add_buttons = page.locator('button')
    add_clicked = False

    # 方法1: 找画布上的 + 按钮
    for i in range(add_buttons.count()):
        btn = add_buttons.nth(i)
        try:
            txt = btn.inner_text()
            if '+' in txt and btn.is_visible():
                btn.click()
                add_clicked = True
                print(f"    点击了按钮: {repr(txt)}")
                break
        except:
            pass

    if not add_clicked:
        # 方法2: 直接找 "Add step" 链接
        try:
            page.click('text=Add step', timeout=3000)
            add_clicked = True
        except:
            pass

    if not add_clicked:
        # 方法3: 找 step-add 相关的元素
        try:
            page.click('[data-testid="add-step"]', timeout=3000)
            add_clicked = True
        except:
            pass

    time.sleep(1.5)

    # 选择 Node.js Code
    try:
        # Pipedream 新版 UI
        page.click('text=Node.js', timeout=3000)
        print("    选择了 Node.js")
    except:
        try:
            page.click('text=Run Node.js code', timeout=3000)
        except:
            try:
                page.click('text=Code', timeout=3000)
                time.sleep(0.5)
                page.click('text=Node.js', timeout=3000)
            except:
                pass

    time.sleep(1.5)

    # 修改步骤名称
    try:
        name_input = page.locator('[placeholder="Step name"]')
        if name_input.is_visible(timeout=3000):
            name_input.click()
            name_input.fill(step_name)
            print(f"    命名为: {step_name}")
    except:
        pass

    time.sleep(0.5)

    # 粘贴代码 - 多种方法尝试
    code_pasted = False
    for selector in ['.CodeMirror textarea', '.cm-content', 'textarea.code-mirror']:
        try:
            editor = page.locator(selector).first
            if editor.is_visible(timeout=2000):
                editor.click()
                editor.fill(code)
                code_pasted = True
                print(f"    代码已粘贴 ({selector})")
                break
        except:
            pass

    if not code_pasted:
        # 使用 keyboard 粘贴
        try:
            page.locator('.CodeMirror').first.click()
            time.sleep(0.3)
            page.keyboard.press("Control+a")
            page.keyboard.type(code)
            code_pasted = True
            print("    代码已通过键盘粘贴")
        except:
            pass

    time.sleep(0.5)

    # 保存
    try:
        page.click('text=Save', timeout=5000)
        print(f"    ✅ 已保存")
    except:
        try:
            page.click('text=Deploy', timeout=5000)
            print(f"    ✅ 已 Deploy")
        except:
            print(f"    ⚠️  请手动保存")

    time.sleep(2)


def main():
    print("=" * 60)
    print("Pipedream 工作流修改工具")
    print("=" * 60)
    print()
    print("⚠️  重要: 脚本将打开浏览器")
    print("   1. 请在打开的浏览器中完成 Google 登录")
    print("   2. 登录后，回到此终端窗口")
    print("   3. 输入 'y' 并回车继续")
    print()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, args=[
            '--disable-blink-images=0',
        ])
        context = browser.new_context(
            viewport={'width': 1400, 'height': 900},
            locale='zh-CN',
        )
        page = context.new_page()

        # 打开 Pipedream
        page.goto("https://pipedream.com/workflows/su-memory-sdk-payment-p_8rCrPR/edit")
        print("\n浏览器已打开，请在浏览器中完成登录...")
        print("登录后回到此终端输入 'y' 继续:")

        # 等待用户输入
        user_input = input("> ").strip().lower()
        if user_input != 'y':
            print("已取消")
            browser.close()
            return

        # 检查是否真的登录了
        page.wait_for_load_state("networkidle", timeout=10000)
        time.sleep(2)

        if "login" in page.url:
            print("⚠️  仍然需要登录，请在浏览器窗口中登录后再次输入 'y'")
            input("> ")
            page.wait_for_load_state("networkidle", timeout=10000)

        page.screenshot(path="pipedream_01_loggedin.png")
        print("✅ 已登录，截图: pipedream_01_loggedin.png")

        # === 检查当前工作流 URL ===
        current_url = page.url
        print(f"\n当前页面: {current_url}")

        # 如果不在工作流编辑器，导航过去
        if "/edit" not in current_url and "/steps" not in current_url:
            page.goto("https://pipedream.com/workflows/su-memory-sdk-payment-p_8rCrPR/edit")
            time.sleep(3)

        page.screenshot(path="pipedream_02_workflow_editor.png")
        print("📸 工作流编辑器截图: pipedream_02_workflow_editor.png")

        # === 添加三个 Code 步骤 ===
        add_code_step(page, "route", ROUTE_CODE, 1)
        page.screenshot(path="pipedream_03_route_added.png")

        add_code_step(page, "create_alipay_order", CREATE_ORDER_CODE, 2)
        page.screenshot(path="pipedream_04_create_order_added.png")

        add_code_step(page, "process_alipay_notify", NOTIFY_CODE, 3)
        page.screenshot(path="pipedream_05_notify_added.png")

        # === 最终截图 ===
        page.screenshot(path="pipedream_06_final.png")
        print("\n📸 最终截图: pipedream_06_final.png")
        print("\n🎉 步骤添加完成!")
        print("\n下一步: 在 Pipedream 中设置环境变量:")
        print("  1. 点击工作流右上角的 Settings")
        print("  2. 进入 Environment Variables")
        print("  3. 添加以下变量:")
        print("     ALIPAY_APP_ID = 2021006151644209")
        print("     ALIPAY_GATEWAY = https://openapi.alipay.com/gateway.do")
        print("     ALIPAY_NOTIFY_URL = https://eo91ihemgrxrlsy.m.pipedream.net/alipay-notify")
        print("     LICENSE_SECRET = 656916eccfca446108d6eea6d026ee53978a83bf7801158d5afeb08cf982815d")
        print("     ALIPAY_PRIVATE_KEY = (从 keys/alipay_private_key.pem 复制)")
        print("     ALIPAY_PUBLIC_KEY = (从 keys/alipay_public_key.pem 复制)")
        print("\n按 Enter 关闭浏览器...")
        input()
        browser.close()


if __name__ == "__main__":
    main()
