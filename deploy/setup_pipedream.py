#!/usr/bin/env python3
"""
Pipedream 工作流代码粘贴脚本
直接操作 CodeMirror 编辑器，使用键盘快捷键粘贴代码
"""

import time
from playwright.sync_api import sync_playwright

WF_EDIT_URL = "https://pipedream.com/@sandysu737-workspace/projects/proj_gYspD2O/su-memory-sdk-payment-p_RRCoZxJ/build"

# === route 步骤代码 (defineComponent 格式) ===
ROUTE_CODE = """export default defineComponent({
  async run({ steps, $ }) {
    const { method, path, body, query } = steps.trigger.event;
    if (method === 'POST' && path === '/create-order') {
      return $.flow.exit({ action: 'create_order', data: body });
    }
    if (method === 'POST' && path === '/alipay-notify') {
      return $.flow.exit({ action: 'alipay_notify', data: body });
    }
    if (method === 'GET' && path === '/alipay-notify') {
      return $.flow.exit({ action: 'ping', data: query });
    }
    return $.flow.exit({ action: 'not_found', path, method });
  }
});"""

# === create_alipay_order 步骤代码 ===
CREATE_ORDER_CODE = """export default defineComponent({
  async run({ steps, $ }) {
    const routeResult = steps.route.$return_value;
    if (routeResult.action !== 'create_order') return;
    const { order_id, plan_type, amount, buyer_email } = routeResult.data;
    if (!order_id || !plan_type || !amount) {
      return $.respond({ status: 400, body: { error: '缺少必填参数' } });
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
      timestamp: formatTimestamp(new Date()),
      version: '1.0',
      notify_url: process.env.ALIPAY_NOTIFY_URL,
      biz_content: bizContent,
    };
    alipayParams.sign = rsaSign(alipayParams, process.env.ALIPAY_PRIVATE_KEY);
    const gateway = process.env.ALIPAY_GATEWAY || 'https://openapi.alipay.com/gateway.do';
    const qs = Object.entries(alipayParams)
      .map(([k, v]) => `${k}=${encodeURIComponent(v)}`)
      .join('&');
    return $.respond({
      status: 200,
      body: {
        success: true,
        order_id,
        payment_url: `${gateway}?${qs}`,
        message: '请跳转至支付宝完成支付',
      },
    });
  }
});

function formatTimestamp(date) {
  const pad = (n) => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

function rsaSign(params, privateKey) {
  const sortedKeys = Object.keys(params)
    .filter(k => k !== 'sign' && params[k] !== undefined && params[k] !== '')
    .sort();
  const signStr = sortedKeys.map(k => `${k}=${params[k]}`).join('&');
  const sign = require('crypto').createSign('RSA-SHA256');
  sign.update(signStr);
  sign.end();
  return sign.sign(privateKey, 'base64');
}"""

# === process_alipay_notify 步骤代码 ===
NOTIFY_CODE = """export default defineComponent({
  async run({ steps, $ }) {
    const routeResult = steps.route.$return_value;
    if (routeResult.action === 'ping') {
      return $.respond({ status: 200, body: 'success' });
    }
    if (routeResult.action !== 'alipay_notify') return;

    const notifyData = routeResult.data;
    if (!verifySign(notifyData, process.env.ALIPAY_PUBLIC_KEY)) {
      return $.respond({ status: 400, body: 'fail' });
    }
    const tradeStatus = notifyData.trade_status;
    if (tradeStatus !== 'TRADE_SUCCESS' && tradeStatus !== 'TRADE_FINISHED') {
      return $.respond({ status: 200, body: 'success' });
    }
    const outTradeNo = notifyData.out_trade_no;
    let planType = 'unknown', buyerEmail = '';
    try {
      const passback = JSON.parse(decodeURIComponent(notifyData.passback_params || '{}'));
      planType = passback.plan_type || 'unknown';
      buyerEmail = passback.buyer_email || '';
    } catch (e) {}
    const pkg = getCapacity(planType);
    const now = new Date();
    const expires = planType === 'on_premise' ? 'never'
      : new Date(now.getFullYear() + 1, now.getMonth(), now.getDate()).toISOString().split('T')[0];
    const features = {};
    if (pkg.features[0] === '*') features['all'] = true;
    else pkg.features.forEach(f => { features[f] = true; });
    const licenseKey = genLicenseKey(planType);
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
      order_id: outTradeNo, plan: planType,
      license_key: licenseKey, email: licenseData.issued_to,
    }, null, 2));
    return $.respond({ status: 200, body: 'success', headers: { 'Content-Type': 'text/plain' } });
  }
});

function verifySign(notifyData, alipayPublicKey) {
  try {
    const sign = notifyData.sign;
    if (!sign) return false;
    const sortedKeys = Object.keys(notifyData)
      .filter(k => k !== 'sign' && k !== 'sign_type' && notifyData[k] !== undefined && notifyData[k] !== '')
      .sort();
    const signStr = sortedKeys.map(k => `${k}=${notifyData[k]}`).join('&');
    const verify = require('crypto').createVerify('RSA-SHA256');
    verify.update(signStr);
    verify.end();
    return verify.verify(alipayPublicKey, sign, 'base64');
  } catch (e) { return false; }
}

function genLicenseKey(planType) {
  const prefixMap = { community: 'COM', starter: 'STD', pro: 'PRO', enterprise: 'ENT', on_premise: 'ONP' };
  const prefix = prefixMap[planType] || 'UNK';
  const ts = Date.now().toString(16).toUpperCase();
  const rand = [...Array(8)].map(() => Math.floor(Math.random() * 16).toString(16)).join('').toUpperCase();
  return `SM-${prefix}-${ts}-${rand}`;
}

function hmacSign(licenseKey, secret) {
  if (!secret) return '';
  return require('crypto').createHmac('sha256', secret).update(licenseKey).digest('hex').slice(0, 16);
}

function getCapacity(planType) {
  const map = {
    community:  { memories: 1000,   features: ['basic_query', 'tfidf', 'session_basic'] },
    starter:    { memories: 50000,  features: ['basic_query', 'tfidf', 'session_basic', 'vector_search'] },
    pro:        { memories: 200000, features: ['basic_query', 'tfidf', 'session_basic', 'vector_search', 'multihop', 'causal_inference', 'temporal', 'prediction'] },
    enterprise: { memories: -1,     features: ['*'] },
    on_premise: { memories: -1,     features: ['*'] },
  };
  return map[planType] || map.community;
}"""


def paste_code_in_editor(page, code):
    """在 CodeMirror 编辑器中粘贴代码"""
    # 方法1: 点击编辑器，使用 Ctrl+A 全选，然后粘贴
    editor = page.locator('.CodeMirror textarea, .cm-editor textarea, .CodeMirror .cm-content').first
    if editor.is_visible():
        editor.click()
        time.sleep(0.3)
        # 全选
        page.keyboard.press("Control+a")
        time.sleep(0.2)
        # 删除
        page.keyboard.press("Backspace")
        time.sleep(0.2)
        # 粘贴
        page.keyboard.type(code, delay=5)
        return True

    # 方法2: 使用 page.evaluate 直接设置 CodeMirror 内容
    try:
        escaped = code.replace('`', '\\`').replace('${', '\\${').replace('$', '\\$')
        eval_code = f"""
() => {{
  const cm = document.querySelector('.CodeMirror');
  if (cm && cm.CodeMirror) {{
    cm.CodeMirror.setValue(`{escaped}`);
  }}
}}
        """
        page.evaluate(eval_code)
        return True
    except Exception as e:
        print(f"  evaluate 失败: {e}")

    return False


def setup_step(page, step_name, code, step_idx):
    """设置单个步骤的代码"""
    print(f"\n--- [{step_idx}] 设置 {step_name} ---")

    # 点击步骤节点
    step_selectors = [
        f'text={step_name}',
        f'[data-testid="step-{step_name}"]',
        f'[data-testid="node-{step_name}"]',
    ]
    clicked = False
    for sel in step_selectors:
        try:
            el = page.locator(sel).first
            if el.is_visible(timeout=3000):
                el.click()
                clicked = True
                print(f"  点击了: {sel}")
                break
        except:
            pass

    if not clicked:
        # 尝试点击画布上的节点
        try:
            page.click(f'.wf-node:has-text("{step_name}")', timeout=3000)
            clicked = True
        except:
            pass

    if not clicked:
        print("  ⚠️  未找到步骤节点，跳过")
        return False

    time.sleep(1.5)

    # 截图步骤详情
    page.screenshot(path=f"step{step_idx}_detail.png")
    print("  📸 截图: step{step_idx}_detail.png")

    # 找到代码编辑区
    success = paste_code_in_editor(page, code)

    if success:
        print("  ✅ 代码已粘贴")
    else:
        print("  ⚠️  代码粘贴失败，请在编辑器中手动粘贴")

    # 点击保存
    time.sleep(0.5)
    save_buttons = ['text=Save', 'text=Deploy', 'text=Save changes']
    for txt in save_buttons:
        try:
            page.click(txt, timeout=3000)
            print("  ✅ 已保存")
            break
        except:
            pass

    time.sleep(2)
    return True


def main():
    print("=" * 60)
    print("Pipedream 代码粘贴脚本")
    print("=" * 60)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            viewport={'width': 1400, 'height': 900},
            locale='zh-CN',
        )
        page = context.new_page()

        print("\n正在打开 Pipedream...")
        page.goto(WF_EDIT_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(3)

        # 检查是否需要登录
        if "login" in page.url or "signin" in page.url:
            print("\n⚠️  需要登录，请在浏览器中完成登录后输入 'y':")
            input("> ")
            page.wait_for_load_state("networkidle")
            time.sleep(2)

        page.screenshot(path="pipedream_edit_start.png")
        print("📸 初始截图: pipedream_edit_start.png")

        # 依次设置三个步骤
        setup_step(page, "route", ROUTE_CODE, 1)
        setup_step(page, "create_alipay_order", CREATE_ORDER_CODE, 2)
        setup_step(page, "process_alipay_notify", NOTIFY_CODE, 3)

        # 最终截图
        page.screenshot(path="pipedream_final.png")
        print("\n📸 最终截图: pipedream_final.png")
        print("\n🎉 完成！请检查工作流编辑器确认代码是否正确。")
        print("\n下一步：设置环境变量（在 Pipedream Settings → Environment Variables）")
        print("按 Enter 关闭...")
        input()
        browser.close()


if __name__ == "__main__":
    main()
