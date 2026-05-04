#!/usr/bin/env python3
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context(viewport={'width': 1400, 'height': 900})
    page = context.new_page()
    
    page.goto("https://pipedream.com/@sandysu737-workspace/projects/proj_gYspD2O/su-memory-sdk-payment-p_RRCoZxJ/build")
    page.wait_for_load_state("networkidle")
    
    page.screenshot(path="debug_workflow.png", full_page=False)
    
    print("=== 可点击文本 ===")
    all_text = page.locator('text').all()
    for el in all_text[:30]:
        try:
            txt = el.inner_text()
            if txt.strip():
                print(f"  '{txt.strip()[:50]}'")
        except:
            pass
    
    print("\n=== 可点击元素 ===")
    clickables = page.locator('button, a, [role="button"]').all()
    for el in clickables[:20]:
        try:
            txt = el.inner_text().strip()
            title = el.get_attribute('title') or ''
            cls = el.get_attribute('class') or ''
            if txt or title:
                print(f"  text='{txt[:40]}' title='{title[:30]}' cls='{cls[:50]}'")
        except:
            pass
    
    print("\n=== SVG 节点文本 ===")
    svg_texts = page.locator('svg text, svg tspan').all()
    for el in svg_texts[:20]:
        try:
            txt = el.inner_text().strip()
            if txt:
                print(f"  SVG: '{txt}'")
        except:
            pass
    
    print("\n按 Enter 关闭...")
    input()
    browser.close()
