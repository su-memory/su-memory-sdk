#!/usr/bin/env python3
"""Convert HTML to PDF using Playwright (Chromium headless)"""
import asyncio
from playwright.async_api import async_playwright
import os

async def html_to_pdf():
    html_path = os.path.expanduser("~/Desktop/MCI_World_Model_v3.5.0_LaTeX.html")
    pdf_path = os.path.expanduser("~/Desktop/MCI_World_Model_v3.5.0_LaTeX.pdf")
    
    file_url = f"file://{html_path}"
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(file_url, wait_until="networkidle", timeout=30000)
        # Wait extra time for MathJax to render
        await page.wait_for_timeout(5000)
        await page.pdf(
            path=pdf_path,
            format="A4",
            margin={"top": "1in", "bottom": "1in", "left": "1in", "right": "1in"},
            print_background=True,
        )
        await browser.close()
    
    size_kb = os.path.getsize(pdf_path) / 1024
    print(f"✅ PDF generated: {pdf_path} ({size_kb:.0f} KB)")

asyncio.run(html_to_pdf())
