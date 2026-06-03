#!/usr/bin/env python3
"""Convert all 3 venue HTML files to PDF."""
import asyncio, os
from playwright.async_api import async_playwright

BASE = "/Users/mac/qoder m5pro/su-memory-sdk/docs"

async def html_to_pdf(venue):
    html_path = os.path.join(BASE, venue, f"MCI_World_Model_v3.5.0_{venue.upper()}.html")
    pdf_path = os.path.join(BASE, venue, f"MCI_World_Model_v3.5.0_{venue.upper()}.pdf")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(f"file://{html_path}", wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(5000)
        await page.pdf(path=pdf_path, format="A4",
            margin={"top":"1in","bottom":"1in","left":"1in","right":"1in"},
            print_background=True)
        await browser.close()
    
    size_kb = os.path.getsize(pdf_path) / 1024
    print(f"✅ {venue.upper()}: {pdf_path} ({size_kb:.0f} KB)")

async def main():
    for venue in ["arxiv", "uai", "jmlr"]:
        await html_to_pdf(venue)
    print("\\nAll PDFs generated!")

asyncio.run(main())
