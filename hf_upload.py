#!/usr/bin/env python3
"""su-memory v2.5.0 → HuggingFace 上传工具 (双击运行)"""
import urllib.request, json, os, sys

TOKEN = os.environ.get("HF_TOKEN", "")
REPO = "su-memory/su-memory-sdk"
API = "https://huggingface.co/api"

# 找到 dist 目录
script_dir = os.path.dirname(os.path.abspath(__file__))
dist_dir = os.path.join(script_dir, "dist")
files = [f for f in os.listdir(dist_dir) if "2.5.0" in f]

if not files:
    input("未找到 v2.5.0 文件，按回车退出")
    sys.exit(1)

print(f"准备上传 {len(files)} 个文件到 {REPO}...")
print()

for fname in files:
    fpath = os.path.join(dist_dir, fname)
    size = os.path.getsize(fpath)
    print(f"上传 {fname} ({size/1024:.0f}KB)...", end=" ", flush=True)
    
    try:
        with open(fpath, "rb") as f:
            data = f.read()
        req = urllib.request.Request(
            f"{API}/models/{REPO}/upload/{fname}",
            data=data,
            headers={"Authorization": f"Bearer {TOKEN}"},
            method="POST"
        )
        resp = urllib.request.urlopen(req, timeout=60)
        print(f"✅ ({resp.status})")
    except Exception as e:
        print(f"❌ {e}")

print()
print(f"检查: https://huggingface.co/{REPO}")
input("按回车退出")
