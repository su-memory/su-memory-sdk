#!/usr/bin/env python3
"""
su-memory 模型下载脚本
自动下载 paraphrase-multilingual-MiniLM-L12-v2 到 HuggingFace 缓存
用法: python install_model.py
"""
import os, time, requests

COMMIT = 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'
BLOBS = os.path.expanduser('~/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/blobs')
os.makedirs(BLOBS, exist_ok=True)

small_files = [
    ('config.json', '配置文件'),
    ('modules.json', '模块配置'),
    ('tokenizer.json', '分词器'),
    ('tokenizer_config.json', '分词器配置'),
    ('1_Pooling/config.json', 'Pooling配置'),
]

for fname, desc in small_files:
    dest = os.path.join(BLOBS, fname.replace('/', '_'))
    url = f'https://hf-mirror.com/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/{COMMIT}/{fname}'
    print(f'下载 {fname} ({desc})...')
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(15, 300), headers={'User-Agent': 'Mozilla/5.0'}) as r:
                r.raise_for_status()
                total = int(r.headers.get('content-length', 0))
                downloaded = 0
                start = time.time()
                with open(dest, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total > 0:
                                pct = downloaded / total * 100
                                speed = downloaded / max(time.time() - start, 1) / 1024
                                print(f'\r  {downloaded/1024/1024:.1f}/{total/1024/1024:.1f}MB ({pct:.1f}%) {speed:.0f}KB/s', end='', flush=True)
            print(f'\n  ✅ {fname}: {os.path.getsize(dest)/1024/1024:.1f}MB')
            break
        except Exception as e:
            print(f'\n  ❌ (attempt {attempt+1}): {str(e)[:80]}')
            if attempt < 2:
                time.sleep(3)

print('\n小文件下载完成!')
print('继续下载 onnx/model.onnx (~449MB)...')

fname = 'onnx/model.onnx'
dest = os.path.join(BLOBS, 'onnx_model.onnx')
url = f'https://hf-mirror.com/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/{COMMIT}/{fname}'
print(f'URL: {url}')
start = time.time()
downloaded = 0
try:
    with requests.get(url, stream=True, timeout=(15, 3600), headers={'User-Agent': 'Mozilla/5.0'}) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        print(f'总大小: {total/1024/1024:.1f}MB')
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    elapsed = time.time() - start
                    speed = downloaded / elapsed / 1024 if elapsed > 0 else 0
                    pct = downloaded / total * 100 if total > 0 else 0
                    eta = (total - downloaded) / speed / 60 if speed > 0 else 0
                    print(f'\r  {downloaded/1024/1024:.1f}/{total/1024/1024:.1f}MB ({pct:.1f}%) {speed:.0f}KB/s ETA:{eta:.0f}min', end='', flush=True)
        print(f'\n✅ 完成! {os.path.getsize(dest)/1024/1024:.1f}MB 总耗时{time.time()-start:.0f}s')
except Exception as e:
    print(f'\n下载中断: {str(e)[:100]}')
    current = os.path.getsize(dest) if os.path.exists(dest) else 0
    print(f'已下载: {current/1024/1024:.1f}MB，可重新运行本脚本继续')
