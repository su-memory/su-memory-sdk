#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/mac/.openclaw/workspace/su-memory')

print('=== su-memory 核心验证 ===')

from memory_engine.extractor import MemoryExtractor

ext = MemoryExtractor()
r = ext.extract_sync('用户孩子5岁平足需要矫正', {'type': 'fact'})

print('encoding_info:', r['encoding_info'])
print('priority:', r['priority'])
print('type:', r['type'])
print('entities:', len(r['entities']))
print()
print('extract_sync OK')
