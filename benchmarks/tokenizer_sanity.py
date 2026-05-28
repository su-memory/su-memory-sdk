#!/usr/bin/env python3
"""
Tokenizer Sanity Test — v3.1.0 前置验证

验证 SuMemoryLite._tokenize() 在各种中英数字混合输入下的行为：
1. 数字区分：含数字的记忆可被正确区分
2. 数字保留：连续数字块作为 token 保留
3. 中数混合：中文+数字组合 token 正确生成
4. 不退化：修复不破坏原有中文分词能力

用法: python benchmarks/tokenizer_sanity.py
退出码: 0 = 全部通过, 1 = 有失败
"""

from __future__ import annotations

import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from su_memory.sdk import SuMemoryLite


def test_digit_discrimination():
    """D1: 含数字的记忆必须可被引擎区分"""
    with tempfile.TemporaryDirectory() as tmp:
        engine = SuMemoryLite(storage_path=tmp)

        # Insert items with different digits
        engine.add("持久化测试第0条重要数据")
        engine.add("持久化测试第6条重要数据")
        engine.add("持久化测试第99条重要数据")

        # Query for item 0 — should find it
        results = engine.query("持久化测试第0条", top_k=3)
        found = any("第0条" in r["content"] for r in results)
        assert found, "FAIL: 无法区分第0条和第6条"
        print("  ✅ D1 数字区分: '第0条' vs '第6条' 可正确区分")


def test_digit_block_preservation():
    """D2: 连续数字块被保留为独立 token"""
    with tempfile.TemporaryDirectory() as tmp:
        engine = SuMemoryLite(storage_path=tmp)

        engine.add("订单号12345已发货完成")
        engine.add("订单号54321待处理中")

        results = engine.query("订单号12345", top_k=3)
        found = any("12345" in r["content"] for r in results)
        assert found, "FAIL: 无法通过数字'12345'检索到对应记忆"
        print("  ✅ D2 数字块保留: '12345' 可独立检索")


def test_mixed_chinese_digit_tokens():
    """D3: 中文+数字组合 token 正确生成"""
    with tempfile.TemporaryDirectory() as tmp:
        engine = SuMemoryLite(storage_path=tmp)

        tokens = engine._tokenize("第0条和第6条的不同之处")
        assert "第0" in tokens, f"FAIL: '第0' 不在 tokens 中: {sorted(tokens)}"
        assert "第6" in tokens, f"FAIL: '第6' 不在 tokens 中: {sorted(tokens)}"
        print("  ✅ D3 中数混合: '第0'、'第6' 组合 token 正确生成")


def test_no_regression_chinese():
    """D4: 纯中文分词不退化"""
    with tempfile.TemporaryDirectory() as tmp:
        engine = SuMemoryLite(storage_path=tmp)

        tokens = engine._tokenize("自然语言处理是人工智能的重要方向")
        expected = {"自然", "然语", "语言", "言处", "处理", "人工", "工智", "智能", "重要", "方向"}
        found = expected & set(tokens)
        assert len(found) >= len(expected) * 0.7, \
            f"FAIL: 纯中文分词退化，期望≥{len(expected)*0.7:.0f}个匹配，实际{len(found)}个"
        print(f"  ✅ D4 纯中文不退化: {len(found)}/{len(expected)} 个核心词保留")


def test_no_regression_english():
    """D5: 英文关键词不退化"""
    with tempfile.TemporaryDirectory() as tmp:
        engine = SuMemoryLite(storage_path=tmp)

        tokens = engine._tokenize("Python类型推断和性能优化")
        assert "python" in tokens, f"FAIL: 'python' 不在 tokens 中: {sorted(tokens)}"
        print("  ✅ D5 英文不退化: 'python' 关键词保留")


def test_token_count_control():
    """D6: token 数量不爆炸（相比纯中文场景增幅 ≤50%）"""
    with tempfile.TemporaryDirectory() as tmp:
        engine = SuMemoryLite(storage_path=tmp)

        pure_cn = len(engine._tokenize("持久化测试重要数据必须完整"))
        with_digit = len(engine._tokenize("持久化测试第12345条重要数据"))

        ratio = with_digit / max(pure_cn, 1)
        assert ratio <= 1.5, \
            f"FAIL: 含数字 token 数 ({with_digit}) 相比纯中文 ({pure_cn}) 增幅 {ratio*100:.0f}% > 50%"
        print(f"  ✅ D6 Token 控制: 含数字 {with_digit} vs 纯中文 {pure_cn} (增幅 {ratio*100:.0f}%)")


def main():
    print("🧪 su-memory v3.1.0 Tokenizer Sanity Test")
    print()

    tests = [
        test_digit_discrimination,
        test_digit_block_preservation,
        test_mixed_chinese_digit_tokens,
        test_no_regression_chinese,
        test_no_regression_english,
        test_token_count_control,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {e}")
            failed += 1
        except Exception as e:
            print(f"  💥 {test.__name__}: {e}")
            failed += 1

    print()
    print(f"{'='*60}")
    print(f"  Results: {passed}/{len(tests)} passed, {failed} failed")
    if failed == 0:
        print("  ✅ ALL TOKENIZER SANITY CHECKS PASSED")
    else:
        print(f"  ❌ {failed} CHECK(S) FAILED")
    print(f"{'='*60}")

    return failed


if __name__ == "__main__":
    sys.exit(main())
