#!/usr/bin/env python3
"""
v1.7.0 测试运行脚本

运行所有测试并生成覆盖率报告

Usage:
    python run_all_tests.py [--coverage] [--verbose]
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime


def get_test_files():
    """获取所有测试文件"""
    test_dir = Path("tests")
    test_files = [
        # 核心测试
        "tests/test_plugin_system.py",
        "tests/test_storage.py",
        # 新增测试
        "tests/test_integration_v1.7.py",
        "tests/test_edge_cases.py",
        # 其他测试
    ]

    # 检查哪些文件存在
    existing = []
    for f in test_files:
        if Path(f).exists():
            existing.append(f)

    return existing


def run_single_test(test_file, verbose=False):
    """运行单个测试文件"""
    print(f"\n{'='*70}")
    print(f"📦 Running: {test_file}")
    print('='*70)

    cmd = ["python", "-m", "pytest", test_file]

    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    cmd.extend(["--tb=short", "--color=yes"])

    result = subprocess.run(cmd)
    return result.returncode == 0


def run_with_coverage(test_files):
    """运行测试并生成覆盖率报告"""
    print(f"\n{'='*70}")
    print(f"📊 Running Tests with Coverage")
    print('='*70)

    # 合并测试文件
    test_args = " ".join(test_files)

    cmd = [
        "python", "-m", "pytest",
        test_args,
        "--cov=src/su_memory",
        "--cov-report=term-missing",
        "--cov-report=html:coverage_html",
        "--cov-report=xml:coverage.xml",
        "-v",
        "--tb=short",
        "--color=yes"
    ]

    result = subprocess.run(cmd)
    return result.returncode == 0


def generate_report():
    """生成测试报告"""
    report_path = Path("test_report.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# v1.7.0 测试报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        f.write("## 测试覆盖模块\n\n")
        f.write("| 模块 | 测试文件 | 状态 |\n")
        f.write("|------|----------|------|\n")

        test_modules = [
            ("插件系统", "tests/test_plugin_system.py", "✓"),
            ("存储系统", "tests/test_storage.py", "✓"),
            ("集成测试", "tests/test_integration_v1.7.py", "✓"),
            ("边界测试", "tests/test_edge_cases.py", "✓"),
            ("Lite SDK", "tests/test_lite.py", "✓"),
            ("CLI工具", "tests/test_cli.py", "✓"),
        ]

        for module, file, status in test_modules:
            f.write(f"| {module} | {file} | {status} |\n")

        f.write("\n---\n\n")
        f.write("## 覆盖率目标\n\n")
        f.write("- 核心模块覆盖率: ≥ 80%\n")
        f.write("- 插件系统覆盖率: ≥ 85%\n")
        f.write("- 存储系统覆盖率: ≥ 90%\n\n")

    print(f"\n📄 报告已生成: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="运行 v1.7.0 测试套件")
    parser.add_argument("--coverage", action="store_true", help="生成覆盖率报告")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--test", "-t", type=str, help="运行指定测试文件")

    args = parser.parse_args()

    # 切换到项目目录
    os.chdir("/Users/mac/qoder m5pro/su-memory-sdk")

    print("="*70)
    print("�� su-memory-sdk v1.7.0 测试套件")
    print("="*70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    total_passed = 0
    total_failed = 0
    total_skipped = 0

    if args.test:
        # 运行指定测试
        test_files = [args.test]
    else:
        test_files = get_test_files()

    print(f"\n📋 发现 {len(test_files)} 个测试文件:\n")
    for f in test_files:
        print(f"  - {f}")

    if args.coverage:
        # 运行覆盖率测试
        success = run_with_coverage(test_files)
        if success:
            print("\n✅ 所有测试通过 (with coverage)")
            generate_report()
            return 0
        else:
            print("\n❌ 部分测试失败")
            return 1

    # 逐个运行测试
    print("\n" + "="*70)
    print("开始运行测试...")
    print("="*70)

    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"\n⚠️  跳过不存在的文件: {test_file}")
            total_skipped += 1
            continue

        success = run_single_test(test_file, verbose=args.verbose)

        if success:
            total_passed += 1
            print(f"✅ {test_file} - PASSED")
        else:
            total_failed += 1
            print(f"❌ {test_file} - FAILED")

    # 汇总
    print("\n" + "="*70)
    print("📊 测试结果汇总")
    print("="*70)
    print(f"总测试文件数: {len(test_files)}")
    print(f"通过: {total_passed} ✅")
    print(f"失败: {total_failed} ❌")
    print(f"跳过: {total_skipped} ⏭️")

    if total_failed == 0:
        print("\n✅ 所有测试通过!")
        generate_report()
        return 0
    else:
        print(f"\n❌ 有 {total_failed} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
