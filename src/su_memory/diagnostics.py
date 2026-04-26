#!/usr/bin/env python3
"""
su-memory SDK 安装诊断工具

用于诊断和修复 su-memory SDK 的安装问题。
"""

import sys
import os
import shutil
import site
import importlib
from pathlib import Path


def print_header(text: str) -> None:
    """打印标题"""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print('=' * 60)


def print_success(text: str) -> None:
    """打印成功信息"""
    print(f"✅ {text}")


def print_error(text: str) -> None:
    """打印错误信息"""
    print(f"❌ {text}")


def print_warning(text: str) -> None:
    """打印警告信息"""
    print(f"⚠️  {text}")


def print_info(text: str) -> None:
    """打印信息"""
    print(f"ℹ️  {text}")


def check_environment() -> dict:
    """检查 Python 环境"""
    print_header("Python 环境检查")
    
    result = {
        "python": sys.executable,
        "pip": shutil.which("pip"),
        "pip3": shutil.which("pip3"),
        "match": False,
        "issues": []
    }
    
    print(f"\n🐍 Python 路径: {sys.executable}")
    print(f"📦 pip 路径:   {result['pip']}")
    print(f"📦 pip3 路径:  {result['pip3']}")
    
    # 检查 pip 和 python 是否匹配
    python_dir = os.path.dirname(os.path.dirname(sys.executable))
    pip_dir = None
    
    if result['pip']:
        pip_dir = os.path.dirname(os.path.dirname(result['pip']))
    
    if pip_dir and python_dir != pip_dir:
        result['match'] = False
        result['issues'].append(f"pip 和 python 指向不同环境:")
        result['issues'].append(f"  Python: {python_dir}")
        result['issues'].append(f"  pip:    {pip_dir}")
        print_warning("pip 和 python 可能指向不同环境!")
    else:
        result['match'] = True
        print_success("pip 和 python 环境一致")
    
    return result


def check_site_packages() -> dict:
    """检查 site-packages"""
    print_header("site-packages 检查")
    
    result = {
        "paths": site.getsitepackages(),
        "user_site": site.getusersitepackages(),
        "su_memory_found": False,
        "su_memory_path": None
    }
    
    print(f"\n📂 全局 site-packages:")
    for p in result['paths']:
        print(f"   {p}")
    
    print(f"\n👤 用户 site-packages: {result['user_site']}")
    
    return result


def find_su_memory() -> dict:
    """查找 su_memory 模块"""
    print_header("su_memory 模块查找")
    
    result = {
        "found": False,
        "path": None,
        "importable": False,
        "version": None,
        "issues": []
    }
    
    # 方法1: 尝试 import
    try:
        import su_memory
        result['importable'] = True
        result['found'] = True
        result['path'] = su_memory.__file__
        if hasattr(su_memory, '__version__'):
            result['version'] = su_memory.__version__
        print_success(f"su_memory 已安装")
        print(f"   位置: {su_memory.__file__}")
    except ImportError:
        print_error("su_memory 无法导入 (ModuleNotFoundError)")
        result['issues'].append("模块未安装或安装路径错误")
    
    # 方法2: 在 sys.path 中查找
    if not result['found']:
        print_info("在 sys.path 中搜索...")
        for path in sys.path:
            su_path = os.path.join(path, 'su_memory')
            if os.path.exists(su_path):
                result['found'] = True
                result['path'] = su_path
                print_success(f"找到目录: {su_path}")
                break
            # 也检查 su-memory 相关目录
            for item in os.listdir(path) if os.path.exists(path) else []:
                if 'su_memory' in item or 'su-memory' in item:
                    print_info(f"发现相关目录: {os.path.join(path, item)}")
    
    # 方法3: 使用 importlib.util
    if not result['found']:
        print_info("使用 importlib 搜索...")
        spec = importlib.util.find_spec("su_memory")
        if spec:
            result['found'] = True
            result['path'] = spec.origin
            print_success(f"通过 importlib 找到: {spec.origin}")
    
    return result


def check_pip_install() -> dict:
    """检查 pip 安装信息"""
    print_header("pip 安装信息")
    
    result = {
        "installed": False,
        "location": None,
        "version": None,
        "issues": []
    }
    
    import subprocess
    try:
        proc = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', 'su-memory'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if proc.returncode == 0:
            result['installed'] = True
            for line in proc.stdout.split('\n'):
                if line.startswith('Location:'):
                    result['location'] = line.split(':', 1)[1].strip()
                if line.startswith('Version:'):
                    result['version'] = line.split(':', 1)[1].strip()
            
            print_success("su-memory 已通过 pip 安装")
            print(f"   位置: {result['location']}")
            print(f"   版本: {result['version']}")
        else:
            print_error("su-memory 未通过 pip 安装")
    except Exception as e:
        print_error(f"无法获取 pip 信息: {e}")
        result['issues'].append(str(e))
    
    return result


def diagnose() -> dict:
    """执行完整诊断"""
    print("\n" + "🔍" * 30)
    print("su-memory SDK 安装诊断工具")
    print("🔍" * 30)
    
    report = {
        "environment": check_environment(),
        "site_packages": check_site_packages(),
        "su_memory": find_su_memory(),
        "pip_install": check_pip_install(),
        "recommendations": []
    }
    
    # 生成建议
    print_header("诊断结果与修复建议")
    
    issues_found = []
    
    # 问题1: pip 和 python 不匹配
    if not report['environment']['match']:
        issues_found.append("环境不匹配")
        report['recommendations'].extend([
            "⚠️  pip 和 python 指向不同环境",
            "",
            "修复方法 (选择一种):",
            "",
            "  方法1: 使用 python -m pip 安装",
            f"    {sys.executable} -m pip install su-memory",
            "",
            "  方法2: 修复 pip 软链接",
            "    which pip  # 查看 pip 位置",
            "    # 确保 pip 和 python 在同一环境",
            "",
            "  方法3: 使用虚拟环境",
            "    python -m venv myenv",
            "    source myenv/bin/activate",
            "    pip install su-memory",
        ])
    
    # 问题2: 模块找不到但 pip 显示已安装
    if report['pip_install']['installed'] and not report['su_memory']['importable']:
        issues_found.append("模块路径问题")
        report['recommendations'].extend([
            "⚠️  su-memory 已安装但无法导入",
            "",
            "原因: pip 安装到 A 环境，但 python 从 B 环境运行",
            "",
            "修复方法:",
            f"  {sys.executable} -m pip install su-memory",
            "",
            "或手动复制到正确位置:",
            f"  cp -r {report['pip_install']['location']}/su_memory <site-packages>/",
        ])
    
    # 问题3: 完全未安装
    if not report['pip_install']['installed']:
        issues_found.append("未安装")
        report['recommendations'].extend([
            "ℹ️  su-memory 未安装",
            "",
            "安装方法 (推荐使用 python -m pip):",
            "",
            "  标准安装:",
            f"    {sys.executable} -m pip install su-memory",
            "",
            "  从 GitHub 安装:",
            f"    {sys.executable} -m pip install git+https://github.com/su-memory/su-memory-sdk.git",
            "",
            "  从源码安装:",
            "    git clone https://github.com/su-memory/su-memory-sdk.git",
            "    cd su-memory-sdk",
            f"    {sys.executable} -m pip install .",
        ])
    
    # 问题4: 成功安装
    if report['su_memory']['importable']:
        issues_found.append("无")
        report['recommendations'].extend([
            "✅ su-memory SDK 安装正常",
            "",
            "验证安装:",
            "  python -c 'from su_memory import SuMemoryLitePro; print(\"OK\")'",
        ])
    
    # 打印建议
    for rec in report['recommendations']:
        print(rec)
    
    # 总结
    print_header("诊断总结")
    if "无" in issues_found and len(issues_found) == 1:
        print_success("安装正常，无需修复")
        return 0
    else:
        print_warning(f"发现问题: {', '.join(issues_found)}")
        print_info("请根据上述建议进行修复")
        return 1


def main():
    """主函数"""
    exit_code = diagnose()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
