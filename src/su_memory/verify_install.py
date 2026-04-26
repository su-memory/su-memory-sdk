#!/usr/bin/env python3
"""
su-memory SDK 安装验证脚本

快速验证 su-memory SDK 是否正确安装。
"""

import sys


def verify_installation():
    """验证安装"""
    print("=" * 60)
    print("su-memory SDK 安装验证")
    print("=" * 60)
    
    all_passed = True
    
    # 测试1: 基础导入
    print("\n[1/5] 测试模块导入...")
    try:
        from su_memory import SuMemoryLitePro
        print("  ✅ SuMemoryLitePro 导入成功")
    except ImportError as e:
        print(f"  ❌ 导入失败: {e}")
        all_passed = False
        return False
    
    # 测试2: 实例化
    print("\n[2/5] 测试实例化...")
    try:
        # 禁用向量服务以加快测试
        import os
        os.environ['OLLAMA_HOST'] = 'http://localhost:11434'
        
        pro = SuMemoryLitePro(enable_vector=False)
        print("  ✅ 实例化成功")
    except Exception as e:
        # 如果是因为没有向量服务，这是正常的
        if "Connection" in str(e) or "HTTP" in str(e):
            print("  ⚠️  实例化成功(向量服务未连接)")
        else:
            print(f"  ❌ 实例化失败: {e}")
            all_passed = False
    
    # 测试3: 添加记忆
    print("\n[3/5] 测试添加记忆...")
    try:
        pro.add("测试记忆: 今天天气很好")
        print("  ✅ 添加记忆成功")
    except Exception as e:
        print(f"  ❌ 添加记忆失败: {e}")
        all_passed = False
    
    # 测试4: 查询记忆
    print("\n[4/5] 测试查询记忆...")
    try:
        results = pro.query("天气")
        print(f"  ✅ 查询成功 (返回 {len(results)} 条结果)")
    except Exception as e:
        print(f"  ❌ 查询失败: {e}")
        all_passed = False
    
    # 测试5: 多跳推理
    print("\n[5/5] 测试多跳推理...")
    try:
        pro.add("测试: 天气好心情就好")
        pro.add("测试: 心情好效率高")
        pro.link_memories(0, 1)  # 建立关联
        
        results = pro.query_multihop("天气", max_hops=2)
        print(f"  ✅ 多跳推理成功 (返回 {len(results)} 条结果)")
    except Exception as e:
        print(f"  ❌ 多跳推理失败: {e}")
        all_passed = False
    
    # 总结
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ su-memory SDK 安装验证通过!")
        print("=" * 60)
        return 0
    else:
        print("⚠️  部分测试未通过，请检查安装")
        print("=" * 60)
        return 1


def quick_check():
    """快速检查"""
    print("快速检查 su-memory 安装状态...\n")
    
    try:
        from su_memory import SuMemoryLitePro
        print("✅ su_memory 模块可以导入")
        
        # 尝试实例化
        import os
        os.environ.setdefault('OLLAMA_HOST', 'http://localhost:11434')
        pro = SuMemoryLitePro(enable_vector=False)
        print("✅ SuMemoryLitePro 可以实例化")
        
        pro.add("测试")
        print("✅ add() 方法正常工作")
        
        print("\n✅ su-memory SDK 安装正常!")
        return 0
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("\n请运行诊断工具:")
        print("  python -c 'from su_memory.diagnostics import main; main()'")
        return 1
    except Exception as e:
        print(f"❌ 错误: {e}")
        return 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        sys.exit(quick_check())
    else:
        sys.exit(verify_installation())
