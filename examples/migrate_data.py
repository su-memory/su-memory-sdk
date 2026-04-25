#!/usr/bin/env python3
"""
su-memory 数据迁移示例脚本

使用方法:
    python examples/migrate_data.py --source json --path data/memories.json
    python examples/migrate_data.py --source csv --path data/notes.csv
    python examples/migrate_data.py --source sqlite --path data/app.db --table memories
    python examples/migrate_data.py --source obsidian --path /path/to/vault
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from su_memory._sys.migrator import (
    MemoryMigrator,
    DataSourceType,
    migrate_json,
    migrate_csv,
    migrate_sqlite,
    migrate_obsidian,
    create_migration_report_file
)
from su_memory.sdk import SuMemoryLitePro


def progress_callback(current: int, total: int, message: str):
    """进度回调函数"""
    percentage = current / total * 100 if total > 0 else 0
    bar_length = 30
    filled = int(bar_length * current / total) if total > 0 else 0
    bar = '█' * filled + '░' * (bar_length - filled)
    
    print(f"\r[{bar}] {percentage:5.1f}% ({current}/{total}) {message}", end='', flush=True)
    
    if current >= total:
        print()


def create_sample_data(output_dir: str):
    """创建示例数据用于测试"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. JSON示例数据
    json_data = {
        "memories": [
            {"id": "mem001", "content": "今天是周一，开始新的一周工作", "timestamp": 1714000000, "tags": ["工作", "日常"]},
            {"id": "mem002", "content": "学习Python编程提升技能", "timestamp": 1714080000, "tags": ["学习", "编程"]},
            {"id": "mem003", "content": "完成项目报告提交", "timestamp": 1714160000, "tags": ["工作", "项目"]},
            {"id": "mem004", "content": "健身锻炼保持健康", "timestamp": 1714240000, "tags": ["健康", "运动"]},
            {"id": "mem005", "content": "阅读技术书籍《Python进阶》", "timestamp": 1714320000, "tags": ["学习", "阅读"]},
        ]
    }
    
    with open(f"{output_dir}/sample_memories.json", 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    # 2. CSV示例数据
    csv_content = """id,content,timestamp,tags
csv001,"团队会议讨论项目进度",1714400000,"工作,会议"
csv002,"学习机器学习基础知识",1714480000,"学习,AI"
csv003,"完成代码review",1714560000,"工作,代码"
csv004,"准备下周演讲稿",1714640000,"工作,演讲"
csv005,"整理工作笔记",1714720000,"工作,笔记"
"""
    
    with open(f"{output_dir}/sample_notes.csv", 'w', encoding='utf-8') as f:
        f.write(csv_content)
    
    # 3. 创建Obsidian风格示例
    obsidian_content = """---
title: Obsidian笔记示例
tags: [笔记, 知识管理]
created: 2024-04-25
---

# Obsidian笔记示例

这是从Obsidian知识库迁移的示例笔记。

## 主要内容

- 知识管理的重要性
- 双链笔记的构建方法
- 定期回顾整理

## 行动项

1. 每天添加新笔记
2. 每周回顾整理
3. 建立知识体系

---
"""
    
    with open(f"{output_dir}/sample_obsidian_note.md", 'w', encoding='utf-8') as f:
        f.write(obsidian_content)
    
    print(f"示例数据已创建在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='su-memory 数据迁移工具')
    parser.add_argument('--source', '-s', 
                       choices=['json', 'csv', 'sqlite', 'obsidian', 'demo'],
                       default='demo',
                       help='数据源类型')
    parser.add_argument('--path', '-p',
                       help='数据源路径')
    parser.add_argument('--table', '-t',
                       default='memories',
                       help='SQLite表名')
    parser.add_argument('--output', '-o',
                       default='./migration_report.md',
                       help='报告输出路径')
    parser.add_argument('--storage', '-st',
                       default='./migration_test_storage',
                       help='目标存储路径')
    parser.add_argument('--create-demo', '-d',
                       action='store_true',
                       help='创建示例数据')
    
    args = parser.parse_args()
    
    # 创建示例数据
    if args.create_demo or args.source == 'demo':
        demo_dir = './demo_data'
        create_sample_data(demo_dir)
        print()
    
    if args.source == 'demo':
        print("=" * 60)
        print("数据迁移演示模式")
        print("=" * 60)
        print()
        
        # 创建目标客户端
        client = SuMemoryLitePro(storage_path=args.storage)
        
        # 迁移JSON数据
        print("【1/2】迁移JSON数据...")
        report1 = migrate_json(
            f"{demo_dir}/sample_memories.json",
            client,
            progress_callback
        )
        print(f"\nJSON迁移完成: 成功 {report1.success_count}/{report1.total_records}")
        print()
        
        # 迁移CSV数据
        print("【2/2】迁移CSV数据...")
        report2 = migrate_csv(
            f"{demo_dir}/sample_notes.csv",
            client,
            progress_callback
        )
        print(f"\nCSV迁移完成: 成功 {report2.success_count}/{report2.total_records}")
        print()
        
        # 保存合并报告
        print("=" * 60)
        print("迁移完成！生成合并报告...")
        print("=" * 60)
        
        migrator = MemoryMigrator(target_client=client)
        migrator.reports = [report1, report2]
        combined = migrator.get_combined_report()
        
        create_migration_report_file(combined, args.output)
        
        # 验证迁移结果
        print()
        print("【验证】检查迁移后的数据...")
        stats = client.get_stats()
        print(f"  总记忆数: {stats.get('total_memories', 0)}")
        
        # 测试检索
        print()
        print("【验证】测试检索功能...")
        results = client.query('工作', top_k=3)
        print(f"  查询'工作'返回 {len(results)} 条结果")
        for r in results[:2]:
            print(f"    - {r['content'][:50]}...")
        
        print()
        print(f"✅ 迁移报告已保存到: {args.output}")
        
    else:
        if not args.path:
            print("错误: 必须指定 --path 参数")
            return 1
        
        print("=" * 60)
        print(f"数据迁移工具 - {args.source.upper()}")
        print("=" * 60)
        print()
        
        # 创建目标客户端
        client = SuMemoryLitePro(storage_path=args.storage)
        
        # 创建迁移器
        migrator = MemoryMigrator(
            target_client=client,
            progress_callback=progress_callback
        )
        
        # 执行迁移
        if args.source == 'json':
            report = migrator.migrate(DataSourceType.JSON, args.path)
        elif args.source == 'csv':
            report = migrator.migrate(DataSourceType.CSV, args.path)
        elif args.source == 'sqlite':
            report = migrator.migrate(
                DataSourceType.SQLITE, args.path,
                table_name=args.table
            )
        elif args.source == 'obsidian':
            report = migrator.migrate(DataSourceType.OBSIDIAN, args.path)
        else:
            print(f"不支持的数据源类型: {args.source}")
            return 1
        
        # 保存报告
        create_migration_report_file(report, args.output)
        
        print()
        print("=" * 60)
        print("迁移结果汇总")
        print("=" * 60)
        print(f"  总记录数: {report.total_records}")
        print(f"  成功迁移: {report.success_count}")
        print(f"  迁移失败: {report.failed_count}")
        print(f"  跳过记录: {report.skipped_count}")
        print(f"  成功率: {report.success_count/max(report.total_records,1)*100:.1f}%")
        print()
        print(f"报告已保存到: {args.output}")
        
        if report.errors:
            print()
            print("前5个错误:")
            for i, err in enumerate(report.errors[:5], 1):
                print(f"  {i}. {err.error_message}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
