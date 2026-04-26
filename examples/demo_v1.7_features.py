"""
su-memory SDK v1.7.0 新功能示例

演示插件系统、本地存储、CLI工具等新功能
"""

import sys
sys.path.insert(0, "src")

def demo_plugin_system():
    """演示插件系统"""
    print("\n" + "="*60)
    print("1. 插件系统演示")
    print("="*60)
    
    from su_memory._sys._plugin_registry import PluginRegistry
    from su_memory._sys._plugin_interface import PluginMetadata
    from su_memory.plugins.embedding_plugin import TextEmbeddingPlugin
    from su_memory.plugins.rerank_plugin import RerankPlugin
    from su_memory.plugins.monitor_plugin import MonitorPlugin
    
    # 获取注册表
    registry = PluginRegistry()
    
    # 注册插件
    plugins = [
        ("TextEmbeddingPlugin", TextEmbeddingPlugin()),
        ("RerankPlugin", RerankPlugin()),
        ("MonitorPlugin", MonitorPlugin()),
    ]
    
    for name, plugin in plugins:
        try:
            registry.register(plugin)
            print(f"  ✓ 注册插件: {name}")
        except Exception as e:
            print(f"  ✗ 注册失败: {name} - {e}")
    
    # 列出插件
    print(f"\n  已注册插件: {registry.list_plugins()}")
    
    # 获取性能统计
    stats = registry.get_performance_stats()
    print(f"  注册统计: {stats.get('register_count', 0)} 次注册")


def demo_storage():
    """演示本地存储"""
    print("\n" + "="*60)
    print("2. 本地存储演示")
    print("="*60)
    
    import tempfile
    import os
    from su_memory.storage.sqlite_backend import SQLiteBackend, MemoryItem
    from su_memory.storage.backup_manager import BackupManager
    from su_memory.storage.exporter import DataExporter
    import time
    
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "demo.db")
    backup_dir = os.path.join(temp_dir, "backups")
    
    try:
        # 创建后端
        backend = SQLiteBackend(db_path)
        print(f"  ✓ 创建数据库: {db_path}")
        
        # 添加记忆
        for i in range(5):
            memory = MemoryItem(
                id=f"mem_{i}",
                content=f"这是第{i}条记忆内容",
                metadata={"index": i, "category": "demo"},
                embedding=None,
                timestamp=time.time()
            )
            backend.add_memory(memory)
        print(f"  ✓ 添加5条记忆")
        
        # 查询
        results = backend.query("记忆", top_k=3)
        print(f"  ✓ 查询结果: {len(results)}条")
        
        # 统计
        stats = backend.get_stats()
        print(f"  ✓ 统计: {stats['count']}条记忆")
        
        # 备份
        manager = BackupManager(db_path, backup_dir)
        backup_path = manager.backup()
        print(f"  ✓ 备份: {os.path.basename(backup_path)}")
        
        # 导出JSON
        json_path = os.path.join(temp_dir, "export.json")
        exporter = DataExporter(db_path)
        exporter.to_json(json_path)
        print(f"  ✓ 导出JSON: {os.path.basename(json_path)}")
        
        backend.close()
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def demo_incremental_learning():
    """演示增量学习"""
    print("\n" + "="*60)
    print("3. 增量学习演示")
    print("="*60)
    
    from su_memory import (
        IncrementalLearningManager,
        FeedbackLoop,
        FeedbackType,
        create_learning_manager
    )
    
    # 创建管理器
    manager = create_learning_manager()
    print("  ✓ 创建增量学习管理器")
    
    # 处理反馈
    manager.process_feedback(
        FeedbackType.POSITIVE,
        {"query": "测试查询"},
        {"memory_key": "test_mem"}
    )
    print("  ✓ 处理正面反馈")
    
    manager.process_feedback(
        FeedbackType.NEGATIVE,
        {"query": "测试查询2"},
        {"memory_key": "test_mem2"}
    )
    print("  ✓ 处理负面反馈")
    
    # 获取状态
    status = manager.get_status()
    print(f"  ✓ 反馈统计: {status['feedback']['total']}条")
    print(f"  ✓ 情感趋势: {status['sentiment_trend']}")
    
    # 更新
    result = manager.update()
    print(f"  ✓ 更新结果: {result.updated_count}个参数更新")


def demo_local_models():
    """演示本地模型"""
    print("\n" + "="*60)
    print("4. 本地预测模型演示")
    print("="*60)
    
    from su_memory import LocalModelManager, create_linear_model, create_tfidf_ranker
    
    # 创建管理器
    manager = LocalModelManager()
    print("  ✓ 创建本地模型管理器")
    
    # 注册模型
    model1 = create_linear_model(input_dim=10)
    manager.register_model("test_model", model1)
    print("  ✓ 注册线性模型")
    
    model2 = create_tfidf_ranker()
    manager.register_model("ranker", model2)
    print("  ✓ 注册TF-IDF排序器")
    
    # 列出模型
    models = manager.list_models()
    print(f"  ✓ 已注册模型: {models}")


def demo_adaptive_engine():
    """演示自适应引擎"""
    print("\n" + "="*60)
    print("5. 自适应引擎演示")
    print("="*60)
    
    from su_memory._sys._adaptive_engine import AdaptiveEngine
    
    # 创建引擎
    engine = AdaptiveEngine()
    print("  ✓ 创建自适应引擎")
    
    # 获取建议
    suggestions = engine.get_suggestions()
    print(f"  ✓ 获取建议: {len(suggestions)}条")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("su-memory SDK v1.7.0 新功能演示")
    print("="*60)
    
    demos = [
        ("插件系统", demo_plugin_system),
        ("本地存储", demo_storage),
        ("增量学习", demo_incremental_learning),
        ("本地模型", demo_local_models),
        ("自适应引擎", demo_adaptive_engine),
    ]
    
    for name, func in demos:
        try:
            func()
        except Exception as e:
            print(f"\n  ✗ {name}演示失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("演示完成!")
    print("="*60)


if __name__ == "__main__":
    main()
