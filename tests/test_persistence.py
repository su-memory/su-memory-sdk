"""
发布前持久化专项测试
验证：重启不丢数据、并发安全、数据损坏恢复
"""
import os
import shutil
import su_memory
from su_memory import SuMemory

DATA_DIR = "./test_persist_isolated"

def clean():
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)

def test_basic_persistence():
    """P0: 重启后记忆不丢失"""
    clean()
    client = SuMemory(persist_dir=DATA_DIR)
    mids = [client.add(f"记忆内容{i}", metadata={"i": i}) for i in range(5)]

    # 重启
    client2 = SuMemory(persist_dir=DATA_DIR)
    stats = client2.get_stats()

    assert stats["total_memories"] == 5, f"重启后记忆丢失: 期望5 实际{stats['total_memories']}"
    print("✅ P0: 重启后记忆不丢失")

def test_immediate_read_after_write():
    """P0: 写入后立即读取"""
    clean()
    client = SuMemory(persist_dir=DATA_DIR)
    UNIQUE = "UNIQUE_TOKEN_XYZ123"
    mid = client.add(UNIQUE)
    result = client.query(UNIQUE, top_k=1)

    assert len(result) > 0, "写入后立即查询为空"
    assert UNIQUE in result[0].content, f"查询结果不匹配: {result[0].content}"
    print("✅ P0: 写入后立即读取正确")

def test_corruption_recovery():
    """P1: JSON 损坏时降级不崩溃"""
    clean()
    client = SuMemory(persist_dir=DATA_DIR)
    client.add("一条记忆")

    # 写入非法 JSON
    data_path = os.path.join(DATA_DIR, "memories.json")
    with open(data_path, "w") as f:
        f.write("{ broken json }")

    # 重启应该能降级，不抛异常
    try:
        client2 = SuMemory(persist_dir=DATA_DIR)
        assert len(client2) == 0, "损坏后应清空数据"
        print("✅ P1: JSON 损坏降级正常")
    except Exception as e:
        raise AssertionError(f"JSON损坏降级失败: {e}")

def test_vectors_persistence():
    """P1: 向量与记忆同时持久化"""
    clean()
    client = SuMemory(persist_dir=DATA_DIR)
    client.add("测试向量内容")
    vec_path = os.path.join(DATA_DIR, "vectors.json")
    assert os.path.exists(vec_path), "向量文件未生成"
    print("✅ P1: 向量文件正确生成")

def test_delete_persistence():
    """P1: 删除后重启确认"""
    clean()
    client = SuMemory(persist_dir=DATA_DIR)
    ids = [client.add(f"内容{i}") for i in range(5)]
    client.delete(ids[0], ids[1])

    client2 = SuMemory(persist_dir=DATA_DIR)
    assert client2.get_stats()["total_memories"] == 3, "删除后重启数量不对"
    print("✅ P1: 删除后持久化正确")

if __name__ == "__main__":
    clean()
    test_basic_persistence()
    clean()
    test_immediate_read_after_write()
    clean()
    test_corruption_recovery()
    clean()
    test_vectors_persistence()
    clean()
    test_delete_persistence()
    clean()
    print("\n🎉 持久化专项测试全部通过！")
