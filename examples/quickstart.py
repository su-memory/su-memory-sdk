#!/usr/bin/env python3
"""su-memory 快速入门示例"""

from su_memory import SuMemory


def main():
    # 1. 一行代码初始化
    client = SuMemory()

    # 2. 添加记忆
    client.add("这个项目的ROI增长了25%，预计明年收益翻倍")
    client.add("市场波动加大，需要优化风险控制策略")
    client.add("团队规模扩大到了50人，需要重新设计组织架构")
    client.add("技术债务积累，需要架构重构")
    client.add("客户满意度提升，NPS得分增长15点")

    # 3. 建立因果关联
    client.link("mem_1", "mem_2")
    client.link("mem_2", "mem_3")
    client.link("mem_3", "mem_4")

    # 4. 语义检索
    results = client.query("投资回报相关")
    print(f"\n检索到 {len(results)} 条记忆:")
    for r in results:
        print(f"  [{r.memory_id}] {r.content[:30]}... (score={r.score})")
        print(f"    八卦:{r.encoding.bagua} 五行:{r.encoding.wuxing} 能量:{r.encoding.energy}")

    # 5. 统计
    stats = client.get_stats()
    print(f"\n记忆统计: {stats['total_memories']} 条")
    print(f"八卦分布: {stats['bagua_distribution']}")


if __name__ == "__main__":
    main()