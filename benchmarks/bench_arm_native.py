#!/usr/bin/env python3
"""su-memory v2.5.0 ARM Native 全面性能基准测试"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import platform

print("=" * 70)
print("  su-memory v2.5.0 ARM Native 性能基准")
print(f"  Python: {platform.python_version()} — {platform.machine()}")
print(f"  macOS:  {platform.mac_ver()[0]}")
print("=" * 70)

from su_memory.sdk.lite_pro import SuMemoryLitePro

# ═════════════════════════════════════════════════
# 1. 写入吞吐
# ═════════════════════════════════════════════════
print("\n[1] 写入吞吐基准...")
pro = SuMemoryLitePro(enable_vector=False, enable_graph=False,
    enable_temporal=False, enable_session=False,
    enable_prediction=False, enable_explainability=False)

pro._energy_cache = {}
N = 500
t0 = time.perf_counter()
for i in range(N):
    pro.add(f"bench_{i:04d}: performance test entry number {i}")
elapsed = time.perf_counter() - t0
throughput = N / elapsed
print(f"  500条写入: {elapsed:.2f}s ({throughput:.1f} 条/秒)")

# ═════════════════════════════════════════════════
# 2. 查询延迟
# ═════════════════════════════════════════════════
print("\n[2] 查询延迟基准...")
latencies = []
queries = ["bench", "test", "entry", "performance", "number"] * 100
for q in queries:
    t0 = time.perf_counter()
    pro.query(q, top_k=5)
    latencies.append((time.perf_counter() - t0) * 1000)
latencies.sort()
print(f"  P50: {latencies[250]:.2f}ms")
print(f"  P95: {latencies[475]:.2f}ms")
print(f"  P99: {latencies[495]:.2f}ms")

# ═════════════════════════════════════════════════
# 3. 能量推断延迟
# ═════════════════════════════════════════════════
print("\n[3] 能量推断延迟...")
contents = [
    "Spring renewal and tree growth in the eastern forest",
    "Summer heat passion fire red south burning bright",
    "Central stability earth grounding balance foundation",
    "Autumn metal harvest precision structure west white",
    "Winter water wisdom flow deep north blue ocean",
] * 20
pro._energy_cache = {}
t0 = time.perf_counter()
for c in contents:
    pro._infer_energy(c)
elapsed = time.perf_counter() - t0
print(f"  100次推断: {elapsed:.2f}s ({100/elapsed:.1f} 次/秒)")

# ═════════════════════════════════════════════════
# 4. 能量分析延迟
# ═════════════════════════════════════════════════
print("\n[4] 能量分析延迟...")
t0 = time.perf_counter()
eco = pro.analyze_memory_ecology()
e1 = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
patterns = pro.distill_patterns()
e2 = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
pro.evolution_pipeline()
e3 = (time.perf_counter() - t0) * 1000

print(f"  analyze_memory_ecology: {e1:.2f}ms")
print(f"  distill_patterns:       {e2:.2f}ms")
print(f"  evolution_pipeline:     {e3:.2f}ms")

# ═════════════════════════════════════════════════
# 5. 能量系统操作
# ═════════════════════════════════════════════════
print("\n[5] 能量系统操作...")
from su_memory._sys._energy_bus import EnergyBus, EnergyNode, EnergyLayer
from su_memory._sys._dimension_map import TaijiMapper

t0 = time.perf_counter()
bus = EnergyBus()
bus.create_five_elements_nodes()
for i in range(100):
    bus.propagate_energy("element_wood", delta=0.3, max_hops=2)
e4 = (time.perf_counter() - t0) * 1000
print(f"  EnergyBus 100次传播: {e4:.2f}ms ({100/(e4/1000):.0f} 次/秒)")

t0 = time.perf_counter()
mapper = TaijiMapper()
for i in range(1000):
    mapper.resolve_trigram_to_semantic(i % 8)
e5 = (time.perf_counter() - t0) * 1000
print(f"  三维映射 1000次: {e5:.2f}ms ({(e5/1000):.3f}ms/次)")

# ═════════════════════════════════════════════════
# 6. 自动能量链接
# ═════════════════════════════════════════════════
print("\n[6] 自动能量链接...")
pro2 = SuMemoryLitePro(enable_vector=False, enable_graph=True,
    enable_temporal=False, enable_session=False,
    enable_prediction=False, enable_explainability=False)
pro2._energy_cache = {}

for i in range(100):
    content = f"memory_{i}: " + [
        "spring wood growth green east",
        "summer fire passion red south",
        "earth stability center yellow",
        "autumn metal harvest west white",
        "winter water wisdom north blue",
    ][i % 5]
    pro2.add(content)

t0 = time.perf_counter()
links = pro2.auto_link_by_energy()
e6 = (time.perf_counter() - t0) * 1000
print(f"  100条记忆 auto_link: {e6:.2f}ms ({links} links)")

# ═════════════════════════════════════════════════
# 综合总结
# ═════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  综合对比 (ARM原生 vs Rosetta x86)")
print("=" * 70)
print(f"  写入吞吐:        {throughput:.0f} 条/秒")
print(f"  查询 P99:        {latencies[495]:.2f}ms")
print(f"  能量推断:        {100/elapsed:.0f} 次/秒")
print(f"  能量传播:        {100/(e4/1000):.0f} 次/秒")
print(f"  三维映射:        {(e5/1000):.3f}ms/次")
print(f"  auto_link(100):  {e6:.0f}ms")
print(f"  evolution_pipe:  {e3:.0f}ms")

pro.clear()
pro2.clear()
