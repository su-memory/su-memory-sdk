"""Task 15 验证：time_code时序全集成"""
import sys
sys.path.insert(0, "/Users/mac/.openclaw/workspace/su-memory")
sys.path.insert(0, "/Users/mac/.openclaw/workspace/su-memory/src")

from datetime import date

def test_monthly_energy_type_state():
    from su_core._sys.chrono import TemporalSystem
    ts = TemporalSystem()
    
    # 春季（3月）
    states = ts.get_monthly_energy_type_state(3, 15)
    print(f"3月strength_state: {states}")
    assert states["wood"] == "strong", f"春季wood应strong，得到 {states['wood']}"
    assert states["fire"] == "restrained", f"春季fire应restrained，得到 {states['fire']}"
    
    # 夏季（6月）
    states2 = ts.get_monthly_energy_type_state(6, 15)
    print(f"6月strength_state: {states2}")
    assert states2["fire"] == "strong", f"夏季fire应strong，得到 {states2['fire']}"
    
    print("PASSED: monthly energy_type state")

def test_cycle_position():
    from su_core._sys.chrono import TemporalSystem
    ts = TemporalSystem()
    
    pos = ts.get_cycle_position(date.today())
    print(f"今日cycle位置: {pos}")
    assert 0 <= pos <= 59, f"cycle位置应在0-59，得到 {pos}"
    
    # 循环距离测试
    d = ts.cycle_distance(0, 59)
    print(f"0→59循环距离: {d}")
    assert d < 0.1, f"0和59应该很近，得到 {d}"
    
    print("PASSED: cycle position")

def test_temporal_similarity():
    from su_core._sys.chrono import TemporalSystem
    ts = TemporalSystem()
    
    sim = ts.temporal_similarity(date(2026, 4, 22), date(2026, 4, 23))
    print(f"相邻日相似度: {sim}")
    assert sim > 0.8, f"相邻日应高度相似，得到 {sim}"
    
    sim2 = ts.temporal_similarity(date(2026, 4, 22), date(2026, 10, 22))
    print(f"半年距相似度: {sim2}")
    assert sim2 < sim, f"半年距应低于相邻日"
    
    print("PASSED: temporal similarity")

def test_time_decay():
    from su_core._sys.chrono import TemporalSystem
    from datetime import date
    import time
    ts = TemporalSystem()
    
    now = int(time.time())
    
    # 找到当前月令下"strong"的energy_type用于测试
    today = date.today()
    monthly_states = ts.get_monthly_energy_type_state(today.month, today.day)
    strong_et = [et for et, st in monthly_states.items() if st == "strong"][0]
    print(f"当前strong能量类型: {strong_et}")
    
    # 1天前（strong能量类型，multiplier=1.3, base=1.0 → capped at 1.0）
    decay1 = ts.calculate_time_decay(now - 86400, strong_et)
    print(f"1天前衰减({strong_et}): {decay1}")
    assert decay1 >= 0.9, f"1天前strong能量衰减应>=0.9，得到 {decay1}"
    
    # 30天前
    decay30 = ts.calculate_time_decay(now - 86400 * 30, strong_et)
    print(f"30天前衰减({strong_et}): {decay30}")
    assert decay30 < decay1
    
    # 100天前
    decay100 = ts.calculate_time_decay(now - 86400 * 100, strong_et)
    print(f"100天前衰减({strong_et}): {decay100}")
    assert decay100 < decay30
    
    # 验证能量类型状态影响衰减速度
    restrained_et = [et for et, st in monthly_states.items() if st == "restrained"][0]
    decay_restrained = ts.calculate_time_decay(now - 86400, restrained_et)
    print(f"1天前衰减({restrained_et}/restrained): {decay_restrained}")
    assert decay_restrained < decay1, "restrained能量衰减应比strong能量更快"
    
    print("PASSED: time decay")

def test_upgraded_priority():
    from su_core._sys.chrono import TemporalSystem
    import time
    ts = TemporalSystem()
    
    time_code = ts.get_current_time_code()
    print(f"当前time_code: {time_code.time_code}, 季节: {time_code.season}")
    
    # 带时间戳的优先级
    priority = ts.calculate_priority(
        base_priority=5,
        time_code_info=time_code,
        memory_energy_type="wood",
        memory_timestamp=int(time.time()) - 86400 * 10
    )
    print(f"优先级: base={priority.base_priority}, season={priority.season_boost}, final={priority.final_priority}")
    
    print("PASSED: upgraded priority")

def test_season_complete():
    from su_core._sys.chrono import TemporalSystem
    ts = TemporalSystem()
    
    # 确保每个月都有确定的季节
    for month in range(1, 13):
        season = ts._get_season(month, 15)
        print(f"  月{month}: {season}")
        assert season != "四季" or month in [4, 7, 10, 1], f"月{month}不应为四季末"
    
    print("PASSED: season complete")

if __name__ == "__main__":
    test_monthly_energy_type_state()
    print("---")
    test_cycle_position()
    print("---")
    test_temporal_similarity()
    print("---")
    test_time_decay()
    print("---")
    test_upgraded_priority()
    print("---")
    test_season_complete()
    print("\n✅ All Task 15 tests passed!")
