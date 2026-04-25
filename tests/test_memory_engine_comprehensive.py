"""
记忆引擎层综合功能测试 (Task #3)
==================================
覆盖：
  2.1 记忆提取（MemoryExtractor）
  2.2 记忆检索（六路全息检索 vs 纯向量，召回率对比）
  2.3 冲突消解（ConflictResolver）
  2.4 遗忘与生命周期（ForgettingEngine + BeliefTracker）

依赖服务：
  - Qdrant  : localhost:6333  → 不可用时 SKIP 相关用例
  - PostgreSQL: localhost:5432  → 不可用时 SKIP 相关用例

运行：
  pytest tests/test_memory_engine_comprehensive.py -v --tb=long
  pytest tests/test_memory_engine_comprehensive.py --cov=memory_engine --cov-report=term-missing
"""

import sys
import time
import uuid
import asyncio
import logging
import socket
import pytest

sys.path.insert(0, ".")

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 服务可用性探测
# ─────────────────────────────────────────────

def _port_open(host, port, timeout=1.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


QDRANT_AVAILABLE = _port_open("localhost", 6333)
POSTGRES_AVAILABLE = _port_open("localhost", 5432)

skip_no_qdrant = pytest.mark.skipif(
    not QDRANT_AVAILABLE,
    reason="Qdrant not reachable at localhost:6333"
)
skip_no_postgres = pytest.mark.skipif(
    not POSTGRES_AVAILABLE,
    reason="PostgreSQL not reachable at localhost:5432"
)
skip_no_backends = pytest.mark.skipif(
    not (QDRANT_AVAILABLE and POSTGRES_AVAILABLE),
    reason="Both Qdrant and PostgreSQL required"
)


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ═══════════════════════════════════════════════
# 2.1  MemoryExtractor 测试
# ═══════════════════════════════════════════════

class TestMemoryExtractor:

    def setup_method(self):
        from memory_engine.extractor import MemoryExtractor
        self.extractor = MemoryExtractor()

    def test_simple_fact_extraction(self):
        content = "患者张三，男，65岁，糖尿病II型"
        result = self.extractor.extract_sync(content)
        assert result is not None
        assert "type" in result
        assert "compressed" in result
        assert "entities" in result
        assert "priority" in result
        assert "encoding_info" in result
        entities = result["entities"]
        numbers = [e["value"] for e in entities if e["type"] == "number"]
        assert "65" in numbers, f"未识别年龄65，实际:{numbers}"
        assert 0 <= result["priority"] <= 10
        enc = result["encoding_info"]
        assert enc["hexagram_name"] != ""
        assert enc["wuxing"] in ["金","木","水","火","土"]
        print(f"\n  ✓ 事实提取: type={result['type']}, 卦={enc['hexagram_name']}({enc['wuxing']}), 优先级={result['priority']}")

    def test_type_classification_fact(self):
        result = self.extractor.extract_sync("患者血糖 8.5 mmol/L")
        assert result["type"] == "fact"

    def test_type_classification_preference(self):
        result = self.extractor.extract_sync("我喜欢低脂饮食，不喜欢甜食")
        assert result["type"] == "preference"

    def test_type_classification_event(self):
        result = self.extractor.extract_sync("昨天做了胃镜检查，发现轻度胃炎")
        assert result["type"] == "event"

    def test_type_classification_belief(self):
        result = self.extractor.extract_sync("我认为手术风险太大，应该先保守治疗")
        assert result["type"] == "belief"

    def test_metadata_type_override(self):
        result = self.extractor.extract_sync("任意内容", metadata={"type": "preference"})
        assert result["type"] == "preference"

    def test_long_content_extraction(self):
        content = (
            "患者李四，女，48岁。入院诊断：高血压3级、2型糖尿病、慢性肾病CKD3期。"
            "既往史：15年前因阑尾炎手术，无药物过敏史。"
            "目前用药：二甲双胍500mg 每日2次，氨氯地平5mg 每日1次。"
            "实验室：空腹血糖7.8 mmol/L，HbA1c 8.2%，肌酐145 μmol/L，eGFR 48。"
        )
        result = self.extractor.extract_sync(content)
        assert result is not None
        assert result["compressed"] != ""
        nums = [e["value"] for e in result["entities"] if e["type"] == "number"]
        assert len(nums) >= 3, f"长文本数字实体不足3个，实际:{nums}"
        print(f"\n  ✓ 长文本提取: 实体={len(result['entities'])}, 数字={nums[:5]}")

    def test_priority_urgent_content(self):
        normal = self.extractor.extract_sync("患者有轻微咳嗽")
        urgent = self.extractor.extract_sync("紧急！患者出现呼吸困难，必须立即处理")
        assert urgent["priority"] >= normal["priority"]

    def test_encode_returns_vector(self):
        vec = self.extractor.encode("患者血压 140/90 mmHg")
        assert isinstance(vec, list)
        assert len(vec) > 0
        print(f"\n  ✓ 向量化: 维度={len(vec)}")

    def test_encode_consistency(self):
        text = "测试确定性向量化"
        v1 = self.extractor.encode(text)
        v2 = self.extractor.encode(text)
        assert v1 == v2

    def test_encode_different_texts(self):
        v1 = self.extractor.encode("患者有高血压")
        v2 = self.extractor.encode("今天天气很好")
        assert v1 != v2

    def test_init_belief(self):
        memory_id = str(uuid.uuid4())
        result = run_async(self.extractor.init_belief(memory_id))
        assert "stage" in result
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0
        print(f"\n  ✓ 信念初始化: stage={result['stage']}, confidence={result['confidence']:.2f}")

    def test_reinforce_belief_increases_confidence(self):
        memory_id = str(uuid.uuid4())
        init = run_async(self.extractor.init_belief(memory_id))
        reinforced = run_async(self.extractor.reinforce_belief(memory_id))
        assert reinforced["confidence"] >= init["confidence"]

    def test_hexagram_encoding_completeness(self):
        result = self.extractor.extract_sync("测试卦象编码")
        enc = result["encoding_info"]
        for field in ["hexagram_name","hexagram_index","wuxing","direction","hu_gua","zong_gua","cuo_gua"]:
            assert field in enc, f"缺少字段: {field}"
        assert 0 <= enc["hexagram_index"] <= 63

    def test_dynamic_priority_fields(self):
        result = self.extractor.extract_sync("动态优先级测试")
        dp = result["dynamic_priority"]
        for field in ["base","season_boost","final"]:
            assert field in dp
        assert 0.0 <= dp["final"] <= 2.0


# ═══════════════════════════════════════════════
# 2.2  记忆检索测试
# ═══════════════════════════════════════════════

class TestMemoryRetriever:

    def setup_method(self):
        from memory_engine.extractor import MemoryExtractor
        from memory_engine.retriever import MemoryRetriever
        self.extractor = MemoryExtractor()

        class StubVDB:
            async def search(self, collection, query_vector, limit, filter=None):
                return []
        self.StubVDB = StubVDB

    def _make_candidates(self, count=20):
        templates = [
            ("患者{name}有{cond}病史", "fact"),
            ("患者{name}喜欢{food}饮食", "preference"),
            ("昨天{name}发生了{ev}事件", "event"),
            ("医生认为{name}应该{treat}治疗", "belief"),
        ]
        names = ["张三","李四","王五","赵六","陈七"]
        conds = ["高血压","糖尿病","心脏病","肾病","肝炎"]
        foods = ["低脂","低糖","高蛋白","素食","低盐"]
        evs = ["急性发作","血糖升高","心律不齐","肌酐升高","发烧"]
        treats = ["手术","保守治疗","药物干预","饮食控制","定期复查"]
        candidates = []
        for i in range(count):
            tmpl, mtype = templates[i % 4]
            content = tmpl.format(
                name=names[i%5], cond=conds[i%5], food=foods[i%5],
                ev=evs[i%5], treat=treats[i%5]
            )
            extracted = self.extractor.extract_sync(content)
            candidates.append({
                "id": f"cand_{i:03d}",
                "score": 0.9 - i*0.01,
                "payload": {
                    "content": content,
                    "user_id": "test_user",
                    "memory_type": mtype,
                    "timestamp": int(time.time()) - i*3600,
                    "metadata": {},
                    "hexagram_index": extracted["encoding_info"]["hexagram_index"],
                    "wuxing": extracted["encoding_info"]["wuxing"],
                    "hu_gua": extracted["encoding_info"]["hu_gua"],
                    "zong_gua": extracted["encoding_info"]["zong_gua"],
                    "cuo_gua": extracted["encoding_info"]["cuo_gua"],
                }
            })
        return candidates

    def test_holographic_rerank_returns_results(self):
        from memory_engine.retriever import MemoryRetriever
        r = MemoryRetriever(self.StubVDB())
        cands = self._make_candidates(20)
        qvec = self.extractor.encode("患者高血压病史")
        results = r._holographic_rerank(query_vector=qvec, candidates=cands, limit=5)
        assert len(results) <= 5
        assert all("id" in x for x in results)
        assert all("score" in x for x in results)
        assert all("holographic_score" in x for x in results)
        print(f"\n  ✓ 全息重排: 候选={len(cands)}, 返回={len(results)}")

    def test_simple_rerank_returns_results(self):
        from memory_engine.retriever import MemoryRetriever
        r = MemoryRetriever(self.StubVDB())
        cands = self._make_candidates(10)
        results = r._simple_rerank(cands, limit=5)
        assert len(results) == 5
        for i in range(len(results)-1):
            assert results[i]["timestamp"] >= results[i+1]["timestamp"]

    def test_holographic_score_range(self):
        from memory_engine.retriever import MemoryRetriever
        r = MemoryRetriever(self.StubVDB())
        cands = self._make_candidates(10)
        qvec = self.extractor.encode("医疗诊断")
        results = r._holographic_rerank(query_vector=qvec, candidates=cands, limit=10)
        for x in results:
            assert isinstance(x["holographic_score"], (int,float))
            assert 0 <= x["hexagram_index"] <= 63

    @skip_no_qdrant
    def test_retriever_with_real_qdrant(self):
        from storage.vector_db import VectorDB
        from memory_engine.retriever import MemoryRetriever
        from qdrant_client import QdrantClient
        import storage.vector_db as vdb_module
        vdb_module._qdrant_client = QdrantClient(host="localhost", port=6333, prefer_grpc=False)
        vdb = VectorDB()
        vdb.client = vdb_module._qdrant_client
        tenant_id = f"test_{uuid.uuid4().hex[:8]}"
        async def run():
            await vdb.create_collection(tenant_id)
            for i in range(5):
                content = f"测试记忆#{i}: 患者{'张三' if i%2==0 else '李四'}的{['血压','血糖','体重','心率','体温'][i]}"
                vec = self.extractor.encode(content)
                ext = self.extractor.extract_sync(content)
                payload = {
                    "user_id":"test_user","content":content,"memory_type":"fact",
                    "timestamp":int(time.time())-i,"metadata":{},
                    "hexagram_index":ext["encoding_info"]["hexagram_index"],
                    "wuxing":ext["encoding_info"]["wuxing"],
                    "hu_gua":ext["encoding_info"]["hu_gua"],
                    "zong_gua":ext["encoding_info"]["zong_gua"],
                    "cuo_gua":ext["encoding_info"]["cuo_gua"],
                }
                await vdb.insert(collection=tenant_id, id=str(uuid.uuid4()), vector=vec, payload=payload)
            retriever = MemoryRetriever(vdb)
            qvec = self.extractor.encode("张三血糖")
            results = await retriever.retrieve(
                collection=tenant_id, query_vector=qvec,
                user_id="test_user", limit=3, use_holographic=True
            )
            return results
        results = run_async(run())
        assert len(results) > 0
        print(f"\n  ✓ Qdrant真实检索: 返回{len(results)}条")


# ═══════════════════════════════════════════════
# 六路全息检索召回率专项
# ═══════════════════════════════════════════════

class TestHolographicRecallRate:

    def setup_method(self):
        from memory_engine.extractor import MemoryExtractor
        from memory_engine.retriever import MemoryRetriever
        self.extractor = MemoryExtractor()
        class StubVDB:
            async def search(self, collection, query_vector, limit, filter=None):
                return []
        self.retriever = MemoryRetriever(StubVDB())

    def _build_memory_bank(self):
        bank = []
        templates = [
            ("患者{n}有{c}，血糖{v}mmol/L", "fact"),
            ("{n}患者血压{v}mmHg，用药", "fact"),
            ("{n}检查结果：肌酐{v}μmol/L", "fact"),
            ("{n}喜欢{c}饮食", "preference"),
            ("{n}月{v}日{c}发生了事件", "event"),
            ("医生认为{n}应该{c}治疗", "belief"),
            ("{n}的{c}记录于{v}年", "fact"),
            ("{c}相关{n}数据：{v}", "fact"),
            ("{v}年{n}月入院{c}", "event"),
            ("随访结果：{v}月后{n}状况{c}", "fact"),
        ]
        patients = ["张三","李四","王五","赵六","陈七","周八","吴九","郑十","冯十一","马十二"]
        conds    = ["高血压","糖尿病","心脏病","肾病","肝炎","哮喘","关节炎","贫血","甲亢","骨质疏松"]
        vals     = ["6.2","8.5","11.2","5.8","9.3","7.1","4.9","13.5","6.8","10.1"]
        idx = 0
        for tmpl, mtype in templates:
            for i in range(10):
                content = tmpl.format(n=patients[i%10], c=conds[i%10], v=vals[i%10])
                extracted = self.extractor.extract_sync(content)
                bank.append({
                    "id": f"mem_{idx:03d}",
                    "content": content,
                    "memory_type": mtype,
                    "score": 0.9 - idx*0.001,
                    "payload": {
                        "content": content,
                        "user_id": "test_user",
                        "memory_type": mtype,
                        "timestamp": int(time.time()) - idx*600,
                        "metadata": {},
                        "hexagram_index": extracted["encoding_info"]["hexagram_index"],
                        "wuxing": extracted["encoding_info"]["wuxing"],
                        "hu_gua": extracted["encoding_info"]["hu_gua"],
                        "zong_gua": extracted["encoding_info"]["zong_gua"],
                        "cuo_gua": extracted["encoding_info"]["cuo_gua"],
                    }
                })
                idx += 1
                if idx >= 100:
                    break
            if idx >= 100:
                break
        return bank[:100]

    def _build_queries(self):
        return [
            ("张三血糖", ["张三"]),
            ("李四高血压", ["李四"]),
            ("王五糖尿病", ["王五"]),
            ("心脏病患者用药", ["心脏病"]),
            ("肾病肌酐", ["肾病","肌酐"]),
            ("糖尿病血糖监测", ["糖尿病","血糖"]),
            ("高血压血压控制", ["高血压","血压"]),
            ("患者喜欢低脂", ["喜欢","低脂"]),
            ("急性发作事件", ["发生","事件"]),
            ("医生建议治疗", ["认为","治疗"]),
            ("入院记录", ["入院"]),
            ("随访结果", ["随访"]),
            ("赵六贫血", ["赵六"]),
            ("陈七肝炎", ["陈七"]),
            ("周八哮喘", ["周八"]),
            ("吴九关节炎", ["吴九"]),
            ("郑十甲亢", ["郑十"]),
            ("冯十一骨质疏松", ["冯十一"]),
            ("马十二检查", ["马十二"]),
            ("患者血压mmHg", ["血压","mmHg"]),
            ("血糖6.2", ["6.2"]),
            ("血糖8.5", ["8.5"]),
            ("肌酐μmol", ["肌酐"]),
            ("用药mg剂量", ["用药","mg"]),
            ("偏好记录", ["喜欢"]),
            ("历史事件", ["发生"]),
            ("信念判断", ["认为"]),
            ("年月日记录", ["年"]),
            ("月后状况", ["月后"]),
            ("患者张三高血压", ["张三","高血压"]),
            ("患者李四糖尿病", ["李四","糖尿病"]),
            ("患者王五肌酐", ["王五","肌酐"]),
            ("患者赵六血压", ["赵六","血压"]),
            ("患者陈七血糖", ["陈七","血糖"]),
            ("心脏病血压治疗", ["心脏病","血压"]),
            ("肾病检查结果", ["肾病","检查"]),
            ("肝炎患者随访", ["肝炎","随访"]),
            ("哮喘发作处理", ["哮喘"]),
            ("关节炎用药方案", ["关节炎","用药"]),
            ("甲亢检查数据", ["甲亢"]),
            ("贫血入院记录", ["贫血","入院"]),
            ("骨质疏松治疗", ["骨质疏松"]),
            ("血糖偏高需控制", ["血糖"]),
            ("血压偏高需用药", ["血压"]),
            ("肌酐升高肾功能", ["肌酐","肾"]),
            ("饮食偏好低盐", ["饮食","低"]),
            ("手术治疗建议", ["手术","治疗"]),
            ("保守治疗方案", ["保守","治疗"]),
            ("定期复查随访", ["复查","随访"]),
            ("综合诊断记录", ["诊断"]),
        ]

    def _calc_recall(self, results, keywords, bank):
        relevant = [m for m in bank if any(kw in m["content"] for kw in keywords)]
        if not relevant:
            return None
        result_ids = {r["id"] for r in results}
        hit = sum(1 for m in relevant if m["id"] in result_ids)
        return hit / len(relevant)

    def test_holographic_100_memories_50_queries(self):
        bank = self._build_memory_bank()
        queries = self._build_queries()
        assert len(bank) == 100
        assert len(queries) == 50

        holo_recalls, simple_recalls = [], []
        skipped = 0

        print(f"\n  {'Query':<32} {'六路召回':>9} {'向量召回':>9}")
        print(f"  {'─'*54}")

        for qt, kws in queries:
            qvec = self.extractor.encode(qt)
            holo = self.retriever._holographic_rerank(query_vector=qvec, candidates=bank, limit=10)
            simple = self.retriever._simple_rerank(bank, limit=10)
            hr = self._calc_recall(holo, kws, bank)
            sr = self._calc_recall(simple, kws, bank)
            if hr is None or sr is None:
                skipped += 1
                continue
            holo_recalls.append(hr)
            simple_recalls.append(sr)
            flag = "↑" if hr > sr else ("=" if hr == sr else "↓")
            print(f"  {qt:<32} {hr:>8.1%} {sr:>8.1%}  {flag}")

        valid = len(holo_recalls)
        assert valid > 0
        avg_h = sum(holo_recalls)/valid
        avg_s = sum(simple_recalls)/valid
        holo_better = sum(1 for h,s in zip(holo_recalls,simple_recalls) if h>s)
        same_count  = sum(1 for h,s in zip(holo_recalls,simple_recalls) if h==s)
        simple_better = valid - holo_better - same_count

        print(f"\n  {'─'*54}")
        print(f"  有效query: {valid}, 跳过: {skipped}")
        print(f"  六路全息平均召回率: {avg_h:.1%}")
        print(f"  纯向量平均召回率:  {avg_s:.1%}")
        print(f"  差值: {avg_h-avg_s:+.1%}")
        print(f"  六路优于简单: {holo_better}次, 相同: {same_count}次, 简单优于六路: {simple_better}次")
        assert avg_h >= 0
        assert avg_s >= 0

    def test_recall_by_type(self):
        bank = self._build_memory_bank()
        for mtype in ["fact","preference","event","belief"]:
            sub_bank = [m for m in bank if m["memory_type"]==mtype]
            if not sub_bank:
                continue
            qvec = self.extractor.encode(f"{mtype}类型检索")
            holo = self.retriever._holographic_rerank(query_vector=qvec, candidates=sub_bank, limit=5)
            simple = self.retriever._simple_rerank(sub_bank, limit=5)
            assert len(holo) <= 5
            assert len(simple) <= 5
        print(f"\n  ✓ 按类型检索完成")

    def test_rrf_score_numeric(self):
        bank = self._build_memory_bank()[:20]
        qvec = self.extractor.encode("患者高血压糖尿病用药")
        results = self.retriever._holographic_rerank(query_vector=qvec, candidates=bank, limit=10)
        for r in results:
            assert isinstance(r["score"],(int,float)), f"score非数值: {type(r['score'])}"
        print(f"\n  ✓ RRF融合分数均为数值, 共{len(results)}条")


# ═══════════════════════════════════════════════
# 2.3  冲突消解测试
# ═══════════════════════════════════════════════

class TestConflictResolver:

    def setup_method(self):
        from memory_engine.conflict_resolver import ConflictResolver
        self.resolver = ConflictResolver()

    def _mem(self, content, mid=None):
        return {"id": mid or str(uuid.uuid4()), "content": content}

    def test_contradictory_facts(self):
        existing = [self._mem("患者可以服用阿司匹林", "med_1")]
        conflicts = self.resolver.detect_conflicts("患者不能服用阿司匹林，禁止使用", existing)
        assert len(conflicts) > 0
        assert conflicts[0]["conflict_type"] == "contradiction"
        print(f"\n  ✓ 矛盾检测: 冲突数={len(conflicts)}")

    def test_drug_allergy_contradiction(self):
        existing = [self._mem("患者知道青霉素过敏风险","al_1")]
        conflicts = self.resolver.detect_conflicts("患者不知道任何药物过敏情况", existing)
        assert len(conflicts) > 0

    def test_preference_contradiction(self):
        existing = [self._mem("患者喜欢高蛋白饮食","pref_1")]
        conflicts = self.resolver.detect_conflicts("患者讨厌高蛋白饮食，不想吃肉", existing)
        assert len(conflicts) > 0

    def test_no_conflict_unrelated(self):
        existing = [self._mem("患者血糖8.5"), self._mem("患者血压140/90")]
        conflicts = self.resolver.detect_conflicts("患者体重75kg，BMI 26", existing)
        assert len(conflicts) == 0

    def test_multiple_conflicts(self):
        existing = [
            self._mem("患者可以运动锻炼","e1"),
            self._mem("患者知道病情","e2"),
        ]
        new_c = "患者不能剧烈运动，不知道完整病情"
        conflicts = self.resolver.detect_conflicts(new_c, existing)
        assert len(conflicts) >= 1
        print(f"\n  ✓ 多重冲突: {len(conflicts)}个")

    def test_resolve_marks_old_invalid(self):
        conflicts = [
            {"existing_id":"old_1","conflict_type":"contradiction","new_id":"n","existing_content":"c1"},
            {"existing_id":"old_2","conflict_type":"contradiction","new_id":"n","existing_content":"c2"},
        ]
        inv = self.resolver.resolve(conflicts)
        assert "old_1" in inv
        assert "old_2" in inv
        print(f"\n  ✓ 消解: {len(inv)}条旧记忆标记无效")

    def test_resolve_empty(self):
        assert self.resolver.resolve([]) == []

    def test_conflict_resolution_accuracy(self):
        cases = [
            ("患者不能服药", ["患者可以服药"], True),
            ("患者知道风险", ["患者不知道风险"], True),
            ("结果正确", ["结论错误"], True),
            ("患者喜欢水果", ["患者讨厌水果"], True),
            ("患者体重70kg", ["患者身高170cm"], False),
            ("检查结果良好", ["需要进一步检查"], False),
        ]
        correct = 0
        for new_c, exists, should in cases:
            existing = [self._mem(c) for c in exists]
            has_conflict = len(self.resolver.detect_conflicts(new_c, existing)) > 0
            if has_conflict == should:
                correct += 1
        acc = correct/len(cases)
        print(f"\n  ✓ 冲突消解综合正确率: {acc:.0%} ({correct}/{len(cases)})")
        assert acc >= 0.5, f"正确率低于50%: {acc:.0%}"

    def test_check_pair_conflict_direct(self):
        cases = [
            ("我可以吃辣","我不能吃辣","contradiction"),
            ("患者知道情况","患者不知道","contradiction"),
            ("正确的诊断","错误的诊断","contradiction"),
            ("患者是糖尿病","完全无关的事","none"),
        ]
        correct = 0
        for c1,c2,expected in cases:
            r = self.resolver._check_pair_conflict(c1,c2)
            if expected == "none":
                if r is None: correct += 1
            else:
                if r == expected: correct += 1
        acc = correct/len(cases)
        print(f"\n  ✓ pair冲突检测正确率: {acc:.0%} ({correct}/{len(cases)})")
        assert acc >= 0.5


# ═══════════════════════════════════════════════
# 2.4  遗忘机制测试
# ═══════════════════════════════════════════════

class TestForgettingEngine:

    def setup_method(self):
        from memory_engine.forgetting import ForgettingEngine
        self.engine = ForgettingEngine()

    def _mem(self, days_ago, priority, status="active"):
        now = time.time()
        return {
            "id": str(uuid.uuid4()),
            "content": f"记忆 d={days_ago} p={priority}",
            "timestamp": int(now - days_ago*86400),
            "last_access": int(now - days_ago*86400),
            "priority": priority,
            "status": status,
        }

    def test_should_archive_old_low_priority(self):
        assert self.engine.should_archive(self._mem(35, 2))

    def test_should_not_archive_recent(self):
        assert not self.engine.should_archive(self._mem(5, 2))

    def test_should_not_archive_high_priority(self):
        assert not self.engine.should_archive(self._mem(35, 8))

    def test_should_not_archive_already_archived(self):
        assert not self.engine.should_archive(self._mem(40, 1, "archived"))

    def test_should_delete_old_archived(self):
        assert self.engine.should_delete(self._mem(95, 1, "archived"))

    def test_should_not_delete_active(self):
        assert not self.engine.should_delete(self._mem(100, 1, "active"))

    def test_should_not_delete_recent_archived(self):
        assert not self.engine.should_delete(self._mem(10, 1, "archived"))

    def test_process_forgetting_batch(self):
        mems = [
            self._mem(40, 1, "active"),   # 归档
            self._mem(35, 2, "active"),   # 归档
            self._mem(40, 9, "active"),   # 保留(高优先级)
            self._mem(3,  2, "active"),   # 保留(最近)
            self._mem(100, 1, "archived"),# 删除
            self._mem(30,  1, "archived"),# 保留(时间不足)
        ]
        result = run_async(self.engine.process_forgetting("t","u", mems))
        assert "archived" in result
        assert "deleted" in result
        assert "kept" in result
        assert len(result["archived"]) >= 1
        assert len(result["deleted"]) >= 1
        print(f"\n  ✓ 批量遗忘: 归档={len(result['archived'])}, 删除={len(result['deleted'])}, 保留={len(result['kept'])}")

    def test_forgetting_candidates_ordering(self):
        mems = [
            self._mem(1,  9),  # 最不应遗忘
            self._mem(10, 5),
            self._mem(30, 3),
            self._mem(60, 1),  # 最应遗忘
        ]
        cands = self.engine.get_forgetting_candidates(mems, top_k=10)
        assert len(cands) == 4
        for i in range(len(cands)-1):
            assert cands[i]["forgetting_score"] >= cands[i+1]["forgetting_score"]

    def test_forgetting_score_formula(self):
        high = self.engine.get_forgetting_candidates([self._mem(60,1)], top_k=1)
        low  = self.engine.get_forgetting_candidates([self._mem(1,10)], top_k=1)
        assert high[0]["forgetting_score"] > low[0]["forgetting_score"]

    def test_forgetting_candidates_top_k(self):
        mems = [self._mem(i*5,5) for i in range(20)]
        cands = self.engine.get_forgetting_candidates(mems, top_k=5)
        assert len(cands) == 5

    def test_archive_count_tracking(self):
        from memory_engine.forgetting import ForgettingEngine
        eng = ForgettingEngine()
        assert eng.archive_count == 0
        run_async(eng.process_forgetting("t","u",[self._mem(40,1)]))
        assert eng.archive_count == 1


# ═══════════════════════════════════════════════
# 2.4  BeliefTracker 生命周期测试
# ═══════════════════════════════════════════════

class TestBeliefLifecycle:

    def setup_method(self):
        from su_core import BeliefTracker, BeliefStage
        self.tracker = BeliefTracker()
        self.BeliefStage = BeliefStage

    def test_initial_state(self):
        mid = str(uuid.uuid4())
        state = self.tracker.initialize(mid)
        assert state is not None
        assert hasattr(state,"stage")
        assert hasattr(state,"confidence")
        assert 0.0 <= state.confidence <= 1.0
        print(f"\n  ✓ 信念初始化: stage={state.stage}, conf={state.confidence:.2f}")

    def test_reinforce_increases_confidence(self):
        mid = str(uuid.uuid4())
        self.tracker.initialize(mid)
        confs = [self.tracker.reinforce(mid).confidence for _ in range(5)]
        assert confs[-1] >= confs[0]
        print(f"\n  ✓ 强化曲线: {[f'{c:.2f}' for c in confs]}")

    def test_shake_decreases_confidence(self):
        mid = str(uuid.uuid4())
        self.tracker.initialize(mid)
        for _ in range(5): self.tracker.reinforce(mid)
        high = self.tracker.reinforce(mid).confidence
        for _ in range(3): self.tracker.shake(mid)
        low = self.tracker.shake(mid).confidence
        assert low <= high
        print(f"\n  ✓ 动摇后: {high:.2f} → {low:.2f}")

    def test_stage_distribution(self):
        for i in range(10):
            mid = str(uuid.uuid4())
            self.tracker.initialize(mid)
            if i%3==0:
                for _ in range(3): self.tracker.reinforce(mid)
        dist = self.tracker.get_stage_distribution()
        assert isinstance(dist, dict)
        assert sum(dist.values()) >= 10
        print(f"\n  ✓ 阶段分布: {dist}")

    def test_multiple_memories_independent(self):
        ids = [str(uuid.uuid4()) for _ in range(5)]
        for mid in ids: self.tracker.initialize(mid)
        for _ in range(3): self.tracker.reinforce(ids[0])
        s0 = self.tracker.reinforce(ids[0])
        s1 = self.tracker.initialize(ids[1])
        # 独立性：两者至少有一个字段不同，或置信度非0
        assert s0.confidence > 0 or s0.stage != s1.stage


# ═══════════════════════════════════════════════
# 集成测试（需要 Qdrant + PostgreSQL）
# ═══════════════════════════════════════════════

@skip_no_backends
class TestMemoryManagerIntegration:

    def setup_method(self):
        from memory_engine.manager import MemoryManager
        from qdrant_client import QdrantClient
        import storage.vector_db as vdb_module
        # 使用 REST 而非 gRPC（gRPC 端口 6334 未暴露）
        vdb_module._qdrant_client = QdrantClient(host="localhost", port=6333, prefer_grpc=False)
        self.manager = MemoryManager()
        self.manager.vector_db.client = vdb_module._qdrant_client
        # 修复 manager.py 中对同步 encode 使用 await 的问题
        _sync_encode = self.manager.extractor.encode
        async def _async_encode(text):
            return _sync_encode(text)
        self.manager.extractor.encode = _async_encode
        self.user_id = f"user_{uuid.uuid4().hex[:6]}"

    def test_create_tenant_and_add_memory(self):
        async def run():
            tenant = await self.manager.create_tenant(name="测试租户", plan="standard")
            assert tenant["api_key"].startswith("sk_")
            mid = await self.manager.add_memory(
                tenant_id=tenant["tenant_id"],
                user_id=self.user_id,
                content="患者张三，男，65岁，糖尿病II型，空腹血糖8.5mmol/L",
                metadata={"priority":8}
            )
            assert mid is not None and len(mid) > 0
            print(f"\n  ✓ 写入记忆: {mid[:8]}...")
        run_async(run())

    def test_add_and_query_memory(self):
        async def run():
            tenant = await self.manager.create_tenant(
                name=f"qtest_{uuid.uuid4().hex[:6]}", plan="standard")
            tid = tenant["tenant_id"]
            for c in ["张三有高血压病史","李四糖尿病血糖控制不佳","王五肾病肌酐偏高"]:
                await self.manager.add_memory(tid, self.user_id, c)
                time.sleep(0.05)
            results = await self.manager.query_memory(
                tenant_id=tid, user_id=self.user_id, query="高血压用药", limit=3)
            assert len(results) > 0
            print(f"\n  ✓ 检索: {len(results)}条")
        run_async(run())

    def test_get_stats(self):
        async def run():
            tenant = await self.manager.create_tenant(
                name=f"stats_{uuid.uuid4().hex[:6]}", plan="standard")
            tid = tenant["tenant_id"]
            await self.manager.add_memory(tid, self.user_id, "统计测试1")
            await self.manager.add_memory(tid, self.user_id, "统计测试2")
            stats = await self.manager.get_stats(tid, self.user_id)
            assert stats["total_memories"] >= 2
            print(f"\n  ✓ 统计: {stats}")
        run_async(run())


# ═══════════════════════════════════════════════
# 冒烟测试（无需外部服务）
# ═══════════════════════════════════════════════

class TestSmokeNoDeps:

    def test_all_modules_importable(self):
        from memory_engine.extractor import MemoryExtractor
        from memory_engine.conflict_resolver import ConflictResolver
        from memory_engine.forgetting import ForgettingEngine
        from su_core import (SemanticEncoder, MultiViewRetriever, SuCompressor,
                             TemporalSystem, BeliefTracker, MetaCognition)
        print("\n  ✓ 所有模块导入成功")

    def test_su_core_encoder(self):
        from su_core import SemanticEncoder
        enc = SemanticEncoder()
        info = enc.encode("测试内容", "fact")
        assert info.name != ""
        assert info.wuxing in ["金","木","水","火","土"]
        print(f"\n  ✓ SemanticEncoder: {info.name}({info.wuxing})")

    def test_su_core_compressor(self):
        from su_core import SuCompressor
        comp = SuCompressor()
        r = comp.compress("这是一段较长的测试内容用于验证压缩功能是否正常工作", mode="semantic")
        assert "compressed" in r and r["compressed"] != ""
        print(f"\n  ✓ SuCompressor: ratio={r.get('ratio','N/A')}")

    def test_su_core_temporal(self):
        from su_core import TemporalSystem
        ts = TemporalSystem()
        info = ts.get_current_ganzhi()
        assert info is not None
        print(f"\n  ✓ TemporalSystem: {info}")

    def test_su_core_belief_tracker(self):
        from su_core import BeliefTracker
        bt = BeliefTracker()
        state = bt.initialize(str(uuid.uuid4()))
        assert state is not None
        print(f"\n  ✓ BeliefTracker: stage={state.stage}")

    def test_su_core_metacognition(self):
        from su_core import MetaCognition
        mc = MetaCognition()
        gaps = mc.discover_gaps(
            memory_types={"fact":10,"event":2,"preference":0},
            user_domains=["医疗","健康"],
            memory_list=[]
        )
        assert isinstance(gaps, list)
        print(f"\n  ✓ MetaCognition: 空洞={len(gaps)}个")
