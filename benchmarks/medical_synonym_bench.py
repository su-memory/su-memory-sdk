#!/usr/bin/env python3
"""
C1 医疗同义召回基准

量化医疗术语同义/近义召回率。
写入术语 A，用同义术语 B 查询，统计 recall@k / precision@k。

用法:
  python benchmarks/medical_synonym_bench.py
  python benchmarks/medical_synonym_bench.py --with-synonym-expand
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

os.environ.setdefault("SU_MEMORY_SKIP_ENV_CHECK", "1")
os.environ.setdefault("MEMORY_EMBEDDING_BACKEND", "none")

# 50 对中文医学术语同义词/近义词
# 格式: (存储术语, 查询同义术语, 语义关系)
MEDICAL_SYNONYM_PAIRS = [
    # 药物通用名 ↔ 商品名/英文
    ("华法林", "warfarin", "中英对照"),
    ("二甲双胍", "metformin", "中英对照"),
    ("甲氨蝶呤", "methotrexate", "中英对照"),
    ("地高辛", "digoxin", "中英对照"),
    ("卡托普利", "captopril", "中英对照"),
    ("呋塞米", "furosemide", "中英对照"),
    ("环丙沙星", "ciprofloxacin", "中英对照"),
    ("左旋甲状腺素", "levothyroxine", "中英对照"),
    # 检验项目同义
    ("白蛋白", "albumin", "中英对照"),
    ("血红蛋白", "hemoglobin", "中英对照"),
    ("血糖", "glucose", "中英对照"),
    ("肌酐", "creatinine", "中英对照"),
    ("前白蛋白", "prealbumin", "中英对照"),
    ("转铁蛋白", "transferrin", "中英对照"),
    ("C反应蛋白", "CRP", "缩写"),
    ("体重指数", "BMI", "缩写"),
    # 临床概念同义/近义
    ("禁忌症", "过敏", "近义"),
    ("禁忌症", "不耐受", "近义"),
    ("营养不良", "营养缺乏", "近义"),
    ("营养不良", "malnutrition", "中英对照"),
    ("高血压", "hypertension", "中英对照"),
    ("糖尿病", "diabetes", "中英对照"),
    ("贫血", "anemia", "中英对照"),
    ("低蛋白血症", "hypoproteinemia", "中英对照"),
    # 营养学术语
    ("蛋白质", "protein", "中英对照"),
    ("维生素K", "vitamin K", "中英对照"),
    ("维生素K", "VK", "缩写"),
    ("维生素B12", "cobalamin", "学名"),
    ("叶酸", "folic acid", "中英对照"),
    ("叶酸", "folate", "近义"),
    ("膳食纤维", "dietary fiber", "中英对照"),
    ("热量", "卡路里", "近义"),
    ("卡路里", "calories", "中英对照"),
    # 症状同义
    ("头晕", "眩晕", "近义"),
    ("恶心", "呕吐", "近义"),
    ("腹泻", "diarrhea", "中英对照"),
    ("便秘", "constipation", "中英对照"),
    ("水肿", "浮肿", "近义"),
    ("消瘦", "体重下降", "近义"),
    # 诊疗术语
    ("医嘱", "处方", "近义"),
    ("处方", "prescription", "中英对照"),
    ("营养方案", "diet plan", "中英对照"),
    ("营养方案", "饮食方案", "近义"),
    ("肠内营养", "enteral nutrition", "中英对照"),
    ("肠外营养", "parenteral nutrition", "中英对照"),
    ("口服", "per os", "拉丁缩写"),
    ("静脉注射", "IV", "缩写"),
    ("肌内注射", "IM", "缩写"),
    ("随访", "复诊", "近义"),
    ("出院", "discharge", "中英对照"),
]


def run_benchmark(
    with_expand: bool = False,
    keyword_only: bool = False,
) -> dict:
    """运行同义召回基准。

    Args:
        with_expand: 是否启用同义词扩展（query 侧扩展）
        keyword_only: 纯关键词模式（禁用向量，模拟医院内网）

    Returns:
        {"recall@5": float, "precision@5": float, "total": int, "hits": int}
    """
    from su_memory.sdk.lite_pro import SuMemoryLitePro
    from su_memory.clinical.synonym_dict import MedicalSynonymDict

    expand_dict: dict[str, list[str]] = {}
    if with_expand:
        syn = MedicalSynonymDict()
        expand_dict = syn.get_reverse_map()

    tmpdir = tempfile.mkdtemp()

    if keyword_only:
        # 纯关键词引擎（模拟内网无向量）
        engine = SuMemoryLitePro(
            storage_path=os.path.join(tmpdir, "kw_bench"),
            enable_vector=False,
            enable_llm_energy=False,
        )
        # 手动写入
        stored_terms = set()
        for store_term, _, _ in MEDICAL_SYNONYM_PAIRS:
            if store_term not in stored_terms:
                engine.add(f"{store_term}相关记录", metadata={"patient_id": "BENCH"})
                stored_terms.add(store_term)
        # 手动查询（含扩展）
        hits = 0
        precision_hits = 0.0
        total = len(MEDICAL_SYNONYM_PAIRS)
        for store_term, query_term, _ in MEDICAL_SYNONYM_PAIRS:
            q = query_term
            if with_expand and query_term in expand_dict:
                q = " ".join([query_term] + expand_dict[query_term])
            results = engine.query(q, top_k=5)
            if any(store_term in r.get("content", "") for r in results):
                hits += 1
            relevant = sum(1 for r in results if any(
                t in r.get("content", "") for t in stored_terms))
            precision_hits += relevant / max(len(results), 1)
        return {
            "recall@5": hits / total,
            "precision@5": precision_hits / total,
            "total_pairs": total, "hits": hits,
            "mode": f"keyword_{'expand' if with_expand else 'baseline'}",
        }

    from su_memory.clinical import ClinicalMemoryClient
    client = ClinicalMemoryClient(
        storage_path=os.path.join(tmpdir, "syn_bench"),
        embedding_backend="none",
        compliance_level=None,
        safety_screen=False,
        synonym_expand=with_expand,
    )

    # 同义词典（扩展用）
    expand_dict: dict[str, list[str]] = {}
    if with_expand:
        from su_memory.clinical.synonym_dict import MedicalSynonymDict
        syn = MedicalSynonymDict()
        expand_dict = syn.get_reverse_map()

    # 写入所有存储术语（每对存术语 A）
    stored_terms = set()
    for store_term, _, _ in MEDICAL_SYNONYM_PAIRS:
        if store_term not in stored_terms:
            client.add_patient_event("BENCH", f"{store_term}相关记录", "term")
            stored_terms.add(store_term)

    # 用同义术语 B 查询，统计召回
    hits = 0
    precision_hits = 0
    total_queries = 0

    for store_term, query_term, relation in MEDICAL_SYNONYM_PAIRS:
        total_queries += 1
        # query 侧扩展
        if with_expand and query_term in expand_dict:
            expanded = " ".join([query_term] + expand_dict[query_term])
        else:
            expanded = query_term

        results = client.recall("BENCH", expanded, top_k=5, max_fetch=500)

        # recall@5: 是否召回了包含 store_term 的记忆
        recalled = any(store_term in r.get("content", "") for r in results)
        if recalled:
            hits += 1

        # precision@5: 召回结果中相关的比例（简化：含任一存储术语即相关）
        relevant = sum(1 for r in results if any(
            t in r.get("content", "") for t in stored_terms
        ))
        precision_hits += relevant / max(len(results), 1)

    recall = hits / total_queries
    precision = precision_hits / total_queries

    return {
        "recall@5": recall,
        "precision@5": precision,
        "total_pairs": total_queries,
        "hits": hits,
        "mode": "with_expand" if with_expand else "baseline",
    }


def main():
    parser = argparse.ArgumentParser(description="医疗同义召回基准")
    parser.add_argument("--with-synonym-expand", action="store_true",
                        help="启用同义词扩展")
    parser.add_argument("--keyword-only", action="store_true",
                        help="纯关键词模式（禁用向量，模拟医院内网）")
    args = parser.parse_args()

    print("=" * 60)
    print("医疗同义召回基准 (Medical Synonym Recall Benchmark)")
    print("=" * 60)
    print(f"同义词对数: {len(MEDICAL_SYNONYM_PAIRS)}")
    print()

    result = run_benchmark(with_expand=args.with_synonym_expand, keyword_only=args.keyword_only)
    print(f"模式: {result['mode']}")
    print(f"同义词对总数: {result['total_pairs']}")
    print(f"命中数: {result['hits']}")
    print(f"Recall@5:    {result['recall@5']:.1%}  ({result['hits']}/{result['total_pairs']})")
    print(f"Precision@5: {result['precision@5']:.1%}")
    print()

    if not args.with_synonym_expand:
        print("提示: 运行 --with-synonym-expand 查看扩展后召回率")
    else:
        print("✅ 同义词扩展已启用")

    # 退出码：recall >= 0.85 为通过
    threshold = 0.85
    passed = result["recall@5"] >= threshold
    print(f"\n{'✅ PASS' if passed else '❌ FAIL'} (阈值 Recall@5 ≥ {threshold:.0%})")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
