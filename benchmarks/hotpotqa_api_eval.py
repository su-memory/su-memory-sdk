"""真实 HotpotQA 评测: 线上 API reader (DeepSeek 等) 标准 EM.

口径: 标准 HotpotQA EM — API reader 抽取 span 经官方 normalize 后 == gold.
这是真实冲击 Hindsight 70.83% 的主力路径 (线上大模型多跳推理远超本地 7B).

自动探测已配置的 API (DEEPSEEK > GLM > KIMI > OPENAI), 三路融合检索 + API reader.

运行: python benchmarks/hotpotqa_api_eval.py [--sample N] [--model X] [--probe-only]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from su_memory.sdk.multi_hop_reader import MultiHopReader
from su_memory.sdk.api_reader import APIReader, probe_api, squad_em, squad_f1
from ab_hotpotqa_real import embed, embed_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--model", type=str, default=None, help="指定模型 (如 deepseek-chat / deepseek-reasoner)")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--probe-only", action="store_true", help="只探测 API 连通性, 不评测")
    args = ap.parse_args()

    provider = probe_api()
    print(f"探测到 API provider: {provider or '无'}")
    if args.probe_only:
        return

    if provider is None:
        print("[error] 未配置任何 API key (DEEPSEEK_API_KEY/GLM_API_KEY/KIMI_API_KEY/OPENAI_API_KEY)")
        sys.exit(1)

    reader_llm = APIReader(model=args.model)
    print(f"reader: {reader_llm.model_id}")

    # 连通性快速测试 (1题)
    with open(ROOT / "benchmarks/data/hotpotqa_validation_200.json") as f:
        data = json.load(f)
    test_ans = reader_llm.extract_answer(
        "When did Beyonce become popular?",
        "Beyonce rose to fame in the late 1990s as lead singer.",
    )
    print(f"连通性测试 (应含 late 1990s): {test_ans!r}")
    if not test_ans:
        print("[error] API 无响应 (网络/key 问题), 终止")
        sys.exit(1)

    if args.sample:
        data = data[: args.sample]

    reader = MultiHopReader(embed, embed_batch, llm_reader=reader_llm)
    em = f1tot = 0
    by_type = {"bridge": [0, 0], "comparison": [0, 0]}
    t0 = time.time()
    for i, d in enumerate(data):
        res = reader.retrieve(d["question"], d["context"], top_k=args.top_k)
        pred = reader.extract_answer(d["question"], res.answer_context)
        gold = d["answer"]
        hit = squad_em(pred, gold)
        em += hit
        f1tot += squad_f1(pred, gold)
        by_type[d["type"]][0] += hit
        by_type[d["type"]][1] += 1
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(data)}] 累计 EM {em/(i+1):.1%} ...", flush=True)

    n = len(data)
    print("=" * 64)
    print(f"真实 HotpotQA 标准 EM (API reader: {reader_llm.model})")
    print(f"数据: {n} 题 (官方 validation, 全 hard)")
    print("=" * 64)
    print(f"标准 EM: {em}/{n} = {em/n:.1%}")
    print(f"F1 (token): {f1tot/n:.1%}")
    if by_type["bridge"][1]:
        print(f"  bridge ({by_type['bridge'][1]}题): {by_type['bridge'][0]/by_type['bridge'][1]:.1%}")
    if by_type["comparison"][1]:
        print(f"  comparison ({by_type['comparison'][1]}题): {by_type['comparison'][0]/by_type['comparison'][1]:.1%}")
    print(f"耗时: {time.time()-t0:.0f}s")
    print(f"对照: Hindsight 70.83% | IRRR+BERT 55.0% | DFGN 48.2% | 本地7B 48.0%")
    delta = em / n - 0.7083
    print(f'vs Hindsight: {delta*100:+.1f}个百分点 {"✓真实超越!" if delta>0 else "✗未超越"}')


if __name__ == "__main__":
    main()
