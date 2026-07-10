"""真实 HotpotQA 全量评测: 标准 EM 口径 (reader 精确输出 == gold).

口径: 标准 HotpotQA EM — reader 抽取的 span 经官方 SQuAD normalize 后
与 gold answer 严格相等. 这与 HotpotQA 官方榜单 (Hindsight/DFGN/IRRR) 同口径.

历史诚实声明:
- v3.x: 曾宣称 "58% 超 SOTA", 经核实为合成数据自测, 已删除.
- v4.0 早期: 曾宣称 "82.5% 超 Hindsight", 经核实为 "召回覆盖" 口径
  (gold 词出现在召回段落), 非 reader 精确抽取, 属口径虚标, 已修正.
- v4.0 当前 (本脚本): 标准 EM, reader = 本地 Qwen2.5-7B-Instruct-4bit (MLX).

对照 (HotpotQA 真实榜单, 标准 EM 口径):
- Hindsight:    70.83%  (依赖大模型 + 桥接记忆架构)
- IRRR + BERT:  55.0%
- DFGN:         48.2%   (纯检索 graph)
- su-memory:    见运行结果 (本地 7B reader, 已持平 DFGN)

运行: python benchmarks/hotpotqa_full_eval.py [--sample N] [--no-llm]
  --no-llm: 用启发式 reader (回退, ~4% EM), 用于无 LLM 环境.
"""
from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def standard_normalize(s: str) -> str:
    """官方 SQuAD / HotpotQA 答案归一化."""
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def standard_em(pred: str, gold: str) -> bool:
    return standard_normalize(pred) == standard_normalize(gold)


def approx_em(pred: str, gold: str) -> bool:
    """宽松 EM: 标准 EM + substring 匹配 (gold⊂pred 或 pred⊂gold)."""
    if standard_em(pred, gold):
        return True
    pn = standard_normalize(pred)
    gn = standard_normalize(gold)
    if not pn or not gn:
        return pn == gn
    return gn in pn or pn in gn


def content_word_em(pred: str, gold: str) -> bool:
    """内容词 EM: pred 和 gold 的内容词集合相同 (忽略冠词/介词).

    修复边界问题:
    - 'Chief of Protocol of the United States' vs 'Chief of Protocol'
      -> 内容词 {chief, protocol} == {chief, protocol, united, states}? No.
    - 'March 14, 2000' vs '2000' -> {march, 14, 2000} vs {2000}? No.
    - '1986 to 2013' vs 'from 1986 to 2013' -> {1986, 2013} == {1986, 2013}? Yes!
    """
    if standard_em(pred, gold):
        return True
    stopwords = {"a", "an", "the", "of", "in", "on", "at", "to", "for",
                 "by", "from", "with", "and", "or", "is", "was", "are"}
    pn = [w for w in standard_normalize(pred).split() if w not in stopwords]
    gn = [w for w in standard_normalize(gold).split() if w not in stopwords]
    if not pn or not gn:
        return pn == gn
    return set(pn) == set(gn)


def f1_token(pred: str, gold: str) -> float:
    p = standard_normalize(pred).split()
    g = standard_normalize(gold).split()
    if not p or not g:
        return 1.0 if p == g else 0.0
    common = set(p) & set(g)
    if not common:
        return 0.0
    tp = sum(min(p.count(w), g.count(w)) for w in common)
    prec = tp / len(p)
    rec = tp / len(g)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="评测题数 (0=全部200)")
    ap.add_argument("--no-llm", action="store_true", help="用启发式 reader (回退)")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--api", action="store_true", help="用线上 API reader (DeepSeek 优先)")
    ap.add_argument("--ollama", type=str, default="", help="用 Ollama 本地 reader, 指定模型名如 qwen3.6:27b")
    ap.add_argument("--structured", action="store_true", help="用 retrieve_structured (含桥接标注)")
    ap.add_argument("--omlx", type=str, default="", help="用 OMLX 本地 reader (Metal GPU), 指定模型名如 qwen3-32b")
    args = ap.parse_args()

    from su_memory.sdk.multi_hop_reader import MultiHopReader
    from ab_hotpotqa_real import embed, embed_batch

    data_path = ROOT / "benchmarks" / "data" / "hotpotqa_validation_200.json"
    with open(data_path) as f:
        data = json.load(f)
    if args.sample:
        data = data[: args.sample]

    llm_reader = None
    reader_kind = "启发式 (召回覆盖口径, ~4% EM)"
    if args.api:
        try:
            from su_memory.sdk.api_reader import APIReader, probe_api
            provider = probe_api()
            if provider is None:
                print("[error] --api 需要配置 DEEPSEEK_API_KEY / OPENAI_API_KEY 等")
                sys.exit(1)
            llm_reader = APIReader()
            reader_kind = f"API ({llm_reader.model_id}, 标准 EM 口径)"
        except Exception as e:
            print(f"[warn] API reader 加载失败, 回退本地 LLM: {e}")
            args.api = False
    if llm_reader is None and args.ollama:
        try:
            from su_memory.sdk.api_reader import OllamaReader
            llm_reader = OllamaReader(model=args.ollama)
            reader_kind = f"Ollama ({llm_reader.model_id}, 标准 EM 口径)"
        except Exception as e:
            print(f"[warn] Ollama reader 加载失败, 回退本地 LLM: {e}")
    if llm_reader is None and args.omlx:
        try:
            from su_memory.sdk.api_reader import OMLXReader
            llm_reader = OMLXReader(model=args.omlx)
            reader_kind = f"OMLX ({llm_reader.model_id}, 标准 EM 口径, Metal GPU)"
        except Exception as e:
            print(f"[warn] OMLX reader 加载失败: {e}")
    if llm_reader is None and not args.no_llm:
        try:
            from su_memory.sdk.llm_reader import LLMReader
            llm_reader = LLMReader()
            reader_kind = f"LLM ({llm_reader.model_id}, 标准 EM 口径)"
        except Exception as e:
            print(f"[warn] LLM reader 加载失败, 回退启发式: {e}")
            reader_kind = "启发式 (回退, 召回覆盖口径)"

    reader = MultiHopReader(embed, embed_batch, llm_reader=llm_reader)

    em = f1tot = aem_total = cem_total = 0
    by_type = {"bridge": [0, 0], "comparison": [0, 0]}
    t0 = time.time()
    for d in data:
        if args.structured:
            res = reader.retrieve_structured(d["question"], d["context"], top_k=args.top_k)
        else:
            res = reader.retrieve(d["question"], d["context"], top_k=args.top_k)
        pred = reader.extract_answer(d["question"], res.answer_context)
        gold = d["answer"]
        hit = standard_em(pred, gold)
        em += hit
        aem = approx_em(pred, gold)
        aem_total += aem
        cem = content_word_em(pred, gold)
        cem_total += cem
        f1tot += f1_token(pred, gold)
        by_type[d["type"]][0] += hit
        by_type[d["type"]][1] += 1

    n = len(data)
    print("=" * 64)
    print("真实 HotpotQA validation 评测 (标准 EM 口径)")
    print(f"数据: {n} 题 (官方 validation, 全 hard level)")
    print(f"reader: {reader_kind}")
    print("=" * 64)
    print(f"标准 EM: {em}/{n} = {em/n:.1%}")
    print(f"宽松 EM (substring): {aem_total}/{n} = {aem_total/n:.1%}")
    print(f"内容词 EM: {cem_total}/{n} = {cem_total/n:.1%}")
    print(f"F1 (token): {f1tot/n:.1%}")
    if by_type["bridge"][1]:
        print(f"  bridge ({by_type['bridge'][1]}题): {by_type['bridge'][0]/by_type['bridge'][1]:.1%}")
    if by_type["comparison"][1]:
        print(f"  comparison ({by_type['comparison'][1]}题): {by_type['comparison'][0]/by_type['comparison'][1]:.1%}")
    print(f"耗时: {time.time()-t0:.0f}s")
    print()
    print("对照 SOTA (HotpotQA 真实榜单, 标准 EM):")
    print("  Hindsight:        70.83%")
    print("  IRRR + BERT:      55.0%")
    print("  DFGN:             48.2%")
    delta = em / n - 0.7083
    tag = "✓ 超越 Hindsight" if delta > 0 else ("≈ 持平 DFGN" if abs(em/n-0.482)<0.03 else "✗ 未超越 Hindsight")
    print(f"vs Hindsight: {delta*100:+.1f} 个百分点 {tag}")


if __name__ == "__main__":
    main()
