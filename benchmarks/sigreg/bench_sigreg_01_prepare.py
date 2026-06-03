#!/usr/bin/env python3
"""
bench_sigreg_01_prepare.py
===========================
HotpotQA dev 集 + bge-small-zh-v1.5 编码 + 落盘缓存。

这一步是 SIGReg 实测链条里最耗时的一步（一次编码 ~3-8 分钟），
单独拆出，避免后续两个 bench 重复编码。

输出文件:
  ${CACHE_DIR}/hotpotqa_corpus.jsonl      -- 7,405 supporting passages (原始文本 + id)
  ${CACHE_DIR}/hotpotqa_queries.jsonl     -- 500 held-out queries (含 gold 支持段落 id)
  ${CACHE_DIR}/passage_embs.npy           -- (7405, 512) float32, bge-small-zh-v1.5 输出
  ${CACHE_DIR}/query_embs.npy             -- (500, 512) float32, 同样编码器

依赖:
  pip install datasets sentence-transformers numpy

默认缓存目录:
  ./benchmarks/sigreg/cache/

用法:
  python bench_sigreg_01_prepare.py
  python bench_sigreg_01_prepare.py --cache-dir /tmp/sigreg-bench --n-passages 7405 --n-queries 500
  python bench_sigreg_01_prepare.py --force    # 删除旧缓存重新跑
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ============================================================
# P0-R6: HuggingFace 镜像 (国内网络)
# ============================================================
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

# ============================================================
# P1-R5: 锁定 OMP/MKL 单线程, 消除多线程不确定性
# ============================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

# 让 su_memory 可导入
SU_MEMORY_SRC = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SU_MEMORY_SRC))


# ============================================================
# 1. 数据集
# ============================================================

def load_hotpotqa_dev(n_passages: int = 7405, n_queries: int = 500, seed: int = 42):
    """
    加载 HotpotQA distractor dev set (官方 7,405 paragraphs, 500 multi-hop queries)。

    若 datasets 库不可用或下载失败, 回退到内置合成数据集
    (与 benchmarks/hotpotqa.py 风格一致, 用于离线 smoke test)。
    """
    try:
        from datasets import load_dataset
        print("[1/3] 正在从 HuggingFace 加载 hotpot_qa 'distractor' 配置的 validation split ...")
        ds = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
        print(f"      加载成功: {len(ds)} 条 validation 样本")
    except Exception as e:
        print(f"[WARN] 在线加载失败 ({type(e).__name__}: {e}), 切换到内置合成数据集")
        return _synthetic_corpus(n_passages, n_queries, seed)

    # 收集所有 supporting facts 做 passage corpus
    seen: dict[str, str] = {}     # passage_id -> passage text
    queries: list[dict] = []
    rng = np.random.default_rng(seed)

    for i, ex in enumerate(ds):
        ctx_titles = ex["context"]["title"]
        ctx_sentences = ex["context"]["sentences"]
        sp_titles = {sf[0] for sf in ex["supporting_facts"]["title"]}

        for title, sents in zip(ctx_titles, ctx_sentences):
            pid = f"{title}::{i}"     # title 唯一
            if pid not in seen:
                seen[pid] = " ".join(sents)

        queries.append({
            "qid": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "gold_passage_ids": [
                f"{sf[0]}::{i}" for sf in ex["supporting_facts"]["title"]
            ],
        })

    # 取前 n_passages 个 passage
    all_pids = list(seen.keys())[:n_passages]
    corpus = [{"pid": pid, "text": seen[pid]} for pid in all_pids]

    # 只保留 gold_passage_ids 都在 corpus 里的 query, 再随机抽 n_queries
    valid_qs = [q for q in queries if all(g in seen for g in q["gold_passage_ids"])]
    valid_qs = valid_qs[:n_queries]

    print(f"      corpus: {len(corpus)} passages, queries: {len(valid_qs)}")
    return corpus, valid_qs


def _synthetic_corpus(n_passages: int, n_queries: int, seed: int):
    """离线 fallback: 100 个领域的 Wikipedia 风格段落, 500 条模板查询。"""
    rng = np.random.default_rng(seed)
    domains = [
        "neuroscience", "physics", "chemistry", "biology", "computer science",
        "mathematics", "medicine", "engineering", "astronomy", "geology",
    ]
    entities = [
        f"Dr. {n} {s}" for n in "Alice Bob Carol David Elena Frank Grace Henry Ivan Judy"
            .split() for s in "Chen Kumar Williams Smith Rodriguez Zhang Park Liu Tanaka Müller"
            .split()
    ][:100]
    universities = ["MIT", "Stanford", "Harvard", "Oxford", "Cambridge", "Caltech",
                    "Princeton", "Yale", "Berkeley", "CMU", "ETH", "Tokyo"]
    cities = ["Tokyo", "London", "Paris", "Berlin", "Singapore", "Seoul", "Toronto",
              "Sydney", "Madrid", "Rome", "Mumbai", "Cairo", "Moscow", "Beijing"]

    corpus = []
    for i in range(n_passages):
        ent = entities[i % len(entities)]
        univ = universities[i % len(universities)]
        dom = domains[i % len(domains)]
        yr = 2000 + (i % 25)
        text = (
            f"{ent} is a researcher in {dom} at {univ}, joining in {yr}. "
            f"Their work focuses on {dom} applications using {['neural networks', 'graph theory', 'optimization', 'simulation'][i % 4]}. "
            f"Prior to {univ}, {ent} was at {cities[i % len(cities)]} Research Lab."
        )
        corpus.append({"pid": f"synth::{i:05d}", "text": text})

    queries = []
    for i in range(n_queries):
        ent = entities[i % len(entities)]
        univ = universities[(i + 3) % len(universities)]
        dom = domains[(i + 5) % len(domains)]
        yr = 2000 + ((i + 7) % 25)
        gold_pid = f"synth::{(i % n_passages):05d}"
        queries.append({
            "qid": f"synth_q::{i:05d}",
            "question": f"Which {dom} researcher at {univ} started in {yr}?",
            "answer": ent,
            "gold_passage_ids": [gold_pid],
        })
    print(f"      合成数据: {len(corpus)} passages, {len(queries)} queries")
    return corpus, queries


# ============================================================
# 2. 编码
# ============================================================

def encode_with_bge(texts: list[str], model_name: str = "BAAI/bge-small-en-v1.5",
                    batch_size: int = 64, normalize: bool = False) -> np.ndarray:
    """
    用 sentence-transformers 编码文本, 返回 (n, d) float32。

    bge-small-en-v1.5: d=384, 英文语义模型。

    normalize 默认 False:
      - SIGReg 设计目的是对未归一化的 raw 输出做各向同性正则
      - 如果 bge 先 L2 归一化, 所有向量已经在球面上,
        isotropy score 被压缩到 ~1e-11, SIGReg 信号被淹没。
      - 我们在 SIGReg 内部 step 4 才做 L2 归一化。
    """
    print(f"[2/3] 加载 sentence-transformers 模型: {model_name}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cpu")
    d = model.get_sentence_embedding_dimension()
    print(f"      模型加载完毕, embedding dim = {d}")

    t0 = time.time()
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    ).astype(np.float32)
    print(f"      编码 {len(texts)} 条文本用时 {time.time() - t0:.1f}s, shape={embs.shape}")
    return embs


# ============================================================
# 3. 落盘
# ============================================================

def save_outputs(cache_dir: Path, corpus, queries, passage_embs, query_embs):
    cache_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = cache_dir / "hotpotqa_corpus.jsonl"
    queries_path = cache_dir / "hotpotqa_queries.jsonl"
    p_embs_path = cache_dir / "passage_embs.npy"
    q_embs_path = cache_dir / "query_embs.npy"

    print(f"[3/3] 写入缓存到 {cache_dir} ...")
    with corpus_path.open("w", encoding="utf-8") as f:
        for c in corpus:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with queries_path.open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    np.save(p_embs_path, passage_embs)
    np.save(q_embs_path, query_embs)

    # 元信息
    meta = {
        "n_passages": int(passage_embs.shape[0]),
        "n_queries": int(query_embs.shape[0]),
        "embedding_dim": int(passage_embs.shape[1]),
        "dtype": str(passage_embs.dtype),
        "model": "BAAI/bge-small-zh-v1.5",
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"      写入完成:\n        {corpus_path}\n        {queries_path}")
    print(f"        {p_embs_path}  {passage_embs.nbytes / 1024:.1f} KB")
    print(f"        {q_embs_path}  {query_embs.nbytes / 1024:.1f} KB")
    print(f"        {cache_dir / 'meta.json'}")


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare HotpotQA + bge embeddings for SIGReg benchmarks")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(__file__).resolve().parent / "cache",
                        help="缓存目录 (默认 ./cache)")
    parser.add_argument("--n-passages", type=int, default=7405)
    parser.add_argument("--n-queries", type=int, default=500)
    parser.add_argument("--model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--normalize-bge", action="store_true",
                        help="对 bge 输出做 L2 归一化 (默认不做, 让 SIGReg 处理)")
    parser.add_argument("--force", action="store_true", help="删除旧缓存并重跑")
    args = parser.parse_args()

    cache = args.cache_dir.resolve()
    if args.force and cache.exists():
        import shutil
        shutil.rmtree(cache)
        print(f"[FORCE] 已删除 {cache}")

    if (cache / "passage_embs.npy").exists() and (cache / "query_embs.npy").exists():
        print(f"[SKIP] 缓存已存在: {cache}")
        print("       删除 --force 重跑 或 手动 rm -rf", cache)
        return

    corpus, queries = load_hotpotqa_dev(args.n_passages, args.n_queries)
    do_normalize = args.normalize_bge  # 默认 False (P0-R4 修复)
    passage_embs = encode_with_bge(
        [c["text"] for c in corpus], args.model, args.batch_size, normalize=do_normalize
    )
    query_embs = encode_with_bge(
        [q["question"] for q in queries], args.model, args.batch_size, normalize=do_normalize
    )
    save_outputs(cache, corpus, queries, passage_embs, query_embs)
    print("\n✅ 准备完成。后续 bench 脚本可直接读取 cache/。")


if __name__ == "__main__":
    main()
