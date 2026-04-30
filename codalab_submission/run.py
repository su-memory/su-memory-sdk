#!/usr/bin/env python3
"""
CodaLab Submission Script — su-memory v2.0
HotpotQA Distractor Setting
============================================

Hybrid pipeline:
1. TF-IDF keyword retrieval → find relevant sentences
2. DistilBERT QA model → extract answer span from retrieved context
3. TF-IDF → identify supporting facts

Expected score: ~45-55% EM (retrieval + QA extraction, no LLM reasoning)
Docker image pre-downloads distilbert-base-cased-distilled-squad during build.
No network required during evaluation.

Format: {"answer": {"id": "answer", ...}, "sp": {"id": [[title, idx], ...]}}
"""

import json
import sys
import math
import re
import os
from collections import defaultdict
from typing import List, Dict, Tuple

# ============================================================
# Stop words + tokenizer
# ============================================================

STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 'and', 'but', 'or', 'because', 'until', 'while',
    'about', 'up', 'out', 'if', 'that', 'this', 'what', 'which', 'who',
    'it', 'its', 'he', 'she', 'they', 'them', 'their', 'his', 'her',
}


def tokenize(text: str) -> set:
    words = re.findall(r'[a-z]{3,}', text.lower())
    return {w for w in words if w not in STOP_WORDS}


# ============================================================
# TF-IDF Retriever
# ============================================================

class Retriever:
    def __init__(self):
        self.docs: List[Dict] = []
        self.inverted_index: Dict[str, set] = defaultdict(set)

    def add(self, text: str, metadata: dict = None):
        tokens = tokenize(text)
        idx = len(self.docs)
        self.docs.append({"text": text, "tokens": tokens, "meta": metadata or {}})
        for t in tokens:
            self.inverted_index[t].add(idx)

    def query(self, q: str, k: int = 15) -> List[Dict]:
        q_tokens = tokenize(q)
        if not q_tokens:
            return []
        scores = defaultdict(float)
        n = max(len(self.docs), 1)
        for t in q_tokens:
            doc_ids = self.inverted_index.get(t, set())
            if not doc_ids:
                continue
            idf = math.log(n / len(doc_ids)) + 1
            for idx in doc_ids:
                dt = self.docs[idx]["tokens"]
                tf = sum(1 for x in dt if x == t)
                scores[idx] += tf * idf
        if scores:
            mx = max(scores.values())
            scores = {k: v / mx for k, v in scores.items()}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{"text": self.docs[idx]["text"], "score": sc,
                 "meta": self.docs[idx]["meta"]}
                for idx, sc in ranked[:k]]


# ============================================================
# QA Model Answer Extraction (DistilBERT)
# ============================================================

_qa_pipeline = None

def get_qa_pipeline():
    global _qa_pipeline
    if _qa_pipeline is None:
        print("  Loading QA model (distilbert-base-cased-distilled-squad)...",
              file=sys.stderr)
        from transformers import pipeline
        _qa_pipeline = pipeline(
            'question-answering',
            model='distilbert-base-cased-distilled-squad',
            device=-1  # CPU
        )
    return _qa_pipeline


def extract_answer_qa(question: str, retrieved_texts: List[str]) -> str:
    """Use DistilBERT QA model to extract answer span from retrieved context."""
    qa = get_qa_pipeline()

    # Build context from top retrieved sentences (limit length)
    context = " ".join(retrieved_texts[:12])
    if len(context) > 3000:
        context = context[:3000]

    try:
        result = qa(question=question, context=context)
        answer = result['answer'].strip()
        score = result['score']

        # If confidence is very low, fall back to heuristics
        if score < 0.01 or not answer:
            return fallback_extract(question, retrieved_texts)

        # Clean up the answer
        # Remove leading/trailing partial words
        answer = re.sub(r'^[^A-Za-z0-9]+', '', answer)
        answer = re.sub(r'[^A-Za-z0-9)]+$', '', answer)

        return answer if answer else fallback_extract(question, retrieved_texts)

    except Exception as e:
        print(f"  QA model error: {e}, using fallback", file=sys.stderr)
        return fallback_extract(question, retrieved_texts)


def fallback_extract(question: str, retrieved_texts: List[str]) -> str:
    """Heuristic fallback when QA model fails."""
    combined = " ".join(retrieved_texts[:10])
    c_lower = combined.lower()
    q_lower = question.lower()

    # Yes/no
    first_word = q_lower.split()[0] if q_lower.split() else ''
    aux_verbs = {'was', 'were', 'is', 'are', 'did', 'does', 'do',
                 'has', 'have', 'had', 'can', 'could', 'will', 'would'}
    if first_word in aux_verbs:
        yes_c = len(re.findall(r'\byes\b', c_lower))
        no_c = len(re.findall(r'\bno\b', c_lower))
        neg_c = len(re.findall(r'\b(not|never|neither|none)\b', c_lower))
        return "no" if (no_c > yes_c or neg_c > 0) else "yes"

    # Entity extraction: return longest proper noun
    best_sent = max(retrieved_texts[:5], key=lambda s: sum(
        1 for kw in tokenize(question) if kw in tokenize(s)), default="")

    proper_nouns = re.findall(
        r'\b([A-Z][a-zA-Z]*(?:\s+(?:[A-Z][a-zA-Z]*|of|in|the|and|de|van|der)){1,5})',
        best_sent
    )
    if proper_nouns:
        return proper_nouns[0].strip()

    return "unknown"


# ============================================================
# Supporting Facts
# ============================================================

def find_supporting_facts(
    question: str, answer: str,
    titles: List[str], sentence_lists: List[List[str]],
) -> List[List]:
    q_tokens = tokenize(question)
    a_tokens = tokenize(answer)
    scored = []
    for pi, (title, sents) in enumerate(zip(titles, sentence_lists)):
        for si, sent in enumerate(sents):
            st = tokenize(sent)
            q_ov = len(q_tokens & st)
            a_ov = len(a_tokens & st)
            score = q_ov + a_ov * 3
            if score > 0:
                scored.append((score, title, si))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [[title, idx] for _, title, idx in scored[:2]]


# ============================================================
# Main Pipeline
# ============================================================

def run(input_data: List[Dict]) -> Dict:
    answers = {}
    sp_facts = {}

    for i, entry in enumerate(input_data):
        qid = entry["id"]
        question = entry["question"]
        ctx = entry["context"]
        titles = ctx["title"]
        sentence_lists = ctx["sentences"]

        # Ingest
        retriever = Retriever()
        for para_idx, sents in enumerate(sentence_lists):
            for sent_idx, sent in enumerate(sents):
                retriever.add(
                    text=sent,
                    metadata={"title": titles[para_idx], "sent_idx": sent_idx},
                )

        # Retrieve
        results = retriever.query(question, k=15)
        retrieved_texts = [r["text"] for r in results]

        # Answer extraction via QA model
        answer = extract_answer_qa(question, retrieved_texts)

        # Supporting facts
        sp = find_supporting_facts(question, answer, titles, sentence_lists)

        answers[qid] = answer
        sp_facts[qid] = sp

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(input_data)}...", file=sys.stderr)

    return {"answer": answers, "sp": sp_facts}


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "input.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "pred.json"

    print(f"su-memory v2.0 — HotpotQA CodaLab Submission", file=sys.stderr)
    print(f"Input: {input_path}, Output: {output_path}", file=sys.stderr)

    with open(input_path, 'r') as f:
        input_data = json.load(f)
    print(f"  {len(input_data)} questions loaded", file=sys.stderr)

    predictions = run(input_data)

    with open(output_path, 'w') as f:
        json.dump(predictions, f)

    n_ans = len(predictions["answer"])
    print(f"Done. {n_ans} answers → {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
