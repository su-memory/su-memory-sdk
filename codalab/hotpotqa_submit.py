#!/usr/bin/env python3
"""
HotpotQA CodaLab Submission Script
Reads test data, runs su-memory + DeepSeek, outputs predictions.
"""
import json, sys, os, math, re, requests
from collections import defaultdict

# DeepSeek API key — set via env on CodaLab
DS_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# su-memory imports
from su_memory.sdk.embedding import OllamaEmbedding

# ⚠️ CodaLab has no GPU/Ollama. Use local embedding fallback.
# If Ollama available, use it. Otherwise hash-based fallback.

class FastRetriever:
    def __init__(self):
        self.docs = []
        self.index = defaultdict(set)
        try:
            self.emb = OllamaEmbedding()
            self.has_vector = True
        except:
            self.has_vector = False
    
    def add(self, text, doc_id=None):
        self.docs.append({"text": text, "id": doc_id or f"d{len(self.docs)}"})
        idx = len(self.docs) - 1
        for word in re.findall(r'[a-z]{3,}', text.lower()):
            self.index[word].add(idx)
    
    def query(self, q, k=10):
        qwords = set(re.findall(r'[a-z]{3,}', q.lower()))
        scores = defaultdict(float)
        n = max(len(self.docs), 1)
        for w in qwords:
            for i in self.index.get(w, set()):
                scores[i] += math.log(n / (len(self.index[w]) + 1)) + 1
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [self.docs[i] for i, s in ranked[:k]]

def deepseek_answer(q, ctxs):
    if not DS_KEY: return ""
    ctx = "\n---\n".join(c[:500] for c in ctxs[:10])[:4000]
    try:
        r = requests.post("https://api.deepseek.com/v1/chat/completions", headers={
            "Authorization": f"Bearer {DS_KEY}",
            "Content-Type": "application/json"
        }, json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": f"Answer briefly based on facts.\n\nFacts:\n{ctx}\n\nQuestion: {q}\n\nAnswer:"}],
            "max_tokens": 20, "temperature": 0
        }, timeout=15)
        return r.json()["choices"][0]["message"]["content"].strip()
    except:
        return ""

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    
    with open(args.input) as f:
        data = json.load(f)
    
    retriever = FastRetriever()
    predictions = {}
    
    for item in data:
        qid = item["_id"]
        question = item["question"]
        
        # Index context
        ctx = item.get("context", {})
        sentences = []
        for doc_sents in ctx.get("sentences", []):
            for s in doc_sents:
                if s and len(s) > 10:
                    retriever.add(s)
                    sentences.append(s)
        
        # Retrieve
        results = retriever.query(question, k=10)
        ctxs = [r["text"] for r in results]
        
        # Answer extraction
        answer = None
        for c in ctxs:
            # Try to find answer span
            pass  # DeepSeek handles this
        
        if DS_KEY:
            answer = deepseek_answer(question, ctxs)
        
        predictions[qid] = answer or ""
    
    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Generated {len(predictions)} predictions")

if __name__ == "__main__":
    main()
