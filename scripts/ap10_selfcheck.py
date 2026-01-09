import json
import time
from typing import List, Set, Tuple, Any

import requests

BASE_URL = "http://127.0.0.1:8080"
TOP_K = 10

# אם יש לך queries_train.json באותה תיקייה - הסקריפט ינסה לטעון ממנו שאילתות
QUERIES_FALLBACK = [
    "python",
    "united states",
    "machine learning",
    "barack obama",
    "computer science",
    "israel",
]

def fetch(endpoint: str, query: str, timeout: float = 60.0) -> List[List[str]]:
    r = requests.get(f"{BASE_URL}{endpoint}", params={"query": query}, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise ValueError(f"{endpoint} returned non-list JSON")
    # strip optional __time__ row if present
    if len(data) > 0 and isinstance(data[0], list) and len(data[0]) == 2 and data[0][0] == "__time__":
        data = data[1:]
    return data

def doc_ids(pairs: List[List[str]], k: int = None) -> List[str]:
    if k is None:
        k = len(pairs)
    out = []
    for p in pairs[:k]:
        if isinstance(p, list) and len(p) == 2:
            out.append(str(p[0]))
    return out

def ap_at_k(ranked: List[str], relevant: Set[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    ranked = ranked[:k]
    hits = 0
    s = 0.0
    for i, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            hits += 1
            s += hits / i
    return s / min(len(relevant), k)

def load_queries_train(path: str = "queries_train.json", max_q: int = 50) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        # בפורמטים שונים: לפעמים זה list של dicts, לפעמים dict
        queries = []
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    q = item.get("query") or item.get("question") or item.get("text")
                    if isinstance(q, str) and q.strip():
                        queries.append(q.strip())
                elif isinstance(item, str) and item.strip():
                    queries.append(item.strip())
        elif isinstance(obj, dict):
            # אם יש מפתח כמו "queries"
            maybe = obj.get("queries")
            if isinstance(maybe, list):
                for q in maybe:
                    if isinstance(q, str) and q.strip():
                        queries.append(q.strip())
        if queries:
            return queries[:max_q]
    except Exception:
        pass
    return []

def main():
    # load queries
    queries = load_queries_train()
    if not queries:
        queries = QUERIES_FALLBACK

    print("===================================================")
    print(" AP@10 Self-Check (Pseudo-Qrels from title+anchor)")
    print(f" Target: {BASE_URL}")
    print(f" Queries: {len(queries)} | TOP_K={TOP_K}")
    print("===================================================")

    ap_scores = []
    t0 = time.time()

    for q in queries:
        # pseudo relevant set
        title_pairs = fetch("/search_title", q, timeout=120.0)
        anchor_pairs = fetch("/search_anchor", q, timeout=120.0)

        rel = set(doc_ids(title_pairs, k=200)) | set(doc_ids(anchor_pairs, k=200))

        # ranked from main search
        search_pairs = fetch("/search", q, timeout=120.0)
        ranked = doc_ids(search_pairs, k=TOP_K)

        ap = ap_at_k(ranked, rel, k=TOP_K)
        ap_scores.append(ap)

        print(f"q={q!r:25} | rel={len(rel):4d} | AP@10={ap:.3f}")

    mean_ap = sum(ap_scores) / len(ap_scores) if ap_scores else 0.0
    total = time.time() - t0

    print("---------------------------------------------------")
    print(f"Mean(AP@10) = {mean_ap:.3f}  (pseudo-qrels)")
    print(f"Total time  = {total:.2f}s")
    print("---------------------------------------------------")
    if mean_ap > 0.1:
        print("✅ PASS (sanity): Mean AP@10 > 0.1")
    else:
        print("⚠️ LOW (sanity): Mean AP@10 <= 0.1")
        print("Note: This is NOT the official course evaluation, only a self-check.")
    print("===================================================")

if __name__ == "__main__":
    main()
