import json
import argparse
import time
from typing import Dict, List, Tuple
import requests

def ap_at_k(rels: set, ranked_doc_ids: List[str], k: int) -> float:
    """Average Precision at K for one query."""
    if not rels:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for i, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in rels:
            hits += 1
            sum_prec += hits / i
    # normalize by min(#relevant, k) (common for AP@K)
    return sum_prec / min(len(rels), k)

def precision_at_k(rels: set, ranked_doc_ids: List[str], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = ranked_doc_ids[:k]
    return sum(1 for d in topk if d in rels) / k

def recall_at_k(rels: set, ranked_doc_ids: List[str], k: int) -> float:
    if not rels:
        return 0.0
    topk = ranked_doc_ids[:k]
    return sum(1 for d in topk if d in rels) / len(rels)

def fetch_search_results(base_url: str, query: str, timeout: float) -> Tuple[str, List[str]]:
    """
    Calls /search?query=...
    Expects JSON list: [("__time__", "..."), (doc_id, title), ...]
    Returns (server_time_str, doc_id_list)
    """
    r = requests.get(f"{base_url}/search", params={"query": query}, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    if not data:
        return ("", [])

    server_time_str = ""
    doc_ids = []

    # first item is ("__time__", "...") in your server
    first = data[0]
    if isinstance(first, (list, tuple)) and len(first) == 2 and first[0] == "__time__":
        server_time_str = str(first[1])

    # rest are (doc_id, title)
    for item in data[1:]:
        if isinstance(item, (list, tuple)) and len(item) >= 1:
            doc_ids.append(str(item[0]))

    return server_time_str, doc_ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="Path to queries_train.json")
    ap.add_argument("--base_url", default="http://127.0.0.1:8080", help="Search server base URL")
    ap.add_argument("--k", type=int, default=10, help="K for AP@K / P@K / R@K")
    ap.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout seconds")
    ap.add_argument("--max_queries", type=int, default=0, help="If >0, run only first N queries")
    args = ap.parse_args()

    with open(args.queries, "r", encoding="utf-8") as f:
        qrels: Dict[str, List[str]] = json.load(f)

    queries = list(qrels.items())
    if args.max_queries and args.max_queries > 0:
        queries = queries[:args.max_queries]

    per_query_rows = []
    ap_list = []
    p_list = []
    r_list = []
    client_times = []

    print(f"Running {len(queries)} queries against {args.base_url} (K={args.k})")
    print("-" * 80)

    for q, rel_doc_ids in queries:
        rels = set(map(str, rel_doc_ids))

        t0 = time.time()
        server_time_str, ranked = fetch_search_results(args.base_url, q, args.timeout)
        t_client = time.time() - t0

        apk = ap_at_k(rels, ranked, args.k)
        pk = precision_at_k(rels, ranked, args.k)
        rk = recall_at_k(rels, ranked, args.k)

        ap_list.append(apk)
        p_list.append(pk)
        r_list.append(rk)
        client_times.append(t_client)

        per_query_rows.append((q, apk, pk, rk, server_time_str, t_client, len(ranked)))

        print(f"q='{q[:40] + ('...' if len(q) > 40 else '')}' | "
              f"AP@{args.k}={apk:.3f} P@{args.k}={pk:.3f} R@{args.k}={rk:.3f} | "
              f"server={server_time_str} client={t_client:.3f}s results={len(ranked)}")

    mapk = sum(ap_list) / len(ap_list) if ap_list else 0.0
    mean_p = sum(p_list) / len(p_list) if p_list else 0.0
    mean_r = sum(r_list) / len(r_list) if r_list else 0.0
    mean_client = sum(client_times) / len(client_times) if client_times else 0.0

    print("-" * 80)
    print(f"MAP@{args.k} = {mapk:.4f}")
    print(f"Mean P@{args.k} = {mean_p:.4f}")
    print(f"Mean R@{args.k} = {mean_r:.4f}")
    print(f"Mean client time = {mean_client:.3f}s")
    print("-" * 80)

if __name__ == "__main__":
    main()
