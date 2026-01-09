import json
import re
import sys
import time
from typing import Any, List, Tuple

import requests

BASE_URL = "http://127.0.0.1:8080"
TIME_LIMIT_SECONDS = 35.0

# תחליפי/תוסיפי שאילתות אם בא לך
QUERIES = [
    "python",
    "united states",
    "computer science",
    "barack obama",
    "machine learning",
]

TIME_ROW_KEY = "__time__"
TIME_RE = re.compile(r"^\s*\d+(\.\d+)?s\s*\(\d+ms\)\s*$")

def fail(msg: str):
    print(f"❌ FAIL: {msg}")
    sys.exit(1)

def ok(msg: str):
    print(f"✅ {msg}")

def is_pair_list(x: Any) -> bool:
    if not isinstance(x, list):
        return False
    for item in x:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            return False
        if not isinstance(item[0], str) or not isinstance(item[1], str):
            return False
    return True

def get_json(path: str, query: str) -> Any:
    url = f"{BASE_URL}{path}"
    t0 = time.time()
    r = requests.get(url, params={"query": query}, timeout=TIME_LIMIT_SECONDS)
    dt = time.time() - t0
    if r.status_code != 200:
        fail(f"GET {path} status_code={r.status_code}, body={r.text[:200]}")
    try:
        data = r.json()
    except Exception:
        fail(f"GET {path} did not return JSON. First 200 chars: {r.text[:200]}")
    return data, dt

def post_json(path: str, payload: Any) -> Any:
    url = f"{BASE_URL}{path}"
    r = requests.post(url, json=payload, timeout=TIME_LIMIT_SECONDS)
    if r.status_code != 200:
        fail(f"POST {path} status_code={r.status_code}, body={r.text[:200]}")
    try:
        return r.json()
    except Exception:
        fail(f"POST {path} did not return JSON. First 200 chars: {r.text[:200]}")

def check_search_like(data: Any, require_time_row: bool, max_results: int | None):
    if not is_pair_list(data):
        fail("Response is not a list of [str,str] pairs.")

    if require_time_row:
        if len(data) == 0:
            fail("Empty response; expected __time__ as first element.")
        k, v = data[0]
        if k != TIME_ROW_KEY:
            fail(f"First row must be ['{TIME_ROW_KEY}', ...] but got {data[0]}")
        if not TIME_RE.match(v):
            fail(f"Time format must look like '0.385s (385ms)' but got '{v}'")

    # verify doc_id/title rows (skip time row)
    start = 1 if require_time_row else 0
    for i, (doc_id, title) in enumerate(data[start:start+5], start=start):
        if not doc_id.strip():
            fail(f"Empty doc_id at row {i}")
        if not title.strip():
            fail(f"Empty title at row {i}")

    if max_results is not None:
        # for /search and /search_body: should be <= 100 results (not counting time row)
        count_results = len(data) - (1 if require_time_row else 0)
        if count_results > max_results:
            fail(f"Expected at most {max_results} results but got {count_results}")

def main():
    print("===================================================")
    print(" Minimal IR Engine Self-Check")
    print(f" Target: {BASE_URL}")
    print("===================================================")

    # 1) /search
    for q in QUERIES[:2]:
        data, dt = get_json("/search", q)
        ok(f"GET /search?query={q!r} (HTTP 200) in {dt:.3f}s")
        if dt > TIME_LIMIT_SECONDS:
            fail(f"/search took {dt:.3f}s which exceeds {TIME_LIMIT_SECONDS}s")
        check_search_like(data, require_time_row=True, max_results=100)
        ok("/search format OK (time row + up to 100 results)")

    # 2) /search_body
    data, dt = get_json("/search_body", QUERIES[0])
    ok(f"GET /search_body (HTTP 200) in {dt:.3f}s")
    check_search_like(data, require_time_row=True, max_results=100)
    ok("/search_body format OK (time row + up to 100 results)")

    # 3) /search_title
    data, dt = get_json("/search_title", QUERIES[0])
    ok(f"GET /search_title (HTTP 200) in {dt:.3f}s")
    check_search_like(data, require_time_row=True, max_results=None)
    ok("/search_title format OK (time row + all results)")

    # 4) /search_anchor
    data, dt = get_json("/search_anchor", QUERIES[0])
    ok(f"GET /search_anchor (HTTP 200) in {dt:.3f}s")
    check_search_like(data, require_time_row=True, max_results=None)
    ok("/search_anchor format OK (time row + all results)")

    # 5) pagerank/pageview endpoints (minimal: zeros, but must match length)
    test_ids = ["1", "2", "12345", "999999"]
    pr = post_json("/get_pagerank", test_ids)
    pv = post_json("/get_pageview", test_ids)

    if not isinstance(pr, list) or len(pr) != len(test_ids):
        fail(f"/get_pagerank must return a list of same length as input. Got: {type(pr)} len={len(pr) if isinstance(pr,list) else 'N/A'}")
    if not isinstance(pv, list) or len(pv) != len(test_ids):
        fail(f"/get_pageview must return a list of same length as input. Got: {type(pv)} len={len(pv) if isinstance(pv,list) else 'N/A'}")

    ok("/get_pagerank + /get_pageview return correct-length lists")

    print("===================================================")
    print("🎉 ALL MINIMAL REQUIREMENTS LOOK GOOD!")
    print("===================================================")

if __name__ == "__main__":
    main()
