# search_frontend.py  (MINIMAL + LIST OUTPUT + RUNTIME AS FIRST ITEM)
# Reads indices + posting_locs from GCS and serves minimal search API.
# Returns LIST of pairs like before, but with runtime as the first pair.

from flask import Flask, request, jsonify
import re
import math
import pickle
import time
import heapq
from collections import Counter, defaultdict
from google.cloud import storage
from inverted_index_gcp import InvertedIndex

###############################################################################
# CONFIG (YOUR GCS PATHS)
###############################################################################
BUCKET_NAME = "maayan-ir-bucket-2025"
INDEX_DIR = "postings_gcp_project"

BODY_INDEX_NAME = "index_body"
TITLE_INDEX_NAME = "index_title"
ANCHOR_INDEX_NAME = "index_anchor"

# Wikipedia size proxy for IDF stability (ok for minimal engine)
N_CORPUS = 6_300_000

# ---------------- Performance guards ----------------
MAX_CANDIDATES = 100_000          # cap on number of docs we score per query
MAX_POSTINGS_PER_TERM = 50_000    # safety cap per term while building candidates
MAX_RESULTS = 100                # results returned

###############################################################################
# Flask app
###############################################################################
app = Flask(__name__)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False

###############################################################################
# Tokenizer + Stopwords (Assignment 3 style)
###############################################################################
english_stopwords = frozenset([
    "a","about","above","after","again","against","all","am","an","and","any","are","as","at","be",
    "because","been","before","being","below","between","both","but","by","could","did","do","does",
    "doing","down","during","each","few","for","from","further","had","has","have","having","he","her",
    "here","hers","herself","him","himself","his","how","i","if","in","into","is","it","its","itself",
    "me","more","most","my","myself","no","nor","not","of","off","on","once","only","or","other",
    "ought","our","ours","ourselves","out","over","own","same","she","should","so","some","such","than",
    "that","the","their","theirs","them","themselves","then","there","these","they","this","those",
    "through","to","too","under","until","up","very","was","we","were","what","when","where","which",
    "while","who","whom","why","with","would","you","your","yours","yourself","yourselves"
])
corpus_stopwords = frozenset([
    "category","references","also","external","links","may","first","see","history","people","one","two",
    "part","thumb","including","second","following","many","however","would","became"
])
ALL_STOPWORDS = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

def tokenize(text: str):
    return [
        m.group().lower()
        for m in RE_WORD.finditer(text.lower())
        if m.group().lower() not in ALL_STOPWORDS
    ]

###############################################################################
# Globals
###############################################################################
gcs_client = None
gcs_bucket = None

body_index = None
title_index = None
anchor_index = None
id2title = None  # optional dict doc_id -> title

###############################################################################
# GCS helpers
###############################################################################
def gcs_init():
    global gcs_client, gcs_bucket
    if gcs_client is None:
        gcs_client = storage.Client()
        gcs_bucket = gcs_client.bucket(BUCKET_NAME)

def gcs_list(prefix: str):
    gcs_init()
    return [b.name for b in gcs_client.list_blobs(BUCKET_NAME, prefix=prefix)]

def gcs_load_pickle(blob_path: str):
    try:
        gcs_init()
        blob = gcs_bucket.blob(blob_path)
        with blob.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def normalize_fname(fname: str) -> str:
    # ensure filenames in posting_locs are just "body_6_003.bin" etc.
    if isinstance(fname, str) and fname.startswith(INDEX_DIR + "/"):
        return fname[len(INDEX_DIR) + 1:]
    return fname

def merge_posting_locs(component_prefix: str):
    """
    Merge gs://BUCKET/INDEX_DIR/{component_prefix}*_posting_locs.pickle
    component_prefix: 'body_' / 'title_' / 'anchor_'
    returns: dict(term -> list[(filename, offset)])
    """
    names = gcs_list(prefix=f"{INDEX_DIR}/{component_prefix}")
    loc_files = [n for n in names if n.endswith("_posting_locs.pickle")]

    merged = defaultdict(list)
    for lf in loc_files:
        d = gcs_load_pickle(lf)
        if not isinstance(d, dict):
            continue
        for term, locs in d.items():
            merged[term].extend([(normalize_fname(fn), off) for fn, off in locs])
    return dict(merged)

def read_posting_list(index_obj: InvertedIndex, term: str):
    # base_dir must be INDEX_DIR (filenames normalized)
    return index_obj.read_a_posting_list(INDEX_DIR, term, bucket_name=BUCKET_NAME)

###############################################################################
# Candidate generation (rare terms first) — key speed improvement
###############################################################################
def build_candidates(query_tokens, max_candidates=MAX_CANDIDATES):
    """
    Build a capped set of candidate doc_ids using BODY index only,
    starting from rare terms (small df first) to avoid huge unions.
    """
    if body_index is None or not query_tokens:
        return set()

    # collect (df, term) for terms existing in body index
    terms = []
    for t in set(query_tokens):
        df = body_index.df.get(t)
        if df:
            terms.append((df, t))
    terms.sort()  # rare first

    cand = set()
    for _df, term in terms:
        pl = read_posting_list(body_index, term)
        # safety cap per term
        for doc_id, _tf in pl[:MAX_POSTINGS_PER_TERM]:
            cand.add(doc_id)
            if len(cand) >= max_candidates:
                return cand
    return cand

###############################################################################
# Minimal scoring (restricted to candidates)
###############################################################################
def tfidf_body_scores(query_tokens, candidates=None):
    """
    Simple TF-IDF dot product restricted to candidates:
      score(doc) += (1+log10(tf_q))*idf * (1+log10(tf_d))*idf
    """
    scores = defaultdict(float)
    if body_index is None or not query_tokens:
        return scores

    cand_set = set(candidates) if candidates is not None else None

    q_tf = Counter(query_tokens)
    for term, qcnt in q_tf.items():
        df = body_index.df.get(term)
        if not df:
            continue

        idf = math.log10((N_CORPUS + 1.0) / (df + 1.0))
        wq = (1.0 + math.log10(qcnt)) * idf

        pl = read_posting_list(body_index, term)
        for doc_id, tf in pl:
            if cand_set is not None and doc_id not in cand_set:
                continue
            wd = (1.0 + math.log10(tf)) * idf
            scores[doc_id] += wq * wd

    return scores

def binary_match_count(index_obj: InvertedIndex, query_tokens, candidates=None):
    """
    Counts how many DISTINCT query tokens appear in doc (title/anchor),
    restricted to candidates if provided.
    """
    scores = defaultdict(int)
    if index_obj is None or not query_tokens:
        return scores

    cand_set = set(candidates) if candidates is not None else None

    for term in set(query_tokens):
        if term not in index_obj.df:
            continue
        pl = read_posting_list(index_obj, term)
        for doc_id, _ in pl:
            if cand_set is not None and doc_id not in cand_set:
                continue
            scores[doc_id] += 1
    return scores

def doc_title(doc_id: int):
    if isinstance(id2title, dict):
        return id2title.get(doc_id, str(doc_id))
    return str(doc_id)

def fmt_time(seconds: float) -> str:
    # show both seconds and ms to avoid confusion
    ms = int(round(seconds * 1000.0))
    return f"{seconds:.3f}s ({ms}ms)"

###############################################################################
# Startup
###############################################################################
@app.before_first_request
def startup():
    global body_index, title_index, anchor_index, id2title

    print("Loading indices from GCS...")
    body_index = InvertedIndex.read_index(INDEX_DIR, BODY_INDEX_NAME, bucket_name=BUCKET_NAME)
    title_index = InvertedIndex.read_index(INDEX_DIR, TITLE_INDEX_NAME, bucket_name=BUCKET_NAME)
    anchor_index = InvertedIndex.read_index(INDEX_DIR, ANCHOR_INDEX_NAME, bucket_name=BUCKET_NAME)
    print("Loaded index PKLs.")

    print("Merging posting_locs from GCS...")
    body_index.posting_locs = merge_posting_locs("body_")
    title_index.posting_locs = merge_posting_locs("title_")
    anchor_index.posting_locs = merge_posting_locs("anchor_")
    print(
        f"posting_locs merged: body={len(body_index.posting_locs)} "
        f"title={len(title_index.posting_locs)} anchor={len(anchor_index.posting_locs)}"
    )

    id2title = gcs_load_pickle(f"{INDEX_DIR}/id2title.pkl")
    if isinstance(id2title, dict):
        print("Loaded id2title.pkl")
    else:
        id2title = None
        print("id2title.pkl not found (ok).")

    print("Startup done.")

###############################################################################
# Required routes (LIST output like staff, with __time__ as first result)
###############################################################################
@app.route("/search")
def search():
    t_start = time.time()
    query = request.args.get("query", "")
    if not query:
        return jsonify([])

    tokens = tokenize(query)

    # minimal weights
    w_body, w_title, w_anchor = 1.0, 2.0, 1.5

    # -------- Build capped candidates first (rare terms first) --------
    t0 = time.time()
    candidates = build_candidates(tokens, MAX_CANDIDATES)
    t_cand = time.time() - t0

    # If no candidates from body (e.g., all OOV), fall back to empty
    if not candidates:
        total = time.time() - t_start
        res = [("__time__", fmt_time(total))]
        return jsonify(res)

    t0 = time.time()
    body_scores = tfidf_body_scores(tokens, candidates=candidates)
    t_body = time.time() - t0

    t0 = time.time()
    title_scores = binary_match_count(title_index, tokens, candidates=candidates)
    t_title = time.time() - t0

    t0 = time.time()
    anchor_scores = binary_match_count(anchor_index, tokens, candidates=candidates)
    t_anchor = time.time() - t0

    # compute final score only on candidates
    ranked = []
    for doc_id in candidates:
        s = (w_body * body_scores.get(doc_id, 0.0) +
             w_title * title_scores.get(doc_id, 0) +
             w_anchor * anchor_scores.get(doc_id, 0))
        if s > 0:
            ranked.append((doc_id, s))

    # Top-K without sorting everything
    top = heapq.nlargest(MAX_RESULTS, ranked, key=lambda x: x[1])

    total = time.time() - t_start

    print(
        f"[SEARCH TIMING] query='{query}' | total={total:.3f}s | "
        f"cand={t_cand:.3f}s | body={t_body:.3f}s | title={t_title:.3f}s | anchor={t_anchor:.3f}s | "
        f"cands={len(candidates)}"
    )

    res = [("__time__", fmt_time(total))]
    res.extend([(str(doc_id), doc_title(doc_id)) for doc_id, _ in top])
    return jsonify(res)

@app.route("/search_body")
def search_body():
    t_start = time.time()
    query = request.args.get("query", "")
    if not query:
        return jsonify([])

    tokens = tokenize(query)

    # Use same capped candidates for fairness + speed
    candidates = build_candidates(tokens, MAX_CANDIDATES)
    scores = tfidf_body_scores(tokens, candidates=candidates)
    ranked = heapq.nlargest(MAX_RESULTS, scores.items(), key=lambda x: x[1])

    total = time.time() - t_start
    print(f"[SEARCH_BODY TIMING] query='{query}' | total={total:.3f}s | cands={len(candidates)}")

    res = [("__time__", fmt_time(total))]
    res.extend([(str(doc_id), doc_title(doc_id)) for doc_id, _ in ranked])
    return jsonify(res)

@app.route("/search_title")
def search_title():
    t_start = time.time()
    query = request.args.get("query", "")
    if not query:
        return jsonify([])

    tokens = tokenize(query)

    candidates = build_candidates(tokens, MAX_CANDIDATES)
    scores = binary_match_count(title_index, tokens, candidates=candidates)

    ranked = heapq.nlargest(MAX_RESULTS, scores.items(), key=lambda x: (x[1], -int(x[0]) if str(x[0]).isdigit() else 0))

    total = time.time() - t_start
    print(f"[SEARCH_TITLE TIMING] query='{query}' | total={total:.3f}s | cands={len(candidates)}")

    res = [("__time__", fmt_time(total))]
    res.extend([(str(doc_id), doc_title(doc_id)) for doc_id, _ in ranked])
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    t_start = time.time()
    query = request.args.get("query", "")
    if not query:
        return jsonify([])

    tokens = tokenize(query)

    candidates = build_candidates(tokens, MAX_CANDIDATES)
    scores = binary_match_count(anchor_index, tokens, candidates=candidates)

    ranked = heapq.nlargest(MAX_RESULTS, scores.items(), key=lambda x: (x[1], -int(x[0]) if str(x[0]).isdigit() else 0))

    total = time.time() - t_start
    print(f"[SEARCH_ANCHOR TIMING] query='{query}' | total={total:.3f}s | cands={len(candidates)}")

    res = [("__time__", fmt_time(total))]
    res.extend([(str(doc_id), doc_title(doc_id)) for doc_id, _ in ranked])
    return jsonify(res)

@app.route("/get_pagerank", methods=["POST"])
def get_pagerank():
    wiki_ids = request.get_json() or []
    return jsonify([0.0 for _ in wiki_ids])

@app.route("/get_pageview", methods=["POST"])
def get_pageview():
    wiki_ids = request.get_json() or []
    return jsonify([0 for _ in wiki_ids])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
