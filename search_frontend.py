from flask import Flask, request, jsonify
import pickle

from inverted_index_gcp import InvertedIndex
from frontend_utils import tokenize, cossim, get_binary_score

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
INVERTED_INDEX_FILE_NAME = "index"

POSTINGS_TEXT_FOLDER_URL   = "postings_gcp_text"
POSTINGS_TITLE_FOLDER_URL  = "postings_gcp_title"
POSTINGS_ANCHOR_FOLDER_URL = "postings_gcp_anchor"

DL_PATH = "dl/dl.pkl"
NF_PATH = "nf/nf.pkl"

# -------------------------------------------------------------------
# Load indexes
# -------------------------------------------------------------------
inverted_index_body = InvertedIndex.read_index(POSTINGS_TEXT_FOLDER_URL, INVERTED_INDEX_FILE_NAME)
inverted_index_title = InvertedIndex.read_index(POSTINGS_TITLE_FOLDER_URL, INVERTED_INDEX_FILE_NAME)
inverted_index_anchor = InvertedIndex.read_index(POSTINGS_ANCHOR_FOLDER_URL, INVERTED_INDEX_FILE_NAME)

# DL / NF are required for cossim
with open(DL_PATH, "rb") as f:
    DL = pickle.load(f)
DL_LEN = len(DL)

with open(NF_PATH, "rb") as f:
    NF = pickle.load(f)

# -------------------------------------------------------------------
# Flask app
# -------------------------------------------------------------------
app = Flask(__name__)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False


@app.route("/search")
def search():
    res = []
    query = request.args.get("query", "")
    if not query:
        return jsonify(res)

    tokens = tokenize(query)  # no stemming, no expansion
    if not tokens:
        return jsonify(res)

    scored = cossim(tokens, inverted_index_body, POSTINGS_TEXT_FOLDER_URL, DL, DL_LEN, NF)
    top100 = scored[:100]
    return jsonify([(int(doc_id), "") for doc_id, _ in top100])


@app.route("/search_body")
def search_body():
    # Same as /search (TFIDF + cosine on body)
    res = []
    query = request.args.get("query", "")
    if not query:
        return jsonify(res)

    tokens = tokenize(query)
    if not tokens:
        return jsonify(res)

    scored = cossim(tokens, inverted_index_body, POSTINGS_TEXT_FOLDER_URL, DL, DL_LEN, NF)
    top100 = scored[:100]
    return jsonify([(int(doc_id), "") for doc_id, _ in top100])


@app.route("/search_title")
def search_title():
    res = []
    query = request.args.get("query", "")
    if not query:
        return jsonify(res)

    tokens = tokenize(query)
    if not tokens:
        return jsonify(res)

    scored = get_binary_score(tokens, inverted_index_title, POSTINGS_TITLE_FOLDER_URL)
    return jsonify([(int(doc_id), "") for doc_id, _ in scored])


@app.route("/search_anchor")
def search_anchor():
    res = []
    query = request.args.get("query", "")
    if not query:
        return jsonify(res)

    tokens = tokenize(query)
    if not tokens:
        return jsonify(res)

    scored = get_binary_score(tokens, inverted_index_anchor, POSTINGS_ANCHOR_FOLDER_URL)
    return jsonify([(int(doc_id), "") for doc_id, _ in scored])


@app.route("/get_pagerank", methods=["POST"])
def get_pagerank():
    wiki_ids = request.get_json() or []
    return jsonify([None for _ in wiki_ids])


@app.route("/get_pageview", methods=["POST"])
def get_pageview():
    wiki_ids = request.get_json() or []
    return jsonify([None for _ in wiki_ids])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
