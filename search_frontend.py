from flask import Flask, request, jsonify, render_template
import gzip
import math
import pickle
import pandas as pd


# Staff modules (must be available on the machine running the server)
from inverted_index_gcp import InvertedIndex
from frontend_utils import (
    tokenize,          # your tokenizer (should remove stopwords)
    BM25,              # your BM25 scoring helper
    cossim,            # your cosine similarity scoring helper
    get_binary_score,  # title/anchor: count distinct query words
)


###############################################################################
# CONFIG — UPDATE THESE TO MATCH YOUR BUCKET LAYOUT
###############################################################################

BUCKET_NAME = "maayan-ir-bucket-2025"

INVERTED_INDEX_FILE_NAME = "index"  # usually index.pkl is produced by staff code

# Posting folders (inside the bucket)
POSTINGS_TEXT_FOLDER_URL   = "postings_gcp_text"
POSTINGS_TITLE_FOLDER_URL  = "postings_gcp_title"
POSTINGS_ANCHOR_FOLDER_URL = "postings_gcp_anchor"

# Optional: if you don't have these, keep None and the code will degrade gracefully
DL_PATH = "dl/dl.pkl"        # doc lengths dict
NF_PATH = "nf/nf.pkl"        # norm factors (for cosine)
DT_PATH = "dt/dt.pkl"        # doc_id -> title

inverted_index_body = InvertedIndex.read_index(POSTINGS_TEXT_FOLDER, INVERTED_INDEX_FILE_NAME)
inverted_index_title = InvertedIndex.read_index(POSTINGS_TITLE_FOLDER, INVERTED_INDEX_FILE_NAME)
inverted_index_anchor = InvertedIndex.read_index(POSTINGS_ANCHOR_FOLDER, INVERTED_INDEX_FILE_NAME)

# Load DL / NF (נדרש ל-cossim)
with open(DL_PATH, "rb") as f:
    DL = pickle.load(f)
DL_LEN = len(DL)

with open(NF_PATH, "rb") as f:
    NF = pickle.load(f)

# Load doc titles (אם אין – לא מפילים את השרת)
DT = {}
if os.path.exists(DT_PATH):
    with open(DT_PATH, "rb") as f:
        DT = pickle.load(f)


# -----------------------------
# FLASK APP
# -----------------------------

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


def _id_to_title(doc_id: int) -> str:
    # מחזיר כותרת אם קיימת, אחרת string ריק כדי לא לשבור API
    return DT.get(doc_id, "")



@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)

    tokens = tokenize(query)  # בלי stemming, בלי expansion
    if not tokens:
        return jsonify(res)

    # Cosine similarity on BODY
    scored = cossim(tokens, inverted_index_body, POSTINGS_TEXT_FOLDER, DL, DL_LEN, NF)
    top100 = scored[:100]

    res = [(doc_id, _id_to_title(doc_id)) for doc_id, _score in top100]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    tokens = tokenize(query)
    if not tokens:
        return jsonify(res)

    scored = cossim(tokens, inverted_index_body, POSTINGS_TEXT_FOLDER, DL, DL_LEN, NF)
    top100 = scored[:100]

    res = [(doc_id, _id_to_title(doc_id)) for doc_id, _score in top100]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    tokens = tokenize(query)
    if not tokens:
        return jsonify(res)

    scored = get_binary_score(tokens, inverted_index_title, POSTINGS_TITLE_FOLDER)  # מחזיר ALL
    res = [(doc_id, _id_to_title(doc_id)) for doc_id, _score in scored]

    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    tokens = tokenize(query)
    if not tokens:
        return jsonify(res)

    scored = get_binary_score(tokens, inverted_index_anchor, POSTINGS_ANCHOR_FOLDER)
    res = [(doc_id, _id_to_title(doc_id)) for doc_id, _score in scored]
    
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    wiki_ids = request.get_json() or []
    return jsonify([None for _ in wiki_ids])

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    wiki_ids = request.get_json() or []
    return jsonify([None for _ in wiki_ids])

def run(**options):
    app.run(**options)

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
