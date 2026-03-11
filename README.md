# Information Retrieval Search Engine

A Python-based search engine project developed as part of the **Information Retrieval course at Ben-Gurion University**.

This project implements a minimal Wikipedia search engine that supports indexing, tokenization, and document retrieval using multiple ranking signals.

## Authors

- Maayan Avraham
- Hila Sagi

## Project Overview

The engine indexes Wikipedia articles and supports search over multiple sources of evidence:

- **Body text** using TF-IDF / cosine similarity
- **Article titles** using binary term matching
- **Anchor text** using binary matching over incoming links

The goal was to build an efficient and functional search engine over a large-scale Wikipedia-based corpus while keeping the implementation relatively simple and interpretable.

## Main Features

- Parsing and indexing Wikipedia content
- Building inverted indexes for:
  - body
  - title
  - anchor text
- Query processing through multiple search endpoints:
  - `/search`
  - `/search_body`
  - `/search_title`
  - `/search_anchor`
- Retrieval based on combined ranking signals
- Deployment using Google Cloud Platform (GCP)

## Technologies

- Python
- Jupyter Notebook
- Flask
- Google Cloud Platform (GCP)
- Google Cloud Storage
- Inverted Index
- TF-IDF
- Cosine Similarity

## Data

The engine uses preprocessed Wikipedia data provided במסגרת הקורס, together with indexes stored on Google Cloud Storage.

## Evaluation Summary

The system was evaluated in three main aspects:

### 1. Functional Correctness
All endpoints were tested using self-check scripts to verify:

- correct HTTP responses
- valid output structure
- proper query handling
- compliance with the required API format

**Result:** all endpoints passed the correctness checks successfully.

### 2. Retrieval Quality
Retrieval quality was evaluated using **AP@10** over a predefined set of sanity queries.

- Required threshold: **AP@10 > 0.1**
- Achieved result: **Mean AP@10 ≈ 0.63–0.65**

This shows that even a simple engine combining TF-IDF body ranking with title and anchor signals can produce strong baseline results.

### 3. Runtime Performance
Runtime was measured using a dedicated evaluation script over 30 predefined queries.

- Mean client-side response time: **2.76 seconds**
- No query exceeded the required time limit
- Slow queries still completed in under 5 seconds after warm-up

## Version Comparison

We evaluated several logical versions of the system:

- **V1 – Body only**
- **V2 – Body + Title**
- **V3 – Body + Title + Anchor (Final)**

Findings:
- Adding title matching improved retrieval quality noticeably
- Adding anchor text provided additional gains
- Runtime remained stable across versions

## Qualitative Analysis

### Good Query: `machine learning`
This query returned highly relevant top results such as:

- Machine learning
- Boosting (machine learning)
- Unsupervised learning
- Journal of Machine Learning Research

**AP@10:** 0.790

### Bad Query: `united states`
This query is broad and ambiguous, so results were spread across many subtopics such as:

- United States Armed Forces
- History of the United States
- Constitution of the United States

**AP@10:** 0.473

This demonstrates one of the main limitations of a minimal lexical engine: broad ambiguous queries are harder to rank well.

## Key Findings

- A simple IR engine can already achieve reasonable retrieval quality
- Combining body, title, and anchor signals improves performance
- Query ambiguity remains a major challenge
- Loading index metadata at startup improves response time significantly

## Future Improvements

Possible next steps include:

- integrating PageRank
- using page views as a ranking signal
- improving ranking with BM25
- handling query ambiguity more effectively

## Repository Structure

- `search_frontend.py` – search server
- `inverted_index_gcp.py` – GCP-based inverted index logic
- `eval_queries.py` – runtime and retrieval evaluation
- notebooks / utilities for indexing and experiments

## Notes

This repository contains the project code and documentation.  
Large index files are stored externally on Google Cloud Storage rather than directly in the repository.
