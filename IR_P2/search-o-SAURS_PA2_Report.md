# Information Retrieval Programming Assignment II

## 1. Group details

- Group: `search-o-SAURS`
- Members: **Replace this line with the group member names and roll numbers before submission.**

## 2. Problem statement

The goal is ranked retrieval over the 1,400-document Cranfield aeronautics collection. The system repeats the Assignment 1 preprocessing, uses sparse vector-space retrieval, compares parameter settings, measures indexing and search time, and produces ranked results for evaluation with human relevance judgments.

## 3. Implementation details

`search-o-SAURS_PA2.py` reads the local group-prefixed corpus, query, stopword, and qrel files from `data/`. Document and query text use the Assignment 1 pipeline: tokenization, case and spelling normalization, stopword removal, and Porter stemming. Terms are not deduplicated because ranked retrieval requires term frequency.

PyTerrier/Terrier is configured with an empty term pipeline because preprocessing is performed explicitly. The Terrier index is written to `output/search-o-SAURS_index/`. Query identifiers are assigned ordinal IDs from 1 through 225 to match the Cranfield qrel convention.

For Terrier's TF-IDF weighting model, the document parameter $c$ was varied. Retrieval is sparse: only indexed query terms contribute to a document score. Results are written in six-column TREC format: query ID, `Q0`, document ID, rank, score, and run name.

## 4. Plan of experiments

Only sparse TF-IDF vector-space retrieval was used, as required by the assignment. The document parameter $c$ was tested at `0.25`, `0.5`, `0.75`, and `1.0`. The qrels contain 1,837 judgments for 225 queries. Evaluation metrics are MAP, nDCG, P@5, P@10, and P@20.

## 5. Results

### 5.1 Retrieval effectiveness

| Model | MAP | nDCG | P@5 | P@10 | P@20 |
|---|---:|---:|---:|---:|---:|
| TF-IDF, c=0.25 | 0.305104 | 0.506464 | 0.324444 | 0.229333 | 0.156889 |
| TF-IDF, c=0.50 | 0.314017 | 0.513593 | 0.331556 | 0.238222 | 0.160889 |
| **TF-IDF, c=0.75** | **0.320598** | **0.519036** | **0.332444** | **0.239556** | 0.164444 |
| TF-IDF, c=1.00 | 0.318271 | 0.516557 | 0.328000 | 0.237333 | **0.165111** |

The selected final run is `TF_IDF_c_0.75`. The complete ranked output is `output/search-o-SAURS_results.txt`, and the machine-readable metrics are in `output/search-o-SAURS_performance_metrics.csv`.

### 5.2 Efficiency

| Measurement | Value |
|---|---:|
| Documents | 1,400 |
| Queries | 225 |
| Relevance judgments | 1,837 |
| Indexing time | 3.87 s |
| Evaluation time for all models | 16.16 s |
| Search time for selected model | 3.05 s |

The timings were measured with PyTerrier/Terrier and are stored in `output/search-o-SAURS_timing.json`.

## 6. Discussion

The `c=0.75` configuration achieved the strongest MAP, nDCG, P@5, and P@10. The `c=1.0`
configuration achieved the strongest P@20, but its MAP and early precision were lower. This
suggests that moderate TF normalization was the best setting for ranking the most relevant
results near the top of this collection.

The implementation preserves term frequencies, applies identical preprocessing to documents and
queries, uses a sparse Terrier index, and produces deterministic experiment artifacts. Before
submission, replace the member placeholder and verify the required filename convention with the
course evaluation script.
