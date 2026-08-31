
```
 ██████╗███████╗ █████╗ ██████╗  ██████╗██╗  ██╗      ██████╗      ███████╗ █████╗ ██╗   ██╗██████╗ ███████╗
██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██║  ██║     ██╔═══██╗     ██╔════╝██╔══██╗██║   ██║██╔══██╗██╔════╝
███████╗█████╗  ███████║██████╔╝██║     ███████║     ██║   ██║     ███████╗███████║██║   ██║██████╔╝███████╗
╚════██║██╔══╝  ██╔══██║██╔══██╗██║     ██╔══██║     ██║   ██║     ╚════██║██╔══██║██║   ██║██╔══██╗╚════██║
███████║███████╗██║  ██║██║  ██║╚██████╗██║  ██║     ╚██████╔╝     ███████║██║  ██║╚██████╔╝██║  ██║███████║
╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝      ╚═════╝      ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝
```



**A Boolean Information Retrieval System for the Cranfield Aeronautics Corpus**

Built as part of Programming Assignment 1 for the Information Retrieval course.

---



## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Pipeline Architecture](#pipeline-architecture)
- [File Descriptions](#file-descriptions)
- [Usage](#usage)
- [Design Decisions](#design-decisions)
- [Optimizations](#optimizations)
- [Testing](#testing)

---



## Overview

search-o-SAURS is a complete Boolean retrieval system that processes the **Cranfield 1400** collection (1,400 aeronautics research abstracts) and supports **AND / OR** queries. The system implements:

- **Multi-stage tokenization** (8 stages) tailored for scientific text
- **British → American spelling normalization** (5 patterns with exception guards)
- **Porter stemming** with the canonical reference implementation
- **Stop word removal** using a curated stop word list
- **Inverted index** with document frequency, sorted lexicographically
- **Binary search** directly on the index file (O(log V) seeks)
- **Adaptive AND merge** — two-pointer for balanced lists, galloping for skewed lists

---



## Project Structure

```
IR_P1/
├── search-o-SAURS_preprocess.py   # Preprocessing pipeline (tokenize, normalize, stem, etc.)
├── search-o-SAURS_indexer.py      # Inverted index builder
├── search-o-SAURS_search.py       # Boolean search engine (binary search + merge)
├── search-o-SAURS_pipeline.ipynb  # Complete pipeline notebook (self-contained)
├── porter.py                      # Porter Stemmer (canonical implementation)
├── README.md                      # This file
├── .gitignore                     # Git ignore rules
│
├── data/                          # Input data and resources
│   ├── cran.all.1400              # Cranfield corpus (input)
│   ├── stopwords.txt              # Stop word list (358 words)
│   └── PA1.pdf                    # Assignment specification
│
├── output/                        # Generated files and results
│   ├── search-o-SAURS_processed.all   # Preprocessed corpus (generated)
│   ├── search-o-SAURS_cran.index      # Inverted index file (generated)
│   └── results/                       # Query result files
│       ├── search-o-SAURS_results_aerodynamic_AND_experimental.txt
│       ├── search-o-SAURS_results_shock_AND_wave.txt
│       └── ... (other result files)
│
└── tests/                         # Testing suite
    └── test_search.py             # Comprehensive test suite (220 tests)
```

---



## Quick Start



### Prerequisites

- Python 3.8+ (no external libraries required — fully self-contained)



### Run the Full Pipeline

#### Option A: One-Command Execution (Recommended)
You can execute the entire pipeline (Preprocessing → Indexing → Search) using the provided helper script. The script allows you to enter your query interactively, or pass it as an argument. Testing is optional and can be enabled via the `--test` flag:
```bash
chmod +x run_pipeline.sh

# Interactive query input, no tests
./run_pipeline.sh

# Provide query as an argument
./run_pipeline.sh "aerodynamic AND experimental"

# Run with tests enabled
./run_pipeline.sh "aerodynamic AND experimental" --test
```

#### Option B: Step-by-Step Execution
```bash
# Step 1: Preprocess the corpus
python search-o-SAURS_preprocess.py

# Step 2: Build the inverted index
python search-o-SAURS_indexer.py

# Step 3: Run a Boolean query
python search-o-SAURS_search.py "aerodynamic AND experimental" output/results/output.txt

# Step 4: Run tests (optional)
python tests/test_search.py
```



### Or Use the Notebook

Open `search-o-SAURS_pipeline.ipynb` in Jupyter — it runs the complete pipeline in a single notebook with an interactive query cell.

---



## Pipeline Architecture

```
                    ┌──────────────────────────────────────────────┐
  cran.all.1400 ──► │  PREPROCESSING  (search-o-SAURS_preprocess.py) │
                    │                                              │
                    │  1. Tokenize    (8-stage tokenizer)          │
                    │  2. Normalize   (case fold + British→American)│
                    │  3. Stop Words  (remove 358 stop words)      │
                    │  4. Stem        (Porter stemmer)              │
                    │  5. Deduplicate (Boolean: presence only)      │
                    │                                              │
                    └────────────────┬─────────────────────────────┘
                                     │
                                     ▼
                    ┌──────────────────────────────────────────────┐
                    │  INDEXING  (search-o-SAURS_indexer.py)        │
                    │                                              │
                    │  Build sorted inverted index with df          │
                    │  Format: token df docid1,docid2,...           │
                    │                                              │
                    └────────────────┬─────────────────────────────┘
                                     │
                                     ▼
                      search-o-SAURS_cran.index
                      (4,183 terms, lexicographically sorted)
                                     │
                                     ▼
                    ┌──────────────────────────────────────────────┐
  "term1 AND term2" │  SEARCH  (search-o-SAURS_search.py)          │
         ──────────►│                                              │
                    │  1. Preprocess query terms (same pipeline)    │
                    │  2. Binary search on index file (O(log V))   │
                    │  3. AND: two-pointer or galloping merge      │
                    │     OR:  two-pointer merge                   │
                    │                                              │
                    └────────────────┬─────────────────────────────┘
                                     │
                                     ▼
                               results/output.txt
                         (matching document IDs, one per line)
```

---



## File Descriptions



### Core Pipeline Files


| File                           | Lines | Purpose                                                                                                                                                                                                                                                                                                                                                                                         |
| ------------------------------ | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `search-o-SAURS_preprocess.py` | 454   | **Preprocessing pipeline.** Parses the Cranfield corpus, then applies a 5-stage pipeline: Tokenize (8-stage tokenizer handling abbreviations, possessives, hyphens, slashes, number-word splits) → Normalize (case fold + 5-pattern British→American spelling equivalence with exception guards) → Stop word removal → Porter stemming → Deduplication. Outputs `search-o-SAURS_processed.all`. |
| `search-o-SAURS_indexer.py`    | 69    | **Index builder.** Reads the preprocessed file, builds an inverted index as a sorted flat file. Each line: `token df docid1,docid2,...`. First line: `vocab_size, max_docid`. The df (document frequency) field enables search-time optimizations.                                                                                                                                              |
| `search-o-SAURS_search.py`     | 435   | **Boolean search engine.** Implements binary search directly on the sorted index file (O(log V) seeks via buffered backward scan in binary mode). For AND queries, adaptively selects between two-pointer merge (O(n+m)) and galloping search (O(k·log(n/k))) based on postings list size ratio. Query terms are preprocessed with the identical normalization+stemming pipeline as documents.  |




### Supporting Files


| File                            | Purpose                                                                                                                   |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `porter.py`                     | Canonical Porter Stemmer implementation (Vivake Gupta, from tartarus.org). Used unchanged as permitted by the assignment. |
| `stopwords.txt`                 | 358 stop words used for filtering (in `data/`).                                                                           |
| `cran.all.1400`                 | The Cranfield collection (in `data/`).                                                                                    |
| `search-o-SAURS_pipeline.ipynb` | Self-contained Jupyter notebook that runs the entire pipeline.                                                            |
| `test_search.py`                | Comprehensive test suite with **220 tests** (in `tests/`).                                                                |




### Generated Files


| File                                  | Generated By    | Contents                                                                                                             |
| ------------------------------------- | --------------- | -------------------------------------------------------------------------------------------------------------------- |
| `output/search-o-SAURS_processed.all` | `preprocess.py` | Preprocessed corpus. Format: `.I docid` + `.S` + space-separated stems. 76,863 unique tokens across 1,400 documents. |
| `output/search-o-SAURS_cran.index`    | `indexer.py`    | Inverted index. 4,183 terms sorted lexicographically with document frequencies and ascending postings lists.         |
| `output/results/*.txt`                | `search.py`     | Query result files. Each contains matching document IDs, one per line, in ascending order.                           |


---



## Usage



### Running a Query

```bash
python search-o-SAURS_search.py "<QUERY>" <OUTPUT_FILE> [INDEX_FILE]
```

**Arguments:**

- `<QUERY>` — A Boolean query: `"term1 AND term2"` or `"term1 OR term2"`
- `<OUTPUT_FILE>` — Path to write result document IDs
- `[INDEX_FILE]` — Optional, defaults to `search-o-SAURS_cran.index`

**Examples:**

```bash
# AND query
python search-o-SAURS_search.py "aerodynamic AND experimental" output/results/output.txt

# OR query
python search-o-SAURS_search.py "boundary OR layer" output/results/output.txt

# British spelling (automatically normalized to American)
python search-o-SAURS_search.py "behaviour AND flow" output/results/output.txt
```

**Sample output:**

```
Original Query: 'aerodynamic AND experimental'
Parsed Terms: 'aerodynam' AND 'experiment'
Searching index file: output\search-o-SAURS_cran.index
  Index contains 4,183 terms, max docid = 1400
  [Binary Search] Found 'aerodynam' (df=179) in 9 seeks
  [Binary Search] Found 'experiment' (df=339) in 7 seeks
Postings for 'aerodynam' (size 179): [1, 5, 11, 13, 14, 29, 32, 33, 36, 44]...
Postings for 'experiment' (size 339): [1, 11, 12, 16, 17, 19, 25, 29, 30, 35]...
  [AND] Algorithm: two-pointer | Lists: 179 vs 339 (ratio 1.9x) | O(179+339)
Match count: 49
Results written to output/results/output.txt
```

---



## Design Decisions



### 1. Pipeline Ordering

The assignment lists the preprocessing steps as: Tokenization → Stemming → Stop word removal → Normalization.

Our execution order is: **Tokenize → Normalize → Stop word removal → Stem → Deduplicate**.

This reordering is **necessary for correctness** due to hard logical dependencies:


| Constraint                       | Reason                                                                                                                                         |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Normalize **before** Stem        | The Porter stemmer only recognizes lowercase vowels (`a,e,i,o,u`). Uppercase input produces incorrect stems: `stem("BOUNDARY")` = `"BOUNDARY"` |
| British→American **before** Stem | Without normalization: `stem("behaviour")` ≠ `stem("behavior")` — they diverge, fragmenting the index                                          |
| Stop words **before** Stem       | 7 stop words (`are→ar`, `has→ha`, `this→thi`, etc.) survive removal after stemming because their stemmed forms don't match the stop word list  |


All four preprocessing functions specified in the assignment are implemented as separate, documented functions.

### 2. British → American Spelling Normalization

The Cranfield corpus mixes British and American authored papers. Without normalization, the Porter stemmer produces different stems for the same concept:


| British    | American   | Stem (without normalization) | Stem (with normalization) |
| ---------- | ---------- | ---------------------------- | ------------------------- |
| behaviour  | behavior   | `behaviour` ≠ `behavior`     | `behavior` ✓              |
| linearised | linearized | `linearis` ≠ `linear`        | `linear` ✓                |
| centre     | center     | `centr` ≠ `center`           | `center` ✓                |
| vapour     | vapor      | `vapour` ≠ `vapor`           | `vapor` ✓                 |


Five rule-based patterns are applied:

- **A:** `-ise`/`-ised`/`-ising`/`-isation` → `-ize`/`-ized`/`-izing`/`-ization`
- **B:** `-our` → `-or`
- **C:** `-tre`/`-bre` → `-ter`/`-ber`
- **D:** `-ogue` → `-og`
- **E:** `-mme` → `-m`

Each pattern has a curated exception set to prevent false conversions (e.g., `noise`, `rise`, `four`, `contour`, `vogue`).

### 3. Token Deduplication

In Boolean retrieval, only term **presence** matters, not frequency. Deduplication achieves a **41.2% token reduction** (130,757 → 76,863 tokens) — reducing index size and search time without affecting retrieval accuracy.

### 4. Query-Document Parity

The query preprocessing pipeline (`preprocess_query_term()` in search.py) applies the **exact same** normalization rules as the document pipeline. This guarantees that a query for `"behaviour"` is normalized to `"behavior"` → stemmed to `"behavior"` — matching the indexed form. This parity is verified by 26 dedicated tests.

---



## Optimizations



### Binary Search on Index File

Instead of loading all 4,183 index entries into memory (O(V) space), we perform **binary search by byte position** directly on the sorted file — finding any term in **O(log V) ≈ 12 seeks**.

Key implementation details:

- **Binary mode (**`rb`**)** for precise byte-level seeking (text mode on Windows breaks `seek()`/`tell()` due to `\r\n` translation)
- **Buffered backward scan** (256-byte chunks) to find line boundaries, handling lines of any length



### Adaptive AND Merge (Galloping Search)

For AND queries, the algorithm adapts based on postings list size ratio:


| Ratio | Algorithm   | Complexity    | Example                                       |
| ----- | ----------- | ------------- | --------------------------------------------- |
| ≤ 10x | Two-pointer | O(n + m)      | `shock(240) AND wave(210)` → O(450)           |
| > 10x | Galloping   | O(k·log(n/k)) | `ab(1) AND flow(730)` → O(1·log(730)) ≈ O(10) |


Galloping search uses exponential jumps (1, 2, 4, 8, ...) followed by binary search in the last interval, giving O(log d) per element where d is the distance to the match.

### Document Frequency in Index

Each index line includes the document frequency: `token df docid1,docid2,...`. This enables the search engine to know postings list sizes without parsing them — used for query term ordering and skew detection.

---



## Testing

Run the comprehensive test suite:

```bash
python3 tests/test_search.py
```



### Test Coverage: 220 tests across 12 groups


| Group                | Tests | What It Validates                                                                    |
| -------------------- | ----- | ------------------------------------------------------------------------------------ |
| 1. Tokenization      | 15    | Abbreviations, possessives, hyphens, slashes, number-word splits, noise, edge cases  |
| 2. Normalization     | 25    | Case folding, all 5 British→American patterns, all exception guards, min-length      |
| 3. Stop Words        | 4     | Removal correctness, empty input, all-stop-words                                     |
| 4. Stemming          | 9     | Porter stemmer on key corpus terms                                                   |
| 5. Deduplication     | 4     | Order preservation, identity, all-same, empty                                        |
| 6. Query↔Doc Parity  | 26    | Exact normalization+stemming match between query and document pipelines              |
| 7. Index Correctness | 8     | Header, lexicographic sort, df accuracy, sorted postings, no duplicates, spot checks |
| 8. Binary Search     | 55    | Stratified sample across alphabet + boundary + nonexistent terms                     |
| 9. AND/OR Merges     | 24    | Gallop search unit tests, AND/OR vs set operations, edge cases                       |
| 10. End-to-End       | 13    | Raw queries → correct results, British spelling queries, missing terms               |
| 11. Spot Checks      | 8     | Specific document membership, British≡American equivalence, commutativity            |
| 12. Invariants       | 30    | `|A∩B| ≤ min(|A|,|B|)`, `|A∪B| ≥ max(|A|,|B|)`, inclusion-exclusion, membership      |


---



## Statistics


| Metric                             | Value                  |
| ---------------------------------- | ---------------------- |
| Documents in corpus                | 1,400                  |
| Stop words                         | 358                    |
| Vocabulary size (indexed terms)    | 4,183                  |
| Total tokens before dedup          | 130,757                |
| Total tokens after dedup           | 76,863                 |
| Deduplication savings              | 41.2%                  |
| Binary search seeks per query term | 7–14 (vs 4,183 linear) |
| Test coverage                      | 220 tests, 12 groups   |

