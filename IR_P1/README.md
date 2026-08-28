<div align="center">

<br/>

```
 ██████╗███████╗ █████╗ ██████╗  ██████╗██╗  ██╗      ██████╗      ███████╗ █████╗ ██╗   ██╗██████╗ ███████╗
██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██║  ██║     ██╔═══██╗     ██╔════╝██╔══██╗██║   ██║██╔══██╗██╔════╝
███████╗█████╗  ███████║██████╔╝██║     ███████║     ██║   ██║     ███████╗███████║██║   ██║██████╔╝███████╗
╚════██║██╔══╝  ██╔══██║██╔══██╗██║     ██╔══██║     ██║   ██║     ╚════██║██╔══██║██║   ██║██╔══██╗╚════██║
███████║███████╗██║  ██║██║  ██║╚██████╗██║  ██║     ╚██████╔╝     ███████║██║  ██║╚██████╔╝██║  ██║███████║
╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝      ╚═════╝      ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝
```

<h1>🦕 search-o-SAURS</h1>

<p><em>A prehistoric-powerful Information Retrieval engine — built from scratch.</em></p>

<br/>

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Cranfield_1400-FF6B6B?style=for-the-badge&logo=databricks&logoColor=white)](http://ir.dcs.gla.ac.uk/resources/test_collections/cran/)
[![Algorithm](https://img.shields.io/badge/Stemmer-Porter_Algorithm-4ECDC4?style=for-the-badge&logo=algorithmia&logoColor=white)](https://tartarus.org/martin/PorterStemmer/)
[![Model](https://img.shields.io/badge/Retrieval-Boolean_Model-FFE66D?style=for-the-badge&logo=searchengineland&logoColor=black)](https://en.wikipedia.org/wiki/Boolean_model_of_information_retrieval)
[![Assignment](https://img.shields.io/badge/IR-Programming_Assignment_1-A8E6CF?style=for-the-badge&logo=academia&logoColor=black)]()

<br/>

> **PA1 · Information Retrieval · IIT Kharagpur**
> _Preprocessing · Indexing · Boolean Search over the Cranfield Collection_

---

</div>

<br/>

## 📖 Table of Contents

- [🌍 Overview](#-overview)
- [🗂️ Project Structure](#️-project-structure)
- [⚙️ The Pipeline](#️-the-pipeline)
  - [Stage 1 — Preprocessing](#stage-1--preprocessing)
  - [Stage 2 — Indexing](#stage-2--indexing)
  - [Stage 3 — Boolean Search](#stage-3--boolean-search)
- [🚀 How to Run](#-how-to-run)
- [📂 File Formats](#-file-formats)
- [📊 Stats & Results](#-stats--results)
- [🧠 Algorithm Deep Dive](#-algorithm-deep-dive)

---

## 🌍 Overview

**search-o-SAURS** is a fully hand-crafted, from-scratch Information Retrieval (IR) system. No IR libraries. No shortcuts. Just raw NLP logic, a classic inverted index, and lightning-fast set-algebra for Boolean queries — all operating over the **Cranfield collection** of 1,400 scientific aeronautics abstracts.

This was built as part of **Programming Assignment 1** for the Information Retrieval course. The system implements the complete classical IR pipeline:

```
Raw Text ──► Tokenization ──► Normalization ──► Stop Word Removal ──► Stemming
                                                                          │
                                                                          ▼
                                                                  Inverted Index
                                                                          │
                                                                          ▼
                                                              Boolean Query Engine
                                                           (AND / OR  ·  O(n) time)
```

---

## 🗂️ Project Structure

```
IR_P1/
│
├── 📄 PA1.pdf                          ← Assignment specification
│
├── 🗃️ cran.all.1400                    ← Raw Cranfield dataset (1400 docs)
├── 🚫 stopwords.txt                    ← English stopwords list
│
├── 🔬 search-o-SAURS_preprocess.py     ← Stage 1: NLP preprocessing pipeline
├── 🗂️  search-o-SAURS_indexer.py       ← Stage 2: Inverted index builder
├── 🔍 search-o-SAURS_search.py         ← Stage 3: Boolean query engine
│
├── 📦 porter.py                        ← Porter Stemmer (1980 algorithm)
│
├── 🗃️  search-o-SAURS_processed.all    ← OUTPUT: Preprocessed token file
├── 📇 search-o-SAURS_cran.index        ← OUTPUT: Inverted index file
│
└── 📋 all_results.txt                  ← OUTPUT: Unified incremental query log
```

---

## ⚙️ The Pipeline

### Stage 1 — Preprocessing

> **Script:** `search-o-SAURS_preprocess.py`

The preprocessor parses the raw **Cranfield SGML-like format** and applies four sequential NLP transformations.

#### 🔤 Cranfield Format Parser
The dataset uses tagged fields. Only **Title (`.T`)** and **Abstract (`.W`)** are extracted — author (`.A`) and bibliography (`.B`) fields are intentionally ignored per the assignment spec.

```
.I 1          ← Document ID
.T            ← Title field (USED ✅)
experimental investigation of the aerodynamics of a wing in a slipstream
.A            ← Authors (IGNORED ❌)
bray, j. r.
.B            ← Bibliography (IGNORED ❌)
j. ae. soc. 65, 1961, 552.
.W            ← Abstract (USED ✅)
an experimental study...
```

#### 🔄 Four NLP Transformations (in order)

| Step | Function | What it does |
|------|----------|-------------|
| **1. Tokenization** | `tokenize(text)` | Splits on non-alphanumeric chars using regex `[a-zA-Z0-9]+`. Extracts only clean word/number tokens. |
| **2. Normalization** | `normalize(tokens)` | Converts all tokens to **lowercase**. `Aerodynamic` → `aerodynamic` |
| **3. Stop Word Removal** | `remove_stopwords(tokens, stopwords)` | Filters tokens present in `stopwords.txt` (e.g., "the", "a", "is", "of") |
| **4. Stemming** | `stem(tokens)` | Applies **Porter Stemmer** to reduce words to their stems. `experimental` → `experiment` |

#### 📤 Output Format: `search-o-SAURS_processed.all`

```
.I 1
.S
experiment investig aerodynam wing slipstream ...
.I 2
.S
effect freestram veloc point vortex ...
```

---

### Stage 2 — Indexing

> **Script:** `search-o-SAURS_indexer.py`

Reads the preprocessed `.all` file and constructs an **inverted index** — the backbone of all fast IR systems.

```
Token ──► { doc_id₁, doc_id₂, doc_id₃, ... }  (sorted ascending)
```

**Key design decisions:**
- Tokens are stored in **lexicographical (alphabetical) order**
- Postings lists contain **comma-separated doc IDs in ascending order** with no duplicates
- The first line stores metadata: `vocabulary_size, max_docid`

#### 📤 Output Format: `search-o-SAURS_cran.index`

```
4615, 1400
aerodynam 1,10,11,29,52,...
experiment 1,7,9,21,27,...
slipstream 1,532,...
```

---

### Stage 3 — Boolean Search

> **Script:** `search-o-SAURS_search.py`

A query engine that operates directly over the inverted index. Supports **two-term Boolean queries** with `AND` or `OR` connectives.

#### 🔍 Query Processing Flow

```
Input Query String
       │
       ▼
   Parse into:  TERM₁  OPERATOR  TERM₂
       │
       ▼
 Preprocess each term (lowercase → strip punctuation → Porter Stem)
       │
       ▼
 Look up postings lists from index: L₁ and L₂
       │
       ├──[AND]──► intersect_postings(L₁, L₂)  ──► Two-pointer sweep  O(|L₁| + |L₂|)
       │
       └──[OR]───► union_postings(L₁, L₂)      ──► Two-pointer sweep  O(|L₁| + |L₂|)
       │
       ▼
  Write matching doc IDs to output file (one per line)
```

#### ⚡ Efficient Algorithms

Both `AND` and `OR` use an **optimal two-pointer merge** — the same algorithm used in production IR systems — achieving linear time `O(|L₁| + |L₂|)`, far better than a naive nested-loop `O(|L₁| × |L₂|)` approach.

**AND (Intersection):**
```python
# Two-pointer sweep — only advances the pointer with the smaller ID
while i < len(L1) and j < len(L2):
    if L1[i] == L2[j]: result.append(L1[i]); i++; j++
    elif L1[i] < L2[j]: i++
    else: j++
```

**OR (Union):**
```python
# Merges both lists in sorted order without duplicates
# Similar to the merge step in merge-sort
```

---

## 🚀 How to Run

> **Prerequisites:** Python 3.x — no additional packages needed.

### Step 1 — Preprocess

```bash
# From the IR_P1/ directory
python3 search-o-SAURS_preprocess.py
```

This reads `cran.all.1400` and `stopwords.txt`, and outputs `search-o-SAURS_processed.all`.

```
✔ Loaded 571 stopwords.
✔ Loaded 1400 documents.
✔ Preprocessing completed. Output written to search-o-SAURS_processed.all
```

---

### Step 2 — Build Index

```bash
python3 search-o-SAURS_indexer.py
```

This reads `search-o-SAURS_processed.all` and outputs `search-o-SAURS_cran.index`.

```
✔ Indexing completed.
✔ Vocabulary Size: 4615
✔ Maximum DocID Indexed: 1400
✔ Index written to search-o-SAURS_cran.index
```

---

### Step 3 — Search!

```bash
# AND query
python3 search-o-SAURS_search.py "aerodynamic AND experimental" all_results.txt

# OR query
python3 search-o-SAURS_search.py "boundary OR layer" all_results.txt

# With a custom index file path (optional 3rd argument)
python3 search-o-SAURS_search.py "flow AND turbulent" all_results.txt path/to/my.index
```

#### Usage

```
python3 search-o-SAURS_search.py "<Term1> <AND|OR> <Term2>" <output_file> [index_file]
```

---

## 📂 File Formats

### Preprocessed File (`.all`)

```
.I <doc_id>
.S
<space-separated stemmed tokens>
```

### Index File (`.index`)

```
<vocabulary_size>, <max_doc_id>
<token> <docid₁>,<docid₂>,<docid₃>,...
<token> <docid₁>,<docid₂>,...
...
```
> All tokens are sorted **lexicographically**. All docid lists are sorted in **ascending order**.

### Results File

```
<docid>
<docid>
<docid>
...
```
> One document ID per line, in ascending order.

---

## 📊 Stats & Results

| Metric | Value |
|--------|-------|
| 📚 Documents Indexed | **1,400** |
| 🔤 Vocabulary Size | **4,615 unique tokens** |
| 🗄️ Index Size | ~372 KB |
| 📝 Stopwords Filtered | 571 words |

### Sample Query Results

**Query: `aerodynamic AND experimental`**

Matched **49 documents**, including:
`1, 11, 29, 52, 137, 142, 202, 203, 216, 225 ...` _(see `all_results.txt` for full log)_

---

## 🧠 Algorithm Deep Dive

### Porter Stemmer (1980)

The `porter.py` module implements **M. F. Porter's landmark 1980 algorithm** _(Program, Vol. 14, no. 3, pp 130-137)_. It reduces English words to their morphological roots through five sequential suffix-stripping steps:

| Step | Purpose | Example |
|------|---------|---------|
| `step1ab` | Remove plurals, `-ed`, `-ing` | `caresses → caress`, `matting → mat` |
| `step1c` | `y → i` when vowel in stem | `happy → happi` |
| `step2` | Map double suffixes | `ization → ize` |
| `step3` | Handle `-ic-`, `-ful`, `-ness` | `electrical → electric` |
| `step4` | Remove residual suffixes | `-ance`, `-ment`, `-ion` |
| `step5` | Final `-e` and `-ll` cleanup | `probate → probat`, `roll → rol` |

The implementation follows the canonical algorithm with two noted **DEPARTURES** that are considered improvements over the original paper.

---

<div align="center">

<br/>

---

_Built with 🦕 by **search-o-SAURS** · IIT Kharagpur · Information Retrieval PA1_

---

</div>
