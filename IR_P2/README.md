**A Sparse Vector-Space Ranked Retrieval System for the Cranfield Aeronautics Corpus**

Built as part of Programming Assignment 2 for the Information Retrieval course.

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
- [Statistics](#statistics)

---



## Overview

search-o-SAURS is a sparse vector-space ranked retrieval system that processes the **Cranfield 1400** collection (1,400 aeronautics research abstracts). The system uses PyTerrier/Terrier to build an inverted index, evaluate TF-IDF parameter settings, and produce ranked results for 225 Cranfield queries.

The implementation includes:

- **Multi-stage tokenization** tailored for scientific text
- **British -> American spelling normalization** with exception guards
- **Porter stemming** using the Assignment 1 implementation
- **Stop word removal** using the group stop word list
- **Term-frequency preservation** required for ranked retrieval
- **Terrier inverted indexing** with document and term statistics
- **Sparse TF-IDF scoring** using Terrier's `TF_IDF` weighting model
- **Cosine-style ranked output** with deterministic TREC formatting
- **MAP, nDCG, P@5, P@10, and P@20 evaluation** against Cranfield qrels

The Assignment 2 system reuses the Assignment 1 preprocessing pipeline, but it does not reuse the Assignment 1 Boolean flat-file indexer. PA2 builds a new Terrier index because ranked retrieval requires term frequencies and document statistics.

---



## Project Structure

```
.
|- README.md
|- search-o-SAURS_PA2.py
|- search-o-SAURS_PA2_Report.md
|- search-o-SAURS_PA2_Report.pdf
|- run_pipeline.sh
|- data/
|  |- search-o-SAURS_cran.all.1400
|  |- search-o-SAURS_cran.qry
|  |- search-o-SAURS_cranqrel
|  |- search-o-SAURS_cranqrel.readme
|  `- search-o-SAURS_stopwords.txt
|- output/
|  |- search-o-SAURS_index/
|  |- search-o-SAURS_results.txt
|  |- search-o-SAURS_performance_metrics.csv
|  |- search-o-SAURS_timing.json
|  `- README.md
`- tests/
   `- test_pa2_structure.py
```

---



## Quick Start



### Prerequisites

- Python 3.8+
- PyTerrier and pandas
- OpenJDK 11 or newer, including `java` and `javac`
- A Python environment containing the project dependencies

The current environment uses OpenJDK 17 from Conda and PyTerrier from `venv`.

Install the Python dependencies if needed:

```bash
pip install pyterrier pandas
```



### Run the Full Pipeline

#### Option A: One Command Execution (Recommended)

From the repository root:

```bash
chmod +x IR_P2/run_pipeline.sh
./IR_P2/run_pipeline.sh
```

The pipeline performs:

1. Local Cranfield corpus parsing
2. Assignment 1 preprocessing
3. Terrier indexing
4. TF-IDF parameter experiments
5. Relevance evaluation
6. Best-model ranked result generation
7. Timing and metrics output

#### Option B: Direct Execution

```bash
python IR_P2/search-o-SAURS_PA2.py
```

The script writes the generated files into `IR_P2/output/`.



### Run Tests

```bash
python3 IR_P2/tests/test_pa2_structure.py
```

The test checks the PA1-style folder structure, group-prefixed input files, Cranfield input coverage, and TREC result formatting.

---



## Pipeline Architecture

```
                    +------------------------------------------------+
  cran.all.1400 --> |  PREPROCESSING  (Assignment 1 pipeline)         |
                    |                                                |
                    |  1. Tokenize                                   |
                    |  2. Normalize case and spelling                |
                    |  3. Remove stop words                          |
                    |  4. Apply Porter stemming                      |
                    |  5. Preserve repeated terms for term frequency|
                    |                                                |
                    +----------------+-------------------------------+
                                     |
                                     v
                    +------------------------------------------------+
                    |  INDEXING  (PyTerrier IterDictIndexer)         |
                    |                                                |
                    |  Build Terrier inverted index                  |
                    |  Store postings, term frequencies, and stats   |
                    |  Output: output/search-o-SAURS_index/          |
                    +----------------+-------------------------------+
                                     |
                                     v
                    +------------------------------------------------+
                    |  RANKED RETRIEVAL  (Terrier TF_IDF)            |
                    |                                                |
                    |  1. Preprocess 225 Cranfield queries           |
                    |  2. Apply sparse TF-IDF weighting               |
                    |  3. Test c = 0.25, 0.5, 0.75, 1.0               |
                    |  4. Rank documents by retrieval score           |
                    |  5. Select the best model by MAP                 |
                    +----------------+-------------------------------+
                                     |
                                     v
                    +------------------------------------------------+
                    |  EVALUATION                                   |
                    |                                                |
                    |  Compare ranked results with Cranfield qrels   |
                    |  Compute MAP, nDCG, P@5, P@10, and P@20        |
                    +----------------+-------------------------------+
                                     |
                                     v
       search-o-SAURS_results.txt + performance_metrics.csv
```

---



## File Descriptions



### Core Pipeline Files


| File | Purpose |
| --- | --- |
| `search-o-SAURS_PA2.py` | **Ranked retrieval pipeline.** Loads local Cranfield files, applies the Assignment 1 preprocessing functions, builds a Terrier index, evaluates sparse TF-IDF configurations, writes TREC results, and records timing. |
| `run_pipeline.sh` | **Pipeline launcher.** Runs the PA2 Python program using the repository virtual environment when available. |
| `search-o-SAURS_PA2_Report.md` | **Report source.** Documents the problem, implementation, experiments, results, and discussion. |
| `search-o-SAURS_PA2_Report.pdf` | **Submission report.** PDF version of the Assignment 2 report. |



### Supporting Files


| File | Purpose |
| --- | --- |
| `../IR_P1/search-o-SAURS_preprocess.py` | Assignment 1 tokenizer, normalizer, stopword filter, Cranfield parser, and Porter-stemming pipeline reused by PA2. |
| `../IR_P1/porter.py` | Canonical Porter stemmer implementation used by the shared preprocessing pipeline. |
| `search-o-SAURS_cran.all.1400` | Group-prefixed Cranfield document collection containing 1,400 records. |
| `search-o-SAURS_cran.qry` | Group-prefixed Cranfield query collection containing 225 queries. |
| `search-o-SAURS_cranqrel` | Human relevance judgments used to evaluate ranked retrieval. |
| `search-o-SAURS_stopwords.txt` | Stopword list used during document and query preprocessing. |
| `test_pa2_structure.py` | PA2 structure, input-coverage, and output-format checks. |



### Generated Files


| File | Generated By | Contents |
| --- | --- | --- |
| `output/search-o-SAURS_index/` | `search-o-SAURS_PA2.py` | Terrier inverted index containing the indexed PA2 collection and retrieval statistics. |
| `output/search-o-SAURS_results.txt` | `search-o-SAURS_PA2.py` | TREC-style ranked results for all 225 queries using the best TF-IDF configuration. |
| `output/search-o-SAURS_performance_metrics.csv` | `search-o-SAURS_PA2.py` | MAP, nDCG, P@5, P@10, and P@20 for each tested TF-IDF setting. |
| `output/search-o-SAURS_timing.json` | `search-o-SAURS_PA2.py` | Backend, collection sizes, indexing time, evaluation time, search time, and selected model. |

---



## Usage



### Running the Ranked Retrieval Pipeline

```bash
./IR_P2/run_pipeline.sh
```

The pipeline reads:

- `IR_P2/data/search-o-SAURS_cran.all.1400`
- `IR_P2/data/search-o-SAURS_cran.qry`
- `IR_P2/data/search-o-SAURS_cranqrel`
- `IR_P2/data/search-o-SAURS_stopwords.txt`

It writes:

- `IR_P2/output/search-o-SAURS_index/`
- `IR_P2/output/search-o-SAURS_results.txt`
- `IR_P2/output/search-o-SAURS_performance_metrics.csv`
- `IR_P2/output/search-o-SAURS_timing.json`



### TREC Result Format

Each result line has six fields:

```text
query_id Q0 document_id rank score run_name
```

Example:

```text
1 Q0 51 0 17.884623022738523 TF_IDF_c_0.75
```

The output contains results for all 225 Cranfield queries and can be passed to a standard evaluation script after confirming its expected rank convention.

---



## Design Decisions



### 1. Reusing Assignment 1 Preprocessing

PA2 imports the Assignment 1 preprocessing module so that document and query terms use the same tokenization, spelling normalization, stopword removal, and Porter stemming behavior.

The PA1 Boolean `deduplicate()` operation is intentionally not applied. Repeated terms must remain available to Terrier because term frequency affects ranked TF-IDF scores.



### 2. Local Cranfield Inputs

The PA2 program reads group-prefixed local files instead of depending on the external `irds:cranfield` download. This makes the submission reproducible and ensures that the indexed documents, queries, and qrels are the files packaged with the assignment.

The query records in `cran.qry` have non-contiguous Cranfield record identifiers. The program maps them to ordinal query IDs `1` through `225` because the standard `cranqrel` file uses ordinal IDs.



### 3. Sparse Vector-Space Retrieval

The final experiments use only Terrier's sparse `TF_IDF` weighting model, satisfying the assignment restriction against dense neural retrieval and advanced retrieval models.

The tested document parameter values are:

```text
c = 0.25, 0.50, 0.75, 1.00
```

The final selected model is `TF_IDF_c_0.75`, chosen by highest MAP.



### 4. Evaluation Against Human Judgments

The qrels are not used to rank documents. They are used only after retrieval to measure how well the ranked output places relevant documents near the top. The reported metrics are MAP, nDCG, P@5, P@10, and P@20.

---



## Optimizations



### Terrier Inverted Index

Terrier stores the lexicon, postings, document metadata, and term statistics in a disk-backed inverted index. This is different from the custom flat-file Boolean index used in Assignment 1 and is appropriate for ranked retrieval.



### Sparse Scoring

Only indexed query terms contribute to document scores. The retrieval process does not construct dense vectors for all vocabulary terms and does not use neural embeddings.



### Parameter Comparison

The pipeline evaluates four TF-IDF parameter settings in one run and selects the best configuration using MAP. This avoids reporting a manually selected model without measuring alternatives.

---



## Testing

Run the PA2 checks:

```bash
python3 IR_P2/tests/test_pa2_structure.py
```

The checks validate:

- PA1-style `data/`, `output/`, and `tests/` directories
- Group-prefixed PA2 program files
- Six-column TREC output
- Presence of local corpus, query, stopword, and qrel files
- Exactly 1,400 corpus documents
- Exactly 225 queries
- Exactly 1,837 relevance judgments
- Qrel coverage for query IDs 1 through 225

The final Terrier run was also checked independently for:

- 225 query IDs in the result file
- Valid document IDs from 1 through 1,400
- No duplicate documents within a query
- Monotonically decreasing scores
- Valid six-column TREC formatting



### Retrieval Results

The final Terrier experiment produced:

| Model | MAP | nDCG | P@5 | P@10 | P@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF, c=0.25 | 0.305104 | 0.506464 | 0.324444 | 0.229333 | 0.156889 |
| TF-IDF, c=0.50 | 0.314017 | 0.513593 | 0.331556 | 0.238222 | 0.160889 |
| **TF-IDF, c=0.75** | **0.320598** | **0.519036** | **0.332444** | **0.239556** | 0.164444 |
| TF-IDF, c=1.00 | 0.318271 | 0.516557 | 0.328000 | 0.237333 | **0.165111** |

---



## Statistics


| Metric | Value |
| --- | ---: |
| Documents in corpus | 1,400 |
| Queries | 225 |
| Relevance judgments | 1,837 |
| Tested TF-IDF configurations | 4 |
| Best model | `TF_IDF_c_0.75` |
| Best MAP | 0.320598 |
| Best nDCG | 0.519036 |
| Best P@10 | 0.239556 |
| Indexing time | 3.71 seconds |
| Evaluation time | 16.14 seconds |
| Best-model search time | 2.96 seconds |
| Backend | PyTerrier/Terrier with OpenJDK 17 |
