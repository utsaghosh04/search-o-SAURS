# Learning to rank on Cranfield

This project implements the two ranking methods in PA3. The first learns how much weight to give a document's title, author, bibliography and body. The second learns a linear SVM over the 22 query-document features listed in the assignment.

Both models produce ranked document lists. Their precision, recall and F1 are measured on queries kept out of training and parameter selection. The LaTeX report is available at `output/pdf/Report_PA3_search-o-SAURS.pdf`; its editable source is in `report/`.

## Getting started

Use Python 3.10 or later. Run these commands from the project folder:

```powershell
python -m pip install -r requirements.txt
python src/train.py
```

Training writes the learned weights, test rankings and evaluation to `results/`. The supplied Cranfield files are already in `data/cran/`.

Scikit-learn is used only to fit the linear SVM, as permitted by PA3. Parsing, features, scaling, grid search, prediction, evaluation and testing use Python's standard library. No extra data downloads are needed.

## Search with either model

After training, try a query:

```powershell
python src/predict.py --method zone --query "heated aircraft" --output results/search_zone.txt
python src/predict.py --method svm --query "heated aircraft" --top-k 10 --output results/search_svm.txt
```

For a file of queries:

```powershell
python src/predict.py --method svm --queries queries.txt --output results/search_results.txt
```

The file can contain Cranfield `.I`/`.W` records, `query_id<TAB>query text`, or one query per line. Explicit IDs are preserved. Add `--qrels judgments.txt` to evaluate the returned lists. Prediction loads the saved weights; it does not train again.

You can also call the models from Python:

```python
import sys
sys.path.insert(0, "src")
from predict import load_models, rank_query

models = load_models()
ranked = rank_query(models, "heated aircraft", method="B", top_k=10)
print(ranked)  # (document ID, score) pairs
print(models["A"].weights)
print(models["B"].weights)
```

`A` and `zone` refer to weighted zones; `B` and `svm` refer to the SVM in the command-line interface. Use `--help` for the remaining options. Default data and weight paths work even when the command is launched from another folder.

## Where to find things

```text
src/
  data.py          Read Cranfield files and build corpus statistics
  features.py      Calculate the 22 ranking features
  zone_model.py    Learn and apply the four Boolean zone weights
  svm_model.py     Fit the linear SVM and apply its learned coefficients
  metrics.py       Calculate precision, recall and F1
  train.py         Run training, validation and test evaluation
  predict.py       Load the models and answer new queries
data/cran/         Original collection, queries and judgments
results/           Learned parameters, rankings and evaluation
tests/
  test_pipeline.py       Parsing, formulas, training and integration tests
  test_model_outputs.py  Independent feature, score and output checks
  run_tests.py           Run tests and optionally reproduce the experiment
```

The main result files are:

| File in `results/` | What it contains |
| --- | --- |
| `weights.txt` | Four zone weights, 22 SVM coefficients, intercept, scaling parameters and selected cutoffs |
| `results_method_a.txt` | Weighted-zone results for the test queries |
| `results_method_b.txt` | SVM results for the same test queries |
| `evaluation.txt` | Each query's precision, recall, F1 and counts, followed by macro/micro summaries |
| `splits.txt` | Which query IDs belong to training, validation and test |
| `validation.txt` | The settings tried and their validation scores |
| `test_queries.txt` | The held-out queries in Cranfield format |
| `query_id_mapping.txt` | Query positions alongside their original `.I` IDs |
| `test_results.txt` | The latest test report |
| `output_verification.txt` | Independent score-error and ranking checks for each test query |
| `retraining_test.txt` | Training log from the full reproduction check |

Weights are plain Python literal data, read using `ast.literal_eval`. There is no JSON or binary model file. Ranking lines use `query_id Q0 document_id rank score run_tag`. Scores descend; equal scores use ascending document ID. Empty results have no ranking lines but remain in the evaluation table.

## How the experiment works

Seed 42 splits the 225 queries into 135 training, 45 validation and 45 test queries. All document pairs for a query stay together. Training learns the weights; validation chooses settings; the test set is used only for the final measurement. Scaling uses training pairs only. Document statistics such as IDF use the searchable collection without consulting relevance judgments.

Method A has one Boolean score for each field. By default, a field matches if it contains any query term. `--zone-match all` requires all terms in that field. A grid searches 286 weight combinations in steps of 0.1, with nonnegative weights summing to one. Training macro F1 at 10 selects the weights. Zero-score documents are excluded. Scores are rounded to 12 decimal places to preserve grid ties.

Method B pairs every training query with every document. A linear SVM learns binary relevance from the 22 features. Features are standardized using population standard deviations; constant columns use scale 1. Class weights are balanced. Validation chooses C from 0.01, 0.1 and 1, and both models choose a cutoff from 5, 10, 15, 20 and 30. Ties prefer smaller settings. Models are not refitted after selection. The SVM's exported coefficients use original feature units, so prediction needs no SVM library. Rankings use decision scores, including negative scores if they fall in the selected top-k.

Precision is relevant documents retrieved divided by documents actually returned. Recall divides those hits by all known relevant documents. F1 is the harmonic mean. Empty denominators give zero. Macro scores average the per-query values; micro scores pool counts. The exported rankings are exactly the lists used to calculate `evaluation.txt`.

For the default experiment, the held-out macro results are:

| Model | Cutoff | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Boolean zones | 30 | 0.0126 | 0.0756 | 0.0197 |
| Linear SVM | 5 | 0.3156 | 0.2608 | 0.2702 |

These numbers describe retrieval quality, not whether the score calculations are correct. Boolean field scores create many ties and give weak results here. A correct implementation does not guarantee that every returned document is relevant.

## Check the implementation and its outputs

Run the tests, save their report, and independently retrain both models:

```powershell
python -B tests/run_tests.py --retrain
```

The script fails on failures, errors or skipped checks. It verifies that the dataset is unchanged and compares all eight generated experiment files with a fresh run in a temporary directory, which is removed afterwards. Omit `--retrain` for a quicker check.

The tests go beyond reproducing the same code. A separate calculator checks all 22 features across generated collections, repeated terms, unknown terms and both stream choices. Exported SVM scores are compared with the solver's own decision function. An independent record reader checks the real corpus, including repeated field sections. Every test query is then scored against all 1,400 documents: zone scores are rebuilt from raw fields, SVM scores are reconstructed in standardized coordinates, and rankings and metrics are checked against the saved text files. Tests also cover query IDs, ties, malformed records, empty inputs, serialization and test-label isolation.

## Data and feature definitions

PA3 names the features but does not specify their exact formulas, preprocessing, split or cutoff. The choices below make this implementation reproducible; they are not additional assignment instructions.

Tokenization lowercases and keeps ASCII alphanumeric sequences, without stemming or stopword removal. Missing fields are empty. Repeated field sections are combined without discarding content. Empty queries return no documents. Document IDs are preserved. The original judgments identify queries by their one-based position in `cran.qry`, whose source `.I` values have gaps. When using the original query file with prediction, select `--query-id-mode ordinal`. Saved test queries already contain judgment-aligned IDs.

Grades 1-4 are relevant. Grade `-1`, explicit 0 or 5, and unjudged pairs are not positive labels. This gives 1,612 positive judgments and 225 grade `-1` entries in the supplied file. Graded ordering is not used by the binary learner or metrics.

The default stream concatenates all four fields and retains repeated content. `--stream body` restricts SVM features to the body. Let `Q` be the distinct query terms, `L` the stream length, `N` the document count, `df(t)` document frequency, and `tf(t,d)` raw term frequency. Missing terms contribute zeros to all statistics; variance is population variance over `Q`.

| Positions | Features | Definition |
| --- | --- | --- |
| 1 | Query term number | Query token count, including repetitions |
| 2 | Query term ratio | Distinct query terms matched divided by `|Q|` |
| 3 | Stream length | `L` |
| 4 | IDF | Sum of IDF over matched query terms |
| 5-9 | TF statistics | Sum, minimum, maximum, mean, variance of `tf(t,d)` |
| 10-14 | Normalized TF statistics | Same five statistics of `tf(t,d)/L` |
| 15-19 | TF-IDF statistics | Same five statistics of `tf(t,d)*idf(t)` |
| 20 | Boolean model | 1 when all distinct query terms occur, otherwise 0 |
| 21 | Vector space model | Raw-TF times smoothed-IDF cosine similarity |
| 22 | BM25 | Saturation and length normalization with `k1=1.2`, `b=0.75` |

Smoothed IDF is `ln((N+1)/(df(t)+1)) + 1`. Cosine uses actual query frequencies and the full document vocabulary for its norm. BM25 sums once per distinct query term:

```text
ln(1 + (N-df(t)+0.5)/(df(t)+0.5)) * tf(t,d)*(k1+1)
 / (tf(t,d) + k1*(1-b+b*L/average_L))
```

Zero-length normalized TF and zero-norm cosine are zero. The SVM ranks all documents by default, so a nonempty unknown query can still receive length-dependent scores. `--svm-candidates matching` restricts it to lexical candidates.

The programming requirements are implemented and tested under these definitions. The report covers the group details, problem statement, implementations, experiment plan, measured results and discussion required by PA3. See `report/BUILD.md` for compilation instructions.
