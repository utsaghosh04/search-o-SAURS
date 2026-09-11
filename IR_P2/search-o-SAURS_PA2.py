"""Run the Assignment 2 sparse vector-space retrieval experiments."""

import importlib.util
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyterrier as pt


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
INDEX_DIR = OUTPUT_DIR / "search-o-SAURS_index"
CORPUS_FILE = DATA_DIR / "search-o-SAURS_cran.all.1400"
QUERY_FILE = DATA_DIR / "search-o-SAURS_cran.qry"
QREL_FILE = DATA_DIR / "search-o-SAURS_cranqrel"
STOPWORDS_FILE = DATA_DIR / "search-o-SAURS_stopwords.txt"


def load_preprocessor():
    pa1_dir = PROJECT_ROOT.parent / "IR_P1"
    if str(pa1_dir) not in sys.path:
        sys.path.insert(0, str(pa1_dir))
    module_path = pa1_dir / "search-o-SAURS_preprocess.py"
    spec = importlib.util.spec_from_file_location("pa1_preprocess", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPROCESSOR = load_preprocessor()


def preprocess_text(text, stopwords):
    tokens = PREPROCESSOR.tokenize(text)
    tokens = PREPROCESSOR.normalize(tokens)
    tokens = PREPROCESSOR.remove_stopwords(tokens, stopwords)
    return PREPROCESSOR.stem(tokens)


def parse_cranfield(filepath):
    documents = []
    current_id = None
    current_field = None
    title = []
    abstract = []

    with filepath.open(encoding="utf-8") as source:
        for raw_line in source:
            line = raw_line.strip()
            if line.startswith(".I"):
                if current_id is not None:
                    documents.append((current_id, " ".join(title + abstract)))
                current_id = str(int(line.split()[1]))
                current_field = None
                title = []
                abstract = []
            elif line.startswith(".T"):
                current_field = "title"
            elif line.startswith(".W"):
                current_field = "abstract"
            elif line.startswith("."):
                current_field = None
            elif current_field == "title":
                title.append(line)
            elif current_field == "abstract":
                abstract.append(line)

    if current_id is not None:
        documents.append((current_id, " ".join(title + abstract)))
    return documents


def parse_queries(filepath):
    queries = []
    current_id = None
    current_field = None
    text = []

    with filepath.open(encoding="utf-8") as source:
        for raw_line in source:
            line = raw_line.strip()
            if line.startswith(".I"):
                if current_id is not None:
                    queries.append((str(len(queries) + 1), " ".join(text)))
                current_id = line.split()[1]
                current_field = None
                text = []
            elif line.startswith(".W"):
                current_field = "query"
            elif line.startswith("."):
                current_field = None
            elif current_field == "query":
                text.append(line)

    if current_id is not None:
        queries.append((str(len(queries) + 1), " ".join(text)))
    return queries


def load_qrels(filepath):
    rows = []
    with filepath.open(encoding="utf-8") as source:
        for line in source:
            fields = line.split()
            if len(fields) != 3:
                continue
            query_id, doc_id, relevance = fields
            rows.append(
                {
                    "qid": str(int(query_id)),
                    "docno": str(int(doc_id)),
                    "label": int(relevance),
                }
            )
    return pd.DataFrame(rows, columns=["qid", "docno", "label"])


def build_topics(queries, stopwords):
    return pd.DataFrame(
        [
            {
                "qid": query_id,
                "query": " ".join(preprocess_text(text, stopwords)),
            }
            for query_id, text in queries
        ]
    )


def build_corpus(documents, stopwords):
    for doc_id, text in documents:
        yield {
            "docno": doc_id,
            "text": " ".join(preprocess_text(text, stopwords)),
        }


def evaluate_run(run, qrels):
    relevant = defaultdict(set)
    for row in qrels.itertuples(index=False):
        if row.label > 0:
            relevant[str(row.qid)].add(str(row.docno))

    metric_rows = []
    for model_name, query_runs in run.items():
        average_precisions = []
        ndcgs = []
        precision = {5: [], 10: [], 20: []}
        for query_id, ranked in query_runs.items():
            judged = relevant[str(query_id)]
            hits = 0
            precision_sum = 0.0
            gains = []
            for rank, (doc_id, _) in enumerate(ranked, start=1):
                is_relevant = doc_id in judged
                gains.append(1 if is_relevant else 0)
                if is_relevant:
                    hits += 1
                    precision_sum += hits / rank
            average_precisions.append(precision_sum / len(judged) if judged else 0.0)
            ideal = min(len(judged), len(ranked))
            dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
            idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal + 1))
            ndcgs.append(dcg / idcg if idcg else 0.0)
            for cutoff in precision:
                precision[cutoff].append(
                    sum(doc_id in judged for doc_id, _ in ranked[:cutoff]) / cutoff
                )
        metric_rows.append(
            {
                "name": model_name,
                "map": sum(average_precisions) / len(average_precisions),
                "ndcg": sum(ndcgs) / len(ndcgs),
                "P_5": sum(precision[5]) / len(precision[5]),
                "P_10": sum(precision[10]) / len(precision[10]),
                "P_20": sum(precision[20]) / len(precision[20]),
            }
        )
    return pd.DataFrame(metric_rows)


def run_sparse_fallback(documents, queries, stopwords, qrels):
    """Run sparse TF-IDF when Terrier cannot start because Java is unavailable."""
    indexing_start = time.perf_counter()
    term_frequencies = {}
    document_lengths = {}
    document_terms = {}
    document_frequency = Counter()
    for doc_id, text in documents:
        counts = Counter(preprocess_text(text, stopwords))
        document_terms[doc_id] = counts
        document_lengths[doc_id] = sum(counts.values())
        term_frequencies[doc_id] = counts
        document_frequency.update(counts.keys())

    total_documents = len(documents)
    idf = {
        term: math.log((total_documents + 1) / (frequency + 1)) + 1
        for term, frequency in document_frequency.items()
    }
    indexing_seconds = time.perf_counter() - indexing_start

    run = {}
    search_start = time.perf_counter()
    for c_value in (0.25, 0.5, 0.75, 1.0):
        model_name = f"TF_IDF_tf_{c_value}"
        model_run = {}
        for query_id, text in queries:
            query_counts = Counter(preprocess_text(text, stopwords))
            query_weights = {
                term: (1 + math.log(count)) * idf[term]
                for term, count in query_counts.items()
                if term in idf
            }
            query_norm = math.sqrt(sum(weight * weight for weight in query_weights.values()))
            scores = defaultdict(float)
            document_norms = defaultdict(float)
            for doc_id, counts in document_terms.items():
                weights = {
                    term: (1 + c_value * math.log(count)) * idf[term]
                    for term, count in counts.items()
                    if term in query_weights
                }
                document_norms[doc_id] = math.sqrt(sum(weight * weight for weight in weights.values()))
                scores[doc_id] = sum(query_weights[term] * weight for term, weight in weights.items())
            ranked = [
                (doc_id, score / (query_norm * document_norms[doc_id]))
                for doc_id, score in scores.items()
                if score > 0 and query_norm and document_norms[doc_id]
            ]
            model_run[query_id] = sorted(ranked, key=lambda item: (-item[1], int(item[0])))
        run[model_name] = model_run

    search_seconds = time.perf_counter() - search_start
    metrics = evaluate_run(run, qrels)
    best_name = metrics.sort_values("map", ascending=False).iloc[0]["name"]
    best_run = run[best_name]
    output_lines = []
    for query_id in sorted(best_run, key=int):
        for rank, (doc_id, score) in enumerate(best_run[query_id], start=1):
            output_lines.append(f"{query_id} Q0 {doc_id} {rank} {score:.12f} {best_name}")
    (OUTPUT_DIR / "search-o-SAURS_results.txt").write_text(
        "\n".join(output_lines) + "\n", encoding="utf-8"
    )
    metrics.to_csv(OUTPUT_DIR / "search-o-SAURS_performance_metrics.csv", index=False)
    timing = {
        "backend": "sparse-python-fallback",
        "documents": len(documents),
        "queries": len(queries),
        "relevance_judgments": len(qrels),
        "indexing_seconds": indexing_seconds,
        "search_seconds_all_models": search_seconds,
        "best_model": best_name,
        "models": list(run),
    }
    (OUTPUT_DIR / "search-o-SAURS_timing.json").write_text(
        json.dumps(timing, indent=2) + "\n", encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(json.dumps(timing, indent=2))


def main():
    required_files = (CORPUS_FILE, QUERY_FILE, QREL_FILE, STOPWORDS_FILE)
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing PA2 input file(s): " + ", ".join(missing))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stopwords = PREPROCESSOR.load_stopwords(str(STOPWORDS_FILE))
    documents = parse_cranfield(CORPUS_FILE)
    queries = parse_queries(QUERY_FILE)
    qrels = load_qrels(QREL_FILE)

    if len(documents) != 1400 or len(queries) != 225:
        raise ValueError(
            f"Unexpected Cranfield size: {len(documents)} documents, {len(queries)} queries"
        )

    try:
        if not pt.java.started():
            pt.java.init()
        pt.java.set_log_level("ERROR")
    except Exception as error:
        print(f"PyTerrier unavailable ({error}); using sparse TF-IDF fallback.")
        run_sparse_fallback(documents, queries, stopwords, qrels)
        return

    indexer = pt.IterDictIndexer(str(INDEX_DIR), overwrite=True)
    indexer.setProperty("termpipelines", "")

    indexing_start = time.perf_counter()
    indexref = indexer.index(build_corpus(documents, stopwords))
    indexing_seconds = time.perf_counter() - indexing_start

    topics = build_topics(queries, stopwords)

    # Every experiment uses Terrier's sparse TF_IDF vector-space model.
    models = []
    search_start = time.perf_counter()
    for c_value in (0.25, 0.5, 0.75, 1.0):
        model = pt.terrier.Retriever(
            indexref,
            wmodel="TF_IDF",
            controls={"c": c_value},
        )
        model.name = f"TF_IDF_c_{c_value}"
        models.append(model)

    evaluation_start = time.perf_counter()
    metrics = pt.Experiment(
        models,
        topics,
        qrels,
        eval_metrics=["map", "ndcg", "P_5", "P_10", "P_20"],
        names=[model.name for model in models],
    )
    evaluation_seconds = time.perf_counter() - evaluation_start
    metrics.to_csv(OUTPUT_DIR / "search-o-SAURS_performance_metrics.csv", index=False)

    best_name = metrics.sort_values("map", ascending=False).iloc[0]["name"]
    best_model = next(model for model in models if model.name == best_name)
    search_start = time.perf_counter()
    results = best_model.transform(topics)
    search_seconds = time.perf_counter() - search_start
    pt.io.write_results(
        results,
        str(OUTPUT_DIR / "search-o-SAURS_results.txt"),
        run_name=best_model.name,
    )

    timing = {
        "backend": "pyterrier-terrier",
        "documents": len(documents),
        "queries": len(queries),
        "relevance_judgments": len(qrels),
        "indexing_seconds": indexing_seconds,
        "evaluation_seconds": evaluation_seconds,
        "best_model_search_seconds": search_seconds,
        "best_model": best_name,
        "models": [model.name for model in models],
    }
    (OUTPUT_DIR / "search-o-SAURS_timing.json").write_text(
        json.dumps(timing, indent=2) + "\n",
        encoding="utf-8",
    )

    print(metrics.to_string(index=False))
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
