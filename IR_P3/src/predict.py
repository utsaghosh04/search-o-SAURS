"""applying saved weights to unknown queries; prediction never retrains"""
import argparse
import ast
import re
from pathlib import Path
from data import (
    ROOT, DEFAULT_DATA, CranfieldCorpus, Query, read_documents,
    read_queries, read_qrels, write_run,
)
from features import FEATURE_NAMES
from zone_model import ZoneRanker
from svm_model import SVMRanker

def load_models(weights_path=ROOT / "results" / "weights.txt", documents_path=DEFAULT_DATA / "cran.all.1400"):
    """Rebuild both models from their learned plain-text parameters."""

    parameters = ast.literal_eval(Path(weights_path).read_text(encoding="utf-8"))
    if parameters.get("feature_definition_version") != 2 or parameters["features"] != list(FEATURE_NAMES):
        raise ValueError("Saved feature definitions do not match this implementation; run src/train.py")
    collection = CranfieldCorpus(read_documents(documents_path), parameters["zone_match"], parameters["stream"])
    return {"collection": collection, "A": ZoneRanker(**parameters["A"]),
            "B": SVMRanker(**parameters["B"])}

def rank_query(models, query, method="B", top_k=None):
    """Return (document ID, score) pairs for one query with either model."""
    if method not in {"A", "B"}:
        raise ValueError("Method must be A or B")
    return models[method].rank(models["collection"], query, top_k)

def rank_queries(models, queries, method="B", top_k=None):
    if any(query.id <= 0 for query in queries) or len({query.id for query in queries}) != len(queries):
        raise ValueError("Query identifiers must be unique and positive")
    return {query.id: rank_query(models, query.text, method, top_k) for query in queries}

def load_external_queries(path, id_mode="source"):
    """Accept Cranfield records, qid<TAB>text, or one plain query per line."""
    content = Path(path).read_text(encoding="utf-8-sig")
    if any(re.match(r"^\s*\.I(?:\s|$)", line) for line in content.splitlines()):
        return read_queries(path, id_mode)
    queries, seen = [], set()
    for line_number, line in enumerate(content.splitlines(), 1):
        if not line.strip():
            continue
        if "\t" in line:
            identifier, text = line.split("\t", 1)
            qid = int(identifier)
        else:
            qid, text = line_number, line
        if qid <= 0 or qid in seen:
            raise ValueError("External queries require unique positive IDs")
        seen.add(qid)
        queries.append(Query(qid, qid, text.strip()))
    if not queries:
        raise ValueError("No external queries found")
    return queries

def main():
    parser = argparse.ArgumentParser(description="Use the trained PA3 models on new queries")
    parser.add_argument("--weights", "--weights_file", default=str(ROOT / "results" / "weights.txt"))
    parser.add_argument("--documents", default=str(DEFAULT_DATA / "cran.all.1400"))
    parser.add_argument("--method", choices=["A", "B", "zone", "svm"], required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query")
    source.add_argument("--queries", help="Query file with .I/.W records")
    parser.add_argument("--query-id", type=int, default=1)
    parser.add_argument("--query-id-mode", choices=["ordinal", "source"], default="source")
    parser.add_argument("--top-k", "--top_k", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--qrels", help="Optional Cranfield judgments for evaluating these queries")
    args = parser.parse_args()
    if args.query_id <= 0:
        parser.error("--query-id must be positive")
    args.method = {"zone": "A", "svm": "B"}.get(args.method, args.method)
    models = load_models(args.weights, args.documents)
    queries = ([Query(args.query_id, args.query_id, args.query)] if args.query is not None
               else load_external_queries(args.queries, args.query_id_mode))
    qrels = read_qrels(args.qrels) if args.qrels else None
    if qrels is not None and {q.id for q in queries} - set(qrels):
        raise ValueError("Some queries have no judgments; check query ID mode")
    rankings = rank_queries(models, queries, args.method, args.top_k)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_run(output, rankings, f"PA3_{args.method}")
    if qrels is not None:
        from metrics import evaluate, evaluation_text
        cutoff = models[args.method].cutoff if args.top_k is None else args.top_k
        evaluations = {args.method: {"cutoff": cutoff, **evaluate(rankings, qrels)}}
        output.with_suffix(".evaluation.txt").write_text(evaluation_text(evaluations), encoding="utf-8")
    print(f"Ranked {len(queries)} queries with Method {args.method}; wrote {output}")

if __name__ == "__main__":
    main()