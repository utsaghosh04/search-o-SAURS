"""PA3 training/validation/test pipeline. Run: python src/train.py."""
from dataclasses import asdict, replace
from pathlib import Path
from pprint import pformat
import argparse
import math
import random
from data import (
    ROOT, DEFAULT_DATA, FIELDS, CranfieldCorpus, read_documents,
    read_queries, read_qrels, rank, tokenize, write_run,
)
from features import FEATURE_NAMES
from zone_model import ZoneRanker, learn_zone_weights
from svm_model import SVMRanker, fit_scaler, scale_rows, fit_linear_svm, candidate_mask
from metrics import relevant_documents, evaluate, evaluation_text


def split_queries(query_ids: list[int], seed: int) -> dict[str, list[int]]:
    if len(query_ids) < 5 or len(set(query_ids)) != len(query_ids):
        raise ValueError("Need at least five unique queries for a three-way split")
    shuffled = sorted(query_ids)
    random.Random(seed).shuffle(shuffled)
    n_train, n_validation = int(0.6 * len(shuffled)), int(0.2 * len(shuffled))
    return {"train": sorted(shuffled[:n_train]),
            "validation": sorted(shuffled[n_train:n_train + n_validation]),
            "test": sorted(shuffled[n_train + n_validation:])}


def train_and_evaluate(data=DEFAULT_DATA, output=ROOT / "results", seed=42, grid_divisions=10,
                       zone_k=10, cutoffs=(5, 10, 15, 20, 30), c_values=(0.01, 0.1, 1.0),
                       zone_match="any", stream="all", svm_candidates="all"):
    """Train both models and return the models and held-out evaluation."""
    if zone_k < 1 or not cutoffs or any(k < 1 for k in cutoffs) or grid_divisions < 1:
        raise ValueError("Cutoffs and grid divisions must be positive and nonempty")
    if not c_values or any(not math.isfinite(c) or c <= 0 for c in c_values):
        raise ValueError("C values must be finite and positive")
    candidate_mask([], svm_candidates)
    data, output = Path(data), Path(output)
    collection = CranfieldCorpus(read_documents(data / "cran.all.1400"), zone_match, stream)
    queries = read_queries(data / "cran.qry")
    query_texts = {query.id: query.text for query in queries}
    qrels = read_qrels(data / "cranqrel")
    if set(qrels) != set(query_texts):
        raise ValueError("Query IDs do not match qrels; check the ordinal query mapping")
    if any(set(judgments) - set(collection.ids) for judgments in qrels.values()):
        raise ValueError("Judgments reference unknown documents")
    splits = split_queries(list(query_texts), seed)
    print(f"Loaded {len(collection.ids)} documents and {len(queries)} queries; "
          f"split sizes: { {name: len(ids) for name, ids in splits.items()} }", flush=True)

    zones = {qid: collection.zone_features(text) for qid, text in query_texts.items()}
    weights, zone_training_f1 = learn_zone_weights(
        collection, zones, qrels, splits["train"], grid_divisions, zone_k)
    zone_model = ZoneRanker(weights, zone_k)
    print(f"Learned zone weights: {dict(zip(FIELDS, weights))}", flush=True)
    zone_trials = {}
    for k in sorted(set(cutoffs)):
        rankings = {qid: zone_model.rank(collection, query_texts[qid], k) for qid in splits["validation"]}
        zone_trials[k] = evaluate(rankings, qrels)["macro"]["f1"]
    zone_model = replace(zone_model, cutoff=max(zone_trials, key=zone_trials.get))

    features = {}
    for position, query in enumerate(queries, 1):
        features[query.id] = collection.features(query.text)
        if position % 75 == 0:
            print(f"Computed 22 features for {position}/{len(queries)} queries", flush=True)
    training_rows = [row for qid in splits["train"] for row in features[qid]]
    labels = []
    for qid in splits["train"]:
        relevant = relevant_documents(qrels[qid])
        labels.extend(int(did in relevant) for did in collection.ids)
    means, scales = fit_scaler(training_rows)
    scaled_rows = scale_rows(training_rows, means, scales)
    best_svm, best_f1, svm_trials = None, -1.0, []
    for c in sorted(set(c_values)):
        fitted_parameters = fit_linear_svm(scaled_rows, labels, means, scales, c, seed)
        svm_model = SVMRanker(**fitted_parameters, c=c, candidates=svm_candidates)
        scores = {qid: svm_model.score_features(features[qid]) for qid in splits["validation"]}
        for k in sorted(set(cutoffs)):
            rankings = {}
            for qid in splits["validation"]:
                if not tokenize(query_texts[qid]):
                    rankings[qid] = []
                    continue
                candidates = candidate_mask(features[qid], svm_candidates)
                rankings[qid] = rank(collection.ids, scores[qid], k, candidates)
            f1 = evaluate(rankings, qrels)["macro"]["f1"]
            svm_trials.append({"C": c, "cutoff": k, "validation_macro_f1": f1})
            if f1 > best_f1:
                best_f1, best_svm = f1, replace(svm_model, cutoff=k)
        print(f"Linear SVM C={c:g} trained and validated", flush=True)

    bundle = {"collection": collection, "A": zone_model, "B": best_svm}
    output.mkdir(parents=True, exist_ok=True)
    evaluations = {}
    for method in ("A", "B"):
        model = bundle[method]
        rankings = {qid: model.rank(collection, query_texts[qid]) for qid in splits["test"]}
        write_run(output / f"results_method_{method.lower()}.txt", rankings, f"PA3_{method}")
        evaluations[method] = {"cutoff": model.cutoff, **evaluate(rankings, qrels)}

    parameters = {"feature_definition_version": 2, "features": list(FEATURE_NAMES), "zone_match": zone_match, "stream": stream,
                  "A": asdict(zone_model), "B": asdict(best_svm)}
    (output / "weights.txt").write_text(pformat(parameters, sort_dicts=False) + "\n", encoding="utf-8")
    (output / "evaluation.txt").write_text(evaluation_text(evaluations), encoding="utf-8")
    (output / "splits.txt").write_text(pformat(splits, sort_dicts=False) + "\n", encoding="utf-8")
    (output / "validation.txt").write_text(pformat({"seed": seed, "grid_divisions": grid_divisions,
        "zone_training_cutoff": zone_k, "zone_training_macro_f1": zone_training_f1,
        "zone_validation": zone_trials, "svm_validation": svm_trials}, sort_dicts=False) + "\n",
        encoding="utf-8")
    (output / "test_queries.txt").write_text("".join(
        f".I {qid}\n.W\n{query_texts[qid]}\n" for qid in splits["test"]), encoding="utf-8")
    (output / "query_id_mapping.txt").write_text("query_id\tsource_id\n" + "".join(
        f"{q.id}\t{q.source_id}\n" for q in queries), encoding="utf-8")
    for method in ("A", "B"):
        print(f"Method {method}: cutoff={evaluations[method]['cutoff']}, "
              f"macro metrics={evaluations[method]['macro']}", flush=True)
    return bundle, evaluations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output", default=str(ROOT / "results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-divisions", type=int, default=10)
    parser.add_argument("--zone-k", type=int, default=10)
    parser.add_argument("--zone-match", choices=["any", "all"], default="any")
    parser.add_argument("--stream", choices=["all", "body"], default="all")
    parser.add_argument("--svm-candidates", choices=["all", "matching"], default="all")
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10, 15, 20, 30])
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    args = parser.parse_args()
    train_and_evaluate(**vars(args))


if __name__ == "__main__":
    main()
