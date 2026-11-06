"""Nested query level validation and paired comparisons; never uses test queries.
Run: python -B src/validate.py
Saved submission weights and test outputs are never replaced.
"""
import argparse
from importlib.metadata import version
import math
from pathlib import Path
from pprint import pformat
import random
import ast
import platform
import sys

from data import ROOT, DEFAULT_DATA, CranfieldCorpus, read_documents, read_queries, read_qrels, rank, tokenize
from metrics import evaluate, relevant_documents
from svm_model import fit_scaler, scale_rows, fit_linear_svm, svm_scores

C_VALUES = (0.01, 0.1, 1.0)
CUTOFFS = tuple(range(1, 31))

def query_folds(query_ids, n_folds=3, seed=42):
    """Each query is held out once; document pairs never cross query boundaries."""
    ids = sorted(query_ids)
    if len(ids) != len(set(ids)) or n_folds < 2 or len(ids) < 2 * n_folds:
        raise ValueError("Need unique query IDs and at least two queries per fold")
    random.Random(seed).shuffle(ids)
    for fold in range(n_folds):
        held_out = sorted(ids[fold::n_folds])
        yield sorted(set(ids) - set(held_out)), held_out

def inner_split(query_ids, seed):
    ids = sorted(query_ids)
    random.Random(seed).shuffle(ids)
    count = max(1, len(ids) // 5)
    return sorted(ids[count:]), sorted(ids[:count])

def paired_uncertainty(differences, seed=42, samples=10000):
    """Query-paired percentile bootstrap CI and two-sided sign-flip test.

    Intervals condition on the fitted folds; they do not measure all possible
    training-set variation. Sign-flip p uses the Monte Carlo +1 correction.
    """
    if not differences or samples < 1:
        raise ValueError("Need paired differences and positive resample count")
    differences = list(differences)
    if not all(math.isfinite(value) for value in differences):
        raise ValueError("Differences must be finite")
    rng = random.Random(seed)
    n = len(differences)
    observed = math.fsum(differences) / n
    bootstraps = sorted(math.fsum(rng.choices(differences, k=n)) / n for _ in range(samples))
    extremes = 0
    for _ in range(samples):
        permuted = math.fsum(value if rng.getrandbits(1) else -value for value in differences) / n
        extremes += abs(permuted) >= abs(observed) - 1e-15
    return {"mean_f1_difference": observed,
            "ci95_low": bootstraps[int(0.025 * (samples - 1))],
            "ci95_high": bootstraps[int(0.975 * (samples - 1))],
            "sign_flip_p": (extremes + 1) / (samples + 1),
            "wins": sum(value > 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "losses": sum(value < 0 for value in differences)}

def choose_cutoff(ordered, qrels, cutoffs):
    """Select on inner validation only; smaller k wins exact ties."""
    trials = {k: evaluate({qid: rows[:k] for qid, rows in ordered.items()}, qrels)["macro"]["f1"]
              for k in sorted(set(cutoffs))}
    best = max(trials, key=trials.get)
    return best, trials[best]

def fit_queries(features, ids, document_ids, qrels, c, seed):
    rows = [row for qid in ids for row in features[qid]]
    labels = [int(did in relevant_documents(qrels[qid])) for qid in ids for did in document_ids]
    means, scales = fit_scaler(rows)
    return fit_linear_svm(scale_rows(rows, means, scales), labels, means, scales, c, seed)

def ordered_svm(corpus, features, ids, parameters, query_texts, maximum):
    return {qid: rank(corpus.ids, svm_scores(features[qid], parameters), maximum)
            if tokenize(query_texts[qid]) else [] for qid in ids}

def nested_validation(corpus, query_texts, qrels, development_ids, seed=42, n_folds=3,
                      c_values=C_VALUES, cutoffs=CUTOFFS, samples=10000):
    """Outer test labels are accessed only for final out-of-fold evaluation."""
    if not c_values or any(c <= 0 or not math.isfinite(c) for c in c_values):
        raise ValueError("C values must be positive and finite")
    if not cutoffs or any(k < 1 for k in cutoffs):
        raise ValueError("Cutoffs must be positive")
    maximum = max(cutoffs)
    features = {qid: corpus.features(query_texts[qid]) for qid in development_ids}
    predictions = {name: {} for name in ("SVM", "BM25", "cosine")}
    manifests = []
    for fold, (outer_train, outer_test) in enumerate(query_folds(development_ids, n_folds, seed), 1):
        inner_train, inner_validation = inner_split(outer_train, seed + fold)
        # Tuning functions receive only the inner labels, not outer held-out labels.
        inner_qrels = {qid: qrels[qid] for qid in inner_train + inner_validation}
        best, trials = None, []
        for c in sorted(set(c_values)):
            parameters = fit_queries(features, inner_train, corpus.ids, inner_qrels, c, seed)
            ordered = ordered_svm(corpus, features, inner_validation, parameters, query_texts, maximum)
            cutoff, f1 = choose_cutoff(ordered, inner_qrels, cutoffs)
            trials.append({"C": c, "cutoff": cutoff, "inner_macro_f1": f1})
            if best is None or f1 > best["inner_macro_f1"]:
                best = trials[-1]
        # Refit using outer-training queries only, with freshly fitted scaling.
        parameters = fit_queries(features, outer_train, corpus.ids,
                                 {qid: qrels[qid] for qid in outer_train}, best["C"], seed)
        predictions["SVM"].update(ordered_svm(corpus, features, outer_test, parameters,
                                                query_texts, best["cutoff"]))
        selected = {"SVM": {"C": best["C"], "cutoff": best["cutoff"]}}
        for name, column in (("BM25", 21), ("cosine", 20)):
            def ordered_baseline(ids):
                return {qid: rank(corpus.ids, [row[column] for row in features[qid]], maximum)
                        if tokenize(query_texts[qid]) else [] for qid in ids}
            cutoff, _ = choose_cutoff(ordered_baseline(inner_validation), inner_qrels, cutoffs)
            selected[name] = {"cutoff": cutoff}
            predictions[name].update({qid: rows[:cutoff] for qid, rows in ordered_baseline(outer_test).items()})
        manifests.append({"fold": fold, "outer_train": outer_train, "outer_test": outer_test,
                          "inner_train": inner_train, "inner_validation": inner_validation,
                          "selected": selected, "svm_trials": trials})
        print(f"Completed outer fold {fold}/{n_folds}: {selected}", flush=True)
    evaluations = {name: evaluate(rows, qrels) for name, rows in predictions.items()}
    comparisons = {}
    for baseline in ("BM25", "cosine"):
        differences = [evaluations["SVM"]["per_query"][qid]["f1"] -
                       evaluations[baseline]["per_query"][qid]["f1"] for qid in sorted(development_ids)]
        comparisons[f"SVM minus {baseline}"] = paired_uncertainty(differences, seed, samples)
    # Holm adjustment across the two predeclared baseline comparisons.
    adjusted = 0.0
    ordered_comparisons = sorted(comparisons.values(), key=lambda row: row["sign_flip_p"])
    for position, comparison in enumerate(ordered_comparisons):
        adjusted = max(adjusted, min(1.0, comparison["sign_flip_p"] * (len(ordered_comparisons) - position)))
        comparison["holm_adjusted_p"] = adjusted
    return {"folds": manifests, "evaluations": evaluations, "comparisons": comparisons}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "cross_validation.txt")
    args = parser.parse_args()
    splits = ast.literal_eval((ROOT / "results/splits.txt").read_text())
    development = sorted(splits["train"] + splits["validation"])
    if set(development) & set(splits["test"]):
        raise ValueError("Development and test queries overlap")
    # Restrict query texts and labels before invoking any experiment function.
    texts = {q.id: q.text for q in read_queries(DEFAULT_DATA / "cran.qry") if q.id in development}
    qrels = {qid: grades for qid, grades in read_qrels(DEFAULT_DATA / "cranqrel").items() if qid in development}
    corpus = CranfieldCorpus(read_documents(DEFAULT_DATA / "cran.all.1400"))
    result = nested_validation(corpus, texts, qrels, development)
    result["protocol"] = {"seed": 42, "outer_folds": 3, "inner_validation_fraction": 0.2,
                          "C_values": C_VALUES, "cutoffs": CUTOFFS, "bootstrap_samples": 10000,
                          "development_ids": development, "excluded_test_ids": splits["test"],
                          "stream": "all", "svm_candidates": "all",
                          "python": sys.version, "sklearn": version("scikit-learn"), "platform": platform.platform()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(pformat(result, sort_dicts=False) + "\n", encoding="utf-8")
    for name, evaluation in result["evaluations"].items():
        print(name, evaluation["macro"], flush=True)
    for name, comparison in result["comparisons"].items():
        print(name, comparison, flush=True)

if __name__ == "__main__":
    main()