"""Build report tables from the submitted weights, rankings and judgments."""

import ast
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import read_qrels, read_queries
from metrics import evaluate
from predict import load_models


def main():
    output = Path(__file__).resolve().parent
    results = ROOT / "results"
    models = load_models()
    corpus = models["collection"]
    parameters = ast.literal_eval((results / "weights.txt").read_text())
    splits = ast.literal_eval((results / "splits.txt").read_text())
    trials = ast.literal_eval((results / "validation.txt").read_text())
    queries = {query.id: query for query in read_queries(ROOT / "data/cran/cran.qry")}
    qrels = read_qrels(ROOT / "data/cran/cranqrel")
    with (results / "evaluation.txt").open() as stream:
        evaluation = list(csv.DictReader(stream, delimiter="\t"))

    summary = []
    for method in ("A", "B"):
        for average in ("macro", "micro"):
            row = next(row for row in evaluation if row["method"] == method and row["query_id"] == average)
            summary.append(f"{method} & {average.capitalize()} & {row['cutoff']} & "
                           + " & ".join(f"{float(row[key]):.6f}" for key in ("precision", "recall", "f1"))
                           + f" & {row['hits']} / {row['retrieved']} / {row['relevant']} " + r"\\")

    common_cutoffs, per_query = [], {}
    for method in ("A", "B"):
        model = models[method]
        full_rankings = {qid: model.rank(corpus, queries[qid].text, 30) for qid in splits["test"]}
        for cutoff in (5, 10, 15, 20, 30):
            measured = evaluate({qid: ranking[:cutoff] for qid, ranking in full_rankings.items()}, qrels)
            common_cutoffs.append((method, cutoff, measured["macro"]))
        per_query[method] = {int(row["query_id"]): row for row in evaluation
                             if row["method"] == method and row["query_id"].isdigit()}

    counts = []
    for name, ids in splits.items():
        positives = sum(sum(grade in {1, 2, 3, 4} for grade in qrels[qid].values()) for qid in ids)
        total = len(ids) * len(corpus.ids)
        counts.append(f"{name.capitalize()} & {len(ids)} & {total:,} & {positives:,} & {total-positives:,} " + r"\\")

    weight_rows = []
    for number, (raw, standardized) in enumerate(zip(parameters["B"]["weights"], parameters["B"]["standardized_weights"]), 1):
        weight_rows.append(f"$f_{{{number}}}$ & {raw:+.6g} & {standardized:+.6f} " + r"\\")

    best_by_c = []
    for c in sorted({row["C"] for row in trials["svm_validation"]}):
        best = max((row for row in trials["svm_validation"] if row["C"] == c), key=lambda row: row["validation_macro_f1"])
        best_by_c.append(f"{c:g} & {best['cutoff']} & {best['validation_macro_f1']:.6f} " + r"\\")

    cutoff_rows = []
    for cutoff in (5, 10, 15, 20, 30):
        entries = [next(row[2] for row in common_cutoffs if row[0] == method and row[1] == cutoff) for method in ("A", "B")]
        cutoff_rows.append(str(cutoff) + " & " + " & ".join(f"{entry[key]:.6f}" for entry in entries for key in ("precision", "recall", "f1")) + " " + r"\\")

    query_id = 223
    example = []
    for position, (did, score) in enumerate(models["B"].rank(corpus, queries[query_id].text), 1):
        grade = qrels[query_id].get(did)
        label = "Relevant" if grade in {1, 2, 3, 4} else "Unjudged"
        example.append(f"{position} & {did} & {score:.6f} & {label} " + r"\\")

    with (results / "output_verification.txt").open() as stream:
        verification = list(csv.DictReader(stream, delimiter="\t"))
    max_error = max(float(row["max_svm_score_error"]) for row in verification)
    improvements = sum(float(per_query["B"][qid]["f1"]) > float(per_query["A"][qid]["f1"]) for qid in splits["test"])
    ties = sum(float(per_query["B"][qid]["f1"]) == float(per_query["A"][qid]["f1"]) for qid in splits["test"])
    zero_hits = {method: sum(int(row["hits"]) == 0 for row in per_query[method].values()) for method in ("A", "B")}
    tex = ["% Generated from actual PA3 outputs by prepare_report.py.",
           r"\newcommand{\CorpusTokens}{" + f"{sum(corpus.lengths):,}" + "}",
           r"\newcommand{\CorpusVocabulary}{" + f"{len(corpus.df):,}" + "}",
           r"\newcommand{\AverageLength}{" + f"{corpus.average_length:.3f}" + "}",
           r"\newcommand{\SelectedC}{" + str(parameters["B"]["c"]) + "}",
           r"\newcommand{\RawIntercept}{" + f"{parameters['B']['intercept']:.6f}" + "}",
           r"\newcommand{\ScaledIntercept}{" + f"{parameters['B']['standardized_intercept']:.6f}" + "}",
           r"\newcommand{\BetterQueries}{" + str(improvements) + "}",
           r"\newcommand{\TiedQueries}{" + str(ties) + "}",
           r"\newcommand{\WorseQueries}{" + str(45-improvements-ties) + "}",
           r"\newcommand{\ZoneZeroHits}{" + str(zero_hits["A"]) + "}",
           r"\newcommand{\SvmZeroHits}{" + str(zero_hits["B"]) + "}",
           r"\newcommand{\MaximumScoreError}{" + f"{max_error:.2e}" + "}"]
    for name, rows in (("SummaryRows", summary), ("SplitRows", counts), ("WeightRows", weight_rows),
                       ("ValidationRows", best_by_c), ("CutoffRows", cutoff_rows), ("ExampleRows", example)):
        tex.append("\\newcommand{\\" + name + "}{%\n" + "\n".join(rows) + "\n}")
    for method in ("A", "B"):
        coordinates = " ".join(f"({cutoff},{values['f1']:.8f})" for m, cutoff, values in common_cutoffs if m == method)
        tex.append("\\newcommand{\\Curve" + method + "}{" + coordinates + "}")
    (output / "report_data.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("Common-cutoff held-out macro metrics:")
    for row in common_cutoffs:
        print(row)
    print("SVM wins/ties/losses", improvements, ties, 45-improvements-ties)
    print("Zero-hit queries", zero_hits, "maximum score error", max_error)
    print("Empty document IDs", [did for did, length in zip(corpus.ids, corpus.lengths) if length == 0])


if __name__ == "__main__":
    main()
