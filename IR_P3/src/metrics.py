"""PA3 precision, recall and F1 for actual returned result lists."""
import math


def relevant_documents(judgments: dict[int, int]) -> set[int]:
    return {did for did, grade in judgments.items() if grade in {1, 2, 3, 4}}


def metrics(retrieved: list[int], relevant: set[int]) -> dict:
    if len(retrieved) != len(set(retrieved)):
        raise ValueError("A result list cannot contain duplicate documents")
    hits = len(set(retrieved) & relevant)
    precision = hits / len(retrieved) if retrieved else 0.0
    recall = hits / len(relevant) if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "retrieved": len(retrieved), "relevant": len(relevant), "hits": hits}


def evaluate(results: dict, qrels: dict) -> dict:
    if set(results) - set(qrels):
        raise ValueError("Results contain queries absent from the judgments")
    per_query = {qid: metrics([d for d, _ in rows], relevant_documents(qrels[qid]))
                 for qid, rows in results.items()}
    macro = {name: math.fsum(result[name] for result in per_query.values()) / len(per_query) if per_query else 0.0
             for name in ("precision", "recall", "f1")}
    hits, returned, relevant = (sum(result[name] for result in per_query.values())
                              for name in ("hits", "retrieved", "relevant"))
    precision = hits / returned if returned else 0.0
    recall = hits / relevant if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"macro": macro, "micro": {"precision": precision, "recall": recall, "f1": f1},
            "per_query": per_query}


def evaluation_text(evaluations):
    """Per-query and aggregate precision/recall/F1 in a readable text table."""
    lines = ["method\tquery_id\tcutoff\tprecision\trecall\tf1\thits\tretrieved\trelevant"]
    for method, result in evaluations.items():
        counts = {key: sum(row[key] for row in result["per_query"].values())
                  for key in ("hits", "retrieved", "relevant")}
        rows = list(sorted(result["per_query"].items()))
        rows += [("macro", {**result["macro"], **counts}), ("micro", {**result["micro"], **counts})]
        for qid, row in rows:
            fields = [method, str(qid), str(result["cutoff"])]
            fields += [format(row[key], ".17g") for key in ("precision", "recall", "f1")]
            fields += [str(row[key]) for key in ("hits", "retrieved", "relevant")]
            lines.append("\t".join(fields))
    return "\n".join(lines) + "\n"

