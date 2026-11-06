"""Independent numerical checks for features, model scores and saved rankings."""

import ast
from collections import Counter
import csv
import math
from pathlib import Path
import random
import re
import statistics
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import CranfieldCorpus, Document, read_documents
from features import FeatureExtractor
from predict import load_models
from svm_model import fit_linear_svm, svm_scores


def read_raw_documents(path):
    parts = re.split(r"(?m)^\.I\s+(\d+)\s*$", Path(path).read_text(encoding="utf-8"))
    documents = {}
    for offset in range(1, len(parts), 2):
        fields = re.split(r"(?m)^\.([TABW])\s*$", parts[offset + 1])
        record = {}
        for i in range(1, len(fields), 2):
            record[fields[i]] = record.get(fields[i], "") + "\n" + fields[i + 1].strip()
        documents[int(parts[offset])] = record
    return documents


def reference_features(streams, query):
    words = query.split()
    terms = sorted(set(words))
    lengths = [len(stream) for stream in streams]
    counts = [Counter(stream) for stream in streams]
    n = len(streams)
    df = {term: sum(term in stream for stream in counts) for stream in counts for term in stream}

    def idf(term):
        return 1 + math.log((n + 1) / (df.get(term, 0) + 1))

    def summary(values):
        return [sum(values), min(values), max(values), statistics.mean(values), statistics.pvariance(values)]

    if not terms:
        return [[0, 0, length] + [0] * 19 for length in lengths]
    query_weights = {term: words.count(term) * idf(term) for term in terms}
    query_norm = math.sqrt(sum(value ** 2 for value in query_weights.values()))
    average_length = statistics.mean(lengths) or 1
    output = []
    for frequencies, length in zip(counts, lengths):
        tf = [frequencies[term] for term in terms]
        covered = sum(term in frequencies for term in terms)
        doc_weights = {term: value * idf(term) for term, value in frequencies.items()}
        doc_norm = math.sqrt(sum(value ** 2 for value in doc_weights.values()))
        cosine = (sum(doc_weights.get(term, 0) * query_weights[term] for term in terms)
                  / (query_norm * doc_norm)) if doc_norm else 0
        bm25 = 0
        for term in terms:
            frequency = frequencies[term]
            term_idf = math.log(1 + (n - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
            bm25 += term_idf * 2.2 * frequency / (frequency + 1.2 * (0.25 + 0.75 * length / average_length))
        output.append([len(words), covered / len(terms), length,
                       sum(idf(term) for term in terms if term in frequencies)]
                      + summary(tf)
                      + summary([value / length if length else 0 for value in tf])
                      + summary([frequencies[term] * idf(term) for term in terms])
                      + [int(covered == len(terms)), cosine, bm25])
    return output


class ModelOutputTests(unittest.TestCase):
    def test_features_against_independent_calculator(self):
        rng = random.Random(431)
        vocabulary = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
        documents, streams = [], []
        for did in range(1, 26):
            fields = [rng.choices(vocabulary, k=rng.randrange(9)) for _ in range(4)]
            documents.append(Document(did, *(" ".join(field) for field in fields)))
            streams.append([word for field in fields for word in field])
        queries = ["", "unknown", "alpha unknown", "alpha alpha beta"]
        queries += [" ".join(rng.choices(vocabulary + ["unknown"], k=rng.randrange(1, 12))) for _ in range(28)]
        for stream_mode in ("all", "body"):
            corpus = CranfieldCorpus(documents, stream=stream_mode)
            source_streams = streams if stream_mode == "all" else [doc.body.split() for doc in documents]
            extractor = FeatureExtractor(corpus)
            for query in queries:
                expected = reference_features(source_streams, query)
                actual = extractor.features(query)
                for did, (found, reference) in enumerate(zip(actual, expected), 1):
                    self.assertEqual(len(found), 22)
                    for column, (value, wanted) in enumerate(zip(found, reference), 1):
                        self.assertTrue(math.isclose(value, wanted, abs_tol=1e-11, rel_tol=1e-11),
                                        (stream_mode, query, did, column, value, wanted))
                    self.assertGreaterEqual(found[20], -1e-12)
                    self.assertLessEqual(found[20], 1 + 1e-12)

    def test_exported_scores_against_solver_decision_function(self):
        from sklearn.svm import LinearSVC

        rng = random.Random(51)
        rows = [[rng.uniform(-3, 3) for _ in range(22)] for _ in range(60)]
        labels = [int(row[0] + 2 * row[1] - row[2] > 0) for row in rows]
        means = [rng.uniform(-100, 100) for _ in range(22)]
        scales = [10 ** rng.uniform(-2, 2) for _ in range(22)]
        solvers = []

        def solver_factory(**kwargs):
            solver = LinearSVC(**kwargs)
            solvers.append(solver)
            return solver

        with patch("sklearn.svm.LinearSVC", side_effect=solver_factory):
            exported = fit_linear_svm(rows, labels, means, scales, 0.1, 42)
        probes = rows[:5] + [[rng.uniform(-7, 7) for _ in range(22)] for _ in range(30)]
        raw_probes = [[mean + scale * x for x, mean, scale in zip(row, means, scales)] for row in probes]
        expected = solvers[0].decision_function(probes)
        actual = svm_scores(raw_probes, exported)
        for found, wanted in zip(actual, expected):
            self.assertAlmostEqual(found, float(wanted), places=10)

    def test_corpus_parser_against_independent_record_parser(self):
        raw = read_raw_documents(ROOT / "data/cran/cran.all.1400")
        parsed = read_documents(ROOT / "data/cran/cran.all.1400")
        self.assertEqual(len(raw), 1400)
        for doc in parsed:
            for field, tag in (("title", "T"), ("author", "A"), ("bibliography", "B"), ("body", "W")):
                self.assertEqual(getattr(doc, field).split(), raw[doc.id].get(tag, "").split())

    def test_every_saved_model_score_ranking_and_metric(self):
        output = ROOT / "results"
        bundle = load_models()
        corpus = bundle["collection"]
        parameters = ast.literal_eval((output / "weights.txt").read_text())
        splits = ast.literal_eval((output / "splits.txt").read_text())
        raw = read_raw_documents(ROOT / "data/cran/cran.all.1400")
        raw_queries = re.split(r"(?m)^\.I\s+\d+\s*$", (ROOT / "data/cran/cran.qry").read_text())[1:]
        queries = {qid: re.sub(r"(?m)^\.W\s*$", "", query).strip() for qid, query in enumerate(raw_queries, 1)}
        judgments = {}
        for line in (ROOT / "data/cran/cranqrel").read_text().splitlines():
            qid, did, grade = map(int, line.split())
            if grade in (1, 2, 3, 4):
                judgments.setdefault(qid, set()).add(did)
        saved = {method: {qid: [] for qid in splits["test"]} for method in ("A", "B")}
        for method in saved:
            for line in (output / f"results_method_{method.lower()}.txt").read_text().splitlines():
                qid, _, did, position, score, _ = line.split()
                rows = saved[method][int(qid)]
                self.assertEqual(int(position), len(rows) + 1)
                rows.append((int(did), float(score)))
        measurements = {method: {} for method in saved}
        report = ["query_id\tdocuments_scored\tmax_svm_score_error\tzone_ranking\tsvm_ranking"]
        for qid in splits["test"]:
            terms = set(re.findall(r"[a-z0-9]+", queries[qid].lower()))
            zone_scores = []
            for did in corpus.ids:
                flags = []
                for tag in ("T", "A", "B", "W"):
                    field_terms = set(re.findall(r"[a-z0-9]+", raw[did].get(tag, "").lower()))
                    flags.append(bool(terms & field_terms) if parameters["zone_match"] == "any"
                                 else bool(terms) and terms <= field_terms)
                zone_scores.append(round(sum(w for present, w in zip(flags, parameters["A"]["weights"]) if present), 12))
            features = FeatureExtractor(corpus).features(queries[qid])
            svm = parameters["B"]
            independent_scores = [math.fsum((x - mean) / scale * weight
                                           for x, mean, scale, weight in zip(row, svm["training_means"],
                                                                          svm["training_scales"], svm["standardized_weights"]))
                                  + svm["standardized_intercept"] for row in features]
            model_scores = bundle["B"].score_features(features)
            error = max(abs(a - b) for a, b in zip(independent_scores, model_scores))
            self.assertLess(error, 1e-10)
            for method, scores in (("A", zone_scores), ("B", independent_scores)):
                candidates = [(did, score) for i, (did, score) in enumerate(zip(corpus.ids, scores))
                              if (method == "A" and score > 0) or
                              (method == "B" and (svm["candidates"] == "all" or features[i][1] > 0))]
                ranked = sorted(candidates, key=lambda row: (-row[1], row[0]))[:parameters[method]["cutoff"]]
                self.assertEqual([did for did, _ in ranked], [did for did, _ in saved[method][qid]])
                self.assertEqual(len(saved[method][qid]), len({did for did, _ in saved[method][qid]}))
                for (_, expected_score), (_, stored_score) in zip(ranked, saved[method][qid]):
                    self.assertAlmostEqual(expected_score, stored_score, places=10)
                hits = sum(did in judgments[qid] for did, _ in ranked)
                retrieved, relevant = len(ranked), len(judgments[qid])
                measurements[method][str(qid)] = (hits, retrieved, relevant)
            report.append(f"{qid}\t{len(corpus.ids)}\t{error:.4g}\tPASS\tPASS")
        with (output / "evaluation.txt").open() as stream:
            table = list(csv.DictReader(stream, delimiter="\t"))
        for row in table:
            records = measurements[row["method"]]
            counts = records[row["query_id"]] if row["query_id"] not in {"macro", "micro"} else tuple(map(sum, zip(*records.values())))
            self.assertEqual(tuple(int(row[key]) for key in ("hits", "retrieved", "relevant")), counts)
            h, n, r = counts
            if row["query_id"] == "macro":
                precision = statistics.mean(h / n if n else 0 for h, n, r in records.values())
                recall = statistics.mean(h / r if r else 0 for h, n, r in records.values())
                f1 = statistics.mean(2 * h / (n + r) if n + r else 0 for h, n, r in records.values())
            else:
                precision, recall, f1 = h / n if n else 0, h / r if r else 0, 2 * h / (n + r) if n + r else 0
            for key, expected in (("precision", precision), ("recall", recall), ("f1", f1)):
                self.assertAlmostEqual(float(row[key]), expected, places=13)
        (output / "output_verification.txt").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main(verbosity=2)
