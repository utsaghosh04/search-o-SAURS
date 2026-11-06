"""Tests for Cranfield parsing, ranking, validation and saved results.

Run python -B tests/run_tests.py; add --retrain to check a fresh training run.
"""
import argparse
import ast
from collections import Counter
from contextlib import redirect_stdout
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import io
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import data as dp
import features as fe
import metrics as ev
import predict as prediction
import svm_model as sv
import train as pipeline
import validate as validation
import zone_model as za
from data import CranfieldCorpus, Document, read_documents, read_queries, read_qrels, rank
from features import FeatureExtractor
from metrics import evaluate
from predict import load_models, load_external_queries
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


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.collection = dp.CranfieldCorpus([
            dp.Document(9, title="alpha alpha", body="beta"),
            dp.Document(2, author="alpha", bibliography="gamma gamma", body="beta beta"),
            dp.Document(17),
        ])

    def assertCloseSequence(self, actual, expected, places=12):
        self.assertEqual(len(actual), len(expected))
        for i, (a, e) in enumerate(zip(actual, expected)):
            self.assertAlmostEqual(a, e, places=places, msg=f"feature {i + 1}")

    def test_parsing_multiline_fields_and_query_ids(self):
        path = self.path / "input"
        path.write_text(".I 001\n.T\nfirst title\nsecond line\n.A\nAlice\n.B\nJournal\n"
                        ".W\nalpha beta\n.I 004\n.W\ngamma\n", encoding="utf-8")
        docs = dp.read_documents(path)
        self.assertEqual(docs[0], dp.Document(1, "first title second line", "Alice", "Journal", "alpha beta"))
        self.assertEqual(docs[1].title, "")
        self.assertEqual([(q.id, q.source_id) for q in dp.read_queries(path)], [(1, 1), (2, 4)])
        self.assertEqual([q.id for q in dp.read_queries(path, "source")], [1, 4])
        with self.assertRaises(ValueError):
            dp.read_queries(path, "invalid")

    def test_malformed_records_rejected(self):
        path = self.path / "bad"
        for text in ("", "orphan", ".W\ntext", ".I 0\n.W\ntext",
                     ".I 1\n.W\na\n.I 1\n.W\nb", ".I 1\norphan"):
            with self.subTest(text=text):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    dp.read_records(path)

    def test_repeated_field_sections_preserve_all_content(self):
        path = self.path / "repeated_fields"
        path.write_text(".I 1\n.T\nalpha\n.A\nfirst author\n.W\nfirst body\n"
                        ".A\nsecond author\n.W\nsecond body\n", encoding="utf-8")
        document = dp.read_documents(path)[0]
        self.assertEqual(document.author, "first author second author")
        self.assertEqual(document.body, "first body second body")

    def test_minimal_preprocessing_preserves_words(self):
        self.assertEqual(dp.tokenize("The WINGS, wing; studies 1958!"),
                         ["the", "wings", "wing", "studies", "1958"])
        self.assertEqual(dp.tokenize("the and of"), ["the", "and", "of"])
        self.assertEqual(dp.tokenize(""), [])

    def test_disjunctive_zone_features(self):
        self.assertEqual(self.collection.zone_features("alpha missing"),
                         [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(self.collection.zone_features(""), [[0] * 4 for _ in range(3)])

    def test_conjunctive_zone_features_and_empty_query(self):
        c = dp.CranfieldCorpus([dp.Document(1, title="alpha beta"),
                            dp.Document(2, title="alpha")], zone_match="all")
        self.assertEqual(c.zone_features("alpha beta"), [[1, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(c.zone_features(""), [[0] * 4, [0] * 4])

    def test_zone_scores_preserve_exact_grid_ties(self):
        scores = za.zone_scores([[1, 1, 0, 0], [0, 0, 1, 0]], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(scores, [0.3, 0.3])

    def test_all_22_features_against_independent_hand_calculation(self):
        actual = self.collection.features("alpha delta")[0]
        a = math.log(4 / 3) + 1
        g = math.log(4 / 2) + 1
        unseen = math.log(4) + 1
        cosine = a * a / (math.sqrt(a*a + 4*a*a + 4*g*g) * math.sqrt(a*a + unseen*unseen))
        bm25 = math.log(1 + 1.5 / 2.5) * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 5 / (8 / 3)))
        expected = [2, 0.5, 5, a, 1, 0, 1, 0.5, 0.25,
                    0.2, 0, 0.2, 0.1, 0.01, a, 0, a, a / 2, a*a / 4, 0, cosine, bm25]
        self.assertEqual(len(fe.FEATURE_NAMES), 22)
        self.assertCloseSequence(actual, expected)

    def test_empty_unknown_and_repeated_terms(self):
        for query in ("", "unknown", "alpha", "alpha alpha beta"):
            with self.subTest(query=query):
                rows = self.collection.features(query)
                self.assertTrue(all(len(row) == 22 and all(math.isfinite(x) for x in row) for row in rows))
                self.assertEqual(rows[2], [float(len(dp.tokenize(query)))] + [0] * 21)
        for row in self.collection.features(""):
            self.assertTrue(all(x == 0 for i, x in enumerate(row) if i != 2))
        for actual, expected in zip(self.collection.features("alpha alpha"),
                                    self.collection.features("alpha")):
            self.assertEqual(actual[0], 2)
            self.assertEqual(expected[0], 1)
            self.assertCloseSequence(actual[1:], expected[1:])
        self.assertEqual(self.collection.features("alpha beta")[0][19], 1)
        # Repetition of only one term changes query direction in cosine.
        self.assertNotEqual(self.collection.features("alpha alpha beta")[0][20],
                            self.collection.features("alpha beta")[0][20])

    def test_empty_collection_streams_are_finite(self):
        c = dp.CranfieldCorpus([dp.Document(1)])
        self.assertEqual(c.features("alpha"), [[1] + [0] * 21])
        with self.assertRaises(ValueError):
            dp.CranfieldCorpus([])
        with self.assertRaises(ValueError):
            dp.CranfieldCorpus([dp.Document(1), dp.Document(1)])

    def test_stream_body_option_does_not_change_four_zones(self):
        c = dp.CranfieldCorpus(self.collection.documents, stream="body")
        self.assertEqual(c.zone_features("alpha"), self.collection.zone_features("alpha"))
        self.assertEqual(c.features("alpha")[0][1], 0)
        self.assertEqual(c.features("beta")[0][2], 2)

    def test_ranking_scores_ties_candidates_and_limits(self):
        ids, scores = [7, 2, 9, 1], [-1, 0.5, 0.5, -5]
        self.assertEqual([d for d, _ in dp.rank(ids, scores, 10)], [2, 9, 7, 1])
        self.assertEqual(dp.rank(ids, scores, 2, [True, False, True, False]), [(9, 0.5), (7, -1)])
        self.assertEqual(dp.rank(ids, scores, 1, [False]*4), [])
        for args in ((ids, scores, 0), (ids, [math.nan]*4, 2), (ids, [1], 2),
                     (ids, scores, 2, [True])):
            with self.assertRaises(ValueError):
                dp.rank(*args)

    def test_negative_one_is_nonrelevant_regression(self):
        judgments = {8: -1, 9: 4, 10: 5, 11: 0, 12: 1, 13: 2, 14: 3}
        self.assertEqual(ev.relevant_documents(judgments), {9, 12, 13, 14})
        result = ev.evaluate({1: [(8, 2), (12, 1)]}, {1: judgments})
        self.assertEqual(result["macro"], {"precision": 0.5, "recall": 0.25, "f1": 1/3})

    def test_qrel_parser_validation(self):
        path = self.path / "qrels"
        path.write_text("1 8 -1\n1 9 4\n", encoding="utf-8")
        self.assertEqual(dp.read_qrels(path), {1: {8: -1, 9: 4}})
        for invalid in ("", "1 2 6", "1 2", "1 2 1\n1 2 2", "0 2 1"):
            path.write_text(invalid, encoding="utf-8")
            with self.assertRaises(ValueError):
                dp.read_qrels(path)

    def test_precision_recall_f1_macro_micro(self):
        result = ev.metrics([1, 2, 3], {2, 3, 4, 5})
        self.assertAlmostEqual(result["precision"], 2/3)
        self.assertEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["f1"], 4/7)
        self.assertEqual(ev.metrics([], {1})["f1"], 0)
        self.assertEqual(ev.metrics([1], set())["recall"], 0)
        evaluated = ev.evaluate({1: [(1, 1), (2, 0.5)], 2: []}, {1: {1: 1}, 2: {9: 1}})
        self.assertEqual(evaluated["macro"], {"precision": 0.25, "recall": 0.5, "f1": 1/3})
        self.assertEqual(evaluated["micro"], {"precision": 0.5, "recall": 0.5, "f1": 0.5})
        with self.assertRaises(ValueError):
            ev.metrics([1, 1], {1})
        with self.assertRaises(ValueError):
            ev.evaluate({99: []}, {})

    def test_query_splits_are_reproducible_disjoint_complete(self):
        split = pipeline.split_queries(list(range(1, 226)), 42)
        self.assertEqual(split, pipeline.split_queries(list(reversed(range(1, 226))), 42))
        groups = [set(v) for v in split.values()]
        self.assertEqual(list(map(len, groups)), [135, 45, 45])
        self.assertEqual(set.union(*groups), set(range(1, 226)))
        self.assertTrue(all(not (groups[i] & groups[j]) for i in range(3) for j in range(i)))
        with self.assertRaises(ValueError):
            pipeline.split_queries([1, 2], 42)

    def test_grid_constraints_and_training_only(self):
        grid = list(za.weight_grid(10))
        self.assertEqual(len(grid), 286)
        self.assertTrue(all(all(w >= 0 for w in row) and math.isclose(sum(row), 1) for row in grid))
        zones = {1: self.collection.zone_features("alpha"), 2: self.collection.zone_features("beta")}
        qrels = {1: {9: 1}, 2: {17: 1}}
        weights, f1 = za.learn_zone_weights(self.collection, zones, qrels, [1], 2, 1)
        self.assertEqual(f1, 1)
        self.assertGreater(weights[0], weights[1])
        qrels[2] = {2: 1}
        self.assertEqual((weights, f1), za.learn_zone_weights(self.collection, zones, qrels, [1], 2, 1))

    def test_cached_grid_search_matches_brute_force_including_ties(self):
        ids = list(range(1, 17))
        c = dp.CranfieldCorpus([dp.Document(i) for i in ids])
        patterns = [[int(bool(p & (1 << j))) for j in range(4)] for p in range(16)]
        zones = {1: patterns, 2: list(reversed(patterns))}
        qrels = {1: {2: 1, 4: 1, 7: 1, 9: 1}, 2: {1: 1, 5: 1, 12: 1, 16: -1}}
        for divisions in (2, 3, 10):
            best = None
            for w in za.weight_grid(divisions):
                runs = {}
                for qid in (1, 2):
                    scores = [round(sum(z[j] * w[j] for j in range(4)), 12) for z in zones[qid]]
                    rows = sorted(((did, s) for did, s in zip(ids, scores) if s > 0),
                                  key=lambda row: (-row[1], row[0]))[:5]
                    runs[qid] = rows
                f1 = ev.evaluate(runs, qrels)["macro"]["f1"]
                if best is None or f1 > best[1]:
                    best = w, f1
            self.assertEqual(za.learn_zone_weights(c, zones, qrels, [1, 2], divisions, 5), best)

    def test_scaling_and_original_unit_scores(self):
        rows = [[float(j + i) for j in range(22)] for i in (0, 2)]
        means, scales = sv.fit_scaler(rows)
        self.assertEqual(means, [float(j + 1) for j in range(22)])
        self.assertEqual(scales, [1]*22)
        self.assertEqual(sv.scale_rows(rows, means, scales), [[-1]*22, [1]*22])
        _, scales = sv.fit_scaler([[3]*22, [3]*22])
        self.assertEqual(scales, [1]*22)
        svm = sv.fit_linear_svm([[-1]*22, [1]*22], [0, 1], means, scales, 0.1, 42)
        standardized_scores = [sum(w*x for w, x in zip(svm["standardized_weights"], row))
                               + svm["standardized_intercept"] for row in [[-1]*22, [1]*22]]
        self.assertCloseSequence(sv.svm_scores(rows, svm), standardized_scores)
        with self.assertRaises(ValueError):
            sv.fit_linear_svm([[-1]*22], [0], means, scales, 0.1, 42)

    def test_svm_candidate_policy_is_explicit(self):
        rows = [[0]*22, [1]*22]
        self.assertIsNone(sv.candidate_mask(rows, "all"))
        self.assertEqual(sv.candidate_mask(rows, "matching"), [False, True])
        with self.assertRaises(ValueError):
            sv.candidate_mask(rows, "bad")

    def test_actual_cranfield_integrity_and_mapping(self):
        docs = dp.read_documents(ROOT / "data/cran/cran.all.1400")
        queries = dp.read_queries(ROOT / "data/cran/cran.qry")
        qrels = dp.read_qrels(ROOT / "data/cran/cranqrel")
        self.assertEqual((len(docs), len(queries)), (1400, 225))
        self.assertEqual([q.source_id for q in queries[:4]], [1, 2, 4, 8])
        self.assertEqual([q.id for q in queries[:4]], [1, 2, 3, 4])
        self.assertEqual(set(qrels), {q.id for q in queries})
        self.assertEqual(sum(len(v) for v in qrels.values()), 1837)
        self.assertEqual(sum(g == -1 for v in qrels.values() for g in v.values()), 225)
        self.assertEqual(sum(len(ev.relevant_documents(v)) for v in qrels.values()), 1612)
        self.assertTrue(all(set(v) <= {d.id for d in docs} for v in qrels.values()))

    def make_dataset(self):
        data = self.path / "cran"
        data.mkdir()
        terms = ["alpha", "beta", "gamma", "delta", "epsilon"]
        (data / "cran.all.1400").write_text("".join(
            f".I {i}\n.T\n{term}\n.A\nwriter\n.B\njournal\n.W\n{term} {term} science\n"
            for i, term in enumerate(terms, 1)), encoding="utf-8")
        (data / "cran.qry").write_text("".join(
            f".I {i*3}\n.W\n{terms[(i-1)%5]}\n" for i in range(1, 16)), encoding="utf-8")
        (data / "cranqrel").write_text("".join(
            f"{i} {(i-1)%5+1} 1\n{i} {i%5+1} -1\n" for i in range(1, 16)), encoding="utf-8")
        return data

    def test_imports_are_standard_library_except_svm_and_no_json(self):
        files = list((ROOT / "src").glob("*.py")) + [Path(__file__)]
        local_modules = {path.stem for path in files}
        for path in files:
            filename = path.name
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module]
                else:
                    continue
                for module in modules:
                    self.assertNotEqual(module, "json")
                    self.assertTrue(module.split(".")[0] in sys.stdlib_module_names or
                                    module in local_modules | {"sklearn.svm", "sklearn.exceptions"}, module)
                    if module.startswith("sklearn"):
                        self.assertIn(filename, {"svm_model.py", Path(__file__).name})
        requirements = [line for line in (ROOT / "requirements.txt").read_text().splitlines()
                        if line.strip() and not line.startswith("#")]
        self.assertEqual(len(requirements), 1)
        self.assertTrue(requirements[0].startswith("scikit-learn"))

    def test_model_parameter_validation(self):
        for weights in ([1, 0], [-1, 1, 1, 0], [1, 1, 1, 1], [math.nan, 0, 0, 1]):
            with self.assertRaises(ValueError):
                za.ZoneRanker(weights)
        model = za.ZoneRanker([1, 0, 0, 0])
        self.assertEqual(model.rank(self.collection, ""), [])
        self.assertEqual(model.rank(self.collection, "unknown"), [])
        self.assertEqual(model.rank(self.collection, "alpha", 1), [(9, 1)])
        with self.assertRaises(ValueError):
            model.rank(self.collection, "", 0)
        with self.assertRaises(ValueError):
            prediction.rank_query({"collection": self.collection}, "alpha", "invalid")

    def test_model_training_reloading_cli_and_test_label_isolation(self):
        data, output = self.make_dataset(), self.path / "results"
        options = dict(data=data, output=output, seed=42, grid_divisions=2, zone_k=1,
                       cutoffs=[1, 2], c_values=[0.1, 1])
        with redirect_stdout(io.StringIO()), patch("train.fit_linear_svm", wraps=sv.fit_linear_svm) as fit:
            bundle, evaluations = pipeline.train_and_evaluate(**options)
        loaded = prediction.load_models(output / "weights.txt", data / "cran.all.1400")
        for method in ("A", "B"):
            self.assertEqual(asdict(bundle[method]), asdict(loaded[method]))
            self.assertEqual(evaluations[method]["macro"]["f1"], 1)
        self.assertFalse(list(output.glob("*.json")))
        splits = ast.literal_eval((output / "splits.txt").read_text())
        queries = {q.id: q for q in dp.read_queries(data / "cran.qry")}
        qrels = dp.read_qrels(data / "cranqrel")
        training_rows = [row for q in splits["train"] for row in loaded["collection"].features(queries[q].text)]
        means, scales = sv.fit_scaler(training_rows)
        self.assertEqual(loaded["B"].training_means, means)
        self.assertEqual(loaded["B"].training_scales, scales)
        labels = [int(d in ev.relevant_documents(qrels[q]))
                  for q in splits["train"] for d in loaded["collection"].ids]
        self.assertEqual(fit.call_args_list[0].args[1], labels)

        # Use both models in a fresh interpreter with site-packages disabled.
        external, judgments = self.path / "external.qry", self.path / "external.qrels"
        external.write_text(".I 801\n.W\nalpha\n.I 905\n.W\nbeta\n", encoding="utf-8")
        judgments.write_text("801 1 1\n905 2 1\n", encoding="utf-8")
        for method in ("A", "B"):
            run = self.path / f"external_{method}.txt"
            completed = subprocess.run([
                sys.executable, "-S", str(ROOT / "src/predict.py"), "--weights", str(output / "weights.txt"),
                "--documents", str(data / "cran.all.1400"), "--method", method,
                "--queries", str(external), "--top-k", "1", "--qrels", str(judgments),
                "--output", str(run)], capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual([line.split()[0] for line in run.read_text().splitlines()], ["801", "905"])
            with run.with_suffix(".evaluation.txt").open() as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertTrue(all(float(row["f1"]) == 1 for row in rows))
            self.assertEqual(prediction.rank_query(loaded, "", method), [])
        # Saved test query IDs must remain ordinal qrel IDs when reloaded.
        test_queries = dp.read_queries(output / "test_queries.txt", "source")
        self.assertEqual([q.id for q in test_queries], splits["test"])
        for method in ("A", "B"):
            runs = prediction.rank_queries(loaded, test_queries, method)
            self.assertEqual(ev.evaluate(runs, qrels),
                             {key: value for key, value in evaluations[method].items() if key != "cutoff"})

        # Changing test labels must not change either model, selected cutoff or ranking.
        for qid in splits["test"]:
            qrels[qid] = {5: 1, 1: -1}
        (data / "cranqrel").write_text("".join(f"{q} {d} {g}\n" for q, ds in qrels.items()
                                               for d, g in ds.items()), encoding="utf-8")
        options["output"] = self.path / "changed_test_labels"
        with redirect_stdout(io.StringIO()):
            second, _ = pipeline.train_and_evaluate(**options)
        for method in ("A", "B"):
            self.assertEqual(asdict(bundle[method]), asdict(second[method]))
            self.assertEqual((output / f"results_method_{method.lower()}.txt").read_bytes(),
                             (options["output"] / f"results_method_{method.lower()}.txt").read_bytes())
        self.assertEqual((output / "weights.txt").read_bytes(), (options["output"] / "weights.txt").read_bytes())

    def test_full_collection_models_reproduce_every_ranking_and_evaluation(self):
        output = ROOT / "results"
        if not (output / "results_method_a.txt").exists():
            self.skipTest("Run python src/train.py to include full-collection output checks")
        bundle = prediction.load_models(output / "weights.txt", ROOT / "data/cran/cran.all.1400")
        splits = ast.literal_eval((output / "splits.txt").read_text())
        test_queries = dp.read_queries(output / "test_queries.txt", "source")
        self.assertEqual([q.id for q in test_queries], splits["test"])
        qrels = dp.read_qrels(ROOT / "data/cran/cranqrel")
        evaluations = {}
        for method in ("A", "B"):
            expected = prediction.rank_queries(bundle, test_queries, method)
            actual = {q.id: [] for q in test_queries}
            previous = {}
            for line in (output / f"results_method_{method.lower()}.txt").read_text().splitlines():
                qid, q0, did, position, score, tag = line.split()
                qid, did, position, score = int(qid), int(did), int(position), float(score)
                self.assertIn(qid, actual)
                self.assertEqual((q0, tag), ("Q0", f"PA3_{method}"))
                self.assertEqual(position, len(actual[qid]) + 1)
                self.assertLessEqual(score, previous.get(qid, math.inf))
                previous[qid] = score
                actual[qid].append((did, score))
            self.assertEqual(actual, expected)
            evaluations[method] = {"cutoff": bundle[method].cutoff, **ev.evaluate(actual, qrels)}
            # Independently recompute real-data metrics without metric helpers.
            independent = []
            for qid, rows in actual.items():
                positive = {did for did, grade in qrels[qid].items() if 1 <= grade <= 4}
                hits = sum(did in positive for did, _ in rows)
                precision = hits / len(rows) if rows else 0.0
                recall = hits / len(positive) if positive else 0.0
                f1 = 2 * hits / (len(rows) + len(positive)) if rows or positive else 0.0
                measured = evaluations[method]["per_query"][qid]
                for key, value in (("precision", precision), ("recall", recall), ("f1", f1)):
                    self.assertAlmostEqual(measured[key], value, places=14)
                independent.append((precision, recall, f1))
            for column, key in enumerate(("precision", "recall", "f1")):
                mean = math.fsum(row[column] for row in independent) / len(independent)
                self.assertAlmostEqual(evaluations[method]["macro"][key], mean, places=14)
        self.assertEqual((output / "evaluation.txt").read_text(), ev.evaluation_text(evaluations))

    def test_external_query_formats_preserve_ids(self):
        path = self.path / "queries"
        cases = [(".I 801\n.W\nalpha\n.I 905\n.W\nbeta\n", [801, 905]),
                 (".I\t801\n.W\nalpha\n.I\t905\n.W\nbeta\n", [801, 905]),
                 ("801\talpha\n905\tbeta\n", [801, 905]),
                 ("alpha\nbeta\n", [1, 2])]
        for content, ids in cases:
            path.write_text(content, encoding="utf-8")
            self.assertEqual([q.id for q in prediction.load_external_queries(path)], ids)
        path.write_text("801\talpha\n801\tbeta\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            prediction.load_external_queries(path)

    def test_utf8_bom_query_formats_preserve_ids_and_text(self):
        path = self.path / "bom_queries"
        cases = [(".I 801\n.W\nalpha\n.I 905\n.W\nbeta\n", [801, 905]),
                 ("801\talpha\n905\tbeta\n", [801, 905]),
                 ("alpha\nbeta\n", [1, 2])]
        for text, ids in cases:
            with self.subTest(text=text):
                path.write_text(text, encoding="utf-8-sig")
                found = prediction.load_external_queries(path)
                self.assertEqual([q.id for q in found], ids)
                self.assertEqual([q.text for q in found], ["alpha", "beta"])
        path.write_text(cases[0][0], encoding="utf-8-sig")
        self.assertEqual([q.id for q in dp.read_queries(path, "source")], [801, 905])

    def test_nonpositive_query_ids_rejected_in_api_and_cli(self):
        for qid in (0, -1):
            with self.subTest(qid=qid):
                with self.assertRaisesRegex(ValueError, "positive"):
                    prediction.rank_queries({}, [dp.Query(qid, qid, "alpha")])
                output = self.path / "invalid_run.txt"
                completed = subprocess.run([
                    sys.executable, "-B", str(ROOT / "src/predict.py"),
                    "--method", "svm", "--query", "alpha", "--query-id", str(qid),
                    "--output", str(output)], capture_output=True, text=True, timeout=60)
                self.assertEqual(completed.returncode, 2)
                self.assertIn("--query-id must be positive", completed.stderr)
                self.assertFalse(output.exists())

    def test_malformed_cranfield_ids_never_fall_back_to_plain_queries(self):
        path = self.path / "invalid_queries"
        for identifier in ("0", "-1", "abc", ""):
            path.write_text(f".I {identifier}\n.W\nalpha\n", encoding="utf-8-sig")
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                prediction.load_external_queries(path)
        for identifier in (0, -1):
            path.write_text(f"{identifier}\talpha\n", encoding="utf-8-sig")
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                prediction.load_external_queries(path)

    def test_modular_layout_and_separate_feature_extractor(self):
        expected = {"data.py", "features.py", "zone_model.py",
                    "svm_model.py", "metrics.py", "train.py", "predict.py"}
        self.assertTrue(expected <= {p.name for p in (ROOT / "src").glob("*.py")})
        self.assertTrue((ROOT / "data/cran/cran.all.1400").is_file())
        extractor = fe.FeatureExtractor(self.collection)
        self.assertEqual(extractor.features("alpha delta"), self.collection.features("alpha delta"))

    def test_prediction_aliases_and_unrelated_working_directory(self):
        data = self.make_dataset()
        output = self.path / "trained"
        with redirect_stdout(io.StringIO()):
            pipeline.train_and_evaluate(data=data, output=output, grid_divisions=1,
                                       zone_k=1, cutoffs=[1], c_values=[0.1])
        for method in ("zone", "svm"):
            path = self.path / f"{method}.txt"
            result = subprocess.run([
                sys.executable, "-S", str(ROOT / "src/predict.py"), "--method", method,
                "--weights_file", str(output / "weights.txt"), "--documents", str(data / "cran.all.1400"),
                "--query", "alpha", "--query-id", "909", "--top_k", "1", "--output", str(path)],
                cwd=self.path, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(path.read_text().split()[:4], ["909", "Q0", "1", "1"])

    def test_svm_conversion_with_nonunit_feature_scales(self):
        means = [float(j + 1) for j in range(22)]
        scales = [float(j + 2) for j in range(22)]
        normalized = [[-1.0] * 22, [1.0] * 22]
        original = [[mean + scale * sign for mean, scale in zip(means, scales)] for sign in (-1, 1)]
        model = sv.fit_linear_svm(normalized, [0, 1], means, scales, 0.1, 42)
        expected = [math.fsum(w * x for w, x in zip(model["standardized_weights"], row))
                    + model["standardized_intercept"] for row in normalized]
        self.assertCloseSequence(sv.svm_scores(original, model), expected)


class ValidationTests(unittest.TestCase):
    def test_folds_are_query_disjoint_complete_and_reproducible(self):
        ids = list(range(1, 19))
        folds = list(validation.query_folds(ids))
        self.assertEqual(folds, list(validation.query_folds(list(reversed(ids)))))
        held_out = []
        for train, test in folds:
            self.assertFalse(set(train) & set(test))
            self.assertEqual(set(train + test), set(ids))
            inner_train, inner_validation = validation.inner_split(train, 43)
            self.assertFalse(set(inner_train) & set(inner_validation))
            self.assertEqual(set(inner_train + inner_validation), set(train))
            held_out.extend(test)
        self.assertEqual(sorted(held_out), ids)
        with self.assertRaises(ValueError):
            list(validation.query_folds([1, 1, 2, 3, 4, 5]))

    def test_paired_uncertainty_known_cases_and_determinism(self):
        zero = validation.paired_uncertainty([0] * 10, samples=1000)
        self.assertEqual((zero["ci95_low"], zero["ci95_high"], zero["sign_flip_p"]), (0, 0, 1))
        positive = validation.paired_uncertainty([0.25] * 20, samples=1000)
        self.assertEqual((positive["mean_f1_difference"], positive["ci95_low"], positive["ci95_high"]),
                         (0.25, 0.25, 0.25))
        self.assertLess(positive["sign_flip_p"], 0.01)
        differences = [0.2, -0.4, 0.1, 0.0]
        found = validation.paired_uncertainty(differences, samples=1000)
        self.assertEqual(found, validation.paired_uncertainty(differences, samples=1000))
        inverse = validation.paired_uncertainty([-x for x in differences], samples=1000)
        self.assertAlmostEqual(found["mean_f1_difference"], -inverse["mean_f1_difference"])
        self.assertEqual(found["wins"], inverse["losses"])

    def test_cutoff_selection_uses_actual_returned_lists_and_smaller_ties(self):
        ordered = {1: [(8, 1.0), (9, 0.0)], 2: []}
        qrels = {1: {8: 1}, 2: {9: 1}}
        self.assertEqual(validation.choose_cutoff(ordered, qrels, [2, 1]), (1, 0.5))
        self.assertEqual(validation.choose_cutoff({1: []}, qrels, [3, 1, 2]), (1, 0))

    def test_nested_training_and_tuning_cannot_access_outer_labels(self):
        terms = ["alpha", "beta", "gamma"]
        corpus = CranfieldCorpus([Document(i + 1, title=word, body=word + " science")
                                  for i, word in enumerate(terms)])
        texts = {qid: terms[(qid - 1) % 3] for qid in range(1, 19)}
        qrels = {qid: {(qid - 1) % 3 + 1: 1} for qid in texts}
        original_fit = validation.fit_queries
        fit_calls, choose_calls = [], []

        def audited_fit(features, ids, document_ids, judgments, c, seed):
            fit_calls.append((set(ids), set(judgments)))
            return original_fit(features, ids, document_ids, judgments, c, seed)

        original_choose = validation.choose_cutoff

        def audited_choose(ordered, judgments, cutoffs):
            choose_calls.append((set(ordered), set(judgments)))
            return original_choose(ordered, judgments, cutoffs)

        with redirect_stdout(io.StringIO()), patch.object(validation, "fit_queries", side_effect=audited_fit), \
                patch.object(validation, "choose_cutoff", side_effect=audited_choose):
            result = validation.nested_validation(corpus, texts, qrels, list(texts),
                                                  c_values=(0.1,), cutoffs=(1, 2), samples=100)
        for index, fold in enumerate(result["folds"]):
            outside = set(fold["outer_test"])
            for used, accessible in fit_calls[index * 2:index * 2 + 2]:
                self.assertTrue(used <= accessible)
                self.assertFalse(outside & accessible)
            self.assertEqual(fit_calls[index * 2 + 1], (set(fold["outer_train"]), set(fold["outer_train"])))
            for used, accessible in choose_calls[index * 3:index * 3 + 3]:
                self.assertEqual(used, set(fold["inner_validation"]))
                self.assertFalse(outside & accessible)
        for measured in result["evaluations"].values():
            self.assertEqual(set(measured["per_query"]), set(texts))
            self.assertEqual(measured["macro"]["f1"], 1)


def run_query_audit():
    lines = []
    checks = 0

    def record(message):
        lines.append(message)
        print(message, flush=True)

    def check(condition, description):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(description)

    models = load_models()
    corpus = models["collection"]
    splits = ast.literal_eval((ROOT / "results/splits.txt").read_text())
    queries = {q.id: q.text for q in read_queries(ROOT / "data/cran/cran.qry")}
    qrels = read_qrels(ROOT / "data/cran/cranqrel")

    start = time.perf_counter()
    for qid in splits["validation"]:
        query = queries[qid]
        variant = "  " + query.upper().replace(" ", "  ") + " !!! "
        for method in ("A", "B"):
            model = models[method]
            full = model.rank(corpus, query, len(corpus.ids))
            check(full == model.rank(corpus, variant, len(corpus.ids)),
                  f"Case/whitespace/punctuation invariance {method}/{qid}")
            check(full[:5] == model.rank(corpus, query, 5), f"Top-k prefix {method}/{qid}")
            check(full[:30] == model.rank(corpus, query, 30), f"Top-k prefix30 {method}/{qid}")
            check(len(full) == len({did for did, _ in full}), f"Unique IDs {method}/{qid}")
    record(f"PASS: {checks} metamorphic assertions across 45 validation queries and both models; "
           f"elapsed={time.perf_counter() - start:.3f}s")

    # Comparison under identical validation-only query and cutoff choices.
    rows = {qid: corpus.features(queries[qid]) for qid in splits["validation"]}
    for name, column in (("BM25", 21), ("TF-IDF cosine", 20)):
        trials = []
        for k in (5, 10, 15, 20, 30):
            runs = {qid: rank(corpus.ids, [row[column] for row in features], k)
                    for qid, features in rows.items()}
            trials.append((k, evaluate(runs, qrels)["macro"]["f1"]))
        record(f"VALIDATION baseline {name}: {trials}; best={max(trials, key=lambda item: item[1])}")
    for method in ("A", "B"):
        runs = {qid: models[method].rank(corpus, queries[qid]) for qid in splits["validation"]}
        record(f"VALIDATION saved method {method}: cutoff={models[method].cutoff}; "
               f"macro={evaluate(runs, qrels)['macro']}")
    ordered = {qid: rank(corpus.ids, models["B"].score_features(features), 30)
               for qid, features in rows.items()}
    fine_cutoffs = [(k, evaluate({qid: ranking[:k] for qid, ranking in ordered.items()}, qrels)["macro"]["f1"])
                    for k in range(1, 31)]
    record(f"FIXED_SPLIT_SVM_CUTOFFS: {fine_cutoffs}")

    # Quantify score ties and lexical mismatch without selecting on test labels.
    ties, zero_overlap = [], 0
    for qid in splits["validation"]:
        scores = models["A"].score(corpus, queries[qid])
        counts = Counter(score for score in scores if score > 0)
        ties.append(max(counts.values(), default=0))
        returned = models["B"].rank(corpus, queries[qid])
        coverage = {did: row[1] for did, row in zip(corpus.ids, rows[qid])}
        zero_overlap += sum(coverage[did] == 0 for did, _ in returned)
    record(f"ZONE ties: mean largest positive-score tie group={sum(ties)/len(ties):.2f} documents; "
           f"maximum={max(ties)}")
    record(f"SVM: zero lexical overlap in {zero_overlap}/"
           f"{len(rows)*models['B'].cutoff} validation results")
    unknown = "zzzxxyyunknownterm"
    record(f"UNKNOWN query: zone={models['A'].rank(corpus, unknown)}; "
           f"svm={models['B'].rank(corpus, unknown)}; "
           f"matching-only svm={replace(models['B'], candidates='matching').rank(corpus, unknown)}")

    with tempfile.TemporaryDirectory(prefix="pa3_audit_") as directory:
        directory = Path(directory)
        for qid in (0, -1):
            output = directory / f"query_{qid}.txt"
            completed = subprocess.run(
                [sys.executable, "-B", str(ROOT / "src/predict.py"), "--method", "svm",
                 "--query", "heated aircraft", "--query-id", str(qid), "--output", str(output)],
                capture_output=True, text=True, timeout=60)
            record(f"INPUT probe query-id={qid}: exit={completed.returncode}; "
                   f"accepted_invalid_id={completed.returncode == 0}")
            check(completed.returncode == 2 and not output.exists(), "Reject invalid CLI query ID before writing")
        bom = directory / "bom.qry"
        bom.write_text(".I 801\n.W\nheated aircraft\n", encoding="utf-8-sig")
        try:
            found = load_external_queries(bom)
            record(f"INPUT probe UTF-8 BOM Cranfield: IDs={[q.id for q in found]}")
            check([q.id for q in found] == [801] and found[0].text == "heated aircraft",
                  "BOM Cranfield parsing preserves query ID and text")
        except ValueError as error:
            record(f"INPUT probe UTF-8 BOM Cranfield: rejected ({error})")
            raise

    report = ROOT / "results/regression_audit.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


class QueryAuditTests(unittest.TestCase):
    def test_validation_queries_and_external_inputs(self):
        with redirect_stdout(io.StringIO()):
            run_query_audit()


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain on the full supplied corpus and compare every generated artifact")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    result_dir = ROOT / "results"
    result_dir.mkdir(exist_ok=True)
    data_paths = sorted((ROOT / "data" / "cran").iterdir())
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in data_paths if path.is_file()}
    with (result_dir / "test_results.txt").open("w", encoding="utf-8") as log:
        stream = Tee(sys.stdout, log)
        stream.write(f"PA3 correctness checks - {datetime.now(timezone.utc).isoformat()}\n\n")
        suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        success = result.wasSuccessful() and not result.skipped and result.testsRun > 0
        if result.skipped:
            stream.write("FAIL: correctness verification requires all checks to run, with no skips.\n")

        if success and args.retrain:
            stream.write("\nRetraining both models on all 1,400 documents and 225 queries...\n")
            with tempfile.TemporaryDirectory(prefix="pa3_verification_") as directory:
                output = Path(directory) / "results"
                command = [sys.executable, "-B", str(ROOT / "src" / "train.py"), "--output", str(output)]
                with (result_dir / "retraining_test.txt").open("w", encoding="utf-8") as training_log:
                    with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          text=True, encoding="utf-8", errors="replace") as process:
                        for line in process.stdout:
                            stream.write(line)
                            training_log.write(line)
                        return_code = process.wait()
                if return_code:
                    stream.write(f"FAIL: fresh training exited with status {return_code}.\n")
                    success = False
                else:
                    names = ("weights.txt", "results_method_a.txt", "results_method_b.txt", "evaluation.txt",
                             "splits.txt", "validation.txt", "test_queries.txt", "query_id_mapping.txt")
                    for name in names:
                        stored, fresh = result_dir / name, output / name
                        equal = stored.is_file() and fresh.is_file() and stored.read_bytes() == fresh.read_bytes()
                        stream.write(f"{'PASS' if equal else 'FAIL'}: fresh experiment reproduces {name}\n")
                        success = success and equal

        after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}
        unchanged = before == after
        stream.write(f"\n{'PASS' if unchanged else 'FAIL'}: supplied dataset unchanged\n")
        success = success and unchanged
        stream.write(f"\nOVERALL: {'PASS' if success else 'FAIL'} ({result.testsRun} tests; "
                     f"{len(result.failures)} failures; {len(result.errors)} errors; {len(result.skipped)} skipped)\n")
        return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
