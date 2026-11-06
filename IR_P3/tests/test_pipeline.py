"""PA3 standard-library tests: python -m unittest discover -s tests -v."""

import ast
import csv
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data as dp
import features as fe
import zone_model as za
import svm_model as sv
import metrics as ev
import train as pipeline
import predict as prediction


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
        files = list((ROOT / "src").glob("*.py")) + list((ROOT / "tests").glob("test_*.py"))
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
                        self.assertIn(filename, {"svm_model.py", "test_model_outputs.py"})
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


if __name__ == "__main__":
    unittest.main(verbosity=2)

