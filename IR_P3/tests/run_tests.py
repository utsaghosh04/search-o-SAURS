"""Run correctness checks and optionally reproduce the full Cranfield experiment.

Usage: python -B tests/run_tests.py --retrain
Uses only Python's standard library; temporary retraining outputs are cleaned up.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


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
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
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
