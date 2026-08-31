"""Validate sample Boolean queries documented in sample_queries.md."""

import os
import subprocess
import sys
import tempfile


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_QUERIES_FILE = os.path.join(PROJECT_ROOT, "sample_queries.md")
SEARCH_SCRIPT = os.path.join(PROJECT_ROOT, "search-o-SAURS_search.py")


def parse_docids(postings_text):
    """Convert a comma-separated postings list into document IDs."""
    return [int(docid.strip()) for docid in postings_text.split(",") if docid.strip()]


def parse_search_output(output_path):
    """Read and validate the count-and-postings result file format."""
    with open(output_path, "r") as output_file:
        output = output_file.read().strip()

    count_text, separator, postings_text = output.partition("|")
    if not separator:
        raise ValueError("Output must use the format: count | docid1, docid2, ...")

    count = int(count_text.strip())
    docids = parse_docids(postings_text)
    if count != len(docids):
        raise ValueError(f"Output count {count} does not match {len(docids)} document IDs.")
    return count, docids


def parse_sample_queries(markdown_path):
    """Read expected AND postings and AND/OR counts from the Markdown table."""
    sample_queries = []
    with open(markdown_path, "r") as markdown_file:
        for line in markdown_file:
            if not line.startswith("|"):
                continue

            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) != 6 or not cells[0].isdigit():
                continue

            sample_queries.append({
                "number": int(cells[0]),
                "and_query": cells[1].strip("`"),
                "or_query": cells[2].strip("`"),
                "and_count": int(cells[3]),
                "or_count": int(cells[4]),
                "and_docids": parse_docids(cells[5]),
            })
    return sample_queries


def run_query(query, output_path):
    """Run one query and return the count and postings written by the searcher."""
    result = subprocess.run(
        [sys.executable, SEARCH_SCRIPT, query, output_path],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout + result.stderr)
    return parse_search_output(output_path)


def main():
    """Check all documented AND and OR sample queries."""
    sample_queries = parse_sample_queries(SAMPLE_QUERIES_FILE)
    failures = 0

    with tempfile.TemporaryDirectory() as temporary_directory:
        for sample in sample_queries:
            and_output = os.path.join(temporary_directory, f"sample_{sample['number']}_and.txt")
            or_output = os.path.join(temporary_directory, f"sample_{sample['number']}_or.txt")

            and_count, and_docids = run_query(sample["and_query"], and_output)
            or_count, _ = run_query(sample["or_query"], or_output)

            and_matches = (
                and_count == sample["and_count"]
                and
                and_docids == sample["and_docids"]
            )
            or_matches = or_count == sample["or_count"]

            if and_matches:
                print(f"OK  #{sample['number']} AND: count and postings match")
            else:
                failures += 1
                print(
                    f"FAIL #{sample['number']} AND: expected count {sample['and_count']} "
                    f"and {sample['and_docids']}; got count {and_count} and {and_docids}"
                )

            if or_matches:
                print(f"OK  #{sample['number']} OR: count matches")
            else:
                failures += 1
                print(
                    f"FAIL #{sample['number']} OR: expected count {sample['or_count']}; "
                    f"got {or_count}"
                )

    if failures:
        print(f"Sample query validation failed: {failures} mismatch(es).")
        return 1

    print(f"Sample query validation passed: {len(sample_queries)} AND and OR pairs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
