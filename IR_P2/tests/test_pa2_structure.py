"""Fast checks for the PA2 project layout and generated run format."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_pa1_style_directories_exist():
    for directory in ("data", "output", "tests"):
        assert (PROJECT_ROOT / directory).is_dir()


def test_group_prefixed_program_files_exist():
    for filename in ("search-o-SAURS_PA2.py", "run_pipeline.sh"):
        assert (PROJECT_ROOT / filename).is_file()


def test_existing_results_use_trec_columns():
    result_file = PROJECT_ROOT / "search-o-SAURS_results.txt"
    if not result_file.exists():
        return

    first_line = result_file.open(encoding="utf-8").readline().split()
    assert len(first_line) == 6
    assert first_line[1] == "Q0"
    assert first_line[2].isdigit()
    assert first_line[3].isdigit()
    float(first_line[4])


def test_assignment_inputs_are_explicitly_tracked():
    data_readme = PROJECT_ROOT / "data" / "README.md"
    text = data_readme.read_text(encoding="utf-8")
    assert "search-o-SAURS_cran.qry" in text
    assert "search-o-SAURS_cranqrel" in text


def test_local_assignment_inputs_have_expected_coverage():
    corpus = PROJECT_ROOT / "data" / "search-o-SAURS_cran.all.1400"
    queries = PROJECT_ROOT / "data" / "search-o-SAURS_cran.qry"
    qrels = PROJECT_ROOT / "data" / "search-o-SAURS_cranqrel"
    assert sum(line.startswith(".I ") for line in corpus.read_text().splitlines()) == 1400
    assert sum(line.startswith(".I ") for line in queries.read_text().splitlines()) == 225
    qrel_lines = [line for line in qrels.read_text().splitlines() if line.strip()]
    assert len(qrel_lines) == 1837
    assert {line.split()[0] for line in qrel_lines} == {str(i) for i in range(1, 226)}


def main():
    tests = (
        test_pa1_style_directories_exist,
        test_group_prefixed_program_files_exist,
        test_existing_results_use_trec_columns,
        test_assignment_inputs_are_explicitly_tracked,
        test_local_assignment_inputs_have_expected_coverage,
    )
    for test in tests:
        test()
        print(f"OK {test.__name__}")
    print(f"PA2 structure checks passed: {len(tests)}")


if __name__ == "__main__":
    main()
