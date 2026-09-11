# Information Retrieval - PA2

This directory is the Assignment 2 companion to the Boolean Assignment 1 project.
It follows the same layout:

```text
IR_P2/
|- data/
|- output/
|- tests/
|- run_pipeline.sh
|- search-o-SAURS_PA2.py
|- search-o-SAURS_PA2_Report.md
`- search-o-SAURS_PA2_Report.pdf
```

## Current status

The pipeline now uses local Cranfield inputs and only sparse TF-IDF vector-space experiments.
It generates TREC results, MAP/nDCG/precision metrics, and timing metadata. OpenJDK 17 is
installed in the user Conda environment, and the included run was generated through Terrier.

The only remaining report edit is replacing the group-member placeholder.

## Run

From the repository root:

```bash
./IR_P2/run_pipeline.sh
```

Or run the program directly:

```bash
python3 IR_P2/search-o-SAURS_PA2.py
```

Dependencies:

```bash
pip install pyterrier pandas ir_datasets
```

An OpenJDK installation is also required by Terrier.

## Included artifacts

- `search-o-SAURS_PA2.py`: Local preprocessing, sparse TF-IDF retrieval, evaluation, and PyTerrier/Terrier backend.
- `output/search-o-SAURS_results.txt`: TREC-style ranked run for all 225 queries.
- `output/search-o-SAURS_performance_metrics.csv`: MAP, nDCG, and precision comparison table.
- `output/search-o-SAURS_timing.json`: Dataset, backend, and timing metadata.
- `search-o-SAURS_PA2_Report.md` and `.pdf`: Report; replace the member placeholder before submission.
- `tests/test_pa2_structure.py`: Fast checks for the PA1-style PA2 layout and run format.

The generated `output/search-o-SAURS_index/` directory is a Terrier index and should be
recreated by the pipeline rather than treated as source code.
