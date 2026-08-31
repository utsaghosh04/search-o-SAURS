#!/bin/bash

# Exit immediately if any command exits with a non-zero status
set -e

# Resolve the directory of this script and cd to it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Step 1: Preprocessing the corpus ==="
python3 search-o-SAURS_preprocess.py

echo "=== Step 2: Building the inverted index ==="
python3 search-o-SAURS_indexer.py

echo "=== Step 3: Running a Boolean query ==="
python3 search-o-SAURS_search.py "aerodynamic AND experimental" output/results/output.txt

echo "=== Step 4: Running the test suite ==="
python3 tests/test_search.py

echo "=== Pipeline execution finished successfully! ==="
