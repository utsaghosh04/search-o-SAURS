#!/bin/bash

# Exit immediately if any command exits with a non-zero status
set -e

# Resolve the directory of this script and cd to it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_TESTS=false
QUERY=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --test) RUN_TESTS=true; shift ;;
        *) QUERY="$1"; shift ;;
    esac
done

if [ -z "$QUERY" ]; then
    read -p "Enter your Boolean query (default: 'aerodynamic AND experimental'): " QUERY
    if [ -z "$QUERY" ]; then
        QUERY="aerodynamic AND experimental"
    fi
fi

echo "=== Step 1: Preprocessing the corpus ==="
python3 search-o-SAURS_preprocess.py

echo "=== Step 2: Building the inverted index ==="
python3 search-o-SAURS_indexer.py

echo "=== Step 3: Running a Boolean query ==="
echo "Query: $QUERY"
python3 search-o-SAURS_search.py "$QUERY" output/results/output.txt

if [ "$RUN_TESTS" = true ]; then
    echo "=== Step 4: Running the test suite ==="
    python3 tests/test_comprehensive.py
else
    echo "=== Step 4: Skipping test suite (use --test to run) ==="
fi

echo "=== Pipeline execution finished successfully! ==="
