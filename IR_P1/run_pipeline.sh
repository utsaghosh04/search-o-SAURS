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
    read -p "Enter your Boolean query (default: 'aeroelastic AND aircraft'): " QUERY
    if [ -z "$QUERY" ]; then
        QUERY="aeroelastic AND aircraft"
    fi
fi

echo "=== Preprocessing the corpus ==="
python3 search-o-SAURS_preprocess.py

echo "=== Building the inverted index ==="
python3 search-o-SAURS_indexer.py

echo "=== Running a Boolean query ==="
echo "Query: $QUERY"
python3 search-o-SAURS_search.py "$QUERY" output/output.txt

if [ "$RUN_TESTS" = true ]; then
    echo "=== Running the test suite ==="
    python3 tests/test_comprehensive.py
    python3 tests/test_sample_queries.py
else
    echo "=== Skipping test suite (use --test to run) ==="
fi

echo "=== Pipeline execution finished ==="
