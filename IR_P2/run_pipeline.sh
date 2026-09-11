#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

if [[ -z "${PYTHON_BIN:-}" ]]; then
	if [[ -x "$SCRIPT_DIR/../venv/bin/python" ]]; then
		PYTHON_BIN="$SCRIPT_DIR/../venv/bin/python"
	else
		PYTHON_BIN="python3"
	fi
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/search-o-SAURS_PA2.py"
