#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR" || exit

if [ -z "$1" ]; then
    OUTPUT_DIR=_build
else # assume it's a path
    OUTPUT_DIR=$1
fi

pip install ..[docs]  --verbose

sphinx-apidoc \
    --no-toc \
    --separate \
    --templatedir _templates/apidoc \
    -o apidoc \
    ../src/elisa

sphinx-build \
    --builder html \
    --doctree-dir _build/doctrees \
    --jobs auto \
    --define language=en \
    --fail-on-warning \
    --keep-going \
    --show-traceback \
    ./ \
    "$OUTPUT_DIR"
