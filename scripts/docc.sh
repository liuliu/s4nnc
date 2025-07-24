#!/usr/bin/env bash

set -euo pipefail

GIT_ROOT=$(git rev-parse --show-toplevel)

cd $GIT_ROOT

# Generate symbol graph
bazel build scripts:s4nnc_symbols
# Copy it into a valid bundle
mkdir -p s4nnc.docc
cp bazel-bin/scripts/s4nnc_symbols.symbolgraphs/*.json s4nnc.docc/
# Remove all docs
rm -rf docs
# Convert into static hosting document
docc convert s4nnc.docc --fallback-display-name="Swift for NNC" --fallback-bundle-identifier org.liuliu.s4nnc --fallback-bundle-version 0.0.1 --output-path docs --hosting-base-path /s4nnc --index --transform-for-static-hosting
# Adding auto-redirect index.html
echo '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=https://liuliu.github.io/s4nnc/documentation/nnc">' > docs/index.html
rm -rf s4nnc.docc
