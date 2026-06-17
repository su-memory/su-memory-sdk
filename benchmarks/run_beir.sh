#!/bin/bash
cd "/Users/mac/qoder m5pro/su-memory-sdk"
rm -rf /tmp/su-memory-bench
/Users/mac/Documents/GitHub/hermes-agent/venv/bin/python benchmarks/beir.py --json --output benchmarks/results/beir_v357.txt 2>&1
echo "BEIR DONE"
