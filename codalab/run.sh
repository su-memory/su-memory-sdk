#!/bin/bash
# HotpotQA CodaLab Submission Script for su-memory v2.0
# 
# Usage on CodaLab:
#   cl run --request-network --request-docker-image python:3.11 \
#     'bash run.sh /path/to/hotpot_test.json /output/predictions.json'

set -e

INPUT="${1:-/mnt/input/hotpot_test.json}"
OUTPUT="${2:-/output/predictions.json}"

echo "=== su-memory v2.0 HotpotQA Submission ==="
echo "Input: $INPUT"
echo "Output: $OUTPUT"

# Install su-memory
pip install su-memory==2.0.0.post3 faiss-cpu requests

# Download benchmark runner
wget -q https://raw.githubusercontent.com/su-memory/su-memory-sdk/main/benchmarks/hotpotqa_submit.py -O /tmp/submit.py

# Run
python3 /tmp/submit.py --input "$INPUT" --output "$OUTPUT"

echo "=== Done ==="
