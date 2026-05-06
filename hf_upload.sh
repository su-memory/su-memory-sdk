#!/bin/bash
# HuggingFace upload script for su-memory
# Run: HF_TOKEN=your_token bash hf_upload.sh
set -e

if [ -z "$HF_TOKEN" ]; then
    echo "Error: Set HF_TOKEN environment variable"
    echo "Usage: HF_TOKEN=hf_xxx bash hf_upload.sh"
    exit 1
fi

echo "Uploading to Hugging Face..."
cd "$(dirname "$0")"

echo ""
echo "✅ Done: https://huggingface.co/su-memory/su-memory-sdk"
