#!/usr/bin/env bash
set -euo pipefail

# Fixed experiment:
#   model: /mnt/share/w00936111/GLM-5.2-W4A4C8-mxfp4-A5-0731
#   workload: 64K input, 4K output, 24 requests, concurrency 24
#   prefix cache: approximately 95% prompt-token hit rate
#
# Prerequisite: the SGLang server is already listening on 127.0.0.1:8818.

export PYTHONPATH="/home/y00951466/sglang/python:${PYTHONPATH:-}"

BASE_URL="http://127.0.0.1:8818"
RESULT_DIR="./benchmark_results/cache95"
RESULT_FILE="${RESULT_DIR}/cache95_64k_4k_$(date +%Y%m%d_%H%M%S).jsonl"

mkdir -p "${RESULT_DIR}"

# Read the live server's KV page size so the shared prefix is page-aligned.
PAGE_SIZE="$({ curl --fail --silent --show-error "${BASE_URL}/server_info"; } | python -c '
import json
import sys

info = json.load(sys.stdin)
if isinstance(info.get("decode"), list):
    info = info["decode"][0]
page_size = info.get("page_size")
if not isinstance(page_size, int) or page_size <= 0:
    raise SystemExit("server_info did not return a valid page_size")
print(page_size)
')"

INPUT_LEN=65536
PREFIX_LEN=$((INPUT_LEN * 95 / 100 / PAGE_SIZE * PAGE_SIZE))
QUESTION_LEN=$((INPUT_LEN - PREFIX_LEN))

echo "Running fixed SGLang cache experiment"
echo "  input/output:    65536/4096 tokens"
echo "  prefix/question: ${PREFIX_LEN}/${QUESTION_LEN} tokens"
echo "  requests:        24"
echo "  concurrency:     24"
echo "  result:          ${RESULT_FILE}"

# Start from an empty cache, then let the benchmark warm the shared prefix once.
curl --fail --silent --show-error \
    --request POST \
    "${BASE_URL}/flush_cache?timeout=60" \
    >/dev/null

# Prevent the benchmark client from flushing the deliberately warmed prefix.
unset SGLANG_IS_IN_CI

python -m sglang.benchmark.serving \
    --dataset-name generated-shared-prefix \
    --backend sglang \
    --model /mnt/share/w00936111/GLM-5.2-W4A4C8-mxfp4-A5-0731 \
    --host 127.0.0.1 \
    --port 8818 \
    --request-rate inf \
    --max-concurrency 24 \
    --gsp-num-groups 1 \
    --gsp-prompts-per-group 24 \
    --gsp-system-prompt-len "${PREFIX_LEN}" \
    --gsp-question-len "${QUESTION_LEN}" \
    --gsp-output-len 4096 \
    --gsp-range-ratio 1.0 \
    --gsp-ordered \
    --warmup-requests 1 \
    --cache-report \
    --seed 42 \
    --output-file "${RESULT_FILE}"