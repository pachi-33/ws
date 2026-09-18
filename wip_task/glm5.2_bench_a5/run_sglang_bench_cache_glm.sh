#!/usr/bin/env bash
set -euo pipefail

# Prefix-cache benchmark for SGLang.
#
# Default: 128K input, 1K output, 90% shared prefix aligned to KV pages.
# Send token IDs directly; warm and verify every DP before native timed requests.
# Settings can be overridden with command-line flags or environment variables:
#   bash run_sglang_bench_cache_glm.sh \
#     --input-len 65536 --output-len 8192 \
#     --target-kv-hit-percent 90 --max-concurrency 32
#
# TARGET_KV_HIT_PERCENT>0: the same shared prefix is warmed on every DP rank.
# TARGET_KV_HIT_PERCENT=0: every request has a distinct prefix; radix cache
# remains enabled so this measures the cache-enabled miss path.
#
# The raw SGLang JSONL keeps per-request details because ITL/TPOT distributions
# and SLOs need them. The generated *.summary.json and terminal output contain
# aggregate statistics only and do not print individual request records.

SGLANG_ROOT="${SGLANG_ROOT:-/home/y00951466/sglang}"
# MODEL="${MODEL:-/home/weights/GLM-5.2-w4a8}"
# MODEL="${MODEL:-/home/weights/GLM-5.2-W4A8C8-A5-0731}"
MODEL="${MODEL:-/mnt/share/GLM-5.2-W4A8C8-A5-0731-copy}"
# MODEL="${MODEL:-/mnt/share/w00936111/GLM-5.2-W4A4C8-mxfp4-A5-0731}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8818}"
BASE_URL="http://${HOST}:${PORT}"

INPUT_LEN="${INPUT_LEN:-131072}" # 16K 16384; 64k 65536; 128K 131072
OUTPUT_LEN="${OUTPUT_LEN:-1024}" # 1K
NUM_REQUESTS="${NUM_REQUESTS:-24}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-24}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
PAGE_SIZE="${PAGE_SIZE:-auto}"
DP_SIZE="${DP_SIZE:-auto}"
CACHE_HIT_TOLERANCE_TOKENS="${CACHE_HIT_TOLERANCE_TOKENS:-0}"
TARGET_KV_HIT_PERCENT="${TARGET_KV_HIT_PERCENT:-${PREFIX_PERCENT:-90}}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEED="${SEED:-42}"
RESULT_DIR="${RESULT_DIR:-./benchmark_results/cache}"

usage() {
    cat <<'EOF'
Usage: run_sglang_bench_cache_glm.sh [options]

Options:
  --input-len N                 Target input length in tokens.
  --output-len N                Target output length in tokens.
  --target-kv-hit-percent N     Target cached-prompt-token rate, integer 0-99.
  --num-requests N              Number of measured requests.
  --max-concurrency N           Client-side maximum in-flight requests.
  --request-rate RATE           Requests/s, or inf for burst traffic.
  --page-size N|auto            KV page size for prefix alignment (default auto).
  --dp-size N|auto              Verify server DP count (default auto).
  --cache-hit-tolerance-tokens N Allowed cached-token deviation (default 0).
  --result-dir DIR              Directory for JSONL and summary files.
  -h, --help                    Show this help.

The same settings can be supplied as environment variables: INPUT_LEN,
OUTPUT_LEN, TARGET_KV_HIT_PERCENT, NUM_REQUESTS, MAX_CONCURRENCY, REQUEST_RATE,
PAGE_SIZE, DP_SIZE, CACHE_HIT_TOLERANCE_TOKENS, and RESULT_DIR. MODEL, HOST,
PORT, and SGLANG_ROOT remain configurable through environment variables.

The companion bench_cache_exact_glm.py must stay beside this script.
Each measured request must match the page-aligned cache target; an invalid
run returns a nonzero status and preserves its raw result and summary.
EOF
}

require_value() {
    if (( $# < 2 )); then
        echo "Missing value for $1" >&2
        usage >&2
        exit 2
    fi
}

while (( $# > 0 )); do
    case "$1" in
        --input-len)
            require_value "$@"
            INPUT_LEN="$2"
            shift 2
            ;;
        --output-len)
            require_value "$@"
            OUTPUT_LEN="$2"
            shift 2
            ;;
        --target-kv-hit-percent)
            require_value "$@"
            TARGET_KV_HIT_PERCENT="$2"
            shift 2
            ;;
        --num-requests)
            require_value "$@"
            NUM_REQUESTS="$2"
            shift 2
            ;;
        --max-concurrency)
            require_value "$@"
            MAX_CONCURRENCY="$2"
            shift 2
            ;;
        --request-rate)
            require_value "$@"
            REQUEST_RATE="$2"
            shift 2
            ;;
        --page-size)
            require_value "$@"
            PAGE_SIZE="$2"
            shift 2
            ;;
        --dp-size)
            require_value "$@"
            DP_SIZE="$2"
            shift 2
            ;;
        --cache-hit-tolerance-tokens)
            require_value "$@"
            CACHE_HIT_TOLERANCE_TOKENS="$2"
            shift 2
            ;;
        --result-dir)
            require_value "$@"
            RESULT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${SGLANG_ROOT}/python:${PYTHONPATH}"
else
    export PYTHONPATH="${SGLANG_ROOT}/python"
fi

if ! command -v jq >/dev/null 2>&1; then
    echo "jq is required to generate the detailed benchmark report." >&2
    exit 3
fi

if [[ ! -f "${SCRIPT_DIR}/bench_cache_exact_glm.py" ]]; then
    echo "Missing companion: ${SCRIPT_DIR}/bench_cache_exact_glm.py" >&2
    exit 3
fi

if ! SERVER_INFO_JSON="$(curl --fail --silent --show-error "${BASE_URL}/server_info")"; then
    echo "Unable to query ${BASE_URL}/server_info." >&2
    exit 3
fi
SERVER_PAGE_SIZE="$(jq -r '(if .decode then .decode[0] else . end).page_size // empty' <<<"${SERVER_INFO_JSON}")"
SERVER_DP_SIZE="$(jq -r '(if .decode then .decode[0] else . end).dp_size // empty' <<<"${SERVER_INFO_JSON}")"
for value_name in SERVER_PAGE_SIZE SERVER_DP_SIZE; do
    if ! [[ "${!value_name}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Could not determine ${value_name} from server_info." >&2
        exit 3
    fi
done
if [[ "${PAGE_SIZE}" == auto ]]; then PAGE_SIZE="${SERVER_PAGE_SIZE}"; fi
if [[ "${DP_SIZE}" == auto ]]; then DP_SIZE="${SERVER_DP_SIZE}"; fi
if [[ "${PAGE_SIZE}" != "${SERVER_PAGE_SIZE}" || "${DP_SIZE}" != "${SERVER_DP_SIZE}" ]]; then
    echo "Configured PAGE_SIZE/DP_SIZE (${PAGE_SIZE}/${DP_SIZE}) differs from server (${SERVER_PAGE_SIZE}/${SERVER_DP_SIZE})." >&2
    exit 2
fi

for value_name in INPUT_LEN OUTPUT_LEN NUM_REQUESTS MAX_CONCURRENCY PAGE_SIZE DP_SIZE; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer, got: ${value}" >&2
        exit 2
    fi
done

if ! [[ "${CACHE_HIT_TOLERANCE_TOKENS}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "CACHE_HIT_TOLERANCE_TOKENS must be a nonnegative integer." >&2
    exit 2
fi

if ! [[ "${TARGET_KV_HIT_PERCENT}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "TARGET_KV_HIT_PERCENT must be an integer from 0 to 99, got: ${TARGET_KV_HIT_PERCENT}" >&2
    exit 2
fi

if (( TARGET_KV_HIT_PERCENT >= 100 )); then
    echo "TARGET_KV_HIT_PERCENT must be less than 100, got: ${TARGET_KV_HIT_PERCENT}" >&2
    exit 2
fi

# Every input contains exactly INPUT_LEN token IDs. Distinct markers inside
# the first suffix page prevent accidental reuse past the shared boundary.
PREFIX_LEN=$((INPUT_LEN * TARGET_KV_HIT_PERCENT / 100 / PAGE_SIZE * PAGE_SIZE))
QUESTION_LEN=$((INPUT_LEN - PREFIX_LEN))
if (( QUESTION_LEN <= 0 || (TARGET_KV_HIT_PERCENT > 0 && PREFIX_LEN <= 0) )); then
    echo "Invalid prefix/question split: prefix=${PREFIX_LEN}, question=${QUESTION_LEN}" >&2
    exit 2
fi
ALIGNED_HIT_PERCENT="$(jq -n --argjson prefix "${PREFIX_LEN}" --argjson input "${INPUT_LEN}" '100 * $prefix / $input')"

mkdir -p "${RESULT_DIR}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_FILE="${RESULT_DIR}/cache${TARGET_KV_HIT_PERCENT}_in${INPUT_LEN}_out${OUTPUT_LEN}_c${MAX_CONCURRENCY}_${RUN_ID}.jsonl"
SUMMARY_FILE="${OUTPUT_FILE%.jsonl}.summary.json"
CONSOLE_LOG="${OUTPUT_FILE%.jsonl}.console.log"

echo "SGLang prefix-cache benchmark"
echo "  server:              ${BASE_URL}"
echo "  model:               ${MODEL}"
echo "  target KV hit rate:  ${TARGET_KV_HIT_PERCENT}% (page-aligned ${ALIGNED_HIT_PERCENT}%)"
echo "  input/output:        ${INPUT_LEN}/${OUTPUT_LEN} tokens"
echo "  prefix/question:     ${PREFIX_LEN}/${QUESTION_LEN} target tokens"
echo "  DP ranks/page size:  ${DP_SIZE}/${PAGE_SIZE}"
echo "  cache tolerance:     ${CACHE_HIT_TOLERANCE_TOKENS} tokens per request"
echo "  request rate:        ${REQUEST_RATE} req/s"
echo "  max concurrency:     ${MAX_CONCURRENCY}"
echo "  per-request details: collected in raw JSONL; omitted from summary output"
echo "  result:              ${OUTPUT_FILE}"
echo "  summary:             ${SUMMARY_FILE}"
echo "  console log:         ${CONSOLE_LOG}"

# The helper flushes once, then explicitly warms and probes every DP.
# bench_serving also flushes automatically when SGLANG_IS_IN_CI is true. That
# would destroy the deliberately warmed prefix for a nonzero hit-rate target.
unset SGLANG_IS_IN_CI

BENCH_ARGS=(
    python "${SCRIPT_DIR}/bench_cache_exact_glm.py"
    --exact-input-len "${INPUT_LEN}"
    --exact-hit-percent "${TARGET_KV_HIT_PERCENT}"
    --exact-num-requests "${NUM_REQUESTS}"
    --exact-page-size "${PAGE_SIZE}"
    --exact-dp-size "${DP_SIZE}"
    --cache-hit-tolerance-tokens "${CACHE_HIT_TOLERANCE_TOKENS}"
    --dataset-name generated-shared-prefix
    --backend sglang
    --model "${MODEL}"
    --host "${HOST}"
    --port "${PORT}"
    --max-concurrency "${MAX_CONCURRENCY}"
    --request-rate "${REQUEST_RATE}"
    --gsp-num-groups 1
    --gsp-prompts-per-group "${NUM_REQUESTS}"
    --gsp-system-prompt-len "${PREFIX_LEN}"
    --gsp-question-len "${QUESTION_LEN}"
    --gsp-output-len "${OUTPUT_LEN}"
    --gsp-range-ratio 1.0
    --gsp-ordered
    --warmup-requests 0
    --cache-report
    --tag "target_kv_hit_${TARGET_KV_HIT_PERCENT}pct"
    --seed "${SEED}"
    --output-file "${OUTPUT_FILE}"
    --output-details
)

BENCH_STATUS=0
"${BENCH_ARGS[@]}" 2>&1 | tee "${CONSOLE_LOG}" || BENCH_STATUS=$?
if (( BENCH_STATUS != 0 && BENCH_STATUS != 4 )); then
    echo "Benchmark setup/execution failed; see ${CONSOLE_LOG}." >&2
    exit "${BENCH_STATUS}"
fi
if [[ ! -s "${OUTPUT_FILE}" ]]; then
    echo "Benchmark did not produce a result; see ${CONSOLE_LOG}." >&2
    exit 3
fi

# bench_serving writes JSONL. Use the last record so an explicitly reused
# output path still summarizes the most recent run. The raw JSONL remains the
# source of truth and includes full server_info plus raw per-request ITL arrays.
jq -s \
    --argjson target_input_tokens "${INPUT_LEN}" \
    --argjson target_output_tokens "${OUTPUT_LEN}" \
    --argjson target_kv_hit_pct "${TARGET_KV_HIT_PERCENT}" \
    --arg result_file "${OUTPUT_FILE}" \
    --arg console_log "${CONSOLE_LOG}" \
    '
    def percentile($pct):
        if length == 0 then null
        else
            sort as $values |
            (((($values | length) - 1) * $pct) / 100) as $position |
            ($position | floor) as $lower |
            ($position | ceil) as $upper |
            if $lower == $upper then $values[$lower]
            else
                $values[$lower]
                + ($values[$upper] - $values[$lower]) * ($position - $lower)
            end
        end;

    def stats:
        map(select(type == "number")) as $values |
        if ($values | length) == 0 then
            {count: 0, min: null, mean: null, p50: null, p90: null, p95: null, p99: null, max: null}
        else
            {
                count: ($values | length),
                min: ($values | min),
                mean: (($values | add) / ($values | length)),
                p50: ($values | percentile(50)),
                p90: ($values | percentile(90)),
                p95: ($values | percentile(95)),
                p99: ($values | percentile(99)),
                max: ($values | max)
            }
        end;

    def slo($requests; $threshold; $duration):
        ($requests | map(select(.success and (.tpot_ms != null)))) as $eligible |
        ($eligible | map(select(.tpot_ms <= $threshold)) | length) as $passed |
        {
            threshold_ms: $threshold,
            eligible_requests: ($eligible | length),
            passed_requests: $passed,
            pass_rate_pct: (
                if ($eligible | length) > 0
                then 100 * $passed / ($eligible | length)
                else null end
            ),
            request_goodput_req_s: (
                if $duration > 0 then $passed / $duration else null end
            )
        };

    last as $run |
    ($run | [
        range(0; ((.input_lens // []) | length)) as $i |
        (.input_lens[$i] // 0) as $input |
        (.output_lens[$i] // 0) as $output |
        (.cached_tokens[$i] // 0) as $cached |
        (.ttfts[$i] // 0) as $ttft |
        (.itls[$i] // []) as $itls |
        (.latencies[$i]) as $latency |
        ($latency - $ttft) as $decode_time |
        (($itls | add) // 0) as $itl_time |
        {
            request_index: $i,
            dp_rank: .dp_ranks[$i],
            success: (((.errors[$i] // "") == "") and ($output > 0)),
            error: (if (.errors[$i] // "") == "" then null else .errors[$i] end),
            input_tokens: $input,
            output_tokens: $output,
            cached_tokens: $cached,
            cache_hit_rate_pct: (
                if $input > 0 then 100 * $cached / $input else null end
            ),
            ttft_ms: (1000 * $ttft),
            decode_time_ms: (1000 * $decode_time),
            e2e_latency_ms: (1000 * $latency),
            tpot_ms: (
                if $output > 1 then 1000 * $decode_time / ($output - 1)
                else null end
            ),
            mean_itl_ms: (
                if ($itls | length) > 0
                then 1000 * $itl_time / ($itls | length)
                else null end
            ),
            p99_itl_ms: (
                if ($itls | length) > 0
                then 1000 * ($itls | percentile(99))
                else null end
            ),
            max_itl_ms: (
                if ($itls | length) > 0 then 1000 * ($itls | max)
                else null end
            )
        }
    ]) as $requests |
    {
        files: {
            raw_result_jsonl: $result_file,
            console_log: $console_log
        },
        test_configuration: {
            tag: $run.tag,
            backend: $run.backend,
            dataset_name: $run.dataset_name,
            target_input_tokens: $target_input_tokens,
            target_output_tokens: $target_output_tokens,
            target_kv_cache_hit_rate_pct: $target_kv_hit_pct,
            request_rate_req_s: $run.request_rate,
            configured_max_concurrency: $run.max_concurrency
        },
        cache_validation: ($run.cache_validation | del(.failures)),
        exact_cache_configuration: $run.exact_cache_configuration,
        completion: {
            requested_requests: ($requests | length),
            completed_requests: $run.completed,
            failed_requests: (($requests | map(select(.success | not))) | length),
            errors: [$requests[] | select(.success | not) | {
                request_index: .request_index,
                error: .error
            }]
        },
        actual_workload: {
            duration_s: $run.duration,
            total_input_tokens: $run.total_input_tokens,
            total_output_tokens: $run.total_output_tokens,
            total_output_tokens_retokenized: $run.total_output_tokens_retokenized,
            input_tokens_per_request: ([$requests[] | select(.success) | .input_tokens] | stats),
            output_tokens_per_request: ([$requests[] | select(.success) | .output_tokens] | stats)
        },
        kv_cache: {
            target_hit_rate_pct: $target_kv_hit_pct,
            aligned_target_hit_rate_pct: $run.exact_cache_configuration.expected_hit_rate_pct,
            expected_cached_tokens_per_request: $run.exact_cache_configuration.prefix_tokens,
            actual_aggregate_hit_rate_pct: (
                if $run.cache_report.total_prompt_tokens > 0
                then 100 * $run.cache_report.total_cached_tokens / $run.cache_report.total_prompt_tokens
                else null end
            ),
            aligned_target_error_percentage_points: (
                if ($run.cache_report.total_prompt_tokens // 0) <= 0 then null
                else (100 * $run.cache_report.total_cached_tokens / $run.cache_report.total_prompt_tokens) - $run.exact_cache_configuration.expected_hit_rate_pct end
            ),
            total_prompt_tokens: $run.cache_report.total_prompt_tokens,
            total_cached_tokens: $run.cache_report.total_cached_tokens,
            device_cached_tokens: $run.cache_report.device_cached_tokens,
            host_cached_tokens: $run.cache_report.host_cached_tokens,
            storage_cached_tokens: $run.cache_report.storage_cached_tokens,
            storage_backend: $run.cache_report.storage_backend,
            per_dp: ($requests | group_by(.dp_rank) | map({
                dp_rank: .[0].dp_rank,
                requests: length,
                cache_hit_rate_pct: ([.[].cache_hit_rate_pct] | stats),
                ttft_ms: ([.[] | select(.success) | .ttft_ms] | stats),
                tpot_ms: ([.[] | select(.success) | .tpot_ms] | stats)
            })),
            per_request_hit_rate_pct: (
                [$requests[] | select(.success) | .cache_hit_rate_pct] | stats
            )
        },
        concurrency: {
            configured_client_max: $run.max_concurrency,
            actual_average_inflight_requests: $run.concurrency,
            approximate_peak_1s_bucket_requests: $run.max_concurrent_requests,
            server_configured_max_running_requests: $run.server_info.max_running_requests,
            effective_max_running_requests_per_dp: (
                $run.server_info.internal_states[0].effective_max_running_requests_per_dp // null
            )
        },
        throughput: {
            request_req_s: $run.request_throughput,
            input_token_s: $run.input_throughput,
            output_token_s: $run.output_throughput,
            total_token_s: $run.total_throughput,
            peak_output_token_s: $run.max_output_tokens_per_s
        },
        latency_ms: {
            e2e: {
                mean: $run.mean_e2e_latency_ms,
                median: $run.median_e2e_latency_ms,
                std: $run.std_e2e_latency_ms,
                p90: $run.p90_e2e_latency_ms,
                p95: $run.p95_e2e_latency_ms,
                p99: $run.p99_e2e_latency_ms
            },
            ttft: {
                mean: $run.mean_ttft_ms,
                median: $run.median_ttft_ms,
                std: $run.std_ttft_ms,
                p90: $run.p90_ttft_ms,
                p95: $run.p95_ttft_ms,
                p99: $run.p99_ttft_ms
            },
            tpot: {
                mean: $run.mean_tpot_ms,
                median: $run.median_tpot_ms,
                std: $run.std_tpot_ms,
                p90: $run.p90_tpot_ms,
                p95: $run.p95_tpot_ms,
                p99: $run.p99_tpot_ms
            },
            itl: {
                mean: $run.mean_itl_ms,
                median: $run.median_itl_ms,
                std: $run.std_itl_ms,
                p90: $run.p90_itl_ms,
                p95: $run.p95_itl_ms,
                p99: $run.p99_itl_ms,
                max: ([$requests[].max_itl_ms] | max)
            }
        },
        tpot_slo: {
            limit_20_ms: slo($requests; 20; $run.duration),
            limit_50_ms: slo($requests; 50; $run.duration)
        },
        server: {
            version: $run.server_info.version,
            model_path: $run.server_info.model_path,
            device: $run.server_info.device,
            dtype: $run.server_info.dtype,
            quantization: $run.server_info.quantization,
            kv_cache_dtype: $run.server_info.kv_cache_dtype,
            context_length: $run.server_info.context_length,
            max_total_num_tokens: $run.server_info.max_total_num_tokens,
            max_req_input_len: $run.server_info.max_req_input_len,
            page_size: $run.server_info.page_size,
            mem_fraction_static: $run.server_info.mem_fraction_static,
            memory_usage: $run.server_info.memory_usage,
            tp_size: $run.server_info.tp_size,
            pp_size: $run.server_info.pp_size,
            dp_size: $run.server_info.dp_size,
            ep_size: $run.server_info.ep_size,
            enable_dp_attention: $run.server_info.enable_dp_attention,
            disable_radix_cache: $run.server_info.disable_radix_cache,
            radix_eviction_policy: $run.server_info.radix_eviction_policy,
            chunked_prefill_size: $run.server_info.chunked_prefill_size,
            max_prefill_tokens: $run.server_info.max_prefill_tokens,
            schedule_policy: $run.server_info.schedule_policy,
            attention_backend: $run.server_info.attention_backend,
            prefill_attention_backend: $run.server_info.prefill_attention_backend,
            decode_attention_backend: $run.server_info.decode_attention_backend
        }
    }' "${OUTPUT_FILE}" > "${SUMMARY_FILE}"

if [[ "$(jq -r '.kv_cache.actual_aggregate_hit_rate_pct' "${SUMMARY_FILE}")" == "null" ]]; then
    echo "Cache statistics are missing from ${OUTPUT_FILE}; verify that the benchmark client supports --cache-report." >&2
    exit 4
fi

echo
echo "Benchmark summary"
jq . "${SUMMARY_FILE}"

if [[ "$(jq -r '.cache_validation.valid' "${SUMMARY_FILE}")" != true ]]; then
    echo "INVALID benchmark: per-request cache validation failed. Results are for diagnosis only." >&2
    exit 4
fi
exit "${BENCH_STATUS}"
