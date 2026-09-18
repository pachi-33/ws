#!/usr/bin/env bash

# GLM-5.2 / Ascend A5 / 8 NPU
# Single-node unified serving: the same server handles both prefill and decode.
#
# Default topology:
#   TP8 + DP Attention 8
#   - Attention TP = TP / DP = 1
#   - MoE communication still spans the 8 ranks through DeepEP
#
# Long-prefill topology:
#   PARALLEL_MODE=prefill_cp bash glm52_single_node_unified.sh
#   - TP8 + CP8, DP Attention disabled
#
# Plain TP baseline:
#   PARALLEL_MODE=tp bash glm52_single_node_unified.sh
#
# DP Attention and prefill CP intentionally cannot be enabled together here.
# On one 8-NPU node both configurations consume the same rank dimension, and
# the GLM-5.2 unified-serving path recommends prefill CP with DP=1.

set -o pipefail

# ================================================================
# User config (all values can be overridden from the shell)
# ================================================================
MODEL_PATH="${MODEL_PATH:-/mnt/share/w00936111/GLM-5.2-W8A8C8-mxfp8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_PATH}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SGLANG_PYTHON_PATH="${SGLANG_PYTHON_PATH:-/mnt/share/chenxu/codes/sglang/python}"
CUSTOM_TRANSFORMER_ENV="${CUSTOM_TRANSFORMER_ENV:-/mnt/share/chenxu/SFA/vendors/custom_transformer/bin/set_env.bash}"
MEMFABRIC_ENV="${MEMFABRIC_ENV:-/usr/local/memfabric_hybrid/set_env.sh}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
HCCL_IFACE="${HCCL_IFACE:-enp35s0f2}"
GLOO_IFACE="${GLOO_IFACE:-data0.173}"

TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-8}"
CP_SIZE="${CP_SIZE:-8}"
PARALLEL_MODE="${PARALLEL_MODE:-dp_attention}" # dp_attention | prefill_cp | tp

MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.94}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-192}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-2048000}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
CONTEXT_LEN="${CONTEXT_LEN:-153600}"
HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-160}"

# Feature switches: 1=enable, 0=disable
ENABLE_HICACHE="${ENABLE_HICACHE:-1}"
ENABLE_MTP="${ENABLE_MTP:-1}"
ENABLE_CUDA_GRAPH="${ENABLE_CUDA_GRAPH:-1}"
ENABLE_DEEPEP="${ENABLE_DEEPEP:-1}"
ENABLE_DYNAMIC_TOKENIZER="${ENABLE_DYNAMIC_TOKENIZER:-1}"
ENABLE_CACHE_REPORT="${ENABLE_CACHE_REPORT:-1}"
ENABLE_METRICS="${ENABLE_METRICS:-1}"

LOG_ROOT="${LOG_ROOT:-/mnt/share/chenxu/codes/sglang/logs}"
LOG_DIR="${LOG_ROOT}/run_single_node_unified_$(date +%Y%m%d_%H%M%S)"

# ================================================================
# Preflight validation
# ================================================================
die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -d "${MODEL_PATH}" ]] || die "MODEL_PATH does not exist: ${MODEL_PATH}"
[[ -d "${SGLANG_PYTHON_PATH}" ]] || die "SGLANG_PYTHON_PATH does not exist: ${SGLANG_PYTHON_PATH}"
[[ -f "${CUSTOM_TRANSFORMER_ENV}" ]] || die "Missing env file: ${CUSTOM_TRANSFORMER_ENV}"
[[ -f "${MEMFABRIC_ENV}" ]] || die "Missing env file: ${MEMFABRIC_ENV}"

case "${PARALLEL_MODE}" in
  dp_attention)
    (( TP_SIZE % DP_SIZE == 0 )) || die "TP_SIZE must be divisible by DP_SIZE"
    ;;
  prefill_cp)
    [[ "${DP_SIZE}" == "1" || "${DP_SIZE}" == "8" ]] || \
      die "Do not provide a separate DP topology in prefill_cp mode"
    ;;
  tp)
    ;;
  *)
    die "PARALLEL_MODE must be one of: dp_attention, prefill_cp, tp"
    ;;
esac

mkdir -p "${LOG_DIR}" || die "Cannot create log directory: ${LOG_DIR}"

# ================================================================
# CPU and process environment
# ================================================================
# These tuning operations require root. Failure is reported but does not abort.
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null \
  || echo "WARN: unable to set CPU governor"
sysctl -w vm.swappiness=0 || true
sysctl -w kernel.numa_balancing=0 || true
sysctl -w kernel.sched_migration_cost_ns=50000 || true

unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING

# Deliberately do not pkill python/sglang globally. Clean up an old service
# manually after confirming its PID if the port or NPU is occupied.

export PYTHONPATH="${SGLANG_PYTHON_PATH}:${PYTHONPATH:-}"

# shellcheck disable=SC1090
source "${CUSTOM_TRANSFORMER_ENV}"
# shellcheck disable=SC1090
source "${MEMFABRIC_ENV}"

export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=68
export HCCL_OP_EXPANSION_MODE=AIV
export ACL_DEVICE_SYNC_TIMEOUT=60
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export HCCL_SOCKET_IFNAME="${HCCL_IFACE}"
export GLOO_SOCKET_IFNAME="${GLOO_IFACE}"
export SGLANG_SET_CPU_AFFINITY=1
export TRANSFORMERS_VERBOSITY=error

# General Ascend optimizations from the original P/D deployment.
export ASCEND_USE_FIA=1
export SGLANG_NPU_USE_MLAPO=1
export USE_MX_FP8_QUANT=1

# DeepEP / communication buffers.
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=36
export DEEPEP_HCCL_BUFFSIZE=250
export HCCL_BUFFSIZE=70

# Decode-side optimizations that are also applicable to unified serving.
export SGLANG_LM_HEAD_TP="${TP_SIZE}"
export SGLANG_NPU_FINE_GRAINED_MOE_DUAL_STREAM=1

# MTP implementation switch. The command-line MTP args are added below.
if [[ "${ENABLE_MTP}" == "1" ]]; then
  export SGLANG_ENABLE_SPEC_V2=1
else
  unset SGLANG_ENABLE_SPEC_V2
fi

# HiCache uses local host/HYBM memory, but unified serving does not need the
# cross-node PD KV store or device_urma transfer endpoint.
if [[ "${ENABLE_HICACHE}" == "1" ]]; then
  export MF_HYBM_USE_VMM_SEGMENT=1
  export SGLANG_LOG_HICACHE_LAYER_TIME=1
  export SGLANG_HICACHE_HYBM_RESERVE_GB="${HICACHE_SIZE_GB}"
  export SGLANG_HICACHE_IO_ASCENDC=1
  export SGLANG_HICACHE_HOST_MEM=hybm
fi
unset ASCEND_MF_STORE_URL ASCEND_MF_TRANSFER_PROTOCOL ASCEND_MF_LOG_LEVEL
unset HCCL_IF_IP HCCL_SOCKET_FAMILY RANK_TABLE_FILE

# ================================================================
# Assemble compatible feature groups
# ================================================================
PARALLEL_ARGS=()
case "${PARALLEL_MODE}" in
  dp_attention)
    PARALLEL_ARGS+=(
      --dp "${DP_SIZE}"
      --enable-dp-attention
      --enable-dp-attention-local-control-broadcast
    )
    ;;
  prefill_cp)
    # This is the flag spelling used by the supplied Ascend branch.
    PARALLEL_ARGS+=(
      --enable-nsa-prefill-context-parallel
      --dsa-prefill-cp-mode in-seq-split
      --attn-cp-size "${CP_SIZE}"
    )
    export DEEP_USE_MODE=allgather
    export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1
    ;;
  tp)
    ;;
esac

HICACHE_ARGS=()
if [[ "${ENABLE_HICACHE}" == "1" ]]; then
  HICACHE_ARGS+=(
    --enable-hierarchical-cache
    --hicache-io-backend kernel_ascend
    --hicache-size "${HICACHE_SIZE_GB}"
  )
fi

MTP_ARGS=()
if [[ "${ENABLE_MTP}" == "1" ]]; then
  MTP_ARGS+=(
    --speculative-algorithm NEXTN
    --speculative-num-steps 5
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 6
    --speculative-draft-kv-cache-dtype bf16
    --enable-draft-prefetch
    --skip-draft-prefetch-seq-lens-cpu-sync
  )
fi

CUDA_GRAPH_ARGS=()
if [[ "${ENABLE_CUDA_GRAPH}" == "1" ]]; then
  CUDA_GRAPH_ARGS+=(--cuda-graph-bs 1 2 3 4 5 6)
fi

MOE_ARGS=()
if [[ "${ENABLE_DEEPEP}" == "1" ]]; then
  MOE_ARGS+=(--moe-a2a-backend deepep --deepep-mode auto)
fi

OBSERVABILITY_ARGS=()
[[ "${ENABLE_CACHE_REPORT}" == "1" ]] && OBSERVABILITY_ARGS+=(--enable-cache-report)
[[ "${ENABLE_METRICS}" == "1" ]] && OBSERVABILITY_ARGS+=(--enable-metrics)

TOKENIZER_ARGS=(--tokenizer-worker-num 12)
[[ "${ENABLE_DYNAMIC_TOKENIZER}" == "1" ]] && TOKENIZER_ARGS+=(--enable-dynamic-batch-tokenizer)

# ================================================================
# Launch one unified server
# ================================================================
LOG_FILE="${LOG_DIR}/unified.log"

echo "================================================"
echo "Launching GLM-5.2 single-node unified server"
echo "model             : ${MODEL_PATH}"
echo "parallel mode      : ${PARALLEL_MODE}"
echo "tp-size            : ${TP_SIZE}"
if [[ "${PARALLEL_MODE}" == "dp_attention" ]]; then
  echo "dp-size            : ${DP_SIZE}"
  echo "attention-tp       : $(( TP_SIZE / DP_SIZE ))"
elif [[ "${PARALLEL_MODE}" == "prefill_cp" ]]; then
  echo "attention-cp       : ${CP_SIZE}"
fi
echo "HiCache / MTP      : ${ENABLE_HICACHE} / ${ENABLE_MTP}"
echo "DeepEP / graph     : ${ENABLE_DEEPEP} / ${ENABLE_CUDA_GRAPH}"
echo "HCCL / GLOO iface  : ${HCCL_IFACE} / ${GLOO_IFACE}"
echo "listen             : ${HOST}:${PORT}"
echo "log                : ${LOG_FILE}"
echo "================================================"

"${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp-size "${TP_SIZE}" \
  "${PARALLEL_ARGS[@]}" \
  --moe-dense-tp-size 1 \
  --load-balance-method round_robin \
  --trust-remote-code \
  --attention-backend ascend \
  --device npu \
  --watchdog-timeout 9000 \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --quantization modelslim \
  --max-prefill-tokens "${MAX_PREFILL_TOKENS}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --context-len "${CONTEXT_LEN}" \
  --kv-cache-dtype fp8_e4m3 \
  "${MOE_ARGS[@]}" \
  "${CUDA_GRAPH_ARGS[@]}" \
  "${TOKENIZER_ARGS[@]}" \
  "${HICACHE_ARGS[@]}" \
  "${MTP_ARGS[@]}" \
  "${OBSERVABILITY_ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"

exit "${PIPESTATUS[0]}"
