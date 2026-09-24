#!/bin/bash

# ================================================================
# CPU
# ================================================================
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

# ================================================================
# Cleanup
# ================================================================
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

pkill -9 python  2>/dev/null || true
pkill -9 sglang 2>/dev/null || true
pkill -9 VLLM   2>/dev/null || true

# ================================================================
# Environment
# ================================================================
export PYTHONPATH=/home/y00951466/sglang/python:$PYTHONPATH
# export PYTHONPATH=/mnt/share/lws/sglang_opt/python:$PYTHONPATH

# source /mnt/share/chenxu/SFA/vendors/custom_transformer/bin/set_env.bash
# source /usr/local/memfabric_hybrid/set_env.sh

# echo "================ MemFabric ENV ================"
# env | sort | grep -E \
# '^(ASCEND_MF_|MF_|HYBM_|URMA_|HCCL_|ASCEND_RT_VISIBLE_DEVICES)' || true

# echo "================ Network ================"
# ip -br addr

# echo "================ MF version ================"
# cat /usr/local/memfabric_hybrid/latest/version.info 2>/dev/null || true

# echo "================ Transfer Protocol Source ================"
# grep -Rni "ASCEND_MF_TRANSFER_PROTOCOL" \
#     /usr/local/memfabric_hybrid/set_env.sh \
#     /home/y00951466/sglang/python/sglang/srt/disaggregation/ascend \
#     2>/dev/null | head -100

# echo "================================================"

export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=0

export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=68
export HCCL_OP_EXPANSION_MODE=AIV
export ACL_DEVICE_SYNC_TIMEOUT=60

# 内存碎片
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

# [FIA]
export ASCEND_USE_FIA=1

# [MLAPO]
export SGLANG_NPU_USE_MLAPO=1
# export SGLANG_NPU_GLM_NEXTN_BF16_KV_CACHE=1

# [DEEPEP]
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=36

# [Prefill Delay]
# export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1
# export SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=200

# export SGLANG_SET_CPU_AFFINITY=1

# [MTP]
export SGLANG_ENABLE_SPEC_V2=1
# export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

export TRANSFORMERS_VERBOSITY=error

# [多机 / 通信]
export HCCL_HOST_SOCKET_PORT_RANGE=auto
# export GLOO_SOCKET_IFNAME=data0.173

# ================================================================
# A5 PD 分离
# ================================================================
# export MF_HYBM_USE_VMM_SEGMENT=1

# export ASCEND_MF_TRANSFER_PROTOCOL=sdma
# export ASCEND_MF_TRANSFER_PROTOCOL="device_urma"

# P 节点作为 MemFabric Store
# export ASCEND_MF_STORE_URL="tcp://141.61.73.103:31001"
# export ASCEND_MF_LOG_LEVEL=3

unset HCCL_IF_IP 2>/dev/null || true
unset HCCL_SOCKET_FAMILY 2>/dev/null || true
unset RANK_TABLE_FILE 2>/dev/null || true

# ================================================================
# Model Config
# ================================================================
MODEL_PATH=/mnt/share/GLM-5.2-W4A8C8-A5-0731-copy 
SERVED_MODEL_NAME=${MODEL_PATH}

# ================================================================
# Cluster Config
#
# PD:
#
#   Prefill:
#     1 node x 8 NPU
#     TP8 + CP8
#
#   Decode:
#     1 node x 8 NPU
#     TP-size = 8
#     DP Attention = 8
#     Effective Attention TP = 8 / 8 = 1
#
# ================================================================

# Prefill: 1 node / 8 NPU
P_IPS=(
  "141.61.73.103"
)

P_IFS=(
  "enp35s0f2"
)

# Decode: 1 node / 8 NPU
D_IPS=(
  "141.61.73.105"
)

D_IFS=(
  "enp35s0f2"
)

NUM_NPUS_PER_NODE=8

P_NNODES=${#P_IPS[@]}
D_NNODES=${#D_IPS[@]}

# ------------------------------------------------
# Parallel Config
# ------------------------------------------------

# Prefill:
# 8 cards total
# TP8 + CP8
P_TP_SIZE=8
P_CP_SIZE=8

# Decode:
# 8 cards total
# global TP size = 8
# DP attention = 8
# effective attention TP = TP / DP = 1
D_TP_SIZE=8
D_DP_SIZE=8

# ================================================================
# Distributed Init
# ================================================================
P_MASTER="${P_IPS[0]}"
D_MASTER="${D_IPS[0]}"

P_DIST_INIT="${P_MASTER}:4567"
D_DIST_INIT="${D_MASTER}:4589"

# ================================================================
# Auto-detect local node
# ================================================================
LOCAL_HOST1=$(hostname -I | awk '{print $1}')
LOCAL_HOST2=$(hostname -I | awk '{print $2}')

echo "Local IPs: ${LOCAL_HOST1} ${LOCAL_HOST2}"

# ================================================================
# Log directory
# ================================================================
LOG_DIR="/home/y00951466/sglang/logs/run_8p8d_pd_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "[run_8p8d_pd] logs will be saved to ${LOG_DIR}"

# ================================================================
# Launch Prefill
#
# Topology:
#
#       8 NPU
#         |
#        TP8
#         |
#   Attention CP8
#
# No PP
# No DP Attention
#
# ================================================================
for i in "${!P_IPS[@]}"; do
  if [[ "$LOCAL_HOST1" == "${P_IPS[$i]}" || "$LOCAL_HOST2" == "${P_IPS[$i]}" ]]; then

    # export HCCL_SOCKET_IFNAME="${P_IFS[$i]}"

    echo "================================================"
    echo "Launching GLM5.2 Prefill"
    echo "node-rank       : ${i}"
    echo "local IPs       : ${LOCAL_HOST1} ${LOCAL_HOST2}"
    echo "dist-init-addr  : ${P_DIST_INIT}"
    echo "nnodes          : ${P_NNODES}"
    echo "tp-size         : ${P_TP_SIZE}"
    echo "cp-size         : ${P_CP_SIZE}"
    echo "HCCL interface  : ${HCCL_SOCKET_IFNAME}"
    echo "GLOO interface  : ${GLOO_SOCKET_IFNAME}"
    echo "================================================"

    export DEEP_USE_MODE="allgather"

    export USE_MX_FP8_QUANT=1
    export SGLANG_LOG_HICACHE_LAYER_TIME=1
    export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1

    # export SGLANG_HICACHE_HYBM_RESERVE_GB=80
    # export SGLANG_HICACHE_IO_ASCENDC=1
    # export SGLANG_HICACHE_HOST_MEM=hybm

    LOG_FILE="${LOG_DIR}/prefill.log"

    python3 -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --host 0.0.0.0 \
      --port 30123 \
      --nnodes ${P_NNODES} \
      --node-rank ${i} \
      --dist-init-addr ${P_DIST_INIT} \
      --tp-size ${P_TP_SIZE} \
      --enable-nsa-prefill-context-parallel \
      --dsa-prefill-cp-mode in-seq-split \
      --attn-cp-size ${P_CP_SIZE} \
      --moe-dense-tp-size 1 \
      --load-balance-method round_robin \
      --disaggregation-mode prefill \
      --disaggregation-transfer-backend ascend \
      --disaggregation-bootstrap-port 8998 \
      --trust-remote-code \
      --attention-backend ascend \
      --device npu \
      --watchdog-timeout 9000 \
      --max-running-requests 36 \
      --mem-fraction-static 0.8 \
      --quantization modelslim \
      --max-prefill-tokens 2048000 \
      --chunked-prefill-size 16384 \
      --kv-cache-dtype "fp8_e4m3" \
      --moe-a2a-backend deepep \
      --deepep-mode auto \
      --skip-server-warmup \
      --tokenizer-worker-num 12 \
      --enable-dynamic-batch-tokenizer \
      --enable-hierarchical-cache \
      --hicache-io-backend kernel_ascend \
      --enable-cache-report \
      --hicache-size 80 \
      --enable-metrics \
      2>&1 | tee "${LOG_FILE}"

    exit ${PIPESTATUS[0]}
  fi
done

# ================================================================
# Launch Decode
#
# Topology:
#
#         8 NPU
#           |
#      global TP = 8
#           |
#     DP Attention = 8
#           |
#   Attention TP = 1
#
# Attention:
#
# rank0 -> request group 0
# rank1 -> request group 1
# ...
# rank7 -> request group 7
#
# MoE:
# DeepEP / A2A across ranks
#
# ================================================================
for i in "${!D_IPS[@]}"; do
  if [[ "$LOCAL_HOST1" == "${D_IPS[$i]}" || "$LOCAL_HOST2" == "${D_IPS[$i]}" ]]; then

    # export HCCL_SOCKET_IFNAME="${D_IFS[$i]}"

    echo "================================================"
    echo "Launching GLM5.2 Decode"
    echo "node-rank       : ${i}"
    echo "local IPs       : ${LOCAL_HOST1} ${LOCAL_HOST2}"
    echo "dist-init-addr  : ${D_DIST_INIT}"
    echo "nnodes          : ${D_NNODES}"
    echo "tp-size         : ${D_TP_SIZE}"
    echo "dp-size         : ${D_DP_SIZE}"
    echo "attention-tp    : $(( D_TP_SIZE / D_DP_SIZE ))"
    echo "HCCL interface  : ${HCCL_SOCKET_IFNAME}"
    echo "GLOO interface  : ${GLOO_SOCKET_IFNAME}"
    echo "================================================"

    export DEEPEP_HCCL_BUFFSIZE=1000
    export HCCL_BUFFSIZE=1000

    # LM-head TP
    export SGLANG_LM_HEAD_TP=8

    # Fine-grained MoE dual stream
    export SGLANG_NPU_FINE_GRAINED_MOE_DUAL_STREAM=1

    # Optional profiling
    # export SGLANG_NPU_PROFILING=1
    # export SGLANG_ENABLE_WAR_BARRIER=1

    LOG_FILE="${LOG_DIR}/decode.log"

    python3 -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --host 0.0.0.0 \
      --port 30000 \
      --nnodes ${D_NNODES} \
      --node-rank ${i} \
      --dist-init-addr ${D_DIST_INIT} \
      --tp-size ${D_TP_SIZE} \
      --dp ${D_DP_SIZE} \
      --enable-dp-attention \
      --moe-dense-tp-size 1 \
      --load-balance-method round_robin \
      --disaggregation-mode decode \
      --disaggregation-transfer-backend ascend \
      --trust-remote-code \
      --attention-backend ascend \
      --device npu \
      --disable-radix-cache \
      --watchdog-timeout 9000 \
      --max-running-requests 192 \
      --mem-fraction-static 0.8 \
      --quantization modelslim \
      --kv-cache-dtype "fp8_e4m3" \
      --moe-a2a-backend deepep \
      --deepep-mode auto \
      --cuda-graph-bs 1 2 3 4 5 6 \
      --context-len 153600 \
      --tokenizer-worker-num 4 \
      --enable-dynamic-batch-tokenizer \
      --speculative-algorithm NEXTN \
      --speculative-num-steps 5 \
      --speculative-eagle-topk 1 \
      --speculative-num-draft-tokens 6 \
      --speculative-draft-kv-cache-dtype bf16 \
      --enable-metrics \
      2>&1 | tee "${LOG_FILE}"

    exit ${PIPESTATUS[0]}
  fi
done

echo "ERROR: local IPs [${LOCAL_HOST1} ${LOCAL_HOST2}] not found."
echo "P_IPS=[${P_IPS[*]}]"
echo "D_IPS=[${D_IPS[*]}]"
exit 1

# ================================================================
# Router
# ================================================================
#
# Example:
#
# python -m sglang_router.launch_router \
#     --pd-disaggregation \
#     --policy cache_aware \
#     --prefill http://141.61.73.103:30123 \
#     --decode http://141.61.73.105:30000 \
#     --host 0.0.0.0 \
#     --port 8818