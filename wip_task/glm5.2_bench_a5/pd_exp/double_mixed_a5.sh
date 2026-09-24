#!/bin/bash

# ================================================================
# CPU
# ================================================================
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000 2>/dev/null || true


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
# Model
# ================================================================
MODEL_PATH=/mnt/share/GLM-5.2-W4A8C8-A5-0731-copy


# ================================================================
# SGLang / custom op
# ================================================================

# 当前双机环境的代码路径先保留
# export PYTHONPATH=/mnt/share/w00964934/kimi-k3/sglang/python:$PYTHONPATH

# 如果 94.139 / 94.143 上也有 PD 已验证的这份代码，
# 建议最终切成完全相同的代码版本：
export PYTHONPATH=/home/y00951466/sglang/python:$PYTHONPATH

# 当前环境如果依赖 custom_transformer，先保留。
# 放在环境变量设置之前，避免 set_env.bash 覆盖下面的配置。
# export PYTHONPATH=/mnt/share/l00890003/codes/sglang/kimi-k3/cann_ops_transformer-1.0.0-py3-none-any:$PYTHONPATH
# source /mnt/share/l00890003/codes/sglang/kimi-k3/vendors/custom_transformer/bin/set_env.bash


# ================================================================
# 清掉旧混部配置里，与已验证 PD 配置冲突的 feature
# ================================================================

unset SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE
# unset SGLANG_K3_SHARED_EXPERTS_ATTN_TP

# unset SGLANG_RAGGED_VERIFY_MODE
# unset SGLANG_DSPARK_FOLDED_PROPOSAL
# unset SGLANG_DSPARK_FOLDED_SAMPLING
# unset SGLANG_DSPARK_STACKED_CTX_KV
# unset SGLANG_DSPARK_EMBED_IN_GRAPH

# unset SGLANG_NPU_USE_FIAS_V2_BSND
# unset SGLANG_USE_FIA_NZ

# unset SGLANG_NPU_MOE_MOE_SITU_MXFP8_FUSED
# unset SGLANG_W4A8_MXFP4_MOE

# unset SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS
# unset SGLANG_NPU_SPARSE_ATTN_A2A

# unset ASCEND_USE_MOJO_MTP

# 防止之前启动过 PD 后，MemFabric 环境残留到混部进程
unset MF_HYBM_USE_VMM_SEGMENT
unset ASCEND_MF_TRANSFER_PROTOCOL
unset ASCEND_GLOBAL_RESOURCE_CONFIG
unset ASCEND_MF_STORE_URL
unset ASCEND_MF_LOG_LEVEL

# export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1
# ================================================================
# 已验证配置：Common Runtime
# ================================================================

export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=0

export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=68
export ACL_DEVICE_SYNC_TIMEOUT=60

# 已验证配置这里是 AIV，不再使用 CCU_SCHED
export HCCL_OP_EXPANSION_MODE=AIV

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

export TRANSFORMERS_VERBOSITY=error

# 多机 socket port 自动选择
export HCCL_HOST_SOCKET_PORT_RANGE=auto


# ================================================================
# Attention
# ================================================================

export ASCEND_USE_FIA=1

# 关键变化：
# 原配置 MLAPO=0 + FIAS_V2
# 改成已验证配置 MLAPO=1
export SGLANG_NPU_USE_MLAPO=1


# ================================================================
# DeepEP / MoE
# ================================================================

# PD 已验证配置为 36
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=36

export DEEPEP_HCCL_BUFFSIZE=1000
export HCCL_BUFFSIZE=1000

# Decode 已验证配置
export SGLANG_NPU_FINE_GRAINED_MOE_DUAL_STREAM=1


# ================================================================
# Speculative / MTP
# ================================================================

export SGLANG_ENABLE_SPEC_V2=1

# 不再使用：
#   DSPARK
#   ASCEND_USE_MOJO_MTP
#
# 改用 launch 参数里的 NEXTN


# ================================================================
# LM Head
# ================================================================

# 已验证 PD decode:
#   TP8 -> SGLANG_LM_HEAD_TP=8
#
# 当前双机混部:
#   TP16 -> 对应设置 16
# export SGLANG_LM_HEAD_TP=16


# ================================================================
# Network
# ================================================================

D_IP=(
    '141.61.73.103'
    '141.61.73.105'
)

LOCAL_IPS=$(hostname -I 2>/dev/null)

# export HCCL_SOCKET_IFNAME=lo
# export GLOO_SOCKET_IFNAME=lo


# ================================================================
# Topology
#
# 2 nodes x 8 NPU
#
# global TP = 16
# DP attention = 2
#
# effective Attention TP = 16 / 2 = 8
# ================================================================

TP_SIZE=16
DP_SIZE=2
NNODES=2

DIST_INIT_ADDR="${D_IP[0]}:5000"


# ================================================================
# Launch
# ================================================================

for i in "${!D_IP[@]}"; do
    for ip in ${LOCAL_IPS}; do

        if [[ "${ip}" == "${D_IP[$i]}" ]]; then
            LOCAL_MAIN_IP="${D_IP[$i]}"

NET_IF=$(
python3 - "${LOCAL_MAIN_IP}" <<'PY'
import socket
import fcntl
import struct
import sys

target_ip = sys.argv[1]

for _, ifname in socket.if_nameindex():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        addr = socket.inet_ntoa(
            fcntl.ioctl(
                s.fileno(),
                0x8915,  # SIOCGIFADDR
                struct.pack("256s", ifname.encode()[:15])
            )[20:24]
        )
        if addr == target_ip:
            print(ifname)
            sys.exit(0)
    except OSError:
        pass

sys.exit(1)
PY
)

            if [[ -z "${NET_IF}" ]]; then
                echo "ERROR: cannot find interface for ${LOCAL_MAIN_IP}"
                ip -o -4 addr show
                exit 1
            fi

            export GLOO_SOCKET_IFNAME="${NET_IF}"
            export HCCL_SOCKET_IFNAME="${NET_IF}"

            echo "Network interface: ${NET_IF}"

            echo "================================================"
            echo "Launching GLM-5.2 mixed deployment"
            echo "node-rank       : ${i}"
            echo "local IPs       : ${LOCAL_IPS}"
            echo "dist-init-addr  : ${DIST_INIT_ADDR}"
            echo "nnodes          : ${NNODES}"
            echo "tp-size         : ${TP_SIZE}"
            echo "dp-size         : ${DP_SIZE}"
            echo "attention-tp    : $(( TP_SIZE / DP_SIZE ))"
            echo "HCCL interface  : ${HCCL_SOCKET_IFNAME}"
            echo "GLOO interface  : ${GLOO_SOCKET_IFNAME}"
            echo "================================================"

            python3 -m sglang.launch_server \
                --model-loader-extra-config '{"enable_multithread_load": true}' \
                --model-path "${MODEL_PATH}" \
                --served-model-name "${MODEL_PATH}" \
                --tokenizer-path "${MODEL_PATH}" \
                --host 0.0.0.0 \
                --port 30000 \
                \
                --nnodes ${NNODES} \
                --node-rank ${i} \
                --dist-init-addr ${DIST_INIT_ADDR} \
                \
                --tp-size ${TP_SIZE} \
                --dp ${DP_SIZE} \
                --enable-dp-attention \
                --enable-dp-attention-local-control-broadcast \
                \
                --moe-dense-tp-size 1 \
                --load-balance-method round_robin \
                \
                --trust-remote-code \
                --attention-backend ascend \
                --device npu \
                --dtype bfloat16 \
                --mamba-ssm-dtype bfloat16 \
                \
                --quantization modelslim \
                --kv-cache-dtype fp8_e4m3 \
                \
                --mem-fraction-static 0.8 \
                --context-len 153600 \
                --chunked-prefill-size 16384 \
                --max-running-requests 36 \
                \
                --moe-a2a-backend deepep \
                --deepep-mode auto \
                \
                --cuda-graph-bs 8 \
                \
                --tokenizer-worker-num 4 \
                --enable-dynamic-batch-tokenizer \
                \
                --speculative-algorithm NEXTN \
                --speculative-num-steps 5 \
                --speculative-eagle-topk 1 \
                --speculative-num-draft-tokens 6 \
                --speculative-draft-kv-cache-dtype bf16 \
                \
                --watchdog-timeout 9000

            exit $?
        fi
    done
done


echo "ERROR: current machine is not in D_IP."
echo "local IPs: ${LOCAL_IPS}"
echo "D_IP: ${D_IP[*]}"
exit 1