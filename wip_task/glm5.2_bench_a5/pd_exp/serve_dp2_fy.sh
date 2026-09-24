#!/bin/bash

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=10
sysctl -w kernel.numa_balancing=0
# 下面这行若内核不支持可注释掉，或加判断
sysctl -w kernel.sched_migration_cost_ns=50000 2>/dev/null || true
# export SGLANG_SET_CPU_AFFINITY=1
# export TRITON_ALL_BLOCKS_PARALLEL=1
export SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE=1
#MODEL_PATH=/mnt/share/fuyong/weights/kimi-k3
#MODEL_PATH=/mnt/share/recipes/kimi/kimi_k3_27
#DRAFT_MODEL_PATH=/mnt/share/fuyong/weights/kimi-k3-dspark
#DRAFT_MODEL_PATH=/mnt/share/recipes/kimi/RadixArk-Kimi-K3-DSpark

#MODEL_PATH=/mnt/share/w00964934/kimi-k3/kimi_k3_mla_10layers_weight
MODEL_PATH=/mnt/share/GLM-5.2-W4A8C8-A5-0731-copy 
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY

pkill -9 python
pkill -9 sglang

export SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE=1
export SGLANG_K3_SHARED_EXPERTS_ATTN_TP=1
#export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_DSPARK_FOLDED_PROPOSAL=0
export SGLANG_DSPARK_FOLDED_SAMPLING=0
export SGLANG_DSPARK_STACKED_CTX_KV=0
export SGLANG_DSPARK_EMBED_IN_GRAPH=0

# mc's opti
#export SGLANG_DSPARK_FUSED_LOCAL_TOP1=1
#export SGLANG_NPU_FUSED_KDA_VERIFY_GATES=1
#export SGLANG_NPU_FUSED_KDA_RAGGED_IO=1
#export SGLANG_NPU_FUSED_KDA_ONORM=1
#export SGLANG_NPU_REUSE_KDA_VERIFY_METADATA=1
#export SGLANG_SPECULATIVE_FUSED_MLP_SYNC=1

# use fias v2
#export SGLANG_NPU_USE_TRITON_MLA_NZ_STORE=1
export SGLANG_NPU_USE_MLAPO=0
export SGLANG_USE_FIA_NZ=0
export SGLANG_NPU_USE_FIAS_V2_BSND=1
# export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/mnt/share/hanwlax/expert_stats

# mxfp8 quant
export SGLANG_NPU_MOE_MOE_SITU_MXFP8_FUSED=1

# experts-stream
export SGLANG_NPU_FINE_GRAINED_MOE_DUAL_STREAM=True

unset ASCEND_LAUNCH_BLOCKING
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

export DEEP_NORMAL_MODE_USE_INT8_QUANT=0

# deepep
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16
export DEEPEP_HCCL_BUFFSIZE=1000

export ASCEND_USE_FIA=1
export SGLANG_W4A8_MXFP4_MOE=1

# profiling
export SGLANG_TORCH_PROFILER_DIR="./sglang_profile/"
export SGLANG_PROFILE_WITH_STACK=False

# 框架路径

export PYTHONPATH=/mnt/share/w00964934/kimi-k3/sglang/python:$PYTHONPATH

export PYTHONPATH=/mnt/share/l00890003/codes/sglang/kimi-k3/cann_ops_transformer-1.0.0-py3-none-any:$PYTHONPATH
source /mnt/share/l00890003/codes/sglang/kimi-k3/vendors/custom_transformer/bin/set_env.bash


export SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS=0
export SGLANG_NPU_SPARSE_ATTN_A2A=1
export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
export ASCEND_USE_MOJO_MTP=1

#export SGLANG_NPU_USE_FLASH_MLA=1
#export SGLANG_MOE_DBG=1

# 定义 Decode 节点 IP 列表（4 机）
D_IP=('141.61.94.139' '141.61.94.143')

# 获取本机所有 IP 地址
LOCAL_IPS=$(hostname -I 2>/dev/null)

# 设置网络接口环境变量
export HCCL_SOCKET_IFNAME=enp35s0f2
export GLOO_SOCKET_IFNAME=data0.173

# 检查是否属于 Decode 节点
for i in "${!D_IP[@]}"; do
    for ip in $LOCAL_IPS; do
        if [[ "$ip" == "${D_IP[$i]}" ]]; then
            echo "Decode -> ${D_IP[$i]} (node-rank $i)"
            # 启动服务
            sglang serve \
                --model-loader-extra-config '{"enable_multithread_load": true}' \
	        --dist-init-addr ${D_IP[0]}:5000 --nnodes 2 --node-rank $i \
                --model-path $MODEL_PATH \
                --tokenizer-path $MODEL_PATH \
                --trust-remote-code \
                --attention-backend ascend \
                --device npu \
                --dtype bfloat16 \
                --mamba-ssm-dtype bfloat16 \
                --tp-size 16 \
                --enable-dp-attention \
                --dp-size 2 \
		--shared-experts-tp-size 4 \
                --enable-dp-lm-head \
                --mem-fraction-static 0.93 \
                --chunked-prefill-size 16384 \
		--cuda-graph-bs 4 8 \
                --max-running-requests 16 \
                --host 0.0.0.0 \
                --port 30000 \
                --max-total-tokens 18000 \
		--disable-cuda-graph \
                --enable-shared-experts-attn-tp \
                --enable-dense-mlp-attn-tp \
                --moe-a2a-backend deepep \
                --deepep-mode auto \
                --linear-attn-verify-backend triton \
		--speculative-algorithm DSPARK \
                --speculative-draft-model-quantization unquant \
                --speculative-dspark-block-size 3 \
                --speculative-draft-attention-backend ascend \
                --speculative-draft-kv-cache-dtype bf16 \
                --speculative-eagle-topk 1 \
                --watchdog-timeout 9000
            exit 0
        fi
    done
done


