unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

# export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7

# export ASCEND_LAUNCH_BLOCKING=1

# export ASCEND_GLOBAL_LOG_LEVEL=1
# export ASCEND_SLOG_PRINT_TO_STDOUT=1
# export HCCL_ENTRY_LOG_ENABLE=1

# export ASCEND_SLOG_PRINT_TO_STDOUT=0 #日志打屏, 可选
# export ASCEND_GLOBAL_LOG_LEVEL=1 #日志级别常用 1 INFO级别; 3 ERROR级别

export ENABLE_PROFILING=0
export PROFILING_BS=5
export PROFILING_STAGE="decode" # "decode" # "prefill"
export PROFILING_step=5

# export PYTHONPATH=/mnt/share/l00951279/sglang/python:$PYTHONPATH
# export PYTHONPATH=/home/y00951466/sglang-my/python:$PYTHONPATH/
# export PYTHONPATH=/home/y00951466/sglang-a5-optim/python:$PYTHONPATH/
export PYTHONPATH=/home/y00951466/sglang/python:$PYTHONPATH/

export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=300
export HCCL_BUFFSIZE=2048
export HCCL_OP_EXPANSION_MODE=""AIV""
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0

# export SGLANG_HICACHE_HYBM_RESERVE_GB=60
# export SGLANG_HICACHE_IO_ASCENDC=1
# export SGLANG_HICACHE_HOST_MEM=hybm

export ACL_DEVICE_SYNC_TIMEOUT=300

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export STREAMS_PER_DEVICE=32

# MODEL_PATH=/mnt/share/w00936111/GLM-5.2-W4A4C8-mxfp4-A5-0731
# MODEL_PATH=/mnt/share/y00951466/GLM-5.2-W4A4C8-mxfp4-A5-0731-clip
# MODEL_PATH=/home/weights/GLM-5.2-W4A8C8-A5-0731
# MODEL_PATH=/mnt/share/GLM-5.2-W4A8C8-0713      
# MODEL_PATH=/mnt/share/w00936111/GLM-5.2-W8A8C8-mxfp8
# MODEL_PATH=/mnt/share/GLM-5.2-W8A8C8-mxfp8 

MODEL_PATH=/mnt/share/GLM-5.2-W4A8C8-A5-0731-copy 

export ASCEND_USE_FIA=1
export SGLANG_NPU_USE_MLAPO=0
export SGLANG_ENABLE_SPEC_V2=1

export TRANSFORMERS_VERBOSITY=error

export HCCL_HOST_SOCKET_PORT_RANGE=auto

export HCCL_SOCKET_IFNAME=lo
# unset HCCL_SOCKET_IFNAME
export GLOO_SOCKET_IFNAME=lo

# export DEEPEP_NORMAL_LONG_SEQ_ROUND=9
# export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=8192
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=0

unset HCCL_IF_IP 2>/dev/null || true
unset HCCL_SOCKET_FAMILY 2>/dev/null || true
unset RANK_TABLE_FILE 2>/dev/null || true

SERVED_MODEL_NAME=glm52
SERVER_HOST=127.0.0.1
SERVER_PORT=8818

python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
--served-model-name "${SERVED_MODEL_NAME}" \
--host "${SERVER_HOST}" \
--port "${SERVER_PORT}" \
--tp-size 8 \
--trust-remote-code \
--attention-backend ascend \
--device npu \
--watchdog-timeout 9000 \
--max-running-requests 20 \
--mem-fraction-static 0.85 \
--quantization modelslim \
--max-prefill-tokens 2048000 \
--chunked-prefill-size 16384 \
--enable-metrics \
--cuda-graph-bs 4 8 12 16 20 \
--load-balance-method round_robin \
--speculative-algorithm EAGLE \
--speculative-num-steps 4 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 5 \
--speculative-draft-kv-cache-dtype bf16 \
--kv-cache-dtype "fp8_e4m3" \
--enable-hierarchical-cache \
--hicache-io-backend kernel_ascend \
--enable-cache-report \
--hicache-size 60



# --moe-dense-tp-size 1 \
# --moe-a2a-backend deepep \
# --deepep-mode auto \


# --enable-draft-prefetch

# --enable-nsa-prefill-context-parallel \
# --dsa-prefill-cp-mode in-seq-split \
# --attn-cp-size 4 \

# --enable-draft-prefetch \
# --skip-draft-prefetch-seq-lens-cpu-sync \

# --dp-size 8 \
# --enable-dp-attention

# --kv-cache-dtype "fp8_e4m3"
# --speculative-draft-model-quantization modelslim
# --max-total-tokens 327680 \
# --cuda-graph-max-bs-decode 8 \
# --cuda-graph-max-bs-prefill 8 \


