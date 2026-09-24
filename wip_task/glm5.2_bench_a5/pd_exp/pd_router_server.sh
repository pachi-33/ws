python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://141.61.73.103:30123 8998\
    --decode http://141.61.73.105:30000 \
    --host 0.0.0.0 \
    --port 8818