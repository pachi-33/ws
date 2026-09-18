SGLANG_ROOT=/home/y00951466/sglang \
MODEL=/mnt/share/GLM-5.2-W4A8C8-A5-0731-copy \
HOST=127.0.0.1 PORT=8818 \
bash run_sglang_bench_cache_glm.sh \
  --input-len 131072 \
  --output-len 1024 \
  --target-kv-hit-percent 90 \
  --num-requests 24 \
  --max-concurrency 24 \
  --request-rate inf \
  --dp-size auto \
  --page-size auto