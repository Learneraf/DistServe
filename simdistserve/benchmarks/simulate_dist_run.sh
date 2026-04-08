RATE=1
python simulate_dist.py \
    --backend distserve \
    --model /users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B/converted_bin_v2 \
    --seed 0 \
    --rate $RATE \
    --N 100 \
    --arrival poisson \
    --workload /users/rh/DistServe/evaluation/2-benchmark-serving/sharegpt.json \
    --output ./results/latency/vllm_ascend/organized_data/llama_3B/rate_$RATE/sharegpt.json.sim.csv \
    --name llama_3B/rate_$RATE \
    --output-request-latency ./results/latency/vllm_ascend/organized_data/llama_3B/rate_$RATE/request_latency.csv \
    --slo-scales "[1.0]" \
    --output-request-event ./results/latency/vllm_ascend/organized_data/llama_3B/rate_$RATE/request_event.csv \
    --output-request-info ./results/latency/vllm_ascend/organized_data/llama_3B/rate_$RATE/request_info.csv
