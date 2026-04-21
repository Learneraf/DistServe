#!/bin/bash
set -euo pipefail

PYTHON="/users/rh/miniconda3/envs/distserve/bin/python"
REPO_ROOT="/users/rh/DistServe"
DATASET_ROOT="/users/rh/DistServe/simdistserve/dataset/splits/sharegpt_four_models_common_ascend1900_seed0"
BENCH_ROOT="/users/rh/DistServe/evaluation/2-benchmark-serving/result_four_models_common_seed0"
SIM_ROOT="/users/rh/DistServe/simdistserve/benchmarks/results/latency/distserve_cuda_four_models_common/organized_data"
SLO_ROOT="/users/rh/DistServe/simdistserve/benchmarks/results/slo/distserve_cuda_four_models_common"
PROFILE_ROOT="/users/rh/DistServe/simdistserve/estimators/profiled_data/distserve-cuda"
PROFILE_DECODE="$PROFILE_ROOT/fit_params_live_decode_four_models_common.json"
PROFILE_FINAL="$PROFILE_ROOT/fit_params_live_four_models_common.json"
RATES=(1 1.5 2 2.5 3 3.5 4)
NUM_PROMPTS=120
PORT=8400

declare -A MODEL_PATHS=(
    [llama_1B]="/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B/converted_bin_v2"
    [llama_3B]="/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B/converted_bin_v2"
    [llama_7B]="/users/rh/.cache/modelscope/hub/models/LLM-Research/llama-2-7b/converted_bin_v2"
    [llama_8B]="/users/rh/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B/converted_bin_v2"
)

declare -A DATASET_DIRS=(
    [llama_1B]="llama-3.2-1B"
    [llama_3B]="llama-3.2-3B"
    [llama_7B]="llama-2-7b"
    [llama_8B]="llama-3.1-8B"
)

MODELS=(llama_1B llama_3B llama_7B llama_8B)
SERVER_PID=""

cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

wait_for_port() {
    local port="$1"
    local timeout_s="${2:-600}"
    local start_ts
    start_ts=$(date +%s)
    while true; do
        if "${PYTHON}" - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(1.0)
    sys.exit(0 if sock.connect_ex(("127.0.0.1", port)) == 0 else 1)
PY
        then
            sleep 5
            return 0
        fi
        if (( $(date +%s) - start_ts >= timeout_s )); then
            return 1
        fi
        sleep 2
    done
}

run_actual_benchmark() {
    local alias="$1"
    local model_path="${MODEL_PATHS[$alias]}"
    local dataset_dir="${DATASET_DIRS[$alias]}"
    local dataset_ds="$DATASET_ROOT/$dataset_dir/test.ds"
    local server_log="$BENCH_ROOT/$alias/server.log"

    mkdir -p "$BENCH_ROOT/$alias"
    echo "============================================================"
    echo "Benchmarking $alias"
    echo "  model:   $model_path"
    echo "  dataset: $dataset_ds"
    echo "============================================================"

    (
        cd "$REPO_ROOT"
        CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m distserve.api_server.distserve_api_server \
            --host 0.0.0.0 \
            --port "$PORT" \
            --model "$model_path" \
            --tokenizer "$model_path" \
            --context-tensor-parallel-size 1 \
            --context-pipeline-parallel-size 1 \
            --decoding-tensor-parallel-size 1 \
            --decoding-pipeline-parallel-size 1 \
            --block-size 16 \
            --max-num-blocks-per-req 128 \
            --gpu-memory-utilization 0.95 \
            --swap-space 16 \
            --context-sched-policy fcfs \
            --context-max-batch-size 128 \
            --context-max-tokens-per-batch 8192 \
            --decoding-sched-policy fcfs \
            --decoding-max-batch-size 1024 \
            --decoding-max-tokens-per-batch 65536 \
            >"$server_log" 2>&1
    ) &
    SERVER_PID="$!"

    if ! wait_for_port "$PORT" 600; then
        echo "Server failed to become ready for $alias. Log: $server_log" >&2
        exit 1
    fi

    (
        cd "$REPO_ROOT/evaluation/2-benchmark-serving"
        CUDA_VISIBLE_DEVICES=0 "${PYTHON}" ./2-benchmark-serving.py \
            --backend distserve \
            --host 127.0.0.1 \
            --dataset "$dataset_ds" \
            --seed 0 \
            --num-prompts-req-rates "[(120, 1), (120, 1.5), (120, 2), (120, 2.5), (120, 3), (120, 3.5), (120, 4)]" \
            --exp-result-root "$BENCH_ROOT" \
            --exp-result-dir "$alias" \
            --exp-result-prefix distserve
    )

    cleanup
    SERVER_PID=""
}

run_simulator() {
    local alias="$1"
    local model_path="${MODEL_PATHS[$alias]}"
    local dataset_dir="${DATASET_DIRS[$alias]}"
    local workload_jsonl="$DATASET_ROOT/$dataset_dir/test.jsonl"

    for rate in "${RATES[@]}"; do
        local output_dir="$SIM_ROOT/$alias/rate_$rate"
        mkdir -p "$output_dir"
        (
            cd "$REPO_ROOT"
            SIMDISTSERVE_DISTSERVE_PROFILE="$PROFILE_FINAL" "${PYTHON}" simdistserve/benchmarks/simulate_dist.py \
                --backend distserve \
                --model "$model_path" \
                --seed 0 \
                --rate "$rate" \
                --N "$NUM_PROMPTS" \
                --arrival poisson \
                --workload "$workload_jsonl" \
                --output "$output_dir/sharegpt.json.sim.csv" \
                --name "$alias/rate_$rate" \
                --output-request-latency "$output_dir/request_latency.csv" \
                --slo-scales "[1.0]" \
                --output-request-event "$output_dir/request_event.csv" \
                --output-request-info "$output_dir/request_info.csv"
        )
    done
}

run_slo_compare() {
    local alias="$1"
    for rate in "${RATES[@]}"; do
        local exp_file="$BENCH_ROOT/$alias/distserve-$NUM_PROMPTS-$rate.exp"
        local csv_file="$SIM_ROOT/$alias/rate_$rate/request_latency.csv"
        local case_root="$SLO_ROOT/$alias/rate_$rate"
        mkdir -p "$case_root/actual" "$case_root/sim"
        (
            cd "$REPO_ROOT/simdistserve/benchmarks"
            "${PYTHON}" merged_analyze.py \
                --exp-file "$exp_file" \
                --csv-file "$csv_file" \
                --prefill-slo 1.0 \
                --decode-slo 1.0 \
                --total-slo 1.0 \
                --exp-output "$case_root/actual/exp_analysis.txt" \
                --csv-output "$case_root/sim/csv_analysis.txt" \
                --compare-output "$case_root/comparison.txt"
        )
    done
}

mkdir -p "$BENCH_ROOT" "$SIM_ROOT" "$SLO_ROOT"

for alias in "${MODELS[@]}"; do
    run_actual_benchmark "$alias"
done

(
    cd "$REPO_ROOT"
    "${PYTHON}" simdistserve/estimators/fit_params/retrain_distserve_live_decode.py \
        --results-root "$BENCH_ROOT" \
        --base-profile "$PROFILE_ROOT/fit_params_live.json" \
        --output "$PROFILE_DECODE"
)

(
    cd "$REPO_ROOT"
    "${PYTHON}" simdistserve/estimators/fit_params/retrain_distserve_live_prefill.py \
        --results-root "$BENCH_ROOT" \
        --base-profile "$PROFILE_DECODE" \
        --output "$PROFILE_FINAL"
)

for alias in "${MODELS[@]}"; do
    run_simulator "$alias"
done

for alias in "${MODELS[@]}"; do
    run_slo_compare "$alias"
done

"${PYTHON}" - "$BENCH_ROOT" "$SIM_ROOT" "$SLO_ROOT" "$PROFILE_FINAL" <<'PY'
import csv
import json
import sys
from pathlib import Path

bench_root = Path(sys.argv[1])
sim_root = Path(sys.argv[2])
slo_root = Path(sys.argv[3])
profile_final = Path(sys.argv[4])
rates = ["1", "1.5", "2", "2.5", "3", "3.5", "4"]
models = ["llama_1B", "llama_3B", "llama_7B", "llama_8B"]

def exp_stats(path: Path):
    data = json.loads(path.read_text())
    total = len(data)
    prefill = decode = total_ok = both = 0
    for req in data:
        lifecycle = {event["event_type"]: event["timestamp"] for event in req.get("lifecycle_events", [])}
        decode_time = lifecycle.get("decoding_end", 0) - lifecycle.get("decoding_begin", 0)
        prefill_ok = req.get("ftl", 0) <= 1.0
        decode_ok = decode_time <= 1.0
        total_met = req.get("latency", 0) <= 1.0
        both_ok = prefill_ok and decode_ok
        prefill += int(prefill_ok)
        decode += int(decode_ok)
        total_ok += int(total_met)
        both += int(both_ok)
    return {
        "prefill_slo_rate": 100.0 * prefill / total,
        "decode_slo_rate": 100.0 * decode / total,
        "total_slo_rate": 100.0 * total_ok / total,
        "both_slo_rate": 100.0 * both / total,
    }

def sim_stats(path: Path):
    rows = list(csv.DictReader(path.open()))
    total = len(rows)
    prefill = decode = total_ok = both = 0
    for row in rows:
        ftl = float(row["first_token_latency"]) / 1000.0
        dec = float(row["decoding_latency"]) / 1000.0
        ttl = float(row["total_latency"]) / 1000.0
        prefill_ok = ftl <= 1.0
        decode_ok = dec <= 1.0
        total_met = ttl <= 1.0
        both_ok = prefill_ok and decode_ok
        prefill += int(prefill_ok)
        decode += int(decode_ok)
        total_ok += int(total_met)
        both += int(both_ok)
    return {
        "prefill_slo_rate": 100.0 * prefill / total,
        "decode_slo_rate": 100.0 * decode / total,
        "total_slo_rate": 100.0 * total_ok / total,
        "both_slo_rate": 100.0 * both / total,
    }

summary_rows = []
metric_names = [
    "prefill_slo_rate",
    "decode_slo_rate",
    "total_slo_rate",
    "both_slo_rate",
]
for model in models:
    for rate in rates:
        exp = exp_stats(bench_root / model / f"distserve-120-{rate}.exp")
        sim = sim_stats(sim_root / model / f"rate_{rate}" / "request_latency.csv")
        row = {
            "model": model,
            "rate": rate,
        }
        for metric in metric_names:
            row[f"actual_{metric}"] = exp[metric]
            row[f"sim_{metric}"] = sim[metric]
            row[f"diff_{metric}"] = exp[metric] - sim[metric]
        summary_rows.append(row)

summary_csv = slo_root / "comparison_summary.csv"
summary_json = slo_root / "comparison_summary.json"
summary_csv.parent.mkdir(parents=True, exist_ok=True)

fieldnames = list(summary_rows[0].keys())
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)

aggregate = {}
for metric in metric_names:
    diffs = [abs(row[f"diff_{metric}"]) for row in summary_rows]
    aggregate[metric] = {
        "mean_abs_diff_pct": sum(diffs) / len(diffs),
        "max_abs_diff_pct": max(diffs),
    }

summary = {
    "profile": str(profile_final),
    "benchmark_root": str(bench_root),
    "sim_root": str(sim_root),
    "summary_csv": str(summary_csv),
    "cases": summary_rows,
    "aggregate_abs_diff_pct": aggregate,
}
summary_json.write_text(json.dumps(summary, indent=2) + "\n")

print(json.dumps(summary["aggregate_abs_diff_pct"], indent=2))
print(summary_csv)
print(summary_json)
PY

echo "CUDA refit and verification completed."
echo "Benchmark root: $BENCH_ROOT"
echo "Simulator root: $SIM_ROOT"
echo "SLO summary:    $SLO_ROOT/comparison_summary.csv"
echo "Profile file:   $PROFILE_FINAL"
