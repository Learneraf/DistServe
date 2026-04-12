#!/usr/bin/env python3
"""
Run a configurable DistServe profiling sweep into a fresh SQLite DB.

This is intended for fitting the simulator from raw single-forward measurements
while controlling:
- tensor parallel size
- prompt/output length coverage
- batch-size coverage
- Ray temp directory

By default this runner targets a broader TP=1 grid than the original benchmark,
including short prompts that are necessary for robust simulation fitting.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import ray

from structs import InputParam, TestParamGroup, WorkerParam
from run_test_params import run_test_params
from sut.sut_distserve import DistServeSUT


PROFILE_RESULT_CSV = Path("/users/rh/DistServe/simdistserve/profilers/profile_result.csv")
DEFAULT_RAY_TMP = Path("/users/rh/ray_tmp")


def read_profiles(profile_csv: Path, tp_world_size: int) -> list[tuple[str, int]]:
    rows = list(csv.DictReader(profile_csv.open()))
    result: list[tuple[str, int]] = []
    for row in rows:
        if int(row["tp"]) != tp_world_size:
            continue
        result.append((row["model"], int(row["max_num_tokens_per_gpu"])))
    return result


def parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_str_csv(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    return values or None


def build_test_params(
    profile_csv: Path,
    tp_world_size: int,
    batch_sizes: tuple[int, ...],
    input_lens: tuple[int, ...],
    output_lens: tuple[int, ...],
    max_req_num: int,
    max_seq_len: int,
    models: tuple[str, ...] | None,
    gpu_memory_utilization: float,
) -> list[TestParamGroup]:
    params: list[TestParamGroup] = []
    for model, num_tokens_limit in read_profiles(profile_csv, tp_world_size):
        if models is not None and model not in models:
            continue
        input_params = [
            InputParam(
                batch_size=batch_size,
                input_len=input_len,
                output_len=output_len,
            )
            for batch_size in batch_sizes
            for input_len in input_lens
            for output_len in output_lens
            if batch_size * ((((input_len + output_len) + 15) // 16) * 16) <= num_tokens_limit
        ]
        params.append(
            TestParamGroup(
                worker_param=WorkerParam(
                    model_dir=model,
                    tp_world_size=tp_world_size,
                    max_req_num=max_req_num,
                    max_seq_len=max_seq_len,
                    use_dummy_weights=True,
                    gpu_memory_utilization=gpu_memory_utilization,
                ),
                input_params=input_params,
            )
        )
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a configurable DistServe profiling sweep into SQLite.")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("/users/rh/DistServe/evaluation/0-test-single-forward-performance/db-identical-req-tp1-robust.sqlite"),
    )
    parser.add_argument("--profile-csv", type=Path, default=PROFILE_RESULT_CSV)
    parser.add_argument("--ray-temp-dir", type=Path, default=DEFAULT_RAY_TMP)
    parser.add_argument("--tp-world-size", type=int, default=1)
    parser.add_argument("--max-req-num", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--measure-rounds", type=int, default=3)
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8")
    parser.add_argument("--input-lens", type=str, default="4,8,16,32,64,128,256,512,768,1024")
    parser.add_argument("--output-lens", type=str, default="16,32,64,128,256,512,1024")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model paths to include.")
    parser.add_argument("--skip-duplicated", action="store_true", default=False)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--exp-output-dir", type=Path, default=None, help="Optional output directory for per-case .exp files in serving-result format.")
    args = parser.parse_args()

    args.ray_temp_dir.mkdir(parents=True, exist_ok=True)

    test_params = build_test_params(
        args.profile_csv,
        args.tp_world_size,
        parse_int_csv(args.batch_sizes),
        parse_int_csv(args.input_lens),
        parse_int_csv(args.output_lens),
        args.max_req_num,
        args.max_seq_len,
        parse_str_csv(args.models),
        args.gpu_memory_utilization,
    )
    total_cases = sum(len(group.input_params) for group in test_params)

    print(f"db_path={args.db_path}")
    print(f"ray_temp_dir={args.ray_temp_dir}")
    print(f"tp_world_size={args.tp_world_size}")
    print(f"models={[group.worker_param.model_dir for group in test_params]}")
    print(f"cases_per_model={[len(group.input_params) for group in test_params]}")
    print(f"total_cases={total_cases}")

    ray.init(address="local", ignore_reinit_error=True, _temp_dir=str(args.ray_temp_dir))
    run_test_params(
        DistServeSUT,
        str(args.db_path),
        test_params,
        warmup_rounds=args.warmup_rounds,
        measure_rounds=args.measure_rounds,
        skip_duplicated=args.skip_duplicated,
        store_into_db=True,
        exp_output_dir=str(args.exp_output_dir) if args.exp_output_dir is not None else None,
    )


if __name__ == "__main__":
    main()
