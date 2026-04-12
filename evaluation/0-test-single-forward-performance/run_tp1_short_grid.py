#!/usr/bin/env python3
"""
Run a TP=1-only DistServe profiling sweep with short prompt lengths included.

This targets the failure mode seen in simulation fitting: the original
`db-identical-req.sqlite` only covered long prompts (256..1024), so the fitted
prefill model extrapolated to negative latency on short ShareGPT prompts.

The sweep uses:
- TP=1 only
- batch sizes: 1, 2, 3, 4
- input lengths: 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024
- output lengths: 16, 64, 256, 1024

Ray is started in local mode with a temp dir under `/users/rh/ray_tmp` so the
run does not depend on `/tmp/ray`.
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
DEFAULT_DB_PATH = Path("/users/rh/DistServe/evaluation/0-test-single-forward-performance/db-identical-req-tp1-short.sqlite")
DEFAULT_RAY_TMP = Path("/users/rh/ray_tmp")

DEFAULT_BATCH_SIZES = (1, 2, 3, 4)
DEFAULT_INPUT_LENS = (4, 8, 16, 32, 64, 128, 256, 512, 768, 1024)
DEFAULT_OUTPUT_LENS = (16, 64, 256, 1024)


def read_tp1_profiles(profile_csv: Path) -> list[tuple[str, int]]:
    rows = list(csv.DictReader(profile_csv.open()))
    result: list[tuple[str, int]] = []
    for row in rows:
        if int(row["tp"]) != 1:
            continue
        result.append((row["model"], int(row["max_num_tokens_per_gpu"])))
    return result


def build_test_params(
    profile_csv: Path,
    batch_sizes: tuple[int, ...],
    input_lens: tuple[int, ...],
    output_lens: tuple[int, ...],
) -> list[TestParamGroup]:
    params: list[TestParamGroup] = []
    for model, num_tokens_limit in read_tp1_profiles(profile_csv):
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
                    tp_world_size=1,
                    max_req_num=256,
                    max_seq_len=2048,
                    use_dummy_weights=True,
                ),
                input_params=input_params,
            )
        )
    return params


def parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TP=1 short-grid DistServe profiling into a fresh SQLite DB.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--profile-csv", type=Path, default=PROFILE_RESULT_CSV)
    parser.add_argument("--ray-temp-dir", type=Path, default=DEFAULT_RAY_TMP)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--measure-rounds", type=int, default=3)
    parser.add_argument("--batch-sizes", type=str, default="1,2,3,4")
    parser.add_argument("--input-lens", type=str, default="4,8,16,32,64,128,256,512,768,1024")
    parser.add_argument("--output-lens", type=str, default="16,64,256,1024")
    args = parser.parse_args()

    args.ray_temp_dir.mkdir(parents=True, exist_ok=True)

    test_params = build_test_params(
        args.profile_csv,
        parse_int_csv(args.batch_sizes),
        parse_int_csv(args.input_lens),
        parse_int_csv(args.output_lens),
    )
    total_cases = sum(len(group.input_params) for group in test_params)

    print(f"db_path={args.db_path}")
    print(f"ray_temp_dir={args.ray_temp_dir}")
    print(f"models={[group.worker_param.model_dir for group in test_params]}")
    print(f"total_cases={total_cases}")

    ray.init(address="local", ignore_reinit_error=True, _temp_dir=str(args.ray_temp_dir))
    run_test_params(
        DistServeSUT,
        str(args.db_path),
        test_params,
        warmup_rounds=args.warmup_rounds,
        measure_rounds=args.measure_rounds,
        skip_duplicated=False,
        store_into_db=True,
    )


if __name__ == "__main__":
    main()
