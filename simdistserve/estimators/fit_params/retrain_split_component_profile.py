#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simdistserve.estimators.fit_params import (
    retrain_vllm_ascend_decode as ascend_decode_fit,
)
from simdistserve.estimators.fit_params import (
    retrain_vllm_ascend_prefill as ascend_prefill_fit,
)
from simdistserve.benchmarks.component_fit_utils import (
    MODEL_CONFIGS,
    DEFAULT_MODELS,
    DEFAULT_RATES,
    add_handoff_and_queue_models,
    actual_ftl_ms,
    actual_total_latency_ms,
    build_compute_only_profile,
    collect_decode_compute_samples,
    collect_prefill_compute_samples,
    default_output_profile,
    default_seed_profile,
    dump_json,
    extract_actual_migration_ms,
    extract_request_components,
    fit_decode_compute,
    fit_huber_linear_model_ms,
    fit_linear_model_ms,
    fit_prefill_compute,
    handoff_tokens_for_request,
    load_actual_requests,
    load_jsonl,
    model_key_for_alias,
    prefill_first_token_visible_immediately_for_backend,
    resolve_actual_exp_path,
    run_simulation,
    workload_path_for_split,
    write_profile,
)


WORKLOAD_ROOT_DEFAULT = Path("/users/rh/DistServe/simdistserve/dataset/splits/sharegpt_four_models_common_ascend1900_seed0")
ASCEND_COMPUTE_GRID_ROOT_DEFAULT = Path("/users/rh/ascend_data/ascend_compute_grid")


def default_real_fit_root(backend: str) -> Path:
    if backend == "distserve":
        preferred = Path("/users/rh/DistServe/evaluation/2-benchmark-serving/result_split/fit")
        if preferred.exists():
            return preferred
        fallback = Path("/users/rh/DistServe/evaluation/2-benchmark-serving/result/fit")
        if fallback.exists():
            return fallback
        return preferred
    if backend == "vllm_ascend":
        return Path("/users/rh/ascend_data/ascend_vllm_split/fit")
    raise ValueError(f"Unsupported backend: {backend}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit split-aware component profiles for DistServe CUDA or vLLM Ascend.")
    parser.add_argument("--backend", choices=["distserve", "vllm_ascend"], required=True)
    parser.add_argument("--seed-profile", type=Path, default=None)
    parser.add_argument("--workload-root", type=Path, default=WORKLOAD_ROOT_DEFAULT)
    parser.add_argument("--real-fit-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--python-bin", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=120)
    parser.add_argument("--arrival", type=str, default="poisson")
    parser.add_argument("--cv", type=float, default=1.0)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=DEFAULT_MODELS)
    parser.add_argument("--rates", nargs="+", default=DEFAULT_RATES)
    parser.add_argument(
        "--ascend-compute-grid-root",
        type=Path,
        default=ASCEND_COMPUTE_GRID_ROOT_DEFAULT,
        help="Real Ascend compute-grid root used to fit Stage-1 compute coefficients for backend=vllm_ascend.",
    )
    parser.add_argument(
        "--grid-ttft-stat",
        type=str,
        default="p50_ttft",
        choices=["mean_ttft", "p50_ttft", "p95_ttft"],
        help="Which TTFT statistic to use from Ascend compute-grid case summaries.",
    )
    parser.add_argument(
        "--exclude-last-interval",
        dest="exclude_last_interval",
        action="store_true",
        default=True,
        help="Drop the last emitted decode interval when fitting Ascend Stage-1 decode from request_metrics.json.",
    )
    parser.add_argument(
        "--include-last-interval",
        dest="exclude_last_interval",
        action="store_false",
        help="Keep the last emitted decode interval when fitting Ascend Stage-1 decode from request_metrics.json.",
    )
    parser.add_argument(
        "--min-interval-ms",
        type=float,
        default=5.0,
        help="Ignore implausibly small decode intervals when fitting Ascend Stage-1 decode from request_metrics.json.",
    )
    args = parser.parse_args()

    if args.seed_profile is None:
        args.seed_profile = default_seed_profile(args.backend)
    if args.real_fit_root is None:
        args.real_fit_root = default_real_fit_root(args.backend)
    if args.output is None:
        args.output = default_output_profile(args.backend)
    if args.work_dir is None:
        args.work_dir = args.output.parent / f"{args.backend}_split_component_fit_work"
    if args.python_bin is None:
        args.python_bin = "python3"
    return args


def ensure_aligned_lengths(workload_items, actual_requests, components, context: str) -> None:
    expected = len(workload_items)
    actual = len(actual_requests)
    sim = len(components)
    if actual != expected or sim != expected:
        raise ValueError(
            f"{context}: length mismatch workload={expected} actual={actual} sim={sim}. "
            "This flow assumes the real exp order matches the workload order exactly."
        )


def _ascend_compute_grid_dir_for_model(results_root: Path, model_alias: str) -> Path:
    model_dir = results_root / MODEL_CONFIGS[model_alias]["ascend_real_dir"]
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Missing Ascend compute-grid directory for {model_alias}: {model_dir}"
        )
    return model_dir


def collect_stage1_ascend_compute_samples(
    results_root: Path,
    model_alias: str,
    ttft_stat: str,
    exclude_last_interval: bool,
    min_interval_ms: float,
) -> tuple[list, list]:
    model_dir = _ascend_compute_grid_dir_for_model(results_root, model_alias)
    case_summary_path = model_dir / "case_summaries.json"
    request_metrics_path = model_dir / "request_metrics.json"
    if not case_summary_path.is_file():
        raise FileNotFoundError(f"Missing Ascend compute-grid case summaries: {case_summary_path}")
    if not request_metrics_path.is_file():
        raise FileNotFoundError(f"Missing Ascend compute-grid request metrics: {request_metrics_path}")

    prefill_samples_by_model, ignored_prefill = ascend_prefill_fit.extract_grid_samples(
        case_summary_path,
        ttft_stat=ttft_stat,
    )
    decode_samples_by_model, ignored_decode = ascend_decode_fit.extract_grid_samples(
        request_metrics_path,
        exclude_last_interval=exclude_last_interval,
        min_interval_ms=min_interval_ms,
    )
    if ignored_prefill:
        print(
            f"Stage1 ignored {len(ignored_prefill)} unknown Ascend prefill model entries from {case_summary_path}",
            flush=True,
        )
    if ignored_decode:
        print(
            f"Stage1 ignored {len(ignored_decode)} unknown Ascend decode model entries from {request_metrics_path}",
            flush=True,
        )

    model_key = model_key_for_alias(model_alias)
    prefill_samples = prefill_samples_by_model.get(model_key, [])
    decode_samples = decode_samples_by_model.get(model_key, [])
    if not prefill_samples:
        raise ValueError(f"No Ascend prefill compute-grid samples found for {model_alias} in {case_summary_path}")
    if not decode_samples:
        raise ValueError(f"No Ascend decode compute-grid samples found for {model_alias} in {request_metrics_path}")
    return prefill_samples, decode_samples


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    prefill_coeffs_by_model: dict[str, list[float]] = {}
    decode_coeffs_by_model: dict[str, list[float]] = {}
    fit_summary: dict[str, object] = {
        "backend": args.backend,
        "seed_profile": str(args.seed_profile),
        "workload_root": str(args.workload_root),
        "real_fit_root": str(args.real_fit_root),
        "output": str(args.output),
        "models": {},
    }

    stage1_root = args.work_dir / "stage1_seed_compute"
    for model_alias in args.models:
        workload_path = workload_path_for_split(args.workload_root, model_alias, "fit")
        if not workload_path.exists():
            raise FileNotFoundError(f"Missing workload: {workload_path}")

        if args.backend == "vllm_ascend":
            print(f"STAGE1 backend={args.backend} model={model_alias} source={args.ascend_compute_grid_root}", flush=True)
            prefill_samples, decode_samples = collect_stage1_ascend_compute_samples(
                results_root=args.ascend_compute_grid_root,
                model_alias=model_alias,
                ttft_stat=args.grid_ttft_stat,
                exclude_last_interval=args.exclude_last_interval,
                min_interval_ms=args.min_interval_ms,
            )
            prefill_coeffs, prefill_metrics = ascend_prefill_fit.fit_relative_error_live_batch(prefill_samples)
            decode_coeffs, decode_metrics = ascend_decode_fit.fit_grid_interval_model(decode_samples)
        else:
            prefill_samples = []
            decode_samples = []
            for rate in args.rates:
                output_dir = stage1_root / model_alias / f"rate_{rate}"
                print(f"STAGE1 backend={args.backend} model={model_alias} rate={rate}", flush=True)
                run_simulation(
                    backend=args.backend,
                    python_bin=args.python_bin,
                    profile=args.seed_profile,
                    model_alias=model_alias,
                    rate=str(rate),
                    output_dir=output_dir,
                    workload_path=workload_path,
                    num_prompts=args.num_prompts,
                    seed=args.seed,
                    arrival=args.arrival,
                    cv=args.cv,
                    handoff_delay_ms=0.0,
                    handoff_delay_per_token_ms=0.0,
                )
                worker_csv = output_dir / "worker_event.csv"
                prefill_samples.extend(collect_prefill_compute_samples(worker_csv))
                decode_samples.extend(collect_decode_compute_samples(worker_csv))

            prefill_coeffs, prefill_metrics = fit_prefill_compute(prefill_samples)
            decode_coeffs, decode_metrics = fit_decode_compute(decode_samples)
        prefill_coeffs_by_model[model_alias] = prefill_coeffs
        decode_coeffs_by_model[model_alias] = decode_coeffs
        fit_summary["models"].setdefault(model_alias, {})
        fit_summary["models"][model_alias]["prefill_compute_fit"] = {
            "coeffs": prefill_coeffs,
            "metrics": prefill_metrics,
        }
        fit_summary["models"][model_alias]["decode_compute_fit"] = {
            "coeffs": decode_coeffs,
            "metrics": decode_metrics,
        }
        if args.backend == "vllm_ascend":
            fit_summary["models"][model_alias]["stage1_source"] = {
                "kind": "ascend_compute_grid",
                "root": str(args.ascend_compute_grid_root),
                "grid_ttft_stat": args.grid_ttft_stat,
                "exclude_last_interval": args.exclude_last_interval,
                "min_interval_ms": args.min_interval_ms,
            }

    compute_only_profile = build_compute_only_profile(
        backend=args.backend,
        prefill_coeffs_by_model=prefill_coeffs_by_model,
        decode_coeffs_by_model=decode_coeffs_by_model,
    )
    compute_only_profile_path = args.work_dir / "compute_only_profile.json"
    write_profile(compute_only_profile_path, compute_only_profile)

    handoff_coeffs_by_model: dict[str, list[float]] = {}
    stage2_root = args.work_dir / "stage2_compute_only_fit"
    for model_alias in args.models:
        workload_path = workload_path_for_split(args.workload_root, model_alias, "fit")
        workload_items = load_jsonl(workload_path)
        rows = []
        targets = []
        exact_count = 0
        proxy_count = 0

        for rate in args.rates:
            exp_path = resolve_actual_exp_path(args.backend, args.real_fit_root, model_alias, args.num_prompts, str(rate))
            if not exp_path.exists():
                raise FileNotFoundError(f"Missing real fit trace: {exp_path}")

            output_dir = stage2_root / model_alias / f"rate_{rate}"
            print(f"STAGE2 backend={args.backend} model={model_alias} rate={rate}", flush=True)
            run_simulation(
                backend=args.backend,
                python_bin=args.python_bin,
                profile=compute_only_profile_path,
                model_alias=model_alias,
                rate=str(rate),
                output_dir=output_dir,
                workload_path=workload_path,
                num_prompts=args.num_prompts,
                seed=args.seed,
                arrival=args.arrival,
                cv=args.cv,
                handoff_delay_ms=0.0,
                handoff_delay_per_token_ms=0.0,
            )

            components = extract_request_components(output_dir / "request_event.csv", backend=args.backend)
            actual_requests = load_actual_requests(exp_path)
            ensure_aligned_lengths(workload_items, actual_requests, components, context=str(exp_path))

            for workload_item, actual_request, component in zip(workload_items, actual_requests, components):
                exact_migration_ms = extract_actual_migration_ms(actual_request)
                if exact_migration_ms is not None:
                    target_ms = exact_migration_ms
                    exact_count += 1
                else:
                    target_ms = actual_ftl_ms(actual_request) - (
                        component.prefill_queue_ms
                        + component.prefill_compute_ms
                        + component.decode_queue_first_ms
                        + component.first_decode_compute_ms
                    )
                    target_ms = max(target_ms, 0.0)
                    proxy_count += 1

                rows.append([1.0, handoff_tokens_for_request(args.backend, workload_item)])
                targets.append(target_ms)

        handoff_coeffs, handoff_metrics = fit_huber_linear_model_ms(rows, targets)
        handoff_coeffs = [max(float(handoff_coeffs[0]), 0.0), max(float(handoff_coeffs[1]), 0.0)]
        handoff_coeffs_by_model[model_alias] = handoff_coeffs
        fit_summary["models"][model_alias]["handoff_fit"] = {
            "fit_method": "huber_irls",
            "coeffs": handoff_coeffs,
            "metrics": handoff_metrics,
            "exact_target_count": exact_count,
            "proxy_target_count": proxy_count,
        }

    handoff_profile = build_compute_only_profile(
        backend=args.backend,
        prefill_coeffs_by_model=prefill_coeffs_by_model,
        decode_coeffs_by_model=decode_coeffs_by_model,
    )
    for model_alias in args.models:
        tp_profile = handoff_profile[model_key_for_alias(model_alias)]["1"]
        tp_profile["handoff_delay_ms"] = float(handoff_coeffs_by_model[model_alias][0])
        tp_profile["handoff_delay_per_token_ms"] = float(handoff_coeffs_by_model[model_alias][1])
    handoff_profile_path = args.work_dir / "compute_handoff_profile.json"
    write_profile(handoff_profile_path, handoff_profile)

    queue_ftl_coeffs_by_model: dict[str, list[float]] = {}
    queue_total_coeffs_by_model: dict[str, list[float]] = {}
    queue_piecewise_by_model: dict[str, dict[str, Any]] = {}
    stage3_root = args.work_dir / "stage3_handoff_fit"
    prefill_first_token_visible_immediately = prefill_first_token_visible_immediately_for_backend(args.backend)
    piecewise_rate_threshold = 2.5
    for model_alias in args.models:
        workload_path = workload_path_for_split(args.workload_root, model_alias, "fit")
        workload_items = load_jsonl(workload_path)
        rows_ftl = []
        targets_ftl = []
        rows_total = []
        targets_total = []
        piecewise_rows_ftl = {"low": [], "high": []}
        piecewise_targets_ftl = {"low": [], "high": []}
        piecewise_rows_total = {"low": [], "high": []}
        piecewise_targets_total = {"low": [], "high": []}

        handoff_delay_ms, handoff_delay_per_token_ms = handoff_coeffs_by_model[model_alias]
        for rate in args.rates:
            exp_path = resolve_actual_exp_path(args.backend, args.real_fit_root, model_alias, args.num_prompts, str(rate))
            if not exp_path.exists():
                raise FileNotFoundError(f"Missing real fit trace: {exp_path}")

            output_dir = stage3_root / model_alias / f"rate_{rate}"
            print(f"STAGE3 backend={args.backend} model={model_alias} rate={rate}", flush=True)
            run_simulation(
                backend=args.backend,
                python_bin=args.python_bin,
                profile=handoff_profile_path,
                model_alias=model_alias,
                rate=str(rate),
                output_dir=output_dir,
                workload_path=workload_path,
                num_prompts=args.num_prompts,
                seed=args.seed,
                arrival=args.arrival,
                cv=args.cv,
                handoff_delay_ms=handoff_delay_ms,
                handoff_delay_per_token_ms=handoff_delay_per_token_ms,
            )

            components = extract_request_components(output_dir / "request_event.csv", backend=args.backend)
            actual_requests = load_actual_requests(exp_path)
            ensure_aligned_lengths(workload_items, actual_requests, components, context=str(exp_path))
            piecewise_bucket = "low" if float(rate) <= piecewise_rate_threshold else "high"

            for actual_request, component in zip(actual_requests, components):
                if prefill_first_token_visible_immediately:
                    target_ftl_ms = actual_ftl_ms(actual_request) - component.prefill_compute_ms
                    ftl_row = [
                        1.0,
                        component.prefill_queue_ms,
                    ]
                else:
                    target_ftl_ms = actual_ftl_ms(actual_request) - (
                        component.prefill_compute_ms
                        + component.handoff_service_ms
                        + component.first_decode_compute_ms
                    )
                    ftl_row = [
                        1.0,
                        component.prefill_queue_ms,
                        component.handoff_queue_ms,
                        component.decode_queue_first_ms,
                    ]
                target_total_ms = actual_total_latency_ms(actual_request) - (
                    component.prefill_compute_ms
                    + component.handoff_service_ms
                    + component.decode_compute_total_ms
                )
                total_row = [
                    1.0,
                    component.prefill_queue_ms,
                    component.handoff_queue_ms,
                    component.decode_queue_total_ms,
                ]
                rows_ftl.append(ftl_row)
                targets_ftl.append(float(target_ftl_ms))
                rows_total.append(total_row)
                targets_total.append(float(target_total_ms))
                piecewise_rows_ftl[piecewise_bucket].append(ftl_row)
                piecewise_targets_ftl[piecewise_bucket].append(float(target_ftl_ms))
                piecewise_rows_total[piecewise_bucket].append(total_row)
                piecewise_targets_total[piecewise_bucket].append(float(target_total_ms))

        queue_ftl_coeffs, queue_ftl_metrics = fit_linear_model_ms(rows_ftl, targets_ftl)
        queue_total_coeffs, queue_total_metrics = fit_linear_model_ms(rows_total, targets_total)
        piecewise_summary: dict[str, Any] = {
            "split_feature": "rate",
            "rate_threshold": piecewise_rate_threshold,
        }
        piecewise_profile: dict[str, Any] = {
            "split_feature": "rate",
            "rate_threshold": piecewise_rate_threshold,
        }
        for bucket in ("low", "high"):
            bucket_ftl_coeffs, bucket_ftl_metrics = fit_linear_model_ms(
                piecewise_rows_ftl[bucket],
                piecewise_targets_ftl[bucket],
            )
            bucket_total_coeffs, bucket_total_metrics = fit_linear_model_ms(
                piecewise_rows_total[bucket],
                piecewise_targets_total[bucket],
            )
            piecewise_summary[bucket] = {
                "ftl_coeffs": bucket_ftl_coeffs,
                "ftl_metrics": bucket_ftl_metrics,
                "total_coeffs": bucket_total_coeffs,
                "total_metrics": bucket_total_metrics,
            }
            piecewise_profile[bucket] = {
                "ftl": bucket_ftl_coeffs,
                "total": bucket_total_coeffs,
            }
        queue_ftl_coeffs_by_model[model_alias] = queue_ftl_coeffs
        queue_total_coeffs_by_model[model_alias] = queue_total_coeffs
        queue_piecewise_by_model[model_alias] = piecewise_profile
        fit_summary["models"][model_alias]["queue_fit_ftl"] = {
            "coeffs": queue_ftl_coeffs,
            "metrics": queue_ftl_metrics,
        }
        fit_summary["models"][model_alias]["queue_fit_total"] = {
            "coeffs": queue_total_coeffs,
            "metrics": queue_total_metrics,
        }
        fit_summary["models"][model_alias]["queue_fit_piecewise"] = piecewise_summary

    final_profile = add_handoff_and_queue_models(
        profile=build_compute_only_profile(
            backend=args.backend,
            prefill_coeffs_by_model=prefill_coeffs_by_model,
            decode_coeffs_by_model=decode_coeffs_by_model,
        ),
        handoff_coeffs_by_model=handoff_coeffs_by_model,
        queue_ftl_coeffs_by_model=queue_ftl_coeffs_by_model,
        queue_total_coeffs_by_model=queue_total_coeffs_by_model,
        queue_piecewise_by_model=queue_piecewise_by_model,
    )
    write_profile(args.output, final_profile)
    fit_summary["final_profile"] = str(args.output)
    fit_summary_path = args.work_dir / "fit_summary.json"
    dump_json(fit_summary_path, fit_summary)

    print(f"WROTE {args.output}")
    print(f"WROTE {fit_summary_path}")


if __name__ == "__main__":
    main()
