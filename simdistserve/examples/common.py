from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import pandas as pd
import simpy

from simdistserve.base.organize_data import (
    calculate_per_request_latency,
    organize_request_df,
    organize_request_event_df,
    organize_worker_event_df,
)
from simdistserve.base.request import Request
from simdistserve.base.scheduler import put_requests_with_interarrivals
from simdistserve.base.worker import WorkerConfig
from simdistserve.clusters.disagg import DisaggCluster
from simdistserve.clusters.vllm import VLLMCluster
from simdistserve.constants import ModelTypes
from simdistserve.estimators.memory_estimator import get_max_num_tokens, is_model_runnable
from simdistserve.estimators.time_estimator import (
    distserve_profile_data,
    vllm_ascend_profile_data,
    vllm_profile_data,
)

Backend = Literal["distserve", "vllm", "vllm_ascend"]


@dataclass(frozen=True)
class ExampleConfig:
    backend: Backend
    model: str = "huggyllama/llama-7b"
    tp_prefill: int = 1
    pp_prefill: int = 1
    tp_decode: int = 1
    pp_decode: int = 1
    prefill_max_batch_size: int = 10 ** 7
    decode_max_batch_size: int = 10 ** 7
    prefill_max_tokens_cap: int | None = None
    decode_max_tokens_cap: int | None = None
    enable_chunked_prefill: bool = False
    decode_back_pressure: float = 0.9
    handoff_delay_ms: float | None = None
    handoff_delay_per_token_ms: float | None = None
    handoff_capacity: int | None = None
    prefill_first_token_visible_immediately: bool | None = None

    @property
    def model_type(self) -> ModelTypes:
        return ModelTypes.model_str_to_object(self.model)

    @property
    def total_gpus(self) -> int:
        if self.backend == "vllm":
            return self.tp_prefill * self.pp_prefill
        return self.tp_prefill * self.pp_prefill + self.tp_decode * self.pp_decode


@dataclass
class SimulationResult:
    config: ExampleConfig
    requests: list[Request]
    request_df: pd.DataFrame
    request_event_df: pd.DataFrame
    latency_df: pd.DataFrame
    worker_df: pd.DataFrame


def build_requests(request_specs: Sequence[tuple[int, int]]) -> list[Request]:
    return [
        Request(env=None, req_id=req_id, prefill_length=prefill_len, output_lens=output_len)
        for req_id, (prefill_len, output_len) in enumerate(request_specs)
    ]


def fixed_interarrivals_from_rate(num_requests: int, rate: float) -> list[float]:
    if num_requests <= 0:
        return []
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}.")
    delay_ms = 1000.0 / rate
    return [0.0] + [delay_ms] * (num_requests - 1)


def _validate_runnability(config: ExampleConfig) -> tuple[int, int]:
    model_type = config.model_type
    formal_model_name = ModelTypes.formalize_model_name(model_type)
    profile_maps = {
        "distserve": distserve_profile_data,
        "vllm": vllm_profile_data,
        "vllm_ascend": vllm_ascend_profile_data,
    }
    profile_map = profile_maps[config.backend]
    if formal_model_name not in profile_map:
        available_models = ", ".join(sorted(profile_map))
        raise ValueError(
            f"{config.backend} does not have profiled coefficients for {formal_model_name}. "
            f"Available models: {available_models}"
        )
    supported_tps = sorted(profile_map[formal_model_name])
    if str(config.tp_prefill) not in supported_tps:
        raise ValueError(
            f"{config.backend} does not have profiled coefficients for prefill TP={config.tp_prefill} "
            f"on {formal_model_name}. Supported TP values: {supported_tps}"
        )
    if config.backend != "vllm" and str(config.tp_decode) not in supported_tps:
        raise ValueError(
            f"{config.backend} does not have profiled coefficients for decode TP={config.tp_decode} "
            f"on {formal_model_name}. Supported TP values: {supported_tps}"
        )
    if not is_model_runnable(model_type, config.tp_prefill, config.pp_prefill):
        raise ValueError(
            f"{config.model} is not runnable with prefill TP={config.tp_prefill}, PP={config.pp_prefill}."
        )
    prefill_max_tokens = get_max_num_tokens(model_type, config.tp_prefill, config.pp_prefill)
    if config.prefill_max_tokens_cap is not None:
        prefill_max_tokens = min(prefill_max_tokens, config.prefill_max_tokens_cap)

    if config.backend == "vllm":
        decode_max_tokens = prefill_max_tokens
        if config.decode_max_tokens_cap is not None:
            decode_max_tokens = min(decode_max_tokens, config.decode_max_tokens_cap)
        return prefill_max_tokens, decode_max_tokens

    if not is_model_runnable(model_type, config.tp_decode, config.pp_decode):
        raise ValueError(
            f"{config.model} is not runnable with decode TP={config.tp_decode}, PP={config.pp_decode}."
        )
    decode_max_tokens = get_max_num_tokens(model_type, config.tp_decode, config.pp_decode)
    if config.decode_max_tokens_cap is not None:
        decode_max_tokens = min(decode_max_tokens, config.decode_max_tokens_cap)
    return prefill_max_tokens, decode_max_tokens


def _build_cluster(env: simpy.Environment, config: ExampleConfig):
    prefill_max_tokens, decode_max_tokens = _validate_runnability(config)
    worker_config: WorkerConfig = WorkerConfig(
        model_type=config.model_type,
        TP=config.tp_prefill,
        TP_Prefill=config.tp_prefill,
        TP_Decode=config.tp_decode,
        prefill_max_batch_size=config.prefill_max_batch_size,
        decode_max_batch_size=config.decode_max_batch_size,
        prefill_max_tokens=prefill_max_tokens,
        decode_max_tokens=decode_max_tokens,
        enable_chunked_prefill=config.enable_chunked_prefill,
        decode_back_pressure=config.decode_back_pressure,
        engine_type=config.backend,
        prefill_generates_first_token=(config.backend == "vllm_ascend"),
    )
    if config.backend != "vllm":
        if config.handoff_delay_ms is not None:
            worker_config["handoff_delay_ms"] = config.handoff_delay_ms
        if config.handoff_delay_per_token_ms is not None:
            worker_config["handoff_delay_per_token_ms"] = config.handoff_delay_per_token_ms
        if config.handoff_capacity is not None:
            worker_config["handoff_capacity"] = config.handoff_capacity
        if config.prefill_first_token_visible_immediately is not None:
            worker_config["prefill_first_token_visible_immediately"] = (
                config.prefill_first_token_visible_immediately
            )

    if config.backend == "vllm":
        return VLLMCluster(env=env, PP=config.pp_prefill, worker_configs=worker_config)

    return DisaggCluster(
        env=env,
        PP_prefill=config.pp_prefill,
        PP_decode=config.pp_decode,
        worker_configs=worker_config,
    )


def run_simulation(
    request_specs: Sequence[tuple[int, int]],
    interarrival_ms: Sequence[float],
    config: ExampleConfig,
) -> SimulationResult:
    if len(request_specs) != len(interarrival_ms):
        raise ValueError(
            f"request_specs and interarrival_ms must have the same length, got "
            f"{len(request_specs)} vs {len(interarrival_ms)}."
        )

    requests = build_requests(request_specs)
    env = simpy.Environment()
    cluster = _build_cluster(env, config)
    cluster.run()
    put_requests_with_interarrivals(env, cluster.scheduler, interarrival_ms, requests)
    env.run()

    request_df = organize_request_df(requests)
    request_event_df = organize_request_event_df(requests).sort_values(
        ["req_id", "start_time"],
        kind="mergesort",
    )
    latency_df = calculate_per_request_latency(
        request_event_df,
        request_df.output_lens,
        request_df.first_token_prefill,
    ).reset_index().rename(columns={"index": "req_id"})
    worker_df = organize_worker_event_df(cluster).sort_values(["start_time", "worker_id"]).reset_index(drop=True)
    return SimulationResult(
        config=config,
        requests=requests,
        request_df=request_df,
        request_event_df=request_event_df.reset_index(drop=True),
        latency_df=latency_df,
        worker_df=worker_df,
    )


def latency_summary(latency_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["first_token_latency", "tpot", "total_latency"]
    rows = []
    for metric in metrics:
        series = latency_df[metric]
        rows.append(
            {
                "metric": metric,
                "mean_ms": series.mean(),
                "p50_ms": series.quantile(0.50),
                "p90_ms": series.quantile(0.90),
                "p99_ms": series.quantile(0.99),
                "max_ms": series.max(),
            }
        )
    return pd.DataFrame(rows)


def attainment_summary(
    latency_df: pd.DataFrame,
    prefill_target_ms: float,
    decode_target_ms: float,
) -> dict[str, float]:
    prefill_ok = (latency_df["first_token_latency"] <= prefill_target_ms).mean() * 100
    decode_ok = (latency_df["tpot"] <= decode_target_ms).mean() * 100
    both_ok = (
        (latency_df["first_token_latency"] <= prefill_target_ms)
        & (latency_df["tpot"] <= decode_target_ms)
    ).mean() * 100
    return {
        "prefill_attainment_pct": prefill_ok,
        "decode_attainment_pct": decode_ok,
        "joint_attainment_pct": both_ok,
    }


def describe_config(config: ExampleConfig) -> str:
    if config.backend == "vllm":
        parts = [
            f"backend={config.backend}",
            f"model={config.model}",
            f"tp={config.tp_prefill}",
            f"pp={config.pp_prefill}",
            f"total_gpus={config.total_gpus}",
        ]
    else:
        parts = [
            f"backend={config.backend}",
            f"model={config.model}",
            f"tp_prefill={config.tp_prefill}",
            f"pp_prefill={config.pp_prefill}",
            f"tp_decode={config.tp_decode}",
            f"pp_decode={config.pp_decode}",
            f"total_gpus={config.total_gpus}",
        ]

    if config.enable_chunked_prefill:
        parts.append("enable_chunked_prefill=True")
    if config.prefill_max_batch_size != 10 ** 7:
        parts.append(f"prefill_max_batch_size={config.prefill_max_batch_size}")
    if config.decode_max_batch_size != 10 ** 7:
        parts.append(f"decode_max_batch_size={config.decode_max_batch_size}")
    if config.prefill_max_tokens_cap is not None:
        parts.append(f"prefill_max_tokens_cap={config.prefill_max_tokens_cap}")
    if config.decode_max_tokens_cap is not None:
        parts.append(f"decode_max_tokens_cap={config.decode_max_tokens_cap}")
    if config.decode_back_pressure != 0.9:
        parts.append(f"decode_back_pressure={config.decode_back_pressure}")
    if config.handoff_delay_ms is not None:
        parts.append(f"handoff_delay_ms={config.handoff_delay_ms}")
    if config.handoff_delay_per_token_ms is not None:
        parts.append(f"handoff_delay_per_token_ms={config.handoff_delay_per_token_ms}")
    if config.handoff_capacity is not None:
        parts.append(f"handoff_capacity={config.handoff_capacity}")
    if config.prefill_first_token_visible_immediately is not None:
        parts.append(
            "prefill_first_token_visible_immediately="
            f"{config.prefill_first_token_visible_immediately}"
        )
    return ", ".join(parts)


def request_table(request_specs: Sequence[tuple[int, int]], interarrival_ms: Sequence[float]) -> pd.DataFrame:
    absolute_arrival = []
    current = 0.0
    for delay in interarrival_ms:
        current += delay
        absolute_arrival.append(current)
    return pd.DataFrame(
        [
            {
                "req_id": req_id,
                "prefill_tokens": prefill_tokens,
                "output_tokens": output_tokens,
                "interarrival_ms": delay,
                "arrival_ms": arrival_time,
            }
            for req_id, ((prefill_tokens, output_tokens), delay, arrival_time) in enumerate(
                zip(request_specs, interarrival_ms, absolute_arrival)
            )
        ]
    )


def request_event_view(request_event_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["req_id", "start_time", "event_type", "worker_id", "duration"]
    return request_event_df.loc[:, cols].rename(columns={"start_time": "time_ms"})


def handoff_event_view(request_event_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["req_id", "start_time", "event_type", "worker_id", "duration"]
    event_types = ["wait_handoff", "do_handoff", "finish_handoff", "first_token_visible"]
    df = request_event_df[request_event_df.event_type.isin(event_types)].loc[:, cols]
    return df.rename(columns={"start_time": "time_ms"})


def worker_event_view(worker_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "worker_id",
        "start_time",
        "event_type",
        "duration",
        "prefill_bs",
        "decode_bs",
        "prefill_batch",
        "decode_batch",
    ]
    return worker_df.loc[:, cols].rename(columns={"start_time": "time_ms"})


def request_latency_view(latency_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["req_id", "first_token_latency", "tpot", "total_latency"]
    return latency_df.loc[:, cols]


def decode_round_view(worker_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["worker_id", "start_time", "duration", "decode_bs", "decode_batch"]
    df = worker_df[worker_df.event_type == "do_decode"].loc[:, cols]
    return df.rename(columns={"start_time": "time_ms"})


def prefill_round_view(worker_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "worker_id",
        "start_time",
        "duration",
        "prefill_bs",
        "decode_bs",
        "prefill_batch",
        "decode_batch",
    ]
    df = worker_df[worker_df.event_type == "do_prefill"].loc[:, cols]
    return df.rename(columns={"start_time": "time_ms"})


def decode_round_membership_view(request_event_df: pd.DataFrame) -> pd.DataFrame:
    decode_df = request_event_df[request_event_df.event_type == "do_decode"].copy()
    if decode_df.empty:
        return pd.DataFrame(
            columns=["time_ms", "worker_id", "batch_size", "req_ids", "joiners", "leavers"]
        )

    grouped = (
        decode_df.groupby(["start_time", "worker_id"], sort=True)["req_id"]
        .apply(lambda s: sorted(s.tolist()))
        .reset_index()
        .rename(columns={"start_time": "time_ms", "req_id": "req_ids"})
    )
    grouped["batch_size"] = grouped["req_ids"].apply(len)

    previous_ids: set[int] = set()
    joiners = []
    leavers = []
    for req_ids in grouped["req_ids"]:
        current_ids = set(req_ids)
        joiners.append(sorted(current_ids - previous_ids))
        leavers.append(sorted(previous_ids - current_ids))
        previous_ids = current_ids
    grouped["joiners"] = joiners
    grouped["leavers"] = leavers
    return grouped.loc[:, ["time_ms", "worker_id", "batch_size", "req_ids", "joiners", "leavers"]]


def chunk_slicing_plan(prefill_len: int, chunk_cap: int) -> pd.DataFrame:
    if prefill_len <= 0:
        raise ValueError(f"prefill_len must be positive, got {prefill_len}.")
    if chunk_cap <= 0:
        raise ValueError(f"chunk_cap must be positive, got {chunk_cap}.")

    rows = []
    remain = prefill_len
    round_idx = 0
    while remain > 0:
        current = min(remain, chunk_cap)
        remain_after = remain - current
        rows.append(
            {
                "round": round_idx,
                "remain_before": remain,
                "current_prefill_lens": current,
                "remain_after": remain_after,
            }
        )
        remain = remain_after
        round_idx += 1
    return pd.DataFrame(rows)


def round_frame(df: pd.DataFrame, digits: int = 2) -> pd.DataFrame:
    rounded = df.copy()
    numeric_cols = rounded.select_dtypes(include="number").columns
    rounded.loc[:, numeric_cols] = rounded.loc[:, numeric_cols].round(digits)
    return rounded


def format_frame(df: pd.DataFrame, digits: int = 2) -> str:
    return round_frame(df, digits=digits).to_string(index=False)


def repeated_pattern(pattern: Sequence[tuple[int, int]], repeats: int) -> list[tuple[int, int]]:
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}.")
    return list(pattern) * repeats


def pretty_join(lines: Iterable[str]) -> str:
    return "\n".join(line for line in lines if line)
