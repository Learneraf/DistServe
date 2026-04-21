#!/usr/bin/env python3
"""
DistServe-style heterogeneous placement search for DistServe CUDA and vLLM Ascend.

The planner searches disaggregated prefill/decode placements that maximize goodput
under TTFT/TPOT SLO attainment. It uses the existing runtime simulator and timing
estimators, but allows the prefill phase and decode phase to run on different
backends.

Topology schema
---------------
The planner consumes a JSON file with this shape:

{
  "objective": "per_device_goodput",
  "ttft_target_ms": 1000.0,
  "tpot_target_ms": 1000.0,
  "attainment_target_pct": 90.0,
  "search": {
    "max_rate": 8.0,
    "rate_epsilon": 0.25,
    "num_requests": 120,
    "seed": 0,
    "arrival": "poisson",
    "cv": 1.0,
    "top_k": 10,
    "pp_candidates": [1, 2, 4]
  },
  "nodes": [
    {
      "name": "node0",
      "devices": {
        "distserve_cuda": 4,
        "vllm_ascend": 8
      }
    }
  ],
  "handoff_ms": {
    "same_node": {
      "distserve_cuda->distserve_cuda": {"base": 0.0, "per_token": 0.0},
      "distserve_cuda->vllm_ascend": {"base": 65.4, "per_token": 0.0944},
      "vllm_ascend->distserve_cuda": {"base": 65.4, "per_token": 0.0944},
      "vllm_ascend->vllm_ascend": {"base": 65.4, "per_token": 0.0944}
    },
    "cross_node": {
      "distserve_cuda->distserve_cuda": {"base": 0.5, "per_token": 0.0},
      "distserve_cuda->vllm_ascend": {"base": 80.0, "per_token": 0.12},
      "vllm_ascend->distserve_cuda": {"base": 80.0, "per_token": 0.12},
      "vllm_ascend->vllm_ascend": {"base": 80.0, "per_token": 0.12}
    }
  }
}
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
from dataclasses import asdict, dataclass, replace
from functools import reduce
from itertools import chain
from pathlib import Path
from typing import Any

import simpy

from simdistserve.base.organize_data import (
    calculate_per_request_latency,
    organize_request_df,
    organize_request_event_df,
)
from simdistserve.base.request import Request
from simdistserve.base.scheduler import Scheduler, put_requests_with_interarrivals
from simdistserve.base.worker import Worker, WorkerConfig
from simdistserve.base.workload import get_fixed_interarrival, get_gamma_interarrival
from simdistserve.constants import ModelTypes
from simdistserve.estimators import time_estimator
from simdistserve.estimators.memory_estimator import is_model_runnable
from simdistserve.utils import set_next_worker


BACKENDS = ("distserve_cuda", "vllm_ascend")
ENGINE_TYPE_BY_BACKEND = {
    "distserve_cuda": "distserve",
    "vllm_ascend": "vllm_ascend",
}
MODEL_ALIAS_TO_PATH = {
    "llama_1B": ModelTypes.LLAMA_3_2_1B_LOCAL_PATH,
    "llama_3B": ModelTypes.LLAMA_3_2_3B_LOCAL_PATH,
    "llama_7B": ModelTypes.LLAMA_2_7B_LOCAL_PATH,
    "llama_8B": ModelTypes.LLAMA_3_1_8B_LOCAL_PATH,
}


@dataclass(frozen=True)
class NodeSpec:
    name: str
    devices: dict[str, int]


@dataclass(frozen=True)
class TransferCost:
    base_ms: float
    per_token_ms: float


@dataclass(frozen=True)
class SearchConfig:
    max_rate: float
    rate_epsilon: float
    num_requests: int
    seed: int
    arrival: str
    cv: float
    top_k: int
    pp_candidates: list[int]


@dataclass(frozen=True)
class TopologyConfig:
    objective: str
    ttft_target_ms: float
    tpot_target_ms: float
    attainment_target_pct: float
    nodes: list[NodeSpec]
    search: SearchConfig
    handoff_same_node: dict[str, TransferCost]
    handoff_cross_node: dict[str, TransferCost]
    prefill_first_token_visible_immediately: bool | None


@dataclass(frozen=True)
class PhaseChoice:
    backend: str
    tp: int
    pp: int
    instances: int
    allocation: dict[str, int]


@dataclass(frozen=True)
class PlacementPlan:
    prefill: PhaseChoice
    decode: PhaseChoice
    same_node_fraction: float
    handoff_base_ms: float
    handoff_per_token_ms: float
    devices_used: dict[str, int]
    total_devices_used: int


@dataclass(frozen=True)
class EvaluationMetrics:
    rate: float
    ttft_attainment_pct: float
    tpot_attainment_pct: float
    both_attainment_pct: float
    mean_ttft_ms: float
    mean_tpot_ms: float


@dataclass(frozen=True)
class PlacementResult:
    plan: PlacementPlan
    sustainable_rate: float
    objective_value: float
    metrics_at_best_rate: EvaluationMetrics


def _normalize_model_path(model: str) -> str:
    return MODEL_ALIAS_TO_PATH.get(model, model)


def configure_profile_paths(
    distserve_profile: Path | None = None,
    vllm_ascend_profile: Path | None = None,
) -> None:
    if distserve_profile is not None:
        os.environ["SIMDISTSERVE_DISTSERVE_PROFILE"] = str(distserve_profile)
    if vllm_ascend_profile is not None:
        os.environ["SIMDISTSERVE_VLLM_ASCEND_PROFILE"] = str(vllm_ascend_profile)
    importlib.reload(time_estimator)


def _to_model_object(model: str) -> str:
    return ModelTypes.model_str_to_object(_normalize_model_path(model))


def _load_workload_jsonl(
    workload_path: Path,
    num_requests: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(workload_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if num_requests >= len(rows):
        return rows
    random.seed(seed)
    return random.sample(rows, num_requests)


def _build_requests(
    workload_rows: list[dict[str, Any]],
) -> list[Request]:
    requests: list[Request] = []
    for idx, row in enumerate(workload_rows):
        requests.append(
            Request(
                env=None,
                req_id=idx,
                source_index=int(row.get("source_index", idx)),
                prefill_length=int(row["prompt_len"]),
                output_lens=int(row["output_len"]),
            )
        )
    return requests


def _build_interarrivals(
    count: int,
    rate: float,
    arrival: str,
    cv: float,
    seed: int,
) -> list[float]:
    if arrival == "fixed":
        return get_fixed_interarrival(count, 1000.0 / rate)
    if arrival in {"poisson", "gamma"}:
        gamma_cv = 1.0 if arrival == "poisson" else cv
        return get_gamma_interarrival(count, rate, gamma_cv, seed=seed)
    raise ValueError(f"Unsupported arrival mode: {arrival}")


class HeterogeneousDisaggCluster:
    def __init__(
        self,
        env: simpy.Environment,
        n_prefill_instances: int,
        n_decode_instances: int,
        pp_prefill: int,
        pp_decode: int,
        prefill_worker_config: WorkerConfig,
        decode_worker_config: WorkerConfig,
        handoff_base_ms: float,
        handoff_per_token_ms: float,
        handoff_capacity: int = 1,
        prefill_first_token_visible_immediately: bool = True,
    ) -> None:
        prefill_instances = []
        decode_instances = []
        worker_id = 0

        prefill_kwargs = dict(global_scheduler=None, **prefill_worker_config)
        decode_kwargs = dict(global_scheduler=None, **decode_worker_config)

        for _ in range(n_prefill_instances):
            instance = []
            for pipe_rank in range(pp_prefill):
                worker = Worker(env, worker_id, cluster=self, pipe_rank=pipe_rank, **prefill_kwargs)
                instance.append(worker)
                worker_id += 1
            reduce(set_next_worker, chain(instance, (instance[0],)))
            instance[-1].is_last_in_pipeline = True
            instance[-1].should_request_stay = False
            prefill_instances.append(instance)

        for _ in range(n_decode_instances):
            instance = []
            for pipe_rank in range(pp_decode):
                worker = Worker(env, worker_id, cluster=self, pipe_rank=pipe_rank, **decode_kwargs)
                instance.append(worker)
                worker_id += 1
            reduce(set_next_worker, chain(instance, (instance[0],)))
            instance[-1].is_last_in_pipeline = True
            decode_instances.append(instance)

        scheduler = Scheduler(
            env,
            prefill_heads=[inst[0] for inst in prefill_instances],
            decode_heads=[inst[0] for inst in decode_instances],
        )
        for last_in_prefill in (inst[-1] for inst in prefill_instances):
            last_in_prefill.global_scheduler = scheduler

        self.env = env
        self.PP_prefill = pp_prefill
        self.PP_decode = pp_decode
        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances
        self.scheduler = scheduler
        self.handoff_delay_ms = float(handoff_base_ms)
        self.handoff_delay_per_token_ms = float(handoff_per_token_ms)
        self.handoff_resource = simpy.Resource(env, capacity=max(1, int(handoff_capacity)))
        self.prefill_first_token_visible_immediately = bool(prefill_first_token_visible_immediately)

    def get_handoff_delay(self, request: Request) -> float:
        return self.handoff_delay_ms + self.handoff_delay_per_token_ms * request.current_context_len

    def start_handoff(self, request: Request, source_worker: Worker, to_scheduler: bool = True) -> None:
        self.env.process(self._run_handoff(request, source_worker, to_scheduler))

    def _run_handoff(self, request: Request, source_worker: Worker, to_scheduler: bool):
        request.wait_handoff(wid=source_worker.wid)
        delay = self.get_handoff_delay(request)
        with self.handoff_resource.request() as handoff_slot:
            yield handoff_slot
            request.do_handoff(wid=source_worker.wid)
            if delay > 0:
                yield self.env.timeout(delay)
        request.finish_handoff(wid=source_worker.wid)
        if (
            request.first_token_prefill
            and not self.prefill_first_token_visible_immediately
            and request.should_finish()
        ):
            request.mark_first_token_visible(wid=source_worker.wid)
        if request.should_finish():
            return
        if to_scheduler:
            self.scheduler.schedule_decode(request)
            return
        next_wid = source_worker.next_worker.wid if source_worker.next_worker else None
        request.wait_decode(wid=next_wid)
        source_worker.forward_decode(request, to_scheduler=False)

    def get_all_workers(self) -> list[Worker]:
        return list(chain(chain(*self.prefill_instances), chain(*self.decode_instances)))

    def run(self) -> "HeterogeneousDisaggCluster":
        for instance in chain(self.prefill_instances, self.decode_instances):
            for worker in instance:
                self.env.process(worker.run())
        return self


def _parse_transfer_cost_map(raw: dict[str, Any]) -> dict[str, TransferCost]:
    result: dict[str, TransferCost] = {}
    for key, value in raw.items():
        result[key] = TransferCost(
            base_ms=float(value.get("base", 0.0)),
            per_token_ms=float(value.get("per_token", 0.0)),
        )
    return result


def load_topology_config(path: Path) -> TopologyConfig:
    payload = json.loads(path.read_text())
    nodes = [
        NodeSpec(
            name=str(node["name"]),
            devices={backend: int(count) for backend, count in node["devices"].items()},
        )
        for node in payload["nodes"]
    ]
    search = payload.get("search", {})
    search_cfg = SearchConfig(
        max_rate=float(search.get("max_rate", 8.0)),
        rate_epsilon=float(search.get("rate_epsilon", 0.25)),
        num_requests=int(search.get("num_requests", 120)),
        seed=int(search.get("seed", 0)),
        arrival=str(search.get("arrival", "poisson")),
        cv=float(search.get("cv", 1.0)),
        top_k=int(search.get("top_k", 10)),
        pp_candidates=[int(x) for x in search.get("pp_candidates", [1])],
    )
    handoff_ms = payload.get("handoff_ms", {})
    return TopologyConfig(
        objective=str(payload.get("objective", "per_device_goodput")),
        ttft_target_ms=float(payload.get("ttft_target_ms", 1000.0)),
        tpot_target_ms=float(payload.get("tpot_target_ms", 1000.0)),
        attainment_target_pct=float(payload.get("attainment_target_pct", 90.0)),
        nodes=nodes,
        search=search_cfg,
        handoff_same_node=_parse_transfer_cost_map(handoff_ms.get("same_node", {})),
        handoff_cross_node=_parse_transfer_cost_map(handoff_ms.get("cross_node", {})),
        prefill_first_token_visible_immediately=payload.get("prefill_first_token_visible_immediately"),
    )


def _phase_capacity_per_node(topology: TopologyConfig, backend: str, tp: int, pp: int) -> dict[str, int]:
    capacities: dict[str, int] = {}
    devices_per_instance = tp * pp
    for node in topology.nodes:
        capacities[node.name] = node.devices.get(backend, 0) // devices_per_instance
    return capacities


def _enumerate_allocations_from_capacities(
    node_caps: list[tuple[str, int]],
    target_instances: int,
    idx: int = 0,
) -> list[dict[str, int]]:
    if target_instances == 0:
        return [{}]
    if idx >= len(node_caps):
        return []

    node_name, capacity = node_caps[idx]
    allocations: list[dict[str, int]] = []
    max_assign = min(capacity, target_instances)
    for assigned in range(max_assign + 1):
        tail_allocations = _enumerate_allocations_from_capacities(
            node_caps,
            target_instances - assigned,
            idx + 1,
        )
        for tail in tail_allocations:
            merged = dict(tail)
            if assigned > 0:
                merged[node_name] = assigned
            allocations.append(merged)
    return allocations


def enumerate_phase_choices(
    topology: TopologyConfig,
    model: str,
    backend: str,
    tp_candidates: list[int],
    pp_candidates: list[int],
) -> list[PhaseChoice]:
    model_obj = _to_model_object(model)
    results: list[PhaseChoice] = []

    for tp in tp_candidates:
        for pp in pp_candidates:
            if not is_model_runnable(model_obj, tp, pp):
                continue
            node_cap_map = _phase_capacity_per_node(topology, backend, tp, pp)
            total_instances = sum(node_cap_map.values())
            if total_instances <= 0:
                continue
            node_caps = sorted(node_cap_map.items())
            for instances in range(1, total_instances + 1):
                for allocation in _enumerate_allocations_from_capacities(node_caps, instances):
                    results.append(
                        PhaseChoice(
                            backend=backend,
                            tp=tp,
                            pp=pp,
                            instances=instances,
                            allocation=allocation,
                        )
                    )
    return results


def _allocations_compatible(
    topology: TopologyConfig,
    prefill: PhaseChoice,
    decode: PhaseChoice,
) -> bool:
    for node in topology.nodes:
        prefill_used = prefill.allocation.get(node.name, 0) * prefill.tp * prefill.pp
        decode_used = decode.allocation.get(node.name, 0) * decode.tp * decode.pp
        if prefill.backend == decode.backend:
            if prefill_used + decode_used > node.devices.get(prefill.backend, 0):
                return False
        else:
            if prefill_used > node.devices.get(prefill.backend, 0):
                return False
            if decode_used > node.devices.get(decode.backend, 0):
                return False
    return True


def _compute_local_fraction(prefill: PhaseChoice, decode: PhaseChoice) -> float:
    colocated = 0
    for node_name in set(prefill.allocation) | set(decode.allocation):
        colocated += min(prefill.allocation.get(node_name, 0), decode.allocation.get(node_name, 0))
    pair_count = max(1, min(prefill.instances, decode.instances))
    return min(1.0, colocated / pair_count)


def _transfer_key(prefill_backend: str, decode_backend: str) -> str:
    return f"{prefill_backend}->{decode_backend}"


def build_placement_plan(
    topology: TopologyConfig,
    prefill: PhaseChoice,
    decode: PhaseChoice,
) -> PlacementPlan:
    local_fraction = _compute_local_fraction(prefill, decode)
    key = _transfer_key(prefill.backend, decode.backend)
    same_node_cost = topology.handoff_same_node.get(key, TransferCost(0.0, 0.0))
    cross_node_cost = topology.handoff_cross_node.get(key, same_node_cost)
    handoff_base_ms = (
        local_fraction * same_node_cost.base_ms
        + (1.0 - local_fraction) * cross_node_cost.base_ms
    )
    handoff_per_token_ms = (
        local_fraction * same_node_cost.per_token_ms
        + (1.0 - local_fraction) * cross_node_cost.per_token_ms
    )
    devices_used = {
        "distserve_cuda": 0,
        "vllm_ascend": 0,
    }
    devices_used[prefill.backend] += prefill.instances * prefill.tp * prefill.pp
    devices_used[decode.backend] += decode.instances * decode.tp * decode.pp
    total_devices_used = devices_used["distserve_cuda"] + devices_used["vllm_ascend"]
    return PlacementPlan(
        prefill=prefill,
        decode=decode,
        same_node_fraction=local_fraction,
        handoff_base_ms=handoff_base_ms,
        handoff_per_token_ms=handoff_per_token_ms,
        devices_used=devices_used,
        total_devices_used=total_devices_used,
    )


def evaluate_plan_at_rate(
    plan: PlacementPlan,
    model: str,
    workload_path: Path,
    rate: float,
    topology: TopologyConfig,
    num_requests: int | None = None,
    seed: int | None = None,
    arrival: str | None = None,
    cv: float | None = None,
) -> EvaluationMetrics:
    search = topology.search
    num_requests = search.num_requests if num_requests is None else num_requests
    seed = search.seed if seed is None else seed
    arrival = search.arrival if arrival is None else arrival
    cv = search.cv if cv is None else cv

    model_path = _normalize_model_path(model)
    model_obj = _to_model_object(model)
    workload_rows = _load_workload_jsonl(workload_path, num_requests, seed)
    requests = _build_requests(workload_rows)
    interarrivals = _build_interarrivals(len(requests), rate, arrival, cv, seed)

    prefill_engine = ENGINE_TYPE_BY_BACKEND[plan.prefill.backend]
    decode_engine = ENGINE_TYPE_BY_BACKEND[plan.decode.backend]

    prefill_worker_config: WorkerConfig = WorkerConfig(
        model_type=model_obj,
        TP=plan.prefill.tp,
        TP_Prefill=plan.prefill.tp,
        TP_Decode=plan.prefill.tp,
        prefill_max_batch_size=10 ** 7,
        decode_max_batch_size=10 ** 7,
        prefill_max_tokens=10 ** 7,
        decode_max_tokens=10 ** 7,
        enable_chunked_prefill=False,
        engine_type=prefill_engine,
        prefill_generates_first_token=(prefill_engine == "vllm_ascend"),
    )
    decode_worker_config: WorkerConfig = WorkerConfig(
        model_type=model_obj,
        TP=plan.decode.tp,
        TP_Prefill=plan.decode.tp,
        TP_Decode=plan.decode.tp,
        prefill_max_batch_size=10 ** 7,
        decode_max_batch_size=10 ** 7,
        prefill_max_tokens=10 ** 7,
        decode_max_tokens=10 ** 7,
        enable_chunked_prefill=False,
        engine_type=decode_engine,
        prefill_generates_first_token=False,
    )

    prefill_first_token_visible_immediately = topology.prefill_first_token_visible_immediately
    if prefill_first_token_visible_immediately is None:
        prefill_first_token_visible_immediately = (prefill_engine != "vllm_ascend")

    env = simpy.Environment()
    cluster = HeterogeneousDisaggCluster(
        env=env,
        n_prefill_instances=plan.prefill.instances,
        n_decode_instances=plan.decode.instances,
        pp_prefill=plan.prefill.pp,
        pp_decode=plan.decode.pp,
        prefill_worker_config=prefill_worker_config,
        decode_worker_config=decode_worker_config,
        handoff_base_ms=plan.handoff_base_ms,
        handoff_per_token_ms=plan.handoff_per_token_ms,
        handoff_capacity=1,
        prefill_first_token_visible_immediately=prefill_first_token_visible_immediately,
    )
    cluster.run()
    put_requests_with_interarrivals(env, cluster.scheduler, interarrivals, requests)
    env.run()

    request_df = organize_request_df(requests)
    request_event_df = organize_request_event_df(requests)
    latency_df = calculate_per_request_latency(
        request_event_df,
        request_df.output_lens,
        request_df.first_token_prefill,
    )

    ttft_target = topology.ttft_target_ms
    tpot_target = topology.tpot_target_ms
    ttft_attainment = float((latency_df["first_token_latency"] <= ttft_target).mean() * 100.0)
    tpot_attainment = float((latency_df["tpot"] <= tpot_target).mean() * 100.0)
    both_attainment = float(
        ((latency_df["first_token_latency"] <= ttft_target) & (latency_df["tpot"] <= tpot_target)).mean() * 100.0
    )
    return EvaluationMetrics(
        rate=rate,
        ttft_attainment_pct=ttft_attainment,
        tpot_attainment_pct=tpot_attainment,
        both_attainment_pct=both_attainment,
        mean_ttft_ms=float(latency_df["first_token_latency"].mean()),
        mean_tpot_ms=float(latency_df["tpot"].mean()),
    )


def rate_meets_slo(metrics: EvaluationMetrics, topology: TopologyConfig) -> bool:
    threshold = topology.attainment_target_pct
    return (
        metrics.ttft_attainment_pct >= threshold
        and metrics.tpot_attainment_pct >= threshold
    )


def find_sustainable_rate(
    plan: PlacementPlan,
    model: str,
    workload_path: Path,
    topology: TopologyConfig,
) -> tuple[float, EvaluationMetrics]:
    low = 0.0
    high = topology.search.max_rate
    best_metrics = evaluate_plan_at_rate(plan, model, workload_path, rate=0.01, topology=topology)
    epsilon = topology.search.rate_epsilon

    while (high - low) > epsilon:
        rate = (low + high) / 2.0
        metrics = evaluate_plan_at_rate(plan, model, workload_path, rate=rate, topology=topology)
        if rate_meets_slo(metrics, topology):
            low = rate
            best_metrics = metrics
        else:
            high = rate

    return low, best_metrics


def objective_value(plan: PlacementPlan, sustainable_rate: float, topology: TopologyConfig) -> float:
    if topology.objective == "sustainable_rate":
        return sustainable_rate
    return sustainable_rate / max(1, plan.total_devices_used)


def search_best_plan(
    topology: TopologyConfig,
    model: str,
    workload_path: Path,
    tp_candidates: list[int] | None = None,
) -> dict[str, Any]:
    tp_candidates = tp_candidates or [1]
    pp_candidates = topology.search.pp_candidates

    prefill_choices = []
    decode_choices = []
    for backend in BACKENDS:
        prefill_choices.extend(enumerate_phase_choices(topology, model, backend, tp_candidates, pp_candidates))
        decode_choices.extend(enumerate_phase_choices(topology, model, backend, tp_candidates, pp_candidates))

    results: list[PlacementResult] = []
    for prefill in prefill_choices:
        for decode in decode_choices:
            if not _allocations_compatible(topology, prefill, decode):
                continue
            plan = build_placement_plan(topology, prefill, decode)
            sustainable_rate, metrics = find_sustainable_rate(plan, model, workload_path, topology)
            value = objective_value(plan, sustainable_rate, topology)
            results.append(
                PlacementResult(
                    plan=plan,
                    sustainable_rate=sustainable_rate,
                    objective_value=value,
                    metrics_at_best_rate=metrics,
                )
            )

    results.sort(
        key=lambda item: (
            item.objective_value,
            item.sustainable_rate,
            -item.plan.total_devices_used,
        ),
        reverse=True,
    )
    top_k = topology.search.top_k
    best = results[0] if results else None
    return {
        "best_plan": _serialize_result(best) if best is not None else None,
        "top_plans": [_serialize_result(item) for item in results[:top_k]],
        "num_candidates_evaluated": len(results),
    }


def _serialize_result(result: PlacementResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "plan": {
            "prefill": asdict(result.plan.prefill),
            "decode": asdict(result.plan.decode),
            "same_node_fraction": result.plan.same_node_fraction,
            "handoff_base_ms": result.plan.handoff_base_ms,
            "handoff_per_token_ms": result.plan.handoff_per_token_ms,
            "devices_used": result.plan.devices_used,
            "total_devices_used": result.plan.total_devices_used,
        },
        "sustainable_rate": result.sustainable_rate,
        "objective_value": result.objective_value,
        "metrics_at_best_rate": asdict(result.metrics_at_best_rate),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search a heterogeneous DistServe-style placement plan.")
    parser.add_argument("--topology", type=Path, required=True, help="Topology JSON file.")
    parser.add_argument("--model", type=str, required=True, help="Model alias or formal model path.")
    parser.add_argument("--workload", type=Path, required=True, help="JSONL workload file.")
    parser.add_argument(
        "--tp-candidates",
        type=str,
        default="[1]",
        help="Python list of TP candidates to consider for each phase.",
    )
    parser.add_argument(
        "--distserve-profile",
        type=Path,
        default=None,
        help="Optional DistServe CUDA fit-params JSON to force during evaluation.",
    )
    parser.add_argument(
        "--vllm-ascend-profile",
        type=Path,
        default=Path(
            "/users/rh/DistServe/simdistserve/estimators/profiled_data/vllm-ascend/fit_params_live_5p4d.json"
        ),
        help="Optional vLLM Ascend fit-params JSON to force during evaluation. Defaults to the repo's live 5p4d profile.",
    )
    parser.add_argument(
        "--prefill-first-token-visible-immediately",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Whether a first token generated during prefill is immediately visible to the user "
            "before the handoff completes. Defaults to false for vllm_ascend prefill and true otherwise."
        ),
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_profile_paths(
        distserve_profile=args.distserve_profile,
        vllm_ascend_profile=args.vllm_ascend_profile,
    )
    topology = load_topology_config(args.topology)
    if args.prefill_first_token_visible_immediately is not None:
        topology = replace(
            topology,
            prefill_first_token_visible_immediately=args.prefill_first_token_visible_immediately,
        )
    tp_candidates = [int(x) for x in eval(args.tp_candidates)]
    result = search_best_plan(topology, args.model, args.workload, tp_candidates=tp_candidates)
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
