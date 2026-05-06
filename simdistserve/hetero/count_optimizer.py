from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from simdistserve.hetero.flow import FlowInputs, solve_flow_allocation
from simdistserve.hetero.enumerate import (
    get_instance_upper_bound,
    resource_footprint,
    static_count_compatible,
)
from simdistserve.hetero.types import DevicePool, FlowSolution, HeteroConfig, RoleConfig, RoleShape


SearchMode = Literal["milp", "no_cross", "cuda_prefill_ascend_decode"]


@dataclass(frozen=True)
class CountOptimizationInputs:
    cuda_pool: DevicePool
    ascend_pool: DevicePool
    cuda_prefill: RoleShape | None
    cuda_decode: RoleShape | None
    ascend_prefill: RoleShape | None
    ascend_decode: RoleShape | None
    mu_cp: float
    mu_cd: float
    mu_ap: float
    mu_ad: float
    h_ca: float
    h_ac: float
    search_mode: SearchMode = "milp"


@dataclass(frozen=True)
class _PoolPair:
    n_prefill: int
    n_decode: int
    prefill_capacity: float
    decode_capacity: float
    resource_used: int


NCP = 0
NCD = 1
NAP = 2
NAD = 3
XCC = 4
XCA = 5
XAC = 6
XAA = 7
NUM_VARS = 8


def optimize_instance_allocation(inputs: CountOptimizationInputs) -> HeteroConfig | None:
    """Optimize instance counts and 2x2 prefill/decode flow for a fixed shape tuple."""
    if os.environ.get("SIMDISTSERVE_HETERO_USE_SCIPY_MILP") == "1":
        return optimize_instance_allocation_milp(inputs)
    return optimize_instance_allocation_fast(inputs)


def optimize_instance_allocation_fast(inputs: CountOptimizationInputs) -> HeteroConfig | None:
    """Fast specialized allocator for the fixed-shape 2x2 flow problem.

    The generic MILP has only eight variables, but invoking SciPy/HiGHS for
    every shape tuple has millisecond-level fixed overhead. This solver avoids
    that overhead by enumerating each pool's feasible count frontier and using
    the closed-form min-cut value of the 2x2 flow network.
    """
    cuda_decode = inputs.cuda_decode
    ascend_prefill = inputs.ascend_prefill
    mu_cd = inputs.mu_cd
    mu_ap = inputs.mu_ap
    if inputs.search_mode == "cuda_prefill_ascend_decode":
        cuda_decode = None
        ascend_prefill = None
        mu_cd = 0.0
        mu_ap = 0.0

    cuda_pairs = _pool_count_frontier(
        pool=inputs.cuda_pool,
        prefill=inputs.cuda_prefill,
        decode=cuda_decode,
        mu_prefill=inputs.mu_cp,
        mu_decode=mu_cd,
    )
    ascend_pairs = _pool_count_frontier(
        pool=inputs.ascend_pool,
        prefill=ascend_prefill,
        decode=inputs.ascend_decode,
        mu_prefill=mu_ap,
        mu_decode=inputs.mu_ad,
    )
    if not cuda_pairs or not ascend_pairs:
        return None

    h_ca = 0.0 if inputs.search_mode == "no_cross" else max(0.0, inputs.h_ca)
    h_ac = 0.0 if inputs.search_mode in {"no_cross", "cuda_prefill_ascend_decode"} else max(0.0, inputs.h_ac)

    best_value = 0.0
    best_resource = math.inf
    best_counts: tuple[int, int, int, int] | None = None

    for cuda in cuda_pairs:
        for ascend in ascend_pairs:
            if cuda.n_prefill + ascend.n_prefill <= 0:
                continue
            if cuda.n_decode + ascend.n_decode <= 0:
                continue
            value = _flow_value_upper_bound(
                p_cuda=cuda.prefill_capacity,
                p_asc=ascend.prefill_capacity,
                d_cuda=cuda.decode_capacity,
                d_asc=ascend.decode_capacity,
                h_ca=h_ca,
                h_ac=h_ac,
            )
            if value <= 1e-9:
                continue
            resource = cuda.resource_used + ascend.resource_used
            if value > best_value + 1e-9 or (
                abs(value - best_value) <= 1e-9 and resource < best_resource
            ):
                best_value = value
                best_resource = resource
                best_counts = (
                    cuda.n_prefill,
                    cuda.n_decode,
                    ascend.n_prefill,
                    ascend.n_decode,
                )

    if best_counts is None:
        return None

    ncp, ncd, nap, nad = best_counts
    flows = solve_flow_allocation(
        FlowInputs(
            mu_cp=inputs.mu_cp,
            mu_cd=inputs.mu_cd,
            mu_ap=inputs.mu_ap,
            mu_ad=inputs.mu_ad,
            ncp=ncp,
            ncd=ncd,
            nap=nap,
            nad=nad,
            h_ca=h_ca,
            h_ac=h_ac,
        )
    )
    return _to_config_from_flows(inputs, flows)


def optimize_instance_allocation_milp(inputs: CountOptimizationInputs) -> HeteroConfig | None:
    """Reference SciPy MILP allocator for debugging and validation."""
    upper_bounds = _variable_upper_bounds(inputs)
    if upper_bounds[NCP] + upper_bounds[NAP] <= 0:
        return None
    if upper_bounds[NCD] + upper_bounds[NAD] <= 0:
        return None

    c = _objective(inputs)
    integrality = np.zeros(NUM_VARS)
    integrality[[NCP, NCD, NAP, NAD]] = 1

    constraints = _constraints(inputs)
    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(np.zeros(NUM_VARS), upper_bounds),
        constraints=constraints,
        options={"disp": False},
    )
    if not result.success or result.x is None:
        return None

    return _to_config(inputs, result.x)


def _pool_count_frontier(
    pool: DevicePool,
    prefill: RoleShape | None,
    decode: RoleShape | None,
    mu_prefill: float,
    mu_decode: float,
) -> tuple[_PoolPair, ...]:
    return _pool_count_frontier_cached(pool, prefill, decode, float(mu_prefill), float(mu_decode))


@lru_cache(maxsize=4096)
def _pool_count_frontier_cached(
    pool: DevicePool,
    prefill: RoleShape | None,
    decode: RoleShape | None,
    mu_prefill: float,
    mu_decode: float,
) -> tuple[_PoolPair, ...]:
    n_prefill_ub = get_instance_upper_bound(pool, prefill)
    n_decode_ub = get_instance_upper_bound(pool, decode)
    pairs: list[_PoolPair] = []

    for n_prefill in range(n_prefill_ub + 1):
        for n_decode in range(n_decode_ub + 1):
            if not static_count_compatible(prefill, decode, n_prefill, n_decode, pool):
                continue
            if prefill is None and n_prefill != 0:
                continue
            if decode is None and n_decode != 0:
                continue
            pairs.append(
                _PoolPair(
                    n_prefill=n_prefill,
                    n_decode=n_decode,
                    prefill_capacity=max(0.0, mu_prefill * n_prefill),
                    decode_capacity=max(0.0, mu_decode * n_decode),
                    resource_used=(
                        n_prefill * resource_footprint(prefill)
                        + n_decode * resource_footprint(decode)
                    ),
                )
            )

    return tuple(_prune_dominated_pairs(pairs))


def _prune_dominated_pairs(pairs: list[_PoolPair]) -> list[_PoolPair]:
    frontier: list[_PoolPair] = []
    for i, candidate in enumerate(pairs):
        dominated = False
        for j, other in enumerate(pairs):
            if i == j:
                continue
            if (
                other.prefill_capacity >= candidate.prefill_capacity - 1e-12
                and other.decode_capacity >= candidate.decode_capacity - 1e-12
                and (
                    other.prefill_capacity > candidate.prefill_capacity + 1e-12
                    or other.decode_capacity > candidate.decode_capacity + 1e-12
                    or other.resource_used < candidate.resource_used
                    or (
                        other.resource_used == candidate.resource_used
                        and other.n_prefill + other.n_decode <= candidate.n_prefill + candidate.n_decode
                        and j < i
                    )
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier


def _flow_value_upper_bound(
    p_cuda: float,
    p_asc: float,
    d_cuda: float,
    d_asc: float,
    h_ca: float,
    h_ac: float,
) -> float:
    """Closed-form max-flow value for the 2x2 prefill/decode graph."""
    return max(
        0.0,
        min(
            p_cuda + p_asc,
            d_cuda + d_asc,
            p_asc + d_cuda + max(0.0, h_ca),
            p_cuda + d_asc + max(0.0, h_ac),
        ),
    )


def _to_config_from_flows(inputs: CountOptimizationInputs, flows: FlowSolution) -> HeteroConfig | None:
    values = np.zeros(NUM_VARS, dtype=float)
    values[XCC] = flows.x_cc
    values[XCA] = flows.x_ca
    values[XAC] = flows.x_ac
    values[XAA] = flows.x_aa
    return _to_config(inputs, values)


def _variable_upper_bounds(inputs: CountOptimizationInputs) -> np.ndarray:
    ncp_ub = get_instance_upper_bound(inputs.cuda_pool, inputs.cuda_prefill)
    ncd_ub = get_instance_upper_bound(inputs.cuda_pool, inputs.cuda_decode)
    nap_ub = get_instance_upper_bound(inputs.ascend_pool, inputs.ascend_prefill)
    nad_ub = get_instance_upper_bound(inputs.ascend_pool, inputs.ascend_decode)

    max_prefill = max(0.0, inputs.mu_cp * ncp_ub + inputs.mu_ap * nap_ub)
    max_decode = max(0.0, inputs.mu_cd * ncd_ub + inputs.mu_ad * nad_ub)
    max_flow = max(max_prefill, max_decode, inputs.h_ca, inputs.h_ac, 1.0)

    upper_bounds = np.array(
        [
            float(ncp_ub),
            float(ncd_ub),
            float(nap_ub),
            float(nad_ub),
            max_flow,
            min(max_flow, max(0.0, inputs.h_ca)),
            min(max_flow, max(0.0, inputs.h_ac)),
            max_flow,
        ],
        dtype=float,
    )
    if inputs.search_mode == "no_cross":
        upper_bounds[XCA] = 0.0
        upper_bounds[XAC] = 0.0
    elif inputs.search_mode == "cuda_prefill_ascend_decode":
        upper_bounds[NCD] = 0.0
        upper_bounds[NAP] = 0.0
        upper_bounds[XCC] = 0.0
        upper_bounds[XAC] = 0.0
        upper_bounds[XAA] = 0.0
    return upper_bounds


def _objective(inputs: CountOptimizationInputs) -> np.ndarray:
    # scipy.optimize.milp minimizes. The tiny resource penalty makes tied optima
    # prefer fewer unused instances and same-pool flow without changing the
    # reported goodput.
    resource_penalty = 1e-9
    cross_flow_penalty = 1e-9
    c = np.zeros(NUM_VARS, dtype=float)
    c[NCP] = resource_penalty * resource_footprint(inputs.cuda_prefill)
    c[NCD] = resource_penalty * resource_footprint(inputs.cuda_decode)
    c[NAP] = resource_penalty * resource_footprint(inputs.ascend_prefill)
    c[NAD] = resource_penalty * resource_footprint(inputs.ascend_decode)
    c[XCC] = -1.0
    c[XCA] = -(1.0 - cross_flow_penalty)
    c[XAC] = -(1.0 - cross_flow_penalty)
    c[XAA] = -1.0
    return c


def _constraints(inputs: CountOptimizationInputs) -> list[LinearConstraint]:
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(row: np.ndarray, lb: float = -np.inf, ub: float = np.inf) -> None:
        rows.append(row)
        lower.append(lb)
        upper.append(ub)

    # Total device budget per pool.
    row = np.zeros(NUM_VARS)
    row[NCP] = resource_footprint(inputs.cuda_prefill)
    row[NCD] = resource_footprint(inputs.cuda_decode)
    add(row, ub=inputs.cuda_pool.total_devices)

    row = np.zeros(NUM_VARS)
    row[NAP] = resource_footprint(inputs.ascend_prefill)
    row[NAD] = resource_footprint(inputs.ascend_decode)
    add(row, ub=inputs.ascend_pool.total_devices)

    # Role goodput bounds.
    row = np.zeros(NUM_VARS)
    row[NCP] = -max(0.0, inputs.mu_cp)
    row[XCC] = 1.0
    row[XCA] = 1.0
    add(row, ub=0.0)

    row = np.zeros(NUM_VARS)
    row[NAP] = -max(0.0, inputs.mu_ap)
    row[XAC] = 1.0
    row[XAA] = 1.0
    add(row, ub=0.0)

    row = np.zeros(NUM_VARS)
    row[NCD] = -max(0.0, inputs.mu_cd)
    row[XCC] = 1.0
    row[XAC] = 1.0
    add(row, ub=0.0)

    row = np.zeros(NUM_VARS)
    row[NAD] = -max(0.0, inputs.mu_ad)
    row[XCA] = 1.0
    row[XAA] = 1.0
    add(row, ub=0.0)

    # Cross-pool handoff goodput bounds.
    row = np.zeros(NUM_VARS)
    row[XCA] = 1.0
    add(row, ub=max(0.0, inputs.h_ca))

    row = np.zeros(NUM_VARS)
    row[XAC] = 1.0
    add(row, ub=max(0.0, inputs.h_ac))

    # At least one prefill and one decode instance must be selected.
    row = np.zeros(NUM_VARS)
    row[NCP] = 1.0
    row[NAP] = 1.0
    add(row, lb=1.0)

    row = np.zeros(NUM_VARS)
    row[NCD] = 1.0
    row[NAD] = 1.0
    add(row, lb=1.0)

    _add_low_affinity_constraints(
        rows,
        lower,
        upper,
        pool=inputs.cuda_pool,
        prefill=inputs.cuda_prefill,
        decode=inputs.cuda_decode,
        n_prefill_idx=NCP,
        n_decode_idx=NCD,
    )
    _add_low_affinity_constraints(
        rows,
        lower,
        upper,
        pool=inputs.ascend_pool,
        prefill=inputs.ascend_prefill,
        decode=inputs.ascend_decode,
        n_prefill_idx=NAP,
        n_decode_idx=NAD,
    )

    return [LinearConstraint(np.vstack(rows), np.array(lower), np.array(upper))]


def _add_low_affinity_constraints(
    rows: list[np.ndarray],
    lower: list[float],
    upper: list[float],
    pool: DevicePool,
    prefill: RoleShape | None,
    decode: RoleShape | None,
    n_prefill_idx: int,
    n_decode_idx: int,
) -> None:
    if pool.high_affinity:
        return
    row = np.zeros(NUM_VARS)
    if prefill is not None:
        row[n_prefill_idx] = prefill.local_devices_per_stage
    if decode is not None:
        row[n_decode_idx] = decode.local_devices_per_stage
    rows.append(row)
    lower.append(-np.inf)
    upper.append(float(pool.devices_per_node))


def _to_config(inputs: CountOptimizationInputs, values: np.ndarray) -> HeteroConfig | None:
    flows = FlowSolution(
        x_cc=_clean_flow(values[XCC]),
        x_ca=_clean_flow(values[XCA]),
        x_ac=_clean_flow(values[XAC]),
        x_aa=_clean_flow(values[XAA]),
    )
    if flows.lambda_est <= 1e-9:
        return None
    ncp = _required_count(flows.x_cc + flows.x_ca, inputs.mu_cp, inputs.cuda_prefill)
    ncd = _required_count(flows.x_cc + flows.x_ac, inputs.mu_cd, inputs.cuda_decode)
    nap = _required_count(flows.x_ac + flows.x_aa, inputs.mu_ap, inputs.ascend_prefill)
    nad = _required_count(flows.x_ca + flows.x_aa, inputs.mu_ad, inputs.ascend_decode)
    if not static_count_compatible(inputs.cuda_prefill, inputs.cuda_decode, ncp, ncd, inputs.cuda_pool):
        return None
    if not static_count_compatible(inputs.ascend_prefill, inputs.ascend_decode, nap, nad, inputs.ascend_pool):
        return None

    return HeteroConfig(
        cuda_prefill=RoleConfig(inputs.cuda_prefill if ncp > 0 else None, ncp),
        cuda_decode=RoleConfig(inputs.cuda_decode if ncd > 0 else None, ncd),
        ascend_prefill=RoleConfig(inputs.ascend_prefill if nap > 0 else None, nap),
        ascend_decode=RoleConfig(inputs.ascend_decode if nad > 0 else None, nad),
        flows=flows,
        estimated_goodput=flows.lambda_est,
    )


def _required_count(required_goodput: float, mu: float, shape: RoleShape | None) -> int:
    if shape is None or required_goodput <= 1e-9:
        return 0
    if mu <= 0:
        return 0
    return max(1, int(math.ceil((required_goodput - 1e-9) / mu)))


def _clean_flow(value: float) -> float:
    return 0.0 if abs(value) < 1e-9 else max(0.0, float(value))
