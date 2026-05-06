from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Literal, Protocol
from tqdm import tqdm

from simdistserve.hetero.count_optimizer import CountOptimizationInputs, SearchMode, optimize_instance_allocation
from simdistserve.hetero.enumerate import (
    enumerate_role_shapes,
    static_shape_compatible,
)
from simdistserve.hetero.types import DevicePool, HandoffGoodput, HeteroConfig, RoleShape


class SingleInstanceGoodputProfiler(Protocol):
    def profile(self, shape: RoleShape) -> float:
        ...


CappedMuPolicy = Literal["keep", "exclude"]


@dataclass(frozen=True)
class HeteroSearchResult:
    configs: list[HeteroConfig]
    num_shape_tuples: int
    num_allocation_problems: int
    timing: dict[str, float]
    search_mode: SearchMode

    @property
    def best_config(self) -> HeteroConfig | None:
        return self.configs[0] if self.configs else None

    @property
    def num_count_tuples(self) -> int:
        return self.num_allocation_problems


def search_hetero_configs(
    model_type,
    cuda_pool: DevicePool,
    ascend_pool: DevicePool,
    handoff: HandoffGoodput,
    profiler: SingleInstanceGoodputProfiler,
    top_k: int = 10,
    search_mode: SearchMode = "milp",
    capped_mu_policy: CappedMuPolicy = "keep",
) -> HeteroSearchResult:
    if capped_mu_policy not in {"keep", "exclude"}:
        raise ValueError(f"Unknown capped_mu_policy: {capped_mu_policy}")

    total_start = time.perf_counter()
    enumerate_start = time.perf_counter()
    p_cuda = enumerate_role_shapes(model_type, cuda_pool, "prefill")
    d_cuda = enumerate_role_shapes(model_type, cuda_pool, "decode")
    p_asc = enumerate_role_shapes(model_type, ascend_pool, "prefill")
    d_asc = enumerate_role_shapes(model_type, ascend_pool, "decode")
    enumerate_seconds = time.perf_counter() - enumerate_start

    mu_cache: dict[RoleShape, float] = {}
    profile_seconds = 0.0
    allocation_seconds = 0.0

    def precompute_mu(shapes: list[RoleShape | None]) -> None:
        nonlocal profile_seconds
        seen: set[RoleShape] = set()
        for shape in tqdm(shapes, desc=search_mode):
            if shape is None or shape in seen:
                continue
            seen.add(shape)
            profile_start = time.perf_counter()
            mu_cache[shape] = float(profiler.profile(shape))
            profile_seconds += time.perf_counter() - profile_start

    def shape_is_capped(shape: RoleShape | None) -> bool:
        if shape is None:
            return False
        is_capped = getattr(profiler, "is_capped", None)
        if not callable(is_capped):
            return False
        return bool(is_capped(shape))

    search_loop_start = time.perf_counter()
    candidates: list[HeteroConfig] = []
    seen_configs: set[tuple] = set()
    num_shape_tuples = 0
    num_allocation_problems = 0

    pc_candidates = [None, *p_cuda]
    dc_candidates = [None, *d_cuda]
    pa_candidates = [None, *p_asc]
    da_candidates = [None, *d_asc]
    if search_mode == "cuda_prefill_ascend_decode":
        pc_candidates = p_cuda
        dc_candidates = [None]
        pa_candidates = [None]
        da_candidates = d_asc

    precompute_mu([*pc_candidates, *dc_candidates, *pa_candidates, *da_candidates])

    if capped_mu_policy == "exclude":
        pc_candidates = [shape for shape in pc_candidates if not shape_is_capped(shape)]
        dc_candidates = [shape for shape in dc_candidates if not shape_is_capped(shape)]
        pa_candidates = [shape for shape in pa_candidates if not shape_is_capped(shape)]
        da_candidates = [shape for shape in da_candidates if not shape_is_capped(shape)]

    def mu(shape: RoleShape | None) -> float:
        if shape is None:
            return 0.0
        return mu_cache[shape]

    for pc in pc_candidates:
        for dc in dc_candidates:
            if not static_shape_compatible(pc, dc, cuda_pool):
                continue
            for pa in pa_candidates:
                for da in da_candidates:
                    if not static_shape_compatible(pa, da, ascend_pool):
                        continue
                    if pc is None and pa is None:
                        continue
                    if dc is None and da is None:
                        continue
                    num_shape_tuples += 1
                    num_allocation_problems += 1
                    mu_pc = mu(pc)
                    mu_dc = mu(dc)
                    mu_pa = mu(pa)
                    mu_da = mu(da)
                    allocation_start = time.perf_counter()
                    config = optimize_instance_allocation(
                        CountOptimizationInputs(
                            cuda_pool=cuda_pool,
                            ascend_pool=ascend_pool,
                            cuda_prefill=pc,
                            cuda_decode=dc,
                            ascend_prefill=pa,
                            ascend_decode=da,
                            mu_cp=mu_pc,
                            mu_cd=mu_dc,
                            mu_ap=mu_pa,
                            mu_ad=mu_da,
                            h_ca=handoff.ca(pc, da),
                            h_ac=handoff.ac(pa, dc),
                            search_mode=search_mode,
                        )
                    )
                    allocation_seconds += time.perf_counter() - allocation_start
                    if config is None:
                        continue
                    key = (
                        config.cuda_prefill.shape,
                        config.cuda_decode.shape,
                        config.ascend_prefill.shape,
                        config.ascend_decode.shape,
                        config.cuda_prefill.num_instances,
                        config.cuda_decode.num_instances,
                        config.ascend_prefill.num_instances,
                        config.ascend_decode.num_instances,
                        round(config.flows.x_cc, 12),
                        round(config.flows.x_ca, 12),
                        round(config.flows.x_ac, 12),
                        round(config.flows.x_aa, 12),
                    )
                    if key in seen_configs:
                        continue
                    seen_configs.add(key)
                    candidates.append(config)
    search_loop_seconds = time.perf_counter() - search_loop_start

    sort_start = time.perf_counter()
    candidates.sort(key=lambda cfg: cfg.estimated_goodput, reverse=True)
    sort_seconds = time.perf_counter() - sort_start
    total_seconds = time.perf_counter() - total_start
    return HeteroSearchResult(
        configs=candidates[:top_k],
        num_shape_tuples=num_shape_tuples,
        num_allocation_problems=num_allocation_problems,
        search_mode=search_mode,
        timing={
            "enumerate_shapes_seconds": enumerate_seconds,
            "profile_seconds": profile_seconds,
            "precompute_mu_seconds": profile_seconds,
            "allocation_milp_seconds": allocation_seconds,
            "search_loop_seconds": search_loop_seconds,
            "sort_seconds": sort_seconds,
            "total_search_seconds": total_seconds,
        },
    )
