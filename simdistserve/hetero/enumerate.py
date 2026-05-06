from __future__ import annotations

from itertools import product

from simdistserve.constants import ModelTypes
from simdistserve.estimators.memory_estimator import get_model_possible_pp, get_model_possible_tp, is_model_runnable
from simdistserve.hetero.types import DevicePool, Role, RoleShape


def _formal_model_name(model_type) -> str:
    return ModelTypes.formalize_model_name(model_type)


def enumerate_role_shapes(model_type, pool: DevicePool, role: Role) -> list[RoleShape]:
    model_name = _formal_model_name(model_type)
    possible_tps = get_model_possible_tp(model_name)
    possible_pps = get_model_possible_pp(model_name)
    total_devices = pool.total_devices

    if pool.high_affinity:
        tp_candidates = [tp for tp in possible_tps if tp <= total_devices]
        pp_cross_candidates = [1]
        pp_local_candidates = [pp for pp in possible_pps if pp <= total_devices]
    else:
        tp_candidates = [tp for tp in possible_tps if tp <= pool.devices_per_node]
        pp_cross_candidates = [pp for pp in possible_pps if pp <= pool.num_nodes]
        pp_local_candidates = [pp for pp in possible_pps if pp <= pool.devices_per_node]

    shapes: list[RoleShape] = []
    for tp, pp_local, pp_cross in product(tp_candidates, pp_local_candidates, pp_cross_candidates):
        total_pp = pp_local * pp_cross
        if total_pp not in possible_pps:
            continue
        if not is_model_runnable(model_type, tp, total_pp):
            continue
        shape = RoleShape(pool.kind, role, tp, pp_local, pp_cross)
        if shape.devices_per_instance > total_devices:
            continue
        if not pool.high_affinity and shape.local_devices_per_stage > pool.devices_per_node:
            continue
        shapes.append(shape)
    return shapes


def resource_footprint(shape: RoleShape | None) -> int:
    return 0 if shape is None else shape.devices_per_instance


def get_instance_upper_bound(pool: DevicePool, shape: RoleShape | None) -> int:
    if shape is None:
        return 0
    if pool.high_affinity:
        return pool.total_devices // shape.devices_per_instance
    if shape.local_devices_per_stage <= 0:
        return 0
    stage_slots_per_node = pool.devices_per_node // shape.local_devices_per_stage
    total_stage_slots = pool.num_nodes * stage_slots_per_node
    return total_stage_slots // shape.pp_cross


def static_shape_compatible(prefill: RoleShape | None, decode: RoleShape | None, pool: DevicePool) -> bool:
    if prefill is None and decode is None:
        return True
    if prefill is not None and prefill.device != pool.kind:
        return False
    if decode is not None and decode.device != pool.kind:
        return False
    if pool.high_affinity:
        return True
    if prefill is not None and decode is not None:
        if prefill.pp_cross != decode.pp_cross:
            return False
        local_used = prefill.local_devices_per_stage + decode.local_devices_per_stage
        if local_used > pool.devices_per_node:
            return False
    return True


def static_count_compatible(
    prefill: RoleShape | None,
    decode: RoleShape | None,
    n_prefill: int,
    n_decode: int,
    pool: DevicePool,
) -> bool:
    if n_prefill < 0 or n_decode < 0:
        return False
    if prefill is None and n_prefill != 0:
        return False
    if decode is None and n_decode != 0:
        return False
    if prefill is not None and prefill.device != pool.kind:
        return False
    if decode is not None and decode.device != pool.kind:
        return False

    total_used = n_prefill * resource_footprint(prefill) + n_decode * resource_footprint(decode)
    if total_used > pool.total_devices:
        return False

    if pool.high_affinity:
        return True
    if prefill is not None and decode is not None:
        if prefill.pp_cross != decode.pp_cross:
            return False
    local_used = 0
    if prefill is not None:
        local_used += n_prefill * prefill.local_devices_per_stage
    if decode is not None:
        local_used += n_decode * decode.local_devices_per_stage
    return local_used <= pool.devices_per_node
