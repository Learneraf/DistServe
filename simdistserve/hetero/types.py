from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DeviceKind = Literal["cuda", "ascend"]
Role = Literal["prefill", "decode"]


@dataclass(frozen=True)
class DevicePool:
    kind: DeviceKind
    num_nodes: int
    devices_per_node: int
    high_affinity: bool

    @property
    def total_devices(self) -> int:
        return self.num_nodes * self.devices_per_node


@dataclass(frozen=True)
class RoleShape:
    device: DeviceKind
    role: Role
    tp: int
    pp_local: int
    pp_cross: int

    @property
    def total_pp(self) -> int:
        return self.pp_local * self.pp_cross

    @property
    def local_devices_per_stage(self) -> int:
        return self.tp * self.pp_local

    @property
    def devices_per_instance(self) -> int:
        return self.local_devices_per_stage * self.pp_cross


@dataclass(frozen=True)
class RoleConfig:
    shape: RoleShape | None
    num_instances: int


@dataclass(frozen=True)
class HeteroConfig:
    cuda_prefill: RoleConfig
    cuda_decode: RoleConfig
    ascend_prefill: RoleConfig
    ascend_decode: RoleConfig
    flows: "FlowSolution"
    estimated_goodput: float


@dataclass(frozen=True)
class HandoffGoodput:
    cuda_to_ascend: float
    ascend_to_cuda: float

    def ca(self, prefill_shape: RoleShape | None, decode_shape: RoleShape | None) -> float:
        if prefill_shape is None or decode_shape is None:
            return 0.0
        return float(self.cuda_to_ascend)

    def ac(self, prefill_shape: RoleShape | None, decode_shape: RoleShape | None) -> float:
        if prefill_shape is None or decode_shape is None:
            return 0.0
        return float(self.ascend_to_cuda)


@dataclass(frozen=True)
class FlowSolution:
    x_cc: float
    x_ca: float
    x_ac: float
    x_aa: float

    @property
    def lambda_est(self) -> float:
        return self.x_cc + self.x_ca + self.x_ac + self.x_aa
