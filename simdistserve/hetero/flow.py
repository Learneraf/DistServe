from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from simdistserve.hetero.types import FlowSolution


@dataclass(frozen=True)
class FlowInputs:
    mu_cp: float
    mu_cd: float
    mu_ap: float
    mu_ad: float
    ncp: int
    ncd: int
    nap: int
    nad: int
    h_ca: float
    h_ac: float


def solve_flow_allocation(inputs: FlowInputs) -> FlowSolution:
    """Solve the fixed-count 2x2 prefill/decode flow problem as max flow."""
    p_cuda = max(0.0, inputs.mu_cp * inputs.ncp)
    p_asc = max(0.0, inputs.mu_ap * inputs.nap)
    d_cuda = max(0.0, inputs.mu_cd * inputs.ncd)
    d_asc = max(0.0, inputs.mu_ad * inputs.nad)

    inf = p_cuda + p_asc + d_cuda + d_asc + inputs.h_ca + inputs.h_ac + 1.0
    source, pc, pa, dc, da, sink = range(6)
    capacity = [[0.0 for _ in range(6)] for _ in range(6)]
    capacity[source][pc] = p_cuda
    capacity[source][pa] = p_asc
    capacity[pc][dc] = inf
    capacity[pc][da] = max(0.0, inputs.h_ca)
    capacity[pa][dc] = max(0.0, inputs.h_ac)
    capacity[pa][da] = inf
    capacity[dc][sink] = d_cuda
    capacity[da][sink] = d_asc

    flow = [[0.0 for _ in range(6)] for _ in range(6)]

    while True:
        parent = [-1] * 6
        parent[source] = source
        q: deque[int] = deque([source])
        while q and parent[sink] == -1:
            u = q.popleft()
            for v in range(6):
                residual = capacity[u][v] - flow[u][v]
                if parent[v] == -1 and residual > 1e-12:
                    parent[v] = u
                    q.append(v)
                    if v == sink:
                        break
        if parent[sink] == -1:
            break

        bottleneck = float("inf")
        v = sink
        while v != source:
            u = parent[v]
            bottleneck = min(bottleneck, capacity[u][v] - flow[u][v])
            v = u

        v = sink
        while v != source:
            u = parent[v]
            flow[u][v] += bottleneck
            flow[v][u] -= bottleneck
            v = u

    return FlowSolution(
        x_cc=max(0.0, flow[pc][dc]),
        x_ca=max(0.0, flow[pc][da]),
        x_ac=max(0.0, flow[pa][dc]),
        x_aa=max(0.0, flow[pa][da]),
    )
