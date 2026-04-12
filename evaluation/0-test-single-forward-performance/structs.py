import dataclasses
from typing import List, Optional

"""
We divide parameters into two types:
- Worker parameters, including model_dir, and tp_world_size. When these parameter
  change, we need to re-create workers
- Input parameters, including batch_size, input_len, output_len.
  We do not need to re-create workers when these parameters change
"""

@dataclasses.dataclass
class WorkerParam:
    """
    Worker parameters, defining how decoding is performed
    """
    model_dir: str
    tp_world_size: int
    max_req_num: int
    max_seq_len: int
    use_dummy_weights: bool = False
    gpu_memory_utilization: float = 0.92

@dataclasses.dataclass
class InputParam:
    """
    Input parameters, defining the request data
    """
    batch_size: int
    input_len: int
    output_len: int

@dataclasses.dataclass
class TestParamGroup:
    """
    One WorkerParam coupled with multiple InputParam
    """
    worker_param: WorkerParam
    input_params: list[InputParam]


@dataclasses.dataclass
class LifetimeEventRecord:
    timestamp: float
    event_type: str


@dataclasses.dataclass
class RequestResultRecord:
    prompt_len: int
    output_len: int
    start_time: float
    end_time: float
    token_timestamps: List[float]
    lifecycle_events: Optional[List[LifetimeEventRecord]]
    latency: float
    ftl: float
    tpot: float
 
