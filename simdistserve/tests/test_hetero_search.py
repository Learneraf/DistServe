import json

import pytest

from simdistserve.base.scheduler import HeteroGreedyScheduler, normalize_delay_table
from simdistserve.benchmarks.compute_handoff_goodput import compute_handoff_goodput
from simdistserve.hetero.count_optimizer import CountOptimizationInputs, optimize_instance_allocation
from simdistserve.hetero.enumerate import static_count_compatible
from simdistserve.hetero.flow import FlowInputs, solve_flow_allocation
from simdistserve.hetero.profile import SimulationGoodputProfiler
from simdistserve.hetero.search import search_hetero_configs
from simdistserve.hetero.types import DevicePool, RoleShape


def test_solve_flow_allocation_cross_handoff_bound():
    solution = solve_flow_allocation(
        FlowInputs(
            mu_cp=10,
            mu_cd=3,
            mu_ap=10,
            mu_ad=10,
            ncp=1,
            ncd=1,
            nap=1,
            nad=1,
            h_ca=2,
            h_ac=1,
        )
    )

    assert solution.x_ca <= 2
    assert solution.x_ac <= 1
    assert solution.lambda_est == 13


def test_low_affinity_count_compatibility():
    pool = DevicePool("cuda", num_nodes=2, devices_per_node=4, high_affinity=False)
    shape = RoleShape("cuda", "prefill", tp=1, pp_local=1, pp_cross=2)

    assert static_count_compatible(shape, None, n_prefill=4, n_decode=0, pool=pool)
    assert not static_count_compatible(shape, None, n_prefill=5, n_decode=0, pool=pool)


def test_optimize_instance_allocation_uses_milp_counts():
    cuda_pool = DevicePool("cuda", num_nodes=1, devices_per_node=2, high_affinity=True)
    ascend_pool = DevicePool("ascend", num_nodes=1, devices_per_node=2, high_affinity=True)
    pc = RoleShape("cuda", "prefill", tp=1, pp_local=1, pp_cross=1)
    dc = RoleShape("cuda", "decode", tp=1, pp_local=1, pp_cross=1)
    pa = RoleShape("ascend", "prefill", tp=1, pp_local=1, pp_cross=1)
    da = RoleShape("ascend", "decode", tp=1, pp_local=1, pp_cross=1)

    config = optimize_instance_allocation(
        CountOptimizationInputs(
            cuda_pool=cuda_pool,
            ascend_pool=ascend_pool,
            cuda_prefill=pc,
            cuda_decode=dc,
            ascend_prefill=pa,
            ascend_decode=da,
            mu_cp=10,
            mu_cd=3,
            mu_ap=10,
            mu_ad=10,
            h_ca=2,
            h_ac=1,
        )
    )

    assert config is not None
    assert config.cuda_prefill.num_instances == 1
    assert config.cuda_decode.num_instances == 1
    assert config.ascend_prefill.num_instances == 1
    assert config.ascend_decode.num_instances == 1
    assert config.flows.x_ca <= 2
    assert config.flows.x_ac <= 1
    assert config.estimated_goodput == pytest.approx(13)


def test_no_cross_mode_disables_cross_pool_flows():
    cuda_pool = DevicePool("cuda", num_nodes=1, devices_per_node=2, high_affinity=True)
    ascend_pool = DevicePool("ascend", num_nodes=1, devices_per_node=2, high_affinity=True)
    pc = RoleShape("cuda", "prefill", tp=1, pp_local=1, pp_cross=1)
    dc = RoleShape("cuda", "decode", tp=1, pp_local=1, pp_cross=1)
    pa = RoleShape("ascend", "prefill", tp=1, pp_local=1, pp_cross=1)
    da = RoleShape("ascend", "decode", tp=1, pp_local=1, pp_cross=1)

    config = optimize_instance_allocation(
        CountOptimizationInputs(
            cuda_pool=cuda_pool,
            ascend_pool=ascend_pool,
            cuda_prefill=pc,
            cuda_decode=dc,
            ascend_prefill=pa,
            ascend_decode=da,
            mu_cp=10,
            mu_cd=3,
            mu_ap=10,
            mu_ad=10,
            h_ca=100,
            h_ac=100,
            search_mode="no_cross",
        )
    )

    assert config is not None
    assert config.flows.x_ca == 0
    assert config.flows.x_ac == 0
    assert config.estimated_goodput == pytest.approx(13)


def test_cuda_prefill_ascend_decode_mode_only_uses_ca_flow():
    cuda_pool = DevicePool("cuda", num_nodes=1, devices_per_node=2, high_affinity=True)
    ascend_pool = DevicePool("ascend", num_nodes=1, devices_per_node=2, high_affinity=True)
    pc = RoleShape("cuda", "prefill", tp=1, pp_local=1, pp_cross=1)
    da = RoleShape("ascend", "decode", tp=1, pp_local=1, pp_cross=1)

    config = optimize_instance_allocation(
        CountOptimizationInputs(
            cuda_pool=cuda_pool,
            ascend_pool=ascend_pool,
            cuda_prefill=pc,
            cuda_decode=None,
            ascend_prefill=None,
            ascend_decode=da,
            mu_cp=10,
            mu_cd=0,
            mu_ap=0,
            mu_ad=4,
            h_ca=7,
            h_ac=0,
            search_mode="cuda_prefill_ascend_decode",
        )
    )

    assert config is not None
    assert config.cuda_decode.num_instances == 0
    assert config.ascend_prefill.num_instances == 0
    assert config.flows.x_cc == 0
    assert config.flows.x_ac == 0
    assert config.flows.x_aa == 0
    assert config.flows.x_ca == pytest.approx(7)


def test_scheduler_normalizes_json_handoff_delay_table():
    table = normalize_delay_table(
        {
            "cuda_to_ascend": {
                "fixed_delay_ms": 2.0,
                "delay_per_token_ms": 0.5,
            },
            "ascend_to_cuda": [3.0, 0.25],
        }
    )

    assert table[("cuda", "ascend")] == (2.0, 0.5)
    assert table[("ascend", "cuda")] == (3.0, 0.25)
    assert HeteroGreedyScheduler._delay_from_table(table, "cuda", "ascend", 10) == 7.0
    assert HeteroGreedyScheduler._delay_from_table(table, "cuda", "cuda", 10) == 0.0


def test_compute_handoff_goodput_outputs_latency_terms(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        """
{
  "hidden_size": 8,
  "num_attention_heads": 2,
  "num_key_value_heads": 2,
  "num_hidden_layers": 4,
  "torch_dtype": "float16"
}
""".strip()
        + "\n"
    )
    workload = tmp_path / "workload.jsonl"
    workload.write_text('{"prompt_len": 10}\n{"prompt_len": 30}\n')
    network = tmp_path / "summary.csv"
    network.write_text(
        "direction_flag,transfer_direction,parallel_streams,mean_receiver_bits_per_second\n"
        "reverse,cuda_to_ascend,8,8000000\n"
        "forward,ascend_to_cuda,8,16000000\n"
    )

    result = compute_handoff_goodput(
        model=str(model_dir),
        workload=workload,
        network_summary_csv=network,
        prompt_len_field="prompt_len",
        length_aggregation="mean",
        dtype=None,
        parallel_streams=8,
        bandwidth_column="mean_receiver_bits_per_second",
        cuda_to_ascend_direction="reverse",
        ascend_to_cuda_direction="forward",
    )

    # KV bytes/token = layers * kv_heads * head_dim * K/V * dtype_bytes
    # = 4 * 2 * 4 * 2 * 2 = 128 bytes = 1024 bits.
    assert result["payload"]["kv_bytes_per_token"] == 128
    assert result["handoff"]["cuda_to_ascend"]["delay_per_token_ms"] == pytest.approx(0.128)
    assert result["handoff"]["ascend_to_cuda"]["delay_per_token_ms"] == pytest.approx(0.064)
    assert result["handoff"]["cuda_to_ascend"]["handoff_goodput_upper_bound"] == pytest.approx(
        8000000 / (20 * 1024)
    )


def test_single_instance_profiler_records_capped_cache_entries(tmp_path):
    class AlwaysPassProfiler(SimulationGoodputProfiler):
        def _meets_slo(self, shape, rate):  # type: ignore[no-untyped-def]
            return True

    cache_path = tmp_path / "mu_cache.json"
    shape = RoleShape("cuda", "prefill", tp=1, pp_local=1, pp_cross=1)
    profiler = AlwaysPassProfiler(
        model="model",
        workload="workload",
        prefill_target_ms=100,
        decode_target_ms=100,
        prefill_attainment=90,
        decode_attainment=90,
        max_rate=8,
        epsilon=0.1,
        cache_path=str(cache_path),
        profile_num_requests=300,
        profile_max_rate_cap=200,
        profile_min_profile_duration_s=3,
    )

    assert profiler.profile(shape) == pytest.approx(100)

    payload = json.loads(cache_path.read_text())
    entry = next(iter(payload["entries"].values()))
    assert entry["goodput"] == pytest.approx(100)
    assert entry["capped"] is True
    assert entry["effective_cap"] == pytest.approx(100)
    assert entry["num_requests"] == 300
    summary = profiler.summary()
    assert summary["num_profiled_shapes"] == 1
    assert summary["num_capped_shapes"] == 1
    assert summary["has_capped_shapes"] is True
    assert profiler.is_capped(shape) is True


def test_search_can_exclude_capped_shapes():
    class StubProfiler:
        def __init__(self):
            self.cache = {}
            self._capped = set()

        def profile(self, shape):
            self.cache[shape] = 10.0
            if shape.tp == 1:
                self._capped.add(shape)
            return 10.0

        def is_capped(self, shape):
            return shape in self._capped

    profiler = StubProfiler()
    pool = DevicePool("cuda", num_nodes=1, devices_per_node=2, high_affinity=True)
    ascend_pool = DevicePool("ascend", num_nodes=1, devices_per_node=2, high_affinity=True)

    class DummyHandoff:
        def ca(self, *_args):
            return 0.0

        def ac(self, *_args):
            return 0.0

    from simdistserve.constants import ModelTypes

    result = search_hetero_configs(
        model_type=ModelTypes.model_str_to_object(ModelTypes.LLAMA_3_2_1B_LOCAL_PATH),
        cuda_pool=pool,
        ascend_pool=ascend_pool,
        handoff=DummyHandoff(),
        profiler=profiler,
        top_k=5,
        search_mode="milp",
        capped_mu_policy="exclude",
    )

    for cfg in result.configs:
        for shape in (
            cfg.cuda_prefill.shape,
            cfg.cuda_decode.shape,
            cfg.ascend_prefill.shape,
            cfg.ascend_decode.shape,
        ):
            assert shape is None or shape.tp != 1
