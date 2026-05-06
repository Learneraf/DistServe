from functools import reduce
from itertools import chain
from typing import List, Optional

import simpy

from simdistserve.base.scheduler import HeteroGreedyScheduler, Scheduler
from simdistserve.base.worker import Worker, WorkerConfig
from simdistserve.utils import set_next_worker


VLLM_ASCEND_HANDOFF_DELAY_MS = 65.40447611397082
VLLM_ASCEND_HANDOFF_DELAY_PER_TOKEN_MS = 0.09436758011886258


class DisaggCluster:
    def __init__(
        self,
        env,
        N_prefill_instance: int = 1,
        N_decode_instance: int = 1,
        PP_prefill: int = 1,
        PP_decode: int = 1,
        worker_configs: 'Optional[WorkerConfig]' = None,
    ):
        prefill_instances = []
        decode_instances = []

        worker_kwargs = dict(
            global_scheduler=None,
            # is_last_in_pipeline
            # should_request_stay: bool = True,
            # prefill_max_batch_size: int = 0,
            # global_scheduler: 'Scheduler' = None,
            **(worker_configs or {})
        )
        cluster_config_keys = (
            "handoff_delay_ms",
            "handoff_delay_per_token_ms",
            "handoff_capacity",
            "handoff_delays",
            "ingress_delays",
            "prefill_first_token_visible_immediately",
            "scheduler_type",
        )
        cluster_config = {
            key: worker_kwargs.pop(key)
            for key in cluster_config_keys
            if key in worker_kwargs
        }
        engine_type = worker_kwargs.get("engine_type", "distserve")

        worker_id = 0
        for inst_id in range(N_prefill_instance):
            instance = []
            for i, p in enumerate(range(PP_prefill)):
                worker = Worker(env, worker_id, cluster=self, pipe_rank=i, **worker_kwargs)
                instance.append(worker)
                worker_id += 1

            # Cyclically chain instance within a GPU
            reduce(set_next_worker, chain(instance, (instance[0],)))
            instance[-1].is_last_in_pipeline = True
            instance[-1].should_request_stay = False
            prefill_instances.append(instance)
            pass

        for inst_id in range(N_decode_instance):
            instance = []
            for i, p in enumerate(range(PP_decode)):
                worker = Worker(env, worker_id, cluster=self, pipe_rank=i, **worker_kwargs)
                instance.append(worker)
                worker_id += 1

            # Cyclically chain instance within a GPU
            reduce(set_next_worker, chain(instance, (instance[0],)))
            instance[-1].is_last_in_pipeline = True
            decode_instances.append(instance)
            pass

        prefill_heads = [i[0] for i in prefill_instances]
        decode_heads = [i[0] for i in decode_instances]
        scheduler_type = cluster_config.get("scheduler_type", "default")
        if scheduler_type == "hetero_greedy":
            scheduler = HeteroGreedyScheduler(
                env,
                prefill_heads=prefill_heads,
                decode_heads=decode_heads,
                handoff_delays=cluster_config.get("handoff_delays"),
                ingress_delays=cluster_config.get("ingress_delays"),
            )
        elif scheduler_type == "default":
            scheduler = Scheduler(env, prefill_heads=prefill_heads, decode_heads=decode_heads)
        else:
            raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

        for last_in_prefill in (instances[-1] for instances in prefill_instances):
            last_in_prefill.global_scheduler = scheduler

        self.env = env
        self.PP_prefill = PP_prefill
        self.PP_decode = PP_decode
        self.engine_type = engine_type
        self.handoff_delay_ms = float(
            cluster_config.get(
                "handoff_delay_ms",
                VLLM_ASCEND_HANDOFF_DELAY_MS if engine_type == "vllm_ascend" else 0.0,
            )
        )
        self.handoff_delay_per_token_ms = float(
            cluster_config.get(
                "handoff_delay_per_token_ms",
                VLLM_ASCEND_HANDOFF_DELAY_PER_TOKEN_MS if engine_type == "vllm_ascend" else 0.0,
            )
        )
        self.handoff_capacity = max(1, int(cluster_config.get("handoff_capacity", 1)))
        self.prefill_first_token_visible_immediately = bool(
            cluster_config.get(
                "prefill_first_token_visible_immediately",
                engine_type != "vllm_ascend",
            )
        )
        self.handoff_resource = None
        if self.handoff_delay_ms > 0 or self.handoff_delay_per_token_ms > 0:
            self.handoff_resource = simpy.Resource(env, capacity=self.handoff_capacity)
        self.prefill_instances: 'List[List[Worker]]' = prefill_instances
        self.decode_instances: 'List[List[Worker]]' = decode_instances
        self.scheduler: 'Scheduler' = scheduler
        pass

    def get_handoff_delay(self, request) -> float:
        return self.handoff_delay_ms + self.handoff_delay_per_token_ms * request.current_context_len

    def start_handoff(self, request, source_worker: Worker, to_scheduler: bool = True):
        self.env.process(self._run_handoff(request, source_worker, to_scheduler))
        return

    def _run_handoff(self, request, source_worker: Worker, to_scheduler: bool):
        request.wait_handoff(wid=source_worker.wid)
        delay = self.get_handoff_delay(request)
        if self.handoff_resource is not None:
            with self.handoff_resource.request() as handoff_slot:
                yield handoff_slot
                request.do_handoff(wid=source_worker.wid)
                if delay > 0:
                    yield self.env.timeout(delay)
        else:
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
        return

    def get_all_workers(self):
        return list(
            chain(
                chain(*self.prefill_instances),
                chain(*self.decode_instances),
            )
        )

    def run(self):
        for instance in chain(self.prefill_instances, self.decode_instances):
            for worker in instance:
                self.env.process(worker.run())
        return self
