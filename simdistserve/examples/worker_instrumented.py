from __future__ import annotations

import argparse
import random
from functools import reduce
from itertools import chain
from typing import Iterable, Optional
from uuid import UUID

import simpy

from simdistserve.base.scheduler import Scheduler, put_requests_with_interarrivals
from simdistserve.base.worker import Worker, WorkerConfig
from simdistserve.clusters.disagg import (
    VLLM_ASCEND_HANDOFF_DELAY_MS,
    VLLM_ASCEND_HANDOFF_DELAY_PER_TOKEN_MS,
)
from simdistserve.estimators.time_estimator import get_decode_time, get_prefill_time
from simdistserve.examples.common import (
    ExampleConfig,
    _validate_runnability,
    build_requests,
    describe_config,
    format_frame,
    request_table,
)
from simdistserve.utils import set_next_worker


def _req_state(req) -> str:
    return (
        f"r{req.req_id}(remain={req.remain_prefill_lens}, cur={req.current_prefill_lens}, "
        f"counter={req.counter}, ctx={req.current_context_len}, "
        f"visible={getattr(req, 'first_token_visible', False)})"
    )


def _req_list(items) -> str:
    return "[" + ", ".join(_req_state(req) for req in items) + "]"


class InstrumentedScheduler(Scheduler):
    def _trace(self, message: str):
        print(f"[t={self.env.now:8.2f}][Scheduler] {message}")

    def schedule_prefill(self, req):
        worker, queue = self._find_best_worker_and_queue(self._prefill_heads, queues=self._prefill_queues)
        self._trace(
            f"schedule_prefill req={_req_state(req)} -> W{worker.wid}; "
            f"target_prefill_queue_before={_req_list(queue)}"
        )
        self._sched_request(req, worker, queue)
        self._trace(
            f"schedule_prefill req={req.req_id} done; "
            f"target_prefill_queue_after={_req_list(queue)}"
        )
        return

    def schedule_decode(self, req):
        if req.should_finish():
            self._trace(f"schedule_decode req={req.req_id}: already finished, force exit")
            req.finish_decode()
            return

        worker, queue = self._find_best_worker_and_queue(self._decode_heads, queues=self._decode_queues)
        self._trace(
            f"schedule_decode req={_req_state(req)} -> W{worker.wid}; "
            f"target_decode_queue_before={_req_list(queue)}"
        )
        req.wait_decode(worker.wid)
        self._sched_request(req, worker, queue)
        self._trace(
            f"schedule_decode req={req.req_id} done; "
            f"target_decode_queue_after={_req_list(queue)}"
        )
        return


class InstrumentedWorker(Worker):
    """Instrumented mirror of the core methods in base/worker.py."""

    def _trace(self, message: str):
        print(
            f"[t={self.env.now:8.2f}][W{self.wid}][pipe={self.pipe_rank}] {message}"
        )

    def _trace_queues(self, label: str):
        self._trace(
            f"{label}: prefill_queue={_req_list(self.prefill_queue)}, "
            f"decode_queue={_req_list(self.decode_queue)}"
        )

    def wakeup(self):
        self._trace("wakeup()")
        return super().wakeup()

    def forward_prefill(self, items):
        if not items:
            self._trace("forward_prefill: nothing to forward")
            return
        if not isinstance(items, Iterable):
            items = [items]
        items = list(items)
        self._trace(
            f"forward_prefill -> W{self.next_worker.wid if self.next_worker else None}: "
            f"{_req_list(items)}"
        )
        return super().forward_prefill(items)

    def forward_decode(self, items, to_scheduler: bool = False):
        if not items:
            self._trace("forward_decode: nothing to forward")
            return
        if not isinstance(items, Iterable):
            items = [items]
        items = list(items)
        dst = "scheduler" if to_scheduler else f"W{self.next_worker.wid if self.next_worker else None}"
        self._trace(f"forward_decode -> {dst}: {_req_list(items)}")
        return super().forward_decode(items, to_scheduler=to_scheduler)

    # Instrumented mirror of Worker.run in base/worker.py.
    def run(self):
        while True:
            if not (self.prefill_queue or self.decode_queue):
                self._trace("run: idle, waiting for wakeup")
                yield self._wakeup_event
                self._trace("run: woke up")

            self._trace_queues("run: loop start")
            decode_load = sum(req.current_context_len for req in self.decode_queue)
            threshold = int(self.decode_max_batch_size * self.decode_back_pressure)
            self._trace(
                f"run: decode_load={decode_load}, "
                f"back_pressure_threshold={threshold}, "
                f"has_back_pressure={self.has_back_pressure}"
            )

            if self.prefill_queue and not self.has_back_pressure:
                self._trace("run: branch -> do_prefill()")
                yield from self.do_prefill()
            else:
                self._trace("run: branch -> do_decode()")
                yield from self.do_decode()

            self._log_event("wait")
            self._trace("run: loop end")

    # Instrumented mirror of Worker._enter_decodes in base/worker.py.
    def _enter_decodes(self, remaining_tok_in_batch: int):
        self._trace(
            f"_enter_decodes: remaining_tok_in_batch={remaining_tok_in_batch}, "
            f"decode_queue_before={_req_list(self.decode_queue)}"
        )

        decode_max_tokens = 50000
        decode_len = min(remaining_tok_in_batch, len(self.decode_queue))
        decode_reqs = []

        for idx in range(decode_len):
            req = self.decode_queue[0]
            self._trace(
                f"_enter_decodes: candidate #{idx} -> {_req_state(req)}, "
                f"decode_max_tokens_left={decode_max_tokens}"
            )
            if (req.current_context_len + 1) > decode_max_tokens:
                self._trace(
                    f"_enter_decodes: stop because ctx+1={req.current_context_len + 1} "
                    f"> decode_max_tokens_left={decode_max_tokens}"
                )
                break
            decode_max_tokens -= (req.current_context_len + 1)
            decode_reqs.append(self.decode_queue.popleft())

        for req in decode_reqs:
            req.do_decode(wid=self.wid)

        self._trace(f"_enter_decodes: selected={_req_list(decode_reqs)}")
        self._trace_queues("_enter_decodes: queue after pop")
        return decode_reqs

    # Instrumented mirror of Worker._enter_prefill in base/worker.py.
    def _enter_prefill(self):
        self._trace_queues("_enter_prefill: queue before pop")
        result = []
        max_request_size = min(self.prefill_max_batch_size, len(self.prefill_queue))
        self._trace(
            f"_enter_prefill: max_request_size={max_request_size}, "
            f"is_first_in_pipeline={self.is_first_in_pipeline}, "
            f"enable_chunked_prefill={self.enable_chunked_prefill}, "
            f"prefill_max_tokens={self.prefill_max_tokens}"
        )

        if not self.is_first_in_pipeline:
            chunk_id = self.prefill_queue[0].chunk_id
            self._trace(f"_enter_prefill: non-first pipeline stage, chunk_id={chunk_id}")
            for idx in range(max_request_size):
                candidate = self.prefill_queue[0]
                if candidate.chunk_id != chunk_id:
                    self._trace(
                        f"_enter_prefill: stop at idx={idx} because candidate.chunk_id={candidate.chunk_id} "
                        f"!= active chunk_id={chunk_id}"
                    )
                    break
                self._trace(f"_enter_prefill: take {_req_state(candidate)}")
                result.append(self.prefill_queue.popleft())
        else:
            chunk_size = 0
            prefill_max_tokens = self.prefill_max_tokens
            chunk_id = UUID(int=random.getrandbits(128))
            self._trace(f"_enter_prefill: new chunk_id={chunk_id}")
            for idx in range(max_request_size):
                candidate = self.prefill_queue[0]
                self._trace(
                    f"_enter_prefill: candidate #{idx} before scheduling -> {_req_state(candidate)}, "
                    f"chunk_size={chunk_size}, prefill_max_tokens_left={prefill_max_tokens}"
                )

                if self.enable_chunked_prefill:
                    sched_size = min(
                        candidate.remain_prefill_lens,
                        prefill_max_tokens - chunk_size,
                    )
                    self._trace(f"_enter_prefill: chunked path, sched_size={sched_size}")
                    if sched_size <= 0:
                        self._trace("_enter_prefill: stop because sched_size <= 0")
                        break
                else:
                    sched_size = candidate.remain_prefill_lens
                    self._trace(f"_enter_prefill: non-chunked path, sched_size={sched_size}")
                    if sched_size > prefill_max_tokens:
                        self._trace(
                            f"_enter_prefill: stop because sched_size={sched_size} > "
                            f"prefill_max_tokens_left={prefill_max_tokens}"
                        )
                        break

                candidate.current_prefill_lens = sched_size
                candidate.remain_prefill_lens -= sched_size
                prefill_max_tokens -= sched_size
                candidate.chunk_id = chunk_id
                chunk_size += sched_size

                self._trace(f"_enter_prefill: candidate after scheduling -> {_req_state(candidate)}")
                result.append(self.prefill_queue.popleft())

        for req in result:
            req.do_prefill(wid=self.wid)

        self._trace(f"_enter_prefill: selected={_req_list(result)}")
        self._trace_queues("_enter_prefill: queue after pop")
        return result

    # Instrumented mirror of Worker._exit_prefill in base/worker.py.
    def _exit_prefill(self, prefill_items):
        for item in prefill_items:
            next_wid = self.next_worker.wid if self.next_worker else None
            self._trace(f"_exit_prefill: before finish_prefill -> {_req_state(item)}, next_wid={next_wid}")
            item.finish_prefill(
                is_finished_one_round=self.is_last_in_pipeline,
                wid=self.wid,
                next_wid=next_wid,
                generated_tokens=(1 if self.prefill_generates_first_token else 0),
                first_token_visible=getattr(
                    self.cluster,
                    "prefill_first_token_visible_immediately",
                    True,
                ),
            )
            self._trace(f"_exit_prefill: after finish_prefill -> {_req_state(item)}")

            if not self.is_last_in_pipeline or (item.remain_prefill_lens > 0):
                self._trace(
                    f"_exit_prefill: req={item.req_id} still has work in prefill "
                    f"(is_last_in_pipeline={self.is_last_in_pipeline}, remain={item.remain_prefill_lens})"
                )
                self.forward_prefill(item)
                continue

            if item.should_finish() and getattr(item, "_terminated", False):
                self._trace(f"_exit_prefill: req={item.req_id} already finished after prefill")
                continue

            if hasattr(self.cluster, "start_handoff"):
                self._trace(f"_exit_prefill: req={item.req_id} transitions from prefill to handoff")
                self.cluster.start_handoff(
                    item,
                    source_worker=self,
                    to_scheduler=(not self.should_request_stay),
                )
                continue

            self._trace(f"_exit_prefill: req={item.req_id} transitions from prefill to decode")
            if self.should_request_stay:
                item.wait_decode(wid=next_wid)
            self.forward_decode(item, to_scheduler=(not self.should_request_stay))

    # Instrumented mirror of Worker._exit_decode in base/worker.py.
    def _exit_decode(self, decode_reqs):
        if not decode_reqs:
            self._trace("_exit_decode: nothing to exit")
            return

        next_wid = self.next_worker.wid if self.next_worker else None
        for req in decode_reqs:
            before_counter = req.counter
            before_visible = getattr(req, "first_token_visible", False)
            self._trace(f"_exit_decode: before finish_decode -> {_req_state(req)}, next_wid={next_wid}")
            req.finish_decode(is_finished_one_round=self.is_last_in_pipeline, next_wid=next_wid, wid=self.wid)
            self._trace(
                f"_exit_decode: after finish_decode -> {_req_state(req)}, "
                f"counter {before_counter} -> {req.counter}, "
                f"visible {before_visible} -> {getattr(req, 'first_token_visible', False)}, "
                f"finished={req.should_finish()}"
            )

        next_decode_batch = tuple(req for req in decode_reqs if not req.should_finish())
        self._trace(f"_exit_decode: next_decode_batch={_req_list(next_decode_batch)}")
        self.forward_decode(next_decode_batch)

    # Instrumented mirror of Worker.do_prefill in base/worker.py.
    def do_prefill(self):
        prefill_items = self._enter_prefill()
        if self.enable_chunked_prefill:
            remaining_tok_in_batch = self.prefill_max_tokens - sum(x.current_prefill_lens for x in prefill_items)
            self._trace(
                f"do_prefill: chunked mode, remaining_tok_in_batch for decode admission={remaining_tok_in_batch}"
            )
            decode_reqs = self._enter_decodes(remaining_tok_in_batch)
        else:
            decode_reqs = []
            self._trace("do_prefill: non-chunked mode, no decode requests admitted into prefill round")

        num_tokens = sum(x.current_prefill_lens for x in prefill_items)
        num_tokens += len(decode_reqs)

        self._log_event(
            "do_prefill",
            num_tokens=num_tokens,
            prefill_bs=len(prefill_items),
            decode_bs=len(decode_reqs),
            prefill_len_list=[x.current_prefill_lens for x in prefill_items],
            decode_len_list=[x.current_context_len for x in decode_reqs],
        )

        delay = get_prefill_time(
            num_tokens,
            bs=len(prefill_items),
            decode_bs=len(decode_reqs),
            pp=self.cluster.PP_prefill,
            model_type=self.model_type,
            TP=self.TP_Prefill,
            prefill_len_list=[x.current_prefill_lens for x in prefill_items],
            engine_type=self.engine_type,
        )
        num_tokens = sum(x.current_context_len for x in (prefill_items + decode_reqs))
        if self.is_first_in_pipeline and self.engine_type != "vllm_ascend":
            ray_overhead = self.add_ray_overhead(num_tokens)
            delay += ray_overhead
            self._trace(f"do_prefill: add_ray_overhead={ray_overhead:.2f}")

        self._trace(
            f"do_prefill: prefill_items={_req_list(prefill_items)}, "
            f"decode_reqs={_req_list(decode_reqs)}, delay={delay:.2f}ms"
        )

        self._prefill_ips = len(prefill_items)
        yield self.env.timeout(delay)
        self._prefill_ips = 0
        self._trace("do_prefill: delay finished")
        self._exit_prefill(prefill_items)
        self._exit_decode(decode_reqs)

    # Instrumented mirror of Worker.do_decode in base/worker.py.
    def do_decode(self):
        decode_reqs = self._enter_decodes(self.decode_max_tokens)
        batch_size = len(decode_reqs)
        self._log_event(
            "do_decode",
            num_tokens=batch_size,
            decode_bs=batch_size,
            decode_len_list=[x.current_context_len for x in decode_reqs],
        )

        token_generated_list = [x.current_context_len + 1 for x in decode_reqs]
        input_lens = [req.prefill_lens for req in decode_reqs]
        output_lens = [req.output_lens for req in decode_reqs]
        current_context_lens = [req.current_context_len for req in decode_reqs]
        delay = get_decode_time(
            batch_size,
            pp=self.cluster.PP_decode,
            model_type=self.model_type,
            TP=self.TP_Decode,
            token_generated_list=token_generated_list,
            input_lens=input_lens,
            output_lens=output_lens,
            current_context_lens=current_context_lens,
            engine_type=self.engine_type,
        )
        num_tokens = sum(x.current_context_len for x in decode_reqs)
        if self.is_first_in_pipeline and self.engine_type != "vllm_ascend":
            ray_overhead = self.add_ray_overhead(num_tokens)
            delay += ray_overhead
            self._trace(f"do_decode: add_ray_overhead={ray_overhead:.2f}")

        self._trace(f"do_decode: decode_reqs={_req_list(decode_reqs)}, delay={delay:.2f}ms")
        yield self.env.timeout(delay)
        self._trace("do_decode: delay finished")
        self._exit_decode(decode_reqs)


class InstrumentedDisaggCluster:
    def __init__(
        self,
        env,
        N_prefill_instance: int = 1,
        N_decode_instance: int = 1,
        PP_prefill: int = 1,
        PP_decode: int = 1,
        worker_configs: Optional[WorkerConfig] = None,
    ):
        prefill_instances = []
        decode_instances = []
        worker_kwargs = dict(global_scheduler=None, **(worker_configs or {}))
        cluster_config_keys = (
            "handoff_delay_ms",
            "handoff_delay_per_token_ms",
            "handoff_capacity",
            "prefill_first_token_visible_immediately",
        )
        cluster_config = {
            key: worker_kwargs.pop(key)
            for key in cluster_config_keys
            if key in worker_kwargs
        }
        engine_type = worker_kwargs.get("engine_type", "distserve")

        worker_id = 0
        for _ in range(N_prefill_instance):
            instance = []
            for i in range(PP_prefill):
                worker = InstrumentedWorker(env, worker_id, cluster=self, pipe_rank=i, **worker_kwargs)
                instance.append(worker)
                worker_id += 1
            reduce(set_next_worker, chain(instance, (instance[0],)))
            instance[-1].is_last_in_pipeline = True
            instance[-1].should_request_stay = False
            prefill_instances.append(instance)

        for _ in range(N_decode_instance):
            instance = []
            for i in range(PP_decode):
                worker = InstrumentedWorker(env, worker_id, cluster=self, pipe_rank=i, **worker_kwargs)
                instance.append(worker)
                worker_id += 1
            reduce(set_next_worker, chain(instance, (instance[0],)))
            instance[-1].is_last_in_pipeline = True
            decode_instances.append(instance)

        scheduler = InstrumentedScheduler(
            env,
            prefill_heads=[instance[0] for instance in prefill_instances],
            decode_heads=[instance[0] for instance in decode_instances],
        )

        for last_in_prefill in (instance[-1] for instance in prefill_instances):
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
        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances
        self.scheduler = scheduler

    def _trace(self, message: str):
        print(f"[t={self.env.now:8.2f}][DisaggCluster] {message}")

    def get_handoff_delay(self, request) -> float:
        return self.handoff_delay_ms + self.handoff_delay_per_token_ms * request.current_context_len

    def start_handoff(self, request, source_worker: InstrumentedWorker, to_scheduler: bool = True):
        self._trace(
            f"start_handoff req={_req_state(request)} source=W{source_worker.wid} "
            f"to_scheduler={to_scheduler}"
        )
        self.env.process(self._run_handoff(request, source_worker, to_scheduler))
        return

    def _run_handoff(self, request, source_worker: InstrumentedWorker, to_scheduler: bool):
        request.wait_handoff(wid=source_worker.wid)
        delay = self.get_handoff_delay(request)
        self._trace(
            f"handoff queued req={_req_state(request)} source=W{source_worker.wid} delay={delay:.2f}ms"
        )
        if self.handoff_resource is not None:
            with self.handoff_resource.request() as handoff_slot:
                yield handoff_slot
                self._trace(
                    f"handoff start req={_req_state(request)} source=W{source_worker.wid} "
                    f"capacity={self.handoff_capacity}"
                )
                request.do_handoff(wid=source_worker.wid)
                if delay > 0:
                    yield self.env.timeout(delay)
        else:
            self._trace(f"handoff start req={_req_state(request)} source=W{source_worker.wid}")
            request.do_handoff(wid=source_worker.wid)
            if delay > 0:
                yield self.env.timeout(delay)

        request.finish_handoff(wid=source_worker.wid)
        self._trace(f"handoff finish req={_req_state(request)} source=W{source_worker.wid}")
        if (
            request.first_token_prefill
            and not self.prefill_first_token_visible_immediately
            and request.should_finish()
        ):
            request.mark_first_token_visible(wid=source_worker.wid)
        if request.should_finish():
            self._trace(f"handoff finish req={request.req_id}: request already completed")
            return

        if to_scheduler:
            self._trace(f"handoff finish req={request.req_id}: schedule_decode via scheduler")
            self.scheduler.schedule_decode(request)
            return

        next_wid = source_worker.next_worker.wid if source_worker.next_worker else None
        request.wait_decode(wid=next_wid)
        self._trace(f"handoff finish req={request.req_id}: forward_decode -> W{next_wid}")
        source_worker.forward_decode(request, to_scheduler=False)
        return

    def get_all_workers(self):
        return list(chain(chain(*self.prefill_instances), chain(*self.decode_instances)))

    def run(self):
        for instance in chain(self.prefill_instances, self.decode_instances):
            for worker in instance:
                self.env.process(worker.run())
        return self


class InstrumentedVLLMCluster:
    def __init__(
        self,
        env,
        N_instance: int = 1,
        PP: int = 1,
        worker_configs: Optional[WorkerConfig] = None,
    ):
        worker_kwargs = dict(global_scheduler=None, **(worker_configs or {}))
        instances = []

        worker_id = 0
        for _ in range(N_instance):
            instance = []
            for i in range(PP):
                worker = InstrumentedWorker(env, worker_id, cluster=self, pipe_rank=i, **worker_kwargs)
                instance.append(worker)
                worker_id += 1
            reduce(set_next_worker, chain(instance, (instance[0],)))
            instance[-1].is_last_in_pipeline = True
            instances.append(instance)

        self.env = env
        self.PP_prefill = PP
        self.PP_decode = PP
        self.instances = instances
        self.scheduler = InstrumentedScheduler(
            env,
            prefill_heads=[instance[0] for instance in instances],
            decode_heads=[instance[0] for instance in instances],
        )

    def get_all_workers(self):
        return list(chain.from_iterable(self.instances))

    def run(self):
        for instance in self.instances:
            for worker in instance:
                self.env.process(worker.run())
        return self


def _build_instrumented_cluster(env: simpy.Environment, config: ExampleConfig):
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
        return InstrumentedVLLMCluster(env=env, PP=config.pp_prefill, worker_configs=worker_config)

    return InstrumentedDisaggCluster(
        env=env,
        PP_prefill=config.pp_prefill,
        PP_decode=config.pp_decode,
        worker_configs=worker_config,
    )


def run_instrumented_case(name: str, request_specs, interarrival_ms, config: ExampleConfig):
    print("=" * 80)
    print(name)
    print("=" * 80)
    print(describe_config(config))
    print()
    print("Workload")
    print(format_frame(request_table(request_specs, interarrival_ms)))
    print()

    requests = build_requests(request_specs)
    env = simpy.Environment()
    cluster = _build_instrumented_cluster(env, config)
    cluster.run()
    put_requests_with_interarrivals(env, cluster.scheduler, interarrival_ms, requests)
    env.run()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Instrumented copy-path of Worker/Scheduler logic with print statements."
    )
    parser.add_argument(
        "--mode",
        choices=["chunked_prefill", "continuous_batching", "back_pressure"],
        default="chunked_prefill",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "chunked_prefill":
        run_instrumented_case(
            name="Instrumented Chunked Prefill",
            request_specs=[(1536, 6), (1024, 16)],
            interarrival_ms=[0.0, 10.0],
            config=ExampleConfig(
                backend="vllm_ascend",
                model="huggyllama/llama-7b",
                enable_chunked_prefill=False,
                prefill_max_tokens_cap=4096,
            ),
        )
        return

    if args.mode == "back_pressure":
        run_instrumented_case(
            name="Instrumented Decode Back-Pressure",
            request_specs=[(512, 4), (64, 3)],
            interarrival_ms=[0.0, 10.0],
            config=ExampleConfig(
                backend="vllm",
                model="facebook/opt-13b",
                tp_prefill=1,
                pp_prefill=2,
                pp_decode=2,
                decode_max_batch_size=100,
                decode_back_pressure=0.9,
            ),
        )
        return

    run_instrumented_case(
        name="Instrumented Continuous Batching",
        request_specs=[(512, 24), (64, 8), (64, 8), (64, 8)],
        interarrival_ms=[0.0, 70.0, 70.0, 70.0],
        config=ExampleConfig(
            backend="vllm",
            model="facebook/opt-13b",
            tp_prefill=1,
            pp_prefill=1,
        ),
    )


if __name__ == "__main__":
    main()
