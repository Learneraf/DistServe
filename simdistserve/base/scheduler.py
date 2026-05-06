from queue import Queue
from typing import List, TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from simdistserve.base.request import Request
    from simdistserve.base.worker import Worker


_DELAY_TABLE_ALIASES = {
    "cuda_to_ascend": ("cuda", "ascend"),
    "ascend_to_cuda": ("ascend", "cuda"),
    "ca": ("cuda", "ascend"),
    "ac": ("ascend", "cuda"),
}


def normalize_delay_table(raw):
    """Normalize JSON-friendly transfer-delay config into tuple-keyed table.

    Accepted forms:
      {("cuda", "ascend"): (fixed_ms, per_token_ms)}
      {"cuda_to_ascend": {"fixed_delay_ms": x, "delay_per_token_ms": y}}
      {"cuda_to_ascend": [x, y]}
    """
    if not raw:
        return {}

    table = {}
    for key, value in raw.items():
        if isinstance(key, tuple):
            src, dst = key
        else:
            src_dst = _DELAY_TABLE_ALIASES.get(str(key))
            if src_dst is None:
                parts = str(key).split("_to_")
                if len(parts) != 2:
                    raise ValueError(f"Unsupported delay-table key: {key!r}")
                src_dst = (parts[0], parts[1])
            src, dst = src_dst

        if isinstance(value, dict):
            fixed = value.get("fixed_delay_ms", value.get("fixed_ms", value.get("fixed", 0.0)))
            per_token = value.get(
                "delay_per_token_ms",
                value.get("per_token_ms", value.get("per_token", 0.0)),
            )
        else:
            fixed, per_token = value
        table[(src, dst)] = (float(fixed), float(per_token))
    return table


class Scheduler:
    def __init__(self, env, prefill_heads, decode_heads):
        self.env = env
        self._prefill_heads: 'List[Worker]' = prefill_heads
        self._prefill_queues = [i.prefill_queue for i in self._prefill_heads]
        self._decode_heads: 'List[Worker]' = decode_heads
        self._decode_queues = [i.decode_queue for i in self._decode_heads]
        pass

    @staticmethod
    def _find_best_worker_and_queue(workers, queues) -> 'Tuple[Worker, Union[Queue, List]]':
        # Peak the queue to find the least loaded worker.
        # Assume round-robin
        # Add the pending tasks in prefill
        worker, queue = min(zip(workers, queues), key=lambda x: x[0]._prefill_ips + len(x[1]))
        return worker, queue

    @staticmethod
    def _sched_request(req, worker, queue):
        queue.append(req)
        worker.wakeup()
        return

    def schedule_new_req(self, req: 'Request'):
        if req.counter < 0:
            return self.schedule_prefill(req)
        # This is for the 'decode-only' case.
        return self.schedule_decode(req)

    def schedule_prefill(self, req: 'Request'):
        assert req.counter < 0
        worker, queue = self._find_best_worker_and_queue(self._prefill_heads, queues=self._prefill_queues)
        self._sched_request(req, worker, queue)
        return

    def schedule_decode(self, req: 'Request'):
        assert req.counter >= 0
        if req.should_finish():
            # Force request to quit.
            req.finish_decode()
            return

        worker, queue = self._find_best_worker_and_queue(self._decode_heads, queues=self._decode_queues)
        req.wait_decode(worker.wid) # Artifact to prevent request having FTL != 0 when decode only.
        self._sched_request(req, worker, queue)
        return

    pass


class HeteroGreedyScheduler(Scheduler):
    def __init__(self, env, prefill_heads, decode_heads, handoff_delays=None, ingress_delays=None):
        super().__init__(env, prefill_heads, decode_heads)
        self.handoff_delays = normalize_delay_table(handoff_delays)
        self.ingress_delays = normalize_delay_table(ingress_delays)

    @staticmethod
    def _remaining_tokens(req: 'Request') -> int:
        return max(req.output_lens - max(req.counter, 0), 0)

    @staticmethod
    def _queued_decode_tokens(queue) -> int:
        return sum(HeteroGreedyScheduler._remaining_tokens(req) for req in queue)

    @staticmethod
    def _delay_from_table(table, src, dst, tokens: int) -> float:
        if src is None or dst is None or src == dst:
            return 0.0
        fixed, per_token = table.get((src, dst), (0.0, 0.0))
        return fixed + per_token * tokens

    def _score_prefill(self, req: 'Request', worker: 'Worker', queue) -> float:
        source_device = getattr(req, "source_device_type", None)
        target_device = getattr(worker, "device_type", None)
        ingress_penalty = self._delay_from_table(
            self.ingress_delays,
            source_device,
            target_device,
            req.prefill_lens,
        )
        backlog = len(queue) + getattr(worker, "_prefill_ips", 0) + 1
        return getattr(worker, "avg_prefill_time", 1.0) * backlog + ingress_penalty

    def _score_decode(self, req: 'Request', worker: 'Worker', queue) -> float:
        prefill_device = getattr(req, "prefill_device_type", None)
        decode_device = getattr(worker, "device_type", None)
        handoff_penalty = self._delay_from_table(
            self.handoff_delays,
            prefill_device,
            decode_device,
            req.current_context_len,
        )
        token_backlog = (
            self._queued_decode_tokens(queue)
            + getattr(worker, "_executing_decode_tokens", 0)
            + self._remaining_tokens(req)
        )
        return getattr(worker, "avg_tpot", 1.0) * token_backlog + handoff_penalty

    def schedule_prefill(self, req: 'Request'):
        assert req.counter < 0
        worker, queue = min(
            zip(self._prefill_heads, self._prefill_queues),
            key=lambda item: self._score_prefill(req, item[0], item[1]),
        )
        req.prefill_device_type = getattr(worker, "device_type", None)
        self._sched_request(req, worker, queue)
        return

    def schedule_decode(self, req: 'Request'):
        assert req.counter >= 0
        if req.should_finish():
            req.finish_decode()
            return
        worker, queue = min(
            zip(self._decode_heads, self._decode_queues),
            key=lambda item: self._score_decode(req, item[0], item[1]),
        )
        req.wait_decode(worker.wid)
        self._sched_request(req, worker, queue)
        return


def put_request(env, scheduler: 'Scheduler', delays, requests):
    for r, delay in zip(requests, delays):
        r.init()
        scheduler.schedule_new_req(r)
        yield env.timeout(delay)
    return


def put_request_at_time(env, scheduler: 'Scheduler', time, request: 'Request'):
    yield env.timeout(time)
    request.init()
    scheduler.schedule_new_req(request)
    return


def put_requests_with_interarrivals(env, scheduler: 'Scheduler', inter_arrivals, requests):
    """Put requests with the inter-arrivals."""
    assert len(inter_arrivals) == len(requests), (
        f"Number of requests ({len(requests)}) and inter-arrivals ({len(inter_arrivals)}) "
        f"should be the same."
    )
    wake_time = 0
    for r, ts in zip(requests, inter_arrivals):
        if r.env is None:
            r.env = env
        assert r.env == env
        wake_time += ts
        env.process(put_request_at_time(env, scheduler, wake_time, r))
    return
