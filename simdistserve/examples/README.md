# simdistserve Runnable Examples

These examples are meant to be the smallest runnable entry points for learning the simulator without going through the
full benchmark and binary-search pipeline.

Run them from the repository root:

```bash
cd /users/rh/DistServe
```

## 1. Inspect a Tiny Request Lifecycle

```bash
python -m simdistserve.examples.basic_lifecycle
```

What it shows:

- the hand-written request list
- each request event, including the current handoff / visibility events on disaggregated paths
- each worker batch
- TTFT / TPOT / total latency summary

Useful when you want to understand how `Request`, `Scheduler`, `Worker`, and `Cluster` interact.

## 2. Feel Continuous Batching

```bash
python -m simdistserve.examples.continuous_batching
```

What it shows:

- later requests arrive while an older request is already decoding
- decode batch size grows over time
- the same worker keeps rebatching active requests each decode round

Useful when you want to see the simulator's default continuous-batching behavior directly.

## 3. Feel Chunked Prefill

```bash
python -m simdistserve.examples.chunked_prefill
```

What it shows:

- one long prompt with chunking disabled
- the same long prompt with `enable_chunked_prefill=True`
- repeated `do_prefill` / `wait_prefill` rounds when the prompt is split

Useful when you want to see exactly how the simulator represents chunk splitting.

## 4. Feel Pipeline Parallelism

```bash
python -m simdistserve.examples.pipeline_parallelism
```

What it shows:

- the same burst workload with `pp=1` and `pp=2`
- how worker timelines change when requests pass through multiple pipeline stages

Useful when you want to see how pipeline stages show up in the simulator logs.

## 5. Trace Worker Mechanics

```bash
python -m simdistserve.examples.worker_mechanics_trace
```

What it shows:

- a plain Python chunk-splitting trace that mirrors `Worker._enter_prefill()`
- the current `finish_prefill -> handoff -> decode` transition on disaggregated runs
- decode rounds reconstructed by request id from simulator events
- a compact explanation of what each transition means

Useful when you want to map the code in `base/worker.py` directly to concrete state transitions.

## 6. Trace `has_back_pressure=True` in the Instrumented Worker

```bash
python -m simdistserve.examples.worker_instrumented --mode back_pressure
```

What it shows:

- one long-running decode request first
- a later prompt arrives while decode queue load is already high
- `run: decode_load=..., back_pressure_threshold=..., has_back_pressure=True`
- the worker chooses `do_decode()` even though `prefill_queue` is not empty

Useful when you want to see the exact branch behind:

```python
if self.prefill_queue and not self.has_back_pressure:
    yield from self.do_prefill()
else:
    yield from self.do_decode()
```

## 7. Compare Burst vs Spread Arrivals

```bash
python -m simdistserve.examples.compare_arrival_patterns
```

What it shows:

- the same mixed workload under two arrival patterns
- aggregate latency comparison
- per-request latency delta

Useful when you want to understand how queueing pressure changes latency even when the request list itself is identical.

## 8. Sweep Arrival Rate and Watch SLO Attainment Change

```bash
python -m simdistserve.examples.arrival_sweep
```

What it shows:

- how TTFT and TPOT change as request rate increases
- prefill / decode / joint SLO attainment percentages

Useful when you want to connect the low-level simulator to the higher-level binary search in `simulate.py`.

## Support Matrix

- `continuous batching`: yes. The simulator continuously rebuilds decode batches from the current decode queue each round.
- `chunked prefill`: yes. It is implemented in `Worker._enter_prefill()` behind `enable_chunked_prefill`.
- `pipeline parallelism`: yes. `PP` in `VLLMCluster` and `PP_prefill` / `PP_decode` in `DisaggCluster`.
- `disaggregated prefill/decode`: yes. `DisaggCluster` uses separate prefill and decode worker groups.
- `handoff latency / capacity`: yes. `DisaggCluster` now models request handoff explicitly between prefill and decode.
- `first token visibility after handoff`: yes. Request latency now uses `first_token_visible` when the backend defers visible output until handoff/decode.
- `decode back-pressure`: partially. `has_back_pressure` exists and can block more prefills when decode queue load grows.
- `decode token cap`: partially. The worker currently hardcodes `decode_max_tokens = 50000` inside `_enter_decodes()`, so that part is not faithfully parameterized yet.
- `speculative decoding`, `prefix caching`, `KV eviction`, real kernel-level scheduling, and other engine internals: not modeled here.

## Notes

- These scripts do not need an external dataset. They build requests directly from `(prefill_tokens, output_tokens)`.
- Disaggregated examples can now show `wait_handoff`, `do_handoff`, `finish_handoff`, and `first_token_visible` even
  when the configured handoff delay is zero, because the lifecycle is modeled explicitly.
- The latency numbers still come from `simdistserve/estimators/profiled_data`, so the examples remain faithful to the
  simulator's performance model.
- If you want to change the scenario, edit the `REQUEST_SPECS` / `INTERARRIVAL_MS` constants in the example files.
- The profiled models currently differ by backend in this repo snapshot:
  `distserve` defaults to `huggyllama/llama-7b`, while `vllm` defaults to `facebook/opt-13b`.
