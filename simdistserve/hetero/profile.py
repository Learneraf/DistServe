from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from simdistserve.hetero.types import RoleShape


ENGINE_BY_DEVICE = {
    "cuda": "distserve",
    "ascend": "vllm_ascend",
}

PROFILE_CACHE_VERSION = 3


@dataclass
class GoodputProfileResult:
    goodput: float
    capped: bool
    effective_cap: float
    num_requests: int
    timed_out: bool = False


class _ProfileTimeout(Exception):
    pass


@dataclass
class SimulationGoodputProfiler:
    model: str
    workload: str
    prefill_target_ms: float
    decode_target_ms: float
    prefill_attainment: int
    decode_attainment: int
    max_rate: float
    epsilon: float
    profile_num_requests: int
    arrival: str = "poisson"
    cv: float = 1.0
    seed: int = 0
    cache_path: str | None = None
    auto_expand_max_rate: bool = True
    profile_max_rate_cap: float = 8192.0
    profile_min_profile_duration_s: float = 1.0
    profile_timeout_s: float | None = None
    cache: dict[RoleShape, float] = field(default_factory=dict)
    profile_results: dict[RoleShape, GoodputProfileResult] = field(default_factory=dict)
    _disk_cache: dict[str, GoodputProfileResult] = field(default_factory=dict, init=False, repr=False)
    _timed_out_shapes: set[RoleShape] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        cache_path = self._resolved_cache_path()
        if not cache_path.exists():
            return
        try:
            payload = json.loads(cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(payload, dict) and "version" in payload and payload.get("version") != PROFILE_CACHE_VERSION:
            return
        entries = payload.get("entries", payload) if isinstance(payload, dict) else {}
        if not isinstance(entries, dict):
            return
        for key, value in entries.items():
            parsed = self._parse_cache_entry(value)
            if parsed is not None:
                self._disk_cache[str(key)] = parsed

    def profile(self, shape: RoleShape) -> float:
        if shape not in self.cache:
            cache_key = self._cache_key(shape)
            if cache_key in self._disk_cache:
                result = self._disk_cache[cache_key]
                self.cache[shape] = result.goodput
                self.profile_results[shape] = result
            else:
                result = self._profile_uncached(shape)
                self.cache[shape] = result.goodput
                self.profile_results[shape] = result
                self._disk_cache[cache_key] = result
                self._save_disk_cache()
        return self.cache[shape]

    def result_for(self, shape: RoleShape) -> GoodputProfileResult | None:
        return self.profile_results.get(shape)

    def is_capped(self, shape: RoleShape) -> bool:
        result = self.result_for(shape)
        return bool(result and result.capped)

    def is_timed_out(self, shape: RoleShape) -> bool:
        result = self.result_for(shape)
        return bool(result and result.timed_out)

    def summary(self) -> dict[str, Any]:
        capped = [
            {
                "shape": {
                    "device": shape.device,
                    "role": shape.role,
                    "tp": shape.tp,
                    "pp_local": shape.pp_local,
                    "pp_cross": shape.pp_cross,
                    "total_pp": shape.total_pp,
                },
                "goodput": result.goodput,
                "effective_cap": result.effective_cap,
                "num_requests": result.num_requests,
            }
            for shape, result in self.profile_results.items()
            if result.capped
        ]
        timed_out = [
            {
                "shape": {
                    "device": shape.device,
                    "role": shape.role,
                    "tp": shape.tp,
                    "pp_local": shape.pp_local,
                    "pp_cross": shape.pp_cross,
                    "total_pp": shape.total_pp,
                },
                "goodput": result.goodput,
                "effective_cap": result.effective_cap,
                "num_requests": result.num_requests,
            }
            for shape, result in self.profile_results.items()
            if result.timed_out
        ]
        return {
            "num_profiled_shapes": len(self.profile_results),
            "num_capped_shapes": len(capped),
            "has_capped_shapes": bool(capped),
            "capped_shapes": capped,
            "num_timed_out_shapes": len(timed_out),
            "has_timed_out_shapes": bool(timed_out),
            "timed_out_shapes": timed_out,
        }

    def _parse_cache_entry(self, value: Any) -> GoodputProfileResult | None:
        try:
            if isinstance(value, dict):
                return GoodputProfileResult(
                    goodput=float(value["goodput"]),
                    capped=bool(value.get("capped", False)),
                    effective_cap=float(value.get("effective_cap", 0.0)),
                    num_requests=int(value.get("num_requests", self._profile_num_requests())),
                    timed_out=bool(value.get("timed_out", False)),
                )
            return GoodputProfileResult(
                goodput=float(value),
                capped=False,
                effective_cap=0.0,
                num_requests=self._profile_num_requests(),
                timed_out=False,
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _resolved_cache_path(self) -> Path:
        if self.cache_path:
            return Path(self.cache_path)
        env_path = os.environ.get("SIMDISTSERVE_HETERO_MU_CACHE")
        if env_path:
            return Path(env_path)
        return Path(__file__).resolve().parent / "results" / "cache" / "single_instance_goodput_cache.json"

    def _cache_key(self, shape: RoleShape) -> str:
        payload: dict[str, Any] = {
            "version": PROFILE_CACHE_VERSION,
            "model": self.model,
            "workload": self.workload,
            "prefill_target_ms": self.prefill_target_ms,
            "decode_target_ms": self.decode_target_ms,
            "prefill_attainment": self.prefill_attainment,
            "decode_attainment": self.decode_attainment,
            "max_rate": self.max_rate,
            "auto_expand_max_rate": self.auto_expand_max_rate,
            "profile_max_rate_cap": self.profile_max_rate_cap,
            "profile_min_profile_duration_s": self.profile_min_profile_duration_s,
            "epsilon": self.epsilon,
            "num_requests": self._profile_num_requests(),
            "profile_timeout_s": self.profile_timeout_s,
            "arrival": self.arrival,
            "cv": self.cv,
            "seed": self.seed,
            "distserve_profile": os.environ.get("SIMDISTSERVE_DISTSERVE_PROFILE", ""),
            "vllm_ascend_profile": os.environ.get("SIMDISTSERVE_VLLM_ASCEND_PROFILE", ""),
            "shape": {
                "device": shape.device,
                "role": shape.role,
                "tp": shape.tp,
                "pp_local": shape.pp_local,
                "pp_cross": shape.pp_cross,
                "total_pp": shape.total_pp,
            },
        }
        rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(rendered.encode("utf-8")).hexdigest()

    def _save_disk_cache(self) -> None:
        cache_path = self._resolved_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": PROFILE_CACHE_VERSION,
            "description": "Persistent cache for heterogeneous single-instance goodput profiling.",
            "entries": {
                key: {
                    "goodput": value.goodput,
                    "capped": value.capped,
                    "effective_cap": value.effective_cap,
                    "num_requests": value.num_requests,
                    "timed_out": value.timed_out,
                }
                for key, value in self._disk_cache.items()
            },
        }
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        tmp_path.replace(cache_path)

    def _profile_uncached(self, shape: RoleShape) -> GoodputProfileResult:
        cap = self._effective_max_rate_cap()
        low = 0.0
        high = min(float(self.max_rate), cap)
        best = 0.0

        if self.auto_expand_max_rate:
            high, high_meets_slo = self._expand_high_bound(shape, high, cap)
            if high >= cap and high_meets_slo:
                return GoodputProfileResult(
                    goodput=cap,
                    capped=True,
                    effective_cap=cap,
                    num_requests=self._profile_num_requests(),
                    timed_out=False,
                )

        while high - low > self.epsilon:
            rate = (low + high) / 2.0
            if self._meets_slo(shape, rate):
                low = rate
                best = rate
            else:
                high = rate
        return GoodputProfileResult(
            goodput=best,
            capped=False,
            effective_cap=cap,
            num_requests=self._profile_num_requests(),
            timed_out=shape in self._timed_out_shapes,
        )

    def _effective_max_rate_cap(self) -> float:
        cap = float(self.profile_max_rate_cap)
        min_duration_s = float(self.profile_min_profile_duration_s)
        if min_duration_s > 0:
            duration_limited_cap = self._profile_num_requests() / min_duration_s
            cap = min(cap, duration_limited_cap)
        return max(float(self.epsilon), cap)

    def _profile_num_requests(self) -> int:
        return max(1, int(self.profile_num_requests))

    def _expand_high_bound(self, shape: RoleShape, high: float, cap: float) -> tuple[float, bool]:
        high = max(float(high), self.epsilon)
        cap = max(high, float(cap))
        high_meets_slo = self._meets_slo(shape, high)
        while high < cap and high_meets_slo:
            high = min(high * 2.0, cap)
            high_meets_slo = self._meets_slo(shape, high)
        return high, high_meets_slo

    def _meets_slo(self, shape: RoleShape, rate: float) -> bool:
        from simdistserve.benchmarks.simulate_dist import parse_args, run_experiment

        backend = ENGINE_BY_DEVICE[shape.device]
        total_pp = shape.total_pp
        very_large_target = 10**12

        if shape.role == "prefill":
            prefill_target = self.prefill_target_ms
            decode_target = very_large_target
        else:
            prefill_target = very_large_target
            decode_target = self.decode_target_ms

        args = [
            "--backend", backend,
            "--model", self.model,
            "--workload", self.workload,
            "--arrival", self.arrival,
            "--cv", str(self.cv),
            "--seed", str(self.seed),
            "--N", str(self._profile_num_requests()),
            "--rate", str(rate),
            "--tp-prefill", str(shape.tp),
            "--pp-prefill", str(total_pp),
            "--tp-decode", str(shape.tp),
            "--pp-decode", str(total_pp),
            "--prefill-containment", str(self.prefill_attainment),
            "--prefill-target", str(int(prefill_target)),
            "--decode-containment", str(self.decode_attainment),
            "--decode-target", str(int(decode_target)),
            "--slas", "[]",
            "--slo-scales", "[1]",
            "--handoff-delay-ms", "0",
            "--handoff-delay-per-token-ms", "0",
            "--prefill-first-token-visible-immediately",
        ]
        parsed = parse_args(args)
        previous_handler = None
        try:
            if self.profile_timeout_s is not None and self.profile_timeout_s > 0:
                previous_handler = signal.getsignal(signal.SIGALRM)

                def _handle_timeout(_signum, _frame):
                    raise _ProfileTimeout()

                signal.signal(signal.SIGALRM, _handle_timeout)
                signal.setitimer(signal.ITIMER_REAL, float(self.profile_timeout_s))
            with contextlib.redirect_stdout(io.StringIO()):
                prefill_ok, decode_ok, _ = run_experiment(parsed)
        except _ProfileTimeout:
            self._timed_out_shapes.add(shape)
            return False
        except Exception:
            return False
        finally:
            if self.profile_timeout_s is not None and self.profile_timeout_s > 0:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                if previous_handler is not None:
                    signal.signal(signal.SIGALRM, previous_handler)

        return bool(prefill_ok if shape.role == "prefill" else decode_ok)
