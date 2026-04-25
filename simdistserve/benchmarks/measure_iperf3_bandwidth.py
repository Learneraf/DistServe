#!/usr/bin/env python3
"""Run iperf3 bandwidth sweeps and persist raw plus summarized results.

This script is intended to be executed on the iperf3 client machine.
The server should already be running, for example:

    ./local/bin/iperf3 -s -B 10.129.165.27

Example client-side usage:

    python simdistserve/benchmarks/measure_iperf3_bandwidth.py \
        --iperf3 iperf3 \
        --server-ip 10.129.165.27 \
        --client-bind-ip 10.0.3.138 \
        --directions forward,reverse \
        --parallel-streams 1,2,4,8,16 \
        --duration 30 \
        --repeats 3 \
        --label ascend-cuda

The script writes:
  - one raw JSON file per run
  - runs.csv with one line per run
  - summary.csv with one line per (direction, parallel_streams)
  - summary.json with metadata and aggregated metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_STREAMS = [1, 2, 4, 8, 16]


@dataclass
class RunRecord:
    run_id: str
    label: str
    timestamp_utc: str
    direction_flag: str
    transfer_direction: str
    parallel_streams: int
    duration_s: int
    server_ip: str
    client_bind_ip: str
    port: int
    iperf3_path: str
    sender_bits_per_second: float
    receiver_bits_per_second: float
    sender_bytes: int
    receiver_bytes: int
    retransmits: int | None
    cpu_util_host_percent: float | None
    cpu_util_remote_percent: float | None
    stdout_json_path: str
    returncode: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run iperf3 sweeps and persist raw plus summarized results."
    )
    parser.add_argument("--iperf3", default="iperf3", help="Path to the iperf3 binary.")
    parser.add_argument("--server-ip", required=True, help="iperf3 server IP address.")
    parser.add_argument(
        "--client-bind-ip",
        default="",
        help="Optional client bind IP passed to iperf3 via -B.",
    )
    parser.add_argument("--port", type=int, default=5201, help="iperf3 server port.")
    parser.add_argument(
        "--directions",
        default="forward,reverse",
        help="Comma-separated directions to test: forward,reverse.",
    )
    parser.add_argument(
        "--parallel-streams",
        default="1,2,4,8,16",
        help="Comma-separated parallel stream counts, for example: 1,2,4,8,16",
    )
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeats per setting.")
    parser.add_argument(
        "--omit",
        type=int,
        default=0,
        help="Seconds to omit at the beginning of each test, passed to iperf3 via -O.",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Optional experiment label stored in output files.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Optional output directory. Default: "
            "simdistserve/benchmarks/results/network/iperf3/<timestamp>"
        ),
    )
    return parser.parse_args()


def parse_csv_ints(raw: str) -> list[int]:
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError("At least one integer value is required.")
    return values


def parse_directions(raw: str) -> list[str]:
    valid = {"forward", "reverse"}
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip().lower()
        if not chunk:
            continue
        if chunk not in valid:
            raise ValueError(f"Unsupported direction '{chunk}'. Expected one of {sorted(valid)}.")
        values.append(chunk)
    if not values:
        raise ValueError("At least one direction is required.")
    return values


def build_output_dir(explicit_output_dir: str) -> Path:
    if explicit_output_dir:
        return Path(explicit_output_dir).expanduser().resolve()

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        Path(__file__).resolve().parent
        / "results"
        / "network"
        / "iperf3"
        / now
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_iperf_command(
    iperf3_path: str,
    server_ip: str,
    client_bind_ip: str,
    port: int,
    duration: int,
    parallel_streams: int,
    omit: int,
    direction_flag: str,
) -> list[str]:
    cmd = [
        iperf3_path,
        "-c",
        server_ip,
        "-p",
        str(port),
        "-t",
        str(duration),
        "-P",
        str(parallel_streams),
        "--json",
    ]
    if client_bind_ip:
        cmd.extend(["-B", client_bind_ip])
    if omit > 0:
        cmd.extend(["-O", str(omit)])
    if direction_flag == "reverse":
        cmd.append("--reverse")
    return cmd


def run_once(
    args: argparse.Namespace,
    direction_flag: str,
    parallel_streams: int,
    repeat_idx: int,
    raw_dir: Path,
) -> RunRecord:
    now = datetime.now(timezone.utc)
    timestamp_utc = now.isoformat()
    run_id = f"{direction_flag}_p{parallel_streams}_r{repeat_idx}"
    transfer_direction = "server_to_client" if direction_flag == "reverse" else "client_to_server"

    cmd = build_iperf_command(
        iperf3_path=args.iperf3,
        server_ip=args.server_ip,
        client_bind_ip=args.client_bind_ip,
        port=args.port,
        duration=args.duration,
        parallel_streams=parallel_streams,
        omit=args.omit,
        direction_flag=direction_flag,
    )

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    raw_json_path = raw_dir / f"{run_id}.json"
    payload: dict[str, Any] = {}
    if proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout)
            raw_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except json.JSONDecodeError:
            raw_json_path.write_text(proc.stdout, encoding="utf-8")
            raise RuntimeError(
                f"iperf3 did not return valid JSON for {run_id}. "
                f"stderr={proc.stderr.strip()!r}"
            ) from None
    else:
        raise RuntimeError(f"iperf3 produced no stdout for {run_id}. stderr={proc.stderr.strip()!r}")

    if proc.returncode != 0 or "error" in payload:
        error_msg = payload.get("error", proc.stderr.strip())
        raise RuntimeError(f"iperf3 failed for {run_id}: {error_msg}")

    end = payload.get("end", {})
    sum_sent = end.get("sum_sent", {}) or {}
    sum_received = end.get("sum_received", {}) or {}
    cpu_utilization = end.get("cpu_utilization_percent", {}) or {}

    return RunRecord(
        run_id=run_id,
        label=args.label,
        timestamp_utc=timestamp_utc,
        direction_flag=direction_flag,
        transfer_direction=transfer_direction,
        parallel_streams=parallel_streams,
        duration_s=args.duration,
        server_ip=args.server_ip,
        client_bind_ip=args.client_bind_ip,
        port=args.port,
        iperf3_path=args.iperf3,
        sender_bits_per_second=float(sum_sent.get("bits_per_second", 0.0)),
        receiver_bits_per_second=float(sum_received.get("bits_per_second", 0.0)),
        sender_bytes=int(sum_sent.get("bytes", 0)),
        receiver_bytes=int(sum_received.get("bytes", 0)),
        retransmits=(int(sum_sent["retransmits"]) if "retransmits" in sum_sent else None),
        cpu_util_host_percent=(
            float(cpu_utilization["host_total"]) if "host_total" in cpu_utilization else None
        ),
        cpu_util_remote_percent=(
            float(cpu_utilization["remote_total"]) if "remote_total" in cpu_utilization else None
        ),
        stdout_json_path=str(raw_json_path),
        returncode=proc.returncode,
    )


def write_runs_csv(records: list[RunRecord], path: Path) -> None:
    fieldnames = list(asdict(records[0]).keys()) if records else list(asdict(
        RunRecord(
            run_id="",
            label="",
            timestamp_utc="",
            direction_flag="",
            transfer_direction="",
            parallel_streams=0,
            duration_s=0,
            server_ip="",
            client_bind_ip="",
            port=0,
            iperf3_path="",
            sender_bits_per_second=0.0,
            receiver_bits_per_second=0.0,
            sender_bytes=0,
            receiver_bytes=0,
            retransmits=None,
            cpu_util_host_percent=None,
            cpu_util_remote_percent=None,
            stdout_json_path="",
            returncode=0,
        )
    ).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def stddev(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) >= 2 else 0.0


def aggregate_records(records: list[RunRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[RunRecord]] = {}
    for record in records:
        grouped.setdefault((record.direction_flag, record.parallel_streams), []).append(record)

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(grouped.keys(), key=lambda item: (item[0], item[1])):
        group = grouped[key]
        sender_bps = [record.sender_bits_per_second for record in group]
        receiver_bps = [record.receiver_bits_per_second for record in group]
        retrans = [float(record.retransmits) for record in group if record.retransmits is not None]

        best_record = max(group, key=lambda record: record.receiver_bits_per_second)
        summary_rows.append(
            {
                "direction_flag": key[0],
                "transfer_direction": group[0].transfer_direction,
                "parallel_streams": key[1],
                "repeats": len(group),
                "mean_sender_bits_per_second": mean(sender_bps),
                "std_sender_bits_per_second": stddev(sender_bps),
                "mean_receiver_bits_per_second": mean(receiver_bps),
                "std_receiver_bits_per_second": stddev(receiver_bps),
                "mean_receiver_mbits_per_second": mean(receiver_bps) / 1e6,
                "best_receiver_bits_per_second": best_record.receiver_bits_per_second,
                "best_receiver_mbits_per_second": best_record.receiver_bits_per_second / 1e6,
                "mean_retransmits": mean(retrans) if retrans else None,
                "max_retransmits": max(retrans) if retrans else None,
                "best_run_id": best_record.run_id,
                "best_raw_json_path": best_record.stdout_json_path,
            }
        )

    return summary_rows


def write_summary_csv(summary_rows: list[dict[str, Any]], path: Path) -> None:
    if not summary_rows:
        raise ValueError("No summary rows to write.")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def write_summary_json(
    args: argparse.Namespace,
    output_dir: Path,
    records: list[RunRecord],
    summary_rows: list[dict[str, Any]],
) -> None:
    payload = {
        "label": args.label,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "server_ip": args.server_ip,
        "client_bind_ip": args.client_bind_ip,
        "port": args.port,
        "iperf3_path": args.iperf3,
        "duration_s": args.duration,
        "omit_s": args.omit,
        "directions": parse_directions(args.directions),
        "parallel_streams": parse_csv_ints(args.parallel_streams),
        "repeats": args.repeats,
        "num_runs": len(records),
        "runs_csv": str(output_dir / "runs.csv"),
        "summary_csv": str(output_dir / "summary.csv"),
        "summary": summary_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    directions = parse_directions(args.directions)
    parallel_streams_values = parse_csv_ints(args.parallel_streams)

    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    if args.duration <= 0:
        raise ValueError("--duration must be positive.")
    if args.port <= 0:
        raise ValueError("--port must be positive.")

    output_dir = build_output_dir(args.output_dir)
    raw_dir = output_dir / "raw"
    ensure_dir(raw_dir)

    records: list[RunRecord] = []
    failures: list[str] = []

    for direction_flag in directions:
        for parallel_streams in parallel_streams_values:
            for repeat_idx in range(1, args.repeats + 1):
                try:
                    record = run_once(
                        args=args,
                        direction_flag=direction_flag,
                        parallel_streams=parallel_streams,
                        repeat_idx=repeat_idx,
                        raw_dir=raw_dir,
                    )
                    records.append(record)
                    print(
                        f"[ok] {record.run_id}: "
                        f"receiver={record.receiver_bits_per_second / 1e6:.3f} Mbits/s "
                        f"retransmits={record.retransmits}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    failures.append(str(exc))
                    print(f"[failed] {direction_flag} p={parallel_streams} r={repeat_idx}: {exc}", file=sys.stderr)

    if not records:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1

    runs_csv_path = output_dir / "runs.csv"
    summary_csv_path = output_dir / "summary.csv"

    write_runs_csv(records, runs_csv_path)
    summary_rows = aggregate_records(records)
    write_summary_csv(summary_rows, summary_csv_path)
    write_summary_json(args, output_dir, records, summary_rows)

    if failures:
        (output_dir / "failures.log").write_text("\n".join(failures) + "\n", encoding="utf-8")

    print(f"Output directory: {output_dir}")
    print(f"Runs CSV:        {runs_csv_path}")
    print(f"Summary CSV:     {summary_csv_path}")
    print(f"Summary JSON:    {output_dir / 'summary.json'}")
    if failures:
        print(f"Failures log:    {output_dir / 'failures.log'}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
