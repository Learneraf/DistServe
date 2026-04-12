#!/usr/bin/env python3
"""
Merge per-case .exp files emitted by run_tp_grid.py into combined .exp files.

Examples:
    python /users/rh/DistServe/evaluation/0-test-single-forward-performance/aggregate_exp_results.py \
        --input-dir /users/rh/tmp/tp1_robust_exp \
        --output-dir /users/rh/tmp/tp1_robust_exp_combined

    python /users/rh/DistServe/evaluation/0-test-single-forward-performance/aggregate_exp_results.py \
        --input-dir /users/rh/tmp/tp1_robust_exp \
        --output-dir /users/rh/tmp/tp1_robust_exp_combined \
        --all-output /users/rh/tmp/tp1_robust_exp_combined/all_models.exp
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


FILENAME_RE = re.compile(r"tp(\d+)-bs(\d+)-in(\d+)-out(\d+)\.exp$")


def sort_key(path: Path) -> tuple[int, int, int, int, str]:
    match = FILENAME_RE.match(path.name)
    if match is None:
        return (10**9, 10**9, 10**9, 10**9, path.name)
    tp, batch_size, input_len, output_len = (int(x) for x in match.groups())
    return (tp, batch_size, input_len, output_len, path.name)


def load_request_results(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a JSON list.")
    return data


def write_json_list(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-case .exp files into combined request-result .exp files.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing per-model subdirectories of per-case .exp files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where combined per-model .exp files will be written.")
    parser.add_argument("--filename", type=str, default="combined.exp", help="Filename to use under each per-model output directory.")
    parser.add_argument("--all-output", type=Path, default=None, help="Optional path for a single combined file spanning all models.")
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    model_to_paths: dict[str, list[Path]] = defaultdict(list)
    for path in args.input_dir.rglob("*.exp"):
        if not path.is_file():
            continue
        if path.parent == args.input_dir:
            model_name = "root"
        else:
            model_name = path.parent.name
        model_to_paths[model_name].append(path)

    if not model_to_paths:
        raise FileNotFoundError(f"No .exp files found under {args.input_dir}")

    all_items: list[dict] = []
    for model_name, paths in sorted(model_to_paths.items()):
        combined: list[dict] = []
        for path in sorted(paths, key=sort_key):
            combined.extend(load_request_results(path))

        output_path = args.output_dir / model_name / args.filename
        write_json_list(output_path, combined)
        all_items.extend(combined)
        print(f"Wrote {len(combined)} requests to {output_path}")

    if args.all_output is not None:
        write_json_list(args.all_output, all_items)
        print(f"Wrote {len(all_items)} requests to {args.all_output}")


if __name__ == "__main__":
    main()
