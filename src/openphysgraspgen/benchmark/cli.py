# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
"""``openphysgraspgen benchmark``: score a benchmark from a dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

EXIT_OK, EXIT_NOT_OK, EXIT_UNREADABLE = 0, 1, 2


def episodes_path(report: Path) -> Path:
    """Where the per-episode records go: beside the report, as JSON lines."""
    return report.with_name(f"{report.stem}.episodes.jsonl")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="openphysgraspgen benchmark",
        description="Score a benchmark from a dataset in the public format.",
    )
    benchmarks = parser.add_subparsers(
        dest="benchmark", required=True, metavar="BENCHMARK"
    )
    t1 = benchmarks.add_parser(
        "pg-t1",
        help="lifted, held and placed under criteria C and B",
        description="Score PG-T1: lifted, held and placed per episode under criteria C "
        "(5 cm for 2 s) and B (12 cm for 4 s). Prints one table per task and writes "
        "the report as JSON, with one record per episode beside it as JSON lines. "
        "Exits 0 when every listed episode was scored and no check failed, 1 when an "
        "episode was refused or unlisted or a check failed, 2 when the dataset cannot "
        "be read.",
    )
    t1.add_argument(
        "dataset", type=Path, help="the dataset's root, which holds metadata.yaml"
    )
    t1.add_argument(
        "--json",
        type=Path,
        default=Path("pg-t1.json"),
        metavar="PATH",
        help="where to write the report (default: pg-t1.json); the episodes go to "
        "PATH with .episodes.jsonl in place of its suffix",
    )
    args = parser.parse_args(argv)
    try:
        from .pg_t1_report import DatasetUnreadable, report_text, score_dataset

        report, episodes = score_dataset(args.dataset)
    except ModuleNotFoundError as exc:
        if exc.name not in ("pyarrow", "yaml"):
            raise
        print(
            f"openphysgraspgen benchmark needs {exc.name}: "
            'pip install "openphysgraspgen[dataset]"',
            file=sys.stderr,
        )
        return EXIT_UNREADABLE
    except DatasetUnreadable as exc:
        print(f"openphysgraspgen benchmark: {exc}", file=sys.stderr)
        return EXIT_UNREADABLE
    lines = episodes_path(args.json)
    try:
        args.json.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        lines.write_text(
            "".join(json.dumps(e, allow_nan=False) + "\n" for e in episodes),
            encoding="utf-8",
        )
    except OSError as exc:
        print(
            f"openphysgraspgen benchmark: cannot write the report: {exc}",
            file=sys.stderr,
        )
        return EXIT_UNREADABLE
    sys.stdout.write(report_text(report))
    print(f"report        {args.json}\nepisodes      {lines}")
    return EXIT_OK if report["status"] == "ok" else EXIT_NOT_OK
