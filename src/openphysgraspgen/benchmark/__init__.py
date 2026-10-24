# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
"""Benchmarks scored from the public dataset alone, on a CPU.

Scoring a dataset needs ``pip install "openphysgraspgen[dataset]"``. PG-T1 is documented
in ``docs/benchmark/pg-t1.md``.
"""

from .pg_t1 import Refused, score_episode, task_block_problems
from .pg_t1_report import (
    DatasetUnreadable,
    report_text,
    score_dataset,
    wilson_interval,
)

__all__ = [
    "DatasetUnreadable",
    "Refused",
    "report_text",
    "score_dataset",
    "score_episode",
    "task_block_problems",
    "wilson_interval",
]
