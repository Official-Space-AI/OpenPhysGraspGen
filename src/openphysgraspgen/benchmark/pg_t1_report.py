# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
"""The PG-T1 report of a dataset: rates, task scores, failures and the check.

The rules are documented in docs/benchmark/pg-t1.md.
"""

from __future__ import annotations

import math
import textwrap
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .pg_t1 import (
    B_HOLD_S,
    B_RISE_M,
    C_HOLD_S,
    C_ID,
    C_RISE_M,
    C_UNDERSIDE_M,
    CLASSES,
    COLUMNS,
    CRITERIA,
    EXACT_CHECKS,
    FLOOR_ARMS,
    FORMAT,
    FORMAT_VERSION,
    SCORER_ID,
    STAGES,
    TASKS,
    TIE_M,
    WILSON_Z,
    Refused,
    Task,
    score_episode,
)


class DatasetUnreadable(ValueError):
    """A dataset the scorer cannot read at all."""


# --- the report -----------------------------------------------------------------------------


def wilson_interval(k: int, m: int, z: float = WILSON_Z) -> list[float] | None:
    """The Wilson score interval of k successes out of m, or None when m is 0."""
    if m <= 0:
        return None
    p = k / m
    d = 1.0 + z * z / m
    centre = (p + z * z / (2.0 * m)) / d
    half = z * math.sqrt(p * (1.0 - p) / m + z * z / (4.0 * m * m)) / d
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _is_decidable(record: Mapping[str, Any], criterion: str) -> bool:
    return record["decidable"] is None or record["decidable"][criterion]


def _stage(
    records: Sequence[Mapping[str, Any]], criterion: str, stage: str
) -> dict[str, Any]:
    pop = [r for r in records if stage == "lifted" or _is_decidable(r, criterion)]
    values = [r[criterion][stage] for r in pop]
    true = sum(v is True for v in values)
    false = sum(v is False for v in values)
    attempted = true + false
    return {
        "planned": len(records),
        "undecidable": len(records) - len(pop),
        "true": true,
        "false": false,
        "not_reached": len(values) - attempted,
        "of_attempted": true / attempted if attempted else None,
        "of_attempted_wilson95": wilson_interval(true, attempted),
        "of_all": true / len(pop) if pop else None,
        "of_all_wilson95": wilson_interval(true, len(pop)),
    }


def _cells(
    task: Task, records: Sequence[Mapping[str, Any]], criterion: str
) -> dict[str, Any]:
    stages = task.stages
    cells: dict[str, Any] = {
        s: _stage(records, criterion, s) if s in stages else None for s in STAGES
    }
    pop = [r for r in records if _is_decidable(r, criterion)]
    if len(stages) < len(STAGES):
        cells["task_score"], cells["task_score_why_null"] = None, "no put-down planned"
    elif not pop:
        cells["task_score"], cells["task_score_why_null"] = None, "no decidable episode"
    else:
        rates = [sum(r[criterion][s] is True for r in pop) / len(pop) for s in STAGES]
        cells["task_score"], cells["task_score_why_null"] = sum(rates) / 3, None
    last = cells[stages[-1]]
    cells["headline"] = (
        None
        if last["of_all"] is None
        else {
            "stage": stages[-1],
            "of_all": last["of_all"],
            "of_all_wilson95": last["of_all_wilson95"],
        }
    )
    counts = Counter(r[criterion]["class"] for r in records)
    reasons: dict[str, Counter[str]] = {}
    for r in records:
        tally = reasons.setdefault(r[criterion]["class"], Counter())
        tally.update(r[criterion]["reasons"])
    cells["failures"] = {
        "counts": {k: counts[k] for k in CLASSES},
        "of_all": {k: counts[k] / len(records) if records else None for k in CLASSES},
        "reasons": {
            k: dict(sorted(reasons[k].items(), key=lambda kv: (-kv[1], kv[0])))
            for k in CLASSES
            if reasons.get(k)
        },
    }
    return cells


def _table(task: Task, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"n": len(records), **{x: _cells(task, records, x) for x in CRITERIA}}


def _common(values: Sequence[Any]) -> Any:
    first = values[0] if values else None
    return first if all(v == first for v in values) else "mixed"


def _task_entry(name: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    task = TASKS[name]
    types: dict[str, list[Mapping[str, Any]]] = {}
    for r in records:
        types.setdefault(r["object_type"], []).append(r)
    labels = Counter(r["label"] for r in records)
    return {
        "criterion_id": task.criterion_id,
        "placed_rule": task.placed_rule,
        "n_object_types": len(types),
        "population": {
            "keep_labels": _common([r["keep_labels"] for r in records]),
            "keep_requires_lift": _common([r["keep_requires_lift"] for r in records]),
            "labels": dict(sorted(labels.items(), key=lambda kv: (-kv[1], str(kv[0])))),
        },
        "overall": _table(task, records),
        "by_object_type": {t: _table(task, rs) for t, rs in sorted(types.items())},
    }


def _tasks(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        name: _task_entry(name, mine)
        for name in TASKS
        if (mine := [r for r in records if r["task"] == name])
    }


def _check_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    exact = {e: {"applied": 0, "failed": 0} for e in EXACT_CHECKS}
    explained: Counter[str] = Counter()
    unavailable: Counter[str] = Counter()
    matched = 0
    failures = []
    for r in records:
        check = r["check"]
        for e, ok in check["exact"].items():
            exact[e]["applied"] += 1
            exact[e]["failed"] += int(not ok)
        explained.update(check["explained"].values())
        unavailable.update(check["unavailable"])
        matched += len(check["matched"])
        if check["failures"]:
            failures.append(
                {"episode_uid": r["episode_uid"], "failures": check["failures"]}
            )
    compared = (
        matched + sum(explained.values()) + sum(e["applied"] for e in exact.values())
    )
    return {
        "result": "failed" if failures else "passed" if compared else "unavailable",
        "exact": exact,
        "matched": matched,
        "explained": dict(sorted(explained.items())),
        "unavailable": dict(sorted(unavailable.items())),
        "failures": failures,
    }


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("openphysgraspgen")
    except PackageNotFoundError:
        return "unknown"


def _listed(root: Path, metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    files = metadata.get("files")
    if not isinstance(files, list):
        raise DatasetUnreadable(f"{root}: metadata.yaml lists no files")
    entries = [
        f for f in files if isinstance(f, Mapping) and f.get("kind") == "trajectory"
    ]
    if not entries:
        raise DatasetUnreadable(f"{root}: metadata.yaml lists no trajectory file")
    return sorted(
        entries, key=lambda f: (str(f.get("object_type")), str(f.get("episode_uid")))
    )


def _text(value: object) -> str | None:
    """A name from metadata.yaml as the report holds it: a string, or null."""
    return value if value is None or isinstance(value, str) else str(value)


def score_dataset(root: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score every listed episode of a dataset: (the report, one record per episode).

    Raises DatasetUnreadable when the dataset has no readable metadata.yaml of the
    format this scorer reads.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml

    from ..dataset import episode_metadata, read_metadata, read_trajectory

    root = Path(root)
    try:
        metadata = read_metadata(root)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise DatasetUnreadable(f"{root}: cannot read metadata.yaml: {exc}") from exc
    if not isinstance(metadata, Mapping):
        raise DatasetUnreadable(f"{root}: metadata.yaml is not a mapping")
    found = (metadata.get("format"), metadata.get("format_version"))
    if found != (FORMAT, FORMAT_VERSION):
        raise DatasetUnreadable(
            f"{root}: the dataset is {found[0]!r} {found[1]!r}; the scorer reads "
            f"{FORMAT!r} {FORMAT_VERSION!r}"
        )
    entries = _listed(root, metadata)
    records: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for entry in entries:
        uid, object_type = entry.get("episode_uid"), entry.get("object_type")
        rel = PurePosixPath(str(entry.get("path")))
        path = root / rel
        try:
            if not (isinstance(uid, str) and isinstance(object_type, str)):
                raise Refused(
                    "listing",
                    f"metadata.yaml lists episode_uid {uid!r} and object_type "
                    f"{object_type!r}; both must be strings",
                )
            if rel.is_absolute() or ".." in rel.parts:
                raise Refused(
                    "listing", f"{entry.get('path')!r} is not a path under the root"
                )
            try:
                meta = episode_metadata(path)
                # score_episode refuses a trajectory that lacks a column, naming it.
                names = set(pq.read_schema(path).names)
                columns = read_trajectory(path, [c for c in COLUMNS if c in names])
            except (OSError, ValueError, KeyError, pa.ArrowException) as exc:
                raise Refused("unreadable", f"{entry.get('path')}: {exc}") from exc
            if not isinstance(meta, Mapping):
                raise Refused(
                    "unreadable", f"{entry.get('path')}: the footer is no mapping"
                )
            record = score_episode(columns, meta, entry)
            records.append(record)
        except Refused as exc:
            record = {
                "episode_uid": _text(uid),
                "object_type": _text(object_type),
                "refused": exc.word,
                "detail": exc.detail,
            }
            refused.append(record)
        episodes.append(record)
    listed = {str(e.get("path")) for e in entries}
    on_disk = sorted(
        p.relative_to(root).as_posix()
        for p in (root / "trajectories").glob("*/*.parquet")
    )
    unlisted = [p for p in on_disk if p not in listed]
    check = _check_summary(records)
    demonstrations = [r for r in records if r["floor_arm"] is None]
    arms = {
        arm: _tasks([r for r in records if r["floor_arm"] == arm])
        for arm in FLOOR_ARMS
        if any(r["floor_arm"] == arm for r in records)
    }
    if check["result"] == "failed":
        status = "check_failed"
    elif refused or unlisted:
        status = "incomplete"
    else:
        status = "ok"
    report = {
        "benchmark": "PG-T1",
        "scorer": {"id": SCORER_ID, "version": _version()},
        "dataset": {
            "name": _text(metadata.get("name")),
            "version": _text(metadata.get("version")),
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "root": str(root),
        },
        "criteria": {
            "C": {
                "id": C_ID,
                "rise_m": C_RISE_M,
                "underside_clearance_m": C_UNDERSIDE_M,
                "tie_m": TIE_M,
                "hold_s": C_HOLD_S,
                "window": "rows k to k + round(2.0 * recorded_fps) - 1, k the first "
                "row up",
            },
            "B": {
                "rise_m": B_RISE_M,
                "tie_m": 0.0,
                "hold_s": B_HOLD_S,
                "window": "rows c to c + round(4.0 * recorded_fps), c the first crossing",
            },
        },
        "episodes": {
            "listed": len(entries),
            "scored": len(records),
            "refused": len(refused),
            "refusals": dict(sorted(Counter(r["refused"] for r in refused).items())),
            "refused_episodes": refused,
            "unlisted": len(unlisted),
            "unlisted_paths": unlisted,
        },
        "tasks": _tasks(demonstrations),
        "floor_arms": arms,
        "check": check,
        "status": status,
    }
    return report, episodes


# --- the printed table ----------------------------------------------------------------------

_CRITERION_NAMES = {"C": "C, 5 cm for 2 s", "B": "B, 12 cm for 4 s"}
_RULES = {
    "put-back": "put back where it rested",
    "put-down": "put down on the mat",
}
_WHY_NULL = {
    "no put-down planned": "not planned",
    "no decidable episode": "not decidable",
}


def _rate(stage: Mapping[str, Any] | None, key: str) -> str:
    if stage is None:
        return "not planned"
    if stage["of_all"] is None:
        return "not decidable"
    if stage[key] is None:
        return "not reached"
    if key == "of_attempted":
        m = stage["true"] + stage["false"]
    else:
        m = stage["planned"] - stage["undecidable"]
    lo, hi = stage[f"{key}_wilson95"]
    return f"{stage['true']:>4}/{m:<4} {stage[key]:.3f} [{lo:.3f}, {hi:.3f}]"


def _population(population: Mapping[str, Any], task: Task) -> str:
    labels, lift = population["keep_labels"], population["keep_requires_lift"]
    if labels == "mixed" or lift == "mixed":
        return "episodes kept under different rules"
    if labels is None and not lift:
        return "every episode the generation run made"
    kept = "the episodes the generation run kept"
    if labels is not None:
        kept += f": labels {', '.join(sorted(labels))}"
    if lift:
        rise = "lifted or rose 5 cm" if task.criterion_id == C_ID else "lifted"
        kept += (
            f"; only those that {rise} or were stopped on a drop, so the rates are "
            "conditional on the lift"
        )
    return kept


# Joins words the printed table keeps on one line.
_KEEP = "\0"


def _field(name: str, text: str, indent: int = 2, column: int = 16) -> list[str]:
    """A labelled field of the printed table, wrapped in the column after its label."""
    head = " " * indent + f"{name:<{column - indent - 1}} "
    lines = textwrap.wrap(
        text,
        96,
        initial_indent=head,
        subsequent_indent=" " * column,
        break_long_words=False,
    )
    return [line.replace(_KEEP, " ") for line in lines]


def _task_text(
    name: str, entry: Mapping[str, Any], arm: str | None = None
) -> list[str]:
    task = TASKS[name]
    overall = entry["overall"]
    types = entry["n_object_types"]
    n = overall["n"]
    head = (
        f"{name}: {n} episode{'s' if n != 1 else ''}, "
        f"{types} object type{'s' if types != 1 else ''}"
    )
    if entry["criterion_id"] is not None:
        head += f", criterion {entry['criterion_id']}"
    if arm is not None:
        head += f", floor arm {arm}"
    rule = entry["placed_rule"]
    labels = ", ".join(f"{k} {v}" for k, v in entry["population"]["labels"].items())
    lines = [
        head,
        *_field("placed", "not planned" if rule is None else _RULES[rule]),
        *_field("population", _population(entry["population"], task)),
        *_field("labels", labels),
    ]
    for x in CRITERIA:
        cells = overall[x]
        lines += ["", f"  {_CRITERION_NAMES[x]:<18}{'over attempted':<31}over all"]
        for s in STAGES:
            attempted, everything = (
                _rate(cells[s], k) for k in ("of_attempted", "of_all")
            )
            lines.append(f"    {s:<16}{attempted:<31}{everything}")
        score = cells["task_score"]
        lines.append(
            f"    {'task score':<16}"
            + (
                _WHY_NULL[cells["task_score_why_null"]]
                if score is None
                else f"{score:.3f}"
            )
        )
        top = cells["headline"]
        lines.append(
            f"    {'headline':<16}"
            + (
                "not decidable"
                if top is None
                else f"{top['stage']} over all {top['of_all']:.3f} "
                f"[{top['of_all_wilson95'][0]:.3f}, {top['of_all_wilson95'][1]:.3f}]"
            )
        )
        failures = cells["failures"]
        first = True
        for k in CLASSES:
            count = failures["counts"][k]
            if not count:
                continue
            words = failures["reasons"].get(k, {})
            text = f"{k.replace('_', _KEEP)}{_KEEP}{count}" + (
                ": " + ", ".join(f"{w}{_KEEP}{c}" for w, c in words.items())
                if words
                else ""
            )
            lines += _field("failures" if first else "", text, 4, 20)
            first = False
    return lines


def _check_text(check: Mapping[str, Any]) -> list[str]:
    names = {
        "E1": "agreement record",
        "E2": "lifted",
        "E3": "held",
        "E4": "put-down record",
        "E5": "carried and placed",
    }
    exact = [
        f"{e} {names[e]} {v['applied'] - v['failed']} of {v['applied']}"
        for e, v in check["exact"].items()
        if v["applied"]
    ]
    explained = ", ".join(f"{k} {v}" for k, v in check["explained"].items())
    unavailable = ", ".join(f"{k} on {v}" for k, v in check["unavailable"].items())
    lines = [
        f"check against the verdicts the episodes carry: {check['result']}",
        *_field("exact", ", ".join(exact) or "none applied"),
        *_field("matched", f"{check['matched']} stage verdicts"),
        *_field("explained", explained or "none"),
        *_field("unavailable", unavailable or "none"),
    ]
    for failure in check["failures"][:10]:
        lines += _field(str(failure["episode_uid"]), "; ".join(failure["failures"]))
    if len(check["failures"]) > 10:
        lines.append(f"  and {len(check['failures']) - 10} more episodes")
    return lines


def _episodes_text(episodes: Mapping[str, Any]) -> list[str]:
    refusals = ", ".join(f"{k} {v}" for k, v in episodes["refusals"].items())
    lines = [
        f"episodes      {episodes['listed']} listed, {episodes['scored']} scored, "
        f"{episodes['refused']} refused"
        + (f" ({refusals})" if refusals else "")
        + f", {episodes['unlisted']} unlisted",
    ]
    refused = episodes["refused_episodes"]
    for r in refused[:10]:
        lines += _field(str(r["episode_uid"]), f"{r['refused']}: {r['detail']}")
    if len(refused) > 10:
        lines.append(f"  and {len(refused) - 10} more refused episodes")
    paths = episodes["unlisted_paths"]
    for i, path in enumerate(paths[:10]):
        lines += _field("" if i else "unlisted", path)
    if len(paths) > 10:
        lines.append(f"  and {len(paths) - 10} more unlisted files")
    return lines


def report_text(report: Mapping[str, Any]) -> str:
    """The report as the command prints it."""
    dataset, scorer = report["dataset"], report["scorer"]
    lines = [
        f"PG-T1, scorer {scorer['id']} (openphysgraspgen {scorer['version']})",
        f"dataset       {dataset['name']} {dataset['version']} at {dataset['root']}",
        *_episodes_text(report["episodes"]),
    ]
    for name, entry in report["tasks"].items():
        lines += ["", *_task_text(name, entry)]
    for arm, tasks in report["floor_arms"].items():
        for name, entry in tasks.items():
            lines += ["", *_task_text(name, entry, arm)]
    lines += [
        "",
        *_check_text(report["check"]),
        "",
        f"status        {report['status']}",
    ]
    return "\n".join(lines) + "\n"
