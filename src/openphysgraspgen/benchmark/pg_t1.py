# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
"""PG-T1: lifted, held and placed per episode under criteria C and B, from the dataset.

score_episode scores one episode from its trajectory columns and its metadata;
pg_t1_report.score_dataset scores a whole dataset. The rules are documented in
docs/benchmark/pg-t1.md.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

SCORER_ID = "pg-t1-v1"
TASK_LAYOUT = "pg-t1-task-1"
# The dataset format this scorer reads.
FORMAT = "physgraspgen-public"
FORMAT_VERSION = "2.0"

B_RISE_M = 0.12
B_HOLD_S = 4.0
C_ID = "agreement-v1"
C_RISE_M = 0.05
C_UNDERSIDE_M = 0.005
C_HOLD_S = 2.0
# A rise this close to a threshold of C reaches it; B has no such tie.
TIE_M = 1.0e-9
REST_ROWS = 10
DESCENT_ROWS = 3
RELEASE_STEP_M = 1.0e-6
SET_DOWN_ROWS = 3
TOUCHDOWN_BAND_M = 0.001
PUSH_FACTOR = 0.5
WILSON_Z = 1.959963984540054

RETRY_PHASES = ("release", "recover")
STOP_REASONS = (None, "dropped", "never_lifted")
FLOOR_ARMS = ("no_vision_pick", "reflection")
LABELS = ("success", "fracture", "fail", "unknown")
STAGES = ("lifted", "held", "placed")
CRITERIA = ("C", "B")
CLASSES = (
    "success",
    "not_lifted",
    "dropped",
    "not_placed",
    "undecided",
    "not_decidable",
)
EXACT_CHECKS = ("E1", "E2", "E3", "E4", "E5")

# The trajectory columns the rules read.
COLUMNS = (
    "sim_step",
    "finite",
    "object.centroid",
    "object.bbox_min",
    "object.bbox_max",
    "robot.ee_pose",
    "robot.gripper_width_cmd_m",
    "object.n_robot_contacts",
    "robot.phase",
    "robot.waypoint",
    "force.ground_force_mag",
)

PLACE_KEYS = (
    "place_tol_m",
    "place_rest_radius_m",
    "place_max_descent_factor",
    "place_release_drop_tol_m",
)
CARRY_KEYS = (
    "carry_slip_tol_m",
    "set_down_speed_max_m_s",
    "grip_force_open_deadband_m",
)
CARRY_COUNTS = ("open_ramp_frames", "opening_to_end_frames", "rest_window_frames")
JUDGE_KEYS = (
    "lift_success_m",
    "approach_height_m",
    *PLACE_KEYS,
    *CARRY_KEYS,
    *CARRY_COUNTS,
    "place_error_tol_m",
)

# The agreement record, as the generator writes it.
AGREEMENT_KEYS = (
    "verdict",
    "reason",
    "rest_z_m",
    "first_up_frame",
    "window_end_frame",
    "min_rise_in_window_m",
    "min_bottom_clearance_in_window_m",
    "peak_rise_m",
    "any_window",
    "rise_at_stop_m",
)
# The put-down record, as the generator writes it, without the two facts that need the
# object's particles.
PLACE_FACTS = (
    "opening_frame",
    "fully_open_frame",
    "first_touch_frame",
    "carry_realised_m",
    "carry_slip_max_m",
    "dragged_max_m",
    "lowest_point_drop_m",
    "release_tcp_z_m",
    "release_tcp_minus_close_tcp_z_m",
    "fall_after_open_m",
    "set_down_speed_m_s",
    "set_down_bench_force_peak_n",
    "came_down_excess_max_m",
    "push_after_release_m",
    "gripper_contacts_after_open",
    "rest_wander_m",
    "final_centroid_xy",
    "place_error_m",
    "on_target_centroid_20mm",
    "prior_error_m",
    "retreat_drag_m",
    "carried",
    "placed",
    "carried_fail_reasons",
    "fail_reasons",
)
NOT_COMPUTED = ("final_yaw_change_rad", "final_tilt_change_rad")

SEARCH = ("search", "search")
PICK = ("approach", "descend", "close", "lift")
PUT_DOWN = (*SEARCH, *PICK, "hold", "carry", "place", "place", "retreat", "retreat")


@dataclass(frozen=True)
class Task:
    """What the scorer needs to know about a task the generator runs."""

    phases: tuple[str, ...]
    hold_s: float | None
    criterion_id: str | None = None
    carry_m: float | None = None
    centroid_bound: bool = False

    @property
    def stages(self) -> tuple[str, ...]:
        planned = {"lifted": True, "held": "hold" in self.phases}
        planned["placed"] = "place" in self.phases
        return tuple(s for s in STAGES if planned[s])

    @property
    def carries(self) -> bool:
        return self.carry_m is not None

    @property
    def placed_rule(self) -> str | None:
        if "place" not in self.phases:
            return None
        return "put-down" if self.carries else "put-back"


TASKS = {
    "pick": Task(PICK, None),
    "lift_hold": Task((*PICK, "hold"), 4.0),
    "lift_hold_place": Task((*PICK, "hold", "place", "place"), 4.0),
    "search_lift_hold": Task((*SEARCH, *PICK, "hold"), 4.0),
    "lift_hold_retry": Task((*PICK, "hold", *RETRY_PHASES, *PICK, "hold"), 4.0),
    "search_lift_hold_2s": Task((*SEARCH, *PICK, "hold"), 2.0, C_ID),
    "search_lift_hold_carry_place": Task(PUT_DOWN, 4.0, "carry-place-v1", 0.30),
    "search_lift_hold_carry_place_10cm": Task(
        PUT_DOWN, 3.0, "carry-place-10cm-v1", 0.10, True
    ),
}


class Refused(ValueError):
    """An episode the scorer does not score: word names the rule, detail the problem."""

    def __init__(self, word: str, detail: str) -> None:
        super().__init__(f"{word}: {detail}")
        self.word = word
        self.detail = detail


# --- the task block ---------------------------------------------------------------------

_MISSING = object()


def _number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _positive(value: object) -> bool:
    return _number(value) and value > 0


def _count(value: object, low: int = 1) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= low


def _schedule(value: object, n_phases: int) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(_count(v) for v in value)
        and (len(value) == 1 or len(value) >= n_phases)
    )


def task_block_problems(block: object, recorded_fps: object = None) -> list[str]:
    """Every missing or malformed key of an episode's task block; empty when it is valid."""
    if not isinstance(block, Mapping):
        return ["task: missing, or not a mapping"]
    problems: list[str] = []

    def get(
        where: Mapping[str, Any],
        prefix: str,
        key: str,
        ok: Callable[[Any], bool],
        rule: str,
        required: bool = True,
    ) -> Any:
        if key not in where:
            if required:
                problems.append(f"{prefix}{key}: missing")
            return _MISSING
        if not ok(where[key]):
            problems.append(f"{prefix}{key}: {rule}, got {where[key]!r}")
            return _MISSING
        return where[key]

    def key(name: str, ok: Callable[[Any], bool], rule: str, **kw: Any) -> Any:
        return get(block, "", name, ok, rule, **kw)

    key("layout", lambda v: v == TASK_LAYOUT, f"must be {TASK_LAYOUT!r}")
    key("source", lambda v: v == "generation", "must be 'generation'")
    name = key(
        "name",
        lambda v: isinstance(v, str) and v in TASKS,
        f"must be one of {', '.join(TASKS)}",
    )
    task = TASKS.get(name) if isinstance(name, str) else None
    if task is None:
        key("phases", lambda v: isinstance(v, list), "must be a list")
        key(
            "frames_per_waypoint",
            lambda v: _schedule(v, 1),
            "must be a list of ints >= 1",
        )
        key(
            "criterion_id",
            lambda v: v is None or isinstance(v, str),
            "must be a string or null",
        )
    else:
        key(
            "phases",
            lambda v: isinstance(v, list) and tuple(v) == task.phases,
            f"must be {list(task.phases)} under {name}",
        )
        key(
            "frames_per_waypoint",
            lambda v: _schedule(v, len(task.phases)),
            "must be a list of ints >= 1, one entry or at least one per phase",
        )
        key(
            "criterion_id",
            lambda v: v == task.criterion_id,
            f"must be {task.criterion_id!r} under {name}",
        )
    key(
        "stop_reason",
        lambda v: v in STOP_REASONS,
        "must be null, 'dropped' or 'never_lifted'",
    )

    def arm_ok(arm: object) -> bool:
        if arm is None:
            return True
        if arm not in FLOOR_ARMS or task is None:
            return arm in FLOOR_ARMS
        if arm == "reflection":
            return task.carries
        return task.criterion_id == C_ID

    arm = key(
        "floor_arm",
        arm_ok,
        "must be null, 'no_vision_pick' under search_lift_hold_2s or 'reflection' "
        "under a task that carries",
    )

    judge = key("judge", lambda v: isinstance(v, Mapping), "must be a mapping")
    if judge is not _MISSING:

        def typed(k: str, ok: Callable[[Any], bool], rule: str, needed: bool) -> Any:
            # A key the task does not need may be absent or null, else it has its type.
            check = ok if needed else lambda v: v is None or ok(v)
            return get(judge, "judge.", k, check, rule, needed)

        def number(k: str, needed: bool = True) -> Any:
            return typed(k, _positive, "must be a finite number > 0", needed)

        lift = number("lift_success_m")
        get(
            judge,
            "judge.",
            "approach_height_m",
            lambda v: _positive(v) and (lift is _MISSING or v > lift),
            "must be a finite number above lift_success_m",
        )
        places = task is not None and "place" in task.phases
        carries = task is not None and task.carries
        for k in PLACE_KEYS:
            number(k, places)
        for k in CARRY_KEYS:
            number(k, carries)
        for k in CARRY_COUNTS:
            typed(k, _count, "must be an int >= 1", carries)
        if task is None or task.centroid_bound:
            number("place_error_tol_m", task is not None)
        else:
            get(
                judge,
                "judge.",
                "place_error_tol_m",
                lambda v: v is None,
                f"must be null or absent under {name}",
                required=False,
            )

    if task is not None and task.carries:
        target = key(
            "place_target", lambda v: isinstance(v, Mapping), "must be a mapping"
        )
        if target is not _MISSING:
            for k in ("x_m", "y_m"):
                get(target, "place_target.", k, _number, "must be a finite number")
            get(target, "place_target.", "half_extent_m", _positive, "must be > 0")
            for k in ("centre_x_m", "centre_y_m"):
                get(
                    target,
                    "place_target.",
                    k,
                    _number,
                    "must be a finite number",
                    False,
                )
    elif task is not None:
        key(
            "place_target",
            lambda v: v is None,
            f"must be null under {name}, which does not carry",
            required=False,
        )

    key(
        "keep_labels",
        lambda v: (
            v is None
            or (isinstance(v, list) and bool(v) and all(x in LABELS for x in v))
        ),
        f"must be null or a list of labels out of {', '.join(LABELS)}",
    )
    key("keep_requires_lift", lambda v: isinstance(v, bool), "must be true or false")
    key(
        "stages",
        lambda v: (
            v is None
            or (
                isinstance(v, Mapping)
                and all(x is None or isinstance(x, bool) for x in v.values())
            )
        ),
        "must be null or a mapping of stage to true, false or null",
    )
    for k in ("agreement", "place"):
        key(
            k,
            lambda v: v is None or isinstance(v, Mapping),
            "must be null or a mapping",
        )
    key(
        "n_retries",
        lambda v: v is None or _count(v, 0),
        "must be an int >= 0",
        required=False,
    )
    if task is not None:
        before_close = set(task.phases[: task.phases.index("close")])
        key(
            "timing_scale",
            lambda v: (
                v is None
                or (
                    isinstance(v, Mapping)
                    and all(p in before_close and _positive(f) for p, f in v.items())
                )
            ),
            "must be null or map phases before the close to factors > 0",
            required=False,
        )
    criterion = key(
        "criterion",
        lambda v: v is None or isinstance(v, Mapping),
        "must be null or a mapping",
        required=False,
    )
    if isinstance(criterion, Mapping) and task is not None:
        _criterion_problems(problems, criterion, block, task, judge, arm, recorded_fps)
    return problems


def _criterion_problems(
    problems: list[str],
    criterion: Mapping[str, Any],
    block: Mapping[str, Any],
    task: Task,
    judge: object,
    arm: object,
    recorded_fps: object,
) -> None:
    def differs(key: str, value: object, against: object, what: str) -> None:
        if value is not None and value != against:
            problems.append(
                f"criterion.{key}: {value!r} differs from {what} {against!r}"
            )

    differs("id", criterion.get("id"), block.get("criterion_id"), "criterion_id")
    differs("task", criterion.get("task"), block.get("name"), "name")
    differs("fps", criterion.get("fps"), recorded_fps, "recorded_fps")
    internal = criterion.get("internal")
    if isinstance(internal, Mapping):
        differs(
            "internal.hold_s", internal.get("hold_s"), task.hold_s, "the task's hold"
        )
        differs(
            "internal.carry_m",
            internal.get("carry_m"),
            task.carry_m,
            "the task's carry",
        )
        if isinstance(judge, Mapping):
            for k, v in judge.items():
                differs(f"internal.{k}", internal.get(k), v, f"judge.{k}")
    placement = criterion.get("placement")
    if (
        isinstance(placement, Mapping)
        and arm is not _MISSING
        and placement.get("floor_arm") != arm
    ):
        problems.append(
            f"criterion.placement.floor_arm: {placement.get('floor_arm')!r} "
            f"differs from floor_arm {arm!r}"
        )


# --- one episode ------------------------------------------------------------------------


def _vectors(
    values: Any, n: int, order: np.ndarray, width: int, finite: np.ndarray | None = None
) -> list[tuple[float, ...] | None]:
    array = np.asarray(values, dtype=np.float64).reshape(n, width)[order]
    ok = np.isfinite(array).all(axis=1)
    if finite is not None:
        ok &= finite
    return [
        tuple(row) if good else None for row, good in zip(array.tolist(), ok.tolist())
    ]


def _scalars(values: Any, n: int, order: np.ndarray) -> list[float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(n)[order]
    return [v if math.isfinite(v) else None for v in array.tolist()]


class Episode:
    """One episode as the rules read it.

    Every per-row list has one entry per trajectory row in sim_step order, None where
    the value is missing or not finite; the object's values are None on a row whose
    finite is false. Build one with read_episode.
    """

    def __init__(
        self, meta: Mapping[str, Any], columns: Mapping[str, Any], order: np.ndarray
    ) -> None:
        block = meta["task"]
        judge = block["judge"]
        self.uid = meta.get("episode_uid")
        self.object_type = meta.get("object_type")
        self.label = meta.get("label")
        self.fps = int(meta["recorded_fps"])
        self.name = block["name"]
        self.task = TASKS[self.name]
        self.phases = self.task.phases
        self.schedule = list(block["frames_per_waypoint"])
        self.stop = block["stop_reason"]
        self.floor_arm = block["floor_arm"]
        self.keep_labels = block["keep_labels"]
        self.keep_requires_lift = block["keep_requires_lift"]
        self.timing_scale = block.get("timing_scale")
        self.n_retries = block.get("n_retries")
        self.stored = {k: block[k] for k in ("stages", "agreement", "place")}
        self.judge = {
            k: (v if k in CARRY_COUNTS else float(v))
            for k, v in judge.items()
            if k in JUDGE_KEYS and v is not None
        }
        target = block.get("place_target")
        self.target = (
            None
            if target is None
            else {k: float(v) for k, v in target.items() if _number(v)}
        )
        self.n = n = len(order)
        finite = np.asarray(columns["finite"], dtype=bool).reshape(n)[order]
        self.c = _vectors(columns["object.centroid"], n, order, 3, finite)
        self.bmin = _vectors(columns["object.bbox_min"], n, order, 3, finite)
        self.bmax = _vectors(columns["object.bbox_max"], n, order, 3, finite)
        pose = np.asarray(columns["robot.ee_pose"], dtype=np.float64).reshape(n, 7)
        self.p = _vectors(pose[:, :3], n, order, 3)
        self.q = _vectors(pose[:, 3:], n, order, 4)
        self.tool_z = _scalars(pose[:, 2], n, order)
        self.w = _scalars(columns["robot.gripper_width_cmd_m"], n, order)
        self.g = _scalars(columns["force.ground_force_mag"], n, order)
        contacts = np.asarray(columns["object.n_robot_contacts"]).reshape(n)[order]
        self.k = [int(v) for v in contacts.tolist()]
        phase = np.asarray(columns["robot.phase"], dtype=object).reshape(n)[order]
        self.phase = [str(v) for v in phase.tolist()]
        waypoint = np.asarray(columns["robot.waypoint"]).reshape(n)[order]
        self.waypoint = [int(v) for v in waypoint.tolist()]
        self.usable = [i for i, c in enumerate(self.c) if c is not None]
        self.n_b = round(B_HOLD_S * self.fps)
        self.h_c = round(C_HOLD_S * self.fps)
        hold = self.task.hold_s
        self.h_t = None if hold is None else round(hold * self.fps)
        self.retried = any(p in RETRY_PHASES for p in self.phase)
        self._final_rows: list[int] | None = None
        self._complete: dict[str, bool] = {}

    def count(self, waypoint: int) -> int:
        return self.schedule[min(waypoint, len(self.schedule) - 1)]

    def final_span(self) -> tuple[int, ...] | None:
        """The waypoints of the attempt the episode ended in, None for a single attempt."""
        if not any(p in RETRY_PHASES for p in self.phases):
            return None
        spans, current = [], []
        for w, p in enumerate(self.phases):
            if p in RETRY_PHASES:
                if current:
                    spans.append(tuple(current))
                    current = []
                continue
            current.append(w)
        if current:
            spans.append(tuple(current))
        last = max(self.waypoint)
        return next((s for s in reversed(spans) if s[0] <= last), None)

    def final_rows(self) -> list[int]:
        if self._final_rows is None:
            span = self.final_span()
            self._final_rows = (
                list(range(self.n))
                if span is None
                else [i for i in range(self.n) if self.waypoint[i] in span]
            )
        return self._final_rows

    def complete(self, phase: str) -> bool:
        """Whether the final attempt recorded every row the schedule plans for phase."""
        if phase not in self._complete:
            rows = [i for i in self.final_rows() if self.phase[i] == phase]
            span = self.final_span()
            planned = [
                w
                for w, p in enumerate(self.phases)
                if p == phase and (span is None or w in span)
            ]
            expected = sum(self.count(w) for w in planned)
            usable = sum(1 for i in rows if self.c[i] is not None)
            self._complete[phase] = bool(rows) and usable >= expected
        return self._complete[phase]


def read_episode(
    columns: Mapping[str, Any],
    meta: Mapping[str, Any],
    listing: Mapping[str, Any] | None = None,
) -> Episode:
    """Check one episode against the scorer's acceptance rules and read it.

    columns are its trajectory columns, as read_trajectory returns them, meta its
    footer metadata and listing its entry in metadata.yaml, when it has one. Raises
    Refused.
    """
    if meta.get("format") != FORMAT or meta.get("format_version") != FORMAT_VERSION:
        raise Refused(
            "format",
            f"the episode is {meta.get('format')} {meta.get('format_version')}, "
            f"not {FORMAT} {FORMAT_VERSION}",
        )
    if listing is not None and (
        meta.get("episode_uid") != listing.get("episode_uid")
        or meta.get("object_type") != listing.get("object_type")
    ):
        raise Refused(
            "listing",
            f"the file holds episode {meta.get('episode_uid')} of "
            f"{meta.get('object_type')}, metadata.yaml lists "
            f"{listing.get('episode_uid')} of {listing.get('object_type')}",
        )
    label = meta.get("label")
    if not (isinstance(label, str) and label in LABELS):
        raise Refused(
            "label", f"label must be one of {', '.join(LABELS)}, got {label!r}"
        )
    fps = meta.get("recorded_fps")
    if not _count(fps):
        raise Refused("recorded_fps", f"recorded_fps must be an int >= 1, got {fps!r}")
    missing = [c for c in COLUMNS if c not in columns]
    if missing:
        raise Refused("columns", f"the trajectory lacks {', '.join(missing)}")
    steps = np.asarray(columns["sim_step"]).reshape(-1)
    n = len(steps)
    order = np.argsort(steps, kind="stable")
    if n == 0 or not np.array_equal(steps[order], np.arange(1, n + 1)):
        raise Refused("rows", "the trajectory rows must be sim_step 1, 2, ..., n")
    listed = None if listing is None else listing.get("n_steps")
    if meta.get("n_steps") != n or (listing is not None and listed != n):
        raise Refused(
            "n_steps",
            f"the trajectory has {n} rows, its metadata says {meta.get('n_steps')}"
            + ("" if listing is None else f" and metadata.yaml {listed}"),
        )
    if meta.get("source_n_frames") != n:
        raise Refused(
            "trajectory_cut",
            f"the trajectory has {n} rows, the generator recorded "
            f"{meta.get('source_n_frames')}",
        )
    problems = task_block_problems(meta.get("task"), fps)
    if problems:
        raise Refused("task_block", "; ".join(problems))
    try:
        episode = Episode(meta, columns, order)
    except (TypeError, ValueError) as exc:
        raise Refused("columns", str(exc)) from exc
    for i, (w, p) in enumerate(zip(episode.waypoint, episode.phase)):
        if not (0 <= w < len(episode.phases) and episode.phases[w] == p):
            raise Refused(
                "phase_row",
                f"row {i} flies waypoint {w} in phase {p!r}, the plan does not",
            )
    return episode


# --- criterion B --------------------------------------------------------------------------


def held_from_crossing(
    episode: Episode, threshold_m: float, hold_rows: int
) -> tuple[bool | None, int | None]:
    """(held, crossing): every finite row from the first crossing c to c + hold_rows."""
    rows = [(i, episode.c[i][2]) for i in episode.usable]
    if not rows:
        return None, None
    rest = rows[0][1]
    crossing = next((i for i, z in rows if z - rest >= threshold_m), None)
    if crossing is None:
        return None, None
    end = crossing + hold_rows
    if rows[-1][0] < end:
        return (False if episode.stop == "dropped" else None), crossing
    return all(
        z - rest >= threshold_m for i, z in rows if crossing <= i <= end
    ), crossing


# --- criterion C --------------------------------------------------------------------------


def agreement(episode: Episode) -> dict[str, Any]:
    """The C verdict with its reason and facts, as the generator records them."""
    record: dict[str, Any] = dict.fromkeys(AGREEMENT_KEYS)
    record["any_window"] = False
    rows = [
        (i, c[2], b[2])
        for i, (c, b) in enumerate(zip(episode.c, episode.bmin))
        if c is not None and b is not None
    ]
    if not rows:
        record["reason"] = "incomplete"
        return record
    rest_z, rest_bottom = rows[0][1], rows[0][2]
    record["rest_z_m"] = rest_z

    def rises(z: float) -> bool:
        return (z - rest_z) + TIE_M >= C_RISE_M

    def clears(bottom: float) -> bool:
        return (bottom - rest_bottom) + TIE_M >= C_UNDERSIDE_M

    record["peak_rise_m"] = max(z - rest_z for _, z, _ in rows)
    hold = episode.h_c
    run, previous = 0, None
    for i, z, bottom in rows:
        up = rises(z) and clears(bottom)
        run = (
            (run + 1 if previous is not None and i == previous + 1 else 1) if up else 0
        )
        previous = i if up else None
        if run >= hold:
            record["any_window"] = True
            break
    k = next((i for i, z, bottom in rows if rises(z) and clears(bottom)), None)
    if k is None:
        if episode.stop == "never_lifted":
            record.update(verdict=False, reason="stopped_never_lifted")
        elif episode.stop == "dropped":
            record.update(verdict=False, reason="stopped_dropped")
        elif episode.complete("lift"):
            record.update(verdict=False, reason="never_crossed")
        else:
            record["reason"] = "incomplete"
        return record
    end = k + hold - 1
    inside = [(z, bottom) for i, z, bottom in rows if k <= i <= end]
    record.update(
        first_up_frame=k,
        window_end_frame=end,
        min_rise_in_window_m=min(z - rest_z for z, _ in inside),
        min_bottom_clearance_in_window_m=min(b - rest_bottom for _, b in inside),
    )
    if any(not rises(z) for z, _ in inside):
        record.update(verdict=False, reason="dipped_in_window")
    elif any(not clears(b) for _, b in inside):
        record.update(verdict=False, reason="still_supported")
    elif rows[-1][0] >= end:
        record["verdict"] = True
    elif episode.stop == "dropped":
        record.update(
            verdict=False,
            reason="stopped_dropped_in_window",
            rise_at_stop_m=rows[-1][1] - rest_z,
        )
    else:
        record["reason"] = "window_truncated"
    return record


# --- placed -------------------------------------------------------------------------------


def put_back(episode: Episode) -> tuple[bool | None, list[str], str | None]:
    """(verdict, reasons, why it is None) of the put-back of lift_hold_place."""
    ep, judge = episode, episode.judge
    hold = (
        ep.schedule[0]
        if len(ep.schedule) == 1
        else ep.schedule[ep.phases.index("hold")]
    )
    usable = ep.usable
    if not usable:
        return None, [], "no_usable_rows"
    rest = ep.c[usable[0]]
    lift = judge["lift_success_m"]
    p = next((q for q, i in enumerate(usable) if ep.c[i][2] - rest[2] >= lift), None)
    if p is None:
        return None, [], "internal_lift_not_met"
    end = p + hold
    if len(usable) <= end:
        return None, [], "internal_hold_truncated"
    if not all(ep.c[usable[q]][2] - rest[2] >= lift for q in range(p, end + 1)):
        return None, [], "internal_hold_not_met"
    after = usable[end:]
    if len(after) < REST_ROWS:
        return None, [], "fewer_than_10_rows_after_hold"
    grip = ep.w[after[0]]
    if grip is None:
        return None, [], "no_grip_width"
    release = next(
        (
            q
            for q in range(1, len(after))
            if (w := ep.w[after[q]]) is not None and w > grip + RELEASE_STEP_M
        ),
        None,
    )
    if release is None:
        return False, ["never_released"], None
    carried = after[: release + 1]
    tool = [ep.tool_z[i] for i in carried]
    if any(z is None for z in tool):
        return None, [], "ee_pose_missing"
    held = [ep.c[i][2] for i in carried]
    if len(held) <= DESCENT_ROWS:
        return None, [], "too_few_carried_rows"
    span = DESCENT_ROWS / ep.fps
    speeds = [
        (
            (held[q] - held[q + DESCENT_ROWS]) / span,
            (tool[q] - tool[q + DESCENT_ROWS]) / span,
        )
        for q in range(len(held) - DESCENT_ROWS)
    ]
    factor = judge["place_max_descent_factor"]
    with_tool = any(t > 0.0 for _, t in speeds) and all(
        o <= factor * max(t, 0.0) for o, t in speeds
    )
    final = ep.c[after[-1]]
    fall = ep.c[after[release]][2] - final[2]
    tail = [ep.c[i] for i in after[-REST_ROWS:]]
    reasons = []
    if not with_tool:
        reasons.append("descent_not_with_tool")
    if not fall <= judge["place_release_drop_tol_m"]:
        reasons.append("fell_on_release")
    if not max(math.dist(c, tail[-1]) for c in tail) <= judge["place_rest_radius_m"]:
        reasons.append("not_at_rest")
    if not math.dist(final, rest) <= judge["place_tol_m"]:
        reasons.append("not_back_at_start")
    return not reasons, reasons, None


def opening(episode: Episode, crossing: int, hold_rows: int) -> int | None:
    """The first row of the first run of open_ramp_frames rows whose commanded width is
    above the widest of the held window plus the deadband."""
    held = [w for w in episode.w[crossing : crossing + hold_rows + 1] if w is not None]
    if not held:
        return None
    line = max(held) + episode.judge["grip_force_open_deadband_m"]
    ramp = episode.judge["open_ramp_frames"]
    run = 0
    for i in range(crossing + 1, episode.n):
        w = episode.w[i]
        run = run + 1 if w is not None and w > line else 0
        if run >= ramp:
            return i - ramp + 1
    return None


def _in_the_jaws(
    quat: tuple[float, ...] | None, offset: tuple[float, float, float]
) -> tuple[float, float]:
    """An offset from the tool point, read in the reached tool frame as its component
    along the jaw axis (the tool's y) and its distance from that axis."""
    vx, vy, vz = offset
    if quat is not None:
        x, y, z, w = quat
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if math.isfinite(norm) and norm != 0.0:
            x, y, z, w = -x / norm, -y / norm, -z / norm, w / norm
            tx = 2.0 * (y * vz - z * vy)
            ty = 2.0 * (z * vx - x * vz)
            tz = 2.0 * (x * vy - y * vx)
            vx, vy, vz = (
                vx + w * tx + (y * tz - z * ty),
                vy + w * ty + (z * tx - x * tz),
                vz + w * tz + (x * ty - y * tx),
            )
    return vy, math.hypot(vx, vz)


def put_down(episode: Episode) -> dict[str, Any]:
    """The put-down of a carrying task: carried, placed, why either is None, and the
    facts of the generator's put-down record."""
    ep, judge = episode, episode.judge
    facts: dict[str, Any] = dict.fromkeys(PLACE_FACTS)
    facts["carried_fail_reasons"], facts["fail_reasons"] = [], []
    out: dict[str, Any] = {"carried": None, "placed": None, "why": None, "facts": facts}
    hold = ep.h_t
    held, crossing = held_from_crossing(ep, judge["lift_success_m"], hold)
    if held is not True or crossing is None:
        out["why"] = (
            "internal_lift_not_met"
            if crossing is None
            else "internal_hold_truncated"
            if held is None
            else "internal_hold_not_met"
        )
        return out
    first = ep.usable[0]
    rest = ep.c[first]
    rest_bottom = None if ep.bmin[first] is None else ep.bmin[first][2]
    target = ep.target
    o = opening(ep, crossing, hold)
    ramp = judge["open_ramp_frames"]
    facts["opening_frame"] = o
    facts["fully_open_frame"] = None if o is None else o + ramp
    tx, ty = target["x_m"], target["y_m"]
    facts["carry_realised_m"] = math.hypot(tx - rest[0], ty - rest[1])
    if "centre_x_m" in target and "centre_y_m" in target:
        gx = 2.0 * target["centre_x_m"] - rest[0]
        gy = 2.0 * target["centre_y_m"] - rest[1]
        facts["prior_error_m"] = math.hypot(tx - gx, ty - gy)

    def clearance(i: int) -> float | None:
        bottom = ep.bmin[i] if 0 <= i < ep.n else None
        return (
            None if bottom is None or rest_bottom is None else bottom[2] - rest_bottom
        )

    def c(i: int) -> tuple[float, ...] | None:
        return ep.c[i] if 0 <= i < ep.n else None

    def p(i: int) -> tuple[float, ...] | None:
        return ep.p[i] if 0 <= i < ep.n else None

    dropped = ep.stop == "dropped"
    window = [
        i for i in range(crossing, ep.n if o is None else o) if ep.c[i] is not None
    ]
    judgeable = dropped or (
        ep.complete("carry") and (o is not None or ep.complete("place"))
    )
    offsets = [
        (
            i,
            _in_the_jaws(
                ep.q[i], (ep.c[i][0] - t[0], ep.c[i][1] - t[1], ep.c[i][2] - t[2])
            ),
        )
        for i in window
        if (t := ep.p[i]) is not None
    ]
    if not judgeable or not offsets:
        out["why"] = "carry_not_judgeable"
        return out
    reasons = []
    if dropped and o is None:
        reasons.append("dropped_in_carry")
    ref = offsets[0][1]
    facts["carry_slip_max_m"] = max(math.dist(v, ref) for _, v in offsets)
    if facts["carry_slip_max_m"] > judge["carry_slip_tol_m"]:
        reasons.append("slipped_in_jaws")
    anchor = ep.c[o] if o is not None else ep.c[window[-1]]
    if anchor is not None:
        low = [
            i
            for i in window
            if (h := clearance(i)) is not None and h < judge["place_release_drop_tol_m"]
        ]
        facts["dragged_max_m"] = max(
            (math.hypot(ep.c[i][0] - anchor[0], ep.c[i][1] - anchor[1]) for i in low),
            default=0.0,
        )
        if facts["dragged_max_m"] > judge["place_tol_m"]:
            reasons.append("dragged")
    facts["carried_fail_reasons"] = reasons
    carried = not reasons
    facts["carried"] = out["carried"] = carried

    def tcp_over_bottom(i: int) -> float | None:
        return (
            None
            if ep.p[i] is None or ep.bmin[i] is None
            else ep.p[i][2] - ep.bmin[i][2]
        )

    close = [i for i, phase in enumerate(ep.phase) if phase == "close"]
    carry = [i for i, phase in enumerate(ep.phase) if phase == "carry"]
    if close and carry:
        a, b = tcp_over_bottom(close[-1]), tcp_over_bottom(carry[-1])
        if a is not None and b is not None:
            facts["lowest_point_drop_m"] = b - a
    if o is not None and close and ep.p[o] is not None and ep.p[close[-1]] is not None:
        facts["release_tcp_minus_close_tcp_z_m"] = ep.p[o][2] - ep.p[close[-1]][2]
    if o is not None and ep.p[o] is not None:
        facts["release_tcp_z_m"] = ep.p[o][2]
    if not carried:
        return out
    if not (ep.complete("place") and ep.complete("retreat")):
        out["why"] = "put_down_incomplete"
        return out
    span_end = None if o is None else min(o + judge["opening_to_end_frames"], ep.n)
    last = ep.n if span_end is None else span_end
    rest_rows = (
        range(0)
        if span_end is None
        else range(max(0, span_end - judge["rest_window_frames"]), last)
    )
    final_index = next(
        (
            i
            for i in range(last - 1, -1, -1)
            if ep.c[i] is not None and ep.bmin[i] is not None and ep.bmax[i] is not None
        ),
        None,
    )
    if final_index is None:
        out["why"] = "no_final_frame"
        return out
    final = ep.c[final_index]
    half = target["half_extent_m"]
    facts["final_centroid_xy"] = [final[0], final[1]]
    facts["place_error_m"] = math.hypot(final[0] - tx, final[1] - ty)
    facts["on_target_centroid_20mm"] = bool(
        facts["place_error_m"] <= judge["place_tol_m"]
    )
    if o is None:
        facts["fail_reasons"] = ["never_opened"]
        facts["placed"] = out["placed"] = False
        return out
    failed = []
    lo, hi = ep.bmin[final_index], ep.bmax[final_index]
    on_mat = (
        lo[0] >= tx - half
        and hi[0] <= tx + half
        and lo[1] >= ty - half
        and hi[1] <= ty + half
    )
    bound = judge.get("place_error_tol_m")
    if not (on_mat and (bound is None or facts["place_error_m"] <= bound)):
        failed.append("off_target")
    fully = o + ramp
    excess = None
    for i in range(crossing, fully - SET_DOWN_ROWS + 1):
        j = i + SET_DOWN_ROWS
        ci, cj, pi, pj = c(i), c(j), p(i), p(j)
        if ci is None or cj is None or pi is None or pj is None:
            continue
        e = (ci[2] - cj[2]) - judge["place_max_descent_factor"] * max(
            pi[2] - pj[2], 0.0
        )
        excess = e if excess is None else max(excess, e)
    facts["came_down_excess_max_m"] = excess
    if excess is not None and excess > judge["place_release_drop_tol_m"]:
        failed.append("came_down_fast")
    touch = next(
        (
            i
            for i in range(crossing + 1, ep.n)
            if (h := clearance(i)) is not None
            and h <= TOUCHDOWN_BAND_M
            and ep.c[i] is not None
        ),
        None,
    )
    facts["first_touch_frame"] = touch
    if touch is not None and c(touch - SET_DOWN_ROWS) is not None:
        speed = (c(touch - SET_DOWN_ROWS)[2] - ep.c[touch][2]) / (
            SET_DOWN_ROWS / ep.fps
        )
        facts["set_down_speed_m_s"] = speed
        if speed > judge["set_down_speed_max_m_s"]:
            failed.append("set_down_too_fast")
    at_open = ep.c[o]
    if at_open is None:
        out["why"] = "no_centroid_at_opening"
        return out
    facts["fall_after_open_m"] = at_open[2] - final[2]
    if facts["fall_after_open_m"] > judge["place_release_drop_tol_m"]:
        failed.append("fell_on_release")
    facts["push_after_release_m"] = math.hypot(
        final[0] - at_open[0], final[1] - at_open[1]
    )
    if facts["push_after_release_m"] > PUSH_FACTOR * judge["place_tol_m"]:
        failed.append("pushed_after_release")
    if (at_full := c(fully)) is not None:
        facts["retreat_drag_m"] = math.hypot(
            final[0] - at_full[0], final[1] - at_full[1]
        )
    after = range(fully + 1, last)
    facts["gripper_contacts_after_open"] = sum(1 for i in after if ep.k[i] > 0)
    if len(after) == 0 or facts["gripper_contacts_after_open"] > 0:
        failed.append("still_touching")
    rested = [ep.c[i] for i in rest_rows if ep.c[i] is not None]
    if not rested:
        out["why"] = "empty_rest_window"
        return out
    facts["rest_wander_m"] = max(math.dist(x, rested[-1]) for x in rested)
    if facts["rest_wander_m"] > judge["place_rest_radius_m"]:
        failed.append("not_at_rest")
    peaks = [g for g in ep.g[crossing : fully + 1] if g is not None]
    facts["set_down_bench_force_peak_n"] = max(peaks) if peaks else None
    facts["fail_reasons"] = failed
    facts["placed"] = out["placed"] = not failed
    return out


def placed_task(episode: Episode) -> dict[str, Any] | None:
    """The task's own placement verdict, or None when the task places nothing."""
    rule = episode.task.placed_rule
    if rule is None:
        return None
    if rule == "put-back":
        verdict, reasons, why = put_back(episode)
        return {
            "rule": rule,
            "verdict": verdict,
            "reasons": reasons,
            "why": why,
            "carried": None,
            "placed": None,
            "facts": None,
        }
    down = put_down(episode)
    facts = down["facts"]
    if down["carried"] is False:
        verdict, reasons = False, facts["carried_fail_reasons"]
    elif down["carried"] is True:
        verdict, reasons = down["placed"], facts["fail_reasons"]
    else:
        verdict, reasons = None, []
    return {
        "rule": rule,
        "verdict": verdict,
        "reasons": list(reasons),
        "why": down["why"],
        "carried": down["carried"],
        "placed": down["placed"],
        "facts": facts,
    }


# --- one episode's outcome ------------------------------------------------------------------


def decidable(episode: Episode) -> dict[str, bool] | None:
    """Whether each criterion's held window, opened at the end of the lift, ends inside
    the planned recording; None for a task that plans no hold."""
    phases = episode.phases
    if "hold" not in phases:
        return None
    before = sum(episode.count(w) for w in range(phases.index("hold")))
    total = sum(episode.count(w) for w in range(len(phases)))
    return {"C": before + episode.h_c - 1 <= total, "B": before + episode.n_b <= total}


def failure_class(
    stages: Sequence[str],
    verdicts: Mapping[str, bool | None],
    stop: str | None,
    is_decidable: bool | None,
) -> str:
    lifted = verdicts["lifted"]
    if "held" in stages and is_decidable is False:
        if lifted is False:
            return "not_lifted"
        return "undecided" if lifted is None else "not_decidable"
    if all(verdicts[s] is True for s in stages):
        return "success"
    if lifted is False:
        return "not_lifted"
    held, last = verdicts.get("held"), verdicts[stages[-1]]
    if lifted is True and (held is False or (stop == "dropped" and last is not True)):
        return "dropped"
    if "placed" in stages and held is True and verdicts["placed"] is False:
        return "not_placed"
    return "undecided"


def _reasons(
    kind: str,
    criterion: str,
    verdicts: Mapping[str, bool | None],
    stop: str | None,
    agreement_reason: str | None,
    placed: Mapping[str, Any] | None,
) -> list[str]:
    on_c = criterion == "C"
    if kind == "not_lifted":
        return [agreement_reason if on_c else "no_crossing"]
    if kind == "dropped":
        if stop == "dropped":
            return ["stopped_dropped"]
        return [agreement_reason if on_c else "below_threshold_in_window"]
    if kind == "not_placed":
        return list(dict.fromkeys(placed["reasons"]))
    if kind == "not_decidable":
        return ["window_longer_than_recording"]
    if kind == "undecided":
        if verdicts["lifted"] is None:
            return [agreement_reason if on_c else "no_usable_rows"]
        if verdicts.get("held") is None:
            return [agreement_reason if on_c else "window_truncated"]
        return [placed["why"]]
    return []


def judge_episode(episode: Episode) -> dict[str, Any]:
    """Every verdict, class and fact the scorer records for one episode."""
    ep, task = episode, episode.task
    stages = task.stages
    record_c = agreement(ep)
    up = record_c["first_up_frame"] is not None
    held_b, crossing = held_from_crossing(ep, B_RISE_M, ep.n_b)
    raw = {
        "C": (True if up else record_c["verdict"], record_c["verdict"] if up else None),
        "B": (
            None if not ep.usable else crossing is not None,
            held_b if crossing is not None else None,
        ),
    }
    placed = placed_task(ep)
    windows = decidable(ep)
    out: dict[str, Any] = {
        "episode_uid": ep.uid,
        "object_type": ep.object_type,
        "task": ep.name,
        "label": ep.label,
        "n_steps": ep.n,
        "stop_reason": ep.stop,
        "floor_arm": ep.floor_arm,
        "keep_labels": ep.keep_labels,
        "keep_requires_lift": ep.keep_requires_lift,
        "timing_scale": ep.timing_scale,
        "n_retries": ep.n_retries,
        "stages": list(stages),
        "decidable": windows,
    }
    for x in CRITERIA:
        lifted, held = raw[x]
        verdicts: dict[str, bool | None] = {"lifted": lifted}
        if "held" in stages:
            verdicts["held"] = held
        if placed is not None:
            verdicts["placed"] = placed["verdict"] if held is True else None
        kind = failure_class(
            stages, verdicts, ep.stop, None if windows is None else windows[x]
        )
        cell: dict[str, Any] = dict(verdicts)
        cell["class"] = kind
        cell["reasons"] = _reasons(
            kind, x, verdicts, ep.stop, record_c["reason"], placed
        )
        if x == "C":
            cell["agreement"] = record_c
        else:
            cell["crossing"] = crossing
            cell["window_end"] = None if crossing is None else crossing + ep.n_b
        out[x] = cell
    out["placed_task"] = placed
    out["check"] = check_episode(
        ep, record_c, raw["B"][0], raw["B"][1], crossing, placed
    )
    return out


def score_episode(
    columns: Mapping[str, Any],
    meta: Mapping[str, Any],
    listing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The scorer's record of one episode (raises Refused); see read_episode."""
    return judge_episode(read_episode(columns, meta, listing))


# --- the check against the verdicts the episode carries -------------------------------------


def _same(a: object, b: object) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a is b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _record_differences(
    name: str,
    stored: Mapping[str, Any],
    scored: Mapping[str, Any],
    skip: Sequence[str] = (),
) -> list[str]:
    out = [f"{name}.{k}: missing" for k in scored if k not in stored and k not in skip]
    for k, v in stored.items():
        if k in skip:
            continue
        if k not in scored:
            out.append(f"{name}.{k}: not a key of the record")
        elif not _same(v, scored[k]):
            out.append(f"{name}.{k}: stored {v!r}, scored {scored[k]!r}")
    return out


def _explain(
    episode: Episode,
    stage: str,
    stored: bool | None,
    scored: bool | None,
    crossing: int | None,
) -> str | None:
    """The one reason a stored stage verdict may differ from the scorer's, or None."""
    ep, task = episode, episode.task
    if stage in ("lifted", "held"):
        if ep.judge["lift_success_m"] != B_RISE_M:
            return "other_lift_threshold"
        if stage == "held" and task.hold_s != B_HOLD_S:
            return "other_hold_length"
        if ep.retried:
            return "retry_whole_recording"
    if stage == "lifted":
        if stored is None:
            return "lift_not_reached"
        if stored is False and scored is True:
            lift = [i for i in ep.final_rows() if ep.phase[i] == "lift"]
            if lift and crossing > lift[-1]:
                return "crossing_after_lift_phase"
            return "below_threshold_at_lift_end"
        return None
    if stage == "held":
        if stored is None:
            return "held_not_judged_by_generator"
        return "window_vs_hold_phase" if task.criterion_id is None else None
    if task.carries:
        return "generator_gate" if stored is None else None
    return "put_back_phase_rule"


def check_episode(
    episode: Episode,
    agreement_record: Mapping[str, Any],
    lifted_b: bool | None,
    held_b: bool | None,
    crossing: int | None,
    placed: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compare the scorer's verdicts with the ones the episode's task block carries.

    exact holds the exact checks that apply, explained the stored stage verdicts that
    differ for a known reason, and failures what an exact check or an unexplained
    difference found.
    """
    ep, task = episode, episode.task
    stages, record, place = (ep.stored[k] for k in ("stages", "agreement", "place"))
    lift = ep.judge["lift_success_m"]
    out: dict[str, Any] = {
        "exact": {},
        "matched": [],
        "explained": {},
        "unavailable": [],
        "failures": [],
    }
    if stages is None:
        out["unavailable"].append("stages")
    if task.criterion_id is not None and record is None:
        out["unavailable"].append("agreement")
    if task.carries and place is None:
        out["unavailable"].append("place")
    stages = stages or {}

    def exact(check: str, problems: list[str]) -> None:
        out["exact"][check] = not problems
        out["failures"] += [f"{check} {p}" for p in problems]

    if record is not None or "agreement" in stages:
        problems = []
        if record is not None:
            problems += _record_differences("agreement", record, agreement_record)
        if "agreement" in stages and not _same(
            stages["agreement"], agreement_record["verdict"]
        ):
            problems.append(
                f"stages.agreement: stored {stages['agreement']!r}, "
                f"scored {agreement_record['verdict']!r}"
            )
        exact("E1", problems)
    if stages.get("lifted") is True and lift >= B_RISE_M:
        exact(
            "E2",
            [] if lifted_b is True else [f"stages.lifted: stored True, B {lifted_b!r}"],
        )
    e3 = ep.name == "search_lift_hold_carry_place" and stages.get("lifted") is True
    e3 = e3 and lift == B_RISE_M
    if e3:
        stored = stages.get("held")
        exact(
            "E3",
            []
            if _same(stored, held_b)
            else [f"stages.held: stored {stored!r}, B {held_b!r}"],
        )
    if place is not None:
        scored = (
            placed["facts"]
            if placed is not None and placed["facts"] is not None
            else {}
        )
        exact(
            "E4",
            _record_differences(
                "place", place, scored, (*NOT_COMPUTED, "carried", "placed")
            ),
        )
    e5 = task.carries and stages.get("held") is True
    if e5:
        problems = [
            f"stages.{k}: stored {stages.get(k)!r}, scored {placed[k]!r}"
            for k in ("carried", "placed")
            if not _same(stages.get(k), placed[k])
        ]
        exact("E5", problems)

    compared = []
    if "lifted" in stages and not (stages["lifted"] is True and lift >= B_RISE_M):
        compared.append(("lifted", stages["lifted"], lifted_b))
    if "held" in stages and "held" in task.stages and not e3:
        compared.append(("held", stages["held"], held_b))
    if task.carries and not e5:
        compared += [
            (k, stages[k], placed[k]) for k in ("carried", "placed") if k in stages
        ]
    if task.placed_rule == "put-back" and "placed" in stages:
        compared.append(("placed", stages["placed"], placed["verdict"]))
    for stage, stored, scored in compared:
        if stored is scored:
            out["matched"].append(stage)
            continue
        code = _explain(ep, stage, stored, scored, crossing)
        if code is None:
            out["failures"].append(
                f"stages.{stage}: stored {stored!r}, scored {scored!r}, "
                "a difference no rule explains"
            )
        else:
            out["explained"][stage] = code
    return out
