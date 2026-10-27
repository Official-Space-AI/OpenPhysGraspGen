# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
"""PG-T1: the portable test vectors of the scoring rules and their edge cases.

Every episode is synthetic. The vectors and the dataset cases are written with
GraspDataRecorder, read back through the dataset reader and scored as the command scores
them; the edge cases are scored from columns in memory by the same function.
"""

import datetime
import json
import math
import subprocess
import sys
from collections import Counter

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("yaml")

import pyarrow.parquet as pq
import yaml

from openphysgraspgen.benchmark import (
    DatasetUnreadable,
    Refused,
    report_text,
    score_dataset,
    score_episode,
    task_block_problems,
    wilson_interval,
)
from openphysgraspgen.benchmark.pg_t1 import COLUMNS, TASKS
from openphysgraspgen.dataset import DatasetSpec, GraspDataRecorder
from openphysgraspgen.pipeline import main

S4 = [30, 90, 30, 30, 60, 30, 240]
S2 = [30, 90, 30, 30, 60, 30, 120]
PK = [30, 30, 60, 30]
LH = [30, 30, 60, 30, 240]
LHP = [30, 30, 30, 30, 240, 30, 60]
RETRY = [30, 30, 60, 30, 240, 30, 90, 30, 30, 60, 30, 240]
CP = [30, 90, 30, 30, 60, 30, 30, 240, 60, 45, 45, 30]
CARRY = "search_lift_hold_carry_place"
CARRY_10CM = "search_lift_hold_carry_place_10cm"
SCHEDULES = {
    "pick": PK,
    "lift_hold": LH,
    "lift_hold_place": LHP,
    "search_lift_hold": S4,
    "lift_hold_retry": RETRY,
    "search_lift_hold_2s": S2,
    CARRY: CP,
    CARRY_10CM: CP,
}
TARGET = {
    "x_m": 0.5,
    "y_m": 0.15,
    "half_extent_m": 0.077,
    "centre_x_m": 0.5,
    "centre_y_m": 0.0,
}
OBJECT = ("object.centroid", "object.bbox_min", "object.bbox_max")
FACTS = {
    "material": {"E_pa": 1.0e5},
    "placement": {"x_m": 0.5},
    "lighting": {"lux": 500.0},
    "camera": {"fov_deg": 60.0},
}


def smooth(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


def clamp(u):
    return min(1.0, max(0.0, u))


def row(
    waypoint, phase, centroid, *, bottom=None, top=None, tool=(0.5, 0.0, 0.3), **kw
):
    """One trajectory row; the bounding box is the centroid +/- 0.04 unless given."""
    low = tuple(c - 0.04 for c in centroid) if bottom is None else bottom
    high = tuple(c + 0.04 for c in centroid) if top is None else top
    out = {
        "finite": True,
        "robot.waypoint": waypoint,
        "robot.phase": phase,
        "object.centroid": tuple(centroid),
        "object.bbox_min": low,
        "object.bbox_max": high,
        "robot.ee_pose": (*tool, 0.0, 0.0, 0.0, 1.0),
        "robot.gripper_width_cmd_m": 0.06,
        "object.n_robot_contacts": 0,
        "force.ground_force_mag": 0.0,
    }
    out.update(kw)
    return out


def waypoints(schedule):
    return [w for w, n in enumerate(schedule) for _ in range(n)]


def blow_up(rows, index=-1):
    rows[index] = dict(rows[index], finite=False)
    return rows


def ramp(
    task,
    schedule,
    top,
    *,
    bottom_frac=1.0,
    dips=(),
    cut=None,
    blowup=False,
    profile="ramp",
):
    """The object rests, rises by top over the lift waypoint and stays there."""
    phases = TASKS[task].phases
    lift = phases.index("lift")
    start, m = sum(schedule[:lift]), schedule[lift]
    rows = []
    for i, w in enumerate(waypoints(schedule)):
        if i < start:
            r = 0.0
        elif i < start + m:
            u = (i - start + 1) / m
            r = top * (smooth(u) if profile == "smooth" else u)
        else:
            r = top
        for a, length, depth in dips:
            if a <= i < a + length:
                r = max(0.0, r - depth)
        rows.append(
            row(
                w,
                phases[w],
                (0.5, 0.0, 0.05 + r),
                bottom=(0.46, -0.04, 0.01 + r * bottom_frac),
                top=(0.54, 0.04, 0.09 + r),
            )
        )
    rows = rows[:cut]
    return blow_up(rows) if blowup else rows


# The put-down: a fruit held under the tool through a smoothstep lift, carried 0.30 m at
# hold height, lowered to 3 mm over the bench and let go of when the jaws open at row 600.
REST = (0.5, -0.15, 0.04)
Z_CLOSE = 0.066
OFFSET = REST[2] - Z_CLOSE
GRIP, OPEN = 0.076, 0.14
STARTS = [sum(CP[:k]) for k in range(len(CP))]


def tool(i, clearance=0.003):
    x, y = REST[0], REST[1]
    if i < STARTS[5]:
        return x, y, Z_CLOSE + 0.17 * (1.0 if i < STARTS[3] else 0.0)
    if i < STARTS[6]:
        return x, y, Z_CLOSE + 0.17 * smooth((i - STARTS[5] + 1) / CP[5])
    if i < STARTS[7]:
        return x, y, Z_CLOSE + 0.17
    if i < STARTS[8]:
        return x, y + 0.30 * smooth((i - STARTS[7] + 1) / CP[7]), Z_CLOSE + 0.17
    if i < STARTS[9]:
        u = smooth((i - STARTS[8] + 1) / CP[8])
        return x, y + 0.30, Z_CLOSE + 0.17 - (0.17 - clearance) * u
    if i < STARTS[10]:
        return x, y + 0.30, Z_CLOSE + clearance
    if i < STARTS[11]:
        return (
            x,
            y + 0.30,
            Z_CLOSE + clearance + 0.17 * smooth((i - STARTS[10] + 1) / CP[10]),
        )
    return x, y + 0.30, Z_CLOSE + clearance + 0.17


def put_down(
    *,
    task=CARRY,
    clearance=0.003,
    fall=None,
    slip=None,
    open_at=600,
    shove=None,
    touch_after_open=None,
    wander=None,
    truncate=None,
    path=None,
):
    """path(i) moves the tool off its plan on the rows where it is not None."""
    phases = TASKS[task].phases
    fall = fall or (lambda k: min(clearance, clearance * k / 3.0))
    rows, settled = [], None
    n = sum(CP)
    for i, w in enumerate(waypoints(CP)):
        t = (path(i) if path else None) or tool(i, clearance)
        if i < STARTS[4]:
            centroid, width = REST, OPEN
        elif open_at is None or i < open_at:
            centroid = (
                t[0],
                t[1],
                max(REST[2], t[2] + OFFSET - (slip(i) if slip else 0.0)),
            )
            width = GRIP
        else:
            if settled is None:
                settled = rows[-1]["object.centroid"]
            centroid = (
                settled[0],
                settled[1],
                max(REST[2], settled[2] - fall(i - open_at)),
            )
            width = OPEN
        for move in (shove, wander):
            if move is not None:
                dx, dy = move(i)
                centroid = (centroid[0] + dx, centroid[1] + dy, centroid[2])
        contacts = 2 if STARTS[4] <= i <= (open_at or n) + 30 else 0
        if i == touch_after_open:
            contacts = 1
        rows.append(
            row(
                w,
                phases[w],
                centroid,
                tool=t,
                **{
                    "robot.ee_pose": (*t, 1.0, 0.0, 0.0, 0.0),
                    "robot.gripper_width_cmd_m": width,
                    "object.n_robot_contacts": contacts,
                },
            )
        )
    return rows[:truncate]


def put_back(
    *, fall=0.0, drift=0.0, never_open=False, drop=False, schedule=LHP, top=0.15
):
    """lift_hold_place: lift top, hold, lower with the tool and let go where it rested;
    with drop the object falls out of the jaws as the tool starts down."""
    phases = TASKS["lift_hold_place"].phases
    starts = [sum(schedule[:k]) for k in range(len(schedule))]
    rows = []
    for i, w in enumerate(waypoints(schedule)):
        if w == 0:
            z_tool, r = 0.22, 0.0
        elif w == 1:
            z_tool, r = 0.22 - 0.17 * (i - starts[1] + 1) / schedule[1], 0.0
        elif w == 2:
            z_tool, r = 0.05, 0.0
        elif w == 3:
            u = (i - starts[3] + 1) / schedule[3]
            z_tool, r = 0.05 + 0.17 * u, top * u
        elif w == 4:
            z_tool, r = 0.22, top
        elif w == 5:
            k = i - starts[5] + 1
            u = k / schedule[5]
            z_tool, r = 0.22 - 0.17 * u, top * (1.0 - u)
            if drop:
                r = max(0.0, top - 0.5 * 9.81 * (k / 60) ** 2)
        else:
            z_tool = 0.05
            r = max(0.0, fall * (1.0 - min(1.0, (i - starts[6] + 1) / 3.0)))
        width = 0.05 if 2 <= w <= 5 or (w == 6 and never_open) else 0.12
        dx = drift if w == 6 else 0.0
        rows.append(
            row(
                w,
                phases[w],
                (0.5 + dx, 0.0, 0.05 + r),
                tool=(0.5, 0.0, z_tool),
                **{"robot.gripper_width_cmd_m": width},
            )
        )
    return rows


def retry(*, first_drop=True, second_top=0.2):
    """lift_hold_retry: the first attempt drops the object, the second lifts second_top."""
    phases = TASKS["lift_hold_retry"].phases
    starts = [sum(RETRY[:k]) for k in range(len(RETRY))]
    rows = []
    for i, w in enumerate(waypoints(RETRY if first_drop else RETRY[:5])):
        if w in (3, 10):
            r = (0.2 if w == 3 else second_top) * (i - starts[w] + 1) / 30
        elif w == 4:
            r = 0.2 if not first_drop or i < starts[4] + 60 else 0.0
        else:
            r = second_top if w == 11 else 0.0
        rows.append(row(w, phases[w], (0.5, 0.0, 0.05 + r)))
    return rows


def block(task, schedule, *, stop=None, judge=None, **extra):
    """A valid task block of the PG-T1 layout."""
    spec = TASKS[task]
    numbers = {"lift_success_m": 0.12, "approach_height_m": 0.17}
    if "place" in spec.phases:
        numbers.update(
            place_tol_m=0.02,
            place_rest_radius_m=0.002,
            place_max_descent_factor=2.0,
            place_release_drop_tol_m=0.01,
        )
    if spec.carry_m is not None:
        numbers.update(
            carry_slip_tol_m=0.02,
            set_down_speed_max_m_s=0.30,
            grip_force_open_deadband_m=0.003,
            open_ramp_frames=30,
            opening_to_end_frames=120,
            rest_window_frames=30,
            place_error_tol_m=0.005 if spec.centroid_bound else None,
        )
    numbers.update(judge or {})
    out = {
        "layout": "pg-t1-task-1",
        "source": "generation",
        "name": task,
        "phases": list(spec.phases),
        "frames_per_waypoint": list(schedule),
        "stop_reason": stop,
        "criterion_id": spec.criterion_id,
        "floor_arm": None,
        "judge": numbers,
        "place_target": dict(TARGET) if spec.carry_m is not None else None,
        "keep_labels": ["success", "fracture"],
        "keep_requires_lift": True,
        "stages": None,
        "agreement": None,
        "place": None,
    }
    out.update(extra)
    return out


def episode(uid, rows, task_block, object_type="apple", label="success", **kw):
    return {
        "uid": uid,
        "rows": rows,
        "task": task_block,
        "object_type": object_type,
        "label": label,
        **kw,
    }


def columns(rows):
    """The trajectory columns of these rows, as read_trajectory returns them."""
    finite = np.array([r["finite"] for r in rows])
    out = {"sim_step": np.arange(1, len(rows) + 1, dtype=np.int64), "finite": finite}
    for name in COLUMNS[2:]:
        values = [r[name] for r in rows]
        if name in ("robot.phase",):
            out[name] = np.array(values, dtype=object)
        elif name in ("robot.waypoint", "object.n_robot_contacts"):
            out[name] = np.array(values, dtype=np.int32)
        else:
            out[name] = np.array(values, dtype=np.float64)
    for name in OBJECT:
        out[name][~finite] = np.nan
    return out


def metadata(rows, task_block, uid="ep-0", fps=60, **kw):
    out = {
        "format": "physgraspgen-public",
        "format_version": "2.0",
        "episode_uid": uid,
        "object_type": "apple",
        "recorded_fps": fps,
        "n_steps": len(rows),
        "source_n_frames": len(rows),
        "label": "success",
        "task": task_block,
    }
    out.update(kw)
    return out


def write(root, episodes, recorded_fps=60):
    """Write the episodes with GraspDataRecorder and finalize the dataset."""
    stride = 2 if recorded_fps % 2 == 0 else 1
    spec = DatasetSpec(
        name="pg-t1-test",
        version="0.0.1",
        fps=recorded_fps // stride,
        recorded_fps=recorded_fps,
        channels=(),
    )
    recorder = GraspDataRecorder(root, spec)
    for ep in episodes:
        rows = ep["rows"]
        with recorder.episode(
            episode_uid=ep["uid"],
            object_type=ep["object_type"],
            object_asset=f"{ep['object_type']}_01",
            n_tets=4,
            label=ep["label"],
            facts=FACTS,
            task=ep["task"],
            source_n_frames=ep.get("source_n_frames", len(rows)),
        ) as writer:
            for i, r in enumerate(rows):
                state = {
                    k: v
                    for k, v in r.items()
                    if k not in ("finite", "force.ground_force_mag")
                }
                state.update(
                    {
                        "robot.ee_pose_cmd": r["robot.ee_pose"],
                        "robot.arm_q_setpoint": np.zeros(6),
                        "robot.arm_q": np.zeros(6),
                        "robot.gripper_width_setpoint_m": r[
                            "robot.gripper_width_cmd_m"
                        ],
                        "robot.gripper_width_m": r["robot.gripper_width_cmd_m"],
                        "robot.grip_contact_width_m": math.nan,
                    }
                )
                if not r["finite"]:
                    state.update(dict.fromkeys(OBJECT))
                writer.add_step(
                    {
                        **state,
                        "sim_step": i + 1,
                        "finite": r["finite"],
                        "force.gripper_force_mag": 0.0,
                        "force.ground_force_mag": r["force.ground_force_mag"],
                    }
                )
                if i + 1 == stride:
                    writer.add_frame(
                        {
                            **state,
                            "sim_step": stride,
                            "source_frame_index": stride - 1,
                            "object.n_soft_contacts": 0,
                            "object.n_rigid_contacts": 0,
                        }
                    )
    recorder.finalize()
    return root


def scored(root):
    report, episodes = score_dataset(root)
    return report, {e["episode_uid"]: e for e in episodes}


# --- the vectors -------------------------------------------------------------------------


def drop(start):
    return lambda i: 0.0 if i < start else 0.5 * 9.81 * ((i - start) / 60) ** 2


VECTORS = {
    "V01": (ramp("search_lift_hold", S4, 0.15), block("search_lift_hold", S4)),
    "V02": (ramp("search_lift_hold_2s", S2, 0.15), block("search_lift_hold_2s", S2)),
    "V03": (ramp("search_lift_hold_2s", S2, 0.07), block("search_lift_hold_2s", S2)),
    "V04": (ramp("search_lift_hold_2s", S2, 0.03), block("search_lift_hold_2s", S2)),
    "V05": (ramp("search_lift_hold", S4, 0.12), block("search_lift_hold", S4)),
    "V06": (
        ramp("search_lift_hold", S4, 0.15, dips=[(400, 5, 0.05)]),
        block("search_lift_hold", S4),
    ),
    "V07": (
        ramp("search_lift_hold", S4, 0.15, dips=[(505, 5, 0.05)]),
        block("search_lift_hold", S4),
    ),
    "V08": (
        ramp("search_lift_hold", S4, 0.15, dips=[(300, 5, 0.12)]),
        block("search_lift_hold", S4),
    ),
    "V09": (ramp("search_lift_hold", S4, 0.15, cut=400), block("search_lift_hold", S4)),
    "V10": (
        ramp("search_lift_hold", S4, 0.15, cut=400),
        block("search_lift_hold", S4, stop="dropped"),
    ),
    "V11": (
        ramp("search_lift_hold", S4, 0.15, cut=320),
        block("search_lift_hold", S4, stop="dropped"),
    ),
    "V12": (
        ramp("search_lift_hold", S4, 0.15, blowup=True),
        block("search_lift_hold", S4),
    ),
    # A blow-up inside the lift waypoint is recorded with the never-lifted stop.
    "V13": (
        ramp("search_lift_hold", S4, 0.15, cut=255, blowup=True),
        block("search_lift_hold", S4, stop="never_lifted"),
    ),
    "V14": (
        ramp("search_lift_hold", S4, 0.15, bottom_frac=0.0),
        block("search_lift_hold", S4),
    ),
    "V15": (
        ramp("search_lift_hold_2s", S2, 0.03, cut=270),
        block("search_lift_hold_2s", S2, stop="never_lifted"),
    ),
    "V16": (
        ramp("search_lift_hold", S4, 0.08, cut=270),
        block("search_lift_hold", S4, stop="never_lifted"),
    ),
    "V17": (ramp("lift_hold", LH, 0.15), block("lift_hold", LH)),
    "V18": (ramp("lift_hold", LH, 0.15, profile="smooth"), block("lift_hold", LH)),
    "V19": (put_down(), block(CARRY, CP)),
    "V20": (put_down(slip=lambda i: 0.030 * clamp((i - 300) / 120)), block(CARRY, CP)),
    "V21": (
        put_down(open_at=None, slip=drop(520), truncate=536),
        block(CARRY, CP, stop="dropped"),
    ),
    "V22": (
        put_down(open_at=None, slip=drop(380), truncate=400),
        block(CARRY, CP, stop="dropped"),
    ),
    "V23": (put_down(), block(CARRY, CP, place_target=dict(TARGET, x_m=0.547))),
    "V24": (
        put_down(clearance=0.015, fall=lambda k: min(0.015, 0.0025 * k)),
        block(CARRY, CP),
    ),
    "V25": (
        put_down(shove=lambda i: (0.020 * clamp((i - 650) / 20), 0.0)),
        block(CARRY, CP),
    ),
    "V26": (put_down(touch_after_open=645), block(CARRY, CP)),
    "V27": (
        put_down(wander=lambda i: (0.003, 0.0) if i in (700, 790) else (0.0, 0.0)),
        block(CARRY, CP),
    ),
    "V28": (put_down(open_at=None), block(CARRY, CP)),
    "V29": (put_down(truncate=650), block(CARRY, CP)),
    "V30": (
        put_down(task=CARRY_10CM),
        block(CARRY_10CM, CP, place_target=dict(TARGET, x_m=TARGET["x_m"] + 0.008)),
    ),
    "V31": (
        put_down(task=CARRY_10CM),
        block(CARRY_10CM, CP, place_target=dict(TARGET, x_m=TARGET["x_m"] + 0.003)),
    ),
    "V32": (put_down(slip=lambda i: 0.08 * clamp((i - 380) / 30)), block(CARRY, CP)),
}

# C and B as lifted, held, placed over the planned stages (T, F, - for None); the
# agreement reason and the first row up k; the 12 cm crossing c; placed_task with its
# words; the failure classes under C and B.
EXPECTED = {
    "V01": ("TT", None, 249, "TT", 264, None, (), "success", "success"),
    "V02": ("TT", None, 249, "T-", 264, None, (), "success", "not_decidable"),
    "V03": ("TT", None, 261, "F-", None, None, (), "success", "not_lifted"),
    "V04": (
        "F-",
        "never_crossed",
        None,
        "F-",
        None,
        None,
        (),
        "not_lifted",
        "not_lifted",
    ),
    "V05": ("TT", None, 252, "F-", None, None, (), "success", "not_lifted"),
    "V06": ("TT", None, 249, "TF", 264, None, (), "success", "dropped"),
    "V07": ("TT", None, 249, "TT", 264, None, (), "success", "success"),
    "V08": ("TF", "dipped_in_window", 249, "TF", 264, None, (), "dropped", "dropped"),
    "V09": ("TT", None, 249, "T-", 264, None, (), "success", "undecided"),
    "V10": ("TT", None, 249, "TF", 264, None, (), "success", "dropped"),
    "V11": (
        "TF",
        "stopped_dropped_in_window",
        249,
        "TF",
        264,
        None,
        (),
        "dropped",
        "dropped",
    ),
    "V12": ("TT", None, 249, "TT", 264, None, (), "success", "success"),
    "V13": (
        "T-",
        "window_truncated",
        249,
        "F-",
        None,
        None,
        (),
        "undecided",
        "not_lifted",
    ),
    "V14": ("F-", "never_crossed", None, "TT", 264, None, (), "not_lifted", "success"),
    "V15": (
        "F-",
        "stopped_never_lifted",
        None,
        "F-",
        None,
        None,
        (),
        "not_lifted",
        "not_lifted",
    ),
    "V16": (
        "T-",
        "window_truncated",
        258,
        "F-",
        None,
        None,
        (),
        "undecided",
        "not_lifted",
    ),
    "V17": ("TT", None, 129, "TT", 144, None, (), "success", "success"),
    "V18": ("TT", None, 131, "TT", 141, None, (), "success", "success"),
    "V19": ("TTT", None, 250, "TTT", 259, True, (), "success", "success"),
    "V20": (
        "TTF",
        None,
        250,
        "TTF",
        259,
        False,
        ("slipped_in_jaws",),
        "not_placed",
        "not_placed",
    ),
    "V21": (
        "TTF",
        None,
        250,
        "TTF",
        259,
        False,
        ("dropped_in_carry", "slipped_in_jaws"),
        "dropped",
        "dropped",
    ),
    "V22": ("TT-", None, 250, "TF-", 259, None, (), "dropped", "dropped"),
    "V23": (
        "TTF",
        None,
        250,
        "TTF",
        259,
        False,
        ("off_target",),
        "not_placed",
        "not_placed",
    ),
    "V24": (
        "TTF",
        None,
        250,
        "TTF",
        259,
        False,
        ("fell_on_release",),
        "not_placed",
        "not_placed",
    ),
    "V25": (
        "TTF",
        None,
        250,
        "TTF",
        259,
        False,
        ("pushed_after_release",),
        "not_placed",
        "not_placed",
    ),
    "V26": (
        "TTF",
        None,
        250,
        "TTF",
        259,
        False,
        ("still_touching",),
        "not_placed",
        "not_placed",
    ),
    "V27": (
        "TTF",
        None,
        250,
        "TTF",
        259,
        False,
        ("not_at_rest",),
        "not_placed",
        "not_placed",
    ),
    "V28": (
        "TTF",
        None,
        250,
        "TTF",
        259,
        False,
        ("never_opened",),
        "not_placed",
        "not_placed",
    ),
    "V29": ("TT-", None, 250, "TT-", 259, None, (), "undecided", "undecided"),
    "V30": (
        "TTF",
        None,
        250,
        "TTF",
        259,
        False,
        ("off_target",),
        "not_placed",
        "not_placed",
    ),
    "V31": ("TTT", None, 250, "TTT", 259, True, (), "success", "success"),
    "V32": ("TT-", None, 250, "TF-", 259, None, (), "undecided", "dropped"),
}
WHY = {
    "V22": "internal_hold_not_met",
    "V29": "put_down_incomplete",
    "V32": "internal_hold_not_met",
}
LETTER = {True: "T", False: "F", None: "-"}


@pytest.fixture(scope="module")
def vectors(tmp_path_factory):
    root = tmp_path_factory.mktemp("vectors")
    write(root, [episode(uid, rows, task) for uid, (rows, task) in VECTORS.items()])
    return scored(root)


@pytest.mark.parametrize("uid", sorted(EXPECTED))
def test_each_vector_is_judged_as_the_rules_say(vectors, uid):
    c, reason, k, b, crossing, verdict, words, class_c, class_b = EXPECTED[uid]
    record = vectors[1][uid]
    stages = record["stages"]
    assert "".join(LETTER[record["C"][s]] for s in stages) == c
    assert "".join(LETTER[record["B"][s]] for s in stages) == b
    assert record["C"]["agreement"]["reason"] == reason
    assert record["C"]["agreement"]["first_up_frame"] == k
    assert record["B"]["crossing"] == crossing
    assert (record["C"]["class"], record["B"]["class"]) == (class_c, class_b)
    placed = record["placed_task"]
    if "placed" in stages:
        assert placed["verdict"] is verdict
        assert tuple(placed["reasons"]) == words
        assert placed["why"] == WHY.get(uid)
    else:
        assert placed is None


def test_the_clean_put_down_records_the_generator_facts(vectors):
    facts = vectors[1]["V19"]["placed_task"]["facts"]
    assert (facts["opening_frame"], facts["fully_open_frame"]) == (600, 630)
    assert facts["first_touch_frame"] == 603
    assert facts["carry_realised_m"] == pytest.approx(0.30, abs=1e-9)
    assert facts["prior_error_m"] == pytest.approx(0.0, abs=1e-9)
    assert facts["fall_after_open_m"] == pytest.approx(0.003, abs=1e-9)
    assert facts["came_down_excess_max_m"] == pytest.approx(0.003, abs=1e-9)
    assert facts["place_error_m"] == pytest.approx(0.0, abs=1e-9)
    assert facts["set_down_speed_m_s"] == pytest.approx(0.06, abs=1e-9)
    assert facts["lowest_point_drop_m"] == pytest.approx(0.0, abs=1e-9)
    assert facts["on_target_centroid_20mm"] is True
    assert facts["gripper_contacts_after_open"] == 0
    assert facts["set_down_bench_force_peak_n"] == 0.0
    assert (facts["carried"], facts["placed"]) == (True, True)
    assert facts["carried_fail_reasons"] == facts["fail_reasons"] == []


def test_the_reason_words_file_each_episode_under_its_class(vectors):
    records = vectors[1]
    assert records["V08"]["C"]["reasons"] == ["dipped_in_window"]
    assert records["V08"]["B"]["reasons"] == ["below_threshold_in_window"]
    assert records["V21"]["C"]["reasons"] == ["stopped_dropped"]
    assert records["V04"]["B"]["reasons"] == ["no_crossing"]
    assert records["V02"]["B"]["reasons"] == ["window_longer_than_recording"]
    assert records["V09"]["B"]["reasons"] == ["window_truncated"]
    assert records["V13"]["C"]["reasons"] == ["window_truncated"]
    assert records["V32"]["C"]["reasons"] == ["internal_hold_not_met"]
    assert records["V20"]["C"]["reasons"] == ["slipped_in_jaws"]


def rate(stage, key):
    lo, hi = stage[f"{key}_wilson95"]
    return round(stage[key], 4), round(lo, 4), round(hi, 4)


def counts(stage):
    return {
        k: stage[k] for k in ("planned", "undecidable", "true", "false", "not_reached")
    }


def printed(report):
    """The printed table's lines with their runs of spaces collapsed."""
    return [" ".join(line.split()) for line in report_text(report).splitlines()]


def test_the_put_down_scope_gives_the_report_vector(vectors):
    entry = vectors[0]["tasks"][CARRY]
    assert (entry["criterion_id"], entry["placed_rule"]) == (
        "carry-place-v1",
        "put-down",
    )
    table = entry["overall"]
    assert table["n"] == 12
    c, b = table["C"], table["B"]
    for cells in (c, b):
        assert rate(cells["lifted"], "of_attempted") == (1.0, 0.7575, 1.0)
        assert rate(cells["placed"], "of_attempted") == (0.1111, 0.0199, 0.435)
        assert rate(cells["placed"], "of_all") == (0.0833, 0.0149, 0.3539)
        assert counts(cells["placed"]) == {
            "planned": 12,
            "undecidable": 0,
            "true": 1,
            "false": 8,
            "not_reached": 3,
        }
        assert cells["headline"]["stage"] == "placed"
    assert rate(c["held"], "of_all") == (1.0, 0.7575, 1.0)
    assert rate(b["held"], "of_attempted") == (0.8333, 0.552, 0.953)
    assert c["task_score"] == pytest.approx(0.69444, abs=1e-5)
    assert b["task_score"] == pytest.approx(0.63889, abs=1e-5)
    assert {k: v for k, v in c["failures"]["counts"].items() if v} == {
        "success": 1,
        "dropped": 2,
        "not_placed": 7,
        "undecided": 2,
    }
    assert {k: v for k, v in b["failures"]["counts"].items() if v} == {
        "success": 1,
        "dropped": 3,
        "not_placed": 7,
        "undecided": 1,
    }
    assert b["failures"]["reasons"]["dropped"] == {
        "stopped_dropped": 2,
        "below_threshold_in_window": 1,
    }
    assert c["failures"]["reasons"]["undecided"] == {
        "internal_hold_not_met": 1,
        "put_down_incomplete": 1,
    }
    assert b["failures"]["of_all"]["dropped"] == 0.25
    # Over attempted 1 of 9, over all 1 of 12, under C and again under B.
    row = "placed 1/9 0.111 [0.020, 0.435] 1/12 0.083 [0.015, 0.354]"
    assert printed(vectors[0]).count(row) == 2


def test_the_two_second_scope_gives_the_report_vector(vectors):
    table = vectors[0]["tasks"]["search_lift_hold_2s"]["overall"]
    c, b = table["C"], table["B"]
    assert (c["lifted"]["true"], c["lifted"]["false"]) == (2, 2)
    assert rate(c["held"], "of_attempted") == (1.0, 0.3424, 1.0)
    assert c["held"]["of_all"] == 0.5
    assert rate(b["lifted"], "of_all") == (0.25, 0.0456, 0.6994)
    held = b["held"]
    assert counts(held) == {
        "planned": 4,
        "undecidable": 4,
        "true": 0,
        "false": 0,
        "not_reached": 0,
    }
    assert (held["of_all"], held["of_attempted"]) == (None, None)
    assert c["task_score"] is b["task_score"] is None
    assert c["task_score_why_null"] == "no put-down planned"
    assert c["placed"] is None
    assert (c["headline"]["stage"], rate(c["headline"], "of_all")) == (
        "held",
        (0.5, 0.15, 0.85),
    )
    assert b["headline"] is None
    assert {k: v for k, v in b["failures"]["counts"].items() if v} == {
        "not_lifted": 3,
        "not_decidable": 1,
    }


def test_a_dataset_report_states_what_it_scored(vectors):
    report = vectors[0]
    assert report["status"] == "ok"
    assert report["episodes"] == {
        "listed": 32,
        "scored": 32,
        "refused": 0,
        "refusals": {},
        "refused_episodes": [],
        "unlisted": 0,
        "unlisted_paths": [],
    }
    assert list(report["tasks"]) == [
        "lift_hold",
        "search_lift_hold",
        "search_lift_hold_2s",
        CARRY,
        CARRY_10CM,
    ]
    assert report["criteria"]["C"]["id"] == "agreement-v1"
    assert report["tasks"]["lift_hold"]["population"] == {
        "keep_labels": ["success", "fracture"],
        "keep_requires_lift": True,
        "labels": {"success": 2},
    }
    assert report["check"]["unavailable"] == {
        "agreement": 18,
        "place": 14,
        "stages": 32,
    }
    assert report["check"]["result"] == "unavailable"
    text = report_text(report)
    assert "search_lift_hold_carry_place: 12 episodes, 1 object type" in text
    assert "not decidable" in text and "status        ok" in text
    assert json.loads(json.dumps(report)) == report


@pytest.mark.parametrize(
    ("k", "m", "expected"),
    [
        (60, 60, [0.93982814785791, 0.9999999999999998]),
        (0, 60, [0.0, 0.06017185214208986]),
        (9, 16, [0.331785563988119, 0.7690134759450765]),
        (900, 1000, [0.8798480368046516, 0.9170905564069044]),
        (10, 100, [0.0552291370606751, 0.17436566150491345]),
        # Clipped: unclipped, these bounds round to 1.0000000000000002 and -1.4e-17.
        (16, 16, [0.8063923194655637, 1.0]),
        (0, 21, [0.0, 0.154639018924847]),
    ],
)
def test_the_wilson_interval_matches_its_vectors(k, m, expected):
    assert wilson_interval(k, m) == expected


def test_no_trial_has_no_interval():
    assert wilson_interval(0, 0) is None


# --- edge cases ----------------------------------------------------------------------------


def score(rows, task_block, **meta):
    return score_episode(columns(rows), metadata(rows, task_block, **meta))


@pytest.mark.parametrize(
    ("top", "up"),
    [(0.05, True), (0.05 - 0.5e-9, True), (0.05 - 1e-9, True), (0.05 - 2e-9, False)],
)
def test_criterion_c_reaches_its_rise_within_the_tie(top, up):
    record = score(
        ramp("search_lift_hold_2s", S2, top), block("search_lift_hold_2s", S2)
    )
    assert record["C"]["lifted"] is up
    assert record["C"]["agreement"]["reason"] == (None if up else "never_crossed")


@pytest.mark.parametrize(
    ("clearance", "up"),
    [(0.005, True), (0.005 - 0.5e-9, True), (0.005 - 2e-9, False)],
)
def test_criterion_c_needs_the_lowest_point_5_mm_up_within_the_tie(clearance, up):
    # The centroid rises 15 cm, the lowest point only clearance.
    rows = ramp("search_lift_hold_2s", S2, 0.15, bottom_frac=clearance / 0.15)
    record = score(rows, block("search_lift_hold_2s", S2))
    assert record["C"]["lifted"] is up
    assert record["C"]["agreement"]["reason"] == (None if up else "never_crossed")
    assert record["B"]["lifted"] is True


def test_criterion_c_waits_for_the_lowest_point():
    # The lowest point rises a twentieth of the centroid's rise: 5 mm at row 259, ten
    # rows after the centroid's 5 cm.
    rows = ramp("search_lift_hold", S4, 0.15, bottom_frac=0.05)
    record = score(rows, block("search_lift_hold", S4))
    agreement = record["C"]["agreement"]
    assert (agreement["first_up_frame"], agreement["window_end_frame"]) == (259, 378)
    assert record["C"]["held"] is True


def supported(r, clearance=0.004):
    """The row with the object's lowest point back within clearance of its rest."""
    low = r["object.bbox_min"]
    return dict(r, **{"object.bbox_min": (low[0], low[1], 0.01 + clearance)})


def test_a_lowest_point_back_near_the_bench_inside_the_window_is_still_supported():
    rows = ramp("search_lift_hold", S4, 0.15)
    rows[300] = supported(rows[300])  # inside C's window, rows 249 to 368
    record = score(rows, block("search_lift_hold", S4))
    c, b = record["C"], record["B"]
    assert (c["lifted"], c["held"]) == (True, False)
    assert c["agreement"]["reason"] == "still_supported"
    assert c["agreement"]["min_bottom_clearance_in_window_m"] == pytest.approx(0.004)
    assert (c["class"], c["reasons"]) == ("dropped", ["still_supported"])
    assert (b["lifted"], b["held"], b["class"]) == (True, True, "success")


def test_a_centroid_dip_in_the_window_outranks_a_supported_row():
    rows = ramp("search_lift_hold", S4, 0.15, dips=[(320, 1, 0.12)])
    rows[300] = supported(rows[300])
    record = score(rows, block("search_lift_hold", S4))
    assert record["C"]["agreement"]["reason"] == "dipped_in_window"
    assert record["C"]["reasons"] == ["dipped_in_window"]


def at_heights(rows, z):
    """The rows with the object's centroid at height z(i) and its box 4 cm around it."""
    return [
        dict(
            r,
            **{
                "object.centroid": (*r["object.centroid"][:2], z(i)),
                "object.bbox_min": (*r["object.bbox_min"][:2], z(i) - 0.04),
                "object.bbox_max": (*r["object.bbox_max"][:2], z(i) + 0.04),
            },
        )
        for i, r in enumerate(rows)
    ]


@pytest.mark.parametrize(
    ("top", "lifted", "held", "crossing"),
    [(0.16, True, True, 269), (math.nextafter(0.16, 0.0), False, None, None)],
)
def test_criterion_b_counts_a_rise_of_exactly_12_cm_and_has_no_tie(
    top, lifted, held, crossing
):
    # The object rests at 0.04 m and stands at top from row 269, the last lift row, on.
    assert 0.16 - 0.04 == 0.12
    rows = at_heights(
        ramp("search_lift_hold", S4, 0.0), lambda i: 0.04 if i < 269 else top
    )
    b = score(rows, block("search_lift_hold", S4))["B"]
    assert (b["lifted"], b["held"], b["crossing"]) == (lifted, held, crossing)


@pytest.mark.parametrize(
    ("dip", "held_c", "held_b"),
    [(368, False, False), (369, True, False), (504, True, False), (505, True, True)],
)
def test_a_dip_counts_up_to_the_last_row_of_each_window(dip, held_c, held_b):
    # C runs from k = 249 to 249 + 120 - 1, B from c = 264 to 264 + 240.
    rows = ramp("search_lift_hold", S4, 0.15, dips=[(dip, 1, 0.14)])
    record = score(rows, block("search_lift_hold", S4))
    assert (record["C"]["held"], record["B"]["held"]) == (held_c, held_b)
    assert record["C"]["agreement"]["window_end_frame"] == 368
    assert record["B"]["window_end"] == 504


@pytest.mark.parametrize(
    ("cut", "held_c", "held_b"),
    [(368, None, None), (369, True, None), (504, True, None), (505, True, True)],
)
def test_a_window_is_cut_short_only_when_its_last_row_is_missing(cut, held_c, held_b):
    # C's window ends at row 368 and B's at row 504; the recording keeps rows 0 to cut - 1.
    rows = ramp("search_lift_hold", S4, 0.15, cut=cut)
    record = score(rows, block("search_lift_hold", S4))
    assert (record["C"]["held"], record["B"]["held"]) == (held_c, held_b)
    reason = None if held_c else "window_truncated"
    assert record["C"]["agreement"]["reason"] == reason


def test_a_row_that_is_not_finite_is_skipped_inside_a_window():
    rows = blow_up(ramp("search_lift_hold", S4, 0.15), 300)
    record = score(rows, block("search_lift_hold", S4))
    assert (record["C"]["held"], record["B"]["held"]) == (True, True)
    assert record["C"]["agreement"]["any_window"] is True


def test_a_window_cut_short_is_decided_by_a_recorded_dip_under_c_only():
    rows = ramp("search_lift_hold", S4, 0.15, dips=[(300, 2, 0.14)], cut=320)
    record = score(rows, block("search_lift_hold", S4))
    assert record["C"]["agreement"]["reason"] == "dipped_in_window"
    assert record["B"]["held"] is None


def test_a_skipped_lift_leaves_an_episode_without_a_rise_undecided():
    rows = [r for r in ramp("search_lift_hold", S4, 0.03) if r["robot.phase"] != "lift"]
    record = score(rows, block("search_lift_hold", S4))
    assert record["C"]["lifted"] is None
    assert record["C"]["agreement"]["reason"] == "incomplete"
    assert record["C"]["class"] == "undecided" and record["C"]["reasons"] == [
        "incomplete"
    ]


@pytest.mark.parametrize("stop", [None, "dropped", "never_lifted"])
def test_an_episode_with_no_usable_row_is_undecided_where_b_cannot_decide(stop):
    # Without a row that has both heights, C is not reached whatever the stop.
    rows = [dict(r, finite=False) for r in ramp("search_lift_hold_2s", S2, 0.15)]
    record = score(rows, block("search_lift_hold_2s", S2, stop=stop))
    c, b = record["C"], record["B"]
    assert record["decidable"] == {"C": True, "B": False}
    assert (c["lifted"], c["class"], c["reasons"]) == (
        None,
        "undecided",
        ["incomplete"],
    )
    assert (b["lifted"], b["class"], b["reasons"]) == (
        None,
        "undecided",
        ["no_usable_rows"],
    )


def test_a_drop_stop_before_any_row_is_up_is_not_lifted_under_c():
    rows = ramp("search_lift_hold_2s", S2, 0.03, cut=300)
    record = score(rows, block("search_lift_hold_2s", S2, stop="dropped"))
    c = record["C"]
    assert (c["lifted"], c["agreement"]["reason"]) == (False, "stopped_dropped")
    assert (c["class"], c["reasons"]) == ("not_lifted", ["stopped_dropped"])


def test_another_recorded_rate_sizes_both_windows(tmp_path):
    s4 = [15, 45, 15, 15, 30, 15, 120]
    write(
        tmp_path,
        [
            episode(
                "ep-30",
                ramp("search_lift_hold", s4, 0.15),
                block("search_lift_hold", s4),
            )
        ],
        30,
    )
    report, records = scored(tmp_path)
    record = records["ep-30"]
    k, c = record["C"]["agreement"]["first_up_frame"], record["B"]["crossing"]
    assert record["C"]["agreement"]["window_end_frame"] == k + 59
    assert record["B"]["window_end"] == c + 120
    assert record["decidable"] == {"C": True, "B": True}
    assert report["status"] == "ok"


@pytest.mark.parametrize(
    ("hold", "c", "b"),
    [(118, False, False), (119, True, False), (239, True, False), (240, True, True)],
)
def test_a_window_is_decidable_when_it_ends_inside_the_plan(hold, c, b):
    # 270 rows come before the hold: C's window needs 270 + 120 - 1 rows, B's 270 + 240.
    schedule = [*S4[:-1], hold]
    rows = ramp("search_lift_hold", schedule, 0.15)
    record = score(rows, block("search_lift_hold", schedule))
    assert record["decidable"] == {"C": c, "B": b}


def test_the_retry_task_reads_every_row():
    record = score(retry(second_top=0.1), block("lift_hold_retry", RETRY))
    assert record["B"]["lifted"] is True
    assert record["B"]["crossing"] < sum(RETRY[:5])
    assert record["B"]["held"] is False
    assert record["C"]["lifted"] is True


def test_the_retry_task_judges_the_lift_of_the_attempt_it_ended_in():
    # One attempt, held to its end, the lowest point never off the bench: no row is up,
    # and that attempt's lift is complete although the plan holds a second lift.
    rows = [supported(r, 0.0) for r in retry(first_drop=False)]
    record = score(rows, block("lift_hold_retry", RETRY))
    assert record["C"]["lifted"] is False
    assert record["C"]["agreement"]["reason"] == "never_crossed"


def test_the_put_back_returns_the_object_where_it_rested():
    task = block("lift_hold_place", LHP)
    assert score(put_back(), task)["placed_task"]["verdict"] is True
    never = score(put_back(never_open=True), task)["placed_task"]
    assert (never["verdict"], never["reasons"]) == (False, ["never_released"])
    moved = score(put_back(drift=0.03), task)["placed_task"]
    assert (moved["verdict"], moved["reasons"]) == (False, ["not_back_at_start"])
    record = score(put_back(fall=0.02), task)
    assert record["placed_task"]["reasons"] == ["fell_on_release"]
    assert record["C"]["class"] == record["B"]["class"] == "not_placed"


def test_a_missing_commanded_width_is_no_width():
    rows, task = put_back(), block("lift_hold_place", LHP)
    # The put-back reads the width at the last row of its hold, 240 rows from the crossing.
    end = score(rows, task)["B"]["crossing"] + 240
    rows[end] = dict(rows[end], **{"robot.gripper_width_cmd_m": math.nan})
    placed = score(rows, task)["placed_task"]
    assert (placed["verdict"], placed["why"]) == (None, "no_grip_width")


def widened(rows, first, last, width):
    """The rows with the commanded width set to width on rows first to last."""
    return [
        dict(r, **{"robot.gripper_width_cmd_m": width}) if first <= i <= last else r
        for i, r in enumerate(rows)
    ]


def moved_at(rows, at, offset):
    """The rows with the object moved by offset (x, y, z) on row at."""
    return [
        dict(r, **{k: tuple(a + d for a, d in zip(r[k], offset)) for k in OBJECT})
        if i == at
        else r
        for i, r in enumerate(rows)
    ]


def without_tool_at(rows, at):
    """The rows with no reached tool height on row at."""
    pose = (0.5, 0.0, math.nan, 0.0, 0.0, 0.0, 1.0)
    return [
        dict(r, **{"robot.ee_pose": pose}) if i == at else r for i, r in enumerate(rows)
    ]


# The put-back's crossing is row 114 and its hold ends at row 354, 240 rows later; the
# place waypoints start at rows 360 and 390, where the jaws open; its last 10 rows are
# 440 to 449.
@pytest.mark.parametrize(
    ("rows", "verdict", "reasons", "why"),
    [
        (put_back(drop=True), False, ["descent_not_with_tool"], None),
        (moved_at(put_back(), 440, (0.003, 0.0, 0.0)), False, ["not_at_rest"], None),
        (moved_at(put_back(), 439, (0.003, 0.0, 0.0)), True, [], None),
        # The jaws hold 0.05 m wide: a width exactly 1 um wider is no release.
        (
            widened(put_back(), 370, 370, math.nextafter(0.05 + 1e-6, 1.0)),
            False,
            ["fell_on_release"],
            None,
        ),
        (widened(put_back(), 370, 370, 0.05 + 1e-6), True, [], None),
        (widened(put_back(), 356, 389, 0.12), None, [], "too_few_carried_rows"),
        (
            widened(put_back(), 357, 389, 0.12),
            False,
            ["descent_not_with_tool", "fell_on_release"],
            None,
        ),
        ([dict(r, finite=False) for r in put_back()], None, [], "no_usable_rows"),
        (put_back(top=0.11), None, [], "internal_lift_not_met"),
        (put_back()[:354], None, [], "internal_hold_truncated"),
        (put_back()[:355], None, [], "fewer_than_10_rows_after_hold"),
        (
            moved_at(put_back(), 200, (0.0, 0.0, -0.05)),
            None,
            [],
            "internal_hold_not_met",
        ),
        (put_back()[:363], None, [], "fewer_than_10_rows_after_hold"),
        (put_back()[:364], False, ["never_released"], None),
        (without_tool_at(put_back(), 370), None, [], "ee_pose_missing"),
    ],
)
def test_each_put_back_bound(rows, verdict, reasons, why):
    placed = score(rows, block("lift_hold_place", LHP))["placed_task"]
    assert (placed["verdict"], placed["reasons"], placed["why"]) == (
        verdict,
        reasons,
        why,
    )


def dragged(height):
    """A tool that lowers the object to height over the bench 5 cm short of the mat at
    row 500 and drags it there by row 600."""
    return lambda i: (
        (0.5, 0.10 + 0.05 * (i - 500) / 100, Z_CLOSE + height)
        if 500 <= i < 600
        else None
    )


def set_down_fast(i):
    """A tool that keeps the hold height until row 580, then comes straight down to the
    bench in 20 rows."""
    if 540 <= i < 600:
        return (*tool(i)[:2], Z_CLOSE + 0.17 * min(1.0, (599 - i) / 20))
    return None


def pushed(dx):
    return lambda i: (dx * clamp((i - 650) / 20), 0.0)


def nudged(at):
    return lambda i: (0.003, 0.0) if i == at else (0.0, 0.0)


def pressed(rows, at, force):
    """The rows with the bench force set to force on row at."""
    return [
        dict(r, **{"force.ground_force_mag": force}) if i == at else r
        for i, r in enumerate(rows)
    ]


def bottom_at(rows, at, z):
    """The rows with the object's lowest point at height z on row at."""
    return [
        dict(r, **{"object.bbox_min": (*r["object.bbox_min"][:2], z)}) if i == at else r
        for i, r in enumerate(rows)
    ]


@pytest.mark.parametrize(
    ("rows", "verdict", "reasons", "fact", "value"),
    [
        (put_down(path=dragged(0.003)), False, ["dragged"], "dragged_max_m", 0.0495),
        (put_down(path=dragged(0.02)), True, [], "dragged_max_m", 0.0),
        (
            put_down(clearance=0.018, slip=lambda i: 0.015 * clamp((i - 400) / 3)),
            False,
            ["came_down_fast"],
            "came_down_excess_max_m",
            0.015,
        ),
        (
            put_down(clearance=0.0, path=set_down_fast),
            False,
            ["set_down_too_fast"],
            "set_down_speed_m_s",
            0.51,
        ),
        (
            put_down(shove=pushed(0.011)),
            False,
            ["pushed_after_release"],
            "push_after_release_m",
            0.011,
        ),
        (put_down(shove=pushed(0.009)), True, [], "push_after_release_m", 0.009),
        # The rest window is the last 30 rows of the span, 690 to 719.
        (put_down(wander=nudged(690)), False, ["not_at_rest"], "rest_wander_m", 0.003),
        (put_down(wander=nudged(689)), True, [], "rest_wander_m", 0.0),
        # The force peak is read up to the row the jaws are fully open, 630.
        (pressed(put_down(), 630, 5.0), True, [], "set_down_bench_force_peak_n", 5.0),
        (pressed(put_down(), 631, 5.0), True, [], "set_down_bench_force_peak_n", 0.0),
        # Let go 3 mm up, the object first comes within 1 mm of its rest height
        # underneath, 0.0, on row 603, or on row 602 when that row is exactly 1 mm up.
        (bottom_at(put_down(), 602, 0.001), True, [], "first_touch_frame", 602),
        (
            bottom_at(put_down(), 602, math.nextafter(0.001, 1.0)),
            True,
            [],
            "first_touch_frame",
            603,
        ),
    ],
)
def test_each_put_down_bound(rows, verdict, reasons, fact, value):
    placed = score(rows, block(CARRY, CP))["placed_task"]
    assert (placed["verdict"], placed["reasons"], placed["why"]) == (
        verdict,
        reasons,
        None,
    )
    assert placed["facts"][fact] == pytest.approx(value, abs=1e-9)


@pytest.mark.parametrize(
    ("dx", "dy", "verdict", "reasons"),
    [
        (0, 0, True, []),
        (1e-9, 0, False, ["off_target"]),
        (-1e-9, 0, False, ["off_target"]),
        (0, 1e-9, False, ["off_target"]),
        (0, -1e-9, False, ["off_target"]),
    ],
)
def test_the_object_must_end_on_the_mat_on_every_side(dx, dy, verdict, reasons):
    # The object ends 8 cm wide around (0.5, 0.15), and the mat is as wide: its bounds,
    # the centroid +/- 0.04, lie on the edges, the mat's centre +/- 0.04. A mat moved a
    # nanometre leaves the object off it on one side.
    target = dict(
        TARGET, x_m=TARGET["x_m"] + dx, y_m=TARGET["y_m"] + dy, half_extent_m=0.04
    )
    placed = score(put_down(), block(CARRY, CP, place_target=target))["placed_task"]
    assert (placed["verdict"], placed["reasons"]) == (verdict, reasons)


@pytest.mark.parametrize(
    "rows",
    [
        # Exactly at the widest held width plus the deadband: not open.
        widened(put_down(), 560, 589, 0.076 + 0.003),
        # The widest held width is on the window's last row, 240 after the crossing.
        widened(widened(put_down(), 499, 499, 0.081), 520, 549, 0.082),
    ],
)
def test_the_jaws_open_only_past_the_widest_held_width_and_the_deadband(rows):
    placed = score(rows, block(CARRY, CP))["placed_task"]
    assert (placed["facts"]["opening_frame"], placed["verdict"]) == (600, True)


def test_a_put_down_with_no_row_after_the_jaws_are_open_is_still_touching():
    # The judged span is rows 600 to 629: no row of it follows row 630, where the jaws
    # are fully open.
    judge = {"opening_to_end_frames": 30, "rest_window_frames": 20}
    placed = score(put_down(), block(CARRY, CP, judge=judge))["placed_task"]
    assert (placed["verdict"], placed["reasons"]) == (False, ["still_touching"])
    assert placed["facts"]["gripper_contacts_after_open"] == 0


def test_the_10_cm_centroid_bound_is_inclusive():
    target = dict(TARGET, x_m=TARGET["x_m"] + 0.003)
    task = block(CARRY_10CM, CP, place_target=target)
    rows = put_down(task=CARRY_10CM)
    error = score(rows, task)["placed_task"]["facts"]["place_error_m"]
    for bound, verdict in ((error, True), (math.nextafter(error, 0.0), False)):
        judge = {"place_error_tol_m": bound}
        task = block(CARRY_10CM, CP, place_target=target, judge=judge)
        assert score(rows, task)["placed_task"]["verdict"] is verdict


def measured(name, times=1.0):
    """The tolerance a put-down meets exactly: its fact name, times times."""
    return lambda facts, _: times * facts[name]


def centroid_at(rows, i):
    return rows[i]["object.centroid"]


def with_the_tool(rows, first, last):
    """The rows with the object's centroid at the tool's height on rows first to last."""
    out = [dict(r) for r in rows]
    for r in out[first : last + 1]:
        r["object.centroid"] = (*r["object.centroid"][:2], r["robot.ee_pose"][2])
    return out


# Each tolerance of the two placements is met at the value the episode measures and
# missed at the float under it.
@pytest.mark.parametrize(
    ("rows", "task", "key", "edge", "word"),
    [
        (
            put_down(slip=lambda i: 0.030 * clamp((i - 300) / 120)),
            CARRY,
            "carry_slip_tol_m",
            measured("carry_slip_max_m"),
            "slipped_in_jaws",
        ),
        (
            put_down(path=dragged(0.003)),
            CARRY,
            "place_tol_m",
            measured("dragged_max_m"),
            "dragged",
        ),
        (
            put_down(clearance=0.018, slip=lambda i: 0.015 * clamp((i - 400) / 3)),
            CARRY,
            "place_release_drop_tol_m",
            measured("came_down_excess_max_m"),
            "came_down_fast",
        ),
        (
            put_down(clearance=0.0, path=set_down_fast),
            CARRY,
            "set_down_speed_max_m_s",
            measured("set_down_speed_m_s"),
            "set_down_too_fast",
        ),
        (
            put_down(clearance=0.015, fall=lambda k: min(0.015, 0.0025 * k)),
            CARRY,
            "place_release_drop_tol_m",
            measured("fall_after_open_m"),
            "fell_on_release",
        ),
        # The push bound is half of place_tol_m.
        (
            put_down(shove=pushed(0.011)),
            CARRY,
            "place_tol_m",
            measured("push_after_release_m", 2.0),
            "pushed_after_release",
        ),
        (
            put_down(wander=nudged(690)),
            CARRY,
            "place_rest_radius_m",
            measured("rest_wander_m"),
            "not_at_rest",
        ),
        # The put-back opens at row 390 and ends at row 449; the object rested at row 0.
        (
            put_back(fall=0.02),
            "lift_hold_place",
            "place_release_drop_tol_m",
            lambda _, rows: centroid_at(rows, 390)[2] - centroid_at(rows, 449)[2],
            "fell_on_release",
        ),
        (
            moved_at(put_back(), 440, (0.003, 0.0, 0.0)),
            "lift_hold_place",
            "place_rest_radius_m",
            lambda _, rows: math.dist(centroid_at(rows, 440), centroid_at(rows, 449)),
            "not_at_rest",
        ),
        (
            put_back(drift=0.03),
            "lift_hold_place",
            "place_tol_m",
            lambda _, rows: math.dist(centroid_at(rows, 449), centroid_at(rows, 0)),
            "not_back_at_start",
        ),
        # Held at the tool point from the end of the hold to the opening, the object
        # comes down exactly as fast as the tool.
        (
            with_the_tool(put_back(), 354, 390),
            "lift_hold_place",
            "place_max_descent_factor",
            lambda *_: 1.0,
            "descent_not_with_tool",
        ),
    ],
)
def test_each_placement_tolerance_is_inclusive(rows, task, key, edge, word):
    schedule = SCHEDULES[task]
    facts = score(rows, block(task, schedule))["placed_task"]["facts"]
    value = edge(facts, rows)
    for tolerance, verdict, reasons in (
        (value, True, []),
        (math.nextafter(value, 0.0), False, [word]),
    ):
        task_block = block(task, schedule, judge={key: tolerance})
        placed = score(rows, task_block)["placed_task"]
        assert (placed["verdict"], placed["reasons"]) == (verdict, reasons)


def test_a_carried_row_counts_as_dragged_only_under_the_drop_tolerance():
    # The tool drags the object over rows 500 to 599 with its lowest point h over its
    # rest height. Only the carry is read: the object then falls h after the opening,
    # which the same key bounds.
    rows = put_down(path=dragged(0.003))
    h = rows[550]["object.bbox_min"][2] - rows[0]["object.bbox_min"][2]
    for tolerance, reasons in ((h, []), (math.nextafter(h, 1.0), ["dragged"])):
        task = block(CARRY, CP, judge={"place_release_drop_tol_m": tolerance})
        facts = score(rows, task)["placed_task"]["facts"]
        assert facts["carried_fail_reasons"] == reasons


def test_the_put_back_counts_a_rise_of_exactly_lift_success_m():
    # The object reaches its full rise at row 119 and keeps it to the end of the hold, at
    # row 359.
    rows = put_back()
    rise = centroid_at(rows, 119)[2] - centroid_at(rows, 0)[2]
    for threshold, outcome in (
        (rise, (True, None)),
        (math.nextafter(rise, 1.0), (None, "internal_lift_not_met")),
    ):
        task = block("lift_hold_place", LHP, judge={"lift_success_m": threshold})
        placed = score(rows, task)["placed_task"]
        assert (placed["verdict"], placed["why"]) == outcome


def test_a_hold_the_recording_cuts_short_leaves_the_put_down_unjudged():
    # No drop stop: the internal hold from the crossing at row 259 needs rows to 499.
    record = score(put_down(truncate=400), block(CARRY, CP))
    assert record["placed_task"]["why"] == "internal_hold_truncated"
    assert (record["C"]["class"], record["C"]["reasons"]) == (
        "undecided",
        ["internal_hold_truncated"],
    )
    assert (record["B"]["held"], record["B"]["reasons"]) == (None, ["window_truncated"])


def test_a_carry_the_recording_cuts_before_the_jaws_open_is_not_judged():
    record = score(put_down(truncate=520), block(CARRY, CP))
    placed = record["placed_task"]
    assert (placed["verdict"], placed["why"]) == (None, "carry_not_judgeable")
    assert placed["facts"]["carry_slip_max_m"] is None
    assert (record["B"]["held"], record["B"]["placed"]) == (True, None)
    assert record["B"]["reasons"] == ["carry_not_judgeable"]


def test_placed_counts_under_a_criterion_only_once_it_held():
    # The 10 cm put-down holds its own 3.0 s, then slips 8 cm in the jaws inside B's 4 s.
    rows = put_down(task=CARRY_10CM, slip=lambda i: 0.08 if 470 <= i < 475 else 0.0)
    record = score(rows, block(CARRY_10CM, CP))
    placed = record["placed_task"]
    assert (placed["verdict"], placed["reasons"]) == (False, ["slipped_in_jaws"])
    assert (record["C"]["held"], record["B"]["held"]) == (True, False)
    assert (record["C"]["placed"], record["B"]["placed"]) == (False, None)
    assert (record["C"]["class"], record["B"]["class"]) == ("not_placed", "dropped")


def test_the_put_down_is_anchored_on_the_generator_threshold():
    # The object hangs 6 cm low in the jaws: it rises 11 cm, and the tool stops 6 cm
    # higher to set it down 3 mm over the bench.
    rows = put_down(clearance=0.063, slip=lambda i: 0.06)
    record = score(rows, block(CARRY, CP, judge={"lift_success_m": 0.10}))
    assert record["placed_task"]["verdict"] is True
    assert (record["C"]["placed"], record["C"]["class"]) == (True, "success")
    assert (record["B"]["lifted"], record["B"]["class"]) == (False, "not_lifted")


def test_several_put_down_failures_are_each_counted(tmp_path):
    rows = put_down(clearance=0.015, fall=lambda k: min(0.015, 0.0025 * k))
    task = block(CARRY, CP, place_target=dict(TARGET, x_m=0.547))
    write(tmp_path, [episode("ep-0", rows, task)])
    report, records = scored(tmp_path)
    reasons = records["ep-0"]["placed_task"]["reasons"]
    assert reasons[0] == "off_target" and "fell_on_release" in reasons
    tally = report["tasks"][CARRY]["overall"]["C"]["failures"]["reasons"]["not_placed"]
    assert tally == dict.fromkeys(reasons, 1)


def test_a_carry_task_without_its_mat_is_refused():
    with pytest.raises(Refused) as refused:
        score(put_down(), block(CARRY, CP, place_target=None))
    assert refused.value.word == "task_block"
    assert "place_target: must be a mapping" in refused.value.detail


@pytest.mark.parametrize(
    ("change", "word"),
    [
        ({"format_version": "1.0"}, "format"),
        ({"format_version": 2.0}, "format"),
        ({"label": "lost"}, "label"),
        ({"label": ["success"]}, "label"),
        ({"recorded_fps": True}, "recorded_fps"),
        ({"n_steps": 10}, "n_steps"),
        ({"source_n_frames": None}, "trajectory_cut"),
        ({"task": {"name": "lift_hold_place"}}, "task_block"),
    ],
)
def test_an_episode_that_breaks_an_acceptance_rule_is_refused(change, word):
    rows = ramp("lift_hold", LH, 0.15)
    with pytest.raises(Refused) as refused:
        score_episode(
            columns(rows), {**metadata(rows, block("lift_hold", LH)), **change}
        )
    assert refused.value.word == word


def test_a_refusal_shows_the_values_it_found():
    rows = ramp("lift_hold", LH, 0.15)
    with pytest.raises(Refused) as refused:
        score(rows, block("lift_hold", LH), format_version=2.0)
    assert refused.value.detail == (
        "the episode is 'physgraspgen-public' 2.0, not 'physgraspgen-public' '2.0'"
    )


def test_a_trajectory_that_does_not_start_at_step_1_is_refused():
    rows = ramp("lift_hold", LH, 0.15)
    cols = columns(rows)
    cols["sim_step"] = cols["sim_step"] + 1
    with pytest.raises(Refused, match="rows"):
        score_episode(cols, metadata(rows, block("lift_hold", LH)))


def test_the_task_block_check_names_every_problem():
    task = block(
        CARRY_10CM, CP, stop="slipped", place_target={"x_m": 0.5, "half_extent_m": 0}
    )
    del task["keep_requires_lift"]
    task["judge"].update(
        lift_success_m=0.2, carry_slip_tol_m=float("nan"), open_ramp_frames=1.5
    )
    task["judge"].pop("place_error_tol_m")
    task["frames_per_waypoint"] = [30, 30]
    problems = task_block_problems(task, 60)
    keys = [p.split(":")[0] for p in problems]
    assert keys == [
        "frames_per_waypoint",
        "stop_reason",
        "judge.approach_height_m",
        "judge.carry_slip_tol_m",
        "judge.open_ramp_frames",
        "judge.place_error_tol_m",
        "place_target.y_m",
        "place_target.half_extent_m",
        "keep_requires_lift",
    ]
    assert task_block_problems(block(CARRY, CP), 60) == []
    every = block(CARRY, CP, keep_labels=None, keep_requires_lift=False)
    assert task_block_problems(every, 60) == []
    assert task_block_problems(None) == ["task: missing, or not a mapping"]


def test_an_unknown_task_name_still_has_its_other_keys_checked():
    task = block(
        "lift_hold",
        LH,
        name="lift_and_hold",
        phases="approach",
        frames_per_waypoint=[],
        criterion_id=1,
        stop_reason="slipped",
    )
    keys = [p.split(":")[0] for p in task_block_problems(task, 60)]
    assert keys == [
        "name",
        "phases",
        "frames_per_waypoint",
        "criterion_id",
        "stop_reason",
    ]


def test_criterion_fps_is_compared_only_with_a_recorded_rate():
    criterion = {"id": "carry-place-v1", "task": CARRY, "fps": 60}
    task = block(CARRY, CP, criterion=dict(criterion, internal={"hold_s": 4.0}))
    assert task_block_problems(task) == []
    assert task_block_problems(task, 60) == []
    assert task_block_problems(task, 30) == [
        "criterion.fps: 60 differs from recorded_fps 30"
    ]


@pytest.mark.parametrize(
    ("key", "value", "rule"),
    [
        ("place_tol_m", "2 cm", "must be a finite number > 0"),
        ("carry_slip_tol_m", [0.02], "must be a finite number > 0"),
        ("open_ramp_frames", 1.5, "must be an int >= 1"),
    ],
)
def test_a_judge_key_the_task_does_not_need_still_has_its_type(key, value, rule):
    rows = ramp("lift_hold", LH, 0.15)
    with pytest.raises(Refused) as refused:
        score(rows, block("lift_hold", LH, judge={key: value}))
    assert refused.value.word == "task_block"
    assert refused.value.detail == f"judge.{key}: {rule}, got {value!r}"
    assert score(rows, block("lift_hold", LH, judge={key: None}))["B"]["held"] is True


@pytest.mark.parametrize(
    ("task", "extra", "problem"),
    [
        (CARRY, {"floor_arm": "no_vision_pick"}, "floor_arm"),
        ("search_lift_hold", {"floor_arm": "reflection"}, "floor_arm"),
        (CARRY, {"timing_scale": {"place": 0.9}}, "timing_scale"),
        ("lift_hold", {"place_target": dict(TARGET)}, "place_target"),
        (CARRY, {"judge": {"place_error_tol_m": 0.005}}, "judge.place_error_tol_m"),
        (
            "lift_hold",
            {"judge": {"approach_height_m": 0.12}},
            "judge.approach_height_m",
        ),
        (CARRY, {"keep_labels": []}, "keep_labels"),
        (CARRY, {"keep_labels": ["kept"]}, "keep_labels"),
        (CARRY, {"keep_labels": None}, "keep_requires_lift"),
        (CARRY, {"stages": {"lifted": "yes"}}, "stages"),
        (CARRY, {"criterion": {"id": "carry-place-10cm-v1"}}, "criterion.id"),
        (CARRY, {"criterion": {"task": CARRY_10CM}}, "criterion.task"),
        (CARRY, {"criterion": {"fps": 30}}, "criterion.fps"),
        (
            CARRY,
            {"criterion": {"internal": {"hold_s": 3.0}}},
            "criterion.internal.hold_s",
        ),
        (
            CARRY,
            {"criterion": {"internal": {"carry_m": 0.10}}},
            "criterion.internal.carry_m",
        ),
        (
            CARRY,
            {"criterion": {"internal": {"lift_success_m": 0.1}}},
            "criterion.internal.lift_success_m",
        ),
        (
            CARRY,
            {"criterion": {"placement": {"floor_arm": "reflection"}}},
            "criterion.placement.floor_arm",
        ),
    ],
)
def test_the_task_block_is_checked_against_itself(task, extra, problem):
    problems = task_block_problems(block(task, SCHEDULES[task], **extra), 60)
    assert [p.split(":")[0] for p in problems] == [problem]


def test_scaled_search_timing_is_accepted_and_reported():
    task = block("search_lift_hold", S4, timing_scale={"search": 0.9, "approach": 1.1})
    record = score(ramp("search_lift_hold", S4, 0.15), task)
    assert record["timing_scale"] == {"search": 0.9, "approach": 1.1}


@pytest.mark.parametrize(
    ("waypoint", "phase"), [(0, "hold"), (5, "approach"), (-1, "hold")]
)
def test_a_row_off_the_plan_is_refused(waypoint, phase):
    rows = ramp("lift_hold", LH, 0.15)
    rows[10] = dict(rows[10], **{"robot.waypoint": waypoint, "robot.phase": phase})
    with pytest.raises(Refused, match="phase_row"):
        score(rows, block("lift_hold", LH))


# --- the dataset ----------------------------------------------------------------------------


def test_refused_and_unlisted_episodes_make_the_run_incomplete(tmp_path):
    good = episode("ep-0", ramp("lift_hold", LH, 0.15), block("lift_hold", LH))
    bad = episode("ep-1", ramp("lift_hold", LH, 0.15), {"name": "lift_hold"})
    cut = episode(
        "ep-2", ramp("lift_hold", LH, 0.15), block("lift_hold", LH), source_n_frames=400
    )
    write(tmp_path, [good, bad, cut])
    trajectory = tmp_path / "trajectories" / "apple"
    (trajectory / "ep-3.parquet").write_bytes(
        (trajectory / "ep-0.parquet").read_bytes()
    )
    report, records = scored(tmp_path)
    assert report["episodes"]["refusals"] == {"task_block": 1, "trajectory_cut": 1}
    assert report["episodes"]["refused_episodes"] == [records["ep-1"], records["ep-2"]]
    assert report["episodes"]["unlisted_paths"] == ["trajectories/apple/ep-3.parquet"]
    assert report["status"] == "incomplete"
    assert records["ep-1"]["refused"] == "task_block"
    assert "layout: missing" in records["ep-1"]["detail"]
    assert report["tasks"]["lift_hold"]["overall"]["n"] == 1
    lines = report_text(report).splitlines()
    assert lines[3].startswith("  ep-1          task_block: layout: missing; source: ")
    assert (
        "  ep-2          trajectory_cut: the trajectory has 390 rows, the generator "
        "recorded 400" in lines
    )
    assert "  unlisted      trajectories/apple/ep-3.parquet" in lines


def test_a_trajectory_without_a_column_the_scorer_reads_is_refused(tmp_path):
    write(
        tmp_path, [episode("ep-0", ramp("lift_hold", LH, 0.15), block("lift_hold", LH))]
    )
    path = tmp_path / "trajectories" / "apple" / "ep-0.parquet"
    footer = {b"physgraspgen": pq.read_metadata(path).metadata[b"physgraspgen"]}
    table = pq.read_table(path).drop_columns(["force.ground_force_mag"])
    pq.write_table(table.replace_schema_metadata(footer), path)
    _, records = scored(tmp_path)
    assert records["ep-0"]["refused"] == "columns"
    assert records["ep-0"]["detail"] == "the trajectory lacks force.ground_force_mag"


def lift_holds(n):
    rows = ramp("lift_hold", LH, 0.15)
    return [episode(f"ep-{k}", rows, block("lift_hold", LH)) for k in range(n)]


def test_the_listing_must_describe_the_file(tmp_path):
    write(tmp_path, lift_holds(5))
    path = tmp_path / "metadata.yaml"
    meta = yaml.safe_load(path.read_text())
    trajectories = [f for f in meta["files"] if f["kind"] == "trajectory"]
    trajectories[0]["n_steps"] += 1
    trajectories[1]["episode_uid"] = "ep-9"
    trajectories[3]["path"] = "../" + trajectories[3]["path"]
    trajectories[4]["path"] = str(tmp_path / trajectories[4]["path"])
    path.write_text(yaml.safe_dump(meta))
    (tmp_path / trajectories[2]["path"]).unlink()
    report, records = scored(tmp_path)
    assert [(uid, e["refused"]) for uid, e in records.items()] == [
        ("ep-0", "n_steps"),
        ("ep-2", "unreadable"),
        ("ep-3", "listing"),
        ("ep-4", "listing"),
        ("ep-9", "listing"),
    ]
    assert records["ep-3"]["detail"] == (
        "'../trajectories/apple/ep-3.parquet' is not a path under the root"
    )
    assert report["episodes"]["unlisted"] == 2


def test_an_episode_listed_twice_is_scored_once(tmp_path):
    write(tmp_path, lift_holds(2))
    path = tmp_path / "metadata.yaml"
    meta = yaml.safe_load(path.read_text())
    trajectories = [f for f in meta["files"] if f["kind"] == "trajectory"]
    meta["files"].append(dict(trajectories[0]))
    path.write_text(yaml.safe_dump(meta))
    report, episodes = score_dataset(tmp_path)
    assert [(e["episode_uid"], e.get("refused")) for e in episodes] == [
        ("ep-0", None),
        ("ep-0", "listing"),
        ("ep-1", None),
    ]
    assert episodes[1]["detail"] == (
        "metadata.yaml lists episode 'ep-0' of 'apple' more than once"
    )
    assert report["tasks"]["lift_hold"]["overall"]["n"] == 2
    assert report["status"] == "incomplete"


def edit_footer(path, **changes):
    """Rewrite a trajectory file with these changes to its footer metadata."""
    meta = json.loads(pq.read_metadata(path).metadata[b"physgraspgen"])
    meta.update(changes)
    footer = {b"physgraspgen": json.dumps(meta).encode()}
    pq.write_table(pq.read_table(path).replace_schema_metadata(footer), path)


def test_a_malformed_label_or_listing_refuses_its_episode_alone(tmp_path):
    root = tmp_path / "d"
    write(root, lift_holds(5))
    trajectory = root / "trajectories" / "apple"
    edit_footer(trajectory / "ep-0.parquet", label=["success"])
    edit_footer(trajectory / "ep-4.parquet", object_type=["apple"])
    path = root / "metadata.yaml"
    meta = yaml.safe_load(path.read_text())
    trajectories = [f for f in meta["files"] if f["kind"] == "trajectory"]
    trajectories[1]["episode_uid"] = ["ep-1"]
    # Written unquoted, a date reads back from YAML as a date, not a string.
    trajectories[2]["episode_uid"] = meta["version"] = datetime.date(2026, 10, 25)
    trajectories[4]["object_type"] = ["apple"]
    path.write_text(yaml.safe_dump(meta))
    report, episodes = score_dataset(root)
    assert [
        (e["object_type"], e["episode_uid"], e.get("refused")) for e in episodes
    ] == [
        ("['apple']", "ep-4", "listing"),
        ("apple", "2026-10-25", "listing"),
        ("apple", "['ep-1']", "listing"),
        ("apple", "ep-0", "label"),
        ("apple", "ep-3", None),
    ]
    assert episodes[3]["detail"] == (
        "label must be one of success, fracture, fail, unknown, got ['success']"
    )
    assert report["dataset"]["version"] == "2026-10-25"
    assert report["status"] == "incomplete"
    assert run("pg-t1", root, "--json", tmp_path / "r.json") == 1


def test_an_unlisted_file_alone_makes_the_run_incomplete(tmp_path):
    root = tmp_path / "d"
    write(root, lift_holds(1))
    trajectory = root / "trajectories" / "apple"
    (trajectory / "ep-1.parquet").write_bytes(
        (trajectory / "ep-0.parquet").read_bytes()
    )
    report, _ = score_dataset(root)
    assert (report["episodes"]["refused"], report["episodes"]["unlisted"]) == (0, 1)
    assert report["status"] == "incomplete"
    assert run("pg-t1", root, "--json", tmp_path / "r.json") == 1


def test_a_dataset_without_metadata_cannot_be_scored(tmp_path):
    with pytest.raises(DatasetUnreadable):
        score_dataset(tmp_path)


def test_floor_arm_baselines_are_scored_apart(tmp_path):
    arm = block(CARRY, CP, floor_arm="reflection")
    write(
        tmp_path,
        [
            episode("ep-0", put_down(), block(CARRY, CP)),
            episode("ep-1", put_down(), arm, object_type="banana"),
        ],
    )
    report, records = scored(tmp_path)
    assert report["tasks"][CARRY]["overall"]["n"] == 1
    assert report["floor_arms"]["reflection"][CARRY]["overall"]["n"] == 1
    assert records["ep-1"]["floor_arm"] == "reflection"
    assert "floor arm reflection" in report_text(report)


def test_the_table_states_the_population_it_scores(tmp_path):
    every = {"keep_labels": None, "keep_requires_lift": False}
    write(
        tmp_path,
        [
            episode(
                "ep-0", ramp("lift_hold", LH, 0.15), block("lift_hold", LH, **every)
            ),
            episode("ep-1", ramp("pick", PK, 0.15), block("pick", PK, **every)),
            episode("ep-2", ramp("pick", PK, 0.15), block("pick", PK)),
        ],
    )
    report = scored(tmp_path)[0]
    assert report["tasks"]["lift_hold"]["population"]["keep_labels"] is None
    assert report["tasks"]["pick"]["population"]["keep_labels"] == "mixed"
    lines = printed(report)
    assert "population every episode the generation run made" in lines
    assert "population episodes kept under different rules" in lines


def test_labels_kept_in_another_order_are_one_population(tmp_path):
    rows = ramp("lift_hold", LH, 0.15)
    reordered = block("lift_hold", LH, keep_labels=["fracture", "success", "fracture"])
    write(
        tmp_path,
        [
            episode("ep-0", rows, block("lift_hold", LH)),
            episode("ep-1", rows, reordered),
        ],
    )
    report = scored(tmp_path)[0]
    assert report["tasks"]["lift_hold"]["population"]["keep_labels"] == [
        "success",
        "fracture",
    ]
    assert "population episodes kept under different rules" not in printed(report)
    assert "kept: labels success, fracture;" in report_text(report)


def test_scores_are_pooled_over_object_types_and_given_per_type(tmp_path):
    write(
        tmp_path,
        [
            episode(
                "ep-0", ramp("lift_hold", LH, 0.15), block("lift_hold", LH), "apple"
            ),
            episode(
                "ep-1", ramp("lift_hold", LH, 0.03), block("lift_hold", LH), "banana"
            ),
            episode(
                "ep-2", ramp("lift_hold", LH, 0.15), block("lift_hold", LH), "banana"
            ),
        ],
    )
    entry = scored(tmp_path)[0]["tasks"]["lift_hold"]
    assert entry["n_object_types"] == 2
    assert entry["overall"]["B"]["lifted"]["of_all"] == pytest.approx(2 / 3)
    assert entry["by_object_type"]["banana"]["B"]["lifted"]["of_all"] == 0.5
    assert entry["by_object_type"]["apple"]["n"] == 1


def test_the_task_score_is_taken_over_the_decidable_episodes(tmp_path):
    # Planned with a 100-row hold, B's window does not fit the second recording.
    short = [*LHP[:4], 100, *LHP[5:]]
    shorter = episode("ep-1", put_back(schedule=short), block("lift_hold_place", short))
    write(
        tmp_path / "both",
        [episode("ep-0", put_back(), block("lift_hold_place", LHP)), shorter],
    )
    report, records = scored(tmp_path / "both")
    assert [records[u]["decidable"]["B"] for u in ("ep-0", "ep-1")] == [True, False]
    assert all(records[u]["placed_task"]["verdict"] for u in ("ep-0", "ep-1"))
    table = report["tasks"]["lift_hold_place"]["overall"]
    c, b = table["C"], table["B"]
    assert (b["lifted"]["of_all"], b["held"]["undecidable"]) == (1.0, 1)
    assert b["task_score"] == c["task_score"] == 1.0
    assert (b["task_score_n"], c["task_score_n"]) == (1, 2)
    lines = printed(report)
    # B's held: 1 of 1 attempted and 1 of the 1 decidable episode; C's: 2 of 2.
    assert "held 1/1 1.000 [0.207, 1.000] 1/1 1.000 [0.207, 1.000]" in lines
    # B's lifted row counts both episodes, its score only the decidable one.
    assert "lifted 2/2 1.000 [0.342, 1.000] 2/2 1.000 [0.342, 1.000]" in lines
    assert "task score 1.000 over 1 decidable" in lines
    assert "task score 1.000 over 2 decidable" in lines
    write(tmp_path / "short", [shorter])
    b = scored(tmp_path / "short")[0]["tasks"]["lift_hold_place"]["overall"]["B"]
    assert (b["task_score"], b["task_score_n"], b["task_score_why_null"]) == (
        None,
        0,
        "no decidable episode",
    )
    assert b["headline"] is None


def test_a_task_without_a_hold_reports_lifted_alone(tmp_path):
    write(
        tmp_path,
        [
            episode("ep-0", ramp("pick", PK, 0.15), block("pick", PK)),
            episode("ep-1", ramp("pick", PK, 0.03), block("pick", PK)),
        ],
    )
    report, records = scored(tmp_path)
    assert (records["ep-0"]["stages"], records["ep-0"]["decidable"]) == (
        ["lifted"],
        None,
    )
    assert "held" not in records["ep-0"]["C"] and "placed" not in records["ep-0"]["B"]
    for x in ("C", "B"):
        cells = report["tasks"]["pick"]["overall"][x]
        assert (
            cells["held"],
            cells["placed"],
            cells["task_score"],
            cells["task_score_n"],
        ) == (None, None, None, None)
        assert cells["task_score_why_null"] == "no put-down planned"
        assert (cells["headline"]["stage"], cells["headline"]["of_all"]) == (
            "lifted",
            0.5,
        )
        assert {k: v for k, v in cells["failures"]["counts"].items() if v} == {
            "success": 1,
            "not_lifted": 1,
        }
    lines = printed(report)
    assert lines.count("held not planned not planned") == 2
    assert lines.count("task score not planned") == 2


# --- the check against the generator's verdicts ---------------------------------------------


def checked(rows, task_block, tmp_path, uid="ep-0"):
    write(tmp_path, [episode(uid, rows, task_block)])
    report, records = scored(tmp_path)
    return report, records[uid]["check"]


def test_a_stored_agreement_record_is_compared_entry_by_entry(tmp_path):
    rows = ramp("search_lift_hold_2s", S2, 0.15)
    record = score(rows, block("search_lift_hold_2s", S2))["C"]["agreement"]
    stages = {"agreement": True, "lifted": True, "held": True}
    good = block("search_lift_hold_2s", S2, agreement=record, stages=stages)
    report, check = checked(rows, good, tmp_path / "good")
    assert check["exact"] == {"E1": True, "E2": True}
    assert check["explained"] == {"held": "other_hold_length"}
    assert report["check"]["exact"]["E1"] == {"applied": 1, "failed": 0}
    assert report["status"] == "ok"
    wrong = block("search_lift_hold_2s", S2, agreement=dict(record, first_up_frame=250))
    report, check = checked(rows, wrong, tmp_path / "wrong")
    assert check["exact"] == {"E1": False}
    assert check["failures"] == ["E1 agreement.first_up_frame: stored 250, scored 249"]
    assert report["check"]["exact"]["E1"] == {"applied": 1, "failed": 1}
    assert report["status"] == "check_failed"
    assert (
        "  ep-0          E1 agreement.first_up_frame: stored 250, scored 249"
        in report_text(report).splitlines()
    )
    # The verdict alone, without the record, is compared too.
    alone = block("search_lift_hold_2s", S2, stages={"agreement": False})
    report, check = checked(rows, alone, tmp_path / "alone")
    assert check["exact"] == {"E1": False}
    assert check["failures"] == ["E1 stages.agreement: stored False, scored True"]
    assert report["check"]["exact"]["E1"] == {"applied": 1, "failed": 1}


def test_a_stored_lift_without_a_twelve_centimetre_crossing_fails_the_check(tmp_path):
    rows = ramp("search_lift_hold", S4, 0.12)
    stages = {"lifted": True, "held": True}
    report, check = checked(
        rows, block("search_lift_hold", S4, stages=stages), tmp_path
    )
    assert check["exact"] == {"E2": False}
    assert report["status"] == "check_failed"


@pytest.mark.parametrize(
    ("rows", "task", "stages", "explained"),
    [
        (
            ramp("search_lift_hold", S4, 0.15),
            "search_lift_hold",
            {"lifted": None, "held": None},
            {"lifted": "lift_not_reached", "held": "held_not_judged_by_generator"},
        ),
        (
            ramp("search_lift_hold", S4, 0.15, dips=[(260, 10, 0.1)]),
            "search_lift_hold",
            {"lifted": False, "held": None},
            {"lifted": "crossing_after_lift_phase"},
        ),
        (
            ramp("search_lift_hold", S4, 0.15, dips=[(505, 5, 0.1)]),
            "search_lift_hold",
            {"lifted": True, "held": False},
            {"held": "window_vs_hold_phase"},
        ),
        (
            ramp("search_lift_hold_2s", S2, 0.15),
            "search_lift_hold_2s",
            {"agreement": True, "lifted": True, "held": True},
            {"held": "other_hold_length"},
        ),
        (
            retry(second_top=0.1),
            "lift_hold_retry",
            {"lifted": False, "held": None},
            {"lifted": "retry_whole_recording", "held": "retry_whole_recording"},
        ),
        (
            # The crossing is the last lift row, 269, itself.
            at_heights(
                ramp("search_lift_hold", S4, 0.0), lambda i: 0.04 if i < 269 else 0.16
            ),
            "search_lift_hold",
            {"lifted": False, "held": None},
            {
                "lifted": "below_threshold_at_lift_end",
                "held": "held_not_judged_by_generator",
            },
        ),
        (
            put_back(),
            "lift_hold_place",
            {"lifted": True, "held": True, "placed": False},
            {"placed": "put_back_phase_rule"},
        ),
    ],
)
def test_known_differences_are_explained(rows, task, stages, explained, tmp_path):
    report, check = checked(rows, block(task, SCHEDULES[task], stages=stages), tmp_path)
    assert check["explained"] == explained
    assert check["failures"] == []
    assert report["check"]["explained"] == dict(
        sorted(Counter(explained.values()).items())
    )
    assert report["status"] == "ok"


@pytest.mark.parametrize(
    ("stages", "explained", "failure"),
    [
        (
            {"lifted": False, "held": None, "carried": False, "placed": None},
            {
                "lifted": "below_threshold_at_lift_end",
                "held": "held_not_judged_by_generator",
                "placed": "generator_gate",
            },
            "stages.carried: stored False, scored True",
        ),
        (
            {"lifted": None, "held": False},
            {"lifted": "lift_not_reached"},
            "stages.held: stored False, scored True",
        ),
    ],
)
def test_a_difference_no_rule_explains_fails_the_check(
    stages, explained, failure, tmp_path
):
    root = tmp_path / "d"
    report, check = checked(put_down(), block(CARRY, CP, stages=stages), root)
    assert check["explained"] == explained
    assert check["failures"] == [f"{failure}, a difference no rule explains"]
    assert report["check"]["failures"] == [
        {"episode_uid": "ep-0", "failures": check["failures"]}
    ]
    assert (report["check"]["result"], report["status"]) == ("failed", "check_failed")
    assert run("pg-t1", root, "--json", tmp_path / "r.json") == 1


def test_the_put_down_record_and_verdicts_are_exact(tmp_path):
    rows = put_down()
    first = score(rows, block(CARRY, CP))
    stages = {
        "agreement": True,
        "lifted": True,
        "held": True,
        "carried": True,
        "placed": True,
    }
    place = dict(
        first["placed_task"]["facts"],
        final_yaw_change_rad=0.1,
        final_tilt_change_rad=None,
    )
    report, check = checked(
        rows, block(CARRY, CP, stages=stages, place=place), tmp_path / "a"
    )
    assert check["exact"] == {
        "E1": True,
        "E2": True,
        "E3": True,
        "E4": True,
        "E5": True,
    }
    assert report["check"]["result"] == "passed"
    place["first_touch_frame"] = 602
    stages["placed"] = False
    report, check = checked(
        rows, block(CARRY, CP, stages=stages, place=place), tmp_path / "b"
    )
    assert check["exact"] == {
        "E1": True,
        "E2": True,
        "E3": True,
        "E4": False,
        "E5": False,
    }
    gated = {"lifted": False, "held": None, "carried": None, "placed": None}
    report, check = checked(rows, block(CARRY, CP, stages=gated), tmp_path / "c")
    assert check["explained"] == {
        "lifted": "below_threshold_at_lift_end",
        "held": "held_not_judged_by_generator",
        "carried": "generator_gate",
        "placed": "generator_gate",
    }
    # A generator that gated the put-down stores it with carried and placed null.
    place = dict(first["placed_task"]["facts"], carried=None, placed=None)
    task = block(CARRY, CP, stages=gated, place=place)
    report, check = checked(rows, task, tmp_path / "d")
    assert check["exact"] == {"E4": True}


def test_another_generator_threshold_takes_the_exact_lift_checks_away(tmp_path):
    rows = ramp("search_lift_hold", S4, 0.11)
    stages = {"lifted": True, "held": True}
    task = block("search_lift_hold", S4, stages=stages, judge={"lift_success_m": 0.10})
    check = score(rows, task)["check"]
    assert check["exact"] == {}
    assert check["explained"] == {
        "lifted": "other_lift_threshold",
        "held": "other_lift_threshold",
    }
    # Under the 4 s put-down too, E3 needs the generator's threshold at 12 cm.
    stages = {"lifted": True, "held": False}
    task = block(CARRY, CP, stages=stages, judge={"lift_success_m": 0.10})
    report, check = checked(put_down(), task, tmp_path)
    assert check["exact"] == {}
    assert check["matched"] == ["lifted"]
    assert check["explained"] == {"held": "other_lift_threshold"}
    assert report["check"]["matched"] == 1
    assert report["check"]["explained"] == {"other_lift_threshold": 1}


# --- the command ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("small")
    write(
        root,
        [
            episode("ep-0", put_down(), block(CARRY, CP)),
            episode(
                "ep-1",
                ramp("search_lift_hold_2s", S2, 0.07),
                block("search_lift_hold_2s", S2),
            ),
        ],
    )
    return root


def run(*argv):
    with pytest.raises(SystemExit) as stopped:
        main(["benchmark", *map(str, argv)])
    return stopped.value.code


def test_the_command_prints_the_table_and_writes_the_report(
    small_dataset, tmp_path, capsys
):
    out = tmp_path / "reports" / "report.json"
    assert run("pg-t1", small_dataset, "--json", out) == 0
    text = capsys.readouterr().out
    assert "search_lift_hold_carry_place: 1 episode, 1 object type" in text
    assert "task score" in text and "status        ok" in text
    report = json.loads(out.read_text())
    assert report["status"] == "ok"
    lines = (out.parent / "report.episodes.jsonl").read_text().splitlines()
    assert [json.loads(line)["episode_uid"] for line in lines] == ["ep-0", "ep-1"]
    assert report == score_dataset(small_dataset)[0]


def test_the_table_is_printed_when_the_report_cannot_be_written(
    small_dataset, tmp_path, capsys
):
    blocked = tmp_path / "file"
    blocked.write_text("")
    assert run("pg-t1", small_dataset, "--json", blocked / "report.json") == 2
    out, err = capsys.readouterr()
    assert "status        ok" in out
    assert "cannot write the report" in err


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            "format: physgraspgen-public\nformat_version: '1.0'\n",
            (
                "the dataset is 'physgraspgen-public' '1.0'; the scorer reads "
                "'physgraspgen-public' '2.0'"
            ),
        ),
        (
            "format: physgraspgen-public\nformat_version: 2.0\n",
            "the dataset is 'physgraspgen-public' 2.0; the scorer reads",
        ),
        ("- physgraspgen-public\n", "metadata.yaml is not a mapping"),
    ],
)
def test_a_dataset_of_another_format_cannot_be_scored(text, message, tmp_path, capsys):
    (tmp_path / "metadata.yaml").write_text(text)
    assert run("pg-t1", tmp_path, "--json", tmp_path / "r.json") == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "r.json").exists()


def test_the_command_exits_by_the_status(tmp_path, capsys):
    assert run("pg-t1", tmp_path / "missing", "--json", tmp_path / "r.json") == 2
    assert "metadata.yaml" in capsys.readouterr().err
    write(
        tmp_path / "d",
        [episode("ep-0", ramp("lift_hold", LH, 0.15), {"name": "lift_hold"})],
    )
    assert run("pg-t1", tmp_path / "d", "--json", tmp_path / "r.json") == 1


def test_the_reference_pipeline_still_runs_without_a_subcommand(capsys):
    main(["--fracture-threshold", "0.25", "--confidence-threshold", "0.5"])
    assert json.loads(capsys.readouterr().out)["training_samples"] > 0


def test_the_module_runs_the_command(small_dataset, tmp_path):
    out = tmp_path / "report.json"
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "openphysgraspgen",
            "benchmark",
            "pg-t1",
            small_dataset,
            "--json",
            out,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(out.read_text())["benchmark"] == "PG-T1"
