# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
"""Check a physgraspgen-public 2.0 dataset against its four test items.

Every dataset carries this file as ``tools/check_dataset.py``, beside the acceptance
values it applies, ``tools/check_criteria.yaml``. It needs Python 3.10 or newer, NumPy,
PyArrow and PyYAML, and nothing else, so it runs where the dataset was downloaded::

    python tools/check_dataset.py <dataset root> --seed <the tester's seed>
    python -I tools/check_dataset.py <root> --seed 20261116 --json report.json

The items:

    scale        every distinct frame of every episode whose files are all clean
    metadata     metadata.yaml against the files: counts, object types, channels, each
                 file's size and sha256, the published tools, SHA256SUMS, and the
                 specification read from it (channels, cameras, licence, DOI, URL)
    consistency  every frame's layers recorded at its own simulation step, and, on a
                 sample drawn with the tester's seed, pressed contact points projected
                 into each camera landing on the object's mask
    diversity    the object types, and a distribution table per variable of the facts

Every data file is hashed before any is opened, and only a file whose sha256 is the one
metadata.yaml states is read; a file read is read in full, and every rule of the format
applied to every value. Video frames are counted from the MP4 index, without a decoder.
A run passes only with the script and the criteria file metadata.yaml lists. It prints
a table and a ``RESULT {...}`` line, writes the whole report as JSON with ``--json``,
and exits 0 when every item run passed, 1 when the run failed, and 2 when the check
could not run.

The same file defines the format: ``openphysgraspgen.dataset`` imports its names, value
rules and file rules, so the writer, finalize and the check share one definition. The
format and the check are documented in ``docs/dataset/format.md``.
"""

import sys

if (
    __name__ == "__main__"
    and sys.path
    and not sys.flags.isolated
    and not getattr(sys.flags, "safe_path", False)
):
    # A script's folder comes first on the path, so a module in tools/ beside this file
    # would be imported in place of the standard library's or NumPy's.
    del sys.path[0]

if sys.version_info < (3, 10):  # noqa: UP036, the file runs on any Python
    _OLD_PYTHON = (
        f"the check needs Python 3.10 or newer; this is {sys.version.split()[0]}"
    )
    if __name__ == "__main__":
        import json

        print(f"check_dataset: {_OLD_PYTHON}", file=sys.stderr)
        print(
            "RESULT " + json.dumps({"pass": False, "reason": _OLD_PYTHON}), flush=True
        )
        sys.exit(2)
    raise ImportError(_OLD_PYTHON)

import argparse
import errno
import hashlib
import heapq
import itertools
import json
import math
import os
import platform
import re
import struct
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:
    np = pa = pq = None
    _IMPORT_ERROR: ImportError | None = exc
else:
    _IMPORT_ERROR = None

FORMAT = "physgraspgen-public"
FORMAT_VERSION = "2.0"
METADATA_NAME = "metadata.yaml"
CHECKSUMS_NAME = "SHA256SUMS"
DATA_DIR = "data"
TRAJECTORY_DIR = "trajectories"
VIDEO_DIR = "videos"
TOOLS_DIR = "tools"
PARQUET_SUFFIX = ".parquet"
VIDEO_SUFFIX = ".mp4"
SCRIPT_NAME = "check_dataset.py"
CRITERIA_NAME = "check_criteria.yaml"
# The tools every dataset publishes: this script and the acceptance values it applies.
TOOL_FILES = (f"{TOOLS_DIR}/{SCRIPT_NAME}", f"{TOOLS_DIR}/{CRITERIA_NAME}")
# The footer key under which both Parquet files of an episode carry its metadata.
METADATA_KEY = "physgraspgen"

CAMERAS = ("front", "wrist")
CHANNELS = ("rgb", "depth", "force", "stress", "deformation_gradient", "segmentation")
CAMERA_CHANNELS = ("rgb", "depth", "segmentation")
SEGMENTATION_CLASSES = ("background", "object", "arm", "gripper", "support", "prop")
LABELS = ("success", "fracture", "fail", "unknown")
# The kinds of an episode's facts that the diversity tables read.
FACT_KINDS = ("material", "placement", "lighting", "camera")
FILE_KINDS = ("frames", "trajectory", "video")
# The keys of the metadata both Parquet files of an episode carry under METADATA_KEY.
EPISODE_KEYS = (
    "format",
    "format_version",
    "dataset_name",
    "dataset_version",
    "license",
    "episode_uid",
    "object_type",
    "object_asset",
    "label",
    "n_frames",
    "n_steps",
    "n_tets",
    "fps",
    "recorded_fps",
    "channels",
    "cameras",
    "image_size",
    "segmentation_classes",
    "video",
    "ffmpeg_version",
    "video_sha256",
    "facts",
    "task",
    "header",
    "generation_provenance",
    "export_provenance",
    "source_n_frames",
    "source_cut_at_row",
)
# Object types, episode ids and camera names become path components, which every file
# system a copy lands on must hold: no trailing dot, no device name of Windows, and a
# length that leaves room for the rest of the path.
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
MAX_NAME_LENGTH = 100
RESERVED_NAMES = frozenset(
    ("con", "prn", "aux", "nul")
    + tuple(f"com{i}" for i in range(1, 10))
    + tuple(f"lpt{i}" for i in range(1, 10))
)
# metadata.yaml's doi is a bare DOI whose registrant code is 4 to 9 digits, as Zenodo's
# is, and its download_url an https URL, in printable ASCII.
DOI_PATTERN = re.compile(r"10\.[0-9]{4,9}/[!-~]+")
URL_PATTERN = re.compile(r"https://[!-~]+")

# A column is (dtype, shape). The shape is () for one value, a tuple of sizes for a
# fixed-size list of that many values stored row-major, "image" for one value per
# pixel of the camera's image, ("tets", k) for k values per tetrahedron and "contacts"
# for one struct per contact. Names with "{camera}" are repeated for every camera.
FRAME_COLUMNS = {
    "episode_uid": ("string", ()),
    "frame_index": ("int32", ()),
    "sim_step": ("int64", ()),
    "sim_time_s": ("float64", ()),
    "source_frame_index": ("int32", ()),
}
CHANNEL_COLUMNS = {
    "rgb": {
        "rgb.{camera}.sim_step": ("int64", ()),
        "rgb.{camera}.video_frame": ("int32", ()),
    },
    "depth": {
        "depth.{camera}.sim_step": ("int64", ()),
        "depth.{camera}.depth_m": ("float32", "image"),
    },
    "segmentation": {
        "segmentation.{camera}.sim_step": ("int64", ()),
        "segmentation.{camera}.mask": ("uint8", "image"),
    },
    "force": {
        "force.sim_step": ("int64", ()),
        "force.n_contacts": ("int32", ()),
        "force.contacts": ("struct", "contacts"),
        "force.gripper_net_force": ("float32", (3,)),
        "force.ground_net_force": ("float32", (3,)),
        "force.gripper_force_mag": ("float64", ()),
        "force.ground_force_mag": ("float64", ()),
    },
    "stress": {
        "stress.sim_step": ("int64", ()),
        "stress.von_mises": ("float32", ("tets", 1)),
        "stress.valid": ("bool", ("tets", 1)),
        "stress.volume_ratio": ("float32", ("tets", 1)),
    },
    "deformation_gradient": {
        "deformation_gradient.sim_step": ("int64", ()),
        "deformation_gradient.F": ("float32", ("tets", 9)),
    },
}
CAMERA_COLUMNS = {
    "camera.{camera}.intrinsics": ("float64", (3, 3)),
    "camera.{camera}.world_to_camera": ("float64", (4, 4)),
}
STATE_COLUMNS = {
    "robot.ee_pose_cmd": ("float64", (7,)),
    "robot.ee_pose": ("float64", (7,)),
    "robot.arm_q_setpoint": ("float64", (6,)),
    "robot.arm_q": ("float64", (6,)),
    "robot.gripper_width_cmd_m": ("float64", ()),
    "robot.gripper_width_setpoint_m": ("float64", ()),
    "robot.gripper_width_m": ("float64", ()),
    "robot.grip_contact_width_m": ("float64", ()),
    "robot.waypoint": ("int32", ()),
    "robot.phase": ("string", ()),
    "object.centroid": ("float64", (3,)),
    "object.bbox_min": ("float64", (3,)),
    "object.bbox_max": ("float64", (3,)),
    "object.n_soft_contacts": ("int32", ()),
    "object.n_rigid_contacts": ("int32", ()),
    "object.n_robot_contacts": ("int32", ()),
}
CONTACT_FIELDS = {
    "force": ("float32", (3,)),
    "point_world": ("float32", (3,)),
    "particle": ("int32", ()),
    "is_ground": ("bool", ()),
    "is_robot": ("bool", ()),
}
TRAJECTORY_COLUMNS = {
    "episode_uid": ("string", ()),
    "sim_step": ("int64", ()),
    "sim_time_s": ("float64", ()),
    "finite": ("bool", ()),
    **{
        name: STATE_COLUMNS[name]
        for name in (
            "robot.waypoint",
            "robot.phase",
            "robot.ee_pose_cmd",
            "robot.ee_pose",
            "robot.arm_q_setpoint",
            "robot.arm_q",
            "robot.gripper_width_cmd_m",
            "robot.gripper_width_setpoint_m",
            "robot.gripper_width_m",
            "robot.grip_contact_width_m",
            "object.centroid",
            "object.bbox_min",
            "object.bbox_max",
            "object.n_robot_contacts",
        )
    },
    "force.gripper_force_mag": ("float64", ()),
    "force.ground_force_mag": ("float64", ()),
}
# The columns a frame shares with the trajectory: at the frame's step they hold the
# same values. The force magnitudes are in a frames file with the force channel.
SHARED_COLUMNS = tuple(
    name
    for name in TRAJECTORY_COLUMNS
    if name in STATE_COLUMNS or name.startswith("force.")
)

# Value rules, which the functions below apply. Every float value of a column of
# FINITE_GROUPS, contact forces and points included, is finite, except where the names
# below allow NaN. A mask value indexes SEGMENTATION_CLASSES, robot.phase is a nonempty
# string, stress.von_mises is NaN exactly where stress.valid is false, and the camera
# matrices are a pinhole K and a rigid transform. Depth and stress.volume_ratio have no
# value rule.
FINITE_GROUPS = ("robot", "object", "camera", "force", "deformation_gradient")
# Finite or NaN: the jaw opening at first contact is NaN until the pads carry force.
MAY_BE_NAN = ("robot.grip_contact_width_m",)
# In the trajectory, NaN exactly where finite is false.
NOT_FINITE_COLUMNS = ("object.centroid", "object.bbox_min", "object.bbox_max")
# Reached robot state and force magnitudes that may also be NaN on such a step: the
# simulation can break down before they are recorded. Commands and setpoints may not.
UNSTABLE_MAY_BE_NAN = (
    "robot.ee_pose",
    "robot.arm_q",
    "robot.gripper_width_m",
    "force.gripper_force_mag",
    "force.ground_force_mag",
)
# How far a world-to-camera rotation may be from orthonormal with determinant +1.
RIGID_TOLERANCE = 1e-6
# Arrow sizes a fixed-size list with an int32, so an image has fewer pixels.
MAX_PIXELS = 2**31 - 1
# JSON integers of the episode metadata fit int64, as Parquet and Arrow readers hold them.
INT64 = (-(2**63), 2**63 - 1)
# How many levels the episode metadata's objects and arrays may nest, the footer's own
# object the first: every Python then reads and walks a footer alike.
MAX_NESTING = 32

CONVENTIONS = {
    "units": "SI: metres, seconds, newtons, pascals, radians",
    "world_frame": "right-handed, z up",
    "quaternion": "x, y, z, w",
    "frames": (
        "a frame is one simulation step; sim_step counts steps at recorded_fps from "
        "the episode start, the first recorded step being 1; an episode's frames are "
        "consecutive multiples of recorded_fps / fps"
    ),
    "trajectory": (
        "one row per recorded step from step 1; object.centroid, object.bbox_min and "
        "object.bbox_max are NaN exactly where finite is false; a frame's robot and "
        "object values and force magnitudes are the trajectory's at its step"
    ),
    "video": (
        "video frame i is the episode's frame i; the video's own timestamps are "
        "i / fps from 0, not sim_time_s"
    ),
    "camera": (
        "pinhole; pixel (u, v), u the column and v the row, sees world point X where "
        "[u, v, 1] ~ K (R X + t) and [R t] is the top of world_to_camera; OpenCV axes "
        "(x right, y down, z forward); pixel centres at integer coordinates"
    ),
    "images": "row-major, row 0 at the top",
    "depth": "forward (z) depth in metres",
    "stress": (
        "von Mises stress per tetrahedron from the deformed mesh through the object's "
        "material model; NaN exactly where stress.valid is false"
    ),
    "deformation_gradient": (
        "per tetrahedron, row-major F[t, r, c], from the object's rest mesh turned by "
        "its placement yaw, so F is the identity at the unloaded start"
    ),
    "segmentation_classes": list(SEGMENTATION_CLASSES),
}

# The one video coding the format allows: AV1 in an av01 sample entry, 8-bit 4:2:0,
# encoded by ffmpeg's SVT-AV1 with settings in the ranges it applies as given.
VIDEO_CODEC = "av01"
VIDEO_BIT_DEPTH = 8
VIDEO_CHROMA = "4:2:0"
VIDEO_ENCODER = "libsvtav1"
VIDEO_CRF = (1, 63)
VIDEO_MIN_GOP = 2
VIDEO_PRESET = (0, 11)
VIDEO_RECORD = ("codec", "pix_fmt", "encoder", "crf", "gop", "preset")
# av1C's (chroma_subsampling_x, chroma_subsampling_y) by name.
CHROMA_SUBSAMPLING = {
    (1, 1): "4:2:0",
    (1, 0): "4:2:2",
    (0, 0): "4:4:4",
    (0, 1): "4:4:0",
}


def frame_columns(
    channels: tuple[str, ...], cameras: tuple[str, ...]
) -> dict[str, tuple[str, object]]:
    """Every column of a frames file with these channels and cameras, in order."""
    columns = dict(FRAME_COLUMNS)
    for channel in CHANNELS:
        if channel not in channels:
            continue
        for name, spec in CHANNEL_COLUMNS[channel].items():
            if "{camera}" in name:
                columns.update({name.format(camera=c): spec for c in cameras})
            else:
                columns[name] = spec
    if any(channel in channels for channel in CAMERA_CHANNELS):
        for name, spec in CAMERA_COLUMNS.items():
            columns.update({name.format(camera=c): spec for c in cameras})
    columns.update(STATE_COLUMNS)
    return columns


def column_camera(name: str) -> str | None:
    """The camera a per-camera column belongs to, or None."""
    parts = name.split(".")
    if len(parts) == 3 and parts[0] in (*CAMERA_CHANNELS, "camera"):
        return parts[1]
    return None


def arrow_type(
    dtype: str, shape: object, image_size: tuple[int, int] | None = None
) -> "pa.DataType":
    """The Arrow type of a column of this dtype and shape."""
    if shape == "contacts":
        return pa.list_(
            pa.struct(
                [
                    pa.field(n, arrow_type(d, s), nullable=False)
                    for n, (d, s) in CONTACT_FIELDS.items()
                ]
            )
        )
    value = pa.string() if dtype == "string" else pa.from_numpy_dtype(np.dtype(dtype))
    if shape == ():
        return value
    if shape == "image":
        if image_size is None:
            raise ValueError("an image column needs the camera's image size")
        height, width = image_size
        return pa.list_(value, height * width)
    if isinstance(shape, tuple) and len(shape) == 2 and shape[0] == "tets":
        return pa.list_(value)
    if isinstance(shape, tuple) and all(isinstance(n, int) for n in shape):
        return pa.list_(value, math.prod(shape))
    raise ValueError(f"unknown column shape {shape!r}")


def _shape_metadata(
    shape: object, image_size: tuple[int, int] | None
) -> dict[str, str] | None:
    if shape == "image":
        return {"shape": json.dumps(list(image_size or ()))}
    if isinstance(shape, tuple) and shape and shape[0] == "tets":
        return {"shape": json.dumps(["n_tets"] if shape[1] == 1 else ["n_tets", 3, 3])}
    if isinstance(shape, tuple) and shape:
        return {"shape": json.dumps(list(shape))}
    return None


def frame_schema(
    channels: tuple[str, ...],
    cameras: tuple[str, ...],
    image_size: Mapping[str, tuple[int, int]],
) -> "pa.Schema":
    """The Arrow schema of a frames file; image_size maps a camera to (height, width)."""
    fields = []
    for name, (dtype, shape) in frame_columns(channels, cameras).items():
        size = image_size[column_camera(name)] if shape == "image" else None
        fields.append(
            pa.field(
                name,
                arrow_type(dtype, shape, size),
                nullable=False,
                metadata=_shape_metadata(shape, size),
            )
        )
    return pa.schema(fields)


def trajectory_schema() -> "pa.Schema":
    """The Arrow schema of a trajectory file."""
    return pa.schema(
        [
            pa.field(
                name,
                arrow_type(dtype, shape),
                nullable=False,
                metadata=_shape_metadata(shape, None),
            )
            for name, (dtype, shape) in TRAJECTORY_COLUMNS.items()
        ]
    )


# --- values ---------------------------------------------------------------------------
#
# Each rule takes one frame's or one row's value; the rules that hold for every element
# also take a stack of rows at once, and then hold exactly when they hold for each row.


def finite_error(name: str, value: object) -> str | None:
    """Why a column's value breaks the rule that its floats are finite, or None."""
    array = np.asarray(value)
    if array.dtype.kind != "f" or name.split(".", 1)[0] not in FINITE_GROUPS:
        return None
    if name in MAY_BE_NAN:
        return "must be finite or NaN" if np.isinf(array).any() else None
    return None if np.isfinite(array).all() else "must be finite"


def text_error(value: object) -> str | None:
    """Why a string value, such as robot.phase, is not a nonempty string, or None."""
    return None if isinstance(value, str) and value else "must be a nonempty string"


def mask_error(mask: object) -> str | None:
    """Why a segmentation mask holds a value that is no class's index, or None."""
    mask = np.asarray(mask)
    if mask.size and mask.max() >= len(SEGMENTATION_CLASSES):
        return "holds a value outside the segmentation classes"
    return None


def stress_error(von_mises: object, valid: object) -> str | None:
    """Why stress.von_mises is not NaN exactly where stress.valid is false, or None."""
    von_mises = np.asarray(von_mises, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if np.isnan(von_mises[~valid]).all() and np.isfinite(von_mises[valid]).all():
        return None
    return "must be NaN exactly where stress.valid is false, and finite elsewhere"


def frame_steps_error(
    steps: Mapping[str, int], previous: int | None, fps: int, recorded_fps: int
) -> str | None:
    """Why a frame's steps break the step rules, or None.

    steps maps sim_step and each <layer>.sim_step column of the frame to its value, and
    previous is the previous frame's sim_step, None for the first frame. Frames are
    consecutive on the fps grid, and every layer is captured at the frame's step.
    """
    step = steps["sim_step"]
    stride = recorded_fps // fps
    if previous is None:
        if step < stride or step % stride:
            return (
                f"sim_step {step} is not on the {fps} fps grid, a positive multiple "
                f"of {stride}"
            )
    elif step != previous + stride:
        return (
            f"sim_step {step} does not follow {previous} on the {fps} fps grid, "
            f"which goes on at {previous + stride}"
        )
    for name, value in steps.items():
        if value != step:
            return f"{name.rsplit('.', 1)[0]} was captured at another step"
    return None


def trajectory_step_error(step: int, previous: int | None) -> str | None:
    """Why a trajectory row's sim_step does not follow the previous row's, or None:
    the trajectory holds every recorded step from step 1."""
    expected = 1 if previous is None else previous + 1
    if step != expected:
        return (
            f"sim_step {step} is not {expected}; the trajectory holds every recorded "
            "step from step 1"
        )
    return None


def trajectory_values_error(row: Mapping[str, object], finite: bool) -> str | None:
    """Why a trajectory row's values break the value rules, or None; finite is the
    row's finite flag."""
    for name, value in row.items():
        if name in NOT_FINITE_COLUMNS:
            if finite and not np.isfinite(value).all():
                return f"{name} must be finite where finite is true"
            if not finite and not np.isnan(value).all():
                return f"{name} must be NaN where finite is false"
        elif not finite and name in UNSTABLE_MAY_BE_NAN:
            if np.isinf(np.asarray(value, dtype=np.float64)).any():
                return f"{name} must be finite or NaN where finite is false"
        else:
            problem = finite_error(name, value)
            if problem:
                return f"{name} {problem}"
    return None


def trajectory_row_error(row: Mapping[str, object], previous: int | None) -> str | None:
    """Why a trajectory row breaks the format's rules, or None.

    row maps every column to its value, and previous is the previous row's sim_step,
    None for the first row. The trajectory holds every recorded step from step 1.
    """
    return trajectory_step_error(row["sim_step"], previous) or trajectory_values_error(
        row, bool(row["finite"])
    )


def intrinsics_error(intrinsics: object) -> str | None:
    """Why a matrix is not a pinhole K [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], or None."""
    k = np.asarray(intrinsics, dtype=np.float64).reshape(-1, 3, 3)
    if not np.isfinite(k).all():
        return "is not finite"
    if (k[:, 0, 1] != 0).any() or (k[:, 1, 0] != 0).any():
        return "is not a pinhole K: K[0][1] (skew) and K[1][0] must be 0"
    if (k[:, 2] != (0.0, 0.0, 1.0)).any():
        return "is not a pinhole K: its last row must be 0 0 1"
    if not ((k[:, 0, 0] > 0) & (k[:, 1, 1] > 0)).all():
        return "is not a pinhole K: its focal lengths must be positive"
    return None


def world_to_camera_error(world_to_camera: object) -> str | None:
    """Why a matrix is not a rigid 4 x 4 transform, or None."""
    t = np.asarray(world_to_camera, dtype=np.float64).reshape(-1, 4, 4)
    if not np.isfinite(t).all():
        return "is not finite"
    if (t[:, 3] != (0.0, 0.0, 0.0, 1.0)).any():
        return "is not a rigid transform: its last row must be 0 0 0 1"
    r = t[:, :3, :3]
    deviation = np.abs(r @ r.transpose(0, 2, 1) - np.eye(3)).max(axis=(1, 2))
    bad = deviation > RIGID_TOLERANCE
    if bad.any():
        return (
            "is not a rigid transform: its rotation is not orthonormal, |R R^T - I| "
            f"reaches {deviation[bad][0]:.2g} (at most {RIGID_TOLERANCE:g})"
        )
    det = np.linalg.det(r)
    bad = np.abs(det - 1.0) > RIGID_TOLERANCE
    if bad.any():
        return (
            f"is not a rigid transform: its rotation's determinant is {det[bad][0]:.8g}, "
            f"not +1 (within {RIGID_TOLERANCE:g})"
        )
    return None


def _equal(a: object, b: object) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind == "f" and b.dtype.kind == "f":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def state_error(frame: Mapping[str, object], step: Mapping[str, object]) -> str | None:
    """Why a frame's values of SHARED_COLUMNS are not the trajectory's row at the
    frame's step, or None. NaN equals NaN; every other value must be equal."""
    for name, value in frame.items():
        if name in SHARED_COLUMNS and name in step and not _equal(value, step[name]):
            return f"{name} differs between the frame and the trajectory at this step"
    return None


def holds_a_value(value: object) -> bool:
    """Whether a value is a mapping with a leaf that is not null, at any depth."""
    stack = [value] if isinstance(value, Mapping) else []
    while stack:
        for item in stack.pop().values():
            if isinstance(item, Mapping):
                stack.append(item)
            elif item is not None:
                return True
    return False


def facts_error(facts: object) -> str | None:
    """Why an episode's facts lack a kind the diversity tables read, or None.

    Each of FACT_KINDS must be a mapping with at least one leaf that is not null.
    """
    if not isinstance(facts, Mapping):
        return "facts must be a mapping"
    for kind in FACT_KINDS:
        if not isinstance(facts.get(kind), Mapping):
            return f"facts has no {kind} mapping"
        if not holds_a_value(facts[kind]):
            return f"facts.{kind} holds no value but null"
    return None


# --- names ----------------------------------------------------------------------------


def name_error(name: object) -> str | None:
    """Why an object type, episode id or tool name cannot be a path component, or None."""
    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
        return f"must match {NAME_PATTERN.pattern}"
    if name.endswith("."):
        return "must not end in a dot, which Windows drops"
    if len(name) > MAX_NAME_LENGTH:
        return f"must be at most {MAX_NAME_LENGTH} characters"
    if name.split(".", 1)[0].lower() in RESERVED_NAMES:
        return "must not be a device name that Windows reserves (CON, NUL, COM1, ...)"
    return None


# --- episode metadata -----------------------------------------------------------------


def _clip(text: str, size: int = 160) -> str:
    return (
        text
        if len(text) <= size
        else f"{text[: size - 20]}... ({len(text):,} characters)"
    )


def _json_int(text: str) -> int:
    digits = text.removeprefix("-")
    value = int(text) if len(digits) <= 19 else None
    if value is None or not INT64[0] <= value <= INT64[1]:
        raise ValueError(f"{_clip(text, 40)} is out of the int64 range")
    return value


def _json_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{_clip(text, 40)} is out of the float64 range")
    return value


def _json_constant(text: str) -> float:
    raise ValueError(f"{text} is not a number")


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"the key {key!r} is repeated in an object")
        result[key] = value
    return result


def _nesting(text: str) -> int:
    """How many levels a JSON text's objects and arrays nest: its escapes are dropped,
    then its strings, and the brackets left are counted, each pass in linear time."""
    bare = re.sub(r'"[^"]*"', "", re.sub(r"\\.", "", text, flags=re.DOTALL))
    steps = (1 if c in "[{" else -1 for c in re.findall(r"[\[\]{}]", bare))
    return max(itertools.accumulate(steps), default=0)


def parse_footer(raw: bytes | str) -> dict[str, Any]:
    """An episode's metadata from its footer value: one JSON object in UTF-8 whose
    numbers are finite floats and int64 integers, whose objects repeat no key, and
    whose objects and arrays nest at most MAX_NESTING levels.

    Raises ValueError with the reason, which reads after "its metadata".
    """
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    except UnicodeDecodeError as exc:
        raise ValueError(f"is not UTF-8: {exc}") from None
    # Counted before parsing: how deep a parser goes depends on the Python.
    if _nesting(text) > MAX_NESTING:
        raise ValueError(f"is not JSON: it nests deeper than {MAX_NESTING} levels")
    try:
        value = json.loads(
            text,
            parse_constant=_json_constant,
            parse_float=_json_float,
            parse_int=_json_int,
            object_pairs_hook=_json_object,
        )
    except ValueError as exc:
        raise ValueError(f"is not JSON: {exc}") from None
    if not isinstance(value, dict):
        raise ValueError("is not a JSON object")  # noqa: TRY004
    return value


def _is_int(value: object, low: int | None = None) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and (low is None or value >= low)
    )


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _listing(value: object, names: tuple[str, ...]) -> bool:
    """Whether value is a list of distinct names from names, in their order."""
    return isinstance(value, list) and value == [n for n in names if n in value]


def video_record_error(record: object) -> str | None:
    """Why a video record is not the settings of an encoding the format allows, or None."""
    if not isinstance(record, Mapping) or sorted(record) != sorted(VIDEO_RECORD):
        return f"must hold exactly {', '.join(VIDEO_RECORD)}"
    if (record["codec"], record["pix_fmt"]) != ("av1", "yuv420p"):
        return "must be av1 in yuv420p"
    if record["encoder"] != VIDEO_ENCODER:
        return f"must be encoded by {VIDEO_ENCODER}"
    low, high = VIDEO_CRF
    if not (_is_int(record["crf"], low) and record["crf"] <= high):
        return f"crf must be an integer from {low} to {high}"
    if not _is_int(record["gop"], VIDEO_MIN_GOP):
        return f"gop must be an integer of at least {VIDEO_MIN_GOP}"
    low, high = VIDEO_PRESET
    if not (_is_int(record["preset"], low) and record["preset"] <= high):
        return f"preset must be an integer from {low} to {high}"
    return None


def footer_sizes(footer: Mapping[str, Any]) -> dict[str, tuple[int, int]] | None:
    """The (height, width) of each camera of a footer, or None when its channels,
    cameras or image sizes are not the format's."""
    channels, cameras, sizes = (
        footer.get("channels"),
        footer.get("cameras"),
        footer.get("image_size"),
    )
    if not _listing(channels, CHANNELS) or not _listing(cameras, CAMERAS):
        return None
    if bool(cameras) != any(c in channels for c in CAMERA_CHANNELS):
        return None
    if not isinstance(sizes, Mapping) or sorted(sizes) != sorted(cameras):
        return None
    out = {}
    for camera in cameras:
        size = sizes[camera]
        if not isinstance(size, Mapping) or sorted(size) != ["height", "width"]:
            return None
        height, width = size["height"], size["width"]
        if (
            not (_is_int(height, 1) and _is_int(width, 1))
            or height * width > MAX_PIXELS
        ):
            return None
        if "rgb" in channels and (height % 2 or width % 2):
            return None
        out[camera] = (height, width)
    return out


def footer_errors(
    footer: Mapping[str, Any],
    object_type: str,
    episode_uid: str,
    n_frames: int | None = None,
) -> list[str]:
    """Why an episode's metadata is not what the writer writes for the episode
    episode_uid of object_type, whose frames file holds n_frames rows when given."""
    errors = []
    missing = [key for key in EPISODE_KEYS if key not in footer]
    if missing:
        errors.append(f"metadata keys {missing} are missing")
    extra = sorted(key for key in footer if key not in EPISODE_KEYS)
    if extra:
        errors.append(f"metadata keys {extra} are not in the format")

    def bad(key: str, requirement: str) -> None:
        errors.append(
            f"metadata {key} is {_clip(repr(footer[key]))}, not {requirement}"
        )

    get = footer.get
    if "format" in footer and get("format") != FORMAT:
        errors.append(f"format is {_clip(repr(get('format')))}, not {FORMAT!r}")
    if "format_version" in footer and get("format_version") != FORMAT_VERSION:
        errors.append(
            f"format_version is {_clip(repr(get('format_version')))}, "
            f"not {FORMAT_VERSION!r}"
        )
    if ("episode_uid" in footer or "object_type" in footer) and (
        get("episode_uid"),
        get("object_type"),
    ) != (episode_uid, object_type):
        errors.append(
            f"holds episode {_clip(repr(get('episode_uid')))} of "
            f"{_clip(repr(get('object_type')))}, not {episode_uid!r} of {object_type!r}"
        )
    for key in ("dataset_name", "dataset_version", "license", "object_asset"):
        if key in footer and not _is_text(footer[key]):
            bad(key, "a nonempty string")
    if "label" in footer and get("label") not in LABELS:
        bad("label", f"one of {', '.join(LABELS)}")
    if "n_frames" in footer:
        if not _is_int(get("n_frames"), 1):
            bad("n_frames", "an integer of at least 1")
        elif n_frames is not None and get("n_frames") != n_frames:
            errors.append(
                f"metadata n_frames is {get('n_frames')} but the file has "
                f"{n_frames} rows"
            )
    for key in ("n_steps", "n_tets", "fps", "recorded_fps"):
        if key in footer and not _is_int(footer[key], 1):
            bad(key, "an integer of at least 1")
    fps, recorded_fps = get("fps"), get("recorded_fps")
    if _is_int(fps, 1) and _is_int(recorded_fps, 1) and recorded_fps % fps:
        errors.append(f"metadata fps {fps} does not divide recorded_fps {recorded_fps}")

    channels = get("channels")
    valid_channels = _listing(channels, CHANNELS)
    if "channels" in footer and not valid_channels:
        bad("channels", f"a list of channels from {', '.join(CHANNELS)} in that order")
    cameras = get("cameras")
    valid_cameras = _listing(cameras, CAMERAS)
    if valid_channels and valid_cameras:
        valid_cameras = bool(cameras) == any(c in channels for c in CAMERA_CHANNELS)
    if "cameras" in footer and not valid_cameras:
        bad(
            "cameras",
            f"a list of cameras from {', '.join(CAMERAS)} in that order, not empty "
            "exactly when a camera channel (rgb, depth, segmentation) is listed",
        )
    rgb = valid_channels and "rgb" in channels
    if (
        valid_channels
        and valid_cameras
        and "image_size" in footer
        and footer_sizes(footer) is None
    ):
        bad(
            "image_size",
            "the height and width of each of its cameras, positive integers "
            f"{'and even ' if rgb else ''}of at most {MAX_PIXELS:,} pixels",
        )
    if valid_channels and "segmentation_classes" in footer:
        if "segmentation" in channels:
            if get("segmentation_classes") != list(SEGMENTATION_CLASSES):
                bad("segmentation_classes", repr(list(SEGMENTATION_CLASSES)))
        elif get("segmentation_classes") is not None:
            bad("segmentation_classes", "null without segmentation")
    if valid_channels and "video" in footer:
        if rgb:
            problem = video_record_error(get("video"))
            if problem:
                bad(
                    "video", f"the settings of an encoding the format allows: {problem}"
                )
        elif get("video") is not None:
            bad("video", "null without rgb")
    if valid_channels and "ffmpeg_version" in footer:
        if rgb and not _is_text(get("ffmpeg_version")):
            bad("ffmpeg_version", "a nonempty string with rgb")
        elif not rgb and get("ffmpeg_version") is not None:
            bad("ffmpeg_version", "null without rgb")
    if valid_channels and valid_cameras and "video_sha256" in footer:
        digests = get("video_sha256")
        if rgb and not (
            isinstance(digests, Mapping)
            and sorted(digests) == sorted(cameras)
            and all(
                isinstance(d, str) and re.fullmatch("[0-9a-f]{64}", d)
                for d in digests.values()
            )
        ):
            bad("video_sha256", "the sha256 of each camera's video, in lower-case hex")
        elif not rgb and digests is not None:
            bad("video_sha256", "null without rgb")
    if "facts" in footer:
        problem = facts_error(get("facts"))
        if problem:
            errors.append(f"metadata {problem}")
    for key in ("task", "header", "generation_provenance", "export_provenance"):
        if key in footer and not (footer[key] is None or isinstance(footer[key], dict)):
            bad(key, "a mapping or null")
    if "source_n_frames" in footer:
        floor = get("n_steps") if _is_int(get("n_steps"), 1) else 0
        value = get("source_n_frames")
        if value is not None and not _is_int(value, floor):
            bad("source_n_frames", f"null or an integer of at least n_steps, {floor}")
    if "source_cut_at_row" in footer and not _is_int(get("source_cut_at_row"), -1):
        bad("source_cut_at_row", "an integer of at least -1")
    return errors


# --- MP4 ------------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoTrack:
    n_frames: int
    timescale: int
    frame_durations: tuple[int, ...]
    codec: str
    width: int
    height: int
    # From the av1C box of an av01 sample entry, None for other codecs.
    bit_depth: int | None = None
    chroma: str | None = None
    # mdhd's duration and the sum of the samples' durations, in the track's timescale.
    duration: int | None = None
    samples_duration: int | None = None
    # mvhd's timescale, and the edit list as (segment duration, media time, rate, rate
    # fraction) entries, None without an edts box.
    movie_timescale: int | None = None
    edits: tuple[tuple[int, int, int, int], ...] | None = None
    # Whether a composition offset (ctts) shows a sample out of its stored order.
    reordered: bool = False

    @property
    def fps(self) -> float | None:
        if len(self.frame_durations) != 1 or self.frame_durations[0] <= 0:
            return None
        return self.timescale / self.frame_durations[0]


def video_errors(
    track: VideoTrack,
    size: tuple[int, int] | None = None,
    fps: int | None = None,
    n_frames: int | None = None,
) -> list[str]:
    """Why a video track is not a camera's video in the format.

    size is the camera's (height, width), fps the dataset's frame rate and n_frames the
    episode's; each is checked when given. Video frame i must be shown i-th, at i / fps.
    """
    errors = []
    if track.codec != VIDEO_CODEC:
        errors.append(f"is coded as {track.codec}, not AV1 ({VIDEO_CODEC})")
    elif (track.bit_depth, track.chroma) != (VIDEO_BIT_DEPTH, VIDEO_CHROMA):
        errors.append(
            f"is {track.bit_depth}-bit {track.chroma}, "
            f"not {VIDEO_BIT_DEPTH}-bit {VIDEO_CHROMA}"
        )
    if size is not None and (track.height, track.width) != tuple(size):
        height, width = size
        errors.append(
            f"is {track.height} x {track.width} pixels, not the camera's "
            f"{height} x {width}"
        )
    if fps is not None and not (
        len(track.frame_durations) == 1
        and track.frame_durations[0] > 0
        and track.timescale == fps * track.frame_durations[0]
    ):
        errors.append(f"does not run at a constant {fps} fps")
    if n_frames is not None and track.n_frames != n_frames:
        errors.append(f"holds {track.n_frames} frames, not {n_frames}")
    if track.duration != track.samples_duration:
        errors.append(
            f"its media lasts {track.duration} units, its frames "
            f"{track.samples_duration}"
        )
    if track.edits is not None and not (
        len(track.edits) == 1
        and track.edits[0][1:] == (0, 1, 0)
        and track.movie_timescale
        and track.duration is not None
        and track.edits[0][0] * track.timescale
        >= track.duration * track.movie_timescale
    ):
        errors.append("its edit list does not play every frame from the first")
    if track.reordered:
        errors.append("shows its frames out of their stored order (ctts)")
    return errors


def video_error(
    track: VideoTrack,
    size: tuple[int, int] | None = None,
    fps: int | None = None,
    n_frames: int | None = None,
) -> str | None:
    """The first of video_errors, or None."""
    errors = video_errors(track, size, fps, n_frames)
    return errors[0] if errors else None


def _boxes(data: bytes, start: int, end: int):
    pos = start
    while pos < end:
        if pos + 8 > end:
            raise ValueError("truncated box header")
        size, kind = struct.unpack_from(">I4s", data, pos)
        header = 8
        if size == 1:
            if pos + 16 > end:
                raise ValueError("truncated box header")
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            header = 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            raise ValueError(f"box {kind!r} runs past its parent")
        yield kind, pos + header, pos + size
        pos += size


def _child(data: bytes, start: int, end: int, kind: bytes) -> tuple[int, int] | None:
    for found, body, stop in _boxes(data, start, end):
        if found == kind:
            return body, stop
    return None


def _unpack(form: str, data: bytes, at: int, end: int, box: str) -> tuple:
    """A box's fields at offset at, refused when they run past the box's end."""
    if at + struct.calcsize(form) > end:
        raise ValueError(f"a truncated {box} box")
    return struct.unpack_from(form, data, at)


def _read_moov(path: Path) -> bytes:
    moov = None
    with open(path, "rb") as stream:
        end = stream.seek(0, 2)
        pos = 0
        while pos < end:
            stream.seek(pos)
            head = stream.read(16)
            if len(head) < 8:
                raise ValueError("truncated box header")
            size, kind = struct.unpack_from(">I4s", head)
            if size == 1:
                if len(head) < 16:
                    raise ValueError("truncated box header")
                size = struct.unpack_from(">Q", head, 8)[0]
            elif size == 0:
                size = end - pos
            if size < 8 or pos + size > end:
                raise ValueError(f"box {kind!r} runs past the end of the file")
            if kind == b"moof":
                raise ValueError("fragmented MP4 is not part of the format")
            if kind == b"moov":
                if moov is not None:
                    raise ValueError("two moov boxes")
                stream.seek(pos)
                moov = stream.read(size)
            pos += size
    if moov is None:
        raise ValueError("no moov box")
    return moov


def _times(moov: bytes, box: tuple[int, int], name: str) -> tuple[int, int]:
    """The timescale and duration of an mvhd or mdhd box, versions 0 and 1."""
    start, end = box
    (version,) = _unpack(">B", moov, start, end, name)
    if version == 1:
        return _unpack(">IQ", moov, start + 20, end, name)
    return _unpack(">II", moov, start + 12, end, name)


def _sample_entry(
    moov: bytes, stsd: tuple[int, int]
) -> tuple[str, int, int, int | None, str | None]:
    """A video track's codec, width, height, bit depth and chroma subsampling."""
    start, end = stsd
    (count,) = _unpack(">I", moov, start + 4, end, "stsd")
    if count != 1:
        raise ValueError(f"a video track with {count} sample descriptions")
    for kind, body, stop in _boxes(moov, start + 8, end):
        # A visual sample entry holds 78 bytes before its own boxes.
        if body + 78 > stop:
            raise ValueError("truncated visual sample entry")
        width, height = struct.unpack_from(">HH", moov, body + 24)
        if kind != b"av01":
            return kind.decode("latin-1"), width, height, None, None
        av1c = _child(moov, body + 78, stop, b"av1C")
        if av1c is None or av1c[0] + 3 > av1c[1]:
            raise ValueError("an av01 sample entry without an av1C box")
        bits = moov[av1c[0] + 2]
        high, twelve, mono, sub_x, sub_y = (bits >> s & 1 for s in (6, 5, 4, 3, 2))
        depth = (12 if twelve else 10) if high else 8
        chroma = "4:0:0" if mono else CHROMA_SUBSAMPLING[sub_x, sub_y]
        return "av01", width, height, depth, chroma
    raise ValueError("a video track without a sample entry")


def _edits(moov: bytes, edts: tuple[int, int]) -> tuple[tuple[int, int, int, int], ...]:
    elst = _child(moov, *edts, b"elst")
    if elst is None:
        return ()
    start, end = elst
    version, _, count = _unpack(">B3sI", moov, start, end, "elst")
    form = ">Qqhh" if version == 1 else ">Iihh"
    size = struct.calcsize(form)
    if start + 8 + count * size > end:
        raise ValueError("a truncated elst box")
    return tuple(
        struct.unpack_from(form, moov, start + 8 + i * size) for i in range(count)
    )


def _reordered(moov: bytes, ctts: tuple[int, int] | None) -> bool:
    if ctts is None:
        return False
    start, end = ctts
    (count,) = _unpack(">I", moov, start + 4, end, "ctts")
    if start + 8 + 8 * count > end:
        raise ValueError("a truncated ctts box")
    return any(
        struct.unpack_from(">I", moov, start + 12 + 8 * i)[0] for i in range(count)
    )


def video_tracks(path: str | Path) -> list[VideoTrack]:
    """The video tracks of an MP4 file, read from its index without decoding.

    Every field is read inside its box; an index that is not whole, or not one the
    format can describe, raises ValueError.
    """
    moov = _read_moov(Path(path))
    header = 16 if struct.unpack_from(">I", moov)[0] == 1 else 8
    mvhd = _child(moov, header, len(moov), b"mvhd")
    if mvhd is None:
        raise ValueError("a moov box without mvhd")
    movie_timescale, _ = _times(moov, mvhd, "mvhd")
    tracks = []
    for kind, start, end in _boxes(moov, header, len(moov)):
        if kind == b"mvex":
            raise ValueError("fragmented MP4 is not part of the format")
        if kind != b"trak":
            continue
        mdia = _child(moov, start, end, b"mdia")
        if mdia is None:
            raise ValueError("a track without mdia")
        hdlr = _child(moov, *mdia, b"hdlr")
        if hdlr is None or hdlr[0] + 12 > hdlr[1]:
            raise ValueError("a track without a handler")
        if moov[hdlr[0] + 8 : hdlr[0] + 12] != b"vide":
            continue
        mdhd = _child(moov, *mdia, b"mdhd")
        minf = _child(moov, *mdia, b"minf")
        stbl = None if minf is None else _child(moov, *minf, b"stbl")
        if mdhd is None or stbl is None:
            raise ValueError("a video track without mdhd or stbl")
        timescale, duration = _times(moov, mdhd, "mdhd")
        stsd = _child(moov, *stbl, b"stsd")
        if stsd is None:
            raise ValueError("a video track without a sample description")
        codec, width, height, depth, chroma = _sample_entry(moov, stsd)
        sizes = _child(moov, *stbl, b"stsz") or _child(moov, *stbl, b"stz2")
        if sizes is None:
            raise ValueError("a video track without a sample size table")
        (n_frames,) = _unpack(">I", moov, sizes[0] + 8, sizes[1], "sample size")
        stts = _child(moov, *stbl, b"stts")
        if stts is None:
            raise ValueError("a video track without a time-to-sample table")
        (n_entries,) = _unpack(">I", moov, stts[0] + 4, stts[1], "stts")
        if stts[0] + 8 + 8 * n_entries > stts[1]:
            raise ValueError("truncated time-to-sample table")
        durations = []
        counted = total = 0
        for i in range(n_entries):
            count, delta = struct.unpack_from(">II", moov, stts[0] + 8 + 8 * i)
            counted += count
            total += count * delta
            if delta not in durations:
                durations.append(delta)
        if counted != n_frames:
            raise ValueError("the time-to-sample table and the sample count disagree")
        edts = _child(moov, start, end, b"edts")
        tracks.append(
            VideoTrack(
                n_frames,
                timescale,
                tuple(durations),
                codec,
                width,
                height,
                depth,
                chroma,
                duration,
                total,
                movie_timescale,
                None if edts is None else _edits(moov, edts),
                _reordered(moov, _child(moov, *stbl, b"ctts")),
            )
        )
    return tracks


def sha256_file(path: str | Path) -> tuple[str, int]:
    """The hex sha256 of a file's bytes, read in 4 MiB chunks, and how many it read."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as stream:
        while chunk := stream.read(1 << 22):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


# --- the check ------------------------------------------------------------------------

# The items of the test, in the order every run reports them.
ITEMS = ("scale", "metadata", "consistency", "diversity")
# How many entries of a listing the table prints; the JSON report holds them all.
MAX_LISTED = 50
# How many failed frames of the step part the JSON report lists; its counts are whole.
MAX_FRAMES_LISTED = 1000
# About how many bytes of values a frames file is read in at a time: decoding a whole
# row group of images at 480 x 640 holds over a gigabyte.
BYTES_PER_READ = 1 << 24
# A sha256sum line, in text (<hash>  <path>) or binary (<hash> *<path>) mode.
CHECKSUM_LINE = re.compile(r"([0-9a-fA-F]{64}) [ *](.+)")
LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"
# The oldest versions the tests run on.
MIN_VERSIONS = (("NumPy", (1, 26)), ("PyArrow", (17, 0)), ("PyYAML", (6, 0)))
PROGRESS_SECONDS = 30.0
# Where the format puts each kind of data file.
SHAPES = {
    DATA_DIR: ("frames", f"{DATA_DIR}/<object_type>/<episode_uid>{PARQUET_SUFFIX}"),
    TRAJECTORY_DIR: (
        "trajectory",
        f"{TRAJECTORY_DIR}/<object_type>/<episode_uid>{PARQUET_SUFFIX}",
    ),
    VIDEO_DIR: (
        "video",
        f"{VIDEO_DIR}/<camera>/<object_type>/<episode_uid>{VIDEO_SUFFIX}",
    ),
}


class CheckError(RuntimeError):
    """The check cannot run: it ends with a RESULT line and exit status 2."""


def _names(
    value: object, allowed: Iterable[str] | None = None, nonempty: bool = False
) -> bool:
    """Whether value is a list of distinct nonempty strings, from allowed when given."""
    return (
        isinstance(value, list)
        and (bool(value) or not nonempty)
        and all(isinstance(v, str) and v for v in value)
        and len(set(value)) == len(value)
        and (allowed is None or set(value) <= set(allowed))
    )


def _number(
    value: object, low: float, high: float | None = None, above: bool = False
) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        finite = math.isfinite(value)
    except OverflowError:  # an integer the check cannot apply as a float
        return False
    return (
        finite
        and (value > low if above else value >= low)
        and (high is None or value <= high)
    )


# What each item reads from the criteria file: key, test, and the requirement it states.
CRITERIA = {
    "scale": (("min_frames", lambda v: _is_int(v, 1), "an integer of at least 1"),),
    "metadata": (
        ("max_mismatches", lambda v: _is_int(v, 0), "an integer of at least 0"),
        (
            "required_channels",
            lambda v: _names(v, CHANNELS),
            f"a list of distinct channels from {', '.join(CHANNELS)}",
        ),
        (
            "required_cameras",
            lambda v: _names(v, CAMERAS, nonempty=True),
            f"a nonempty list of distinct cameras from {', '.join(CAMERAS)}",
        ),
        ("license", _is_text, "a nonempty string"),
    ),
    "consistency": (
        ("min_pass_fraction", lambda v: _number(v, 0, 1), "a number from 0 to 1"),
        (
            "layers",
            lambda v: _names(v, CHANNELS, nonempty=True),
            f"a nonempty list of distinct channels from {', '.join(CHANNELS)}",
        ),
        ("sample_frames", lambda v: _is_int(v, 1), "an integer of at least 1"),
        ("min_contact_frames", lambda v: _is_int(v, 0), "an integer of at least 0"),
        (
            "projection_radius_px",
            lambda v: _number(v, 0, above=True),
            "a finite number above 0",
        ),
        ("min_point_fraction", lambda v: _number(v, 0, 1), "a number from 0 to 1"),
        ("distance_reach_px", lambda v: _number(v, 0), "a finite number of at least 0"),
        ("projection_tolerance_provenance", _is_text, "a nonempty string"),
    ),
    "diversity": (
        ("min_object_types", lambda v: _is_int(v, 1), "an integer of at least 1"),
        ("required_fact_kinds", _names, "a list of distinct nonempty strings"),
        ("bins", lambda v: _is_int(v, 1), "an integer of at least 1"),
        (
            "log_bins_min_ratio",
            lambda v: _number(v, 1),
            "a finite number of at least 1",
        ),
        ("max_categories", lambda v: _is_int(v, 1), "an integer of at least 1"),
    ),
}


def validate_criteria(
    criteria: Mapping[str, Any], items: Iterable[str] = ITEMS
) -> None:
    """Refuse criteria that lack a table or a key an item reads, or hold a value the
    item cannot apply as it is: the values are never converted."""
    items = [item for item in ITEMS if item in items]
    for item in items:
        if not isinstance(criteria.get(item), Mapping):
            raise CheckError(f"the criteria have no [{item}] values")
        missing = [key for key, _, _ in CRITERIA[item] if key not in criteria[item]]
        if missing:
            raise CheckError(f"the criteria's [{item}] lacks {missing}")
    for item in items:
        for key, valid, requirement in CRITERIA[item]:
            value = criteria[item][key]
            if not valid(value):
                raise CheckError(
                    f"the criteria's [{item}] {key} is {_clip(repr(value))}, "
                    f"not {requirement}"
                )
    if "consistency" in items and "metadata" in criteria:
        # The cameras every frame is checked for never drop below these.
        floor = criteria["metadata"]
        if not isinstance(floor, Mapping):
            raise CheckError("the criteria's [metadata] is not a mapping")
        if "required_cameras" in floor:
            key, valid, requirement = CRITERIA["metadata"][2]
            if not valid(floor[key]):
                raise CheckError(
                    f"the criteria's [metadata] {key} is {_clip(repr(floor[key]))}, "
                    f"not {requirement}"
                )


def _required(exc: ImportError) -> str:
    """Why the check cannot run without a package it imports."""
    reason = (
        "NumPy, PyArrow and PyYAML are required: pip install numpy pyarrow pyyaml "
        f"({exc})"
    )
    if sys.flags.no_user_site:
        reason += (
            "; python -I and -s leave out packages that pip install --user installed: "
            "install them in a virtual environment, or run the script without -I or -s"
        )
    return reason


def _yaml() -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise CheckError(_required(exc)) from None
    return yaml


class _RepeatedKey(Exception):
    def __init__(self, key: object, line: int) -> None:
        super().__init__(key, line)
        self.key, self.line = key, line


def load_yaml(data: bytes, what: str) -> Any:
    """A YAML document from its bytes as PyYAML's safe loader reads it (its C loader when
    there is one), refusing a key repeated in a mapping: parsers disagree on which of
    the two values counts. what names the file in a refusal."""
    yaml = _yaml()

    class Loader(getattr(yaml, "CSafeLoader", yaml.SafeLoader)):
        def construct_mapping(self, node: Any, deep: bool = False) -> Any:
            seen = set()
            for key_node, _ in node.value:
                key = self.construct_object(key_node, deep=deep)
                try:
                    repeated = key in seen
                except TypeError:  # unhashable: the base loader refuses it
                    continue
                if repeated:
                    raise _RepeatedKey(key, key_node.start_mark.line + 1)
                seen.add(key)
            return super().construct_mapping(node, deep=deep)

    try:
        return yaml.load(data.decode("utf-8"), Loader=Loader)
    except _RepeatedKey as exc:
        raise CheckError(
            f"{what} repeats the key {exc.key!r} (line {exc.line})"
        ) from None
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise CheckError(f"{what} is not valid YAML: {exc}") from None


def load_criteria(path: str | Path) -> tuple[dict[str, Any], str]:
    """The acceptance values in a criteria file, and the sha256 of the bytes they were
    read from, so the published-tools check judges the file this run applied."""
    path = Path(path)
    if not path.is_file():
        raise CheckError(f"criteria file {path} does not exist (pass --criteria)")
    data = path.read_bytes()
    criteria = load_yaml(data, f"criteria file {path}")
    if not isinstance(criteria, dict):
        raise CheckError(f"criteria file {path} is not a YAML mapping")
    return criteria, hashlib.sha256(data).hexdigest()


def read_metadata(root: Path) -> tuple[dict[str, Any], str]:
    """metadata.yaml, and the sha256 of its bytes."""
    path = root / METADATA_NAME
    if not path.is_file():
        raise CheckError(
            f"{path} does not exist; the dataset root is the folder that holds it"
        )
    data = path.read_bytes()
    metadata = load_yaml(data, f"metadata {path}")
    if not isinstance(metadata, dict):
        raise CheckError(f"metadata {path} is not a YAML mapping")
    return metadata, hashlib.sha256(data).hexdigest()


# --- files ----------------------------------------------------------------------------


@dataclass(frozen=True)
class FileId:
    """What a data file's path says it is."""

    kind: str
    object_type: str
    episode_uid: str
    camera: str | None = None


@dataclass
class DataFile:
    """What the check keeps of one data file: never an open reader."""

    rel: str
    kind: str
    id: FileId | None = None
    size: int | None = None
    sha256: str | None = None
    # (size, mtime, ctime, inode, device) when it was hashed, to see a later change.
    stat: tuple[int, ...] | None = None
    opened: bool = False
    # A frames file's frames, a trajectory's steps, a video's samples.
    n_rows: int | None = None
    footer: dict[str, Any] | None = None
    raw_footer: bytes | None = None
    # The cameras' (height, width) when the footer's channels, cameras and sizes are
    # the format's.
    sizes: dict[str, tuple[int, int]] | None = None
    schema: Any = None
    track: VideoTrack | None = None
    errors: list[str] = field(default_factory=list)


@dataclass
class Episode:
    """A frames file and the files that belong to it."""

    frames: DataFile
    trajectory: DataFile | None = None
    videos: dict[str, DataFile] = field(default_factory=dict)
    # The cameras whose video its footer calls for, and its validated channels.
    expected: tuple[str, ...] = ()
    channels: tuple[str, ...] = ()

    def files(self) -> list[DataFile]:
        rest = [] if self.trajectory is None else [self.trajectory]
        return [self.frames, *rest, *self.videos.values()]

    @property
    def has_errors(self) -> bool:
        return any(rec.errors for rec in self.files())


def _shape(rel: str) -> tuple[str, str]:
    return SHAPES[rel.split("/", 1)[0]]


def classify(rel: str) -> tuple[FileId | None, str | None]:
    """A data file's identity from its path relative to the root, or None and why the
    path is not where the format puts a file of its folder."""
    parts = rel.split("/")
    kind, shape = _shape(rel)
    misplaced = f"is not where the format puts a {kind} file ({shape})"
    suffix = VIDEO_SUFFIX if kind == "video" else PARQUET_SUFFIX
    if len(parts) != (4 if kind == "video" else 3) or not parts[-1].endswith(suffix):
        return None, misplaced
    camera = parts[1] if kind == "video" else None
    object_type, uid = parts[-2], parts[-1][: -len(suffix)]
    if camera is not None and camera not in CAMERAS:
        return None, f"{misplaced}: its camera {camera!r} is not one of the format's"
    for what, name in (("object type", object_type), ("episode id", uid)):
        problem = name_error(name)
        if problem:
            return None, f"{misplaced}: its {what} {name!r} {problem}"
    return FileId(kind, object_type, uid, camera), None


def data_file(rel: str) -> DataFile:
    """The record of the data file at rel; a misplaced one starts with its error."""
    identity, problem = classify(rel)
    rec = DataFile(rel, identity.kind if identity else _shape(rel)[0], identity)
    if problem:
        rec.errors.append(problem)
    return rec


def _format_path(path: object, key: str) -> bool:
    """Whether a path metadata.yaml lists is one the format defines, so it can be
    joined to the root without leaving it."""
    if (
        not isinstance(path, str)
        or path.startswith("/")
        or "\\" in path
        or "\0" in path
    ):
        return False
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return False
    if key == "tools":
        return len(parts) == 2 and parts[0] == TOOLS_DIR and not name_error(parts[1])
    return parts[0] in SHAPES and classify(path)[0] is not None


def listed_entries(
    metadata: Mapping[str, Any], key: str
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """The valid entries of metadata.yaml's files or tools by path, the first of each
    path, and the problems of the list."""
    raw = metadata.get(key)
    if raw is None:
        return {}, []
    if not isinstance(raw, list):
        return {}, [f"{key} is not a list"]
    entries: dict[str, dict[str, Any]] = {}
    problems = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            problems.append(f"{key}[{i}] is not a mapping with a path")
        elif not _format_path(entry["path"], key):
            problems.append(
                f"{key}[{i}] path {_clip(repr(entry['path']))} is not a path the "
                "format defines"
            )
        elif entry["path"] in entries:
            problems.append(f"{key}: {entry['path']} is listed twice")
        else:
            entries[entry["path"]] = entry
    return entries, problems


def find_data_files(root: Path) -> tuple[list[str], list[str]]:
    """Every data file under the data folders, and what they hold that is not one: a
    file of another suffix, a hidden one, a dangling link, a linked folder (not
    followed). Paths are POSIX and relative to root, in path order."""
    found, ignored = [], []
    for top in (DATA_DIR, TRAJECTORY_DIR, VIDEO_DIR):
        suffix = VIDEO_SUFFIX if top == VIDEO_DIR else PARQUET_SUFFIX
        if not (root / top).is_dir():
            continue
        for directory, subdirs, names in os.walk(root / top):
            here = Path(directory)
            for name in subdirs:
                if (here / name).is_symlink():
                    ignored.append(f"{(here / name).relative_to(root).as_posix()}/")
            for name in names:
                rel = (here / name).relative_to(root).as_posix()
                hidden = any(part.startswith(".") for part in rel.split("/")[1:])
                if hidden or not name.endswith(suffix) or not (here / name).is_file():
                    ignored.append(rel)
                else:
                    found.append(rel)
    return sorted(found), sorted(ignored)


def _signature(path: Path) -> tuple[int, ...]:
    st = os.stat(path)
    return (st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino, st.st_dev)


def _resource(exc: BaseException) -> str | None:
    """What the machine ran out of, when exc says so: that is no finding on a file."""
    if isinstance(exc, MemoryError):
        return "memory"
    if isinstance(exc, OSError) and exc.errno in (errno.EMFILE, errno.ENFILE):
        return "file descriptors"
    if isinstance(exc, OSError) and exc.errno == errno.ENOMEM:
        return "memory"
    return None


def _failed(rec: DataFile, exc: Exception, what: str = "cannot be read") -> str:
    """What reading a file raised, as the file's finding; running out of memory or file
    descriptors stops the run instead."""
    resource = _resource(exc)
    if resource:
        raise CheckError(
            f"the check ran out of {resource} while reading {rec.rel}: {exc}"
        ) from exc
    return f"{what}: {type(exc).__name__}: {exc}"


class _Progress:
    """A line on stream at most every PROGRESS_SECONDS, and one when the work is done."""

    def __init__(self, stream: IO[str] | None, label: str, total: int) -> None:
        self.stream, self.label, self.total = stream, label, total
        self.start = self.last = time.monotonic()

    def __call__(self, done: int, extra: str = "") -> None:
        now = time.monotonic()
        if self.stream is None or not self.total:
            return
        if done < self.total and now - self.last < PROGRESS_SECONDS:
            return
        self.last = now
        print(
            f"{self.label}: {done:,}/{self.total:,} {extra}, {now - self.start:,.0f} s",
            file=self.stream,
            flush=True,
        )


def _pool(
    workers: int, jobs: Iterable[Any], run: Callable[[Any], Any]
) -> Iterator[Any]:
    """run(job) for every job in max(1, workers) threads, as each finishes."""
    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        futures = {pool.submit(run, job): job for job in jobs}
        for future in as_completed(futures):
            yield futures[future], future.result()
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()


def hash_files(
    root: Path, rels: list[str], workers: int, progress: IO[str] | None = None
) -> tuple[dict[str, str | None], dict[str, tuple[int, ...]]]:
    """The sha256 of each file, None for one that cannot be read, and each file's stat
    signature taken before it was hashed."""
    stats, sizes = {}, {}
    for rel in rels:
        try:
            stats[rel] = _signature(root / rel)
        except OSError:
            pass
        sizes[rel] = stats[rel][0] if rel in stats else 0
    total = sum(sizes.values())
    report = _Progress(progress, "hashing", len(rels))

    def one(rel: str) -> str | None:
        try:
            return sha256_file(root / rel)[0]
        except OSError as exc:
            if _resource(exc):
                raise CheckError(
                    f"the check ran out of {_resource(exc)} while hashing {rel}: {exc}"
                ) from exc
            return None

    hashes: dict[str, str | None] = {}
    done_bytes = 0
    for rel, digest in _pool(workers, rels, one):
        hashes[rel] = digest
        done_bytes += sizes[rel]
        report(len(hashes), f"files, {done_bytes / 1e9:,.1f}/{total / 1e9:,.1f} GB")
    return hashes, stats


def _lfs_hint(path: Path) -> str:
    try:
        with open(path, "rb") as stream:
            head = stream.read(100)
    except OSError:
        return ""
    if head.startswith(LFS_POINTER):
        return "; it is a Git LFS pointer, not the file (fetch it with git lfs pull)"
    return ""


# --- reading one file -----------------------------------------------------------------


class _Tally:
    """Per rule, the first frame or row that breaks it and how many do."""

    def __init__(self, unit: str) -> None:
        self.unit = unit
        self.found: dict[str, list[Any]] = {}

    def add(self, rule: str, index: int, problem: str, count: int = 1) -> None:
        entry = self.found.setdefault(rule, [index, problem, 0])
        entry[2] += count

    def judge(
        self,
        rule: str,
        start: int,
        check: Callable[[Any], str | None],
        batch: Any,
        rows: Iterable[Any],
    ) -> None:
        """check(batch); only when it fails, check each row for the first and the count."""
        if check(batch) is None:
            return
        first, count = None, 0
        for i, value in enumerate(rows):
            problem = check(value)
            if problem:
                count += 1
                if first is None:
                    first = (i, problem)
        if first is not None:
            self.add(rule, start + first[0], first[1], count)

    def errors(self, prefix: str = "") -> list[str]:
        out = []
        for index, problem, count in self.found.values():
            more = f" (and {count - 1:,} more {self.unit}s)" if count > 1 else ""
            out.append(f"{prefix}{self.unit} {index}: {problem}{more}")
        return out


def _parquet(path: Path) -> Any:
    return pq.ParquetFile(path, page_checksum_verification=True)


def _array(column: Any) -> Any:
    return column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column


def _fixed(array: Any) -> Any:
    """A fixed-size list array as an (n, k) NumPy array."""
    size = array.type.list_size
    return array.flatten().to_numpy(zero_copy_only=False).reshape(len(array), size)


def _offsets(array: Any) -> Any:
    offsets = np.asarray(array.offsets, dtype=np.int64)
    return offsets - offsets[0]


def _has_nulls(array: Any) -> bool:
    if array.null_count:
        return True
    kind = array.type
    if pa.types.is_struct(kind):
        return any(_has_nulls(child) for child in array.flatten())
    if pa.types.is_list(kind) or pa.types.is_fixed_size_list(kind):
        return _has_nulls(array.flatten())
    return False


def _footer(metadata: Any, rec: DataFile) -> dict[str, Any] | None:
    """F1: the episode metadata in a Parquet footer."""
    raw = (metadata.metadata or {}).get(METADATA_KEY.encode())
    if raw is None:
        rec.errors.append(f"carries no {METADATA_KEY} metadata")
        return None
    rec.raw_footer = raw
    try:
        rec.footer = parse_footer(raw)
    except ValueError as exc:
        rec.errors.append(f"its {METADATA_KEY} metadata {exc}")
    return rec.footer


def _field_metadata(field_: Any) -> dict[str, str]:
    return {
        k.decode(errors="replace"): v.decode(errors="replace")
        for k, v in (field_.metadata or {}).items()
    }


def schema_errors(schema: Any, expected: Any, stored: bool = True) -> list[str]:
    """Why a Parquet file's Arrow schema is not the format's: the columns, their order,
    types, nullability and field metadata. stored says whether the file holds the
    ARROW:schema entry, without which no reader sees the types the writer wrote."""
    errors = []
    if not stored:
        errors.append(
            "its footer holds no ARROW:schema entry, which states each column's type "
            "and shape"
        )
    names, wanted = schema.names, expected.names
    missing = sorted(n for n in wanted if n not in names)
    extra = sorted(n for n in set(names) if n not in wanted)
    repeated = sorted(n for n in set(names) if names.count(n) > 1)
    if missing:
        errors.append(f"columns {missing} are missing")
    if extra:
        errors.append(f"columns {extra} are not in the format")
    if repeated:
        errors.append(f"columns {repeated} are repeated")
    if not (missing or extra or repeated) and names != wanted:
        errors.append("its columns are not in the format's order")
    want = {f.name: f for f in expected}
    for found in schema:
        if found.name not in want or found.name in repeated:
            continue
        model = want[found.name]
        if not found.type.equals(model.type):
            errors.append(f"column {found.name} is {found.type}, not {model.type}")
        if found.nullable:
            errors.append(f"column {found.name} may hold nulls")
        if _field_metadata(found) != _field_metadata(model):
            errors.append(
                f"column {found.name} has the field metadata {_field_metadata(found)}, "
                f"not {_field_metadata(model)}"
            )
    return errors


def _stored(metadata: Any) -> bool:
    return b"ARROW:schema" in (metadata.metadata or {})


@dataclass
class _Frames:
    """What a frames file's rows give the rules of its trajectory."""

    steps: Any
    shared: dict[str, Any]


def read_frames_file(path: Path, rec: DataFile) -> _Frames | None:
    """Read a frames file in full and apply the format's rules to it: its footer, its
    columns, and every value of every frame. Findings go to rec.errors."""
    try:
        with _parquet(path) as parquet:
            metadata = parquet.metadata
            rec.n_rows = metadata.num_rows
            footer = _footer(metadata, rec)
            if footer is not None:
                rec.errors.extend(
                    footer_errors(
                        footer, rec.id.object_type, rec.id.episode_uid, rec.n_rows
                    )
                )
            if rec.n_rows < 1:
                rec.errors.append("holds no frame; an episode has at least one")
            if footer is None:
                return None
            rec.sizes = footer_sizes(footer)
            if rec.sizes is None:
                rec.errors.append(
                    "its columns cannot be checked: its channels, cameras or image "
                    "sizes are not valid"
                )
                return None
            rec.schema = parquet.schema_arrow
            expected = frame_schema(
                tuple(footer["channels"]), tuple(footer["cameras"]), rec.sizes
            )
            rec.errors.extend(schema_errors(rec.schema, expected, _stored(metadata)))
            if rec.errors:
                return None
            return _frame_rows(parquet, rec, footer)
    except Exception as exc:  # noqa: BLE001, what a damaged file raises is its finding
        rec.errors.append(_failed(rec, exc))
    return None


def frames_per_read(sizes: Mapping[str, tuple[int, int]], n_tets: int) -> int:
    """How many frames of a frames file hold about BYTES_PER_READ bytes of values: a
    pixel's depth and class take 5 bytes, a tetrahedron's values 45, the rest less than
    a kilobyte."""
    per_frame = 5 * sum(h * w for h, w in sizes.values()) + 45 * n_tets + 1024
    return max(1, BYTES_PER_READ // per_frame)


def _frame_rows(parquet: Any, rec: DataFile, footer: Mapping[str, Any]) -> _Frames:
    """The rules on every frame, frames_per_read frames at a time, with every column
    decoded and every page checksum the writer wrote verified."""
    uid, n_tets = footer["episode_uid"], footer["n_tets"]
    fps, recorded_fps = footer["fps"], footer["recorded_fps"]
    channels, cameras = footer["channels"], footer["cameras"]
    names = rec.schema.names
    stamps = ["sim_step"] + [n for n in names if n.endswith(".sim_step")]
    shared = {n: [] for n in SHARED_COLUMNS if n in names}
    tally = _Tally("frame")
    nulls: list[str] = []
    steps: list[Any] = []
    previous = None
    start = 0
    size = frames_per_read(rec.sizes, n_tets)
    for batch in parquet.iter_batches(batch_size=size):
        n = batch.num_rows
        col = dict(zip(batch.schema.names, batch.columns))
        nulls += [k for k, v in col.items() if k not in nulls and _has_nulls(v)]
        if nulls:
            start += n
            continue
        index = np.arange(start, start + n)

        values = col["episode_uid"].to_pylist()
        bad = [i for i, v in enumerate(values) if v != uid]
        if bad:
            tally.add(
                "episode_uid",
                start + bad[0],
                f"episode_uid is {values[bad[0]]!r}, not {uid!r}",
                len(bad),
            )
        frame_index = col["frame_index"].to_numpy(zero_copy_only=False)
        bad = np.flatnonzero(frame_index != index)
        if bad.size:
            i = int(bad[0])
            tally.add(
                "frame_index",
                start + i,
                f"frame_index is {frame_index[i]}, not {start + i}",
                bad.size,
            )
        stamp_values = {
            name: col[name].to_numpy(zero_copy_only=False).tolist() for name in stamps
        }
        for i in range(n):
            row = {name: stamp_values[name][i] for name in stamps}
            problem = frame_steps_error(row, previous, fps, recorded_fps)
            if problem:
                tally.add("steps", start + i, problem)
            previous = row["sim_step"]
        sim_step = col["sim_step"].to_numpy(zero_copy_only=False)
        sim_time = col["sim_time_s"].to_numpy(zero_copy_only=False)
        bad = np.flatnonzero(sim_time != sim_step / recorded_fps)
        if bad.size:
            i = int(bad[0])
            tally.add(
                "sim_time_s",
                start + i,
                f"sim_time_s is {float(sim_time[i])!r}, not sim_step / recorded_fps "
                f"({int(sim_step[i]) / recorded_fps!r})",
                bad.size,
            )
        if "rgb" in channels:
            for camera in cameras:
                name = f"rgb.{camera}.video_frame"
                video_frame = col[name].to_numpy(zero_copy_only=False)
                bad = np.flatnonzero(video_frame != index)
                if bad.size:
                    i = int(bad[0])
                    tally.add(
                        name,
                        start + i,
                        f"{name} is {video_frame[i]}, not the frame's index {start + i}",
                        bad.size,
                    )
        lists = {}
        for name, k in (
            ("stress.von_mises", n_tets),
            ("stress.valid", n_tets),
            ("stress.volume_ratio", n_tets),
            ("deformation_gradient.F", 9 * n_tets),
        ):
            if name in col:
                offsets = _offsets(col[name])
                lengths = np.diff(offsets)
                bad = np.flatnonzero(lengths != k)
                if bad.size:
                    i = int(bad[0])
                    tally.add(
                        f"length:{name}",
                        start + i,
                        f"{name} holds {lengths[i]} values, not n_tets x "
                        f"{k // n_tets} = {k}",
                        bad.size,
                    )
                else:
                    flat = col[name].flatten().to_numpy(zero_copy_only=False)
                    lists[name] = (offsets, flat)
        contacts = None
        if "force.contacts" in col:
            offsets = _offsets(col["force.contacts"])
            counts = col["force.n_contacts"].to_numpy(zero_copy_only=False)
            bad = np.flatnonzero(np.diff(offsets) != counts)
            if bad.size:
                i = int(bad[0])
                tally.add(
                    "force.n_contacts",
                    start + i,
                    f"force.n_contacts is {counts[i]}, not the "
                    f"{offsets[i + 1] - offsets[i]} entries of force.contacts",
                    bad.size,
                )
            children = dict(
                zip(
                    [f.name for f in col["force.contacts"].type.value_type],
                    col["force.contacts"].flatten().flatten(),
                )
            )
            contacts = (offsets, children)

        # The value rules: finite floats, mask classes, camera matrices, stress, text.
        for name in names:
            group_ = name.split(".", 1)[0]
            if group_ not in FINITE_GROUPS or name == "force.contacts":
                continue
            array = col[name]
            if name in lists:
                offsets, values = lists[name]
                _judge_list(tally, f"finite:{name}", start, name, offsets, values)
                continue
            if pa.types.is_fixed_size_list(array.type):
                values = _fixed(array)
            elif pa.types.is_floating(array.type):
                values = array.to_numpy(zero_copy_only=False)
            else:
                continue
            tally.judge(
                f"finite:{name}",
                start,
                _named(name, finite_error),
                values,
                values,
            )
        if contacts is not None:
            offsets, children = contacts
            for child in ("force", "point_world"):
                name = f"force.contacts.{child}"
                values = _fixed(children[child])
                _judge_list(tally, f"finite:{name}", start, name, offsets, values)
        for camera in cameras:
            for name, rule in (
                (f"segmentation.{camera}.mask", mask_error),
                (f"camera.{camera}.intrinsics", intrinsics_error),
                (f"camera.{camera}.world_to_camera", world_to_camera_error),
            ):
                if name in col:
                    values = _fixed(col[name])
                    tally.judge(
                        f"value:{name}",
                        start,
                        _prefixed(name, rule),
                        values,
                        values,
                    )
        if "stress.von_mises" in lists and "stress.valid" in lists:
            (offsets, von_mises), (_, valid) = (
                lists["stress.von_mises"],
                lists["stress.valid"],
            )
            tally.judge(
                "stress",
                start,
                lambda pair: _prefixed("stress.von_mises", stress_error)(*pair),
                (von_mises, valid),
                ((von_mises[a:b], valid[a:b]) for a, b in itertools.pairwise(offsets)),
            )
        phases = col["robot.phase"].to_pylist()
        bad = [i for i, v in enumerate(phases) if text_error(v)]
        if bad:
            tally.add(
                "robot.phase",
                start + bad[0],
                f"robot.phase {text_error(phases[bad[0]])}",
                len(bad),
            )

        steps.append(sim_step)
        for name, parts in shared.items():
            array = col[name]
            if pa.types.is_fixed_size_list(array.type):
                parts.append(_fixed(array))
            elif pa.types.is_string(array.type):
                parts.append(np.asarray(array.to_pylist(), dtype=object))
            else:
                parts.append(array.to_numpy(zero_copy_only=False))
        start += n
    rec.errors.extend(f"column {name} holds nulls" for name in nulls)
    rec.errors.extend(tally.errors())
    if nulls:
        return _Frames(np.zeros(0, dtype=np.int64), {})
    return _Frames(
        np.concatenate(steps) if steps else np.zeros(0, dtype=np.int64),
        {name: np.concatenate(parts) for name, parts in shared.items() if parts},
    )


def _named(
    name: str, rule: Callable[[str, Any], str | None]
) -> Callable[[Any], str | None]:
    def check(value: Any) -> str | None:
        problem = rule(name, value)
        return f"{name} {problem}" if problem else None

    return check


def _prefixed(name: str, rule: Callable[..., str | None]) -> Callable[..., str | None]:
    def check(*values: Any) -> str | None:
        problem = rule(*values)
        return f"{name} {problem}" if problem else None

    return check


def _judge_list(
    tally: _Tally, rule: str, start: int, name: str, offsets: Any, values: Any
) -> None:
    """finite_error on a column of one variable-length list per frame."""
    tally.judge(
        rule,
        start,
        _named(name, finite_error),
        values,
        (values[a:b] for a, b in itertools.pairwise(offsets)),
    )


def read_trajectory_file(
    path: Path, rec: DataFile, frames: DataFile | None, data: _Frames | None
) -> None:
    """Read a trajectory file in full and apply the format's rules: its footer is its
    frames file's, byte for byte; its columns; every row's values; every frame's step
    among its steps, with the frame's robot and object values.

    The rules on its rows apply when its footer, columns and step count are right and
    the footer's episode_uid and recorded_fps, which they read, are valid, whatever
    else the frames file's footer gets wrong: that is the frames file's finding."""
    try:
        with _parquet(path) as parquet:
            metadata = parquet.metadata
            rec.n_rows = rows = metadata.num_rows
            footer = _footer(metadata, rec)
            if footer is None:
                return
            if frames is not None and frames.raw_footer is not None:
                if rec.raw_footer != frames.raw_footer:
                    differ = ""
                    if frames.footer is not None:
                        keys = sorted(
                            k
                            for k in {*footer, *frames.footer}
                            if footer.get(k) != frames.footer.get(k)
                        )
                        differ = f" (it differs in {keys})" if keys else ""
                    rec.errors.append(
                        f"its {METADATA_KEY} metadata is not that of {frames.rel}{differ}"
                    )
            else:
                rec.errors.extend(
                    footer_errors(footer, rec.id.object_type, rec.id.episode_uid)
                )
            rec.schema = parquet.schema_arrow
            rec.errors.extend(
                schema_errors(rec.schema, trajectory_schema(), _stored(metadata))
            )
            n_steps = footer.get("n_steps")
            if rows != n_steps:
                rec.errors.append(f"holds {rows} steps, its metadata says {n_steps!r}")
            if (
                rec.errors
                or footer.get("episode_uid") != rec.id.episode_uid
                or not _is_int(footer.get("recorded_fps"), 1)
            ):
                return
            _trajectory_rows(parquet, rec, footer, frames, data)
    except Exception as exc:  # noqa: BLE001, what a damaged file raises is its finding
        rec.errors.append(_failed(rec, exc))


def _trajectory_rows(
    parquet: Any,
    rec: DataFile,
    footer: Mapping[str, Any],
    frames: DataFile | None,
    data: _Frames | None,
) -> None:
    uid, recorded_fps = footer["episode_uid"], footer["recorded_fps"]
    tally = _Tally("row")
    nulls: list[str] = []
    steps: list[Any] = []
    shared: dict[str, list[Any]] = {n: [] for n in SHARED_COLUMNS}
    previous = None
    start = 0
    for group in range(parquet.num_row_groups):
        table = parquet.read_row_group(group)
        n = table.num_rows
        col = {name: _array(table.column(name)) for name in table.column_names}
        nulls += [k for k, v in col.items() if k not in nulls and _has_nulls(v)]
        if nulls:
            start += n
            continue
        values = {}
        for name, array in col.items():
            if pa.types.is_fixed_size_list(array.type):
                values[name] = _fixed(array)
            elif pa.types.is_string(array.type):
                values[name] = np.asarray(array.to_pylist(), dtype=object)
            else:
                values[name] = array.to_numpy(zero_copy_only=False)
        bad = np.flatnonzero(values["episode_uid"] != uid)
        if bad.size:
            i = int(bad[0])
            tally.add(
                "episode_uid",
                start + i,
                f"episode_uid is {values['episode_uid'][i]!r}, not {uid!r}",
                bad.size,
            )
        sim_step = values["sim_step"]
        for i, step in enumerate(sim_step.tolist()):
            problem = trajectory_step_error(step, previous)
            if problem:
                tally.add("sim_step", start + i, problem)
            previous = step
        judged = {
            name: value
            for name, value in values.items()
            if name not in ("episode_uid", "sim_step", "sim_time_s", "finite")
        }
        failures = []
        for flag in (True, False):
            where = np.flatnonzero(values["finite"] == flag)
            batch = {name: value[where] for name, value in judged.items()}
            if not where.size or trajectory_values_error(batch, flag) is None:
                continue
            for i in where.tolist():
                problem = trajectory_values_error(
                    {name: value[i] for name, value in judged.items()}, flag
                )
                if problem:
                    failures.append((i, problem))
        if failures:
            failures.sort()
            tally.add("values", start + failures[0][0], failures[0][1], len(failures))
        sim_time = values["sim_time_s"]
        bad = np.flatnonzero(sim_time != sim_step / recorded_fps)
        if bad.size:
            i = int(bad[0])
            tally.add(
                "sim_time_s",
                start + i,
                f"sim_time_s is {float(sim_time[i])!r}, not sim_step / recorded_fps "
                f"({int(sim_step[i]) / recorded_fps!r})",
                bad.size,
            )
        phases = values["robot.phase"].tolist()
        bad = [i for i, v in enumerate(phases) if text_error(v)]
        if bad:
            tally.add(
                "robot.phase",
                start + bad[0],
                f"robot.phase {text_error(phases[bad[0]])}",
                len(bad),
            )
        steps.append(sim_step)
        for name, parts in shared.items():
            parts.append(values[name])
        start += n
    rec.errors.extend(f"column {name} holds nulls" for name in nulls)
    rec.errors.extend(tally.errors())
    if nulls or tally.found or frames is None or data is None or not data.steps.size:
        return
    missing = np.setdiff1d(data.steps, np.concatenate(steps))
    if missing.size:
        rec.errors.append(f"lacks the step {missing[0]} of a frame of {frames.rel}")
        return
    # The steps run from 1, so a frame's step is its row's number plus 1.
    rows = data.steps - 1
    mine = data.shared
    theirs = {name: np.concatenate(shared[name])[rows] for name in mine}
    if state_error(mine, theirs) is None:
        return
    first, count = None, 0
    for i in range(len(rows)):
        problem = state_error(
            {n: v[i] for n, v in mine.items()}, {n: v[i] for n, v in theirs.items()}
        )
        if problem:
            count += 1
            first = first or (i, problem)
    if first is not None:
        more = f" (and {count - 1:,} more frames)" if count > 1 else ""
        rec.errors.append(f"frame {first[0]} of {frames.rel}: {first[1]}{more}")


def read_video_file(path: Path, rec: DataFile, frames: DataFile | None) -> None:
    """Read a video's MP4 index, never a frame, and apply the format's rules: one AV1
    track, yuv420p, the camera's size, the episode's frame count at a constant fps,
    every frame shown in order, and the sha256 its episode's footer states.

    frames is the episode's frames file when it was read. Each rule that compares the
    video with its episode applies whenever the value it needs was read and is valid,
    whatever else the episode's footer gets wrong."""
    try:
        try:
            tracks = video_tracks(path)
        except ValueError as exc:
            rec.errors.append(f"its MP4 index cannot be read: {exc}")
            return
        if len(tracks) != 1:
            rec.errors.append(f"holds {len(tracks)} video tracks, not one")
            return
        rec.track = tracks[0]
        rec.n_rows = rec.track.n_frames
        camera = rec.id.camera
        footer = (frames.footer if frames is not None else None) or {}
        sizes = frames.sizes if frames is not None else None
        if sizes is not None and (
            "rgb" not in footer["channels"] or camera not in sizes
        ):
            # Not a video of its episode (link_episodes): its own rules alone.
            rec.errors.extend(video_errors(rec.track))
            return
        fps = footer.get("fps")
        rec.errors.extend(
            video_errors(
                rec.track,
                None if sizes is None else sizes[camera],
                fps if _is_int(fps, 1) else None,
                None if frames is None else frames.n_rows,
            )
        )
        digests = footer.get("video_sha256")
        stated = digests.get(camera) if isinstance(digests, Mapping) else None
        if (
            isinstance(stated, str)
            and re.fullmatch("[0-9a-f]{64}", stated)
            and (rec.sha256 or sha256_file(path)[0]) != stated
        ):
            rec.errors.append("is not the video its episode was written with")
    except Exception as exc:  # noqa: BLE001, what a damaged file raises is its finding
        rec.errors.append(_failed(rec, exc))


def _open_all(rec: DataFile) -> bool:
    rec.opened = rec.id is not None
    return rec.opened


def _read_unit(
    root: Path, unit: dict[str, Any], gate: Callable[[DataFile], bool]
) -> None:
    frames, trajectory, videos = unit["frames"], unit["trajectory"], unit["videos"]
    data = None
    if frames is not None and gate(frames):
        data = read_frames_file(root / frames.rel, frames)
    read = frames if frames is not None and frames.opened else None
    if trajectory is not None and gate(trajectory):
        read_trajectory_file(root / trajectory.rel, trajectory, read, data)
    for camera in CAMERAS:
        if camera in videos and gate(videos[camera]):
            read_video_file(root / videos[camera].rel, videos[camera], read)
    for rec in (frames, trajectory, *videos.values()):
        if rec is not None and rec.opened and rec.stat is not None:
            try:
                changed = _signature(root / rec.rel) != rec.stat
            except OSError:
                changed = True
            if changed:
                rec.errors.append(
                    "changed while the check ran: it is not the file that was hashed"
                )


def _unit_files(unit: dict[str, Any]) -> list[DataFile]:
    rest = [unit[kind] for kind in ("frames", "trajectory") if unit[kind] is not None]
    return [*rest, *unit["videos"].values()]


def read_files(
    root: Path,
    files: list[DataFile],
    *,
    workers: int = 1,
    gate: Callable[[DataFile], bool] | None = None,
    progress: IO[str] | None = None,
) -> dict[str, Episode]:
    """Read every data file that gate lets through (every file with an identity by
    default), an episode's files together in one of workers threads, then apply the
    episode rules. Returns the episodes by the path of their frames file."""
    units: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in files:
        if rec.id is None:
            continue
        unit = units.setdefault(
            (rec.id.object_type, rec.id.episode_uid),
            {"frames": None, "trajectory": None, "videos": {}},
        )
        if rec.kind == "video":
            unit["videos"][rec.id.camera] = rec
        else:
            unit[rec.kind] = rec
    jobs = sorted(units.values(), key=lambda u: _unit_files(u)[0].rel)
    report = _Progress(progress, "reading", sum(len(_unit_files(u)) for u in jobs))
    done = 0
    for unit, _ in _pool(
        workers, jobs, lambda u: _read_unit(root, u, gate or _open_all)
    ):
        done += len(_unit_files(unit))
        report(done, "files")
    return link_episodes(files)


def _frames_rel(object_type: str, uid: str) -> str:
    return f"{DATA_DIR}/{object_type}/{uid}{PARQUET_SUFFIX}"


def _video_rel(camera: str, object_type: str, uid: str) -> str:
    return f"{VIDEO_DIR}/{camera}/{object_type}/{uid}{VIDEO_SUFFIX}"


def link_episodes(files: list[DataFile]) -> dict[str, Episode]:
    """The episode rules: every file of an episode on disk, no file of no episode, one
    episode one frames file, and no two paths that a case-insensitive file system holds
    as one. Their findings follow each file's own."""
    episodes: dict[str, Episode] = {}
    by_id: dict[tuple[str, str], Episode] = {}
    for rec in files:
        if rec.id is not None and rec.kind == "frames":
            episodes[rec.rel] = by_id[rec.id.object_type, rec.id.episode_uid] = Episode(
                rec
            )
    for rec in files:
        if rec.id is None or rec.kind == "frames":
            continue
        episode = by_id.get((rec.id.object_type, rec.id.episode_uid))
        if episode is None:
            frames_rel = _frames_rel(rec.id.object_type, rec.id.episode_uid)
            rec.errors.append(f"belongs to no episode: {frames_rel} is not on disk")
        elif rec.kind == "trajectory":
            episode.trajectory = rec
        else:
            episode.videos[rec.id.camera] = rec
    for episode in episodes.values():
        frames = episode.frames
        object_type, uid = frames.id.object_type, frames.id.episode_uid
        if episode.trajectory is None:
            rel = f"{TRAJECTORY_DIR}/{object_type}/{uid}{PARQUET_SUFFIX}"
            frames.errors.append(f"its trajectory {rel} is not on disk")
        footer = frames.footer if frames.sizes is not None else None
        if footer is not None and "rgb" in footer["channels"]:
            episode.expected = tuple(footer["cameras"])
        for camera in episode.expected:
            if camera not in episode.videos:
                rel = _video_rel(camera, object_type, uid)
                frames.errors.append(f"its video {rel} is not on disk")
        for camera, video in episode.videos.items():
            if footer is None:
                continue
            if "rgb" not in footer["channels"]:
                video.errors.append(
                    "is not a video of its episode, which records no rgb"
                )
            elif camera not in footer["cameras"]:
                video.errors.append(
                    f"is not a video of its episode, whose cameras are {footer['cameras']}"
                )
    first: dict[str, str] = {}
    for episode in episodes.values():
        uid = episode.frames.id.episode_uid
        if uid in first:
            episode.frames.errors.append(
                f"episode {uid} is also the episode of {first[uid]}"
            )
        else:
            first[uid] = episode.frames.rel
    seen: dict[str, str] = {}
    for rec in files:
        if rec.rel.lower() in seen:
            rec.errors.append(
                f"differs from {seen[rec.rel.lower()]} only in case; a "
                "case-insensitive file system holds one of them"
            )
        else:
            seen[rec.rel.lower()] = rec.rel
    for episode in episodes.values():
        episode.channels = _validated_channels(episode)
    return episodes


def _channel_names(channel: str, cameras: Iterable[str]) -> list[str]:
    names = []
    for name in CHANNEL_COLUMNS[channel]:
        names += (
            [name.format(camera=c) for c in cameras] if "{camera}" in name else [name]
        )
    if channel in CAMERA_CHANNELS:
        names += [name.format(camera=c) for name in CAMERA_COLUMNS for c in cameras]
    return names


def _available(video: DataFile | None) -> bool:
    return video is not None and video.opened and not video.errors


def _validated_channels(episode: Episode) -> tuple[str, ...]:
    """The channels an episode carries: listed by its footer, every column of each in
    its schema with the format's type, a camera for a camera channel, and every video
    for rgb."""
    frames = episode.frames
    if not frames.opened or frames.sizes is None or frames.schema is None:
        return ()
    footer = frames.footer
    cameras = tuple(footer["cameras"])
    found = {f.name: f.type for f in frames.schema}
    model = {
        f.name: f.type
        for f in frame_schema(tuple(footer["channels"]), cameras, frames.sizes)
    }
    out = []
    for channel in CHANNELS:
        if channel not in footer["channels"]:
            continue
        if channel in CAMERA_CHANNELS and not cameras:
            continue
        names = _channel_names(channel, cameras)
        if not all(n in found and found[n].equals(model[n]) for n in names):
            continue
        if channel == "rgb" and not all(
            _available(episode.videos.get(c)) for c in cameras
        ):
            continue
        out.append(channel)
    return tuple(out)


# --- reading the dataset --------------------------------------------------------------


@dataclass
class Dataset:
    """Everything the items compare: the data files, the episodes and what
    metadata.yaml lists."""

    root: Path
    files: list[DataFile]
    episodes: dict[str, Episode]
    listed: dict[str, dict[str, Any]]
    tools: dict[str, dict[str, Any]]
    problems: list[str]
    hashes: dict[str, str | None]
    ignored: list[str]
    checksums: bool


def read_dataset(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    workers: int = 1,
    checksums: bool = True,
    progress: IO[str] | None = None,
) -> Dataset:
    """Find the data files, hash every one before any is opened, and read only those
    whose sha256 is the one metadata.yaml states (every one without checksums)."""
    listed, problems = listed_entries(metadata, "files")
    tools, tool_problems = listed_entries(metadata, "tools")
    rels, ignored = find_data_files(root)
    if (root / TOOLS_DIR).is_dir():
        for path in sorted((root / TOOLS_DIR).rglob("*")):
            rel = path.relative_to(root).as_posix()
            if path.is_file() and rel not in TOOL_FILES and rel not in tools:
                ignored.append(rel)
    files = [data_file(rel) for rel in rels]
    hashes: dict[str, str | None] = {}
    if checksums:
        wanted = set(rels) | {
            rel for rel in (*listed, *tools, *TOOL_FILES) if (root / rel).is_file()
        }
        hashes, stats = hash_files(root, sorted(wanted), workers, progress)
    else:
        stats = {}
        for rel in rels:
            try:
                stats[rel] = _signature(root / rel)
            except OSError:
                pass
    for rec in files:
        rec.sha256 = hashes.get(rec.rel)
        rec.stat = stats.get(rec.rel)
        rec.size = rec.stat[0] if rec.stat else 0

    def gate(rec: DataFile) -> bool:
        if rec.id is None:
            return False
        if checksums:
            entry = listed.get(rec.rel)
            stated = str(entry.get("sha256", "")).lower() if entry is not None else None
            if rec.sha256 is None or rec.sha256 != stated:
                rec.errors.append(
                    (
                        "not opened: its sha256 is not the one metadata.yaml states"
                        if entry is not None
                        else "not opened: metadata.yaml does not list it"
                    )
                    + _lfs_hint(root / rec.rel)
                )
                return False
        rec.opened = True
        return True

    episodes = read_files(root, files, workers=workers, gate=gate, progress=progress)
    return Dataset(
        root,
        files,
        episodes,
        listed,
        tools,
        problems + tool_problems,
        hashes,
        sorted(ignored),
        checksums,
    )


def read_checksum_list(root: Path) -> tuple[dict[str, str], list[str]] | None:
    """SHA256SUMS as {path: sha256} and its problems, None without the file. Lines end
    at a newline, with or without a carriage return, as sha256sum reads them."""
    path = root / CHECKSUMS_NAME
    if not path.is_file():
        return None
    digests: dict[str, str] = {}
    problems = []
    text = path.read_bytes().decode("utf-8", errors="replace")
    for number, line in enumerate(text.split("\n"), 1):
        line = line.removesuffix("\r")
        if not line.strip():
            continue
        match = CHECKSUM_LINE.fullmatch(line)
        if match is None:
            problems.append(f"{CHECKSUMS_NAME} line {number} is not '<sha256>  <path>'")
            continue
        digest, rel = match.group(1).lower(), match.group(2)
        if rel in digests:
            problems.append(f"{CHECKSUMS_NAME}: {rel} is listed twice")
        digests[rel] = digest
    return digests, problems


# --- scale ----------------------------------------------------------------------------


def check_scale(ds: Dataset, criteria: Mapping[str, Any]) -> dict[str, Any]:
    """Every distinct frame of every episode whose files are all clean, counted from the
    Parquet row counts; one file with an error fails the item."""
    minimum = criteria["min_frames"]
    clean = [e for e in ds.episodes.values() if not e.has_errors and e.frames.n_rows]
    dirty = [e for e in ds.episodes.values() if e.has_errors]
    bad = [
        {"file": rec.rel, "kind": rec.kind, "errors": rec.errors}
        for rec in ds.files
        if rec.errors
    ]
    per_type: dict[str, dict[str, int]] = {}
    for episode in clean:
        row = per_type.setdefault(
            episode.frames.id.object_type, {"n_episodes": 0, "n_frames": 0}
        )
        row["n_episodes"] += 1
        row["n_frames"] += episode.frames.n_rows
    n_frames = sum(e.frames.n_rows for e in clean)
    return {
        "pass": bool(ds.episodes) and not bad and n_frames >= minimum,
        "n_frames": n_frames,
        "n_episodes": len(clean),
        "n_files": len(ds.files),
        "criterion": {"min_frames": minimum},
        "n_files_with_errors": len(bad),
        "n_episodes_with_errors": len(dirty),
        "n_frames_in_episodes_with_errors": sum(e.frames.n_rows or 0 for e in dirty),
        "files_with_errors": bad,
        "per_object_type": dict(sorted(per_type.items())),
    }


# --- metadata -------------------------------------------------------------------------


def _same(a: Any, b: Any) -> bool:
    """Equal and of one JSON type: 1 is neither 1.0 nor true."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    for kind in (int, float, str):
        if isinstance(a, kind) or isinstance(b, kind):
            return isinstance(a, kind) and isinstance(b, kind) and a == b
    return a is None and b is None


class _Ledger:
    """Every comparison counted, every disagreement recorded as one mismatch."""

    def __init__(self) -> None:
        self.n_checked = 0
        self.items: list[dict[str, Any]] = []

    def compare(self, what: str, metadata: Any, files: Any) -> None:
        self.n_checked += 1
        if not _same(metadata, files):
            self.items.append({"what": what, "metadata": metadata, "files": files})

    def missing(self, what: str, metadata: Any, files: Any) -> None:
        self.n_checked += 1
        self.items.append({"what": what, "metadata": metadata, "files": files})


def _frames_files(ds: Dataset) -> list[DataFile]:
    return [rec for rec in ds.files if rec.kind == "frames" and rec.id is not None]


def _footer_cameras(rec: DataFile) -> list[str]:
    return list(rec.footer["cameras"]) if rec.sizes is not None else []


def _footer_strings(records: Iterable[DataFile], key: str) -> set[str]:
    """The distinct strings of a footer key: a value of another type is an error of its
    file (footer_errors), not a value to count."""
    return {value for rec in records if isinstance(value := rec.footer.get(key), str)}


def _distinct(records: Iterable[Mapping[str, Any] | None]) -> list[Any]:
    out: dict[str, Any] = {}
    for record in records:
        if record is not None:
            out.setdefault(json.dumps(record, sort_keys=True), record)
    return list(out.values())


def check_spec(
    ds: Dataset, metadata: Mapping[str, Any], criteria: Mapping[str, Any]
) -> dict[str, Any]:
    """The dataset specification the tester reads from the metadata: the channels and
    cameras every episode carries, the licence, a DOI and a download URL."""
    episodes = list(ds.episodes.values())
    everywhere = [
        c for c in CHANNELS if episodes and all(c in e.channels for e in episodes)
    ]
    cameras = [
        c
        for c in CAMERAS
        if episodes and all(c in _footer_cameras(e.frames) for e in episodes)
    ]
    missing = [c for c in criteria["required_channels"] if c not in everywhere]
    missing_cameras = [c for c in criteria["required_cameras"] if c not in cameras]
    doi, url = metadata.get("doi"), metadata.get("download_url")
    checks = {
        "channels": not missing,
        "cameras": not missing_cameras,
        "license": _same(metadata.get("license"), criteria["license"]),
        "doi": isinstance(doi, str) and bool(DOI_PATTERN.fullmatch(doi)),
        "download_url": isinstance(url, str) and bool(URL_PATTERN.fullmatch(url)),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "n_channels": len(everywhere),
        "channels_in_every_episode": everywhere,
        "missing_channels": missing,
        "cameras_in_every_episode": cameras,
        "missing_cameras": missing_cameras,
        "license": metadata.get("license"),
        "doi": doi,
        "download_url": url,
    }


def check_metadata(
    ds: Dataset, metadata: Mapping[str, Any], criteria: Mapping[str, Any]
) -> dict[str, Any]:
    """metadata.yaml against the files: each value it states compared once, each
    disagreement one mismatch, and the specification. Without checksums the sha256
    comparisons are skipped and the item cannot pass."""
    ledger = _Ledger()
    ledger.compare("format", metadata.get("format"), FORMAT)
    ledger.compare("format_version", metadata.get("format_version"), FORMAT_VERSION)
    frames = _frames_files(ds)
    read = [rec for rec in frames if rec.footer is not None]
    counted = {
        "n_frames": sum(rec.n_rows or 0 for rec in frames),
        "n_episodes": len({rec.id.episode_uid for rec in frames}),
        "n_files": len(ds.files),
        "n_object_types": len({rec.id.object_type for rec in frames}),
        "n_object_assets": len(_footer_strings(read, "object_asset")),
    }
    counts = metadata.get("counts")
    counts = counts if isinstance(counts, dict) else {}
    for key, value in counted.items():
        ledger.compare(f"counts.{key}", counts.get(key), value)
    channels = [
        c for c in CHANNELS if any(c in e.channels for e in ds.episodes.values())
    ]
    ledger.compare("counts.n_channels", counts.get("n_channels"), len(channels))
    ledger.compare("channels", metadata.get("channels"), channels)
    conventions = metadata.get("conventions")
    stated = conventions
    if isinstance(conventions, dict):
        stated = {k: v for k, v in conventions.items() if k != "segmentation_classes"}
    ledger.compare(
        "conventions",
        stated,
        {k: v for k, v in CONVENTIONS.items() if k != "segmentation_classes"},
    )

    actual: dict[str, dict[str, Any]] = {}
    for rec in frames:
        row = actual.setdefault(
            rec.id.object_type, {"uids": set(), "n_frames": 0, "assets": set()}
        )
        row["uids"].add(rec.id.episode_uid)
        row["n_frames"] += rec.n_rows or 0
        if rec.footer is not None and isinstance(rec.footer.get("object_asset"), str):
            row["assets"].add(rec.footer["object_asset"])
    types = metadata.get("object_types")
    types = types if isinstance(types, dict) else {}
    for name in types:
        if not isinstance(name, str):
            ledger.missing(f"object_types: {name!r} is not a type name", name, None)
    for name in sorted({n for n in types if isinstance(n, str)} | set(actual)):
        stated_row = types.get(name) if isinstance(types.get(name), dict) else {}
        row = actual.get(name, {"uids": set(), "n_frames": 0, "assets": set()})
        ledger.compare(
            f"object_types.{name}.n_episodes",
            stated_row.get("n_episodes"),
            len(row["uids"]),
        )
        ledger.compare(
            f"object_types.{name}.n_frames", stated_row.get("n_frames"), row["n_frames"]
        )
        ledger.compare(
            f"object_types.{name}.assets",
            stated_row.get("assets"),
            sorted(row["assets"]),
        )

    video = metadata.get("video")
    shared_video = video
    if isinstance(video, dict):
        shared_video = {k: v for k, v in video.items() if k != "ffmpeg_versions"}
    classes = (
        conventions.get("segmentation_classes")
        if isinstance(conventions, dict)
        else None
    )
    for rec in frames:
        footer = rec.footer or {}
        validated = list(ds.episodes[rec.rel].channels)
        ledger.compare(f"{rec.rel}: channels", metadata.get("channels"), validated)
        for key, value in (
            ("cameras", metadata.get("cameras")),
            ("image_size", metadata.get("image_size")),
            ("fps", metadata.get("fps")),
            ("recorded_fps", metadata.get("recorded_fps")),
            ("license", metadata.get("license")),
            ("format_version", metadata.get("format_version")),
            ("dataset_name", metadata.get("name")),
            ("dataset_version", metadata.get("version")),
            ("segmentation_classes", classes),
            ("video", shared_video),
        ):
            ledger.compare(f"{rec.rel}: {key}", value, footer.get(key))
    if isinstance(video, dict):
        ledger.compare(
            "video.ffmpeg_versions",
            video.get("ffmpeg_versions"),
            sorted(_footer_strings(read, "ffmpeg_version")),
        )
    ledger.compare(
        "provenance",
        metadata.get("provenance"),
        {
            "generation": _distinct(
                rec.footer.get("generation_provenance") for rec in read
            ),
            "export": _distinct(rec.footer.get("export_provenance") for rec in read),
        },
    )

    for problem in ds.problems:
        ledger.missing(problem, None, None)
    by_rel = {rec.rel: rec for rec in ds.files}
    for rel, entry in sorted(ds.listed.items()):
        rec = by_rel.get(rel)
        if rec is None:
            ledger.missing(f"files: {rel} is listed but not on disk", rel, None)
            continue
        ledger.compare(f"{rel}: kind", entry.get("kind"), rec.kind)
        ledger.compare(f"{rel}: bytes", entry.get("bytes"), rec.size)
        if rec.kind == "trajectory":
            ledger.compare(f"{rel}: n_steps", entry.get("n_steps"), rec.n_rows)
        elif rec.kind == "video":
            count = rec.track.n_frames if rec.track is not None else None
            ledger.compare(f"{rel}: n_frames", entry.get("n_frames"), count)
        else:
            ledger.compare(f"{rel}: n_frames", entry.get("n_frames"), rec.n_rows)
        ledger.compare(
            f"{rel}: object_type", entry.get("object_type"), rec.id.object_type
        )
        ledger.compare(
            f"{rel}: episode_uid", entry.get("episode_uid"), rec.id.episode_uid
        )
        if ds.checksums:
            stated_sha = str(entry.get("sha256", "")).lower()
            ledger.compare(f"{rel}: sha256", stated_sha, rec.sha256)
    for rec in ds.files:
        if rec.rel not in ds.listed:
            ledger.missing(f"files: {rec.rel} is on disk but not listed", None, rec.rel)
    for rel in TOOL_FILES:
        if rel not in ds.tools:
            ledger.missing(f"tools: {rel} is not listed", None, rel)
    for rel, entry in sorted(ds.tools.items()):
        if rel not in TOOL_FILES:
            ledger.missing(f"tools: {rel} is not a tool of the format", rel, None)
            continue
        path = ds.root / rel
        if not path.is_file():
            ledger.missing(f"tools: {rel} is listed but missing", rel, None)
            continue
        ledger.compare(f"{rel}: bytes", entry.get("bytes"), path.stat().st_size)
        if ds.checksums:
            stated_sha = str(entry.get("sha256", "")).lower()
            ledger.compare(f"{rel}: sha256", stated_sha, ds.hashes.get(rel))
    sums = read_checksum_list(ds.root)
    if sums is None:
        ledger.missing(f"{CHECKSUMS_NAME} is missing", CHECKSUMS_NAME, None)
    else:
        digests, problems = sums
        for problem in problems:
            ledger.missing(problem, None, None)
        stated_sums = {
            rel: str(e.get("sha256", "")).lower()
            for rel, e in {**ds.listed, **ds.tools}.items()
        }
        ledger.compare(f"{CHECKSUMS_NAME}: paths", sorted(stated_sums), sorted(digests))
        for rel in sorted(set(stated_sums) & set(digests)):
            ledger.compare(f"{CHECKSUMS_NAME}: {rel}", stated_sums[rel], digests[rel])

    spec = check_spec(ds, metadata, criteria)
    maximum = criteria["max_mismatches"]
    return {
        "pass": ds.checksums
        and bool(frames)
        and len(ledger.items) <= maximum
        and spec["pass"],
        "n_mismatches": len(ledger.items),
        "n_checked": ledger.n_checked,
        "checksums_verified": ds.checksums,
        "criterion": {
            "max_mismatches": maximum,
            "required_channels": list(criteria["required_channels"]),
            "required_cameras": list(criteria["required_cameras"]),
            "license": criteria["license"],
        },
        "counted": {**counted, "n_channels": len(channels), "channels": channels},
        "mismatches": ledger.items,
        "spec": spec,
    }


# --- consistency ----------------------------------------------------------------------

# The segmentation class whose pixels are the object, looked up by name.
OBJECT_CLASS = "object"


def dataset_cameras(
    metadata: Mapping[str, Any], floor: Iterable[str] = CAMERAS
) -> list[str]:
    """The cameras every frame is checked for: metadata.yaml's, never fewer than floor
    (the criteria's required cameras), and all of the format's when metadata.yaml
    names no valid set."""
    cameras = metadata.get("cameras")
    if _names(cameras, CAMERAS, nonempty=True):
        wanted = set(cameras) | set(floor)
        return [c for c in CAMERAS if c in wanted]
    return list(CAMERAS)


def file_cameras(rec: DataFile, cameras: Iterable[str]) -> list[str]:
    """The cameras one file's frames are checked for: the dataset's and its own, so a
    file that dropped a camera cannot shrink what its frames are checked for."""
    own = rec.footer.get("cameras") if rec.footer is not None else None
    own = [c for c in own if isinstance(c, str)] if isinstance(own, list) else []
    wanted = set(cameras) | {c for c in own if c in CAMERAS}
    return [c for c in CAMERAS if c in wanted]


def sample_frames(
    candidates: Iterable[tuple[str, int]], k: int, seed: int, purpose: str
) -> list[tuple[str, int]]:
    """A seeded sample of k (file, frame) pairs, in path and frame order.

    Each pair's key is the first 8 bytes of the sha256 of "seed NUL purpose NUL path NUL
    frame", and the k smallest keys are the sample. It depends on nothing else: not the
    order files are read in, the Python, NumPy or PyArrow version, or the platform, so a
    seed draws the same frames on any machine. purpose keeps two samples independent.
    """

    def keys() -> Iterator[tuple[bytes, str, int]]:
        for rel, i in candidates:
            text = f"{seed}\0{purpose}\0{rel}\0{i}"
            yield hashlib.sha256(text.encode()).digest()[:8], rel, i

    return sorted((rel, i) for _, rel, i in heapq.nsmallest(k, keys()))


def _stated_frames(entry: Any) -> int:
    """The frames a files entry states, 0 when it states no non-negative integer."""
    stated = entry.get("n_frames") if isinstance(entry, dict) else None
    return max(stated, 0) if _is_int(stated) else 0


def _reopen(root: Path, rec: DataFile) -> Any:
    """A data file opened again; refused when it is no longer the file read first."""
    if rec.stat is not None and _signature(root / rec.rel) != rec.stat:
        raise ValueError("it changed after it was hashed")
    return _parquet(root / rec.rel)


def step_reasons(
    root: Path,
    rec: DataFile,
    layers: Iterable[str],
    cameras: list[str],
    videos: Mapping[str, DataFile],
) -> list[list[str]]:
    """Per frame, why its layers were not recorded at its own step; an empty list
    passes. A layer's stamp is the column <layer>[.<camera>].sim_step, and rgb also
    needs each frame's index into an available video of the episode."""
    n = rec.n_rows
    reasons: list[list[str]] = [[] for _ in range(n)]
    everyone = np.ones(n, dtype=bool)

    def fail(rows: Any, reason: str) -> None:
        for i in np.flatnonzero(rows):
            reasons[i].append(reason)

    own = rec.footer.get("cameras") if rec.footer is not None else None
    own = own if isinstance(own, list) else []
    missing = [c for c in cameras if c not in own]
    if missing:
        fail(everyone, f"the file does not record the dataset's cameras {missing}")
    try:
        with _reopen(root, rec) as parquet:
            types = {f.name: f.type for f in parquet.schema_arrow}

            def has(name: str, kind: Any) -> bool:
                return name in types and types[name].equals(kind)

            if not has("sim_step", pa.int64()):
                fail(everyone, "sim_step has no step for each frame")
                return reasons
            prefixes = []
            for layer in layers:
                if layer in CAMERA_CHANNELS:
                    prefixes += [f"{layer}.{c}" for c in cameras]
                else:
                    prefixes.append(layer)
            stamps = [p for p in prefixes if has(f"{p}.sim_step", pa.int64())]
            indices = [c for c in cameras if has(f"rgb.{c}.video_frame", pa.int32())]
            table = parquet.read(
                columns=["sim_step"]
                + [f"{p}.sim_step" for p in stamps]
                + [f"rgb.{c}.video_frame" for c in indices]
            )
        steps = table.column("sim_step").to_numpy(zero_copy_only=False)
        own_step = steps < 1
        own_step[1:] |= steps[1:] <= steps[:-1]
        fail(own_step, "its step is below 1 or not later than the previous frame's")
        for prefix in prefixes:
            if prefix in stamps:
                stamp = table.column(f"{prefix}.sim_step").to_numpy(
                    zero_copy_only=False
                )
                fail(stamp != steps, f"{prefix} was recorded at another step")
            else:
                fail(everyone, f"{prefix} has no step for each frame")
            if not prefix.startswith("rgb."):
                continue
            camera = prefix.split(".")[1]
            if camera in indices:
                index = table.column(f"rgb.{camera}.video_frame").to_numpy(
                    zero_copy_only=False
                )
                fail(
                    index != np.arange(n),
                    f"rgb.{camera} is not the frame's video frame",
                )
            else:
                fail(everyone, f"rgb.{camera} has no video frame for each frame")
            video = videos.get(camera)
            if not _available(video):
                rel = _video_rel(camera, rec.id.object_type, rec.id.episode_uid)
                why = "is not on disk" if video is None else video.errors[0]
                fail(
                    everyone,
                    f"rgb.{camera} has no video of the episode's frames: {rel} {why}",
                )
    except Exception as exc:  # noqa: BLE001, it fails the file's frames, not the run
        fail(everyone, _failed(rec, exc, "the file cannot be read"))
    return reasons


def projection_inputs(
    schema: Any, footer: Mapping[str, Any], cameras: Iterable[str]
) -> tuple[list[str], int | None, list[str]]:
    """What a frames file lacks to project its contacts into cameras, the object's mask
    value and the class names, read by name: the value is never assumed to be 1."""
    types = {f.name: f.type for f in schema}
    lacking = []

    def need(name: str, kind: Any) -> None:
        if not (name in types and types[name].equals(kind)):
            lacking.append(name)

    need("sim_step", pa.int64())
    need("force.contacts", arrow_type("struct", "contacts"))
    sizes = footer.get("image_size")
    sizes = sizes if isinstance(sizes, dict) else {}
    for camera in cameras:
        size = sizes.get(camera)
        if (
            isinstance(size, dict)
            and _is_int(size.get("height"), 1)
            and _is_int(size.get("width"), 1)
        ):
            hw = (size["height"], size["width"])
            need(f"segmentation.{camera}.mask", arrow_type("uint8", "image", hw))
        else:
            lacking.append(f"image_size.{camera}")
        need(f"segmentation.{camera}.sim_step", pa.int64())
        need(f"camera.{camera}.intrinsics", arrow_type("float64", (3, 3)))
        need(f"camera.{camera}.world_to_camera", arrow_type("float64", (4, 4)))
    classes = footer.get("segmentation_classes")
    if not (isinstance(classes, list) and OBJECT_CLASS in classes):
        lacking.append(f"segmentation_classes naming {OBJECT_CLASS!r}")
        return lacking, None, []
    return lacking, classes.index(OBJECT_CLASS), [str(c) for c in classes]


def _contacts(array: Any) -> tuple[Any, Any, Any, Any]:
    """force.contacts as offsets, is_robot, force (M, 3) and point_world (M, 3)."""
    children = dict(
        zip([f.name for f in array.type.value_type], array.flatten().flatten())
    )
    robot = children["is_robot"].to_numpy(zero_copy_only=False).astype(bool)
    force = _fixed(children["force"]).astype(np.float64)
    points = _fixed(children["point_world"]).astype(np.float64)
    return _offsets(array), robot, force, points


def pressed_frames(contacts: Any) -> Any:
    """Per frame, whether a contact presses on the robot: is_robot, and a component of
    its force that is not 0 (NaN is not 0)."""
    offsets, robot, force, _ = _contacts(contacts)
    pressed = robot & np.any(force != 0.0, axis=1)
    totals = np.concatenate([[0], np.cumsum(pressed, dtype=np.int64)])
    return (totals[offsets[1:]] - totals[offsets[:-1]]) > 0


def project_points(points: Any, intrinsics: Any, world_to_camera: Any) -> tuple:
    """World points to pixels: (u, v, z), u the column, v the row, z the forward depth."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    t = np.asarray(world_to_camera, dtype=np.float64).reshape(4, 4)
    cam = pts @ t[:3, :3].T + t[:3, 3]
    pix = cam @ np.asarray(intrinsics, dtype=np.float64).reshape(3, 3).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return pix[:, 0] / pix[:, 2], pix[:, 1] / pix[:, 2], cam[:, 2]


def object_distances(
    mask: Any, u: Any, v: Any, z: Any, object_value: int, reach: float
) -> tuple[Any, Any]:
    """Per point, whether it is in view (in front of the camera and inside the image)
    and its distance from where it lands to the centre of the nearest object pixel,
    measured out to reach and infinite beyond. The nearest object pixel, not the class
    of the pixel it lands in: a contact lies between a pad and the object."""
    height, width = mask.shape
    with np.errstate(invalid="ignore"):
        in_view = (
            np.isfinite(u)
            & np.isfinite(v)
            & (z > 0)
            & (u > -0.5)
            & (u < width - 0.5)
            & (v > -0.5)
            & (v < height - 0.5)
        )
    dist = np.full(u.shape, np.inf)
    r = math.ceil(reach) + 1
    for j in np.flatnonzero(in_view):
        row, col = int(np.rint(v[j])), int(np.rint(u[j]))
        r0, c0 = max(row - r, 0), max(col - r, 0)
        rows, cols = np.nonzero(
            mask[r0 : row + r + 1, c0 : col + r + 1] == object_value
        )
        if rows.size:
            d = float(np.hypot(cols + c0 - u[j], rows + r0 - v[j]).min())
            dist[j] = d if d <= reach else np.inf
    return in_view, dist


def project_frame(
    frame: Mapping[str, Any],
    cameras: list[str],
    object_value: int,
    classes: list[str],
    radius: float,
    min_points: float,
    reach: float,
) -> dict[str, Any]:
    """Project one frame's pressed contact points into each camera and judge them.

    Each camera's mask must be of the frame's step and its matrices the format's; a
    camera that sees points passes when min_points of them lie within radius pixels of
    the object, and the frame passes when a camera sees one and every such camera
    passes. Each point in view is also counted under the class it lands in, and its
    depth residual (its depth minus the depth image's) kept, for the report only.
    """
    step = frame["sim_step"]
    robot, force, points = frame["contacts"]
    points = points[robot & np.any(force != 0.0, axis=1)]
    per_camera, raw = {}, {}
    for camera in cameras:
        mask_step = frame[f"segmentation.{camera}.sim_step"]
        if mask_step != step:
            return {
                "status": "mask_step",
                "pass": False,
                "reason": f"segmentation.{camera} is of step {mask_step}, not the "
                f"frame's {step}",
            }
        k = frame[f"camera.{camera}.intrinsics"]
        t = frame[f"camera.{camera}.world_to_camera"]
        for name, rule, matrix in (
            (f"camera.{camera}.intrinsics", intrinsics_error, k),
            (f"camera.{camera}.world_to_camera", world_to_camera_error, t),
        ):
            problem = rule(matrix)
            if problem:
                return {
                    "status": "camera_matrices",
                    "pass": False,
                    "reason": f"{camera}: {name} {problem}",
                }
        mask = frame[f"segmentation.{camera}.mask"]
        u, v, z = project_points(points, k, t)
        in_view, dist = object_distances(mask, u, v, z, object_value, reach)
        on = in_view & (dist <= radius)
        landing: dict[str, int] = {}
        residuals: list[float] = []
        if in_view.any():
            rows = np.rint(v[in_view]).astype(int)
            cols = np.rint(u[in_view]).astype(int)
            for value in mask[rows, cols].tolist():
                name = classes[value] if 0 <= value < len(classes) else str(value)
                landing[name] = landing.get(name, 0) + 1
            depth = frame.get(f"depth.{camera}.depth_m")
            if depth is not None:
                at = depth[rows, cols].astype(np.float64)
                hit = at > 0
                residuals = (z[in_view][hit] - at[hit]).tolist()
        seen = dist[in_view]
        per_camera[camera] = {
            "n_points": len(points),
            "n_in_view": int(in_view.sum()),
            "n_on_object": int(on.sum()),
            "landing": landing,
            "max_distance_px": float(seen.max())
            if seen.size and np.isfinite(seen).all()
            else None,
        }
        raw[camera] = {
            "distances": seen[np.isfinite(seen)].tolist(),
            "n_beyond": int(np.isinf(seen).sum()),
            "residuals": residuals,
        }
    seen_by = {c: r for c, r in per_camera.items() if r["n_in_view"]}
    ok = bool(seen_by) and all(
        r["n_on_object"] >= min_points * r["n_in_view"] for r in seen_by.values()
    )
    status = "pass" if ok else ("not_in_view" if not seen_by else "off_object")
    reason = {
        "pass": None,
        "not_in_view": "no camera sees the pressed contact points",
        "off_object": f"pressed contact points lie more than {radius:g} px from the "
        "object",
    }[status]
    return {
        "status": status,
        "pass": ok,
        "reason": reason,
        "cameras": per_camera,
        "raw": raw,
    }


def _sampled_frame(
    row: Any,
    cameras: list[str],
    sizes: Mapping[str, tuple[int, int]],
    columns: list[str],
) -> dict[str, Any]:
    """A sampled frame's values from its one-row batch, as project_frame takes them."""
    frame: dict[str, Any] = {
        "sim_step": row.column("sim_step")[0].as_py(),
        "contacts": _contacts(_array(row.column("force.contacts")))[1:],
    }
    for camera in cameras:
        height, width = sizes[camera]
        for name in (
            f"segmentation.{camera}.sim_step",
            f"camera.{camera}.intrinsics",
            f"camera.{camera}.world_to_camera",
        ):
            array = _array(row.column(name))
            frame[name] = (
                _fixed(array)[0]
                if pa.types.is_fixed_size_list(array.type)
                else array[0].as_py()
            )
        for name in (f"segmentation.{camera}.mask", f"depth.{camera}.depth_m"):
            if name in columns:
                frame[name] = _fixed(_array(row.column(name)))[0].reshape(height, width)
    return frame


def _project_file(
    root: Path,
    rec: DataFile,
    frames: list[int],
    cameras: list[str],
    inputs: tuple[int, list[str]],
    judge: tuple[float, float, float],
) -> list[dict[str, Any]]:
    """project_frame for a file's sampled frames, the file opened once and only the row
    groups that hold them read, frames_per_read frames at a time. What one frame raises
    fails that frame."""

    def failure(what: str, exc: Exception) -> dict[str, Any]:
        return {
            "status": "unreadable",
            "pass": False,
            "reason": _failed(rec, exc, what),
        }

    try:
        with _reopen(root, rec) as parquet:
            types = {f.name: f.type for f in parquet.schema_arrow}
            sizes = {
                c: (
                    rec.footer["image_size"][c]["height"],
                    rec.footer["image_size"][c]["width"],
                )
                for c in cameras
            }
            columns = ["sim_step", "force.contacts"]
            depths = set()
            for camera in cameras:
                columns += [
                    f"segmentation.{camera}.mask",
                    f"segmentation.{camera}.sim_step",
                    f"camera.{camera}.intrinsics",
                    f"camera.{camera}.world_to_camera",
                ]
                depth = f"depth.{camera}.depth_m"
                # Only for the residual, which is never judged: of the image's size.
                if depth in types and types[depth].equals(
                    arrow_type("float32", "image", sizes[camera])
                ):
                    columns.append(depth)
                    depths.add(depth)
            ends = np.cumsum(
                [
                    parquet.metadata.row_group(g).num_rows
                    for g in range(parquet.num_row_groups)
                ]
            )
            out: dict[int, dict[str, Any]] = {}
            wanted = sorted(set(frames))
            size = frames_per_read(sizes, 0)
            for group in sorted(
                {int(np.searchsorted(ends, i, side="right")) for i in wanted}
            ):
                first = int(ends[group - 1]) if group else 0
                for batch in parquet.iter_batches(
                    batch_size=size, row_groups=[group], columns=columns
                ):
                    last = first + batch.num_rows
                    for i in (i for i in wanted if first <= i < last):
                        try:
                            frame = _sampled_frame(
                                batch.slice(i - first, 1), cameras, sizes, columns
                            )
                            out[i] = project_frame(frame, cameras, *inputs, *judge)
                        except Exception as exc:  # noqa: BLE001, it fails the frame
                            out[i] = failure("the frame could not be projected", exc)
                    first = last
            return [out[i] for i in frames]
    except Exception as exc:  # noqa: BLE001, it fails the file's frames
        return [failure("the file cannot be read", exc) for _ in frames]


def _distribution(values: list[float], n_beyond: int = 0) -> dict[str, Any]:
    """The median, 99th percentile and maximum of a camera's values; n_beyond more lie
    past the reach they were measured to and leave the maximum unknown (None)."""
    array = np.asarray(values, dtype=np.float64)
    out: dict[str, Any] = {"n": int(array.size) + n_beyond, "n_beyond_reach": n_beyond}
    if array.size:
        out.update(
            {
                "median": float(np.median(array)),
                "p99": float(np.percentile(array, 99)),
                "max": None if n_beyond else float(array.max()),
            }
        )
    return out


def check_consistency(
    ds: Dataset,
    metadata: Mapping[str, Any],
    criteria: Mapping[str, Any],
    *,
    seed: int,
    sample: int | None = None,
    floor: Iterable[str] = CAMERAS,
    progress: IO[str] | None = None,
) -> dict[str, Any]:
    """Every frame's layers recorded at its own step, and, on a sample drawn with the
    tester's seed from the frames whose contacts press on the robot, those contact
    points landing on the object in each camera. Both parts must reach
    min_pass_fraction; a file without a frame count, or listed and missing, fails both,
    and its stated frames stay in the count."""
    fraction = criteria["min_pass_fraction"]
    layers = list(criteria["layers"])
    wanted = criteria["sample_frames"]
    k = wanted if sample is None else sample
    radius = criteria["projection_radius_px"]
    min_points = criteria["min_point_fraction"]
    reach = max(criteria["distance_reach_px"], radius)
    cameras = dataset_cameras(metadata, floor)

    on_disk = {rec.rel for rec in ds.files}
    n_frames = n_pass = 0
    by_reason: Counter[str] = Counter()
    failed: list[dict[str, Any]] = []
    no_count: list[dict[str, Any]] = []
    judged: dict[str, Any] = {}
    for rel, entry in sorted(ds.listed.items()):
        if rel.startswith(f"{DATA_DIR}/") and rel not in on_disk:
            stated = _stated_frames(entry)
            n_frames += stated
            no_count.append(
                {
                    "file": rel,
                    "frames_stated": stated,
                    "reasons": ["metadata.yaml lists it, but it is not on disk"],
                }
            )
    data_files = [rec for rec in ds.files if rec.rel.startswith(f"{DATA_DIR}/")]
    for rec in data_files:
        if not rec.opened or not rec.n_rows:
            stated = _stated_frames(ds.listed.get(rec.rel))
            n_frames += stated
            no_count.append(
                {
                    "file": rec.rel,
                    "frames_stated": stated,
                    "reasons": rec.errors or ["the file has no frame count"],
                }
            )
            continue
        episode = ds.episodes.get(rec.rel)
        reasons = step_reasons(
            ds.root,
            rec,
            layers,
            file_cameras(rec, cameras),
            episode.videos if episode is not None else {},
        )
        ok = np.array([not r for r in reasons], dtype=bool)
        judged[rec.rel] = ok
        n_frames += int(ok.size)
        n_pass += int(ok.sum())
        for i in np.flatnonzero(~ok):
            by_reason.update(reasons[i])
            if len(failed) < MAX_FRAMES_LISTED:
                failed.append({"file": rec.rel, "frame": int(i), "reasons": reasons[i]})
    if no_count:
        by_reason["files without a frame count, or missing"] = len(no_count)
    steps_ok = n_frames > 0 and not no_count and n_pass >= fraction * n_frames
    steps_sample = sample_frames(
        ((rel, i) for rel, ok in judged.items() for i in range(ok.size)),
        k,
        seed,
        "steps",
    )

    unavailable: dict[str, list[str]] = {e["file"]: e["reasons"] for e in no_count}
    plan: dict[str, tuple[list[str], tuple[int, list[str]]]] = {}
    n_candidates = 0

    def candidates() -> Iterator[tuple[str, int]]:
        nonlocal n_candidates
        for rec in data_files:
            if rec.rel not in judged:
                continue
            file_cams = file_cameras(rec, cameras)
            try:
                with _reopen(ds.root, rec) as parquet:
                    lacking, value, classes = projection_inputs(
                        parquet.schema_arrow, rec.footer or {}, file_cams
                    )
                    if lacking:
                        unavailable[rec.rel] = lacking
                        continue
                    contacts = _array(
                        parquet.read(columns=["force.contacts"]).column(0)
                    )
                pressed = np.flatnonzero(pressed_frames(contacts))
            except Exception as exc:  # noqa: BLE001, the file is unavailable
                unavailable[rec.rel] = [_failed(rec, exc)]
                continue
            plan[rec.rel] = (file_cams, (value, classes))
            n_candidates += int(pressed.size)
            yield from ((rec.rel, int(i)) for i in pressed)

    picked = sample_frames(candidates(), k, seed, "projection")
    statuses: Counter[str] = Counter()
    per_camera = {
        c: {
            "distances": [],
            "n_beyond": 0,
            "residuals": [],
            "landing": Counter(),
            "n_frames_in_view": 0,
        }
        for c in CAMERAS
    }
    projection_failures = []
    n_points = n_in_view = n_on = 0
    by_file: dict[str, list[int]] = {}
    for rel, i in picked:
        by_file.setdefault(rel, []).append(i)
    records = {rec.rel: rec for rec in data_files}
    report = _Progress(progress, "projecting", len(picked))
    done = 0
    for rel, frames in by_file.items():
        file_cams, inputs = plan[rel]
        results = _project_file(
            ds.root,
            records[rel],
            frames,
            file_cams,
            inputs,
            (radius, min_points, reach),
        )
        for i, got in zip(frames, results):
            statuses[got["status"]] += 1
            for camera, r in got.get("cameras", {}).items():
                n_points += r["n_points"]
                n_in_view += r["n_in_view"]
                n_on += r["n_on_object"]
                per_camera[camera]["landing"].update(r["landing"])
                per_camera[camera]["n_frames_in_view"] += bool(r["n_in_view"])
            for camera, r in got.get("raw", {}).items():
                per_camera[camera]["distances"] += r["distances"]
                per_camera[camera]["n_beyond"] += r["n_beyond"]
                per_camera[camera]["residuals"] += r["residuals"]
            if not got["pass"]:
                projection_failures.append(
                    {
                        "file": rel,
                        "frame": i,
                        "status": got["status"],
                        "reason": got["reason"],
                        "cameras": got.get("cameras"),
                    }
                )
        done += len(frames)
        report(done, "sampled frames")
    n_sampled = len(picked)
    n_projected_pass = statuses.get("pass", 0)
    whole_sample = n_sampled >= wanted or n_sampled == n_candidates
    # Each camera must see the contacts: one that sees none in the sample proves nothing.
    blind = [
        f"{c} sees no pressed contact point in the {n_sampled} sampled frames"
        for c in cameras
        if not per_camera[c]["n_frames_in_view"]
    ]
    projection_ok = (
        n_candidates >= criteria["min_contact_frames"]
        and whole_sample
        and not unavailable
        and n_sampled > 0
        and n_projected_pass >= fraction * n_sampled
        and not blind
    )
    cameras_report = {
        c: {
            "n_frames_in_view": r["n_frames_in_view"],
            "distance_px": _distribution(r["distances"], r["n_beyond"]),
            "depth_residual_m": _distribution(r["residuals"]),
            "landing_classes": dict(sorted(r["landing"].items())),
        }
        for c, r in per_camera.items()
        if c in cameras or r["landing"] or r["distances"]
    }
    return {
        "pass": steps_ok and projection_ok,
        "cameras": cameras,
        "steps": {
            "pass": steps_ok,
            "n_frames": n_frames,
            "n_pass": n_pass,
            "layers": layers,
            "files_without_frame_count": no_count,
            "failed_by_reason": dict(by_reason),
            "failed_frames": failed,
            "sample": {
                "seed": seed,
                "n": len(steps_sample),
                "n_pass": int(sum(bool(judged[rel][i]) for rel, i in steps_sample)),
            },
        },
        "projection": {
            "pass": projection_ok,
            "seed": seed,
            "n_candidates": n_candidates,
            "n_sampled": n_sampled,
            "sample_override": sample,
            "n_pass": n_projected_pass,
            "statuses": dict(sorted(statuses.items())),
            "n_points_projected": n_points,
            "n_points_in_view": n_in_view,
            "n_points_on_object": n_on,
            "per_camera": cameras_report,
            "cameras_without_points": blind,
            "unavailable_files": unavailable,
            "failed_frames": projection_failures,
        },
        "criterion": {
            "min_pass_fraction": fraction,
            "layers": layers,
            "sample_frames": wanted,
            "min_contact_frames": criteria["min_contact_frames"],
            "projection_radius_px": radius,
            "min_point_fraction": min_points,
            "distance_reach_px": reach,
            "projection_tolerance_provenance": criteria[
                "projection_tolerance_provenance"
            ],
        },
    }


# --- diversity ------------------------------------------------------------------------


def _leaves(value: Any) -> dict[str, Any]:
    """A nested mapping as {"a.b.c": leaf}, in its order; lists stay leaves."""
    out: dict[str, Any] = {}
    stack = [("", value)]
    while stack:
        path, item = stack.pop()
        if not isinstance(item, dict):
            out[path] = item
            continue
        stack += reversed(
            [(f"{path}.{key}" if path else str(key), v) for key, v in item.items()]
        )
    return out


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _numeric_rows(
    values: list[float], weights: list[int], bins: int, log_ratio: float
) -> list[dict[str, Any]]:
    """Equal-width bins over the values' range, log-spaced when the largest is at least
    log_ratio times the smallest; the last bin is closed."""
    lo, hi = min(values), max(values)
    if lo == hi:
        edges = np.array([lo, hi])
    elif lo > 0 and hi / lo >= log_ratio:
        edges = np.geomspace(lo, hi, bins + 1)
    else:
        edges = np.linspace(lo, hi, bins + 1)
    index = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
    weights_ = np.asarray(weights, dtype=np.int64)
    return [
        {
            "bin": f"[{edges[b]:.6g}, {edges[b + 1]:.6g}]",
            "n_episodes": int((index == b).sum()),
            "n_frames": int(weights_[index == b].sum()),
        }
        for b in range(len(edges) - 1)
    ]


def _categorical_rows(
    pairs: list[tuple[Any, int]], max_categories: int
) -> list[dict[str, Any]]:
    counts: dict[str, list[int]] = {}
    for value, weight in pairs:
        key = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        entry = counts.setdefault(key, [0, 0])
        entry[0] += 1
        entry[1] += weight
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1][1], kv[0]))
    rows = [
        {"value": key, "n_episodes": c[0], "n_frames": c[1]}
        for key, c in ordered[:max_categories]
    ]
    rest = ordered[max_categories:]
    if rest:
        rows.append(
            {
                "value": f"other ({len(rest)} values)",
                "n_episodes": sum(c[0] for _, c in rest),
                "n_frames": sum(c[1] for _, c in rest),
            }
        )
    return rows


def check_diversity(ds: Dataset, criteria: Mapping[str, Any]) -> dict[str, Any]:
    """The object types, and a distribution table per variable of the episodes' facts.

    A type is a frames file's footer object_type, which must be the type of its path.
    The item fails when a frames file was not read, has no valid type, when two types
    are one after lower-casing, one asset is under two types, one episode is in two
    files, or a file's facts lack a required kind.
    """
    minimum = criteria["min_object_types"]
    kinds = list(criteria["required_fact_kinds"])
    frames = _frames_files(ds)
    not_read = [rec.rel for rec in frames if not rec.opened or rec.footer is None]
    read = [rec for rec in frames if rec.opened and rec.footer is not None]

    def label(rec: DataFile) -> str | None:
        value = rec.footer.get("object_type")
        if not name_error(value) and value == rec.id.object_type:
            return value
        return None

    typed = [rec for rec in read if label(rec) is not None]
    without_type = [rec.rel for rec in read if label(rec) is None]
    spellings: dict[str, set[str]] = {}
    assets: dict[str, set[str]] = {}
    episodes: dict[str, list[str]] = {}
    for rec in typed:
        spellings.setdefault(label(rec).strip().lower(), set()).add(label(rec))
        asset = rec.footer.get("object_asset")
        if _is_text(asset):
            assets.setdefault(asset, set()).add(label(rec))
        uid = rec.footer.get("episode_uid")
        if isinstance(uid, str) and uid:
            episodes.setdefault(uid, []).append(rec.rel)
    collisions = {k: sorted(v) for k, v in sorted(spellings.items()) if len(v) > 1}
    shared_assets = {k: sorted(v) for k, v in sorted(assets.items()) if len(v) > 1}
    repeated = {k: v for k, v in sorted(episodes.items()) if len(v) > 1}

    total = sum(rec.n_rows or 0 for rec in typed) or 1
    types: dict[str, dict[str, Any]] = {}
    outcomes: dict[str, list[int]] = {}
    for rec in typed:
        row = types.setdefault(label(rec), {"uids": set(), "n_frames": 0})
        row["uids"].add(rec.id.episode_uid)
        row["n_frames"] += rec.n_rows or 0
        if rec.footer.get("label") in LABELS:
            entry = outcomes.setdefault(rec.footer["label"], [0, 0])
            entry[0] += 1
            entry[1] += rec.n_rows or 0
    object_types = {
        name: {
            "n_episodes": len(row["uids"]),
            "n_frames": row["n_frames"],
            "frame_fraction": round(row["n_frames"] / total, 6),
        }
        for name, row in sorted(types.items())
    }
    labels = {
        name: {
            "n_episodes": outcomes[name][0],
            "n_frames": outcomes[name][1],
            "frame_fraction": round(outcomes[name][1] / total, 6),
        }
        for name in LABELS
        if name in outcomes
    }

    per_variable: dict[str, list[tuple[Any, int]]] = {}
    missing_facts = []
    for rec in typed:
        facts = rec.footer.get("facts")
        lacking = [
            kind
            for kind in kinds
            if not isinstance(facts, dict) or not holds_a_value(facts.get(kind))
        ]
        if lacking:
            missing_facts.append({"file": rec.rel, "missing": lacking})
        if isinstance(facts, dict):
            for path, value in _leaves(facts).items():
                per_variable.setdefault(path, []).append((value, rec.n_rows or 0))
    tables: dict[str, Any] = {}
    for path, pairs in sorted(per_variable.items()):
        values = [v for v, _ in pairs]
        if all(_is_number(v) for v in values):
            rows = _numeric_rows(
                [float(v) for v in values],
                [w for _, w in pairs],
                criteria["bins"],
                criteria["log_bins_min_ratio"],
            )
            kind = "numeric"
        else:
            rows = _categorical_rows(pairs, criteria["max_categories"])
            kind = "categorical"
        for row in rows:
            row["frame_fraction"] = round(row["n_frames"] / total, 6)
        tables[path] = {"kind": kind, "rows": rows}
    clean = not (
        not_read
        or without_type
        or collisions
        or shared_assets
        or repeated
        or missing_facts
    )
    return {
        "pass": clean and len(object_types) >= minimum,
        "n_object_types": len(object_types),
        "object_types": object_types,
        "criterion": {"min_object_types": minimum, "required_fact_kinds": kinds},
        "files_not_read": not_read,
        "files_without_type": without_type,
        "label_collisions": collisions,
        "assets_under_several_types": shared_assets,
        "episodes_in_several_files": repeated,
        "files_missing_facts": missing_facts,
        "labels": labels,
        "tables": tables,
    }


# --- the published tools --------------------------------------------------------------


def check_published_tools(
    ds: Dataset,
    criteria: Mapping[str, Any],
    criteria_path: Path | None,
    criteria_sha256: str | None,
    script_sha256: str,
) -> dict[str, Any]:
    """Whether this run applies the criteria file and runs the script whose sha256
    metadata.yaml lists. The digests are those of the bytes this run read: the criteria
    file's when it was loaded, the script's when the run began."""
    stated = {
        rel: str(ds.tools[rel].get("sha256", "")).lower() if rel in ds.tools else None
        for rel in TOOL_FILES
    }
    script_rel, criteria_rel = TOOL_FILES
    if criteria_path is not None:
        used = Path(criteria_path).resolve()
        same = criteria_sha256 is not None and criteria_sha256 == stated[criteria_rel]
    else:
        used = (ds.root / criteria_rel).resolve()
        try:
            published, criteria_sha256 = load_criteria(used)
        except CheckError:
            published, criteria_sha256 = None, None
        same = criteria_sha256 == stated[criteria_rel] and published == criteria
    return {
        "pass": bool(same) and script_sha256 == stated[script_rel],
        "criteria": {
            "path": str(used),
            "sha256": criteria_sha256,
            "published_sha256": stated[criteria_rel],
            "matches_published": bool(same),
        },
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": script_sha256,
            "published_sha256": stated[script_rel],
            "matches_published": script_sha256 == stated[script_rel],
        },
    }


# --- running --------------------------------------------------------------------------


def environment() -> dict[str, Any]:
    yaml = _yaml()
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
        "pyyaml": yaml.__version__,
        "libyaml": hasattr(yaml, "CSafeLoader"),
        "platform": platform.platform(),
    }


def run_checks(
    root: str | Path,
    criteria: Mapping[str, Any],
    *,
    items: Iterable[str] = ITEMS,
    workers: int = 4,
    checksums: bool = True,
    criteria_path: str | Path | None = None,
    criteria_sha256: str | None = None,
    seed: int | None = None,
    sample: int | None = None,
    progress: IO[str] | None = None,
    script_sha256: str | None = None,
) -> dict[str, Any]:
    """Run the items and return the report, which --json writes.

    The verdict needs every item run to pass and the run to apply the criteria file and
    run the script that metadata.yaml publishes. criteria_path is the file the criteria
    were read from, and criteria_sha256 the digest of the bytes they were read from;
    given a path alone, the file is read, and its digest counts only when it holds the
    criteria. Without a path the criteria must equal the dataset's own. seed, the
    tester's, draws the consistency samples and is required when that item runs; sample
    overrides their size.
    """
    if script_sha256 is None:
        script_sha256 = sha256_file(Path(__file__).resolve())[0]
    root = Path(root).resolve()
    if not root.is_dir():
        raise CheckError(f"dataset root {root} is not a directory")
    items = tuple(items)
    if not items:
        raise CheckError(f"no item to run; the items are {list(ITEMS)}")
    unknown = [item for item in items if item not in ITEMS]
    if unknown:
        raise CheckError(f"unknown item(s) {unknown}; known: {list(ITEMS)}")
    if "consistency" in items and seed is None:
        raise CheckError(
            "the consistency item draws its samples with the tester's seed: pass --seed"
        )
    if sample is not None and not _is_int(sample, 1):
        raise CheckError(f"a sample of {sample!r} frames is not one of at least 1")
    validate_criteria(criteria, items)
    if criteria_path is not None and criteria_sha256 is None:
        loaded, criteria_sha256 = load_criteria(criteria_path)
        if loaded != criteria:
            # Criteria that are not the file's were read from no bytes it holds.
            criteria_sha256 = None
    metadata, metadata_sha256 = read_metadata(root)
    ds = read_dataset(
        root, metadata, workers=workers, checksums=checksums, progress=progress
    )
    results: dict[str, Any] = {}
    run = [item for item in ITEMS if item in items]
    if "scale" in run:
        results["scale"] = check_scale(ds, criteria["scale"])
    if "metadata" in run:
        results["metadata"] = check_metadata(ds, metadata, criteria["metadata"])
    if "consistency" in run:
        floor = criteria.get("metadata", {}).get("required_cameras", CAMERAS)
        results["consistency"] = check_consistency(
            ds,
            metadata,
            criteria["consistency"],
            seed=seed,
            sample=sample,
            floor=floor,
            progress=progress,
        )
    if "diversity" in run:
        results["diversity"] = check_diversity(ds, criteria["diversity"])
    published = check_published_tools(
        ds,
        criteria,
        None if criteria_path is None else Path(criteria_path),
        criteria_sha256,
        script_sha256,
    )
    return {
        "pass": all(r["pass"] for r in results.values()) and published["pass"],
        "root": str(root),
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "items_run": list(results),
        "items_not_run": [item for item in ITEMS if item not in results],
        "published_tools": published,
        "dataset": {
            **{k: metadata.get(k) for k in ("name", "version", "doi", "download_url")},
            "metadata_sha256": metadata_sha256,
        },
        "environment": environment(),
        "ignored_files": {"n": len(ds.ignored), "paths": ds.ignored},
        "items": results,
    }


# --- the report -----------------------------------------------------------------------


def _measured(item: str, r: Mapping[str, Any]) -> tuple[str, str]:
    """The Measured and Criterion cells of an item's row."""
    if item == "scale":
        errors = r["n_files_with_errors"]
        return (
            f"{r['n_frames']:,} frames in {r['n_episodes']:,} episodes "
            f"({r['n_files']:,} files)"
            + (f", {errors:,} with errors" if errors else ""),
            f">= {r['criterion']['min_frames']:,} frames",
        )
    if item == "metadata":
        text = f"{r['n_mismatches']:,} mismatches in {r['n_checked']:,} comparisons"
        if not r["checksums_verified"]:
            text += " (sha256 not checked)"
        return text, f"<= {r['criterion']['max_mismatches']} mismatches, and spec"
    if item == "consistency":
        steps, proj, crit = r["steps"], r["projection"], r["criterion"]
        # In integers: a count metadata.yaml states can be beyond the float range.
        pct = 100 * steps["n_pass"] / steps["n_frames"] if steps["n_frames"] else 0.0
        override = (
            f", sample overridden to {proj['sample_override']}"
            if proj["sample_override"] is not None
            else ""
        )
        return (
            (
                f"steps {steps['n_pass']:,}/{steps['n_frames']:,} frames ({pct:.2f} %), "
                f"sample {steps['sample']['n_pass']}/{steps['sample']['n']}; projection "
                f"{proj['n_pass']}/{proj['n_sampled']} of {proj['n_candidates']:,} frames "
                f"with pressed robot contacts, seed {proj['seed']}{override}"
            ),
            (
                f"{100 * crit['min_pass_fraction']:g} % of frames; sample "
                f"{crit['sample_frames']} of >= {crit['min_contact_frames']} contact frames, "
                f"within {crit['projection_radius_px']:g} px "
                f"({crit['projection_tolerance_provenance']})"
            ),
        )
    problems = sum(
        1
        for key in (
            "files_not_read",
            "files_without_type",
            "label_collisions",
            "assets_under_several_types",
            "episodes_in_several_files",
            "files_missing_facts",
        )
        if r[key]
    )
    return (
        f"{r['n_object_types']} object types, {len(r['tables'])} variable tables"
        + (f"; {problems} kind(s) of problem below" if problems else ""),
        (
            f">= {r['criterion']['min_object_types']} object types; facts "
            f"{', '.join(r['criterion']['required_fact_kinds'])}"
        ),
    )


def _listings(item: str, r: Mapping[str, Any]) -> list[tuple[str, list[Any]]]:
    if item == "scale":
        return [("files with errors", r["files_with_errors"])]
    if item == "metadata":
        return [("mismatches", r["mismatches"])]
    if item == "consistency":
        proj = r["projection"]
        return [
            ("files without a frame count", r["steps"]["files_without_frame_count"]),
            ("step failures", r["steps"]["failed_frames"]),
            (
                "projection unavailable",
                [
                    {"file": k, "missing": v}
                    for k, v in proj["unavailable_files"].items()
                ],
            ),
            ("cameras without points", proj["cameras_without_points"]),
            ("projection failures", proj["failed_frames"]),
        ]
    return [
        ("files not read", r["files_not_read"]),
        ("files without an object type", r["files_without_type"]),
        (
            "labels that are one type",
            [{k: v} for k, v in r["label_collisions"].items()],
        ),
        (
            "assets under several types",
            [{k: v} for k, v in r["assets_under_several_types"].items()],
        ),
        (
            "episodes in several files",
            [{k: v} for k, v in r["episodes_in_several_files"].items()],
        ),
        ("files missing facts", r["files_missing_facts"]),
    ]


def _block(title: str, rows: Mapping[str, Mapping[str, Any]] | list[Any]) -> list[str]:
    lines = ["", title]
    for label, row in rows.items() if isinstance(rows, Mapping) else rows:
        lines.append(
            f"  {str(label)[:48]:48s} {row['n_episodes']:9,d} {row['n_frames']:12,d} "
            f"{100 * row['frame_fraction']:8.2f}%"
        )
    return lines


def _diversity_lines(r: Mapping[str, Any]) -> list[str]:
    """The distribution tables: object types, labels, then one per variable."""
    header = f"  {'':48s} {'episodes':>9s} {'frames':>12s} {'fraction':>9s}"
    lines = _block("diversity: object types", r["object_types"])
    lines.insert(2, header)
    lines += _block("diversity: labels", r["labels"])
    for path, table in r["tables"].items():
        lines += _block(
            f"diversity: {path} ({table['kind']})",
            [(row.get("bin", row.get("value")), row) for row in table["rows"]],
        )
    return lines


def _spec_row(r: Mapping[str, Any]) -> tuple[str, str, str, str]:
    spec, criterion = r["spec"], r["criterion"]
    failed = [
        k
        for k, ok in spec["checks"].items()
        if not ok and k not in ("channels", "cameras")
    ]
    measured = (
        f"{spec['n_channels']} of {len(criterion['required_channels'])} GT channels"
        + (
            f" (missing {', '.join(spec['missing_channels'])})"
            if spec["missing_channels"]
            else ""
        )
        + f", cameras {', '.join(spec['cameras_in_every_episode']) or 'none'}"
        + (f"; failed: {', '.join(failed)}" if failed else "")
    )
    wanted = (
        f"{len(criterion['required_channels'])} channels, "
        f"{', '.join(criterion['required_cameras'])}, {criterion['license']}, DOI, URL"
    )
    return "  spec (in metadata)", measured, wanted, "PASS" if spec["pass"] else "FAIL"


def format_table(report: Mapping[str, Any]) -> str:
    """The report people read: what was checked, one row per item, then the listings."""
    published = report["published_tools"]
    lines = [
        f"{FORMAT} {FORMAT_VERSION} dataset check",
        f"root: {report['root']}",
        f"metadata.yaml: sha256 {report['dataset']['metadata_sha256']}",
    ]
    for key in ("criteria", "script"):
        p = published[key]
        lines.append(
            f"{key}: {p['path']} (sha256 {(p['sha256'] or '?')[:16]}; the published "
            f"one: {'yes' if p['matches_published'] else 'NO'})"
        )
    lines.append(
        "items run: "
        + ", ".join(report["items_run"])
        + (
            f" (not run: {', '.join(report['items_not_run'])})"
            if report["items_not_run"]
            else ""
        )
    )
    if report["ignored_files"]["n"]:
        lines.append(
            f"ignored: {report['ignored_files']['n']:,} files in data/, trajectories/, "
            "videos/ or tools/ that are not the format's (see the JSON report)"
        )
    rows = [("Item", "Measured", "Criterion", "Result")]
    for item, r in report["items"].items():
        rows.append((item, *_measured(item, r), "PASS" if r["pass"] else "FAIL"))
        if item == "metadata":
            rows.append(_spec_row(r))
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    lines.append("")
    for row in rows:
        lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())
    for item, r in report["items"].items():
        for title, listed in _listings(item, r):
            if not listed:
                continue
            more = (
                f", {len(listed) - MAX_LISTED:,} more in the JSON report"
                if len(listed) > MAX_LISTED
                else ""
            )
            lines += ["", f"{item}: {title} ({len(listed):,}{more})"]
            for entry in listed[:MAX_LISTED]:
                lines.append(f"  {json.dumps(jsonable(entry), ensure_ascii=True)}")
    if "diversity" in report["items"]:
        lines += _diversity_lines(report["items"]["diversity"])
    verdict = "PASS" if report["pass"] else "FAIL"
    if not published["pass"]:
        verdict += (
            " (the criteria or the script are not the ones metadata.yaml publishes)"
        )
    elif report["items_not_run"]:
        verdict += f" for the items run; not run: {', '.join(report['items_not_run'])}"
    lines += ["", f"Overall: {verdict}"]
    return "\n".join(lines)


def summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """The RESULT line's payload: the verdict and each item's headline numbers."""
    out: dict[str, Any] = {"pass": report["pass"], "items_run": report["items_run"]}
    for key in ("criteria", "script"):
        p = report["published_tools"][key]
        out[key] = {"sha256": p["sha256"], "matches_published": p["matches_published"]}
    out["metadata_yaml"] = {"sha256": report["dataset"]["metadata_sha256"]}
    for item, r in report["items"].items():
        if item == "scale":
            keys = ("n_frames", "n_episodes", "n_files", "n_files_with_errors")
            out[item] = {"pass": r["pass"], **{k: r[k] for k in keys}}
        elif item == "metadata":
            spec = r["spec"]
            out[item] = {
                "pass": r["pass"],
                "n_mismatches": r["n_mismatches"],
                "n_checked": r["n_checked"],
                "spec": {
                    k: spec[k]
                    for k in (
                        "pass",
                        "n_channels",
                        "missing_channels",
                        "missing_cameras",
                    )
                },
            }
        elif item == "consistency":
            steps, proj = r["steps"], r["projection"]
            out[item] = {
                "pass": r["pass"],
                "steps": {
                    **{k: steps[k] for k in ("pass", "n_frames", "n_pass")},
                    "sample": steps["sample"],
                },
                "projection": {
                    k: proj[k]
                    for k in (
                        "pass",
                        "seed",
                        "n_candidates",
                        "n_sampled",
                        "sample_override",
                        "n_pass",
                        "statuses",
                    )
                },
                "projection_tolerance_provenance": r["criterion"][
                    "projection_tolerance_provenance"
                ],
            }
        else:
            out[item] = {
                "pass": r["pass"],
                "n_object_types": r["n_object_types"],
                "n_files_not_read": len(r["files_not_read"]),
                "n_files_without_type": len(r["files_without_type"]),
                "n_files_missing_facts": len(r["files_missing_facts"]),
            }
    return out


def jsonable(value: Any) -> Any:
    """value as strict JSON can hold it: a float that is not finite becomes "NaN",
    "Infinity" or "-Infinity", a NumPy scalar its Python value, anything else its str."""
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if np is not None and isinstance(value, np.bool_):
        return bool(value)
    if np is not None and isinstance(value, np.integer):
        return int(value)
    if np is not None and isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return (
            "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
        )
    return str(value)


# --- the command line -----------------------------------------------------------------


class _UsageError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        raise _UsageError(message)


def _default_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        description=(
            f"Check a {FORMAT} {FORMAT_VERSION} dataset against its four test items "
            "and print a report."
        ),
        epilog=(
            "exit status: 0 every item run passed with the published criteria and "
            "script; 1 the run completed and failed; 2 the check could not run"
        ),
    )
    parser.add_argument(
        "root", help=f"the dataset folder, the one holding {METADATA_NAME}"
    )
    parser.add_argument(
        "--criteria",
        default=None,
        help=f"acceptance values (default: <root>/{TOOLS_DIR}/{CRITERIA_NAME}); a run "
        f"passes only with the file {METADATA_NAME} publishes",
    )
    parser.add_argument(
        "--only",
        default=",".join(ITEMS),
        help=f"comma-separated items to run, from {','.join(ITEMS)}",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_default_workers(),
        help="threads that hash and read the files",
    )
    parser.add_argument(
        "--no-checksums",
        action="store_true",
        help="skip the sha256 comparisons and read every file without them; a damaged "
        "file is then read as it is, and the metadata item cannot pass",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="the tester's seed, which draws the consistency samples (required when "
        "that item runs)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="frames in each consistency sample (default: the criteria's; a smaller "
        "sample cannot pass unless it holds every candidate frame)",
    )
    parser.add_argument(
        "--json", default=None, help="write the full report to this file"
    )
    return parser


def _refuse(reason: str) -> int:
    print(f"check_dataset: {reason}", file=sys.stderr)
    print("RESULT " + json.dumps({"pass": False, "reason": reason}), flush=True)
    return 2


def _report_path_problem(target: Path) -> str | None:
    """Why --json cannot be written, found before hours of hashing are spent."""
    folder = target.resolve().parent
    if target.is_dir():
        return f"--json {target} is a folder, not a file"
    if not folder.is_dir():
        return f"the folder of --json {target} does not exist"
    if target.exists() and not os.access(target, os.W_OK):
        return f"--json {target} cannot be written"
    if not target.exists() and not os.access(folder, os.W_OK):
        return f"the folder of --json {target} cannot be written to"
    return None


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text)[:2])


def check_environment() -> None:
    """Refuse a run without NumPy, PyArrow and PyYAML, or with one older than the tests
    cover."""
    if _IMPORT_ERROR is not None:
        raise CheckError(_required(_IMPORT_ERROR))
    versions = {"NumPy": np.__version__, "PyArrow": pa.__version__}
    versions["PyYAML"] = _yaml().__version__
    for name, floor in MIN_VERSIONS:
        if _version(versions[name]) < floor:
            raise CheckError(
                f"the check needs {name} {'.'.join(map(str, floor))} or newer; this is "
                f"{versions[name]}"
            )


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        # Text read from a damaged file is written as escapes, never as an error.
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.sample is not None and args.sample < 1:
            parser.error(f"argument --sample: {args.sample} is below 1")
    except _UsageError as exc:
        return _refuse(str(exc))
    try:
        check_environment()
        items = tuple(
            dict.fromkeys(i.strip() for i in args.only.split(",") if i.strip())
        )
        if not items:
            raise CheckError(f"--only names no item; the items are {', '.join(ITEMS)}")
        if "consistency" in items and args.seed is None:
            raise CheckError(
                "the consistency item draws its samples with the tester's seed: pass "
                "--seed"
            )
        problem = _report_path_problem(Path(args.json)) if args.json else None
        if problem:
            raise CheckError(problem)
        if args.no_checksums:
            print(
                "check_dataset: --no-checksums reads every file without comparing its "
                "sha256; a damaged file is then read as it is, and the metadata item "
                "cannot pass",
                file=sys.stderr,
            )
        script_sha256 = sha256_file(Path(__file__).resolve())[0]
        root = Path(args.root)
        criteria_path = (
            Path(args.criteria) if args.criteria else root / TOOLS_DIR / CRITERIA_NAME
        )
        criteria, criteria_sha256 = load_criteria(criteria_path)
        report = run_checks(
            root,
            criteria,
            items=items,
            workers=max(1, args.workers),
            checksums=not args.no_checksums,
            criteria_path=criteria_path,
            criteria_sha256=criteria_sha256,
            seed=args.seed,
            sample=args.sample,
            progress=sys.stderr,
            script_sha256=script_sha256,
        )
        report["criteria"] = criteria
        table = format_table(report)
        result = "RESULT " + json.dumps(
            jsonable(summary(report)), ensure_ascii=True, allow_nan=False
        )
        text = None
        if args.json:
            text = json.dumps(
                jsonable(report), indent=2, ensure_ascii=True, allow_nan=False
            )
    except CheckError as exc:
        return _refuse(str(exc))
    except KeyboardInterrupt:
        return _refuse("the check was interrupted")
    except Exception as exc:  # noqa: BLE001, every run ends with a RESULT line
        return _refuse(f"the check stopped: {type(exc).__name__}: {exc}")
    print(table)
    # The verdict is out before the report is written: a write that fails cannot hide it.
    print(result, flush=True)
    if text is not None:
        try:
            Path(args.json).write_text(text + "\n", encoding="utf-8")
        except OSError as exc:
            print(
                f"check_dataset: the report could not be written to {args.json}: {exc}",
                file=sys.stderr,
            )
            return 2
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
