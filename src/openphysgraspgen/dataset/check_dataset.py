# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
"""The public dataset's format, in the file that ships with every dataset.

Every dataset carries this file as ``tools/check_dataset.py``, so it imports nothing
from ``openphysgraspgen`` and needs only Python 3.10 or newer, NumPy, PyArrow and PyYAML.
``openphysgraspgen.dataset`` imports the format from here, so the writer and the check
cannot disagree about a file, a column or a value rule. The format is documented in
``docs/dataset/format.md``.
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

import hashlib
import json
import math
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
# metadata.yaml's doi is a bare DOI and its download_url an https URL, in ASCII.
DOI_PATTERN = re.compile(r"10\.[0-9]{4,9}(?:\.[0-9]+)*/[!-~]+")
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
    if isinstance(value, Mapping):
        return any(_has_value(item) for item in value.values())
    return False


def _has_value(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(_has_value(item) for item in value.values())
    return value is not None


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


def parse_footer(raw: bytes | str) -> dict[str, Any]:
    """An episode's metadata from its footer value: one JSON object in UTF-8 whose
    numbers are finite floats and int64 integers and whose objects repeat no key.

    Raises ValueError with the reason, which reads after "its metadata".
    """
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    except UnicodeDecodeError as exc:
        raise ValueError(f"is not UTF-8: {exc}") from None
    try:
        value = json.loads(
            text,
            parse_constant=_json_constant,
            parse_float=_json_float,
            parse_int=_json_int,
            object_pairs_hook=_json_object,
        )
    except RecursionError:
        raise ValueError("is not JSON: it nests too deeply") from None
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
