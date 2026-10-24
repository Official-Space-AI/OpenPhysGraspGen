# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
"""The public dataset's format, in the file that ships with every dataset.

Every dataset carries this file as ``tools/check_dataset.py``, so it imports nothing
from ``physgraspgen`` and needs only Python 3.10 or newer, NumPy, PyArrow and PyYAML.
``physgraspgen.dataset`` imports the format from here, so the writer and the check
cannot disagree about a file, a column or a value rule. The format is documented in
``docs/dataset/format.md``.
"""

from __future__ import annotations

import json
import math
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa

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
    "facts",
    "task",
    "header",
    "generation_provenance",
    "export_provenance",
    "source_n_frames",
    "source_cut_at_row",
)
# Object types, episode ids and camera names become path components.
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# metadata.yaml's doi is a bare DOI and its download_url an https URL.
DOI_PATTERN = re.compile(r"10\.\d{4,9}(?:\.\d+)*/\S+")
URL_PATTERN = re.compile(r"https://\S+")

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

# Value rules, which the functions below apply. Every float value of a column of
# FINITE_GROUPS, contact forces and points included, is finite, except where the names
# below allow NaN. A mask value indexes SEGMENTATION_CLASSES, stress.von_mises is NaN
# exactly where stress.valid is false, and the camera matrices are a pinhole K and a
# rigid transform. Depth and stress.volume_ratio have no value rule.
FINITE_GROUPS = ("robot", "object", "camera", "force", "deformation_gradient")
# Finite or NaN: the jaw opening at first contact is NaN until the pads carry force.
MAY_BE_NAN = ("robot.grip_contact_width_m",)
# In the trajectory, NaN exactly where finite is false, where the robot's values may be
# NaN too.
NOT_FINITE_COLUMNS = ("object.centroid", "object.bbox_min", "object.bbox_max")
# In the trajectory, both NaN on a step without a force layer, such as the step at which
# the simulation blew up, and both finite elsewhere.
FORCE_LAYER_COLUMNS = ("force.gripper_force_mag", "force.ground_force_mag")
# How far a world-to-camera rotation may be from orthonormal with determinant +1.
RIGID_TOLERANCE = 1e-6

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
        "one row per recorded step from step 1 to the last; where finite is false the "
        "object's state is NaN and the robot's may be; the force magnitudes are NaN on "
        "a step without a force layer"
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
) -> pa.DataType:
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
) -> pa.Schema:
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


def trajectory_schema() -> pa.Schema:
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


def finite_error(name: str, value: object, nan: bool = False) -> str | None:
    """Why a column's value breaks the rule that its floats are finite, or None.

    nan allows NaN as well, as a MAY_BE_NAN column always does.
    """
    array = np.asarray(value)
    if array.dtype.kind != "f" or name.split(".", 1)[0] not in FINITE_GROUPS:
        return None
    if nan or name in MAY_BE_NAN:
        return "must be finite or NaN" if np.isinf(array).any() else None
    return None if np.isfinite(array).all() else "must be finite"


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


def trajectory_row_error(row: Mapping[str, object], previous: int | None) -> str | None:
    """Why a trajectory row breaks the format's rules, or None.

    row maps every column to its value, and previous is the previous row's sim_step,
    None for the first row. The trajectory holds every recorded step from step 1.
    """
    step = row["sim_step"]
    expected = 1 if previous is None else previous + 1
    if step != expected:
        return (
            f"sim_step {step} is not {expected}; the trajectory holds every recorded "
            "step from step 1"
        )
    if len({bool(np.isnan(row[name])) for name in FORCE_LAYER_COLUMNS}) > 1:
        return (
            f"{' and '.join(FORCE_LAYER_COLUMNS)} must both be NaN, on a step without "
            "a force layer, or both finite"
        )
    finite = bool(row["finite"])
    for name, value in row.items():
        if name not in NOT_FINITE_COLUMNS:
            nan = name in FORCE_LAYER_COLUMNS or (
                not finite and name.startswith("robot.")
            )
            problem = finite_error(name, value, nan)
            if problem:
                return f"{name} {problem}"
        elif finite and not np.isfinite(value).all():
            return f"{name} must be finite where finite is true"
        elif not finite and not np.isnan(value).all():
            return f"{name} must be NaN where finite is false"
    return None


def intrinsics_error(intrinsics: object) -> str | None:
    """Why a matrix is not a pinhole K [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], or None."""
    k = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(k).all():
        return "is not finite"
    if k[0, 1] != 0 or k[1, 0] != 0:
        return "is not a pinhole K: K[0][1] (skew) and K[1][0] must be 0"
    if not np.array_equal(k[2], (0.0, 0.0, 1.0)):
        return "is not a pinhole K: its last row must be 0 0 1"
    if not (k[0, 0] > 0 and k[1, 1] > 0):
        return "is not a pinhole K: its focal lengths must be positive"
    return None


def world_to_camera_error(world_to_camera: object) -> str | None:
    """Why a matrix is not a rigid 4 x 4 transform, or None."""
    t = np.asarray(world_to_camera, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(t).all():
        return "is not finite"
    if not np.array_equal(t[3], (0.0, 0.0, 0.0, 1.0)):
        return "is not a rigid transform: its last row must be 0 0 0 1"
    r = t[:3, :3]
    deviation = np.abs(r @ r.T - np.eye(3)).max()
    if deviation > RIGID_TOLERANCE:
        return (
            "is not a rigid transform: its rotation is not orthonormal, |R R^T - I| "
            f"reaches {deviation:.2g} (at most {RIGID_TOLERANCE:g})"
        )
    det = np.linalg.det(r)
    if abs(det - 1.0) > RIGID_TOLERANCE:
        return (
            f"is not a rigid transform: its rotation's determinant is {det:.8g}, "
            f"not +1 (within {RIGID_TOLERANCE:g})"
        )
    return None


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
        if not _has_value(facts[kind]):
            return f"facts.{kind} holds no value but null"
    return None


# --- MP4 ------------------------------------------------------------------------------

# The one video coding the format allows: AV1 in an av01 sample entry, 8-bit 4:2:0.
VIDEO_CODEC = "av01"
VIDEO_BIT_DEPTH = 8
VIDEO_CHROMA = "4:2:0"
# av1C's (chroma_subsampling_x, chroma_subsampling_y) by name.
CHROMA_SUBSAMPLING = {
    (1, 1): "4:2:0",
    (1, 0): "4:2:2",
    (0, 0): "4:4:4",
    (0, 1): "4:4:0",
}


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

    @property
    def fps(self) -> float | None:
        if len(self.frame_durations) != 1 or self.frame_durations[0] <= 0:
            return None
        return self.timescale / self.frame_durations[0]


def video_error(
    track: VideoTrack,
    size: tuple[int, int],
    fps: float,
    n_frames: int | None = None,
) -> str | None:
    """Why a video track is not a camera's video in the format, or None.

    size is the camera's (height, width), fps the dataset's frame rate.
    """
    if track.codec != VIDEO_CODEC:
        return f"is coded as {track.codec}, not AV1 ({VIDEO_CODEC})"
    if (track.bit_depth, track.chroma) != (VIDEO_BIT_DEPTH, VIDEO_CHROMA):
        return (
            f"is {track.bit_depth}-bit {track.chroma}, "
            f"not {VIDEO_BIT_DEPTH}-bit {VIDEO_CHROMA}"
        )
    height, width = size
    if (track.height, track.width) != (height, width):
        return (
            f"is {track.height} x {track.width} pixels, not the camera's "
            f"{height} x {width}"
        )
    if track.fps is None or abs(track.fps - fps) > 1e-9 * fps:
        return f"does not run at a constant {fps} fps"
    if n_frames is not None and track.n_frames != n_frames:
        return f"holds {track.n_frames} frames, not {n_frames}"
    return None


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


def _sample_entry(
    moov: bytes, stsd: tuple[int, int]
) -> tuple[str, int, int, int | None, str | None]:
    """A video track's codec, width, height, bit depth and chroma subsampling."""
    start, end = stsd
    if start + 8 > end:
        raise ValueError("truncated sample description")
    (count,) = struct.unpack_from(">I", moov, start + 4)
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


def video_tracks(path: str | Path) -> list[VideoTrack]:
    """The video tracks of an MP4 file, read from its index without decoding."""
    moov = _read_moov(Path(path))
    header = 16 if struct.unpack_from(">I", moov)[0] == 1 else 8
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
        # mdhd version 1 holds 64-bit times before the timescale.
        at = mdhd[0] + (20 if mdhd[0] < mdhd[1] and moov[mdhd[0]] == 1 else 12)
        if at + 4 > mdhd[1]:
            raise ValueError("a truncated mdhd box")
        timescale = struct.unpack_from(">I", moov, at)[0]
        stsd = _child(moov, *stbl, b"stsd")
        if stsd is None:
            raise ValueError("a video track without a sample description")
        codec, width, height, depth, chroma = _sample_entry(moov, stsd)
        sizes = _child(moov, *stbl, b"stsz") or _child(moov, *stbl, b"stz2")
        if sizes is None or sizes[0] + 12 > sizes[1]:
            raise ValueError("a video track without a sample size table")
        n_frames = struct.unpack_from(">I", moov, sizes[0] + 8)[0]
        stts = _child(moov, *stbl, b"stts")
        if stts is None or stts[0] + 8 > stts[1]:
            raise ValueError("a video track without a time-to-sample table")
        (n_entries,) = struct.unpack_from(">I", moov, stts[0] + 4)
        if stts[0] + 8 + 8 * n_entries > stts[1]:
            raise ValueError("truncated time-to-sample table")
        durations = []
        counted = 0
        for i in range(n_entries):
            count, delta = struct.unpack_from(">II", moov, stts[0] + 8 + 8 * i)
            counted += count
            if delta not in durations:
                durations.append(delta)
        if counted != n_frames:
            raise ValueError("the time-to-sample table and the sample count disagree")
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
            )
        )
    return tracks
