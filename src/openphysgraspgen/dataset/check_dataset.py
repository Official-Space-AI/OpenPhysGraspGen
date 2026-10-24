# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
"""The public dataset's format, in the file that ships with every dataset.

Every dataset carries this file as ``tools/check_dataset.py``, so it imports nothing
from ``openphysgraspgen`` and needs only Python 3.10 or newer, NumPy, PyArrow and PyYAML.
``openphysgraspgen.dataset`` imports the format from here, so the writer and the check
cannot disagree about a file or a column. The format is documented in
``docs/dataset/format.md``.
"""

from __future__ import annotations

import re
import struct
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
FILE_KINDS = ("frames", "trajectory", "video")
# Object types, episode ids and camera names become path components.
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

# A column is (dtype, shape). The shape is () for one value, (n,) for a fixed-size
# list, "image" for one value per pixel of the camera's image, ("tets", k) for k
# values per tetrahedron and "contacts" for one struct per contact. Names with
# "{camera}" are repeated for every camera.
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
    "camera.{camera}.intrinsics": ("float64", (9,)),
    "camera.{camera}.world_to_camera": ("float64", (16,)),
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
# Trajectory columns that are NaN where the object's state is not finite.
NOT_FINITE_COLUMNS = ("object.centroid", "object.bbox_min", "object.bbox_max")

CONVENTIONS = {
    "units": "SI: metres, seconds, newtons, pascals, radians",
    "world_frame": "right-handed, z up",
    "quaternion": "x, y, z, w",
    "frames": (
        "a frame is one simulation step; sim_step counts steps at recorded_fps from "
        "the episode start, the first recorded step being 1; a published frame's step "
        "is a multiple of recorded_fps / fps"
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
        "material model; NaN where not valid"
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
    if isinstance(shape, tuple) and len(shape) == 1:
        return pa.list_(value, shape[0])
    raise ValueError(f"unknown column shape {shape!r}")


# --- MP4 ------------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoTrack:
    n_frames: int
    timescale: int
    frame_durations: tuple[int, ...]

    @property
    def fps(self) -> float | None:
        if len(self.frame_durations) != 1 or self.frame_durations[0] <= 0:
            return None
        return self.timescale / self.frame_durations[0]


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
        if hdlr is None or moov[hdlr[0] + 8 : hdlr[0] + 12] != b"vide":
            continue
        mdhd = _child(moov, *mdia, b"mdhd")
        minf = _child(moov, *mdia, b"minf")
        stbl = None if minf is None else _child(moov, *minf, b"stbl")
        if mdhd is None or stbl is None:
            raise ValueError("a video track without mdhd or stbl")
        version = moov[mdhd[0]]
        timescale = struct.unpack_from(
            ">I", moov, mdhd[0] + (20 if version == 1 else 12)
        )[0]
        stsz = _child(moov, *stbl, b"stsz")
        stz2 = _child(moov, *stbl, b"stz2")
        if stsz is not None:
            n_frames = struct.unpack_from(">I", moov, stsz[0] + 8)[0]
        elif stz2 is not None:
            n_frames = struct.unpack_from(">I", moov, stz2[0] + 8)[0]
        else:
            raise ValueError("a video track without a sample size table")
        stts = _child(moov, *stbl, b"stts")
        if stts is None:
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
        tracks.append(VideoTrack(n_frames, timescale, tuple(durations)))
    return tracks
