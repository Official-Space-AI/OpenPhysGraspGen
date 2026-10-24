# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
"""Builders of episodes, datasets and MP4 indexes that the dataset tests share."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("yaml")

import pyarrow.parquet as pq
from mp4_index import box as _box  # noqa: F401
from mp4_index import minimal_mp4, sample_entry  # noqa: F401

from openphysgraspgen.dataset import DatasetSpec, episode_metadata
from openphysgraspgen.dataset import check_dataset as fmt

TESTS = Path(__file__).resolve().parent
# A non-square image, so that a swapped width and height cannot pass.
H, W = 64, 96
T = 5
NO_RGB = tuple(c for c in fmt.CHANNELS if c != "rgb")
FACTS = {
    "material": {"E_pa": 1.0e5, "nu": 0.45},
    "placement": {"yaw_deg": 30.0, "offset_m": [0.01, -0.02]},
    "lighting": {"elevation_deg": 45.0},
    "camera": {"front_distance_m": 0.6},
}
TASK = {"name": "lift_hold_place"}
# Solid colours far apart, one per frame and camera of a short video.
PALETTE = np.array(
    [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (0, 128, 128),
    ],
    dtype=np.uint8,
)
# Contacts per frame: with two frames per row group, the first group holds none.
CONTACTS = (0, 0, 1, 2, 0, 3)
DROP = object()
POSIX = pytest.mark.skipif(sys.platform == "win32", reason="uses a shell script")


def _has_av1():
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    encoders = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return "libsvtav1" in encoders


needs_av1 = pytest.mark.skipif(not _has_av1(), reason="needs ffmpeg with libsvtav1")


def make_spec(channels=NO_RGB, frames_per_row_group=2, version="0.0.1", **kwargs):
    return DatasetSpec(
        name="unit-test",
        version=version,
        fps=30,
        recorded_fps=60,
        image_size={"front": (H, W), "wrist": (H, W)},
        channels=channels,
        frames_per_row_group=frames_per_row_group,
        **kwargs,
    )


def colour(step, camera):
    return PALETTE[(2 * step + fmt.CAMERAS.index(camera)) % len(PALETTE)]


def intrinsics(camera):
    f = 50.0 if camera == "front" else 40.0
    return np.array([[f, 0.0, (W - 1) / 2], [0.0, 1.1 * f, (H - 1) / 2], [0, 0, 1.0]])


def world_to_camera(step, camera):
    angle = 0.1 * step + (0.5 if camera == "wrist" else 0.0)
    c, s = np.cos(angle), np.sin(angle)
    matrix = np.eye(4)
    matrix[:3, :3] = [[c, -s, 0.0], [0.0, 0.0, -1.0], [s, c, 0.0]]
    matrix[:3, 3] = [0.1 * step, -0.2, 0.6]
    return matrix


def deformation(step):
    f = np.tile(np.eye(3), (T, 1, 1))
    f[:, 0, 1] = 0.01 * step + 0.001 * np.arange(T)
    f[:, 2, 0] = -0.02
    return f


def contacts(step, n):
    force = np.arange(3 * n, dtype=float).reshape(n, 3) + step
    return {
        "force.contact_force": force,
        "force.contact_point_world": 0.01 * force,
        "force.contact_particle": np.arange(n) + 10 * step,
        "force.contact_is_ground": np.arange(n) % 2 == 0,
        "force.contact_is_robot": np.arange(n) % 2 == 1,
    }


def state(step):
    return {
        "robot.ee_pose_cmd": np.array([0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]),
        "robot.ee_pose": np.array([0.01 * step, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]),
        "robot.arm_q_setpoint": np.linspace(0.0, 1.0, 6) + step,
        "robot.arm_q": np.linspace(0.0, 1.0, 6) + step + 0.001,
        "robot.gripper_width_cmd_m": 0.05,
        "robot.gripper_width_setpoint_m": 0.05,
        "robot.gripper_width_m": 0.05 - 0.001 * step,
        "robot.grip_contact_width_m": np.nan if step < 6 else 0.04,
        "robot.waypoint": step // 4,
        "robot.phase": "lift",
        "object.centroid": np.array([0.0, 0.0, 0.01 * step]),
        "object.bbox_min": np.array([-0.02, -0.02, 0.01 * step - 0.02]),
        "object.bbox_max": np.array([0.02, 0.02, 0.01 * step + 0.02]),
        "object.n_robot_contacts": 2,
    }


def make_frame(step, spec, n_contacts=2):
    rng = np.random.default_rng(step)
    frame = {"sim_step": step, "source_frame_index": step - 1}
    for camera in spec.cameras:
        if "rgb" in spec.channels:
            frame[f"rgb.{camera}.image"] = np.broadcast_to(
                colour(step, camera), (*spec.image_size[camera], 3)
            ).copy()
        if "depth" in spec.channels:
            depth = rng.random(spec.image_size[camera], dtype=np.float32)
            depth[0, :2] = (np.inf, np.nan)  # depth has no value rule
            frame[f"depth.{camera}.depth_m"] = depth
        if "segmentation" in spec.channels:
            frame[f"segmentation.{camera}.mask"] = rng.integers(
                0, 6, spec.image_size[camera], dtype=np.uint8
            )
        frame[f"camera.{camera}.intrinsics"] = intrinsics(camera)
        frame[f"camera.{camera}.world_to_camera"] = world_to_camera(step, camera)
    if "force" in spec.channels:
        frame.update(contacts(step, n_contacts))
        frame["force.gripper_net_force"] = np.array([0.1, 0.2, 0.3]) * step
        frame["force.ground_net_force"] = np.array([-0.1, 0.0, 9.8])
        frame["force.gripper_force_mag"] = 1.5 * step
        frame["force.ground_force_mag"] = 0.5
    if "stress" in spec.channels:
        valid = (np.arange(T) + step) % 3 != 0
        frame["stress.von_mises"] = np.where(
            valid, 1000.0 * step + np.arange(T), np.nan
        ).astype(np.float32)
        frame["stress.valid"] = valid
        frame["stress.volume_ratio"] = 1.0 + 0.001 * np.arange(T)
    if "deformation_gradient" in spec.channels:
        frame["deformation_gradient.F"] = deformation(step)
    frame.update(state(step))
    frame.update({"object.n_soft_contacts": n_contacts, "object.n_rigid_contacts": 0})
    return frame


def make_step(step, finite=True):
    row = state(step)
    row.update(
        {
            "sim_step": step,
            "finite": finite,
            "force.gripper_force_mag": 1.5 * step,
            "force.ground_force_mag": 0.5,
        }
    )
    if not finite:
        row.update(dict.fromkeys(fmt.NOT_FINITE_COLUMNS))
    return row


# What a step whose object state is not finite changes, as make_step writes it.
NON_FINITE = {"finite": False, **dict.fromkeys(fmt.NOT_FINITE_COLUMNS)}
FORCE_MAGNITUDES = ("force.gripper_force_mag", "force.ground_force_mag")


def open_episode(recorder, uid="ep-0", object_type="apple", **kwargs):
    kwargs = {
        "object_asset": f"{object_type}_01",
        "n_tets": T,
        "label": "success",
        "facts": FACTS,
        "source_n_frames": None,
        "task": TASK,
        **kwargs,
    }
    return recorder.episode(episode_uid=uid, object_type=object_type, **kwargs)


def fill(episode, n_frames=3, frame=None):
    """Frames at steps 2, 4, ... and the trajectory up to the last of them."""
    spec = episode.spec
    frame = frame or make_frame
    frames = [
        frame(spec.stride * (i + 1), spec, CONTACTS[i % len(CONTACTS)])
        for i in range(n_frames)
    ]
    for each in frames:
        episode.add_frame(each)
    for step in range(1, frames[-1]["sim_step"] + 1):
        episode.add_step(make_step(step, finite=step != 3))
    return frames


def write_episode(
    recorder, uid="ep-0", object_type="apple", n_frames=5, frame=None, **kwargs
):
    with open_episode(recorder, uid, object_type, **kwargs) as episode:
        return fill(episode, n_frames, frame)


def rewrite(path, change):
    """Rewrite a Parquet file through change(table, footer) -> (table, footer)."""
    table = pq.read_table(path)
    footer = episode_metadata(path)
    table, footer = change(table, footer)
    metadata = None if footer is None else {fmt.METADATA_KEY: json.dumps(footer)}
    pq.write_table(table.replace_schema_metadata(metadata), path)


def nested(depth, leaf=1.0):
    """leaf inside depth mappings: {"a": {"a": ... leaf}}."""
    for _ in range(depth):
        leaf = {"a": leaf}
    return leaf


def edit(array, index, value):
    out = np.array(array, dtype=float)
    out[index] = value
    return out


def files_under(root):
    return sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
    )


def _fake_ffmpeg(directory, script):
    path = directory / "ffmpeg"
    path.write_text(f"#!/bin/sh\n{script}\n")
    path.chmod(0o755)
    return str(path)


def fake_encoder(directory, video, version="ffmpeg version 0.0-test"):
    """An ffmpeg that lists libsvtav1 and writes the MP4 video for every encoding, or
    fails every encoding when video is None."""
    directory.mkdir(parents=True, exist_ok=True)
    encode = "echo 'cannot encode' >&2; exit 1"
    if video is not None:
        (directory / "ffmpeg.mp4").write_bytes(video)
        encode = 'for last; do :; done; cp "$0.mp4" "$last"'
    return _fake_ffmpeg(
        directory,
        'case "$2" in\n'
        "  -encoders) echo ' ------'; echo ' V....D libsvtav1  SVT-AV1 (codec av1)';;\n"
        f"  -version) echo '{version}';;\n"
        f"  *) cat > /dev/null; {encode};;\n"
        "esac",
    )


def fake_av1(directory):
    """An ffmpeg that writes, for every encoding, the index ffmpeg with SVT-AV1 writes
    for the frames it reads, and nothing to decode (tests/fake_ffmpeg.py)."""
    directory.mkdir(parents=True, exist_ok=True)
    return _fake_ffmpeg(
        directory, f'exec "{sys.executable}" -S "{TESTS / "fake_ffmpeg.py"}" "$@"'
    )


# --- datasets for the check -----------------------------------------------------------

# A small image: the object fills rows 3 to 12 and columns 6 to 17, the gripper's pads
# the first three columns. Both cameras look along +z from near the origin, so a point
# at z = 1 m with |x|, |y| <= 0.1 m lands on the object.
SIZE = (16, 24)
OBJECT_ROWS, OBJECT_COLS = slice(3, 13), slice(6, 18)
DOI = "10.5281/zenodo.1234567"
URL = "https://huggingface.co/datasets/org/unit-test"
# Types, episode ids and frame counts of the dataset the check tests build.
EPISODES = (
    ("apple", "ep-0", 3),
    ("apple", "ep-1", 2),
    ("banana", "ep-2", 3),
    ("cherry", "ep-3", 2),
)
N_FRAMES = sum(n for _, _, n in EPISODES)


def scene_spec(channels=fmt.CHANNELS, cameras=fmt.CAMERAS, **kwargs):
    return DatasetSpec(
        name="unit-test",
        version="0.0.1",
        fps=30,
        recorded_fps=60,
        image_size=dict.fromkeys(cameras, SIZE),
        channels=channels,
        frames_per_row_group=2,
        **kwargs,
    )


def scene_intrinsics(camera):
    f = 20.0 if camera == "front" else 18.0
    return np.array(
        [[f, 0.0, (SIZE[1] - 1) / 2], [0.0, f, (SIZE[0] - 1) / 2], [0, 0, 1]]
    )


def scene_world_to_camera(camera):
    matrix = np.eye(4)
    matrix[0, 3] = 0.0 if camera == "front" else 0.01
    return matrix


def scene_mask():
    mask = np.zeros(SIZE, dtype=np.uint8)
    mask[OBJECT_ROWS, OBJECT_COLS] = 1
    mask[:, :3] = 3
    return mask


def scene_contacts(step, offset=0.0, pressed=True):
    """Two contacts that press on the robot, then one on the ground; offset moves the
    robot's contacts along x, in metres at 1 m."""
    x = np.array([0.05, -0.05, 0.0]) + np.array([offset, offset, 0.0])
    points = np.stack([x, [0.02, -0.03, 0.2], [1.0, 1.0, 1.0]], axis=1)
    force = np.array([[0.0, 0.0, 2.0 + step], [0.0, 1.0, 0.0], [0.0, 0.0, 9.8]])
    if not pressed:
        force[:2] = 0.0
    return {
        "force.contact_force": force,
        "force.contact_point_world": points,
        "force.contact_particle": np.array([3, 7, 11]),
        "force.contact_is_ground": np.array([False, False, True]),
        "force.contact_is_robot": np.array([True, True, False]),
    }


def scene_frame(step, spec, n_contacts=None, offset=0.0, pressed=True):
    """A frame of the scene: its pressed contacts land on the object in both cameras."""
    frame = make_frame(step, spec)
    for camera in spec.cameras:
        if "rgb" in spec.channels:
            frame[f"rgb.{camera}.image"] = np.broadcast_to(
                colour(step, camera), (*SIZE, 3)
            ).copy()
        if "depth" in spec.channels:
            depth = np.full(SIZE, 2.0, dtype=np.float32)
            depth[OBJECT_ROWS, OBJECT_COLS] = 1.0
            frame[f"depth.{camera}.depth_m"] = depth
        if "segmentation" in spec.channels:
            frame[f"segmentation.{camera}.mask"] = scene_mask()
        frame[f"camera.{camera}.intrinsics"] = scene_intrinsics(camera)
        frame[f"camera.{camera}.world_to_camera"] = scene_world_to_camera(camera)
    if "force" in spec.channels:
        frame.update(scene_contacts(step, offset, pressed))
    frame["object.n_soft_contacts"] = 3
    return frame


def scene_facts(object_type, uid):
    i = int(uid.rsplit("-", 1)[1])
    return {
        "material": {"E_pa": 1.0e4 * 10**i, "model": "yeoh" if i % 2 else "ogden"},
        "placement": {"yaw_deg": 15.0 * i, "offset_m": [0.01 * i, 0.0]},
        "lighting": {"elevation_deg": 30.0 + i},
        "camera": {"front_distance_m": 0.6, "rig": object_type},
    }


def criteria_text(**values):
    """The packaged criteria, with values in place of the published ones."""
    text = Path(fmt.__file__).with_name("check_criteria.yaml").read_text()
    for key, value in values.items():
        lines = text.split("\n")
        (i,) = [n for n, line in enumerate(lines) if line.strip().startswith(f"{key}:")]
        indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
        lines[i] = f"{indent}{key}: {value}"
        text = "\n".join(lines)
    return text


# Values a dataset of EPISODES meets; the others are the published ones.
SMALL = {
    "min_frames": N_FRAMES,
    "sample_frames": 5,
    "min_contact_frames": 4,
    "min_object_types": 3,
}


def build_dataset(
    root,
    *,
    ffmpeg=None,
    episodes=EPISODES,
    channels=fmt.CHANNELS,
    cameras=fmt.CAMERAS,
    criteria=None,
    doi=DOI,
    download_url=URL,
    frame=scene_frame,
):
    """A dataset written by GraspDataRecorder and finalized with small criteria."""
    from openphysgraspgen.dataset import GraspDataRecorder

    spec = scene_spec(channels, cameras)
    recorder = GraspDataRecorder(root, spec, ffmpeg=ffmpeg or "ffmpeg")
    for object_type, uid, n in episodes:
        write_episode(
            recorder,
            uid=uid,
            object_type=object_type,
            n_frames=n,
            frame=frame,
            facts=scene_facts(object_type, uid),
            source_n_frames=2 * n,
        )
    path = Path(root).parent / f"{Path(root).name}-criteria.yaml"
    path.write_text(criteria if criteria is not None else criteria_text(**SMALL))
    return recorder.finalize(doi=doi, download_url=download_url, criteria=path)
