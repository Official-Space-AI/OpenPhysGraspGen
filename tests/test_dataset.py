# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("yaml")

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from openphysgraspgen.dataset import (
    DatasetError,
    DatasetSpec,
    GraspDataRecorder,
    VideoSpec,
    card_front_matter,
    episode_metadata,
    read_frames,
    read_metadata,
    read_trajectory,
    read_video,
    video_tracks,
)
from openphysgraspgen.dataset import check_dataset as fmt
from openphysgraspgen.dataset import video as video_module
from openphysgraspgen.dataset import writer as writer_module

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
                colour(step, camera), (H, W, 3)
            ).copy()
        if "depth" in spec.channels:
            depth = rng.random((H, W), dtype=np.float32)
            depth[0, :2] = (np.inf, np.nan)  # depth has no value rule
            frame[f"depth.{camera}.depth_m"] = depth
        if "segmentation" in spec.channels:
            frame[f"segmentation.{camera}.mask"] = rng.integers(
                0, 6, (H, W), dtype=np.uint8
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
        "task": TASK,
        **kwargs,
    }
    return recorder.episode(episode_uid=uid, object_type=object_type, **kwargs)


def fill(episode, n_frames=3):
    """Frames at steps 2, 4, ... and the trajectory up to the last of them."""
    spec = episode.spec
    frames = [
        make_frame(spec.stride * (i + 1), spec, CONTACTS[i % len(CONTACTS)])
        for i in range(n_frames)
    ]
    for frame in frames:
        episode.add_frame(frame)
    for step in range(1, frames[-1]["sim_step"] + 1):
        episode.add_step(make_step(step, finite=step != 3))
    return frames


def write_episode(recorder, uid="ep-0", object_type="apple", n_frames=5, **kwargs):
    with open_episode(recorder, uid, object_type, **kwargs) as episode:
        return fill(episode, n_frames)


def rewrite(path, change):
    """Rewrite a Parquet file through change(table, footer) -> (table, footer)."""
    table = pq.read_table(path)
    footer = episode_metadata(path)
    table, footer = change(table, footer)
    metadata = None if footer is None else {fmt.METADATA_KEY: json.dumps(footer)}
    pq.write_table(table.replace_schema_metadata(metadata), path)


def edit(array, index, value):
    out = np.array(array, dtype=float)
    out[index] = value
    return out


def files_under(root):
    return sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
    )


# --- writing and reading -----------------------------------------------------------


def test_an_episode_reads_back_as_written(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    frames = write_episode(recorder)
    files = recorder.files("apple", "ep-0")

    data = read_frames(files.frames)
    assert data["sim_step"].tolist() == [2, 4, 6, 8, 10]
    assert data["sim_time_s"].tolist() == pytest.approx(
        [s / 60 for s in (2, 4, 6, 8, 10)]
    )
    assert data["frame_index"].tolist() == [0, 1, 2, 3, 4]
    assert data["source_frame_index"].tolist() == [1, 3, 5, 7, 9]
    assert data["depth.front.depth_m"].shape == (5, H, W)
    assert data["camera.front.intrinsics"].shape == (5, 3, 3)
    assert data["camera.wrist.world_to_camera"].shape == (5, 4, 4)
    assert data["deformation_gradient.F"].shape == (5, T, 3, 3)
    for i, frame in enumerate(frames):
        for name in (
            "depth.front.depth_m",
            "depth.wrist.depth_m",
            "segmentation.front.mask",
            "segmentation.wrist.mask",
            "camera.front.intrinsics",
            "camera.wrist.intrinsics",
            "camera.front.world_to_camera",
            "camera.wrist.world_to_camera",
            "deformation_gradient.F",
            "stress.von_mises",
            "stress.valid",
            "stress.volume_ratio",
            "force.gripper_net_force",
            "force.gripper_force_mag",
            "robot.ee_pose",
            "robot.arm_q",
            "robot.grip_contact_width_m",
            "object.centroid",
        ):
            np.testing.assert_array_equal(
                data[name][i], np.asarray(frame[name], dtype=data[name].dtype), name
            )
    assert data["force.n_contacts"].tolist() == [0, 0, 1, 2, 0]
    offsets = data["force.contact_offset"]
    assert offsets.tolist() == [0, 0, 0, 1, 3, 3]
    for i, frame in enumerate(frames):
        for key in (
            "force.contact_force",
            "force.contact_point_world",
            "force.contact_particle",
            "force.contact_is_ground",
            "force.contact_is_robot",
        ):
            np.testing.assert_array_equal(
                data[key][offsets[i] : offsets[i + 1]],
                np.asarray(frame[key], dtype=data[key].dtype),
            )
    assert data["robot.phase"].tolist() == ["lift"] * 5
    for layer in ("depth.front", "force", "stress", "deformation_gradient"):
        assert data[f"{layer}.sim_step"].tolist() == data["sim_step"].tolist()

    trajectory = read_trajectory(files.trajectory)
    assert trajectory["sim_step"].tolist() == list(range(1, 11))
    assert trajectory["finite"].tolist() == [s != 3 for s in range(1, 11)]
    assert np.isnan(trajectory["object.centroid"][2]).all()
    for i, step in enumerate(range(1, 11)):
        row = make_step(step)
        for name in ("robot.arm_q", "robot.grip_contact_width_m", "object.bbox_max"):
            if step != 3 or name not in fmt.NOT_FINITE_COLUMNS:
                np.testing.assert_array_equal(trajectory[name][i], row[name], name)

    meta = episode_metadata(files.frames)
    assert set(meta) == set(fmt.EPISODE_KEYS)
    assert meta["format_version"] == fmt.FORMAT_VERSION
    assert (meta["n_frames"], meta["n_steps"], meta["n_tets"]) == (5, 10, T)
    assert (meta["facts"], meta["task"]) == (FACTS, TASK)
    assert (meta["video"], meta["ffmpeg_version"]) == (None, None)
    assert episode_metadata(files.trajectory) == meta

    schema = pq.read_schema(files.frames)
    shapes = {
        name: json.loads(schema.field(name).metadata[b"shape"])
        for name in (
            "camera.front.intrinsics",
            "camera.front.world_to_camera",
            "depth.front.depth_m",
            "deformation_gradient.F",
        )
    }
    assert shapes == {
        "camera.front.intrinsics": [3, 3],
        "camera.front.world_to_camera": [4, 4],
        "depth.front.depth_m": [H, W],
        "deformation_gradient.F": ["n_tets", 3, 3],
    }


def test_columns_can_be_read_alone_across_row_groups(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    frames = write_episode(recorder)
    path = recorder.files("apple", "ep-0").frames
    assert pq.read_metadata(path).num_rows == 5
    data = read_frames(path, columns=["object.centroid"])
    assert list(data) == ["object.centroid"]
    assert data["object.centroid"].shape == (5, 3)

    # The first row group (frames 0 and 1) holds no contacts.
    data = read_frames(path, columns=["force.contacts"])
    assert data["force.contact_offset"].tolist() == [0, 0, 0, 1, 3, 3]
    np.testing.assert_array_equal(data["force.contact_particle"], [60, 80, 81])
    np.testing.assert_array_equal(data["force.contact_is_robot"], [False, False, True])
    data = read_frames(path, columns=["stress.von_mises", "deformation_gradient.F"])
    for i, frame in enumerate(frames):
        np.testing.assert_array_equal(
            data["stress.von_mises"][i], frame["stress.von_mises"]
        )
        np.testing.assert_array_equal(
            data["deformation_gradient.F"][i],
            frame["deformation_gradient.F"].astype(np.float32),
        )
    data = read_frames(path, columns=["depth.wrist.depth_m"])
    np.testing.assert_array_equal(
        data["depth.wrist.depth_m"][3], frames[3]["depth.wrist.depth_m"]
    )


def test_columns_are_encoded_for_their_values(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    write_episode(recorder)
    metadata = pq.read_metadata(recorder.files("apple", "ep-0").frames)
    columns = metadata.row_group(0)
    encodings = {
        columns.column(i).path_in_schema: columns.column(i).encodings
        for i in range(columns.num_columns)
    }
    assert "BYTE_STREAM_SPLIT" in encodings["depth.front.depth_m.list.element"]
    assert "RLE_DICTIONARY" in encodings["segmentation.front.mask.list.element"]
    assert "RLE_DICTIONARY" in encodings["segmentation.wrist.mask.list.element"]
    assert metadata.num_row_groups == 3


def test_the_writer_keeps_copies_of_what_it_is_given(tmp_path):
    spec = make_spec(frames_per_row_group=8)
    recorder = GraspDataRecorder(tmp_path, spec)
    facts = copy.deepcopy(FACTS)
    first = open_episode(recorder, "ep-0", facts=facts)
    facts["material"]["E_pa"] = 2.0e5
    second = open_episode(recorder, "ep-1", facts=facts)
    facts["material"]["E_pa"] = 3.0e5

    # One buffer per column, refilled in place for every frame and step.
    frame = make_frame(2, spec)
    step = make_step(1)
    for episode in (first, second):
        for i in range(4):
            for key, value in make_frame(2 * (i + 1), spec).items():
                if isinstance(value, np.ndarray):
                    frame[key][...] = value
                else:
                    frame[key] = value
            episode.add_frame(frame)
        for s in range(1, 9):
            for key, value in make_step(s).items():
                if isinstance(value, np.ndarray):
                    step[key][...] = value
                else:
                    step[key] = value
            episode.add_step(step)
        episode.close()

    for uid, modulus in (("ep-0", 1.0e5), ("ep-1", 2.0e5)):
        files = recorder.files("apple", uid)
        assert episode_metadata(files.frames)["facts"]["material"]["E_pa"] == modulus
        data = read_frames(files.frames)
        for i in range(4):
            expected = make_frame(2 * (i + 1), spec)
            for name in (
                "depth.front.depth_m",
                "segmentation.wrist.mask",
                "stress.valid",
                "camera.wrist.world_to_camera",
                "force.contact_force",
            ):
                got = (
                    data[name][2 * i : 2 * i + 2]
                    if "contact" in name
                    else data[name][i]
                )
                np.testing.assert_array_equal(
                    got, np.asarray(expected[name], dtype=got.dtype), name
                )
        trajectory = read_trajectory(files.trajectory)
        assert trajectory["object.centroid"][:, 2].tolist() == pytest.approx(
            [0.01 * s for s in range(1, 9)]
        )


def test_the_value_rules_accept_what_the_reader_returns(tmp_path):
    # The dataset's check applies the writer's rules, in check_dataset.py, to read data.
    spec = make_spec()
    recorder = GraspDataRecorder(tmp_path, spec)
    write_episode(recorder)
    files = recorder.files("apple", "ep-0")
    data = read_frames(files.frames)
    for name, values in data.items():
        assert fmt.finite_error(name, values) is None, name
    previous = None
    for i, step in enumerate(data["sim_step"]):
        steps = {k: v[i] for k, v in data.items() if k.endswith("sim_step")}
        assert fmt.frame_steps_error(steps, previous, 30, 60) is None
        previous = step
        for camera in spec.cameras:
            assert fmt.mask_error(data[f"segmentation.{camera}.mask"][i]) is None
            assert fmt.intrinsics_error(data[f"camera.{camera}.intrinsics"][i]) is None
            matrix = data[f"camera.{camera}.world_to_camera"][i]
            assert fmt.world_to_camera_error(matrix) is None
        valid = data["stress.valid"][i]
        assert fmt.stress_error(data["stress.von_mises"][i], valid) is None
    trajectory = read_trajectory(files.trajectory)
    previous = None
    for i in range(len(trajectory["sim_step"])):
        row = {name: values[i] for name, values in trajectory.items()}
        assert fmt.trajectory_row_error(row, previous) is None
        previous = row["sim_step"]


# --- what the writer refuses -------------------------------------------------------

FRAME_REFUSALS = [
    (lambda f: {"unknown": 1}, r"missing \[\], unknown \['unknown'\]"),
    (lambda f: {"source_frame_index": DROP}, r"missing \['source_frame_index'\]"),
    (lambda f: {"sim_step": 4}, "sim_step 4 does not follow 4"),
    (lambda f: {"sim_step": 7}, "sim_step 7 does not follow 4"),
    (lambda f: {"sim_step": 8}, "fps grid, which goes on at 6"),
    (lambda f: {"force.sim_step": 4}, "force was captured at another step"),
    (lambda f: {"robot.arm_q": np.zeros(5)}, r"robot.arm_q must have shape \(6,\)"),
    (
        lambda f: {"depth.front.depth_m": np.zeros((W, H), np.float32)},
        r"depth.front.depth_m must have shape \(64, 96\)",
    ),
    (lambda f: {"robot.waypoint": 1.5}, "robot.waypoint must be integers"),
    (lambda f: {"robot.waypoint": 2**31}, "robot.waypoint does not fit int32"),
    (
        lambda f: {"force.gripper_net_force": np.array([1e39, 0.0, 0.0])},
        "force.gripper_net_force does not fit float32",
    ),
    (
        lambda f: {"segmentation.front.mask": np.full((H, W), 6, dtype=np.uint8)},
        "outside the segmentation classes",
    ),
    (
        lambda f: {"force.contact_particle": np.arange(3)},
        "contact arrays differ in length",
    ),
    (
        lambda f: {"force.contact_force": np.zeros((2, 2))},
        r"force.contact_force must have shape \(M, \*\(3,\)\)",
    ),
    (
        lambda f: {"force.contact_is_robot": np.array([2, 0])},
        "force.contact_is_robot must be boolean",
    ),
    (
        lambda f: {"stress.valid": np.ones(T, dtype=bool)},
        "stress.von_mises must be NaN exactly where stress.valid is false",
    ),
    (
        lambda f: {"stress.von_mises": np.ones(T, dtype=np.float32)},
        "stress.von_mises must be NaN exactly where stress.valid is false",
    ),
    (
        lambda f: {"robot.ee_pose": edit(f["robot.ee_pose"], 3, np.nan)},
        "robot.ee_pose must be finite",
    ),
    (
        lambda f: {"object.centroid": [np.inf, 0.0, 0.0]},
        "object.centroid must be finite",
    ),
    (
        lambda f: {"force.ground_force_mag": np.nan},
        "force.ground_force_mag must be finite",
    ),
    (
        lambda f: {"force.contact_point_world": np.full((2, 3), np.nan)},
        "force.contact_point_world must be finite",
    ),
    (
        lambda f: {
            "deformation_gradient.F": edit(f["deformation_gradient.F"], 0, np.inf)
        },
        "deformation_gradient.F must be finite",
    ),
    (
        lambda f: {"robot.grip_contact_width_m": np.inf},
        "robot.grip_contact_width_m must be finite or NaN",
    ),
    (
        lambda f: {"camera.front.intrinsics": edit(intrinsics("front"), 0, np.nan)},
        "camera.front.intrinsics must be finite",
    ),
    (
        lambda f: {"camera.front.intrinsics": edit(intrinsics("front"), (0, 1), 0.3)},
        "skew",
    ),
    (
        lambda f: {"camera.front.intrinsics": edit(intrinsics("front"), (1, 0), 0.3)},
        "skew",
    ),
    (
        lambda f: {"camera.front.intrinsics": edit(intrinsics("front"), (2, 2), 2.0)},
        "last row must be 0 0 1",
    ),
    (
        lambda f: {"camera.front.intrinsics": edit(intrinsics("front"), (1, 1), -40)},
        "focal lengths must be positive",
    ),
    (
        lambda f: {"camera.wrist.world_to_camera": world_to_camera(6, "wrist").T},
        "last row must be 0 0 0 1",
    ),
    (
        lambda f: {
            "camera.wrist.world_to_camera": world_to_camera(6, "wrist")
            @ np.diag([2.0, 2.0, 2.0, 1.0])
        },
        "rotation is not orthonormal",
    ),
    (
        lambda f: {
            "camera.wrist.world_to_camera": world_to_camera(6, "wrist")
            @ np.diag([-1.0, 1.0, 1.0, 1.0])
        },
        r"determinant is -1, not \+1",
    ),
    (
        lambda f: {
            "camera.wrist.world_to_camera": world_to_camera(6, "wrist")
            @ np.diag([1 + 1e-5, 1 + 1e-5, 1 + 1e-5, 1.0])
        },
        r"not orthonormal, \|R R\^T - I\| reaches 2e-05 \(at most 1e-06\)",
    ),
]


@pytest.mark.parametrize(("change", "message"), FRAME_REFUSALS)
def test_a_bad_frame_is_refused_and_the_episode_goes_on(tmp_path, change, message):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    with open_episode(recorder, label="fail") as episode:
        episode.add_frame(make_frame(4, recorder.spec))
        frame = make_frame(6, recorder.spec)
        frame.update(change(frame))
        frame = {k: v for k, v in frame.items() if v is not DROP}
        with pytest.raises(ValueError, match=message):
            episode.add_frame(frame)
        episode.add_frame(make_frame(6, recorder.spec))
        for step in range(1, 7):
            episode.add_step(make_step(step))
    data = read_frames(recorder.files("apple", "ep-0").frames)
    assert data["sim_step"].tolist() == [4, 6]


@pytest.mark.parametrize("step", [0, 1, 3])
def test_the_first_frame_is_on_the_fps_grid(tmp_path, step):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    with open_episode(recorder) as episode:
        with pytest.raises(ValueError, match=f"frame 0: sim_step {step} is not on"):
            episode.add_frame(make_frame(step, recorder.spec))
        episode.abort()
    assert files_under(tmp_path) == []


STEP_REFUSALS = [
    ({"extra": 1}, r"step 2: missing \[\], unknown \['extra'\]"),
    ({"finite": DROP}, r"step 2: missing \['finite'\]"),
    ({"sim_step": 4}, "sim_step 4 is not 3"),
    ({"sim_step": 2}, "sim_step 2 is not 3"),
    (
        {"object.centroid": [np.nan, 0.0, 0.0]},
        "object.centroid must be finite where finite is true",
    ),
    ({"object.bbox_max": None}, "object.bbox_max must be finite where finite is true"),
    ({"finite": False}, "object.centroid must be NaN where finite is false"),
    (
        {"finite": False, "object.centroid": None, "object.bbox_min": [np.nan] * 3},
        "object.bbox_max must be NaN where finite is false",
    ),
    ({"robot.arm_q": np.full(6, np.nan)}, "robot.arm_q must be finite"),
    ({"robot.gripper_width_m": None}, "robot.gripper_width_m must be finite"),
    ({"robot.gripper_width_m": np.inf}, "robot.gripper_width_m must be finite"),
    ({"robot.grip_contact_width_m": -np.inf}, "must be finite or NaN"),
    ({"force.gripper_force_mag": np.nan}, "force.gripper_force_mag must be finite"),
    ({"force.ground_force_mag": None}, "force.ground_force_mag must be finite"),
    # Where finite is false, only the object's state is NaN.
    (
        {**NON_FINITE, **dict.fromkeys(FORCE_MAGNITUDES)},
        "force.gripper_force_mag must be finite",
    ),
    (
        {**NON_FINITE, "robot.ee_pose_cmd": np.full(7, np.nan)},
        "robot.ee_pose_cmd must be finite",
    ),
    (
        {**NON_FINITE, "robot.gripper_width_cmd_m": None},
        "robot.gripper_width_cmd_m must be finite",
    ),
    ({"robot.waypoint": None}, "robot.waypoint must be integers"),
]


@pytest.mark.parametrize(("change", "message"), STEP_REFUSALS)
def test_a_bad_step_is_refused_and_the_episode_goes_on(tmp_path, change, message):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    with open_episode(recorder) as episode:
        episode.add_frame(make_frame(2, recorder.spec))
        episode.add_frame(make_frame(4, recorder.spec))
        episode.add_step(make_step(1))
        episode.add_step(make_step(2))
        step = {**make_step(3), **change}
        with pytest.raises(ValueError, match=message):
            episode.add_step({k: v for k, v in step.items() if v is not DROP})
        for s in (3, 4):
            episode.add_step(make_step(s))
    trajectory = read_trajectory(recorder.files("apple", "ep-0").trajectory)
    assert trajectory["sim_step"].tolist() == [1, 2, 3, 4]


def test_the_trajectory_starts_at_step_1(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    with open_episode(recorder) as episode:
        episode.add_frame(make_frame(2, recorder.spec))
        with pytest.raises(ValueError, match="step 0: sim_step 2 is not 1"):
            episode.add_step(make_step(2))
        episode.abort()


def test_a_non_finite_step_holds_nan_object_state(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    with open_episode(recorder, source_n_frames=3) as episode:
        frame = make_frame(2, recorder.spec)
        frame["robot.grip_contact_width_m"] = None
        episode.add_frame(frame)
        episode.add_step(make_step(1))
        episode.add_step({**make_step(2), "robot.grip_contact_width_m": None})
        nan = {name: np.full(3, np.nan) for name in fmt.NOT_FINITE_COLUMNS}
        episode.add_step({**make_step(3), "finite": False, **nan})
    files = recorder.files("apple", "ep-0")
    trajectory = read_trajectory(files.trajectory)
    assert trajectory["finite"].tolist() == [True, True, False]
    for name in fmt.NOT_FINITE_COLUMNS:
        assert np.isnan(trajectory[name][2]).all(), name
        assert np.isfinite(trajectory[name][:2]).all(), name
    for name in ("robot.ee_pose", "robot.arm_q", *FORCE_MAGNITUDES):
        np.testing.assert_array_equal(trajectory[name][2], make_step(3)[name], name)
    assert np.isnan(trajectory["robot.grip_contact_width_m"]).all()
    assert np.isnan(read_frames(files.frames)["robot.grip_contact_width_m"]).all()
    meta = episode_metadata(files.trajectory)
    assert meta["n_steps"] == meta["source_n_frames"] == 3


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        ({"uid": "../x"}, ValueError, "episode_uid must match"),
        ({"object_type": 5}, TypeError, "object_type must be a string"),
        ({"object_asset": ""}, ValueError, "object_asset must not be empty"),
        ({"label": "won"}, ValueError, "label must be one of"),
        ({"n_tets": 0}, ValueError, "n_tets must be at least 1"),
        ({"facts": [("material", {})]}, TypeError, "facts must be a mapping"),
        ({"facts": {}}, ValueError, "facts has no material mapping"),
        (
            {"facts": {k: v for k, v in FACTS.items() if k != "lighting"}},
            ValueError,
            "facts has no lighting mapping",
        ),
        ({"facts": {**FACTS, "camera": 0.6}}, ValueError, "no camera mapping"),
        (
            {"facts": {**FACTS, "placement": {"yaw": None, "at": {"x": None}}}},
            ValueError,
            "facts.placement holds no value but null",
        ),
        ({"task": {"tol": float("nan")}}, ValueError, "JSON without NaN"),
        ({"source_n_frames": 2.5}, TypeError, "cannot be interpreted as an integer"),
    ],
)
def test_episode_arguments_are_checked(tmp_path, change, error, message):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    with pytest.raises(error, match=message):
        open_episode(recorder, **change)
    assert files_under(tmp_path) == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"name": " "}, "name must be a nonempty string"),
        ({"fps": 25}, "fps must be positive and divide recorded_fps"),
        ({"fps": 0}, "fps must be positive"),
        ({"channels": ("depth", "smell")}, "unknown channels"),
        ({"image_size": {"top": (H, W)}}, "unknown cameras"),
        ({"image_size": {"front": (0, W)}}, "image size must be positive"),
        (
            {"channels": fmt.CHANNELS, "image_size": {"front": (H, W + 1)}},
            "even width and height",
        ),
        ({"image_size": {}}, "image size of at least one camera"),
        ({"frames_per_row_group": 0}, "frames_per_row_group must be at least 1"),
    ],
)
def test_a_dataset_spec_is_checked(change, message):
    kwargs = {
        "name": "unit-test",
        "version": "0.0.1",
        "fps": 30,
        "recorded_fps": 60,
        "image_size": {"front": (H, W)},
        "channels": NO_RGB,
        **change,
    }
    with pytest.raises(ValueError, match=message):
        DatasetSpec(**kwargs)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"encoder": "libaom-av1"}, "encoder must be libsvtav1"),
        ({"encoder": "librav1e"}, "encoder must be libsvtav1"),
        ({"crf": 0}, "crf must be between 1 and 63"),
        ({"crf": 64}, "crf must be between 1 and 63"),
        ({"gop": 1}, "gop must be at least 2"),
        ({"preset": -1}, "preset must be between 0 and 11"),
        ({"preset": 12}, "preset must be between 0 and 11"),
    ],
)
def test_video_settings_are_ones_the_encoder_applies(change, message):
    with pytest.raises(ValueError, match=message):
        VideoSpec(**change)


def test_a_failed_episode_leaves_nothing(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    with (
        pytest.raises(RuntimeError, match="interrupted"),
        open_episode(recorder, label="fail") as episode,
    ):
        for i in range(3):
            episode.add_frame(make_frame(2 * (i + 1), recorder.spec))
        raise RuntimeError("interrupted")
    assert files_under(tmp_path) == []


def test_an_episode_needs_a_frame(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    episode = open_episode(recorder)
    episode.add_step(make_step(1))
    with pytest.raises(ValueError, match="at least one frame"):
        episode.close()
    assert files_under(tmp_path) == []


def test_the_trajectory_must_cover_every_frame(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    episode = open_episode(recorder, label="fail")
    episode.add_frame(make_frame(2, recorder.spec))
    episode.add_frame(make_frame(4, recorder.spec))
    episode.add_step(make_step(1))
    episode.add_step(make_step(2))
    with pytest.raises(ValueError, match="lacks frame steps, first 4"):
        episode.close()
    assert files_under(tmp_path) == []


def test_an_episode_is_not_written_twice(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    write_episode(recorder)
    with pytest.raises(FileExistsError, match="already written"):
        write_episode(recorder)
    write_episode(recorder, overwrite=True, label="fail")
    assert episode_metadata(recorder.files("apple", "ep-0").frames)["label"] == "fail"
    assert not list(tmp_path.rglob("*.part"))


def test_two_writers_cannot_write_one_episode(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    first = open_episode(recorder)
    with pytest.raises(FileExistsError, match="ep-0 is being written"):
        open_episode(recorder)
    fill(first)
    first.close()
    with pytest.raises(FileExistsError, match="already written"):
        open_episode(recorder)


def test_a_leftover_part_file_is_reported_and_overwrite_replaces_it(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    files = recorder.files("apple", "ep-0")
    for path in (files.frames, files.trajectory):
        path.parent.mkdir(parents=True)
        path.with_name(path.name + ".part").write_bytes(b"left by a crash")
    with pytest.raises(FileExistsError, match="overwrite=True replaces it"):
        open_episode(recorder)
    write_episode(recorder, overwrite=True)
    assert files_under(tmp_path) == [
        "data/apple/ep-0.parquet",
        "trajectories/apple/ep-0.parquet",
    ]
    assert read_frames(files.frames)["sim_step"].tolist() == [2, 4, 6, 8, 10]


def test_claiming_an_episode_removes_its_leftover_part_files(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    trajectory = recorder.files("apple", "ep-0").trajectory
    stale = trajectory.with_name(trajectory.name + ".part")
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"left by a crash")
    episode = open_episode(recorder)
    assert not stale.exists()
    episode.abort()
    assert files_under(tmp_path) == []


def test_a_writer_whose_episode_was_taken_over_leaves_it_alone(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    old = open_episode(recorder, label="fail")
    new = open_episode(recorder, overwrite=True)
    fill(old)
    with pytest.raises(RuntimeError, match="another writer took episode ep-0 over"):
        old.close()
    fill(new)
    new.close()
    files = recorder.files("apple", "ep-0")
    assert episode_metadata(files.frames)["label"] == "success"
    assert files_under(tmp_path) == [
        "data/apple/ep-0.parquet",
        "trajectories/apple/ep-0.parquet",
    ]


@pytest.mark.parametrize("renaming", ["trajectories", "data"])
def test_a_writer_taken_over_while_renaming_publishes_no_file_of_the_other(
    tmp_path, monkeypatch, renaming
):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    old = open_episode(recorder, label="fail")
    fill(old)
    taker = []
    replace = os.replace

    def take_over_first(src, dst):
        if Path(src).parts[-3] == renaming and not taker:
            taker.append(open_episode(recorder, overwrite=True))
        replace(src, dst)

    monkeypatch.setattr(writer_module.os, "replace", take_over_first)
    with pytest.raises(RuntimeError, match="another writer took episode ep-0 over"):
        old.close()
    monkeypatch.undo()
    (new,) = taker
    fill(new)
    files = new.close()
    assert episode_metadata(files.frames)["label"] == "success"
    assert files_under(tmp_path) == [
        "data/apple/ep-0.parquet",
        "trajectories/apple/ep-0.parquet",
    ]
    recorder.finalize()


# --- finalize --------------------------------------------------------------------


def test_finalize_lists_and_hashes_every_file(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    write_episode(recorder, uid="ep-0", object_type="apple")
    write_episode(recorder, uid="ep-1", object_type="banana", n_frames=3)
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "check_criteria.yaml").write_text("scale: {min_frames: 1}\n")
    (tmp_path / "README.md").write_text(card_front_matter() + "\n# unit-test\n")

    metadata = recorder.finalize(
        doi="10.5281/zenodo.1234567",
        download_url="https://huggingface.co/datasets/org/unit-test",
        workers=2,
    )
    assert metadata == read_metadata(tmp_path)
    assert metadata["doi"] == "10.5281/zenodo.1234567"
    assert metadata["counts"]["n_frames"] == 8
    assert metadata["counts"]["n_episodes"] == 2
    assert metadata["counts"]["n_object_types"] == 2
    assert metadata["counts"]["n_files"] == len(metadata["files"]) == 4
    assert metadata["object_types"]["banana"] == {
        "n_episodes": 1,
        "n_frames": 3,
        "assets": ["banana_01"],
    }
    assert {f["kind"] for f in metadata["files"]} == {"frames", "trajectory"}
    assert [f["n_steps"] for f in metadata["files"] if f["kind"] == "trajectory"] == [
        10,
        6,
    ]
    assert metadata["video"] is None
    assert metadata["conventions"]["segmentation_classes"] == list(
        fmt.SEGMENTATION_CLASSES
    )
    assert [t["path"] for t in metadata["tools"]] == ["tools/check_criteria.yaml"]
    lines = (tmp_path / "SHA256SUMS").read_text().splitlines()
    assert len(lines) == len(metadata["files"]) + 1
    for line in lines:
        digest, rel = line.split("  ")
        assert hashlib.sha256((tmp_path / rel).read_bytes()).hexdigest() == digest
    assert (tmp_path / ".gitattributes").read_text().splitlines() == [
        "/metadata.yaml -text",
        "/SHA256SUMS -text",
        "/tools/** -text",
        "*.parquet filter=lfs diff=lfs merge=lfs -text",
        "*.mp4 filter=lfs diff=lfs merge=lfs -text",
    ]
    # A second run finds the files it wrote and the card where it expects them.
    assert recorder.finalize()["counts"] == metadata["counts"]


def test_finalize_states_no_classes_without_masks(tmp_path):
    channels = ("depth", "force", "stress", "deformation_gradient")
    recorder = GraspDataRecorder(tmp_path, make_spec(channels=channels))
    write_episode(recorder)
    metadata = recorder.finalize()
    assert metadata["conventions"]["segmentation_classes"] is None
    assert metadata["channels"] == list(channels)


def test_the_card_front_matter_declares_each_table(tmp_path):
    text = card_front_matter()
    assert text.startswith("---\n") and text.endswith("---\n")
    header = yaml.safe_load(text.strip("-\n"))
    assert header == {
        "license": "cc-by-4.0",
        "configs": [
            {
                "config_name": "frames",
                "data_files": "data/*/*.parquet",
                "default": True,
            },
            {"config_name": "trajectories", "data_files": "trajectories/*/*.parquet"},
        ],
    }


def _case_sensitive(path):
    probe = path / "case-probe"
    probe.mkdir()
    sensitive = not (path / "CASE-PROBE").exists()
    probe.rmdir()
    return sensitive


def _drop_last_frame(table, footer):
    return table.slice(0, table.num_rows - 1), footer


def _damage_a_page(root):
    path = root / "data/apple/ep-0.parquet"
    group = pq.read_metadata(path).row_group(0)
    column = next(
        group.column(i)
        for i in range(group.num_columns)
        if group.column(i).path_in_schema == "depth.front.depth_m.list.element"
    )
    data = bytearray(path.read_bytes())
    data[column.data_page_offset + column.total_compressed_size // 2] ^= 0x01
    path.write_bytes(bytes(data))


def _drop_last_step(table, footer):
    return table.slice(0, table.num_rows - 1), footer


def _footers(root, change):
    for kind in ("data", "trajectories"):
        rewrite(root / kind / "apple/ep-0.parquet", lambda t, f: (t, change(f)))


def _field_metadata(name, metadata):
    def change(table, footer):
        i = table.schema.get_field_index(name)
        field = table.schema.field(i).with_metadata(metadata)
        return table.set_column(i, field, table.column(i)), footer

    return change


def _swap_trajectory_of_an_earlier_write(root):
    recorder = GraspDataRecorder(root, make_spec())
    trajectory = recorder.files("apple", "ep-0").trajectory
    earlier = trajectory.read_bytes()
    write_episode(recorder, overwrite=True, label="fail")
    trajectory.write_bytes(earlier)


FINALIZE_REFUSALS = [
    (
        lambda root: (root / "data/apple/ep-9.parquet.part").write_bytes(b""),
        "data/apple/ep-9.parquet.part is not part of the format",
    ),
    (
        lambda root: (root / "trajectories/banana/ep-1.parquet").unlink(),
        "trajectories/banana/ep-1.parquet is missing",
    ),
    (
        lambda root: shutil.copy(
            root / "trajectories/banana/ep-1.parquet",
            root / "trajectories/banana/ep-7.parquet",
        ),
        "trajectories/banana/ep-7.parquet belongs to no episode",
    ),
    (
        lambda root: rewrite(root / "data/apple/ep-0.parquet", lambda t, f: (t, None)),
        "carries no physgraspgen metadata",
    ),
    (
        lambda root: rewrite(
            root / "data/apple/ep-0.parquet",
            lambda t, f: (t, {**f, "format_version": "1.0"}),
        ),
        "data/apple/ep-0.parquet is not physgraspgen-public 2.0",
    ),
    (
        lambda root: rewrite(
            root / "data/apple/ep-0.parquet",
            lambda t, f: (t, {k: v for k, v in f.items() if k != "video"}),
        ),
        r"its metadata lacks \['video'\]",
    ),
    (
        lambda root: shutil.copy(
            root / "data/apple/ep-0.parquet", root / "data/apple/ep-5.parquet"
        ),
        "data/apple/ep-5.parquet holds episode ep-0 of apple",
    ),
    (
        lambda root: write_episode(
            GraspDataRecorder(root, make_spec()), uid="ep-0", object_type="cherry"
        ),
        "episode ep-0 is under two object types",
    ),
    (
        lambda root: write_episode(
            GraspDataRecorder(root, make_spec()),
            uid="ep-2",
            object_type="cherry",
            object_asset="apple_01",
        ),
        r"object asset apple_01 is under object types \['apple', 'cherry'\]",
    ),
    (
        lambda root: write_episode(
            GraspDataRecorder(root, make_spec(version="0.0.2")), uid="ep-2"
        ),
        "episodes disagree on dataset_version",
    ),
    (
        lambda root: rewrite(
            root / "data/apple/ep-0.parquet",
            lambda t, f: (t, {**f, "facts": {**f["facts"], "lighting": {}}}),
        ),
        "facts.lighting holds no value but null",
    ),
    (
        lambda root: shutil.copy(
            root / "trajectories/banana/ep-1.parquet",
            root / "trajectories/apple/ep-0.parquet",
        ),
        "trajectories/apple/ep-0.parquet: its metadata differs from the frames file's",
    ),
    (
        _swap_trajectory_of_an_earlier_write,
        r"its metadata differs from the frames file's in \['label'\]",
    ),
    (
        lambda root: rewrite(root / "data/apple/ep-0.parquet", _drop_last_frame),
        "holds 4 frames, its metadata says 5",
    ),
    (
        lambda root: rewrite(root / "trajectories/apple/ep-0.parquet", _drop_last_step),
        "trajectories/apple/ep-0.parquet: holds 9 steps, its metadata says 10",
    ),
    (
        lambda root: [shutil.rmtree(root / d) for d in ("data", "trajectories")],
        "no episodes",
    ),
    (
        lambda root: _footers(root, lambda f: {**f, "object_asset": 5}),
        "data/apple/ep-0.parquet: its object_asset is not a nonempty string",
    ),
    (
        lambda root: _footers(root, lambda f: {**f, "image_size": {"front": {}}}),
        "its metadata does not describe its columns: KeyError",
    ),
    (
        lambda root: rewrite(root / "data/apple/ep-0.parquet", lambda t, f: (t, [f])),
        "data/apple/ep-0.parquet: its physgraspgen metadata is not a mapping",
    ),
    (
        lambda root: rewrite(
            root / "trajectories/apple/ep-0.parquet", lambda t, f: (t, [])
        ),
        "trajectories/apple/ep-0.parquet: its physgraspgen metadata is not a mapping",
    ),
    (
        lambda root: rewrite(
            root / "data/apple/ep-0.parquet",
            _field_metadata("camera.front.intrinsics", {"shape": "[9]"}),
        ),
        r"the field metadata of \['camera.front.intrinsics'\] is not the format's",
    ),
    (
        lambda root: rewrite(
            root / "data/apple/ep-0.parquet",
            _field_metadata("depth.wrist.depth_m", None),
        ),
        r"the field metadata of \['depth.wrist.depth_m'\] is not the format's",
    ),
    (_damage_a_page, "data/apple/ep-0.parquet: cannot be read"),
    (
        lambda root: rewrite(
            root / "data/apple/ep-0.parquet",
            lambda t, f: (t.append_column("extra", pa.array([0] * t.num_rows)), f),
        ),
        "data/apple/ep-0.parquet: its columns are not the format's",
    ),
    (
        lambda root: rewrite(
            root / "trajectories/apple/ep-0.parquet",
            lambda t, f: (
                t.set_column(
                    t.schema.get_field_index("robot.waypoint"),
                    "robot.waypoint",
                    t.column("robot.waypoint").cast(pa.int64()),
                ),
                f,
            ),
        ),
        "trajectories/apple/ep-0.parquet: its columns are not the format's",
    ),
]


@pytest.mark.parametrize(("damage", "message"), FINALIZE_REFUSALS)
def test_finalize_refuses_what_the_format_does_not_describe(tmp_path, damage, message):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    write_episode(recorder, uid="ep-0", object_type="apple")
    write_episode(recorder, uid="ep-1", object_type="banana", n_frames=3)
    damage(tmp_path)
    with pytest.raises(DatasetError, match=message):
        recorder.finalize(workers=2)
    assert not (tmp_path / "metadata.yaml").exists()
    assert not (tmp_path / "SHA256SUMS").exists()


def test_finalize_replaces_the_part_files_of_a_finalize_that_was_stopped(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    write_episode(recorder)
    for name in ("metadata.yaml.part", "SHA256SUMS.part", ".gitattributes.part"):
        (tmp_path / name).write_text("half written\n")
    recorder.finalize()
    assert files_under(tmp_path) == [
        ".gitattributes",
        "SHA256SUMS",
        "data/apple/ep-0.parquet",
        "metadata.yaml",
        "trajectories/apple/ep-0.parquet",
    ]


def test_finalize_refuses_object_types_that_differ_in_case(tmp_path):
    sensitive = _case_sensitive(tmp_path)
    recorder = GraspDataRecorder(tmp_path, make_spec())
    write_episode(recorder, uid="ep-0", object_type="Apple")
    write_episode(recorder, uid="ep-1", object_type="apple")
    # A case-insensitive file system puts both episodes in one folder.
    message = (
        r"object types \['Apple', 'apple'\] are one type in lower case"
        if sensitive
        else "data/Apple/ep-1.parquet holds episode ep-1 of apple"
    )
    with pytest.raises(DatasetError, match=message):
        recorder.finalize()


@pytest.mark.parametrize(
    ("doi", "url", "message"),
    [
        ("https://doi.org/10.5281/zenodo.1234567", None, "doi must be a bare DOI"),
        ("zenodo.1234567", None, "doi must be a bare DOI"),
        (None, "http://example.org/dataset", "download_url must be an https:// URL"),
        (None, "https://", "download_url must be an https:// URL"),
    ],
)
def test_finalize_refuses_a_doi_or_url_of_another_form(tmp_path, doi, url, message):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    write_episode(recorder)
    with pytest.raises(DatasetError, match=message):
        recorder.finalize(doi=doi, download_url=url)


# --- video ----------------------------------------------------------------------


def _box(kind, payload):
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def sample_entry(codec=b"av01", width=W, height=H, flags=0x0C):
    """A visual sample entry; flags is the third byte of av1C (0x0C: 8-bit 4:2:0)."""
    body = (
        bytes(6)
        + struct.pack(">H", 1)
        + bytes(16)
        + struct.pack(">HHII", width, height, 0x00480000, 0x00480000)
        + bytes(4)
        + struct.pack(">H", 1)
        + bytes(32)
        + struct.pack(">Hh", 0x18, -1)
    )
    child = _box(b"av1C", bytes([0x81, 0x00, flags, 0x00])) if codec == b"av01" else b""
    return _box(codec, body + child)


def minimal_mp4(
    n_frames,
    timescale=15360,
    delta=512,
    fragmented=False,
    entry=None,
    handler=b"vide",
    mdhd=None,
):
    if mdhd is None:
        mdhd = _box(b"mdhd", bytes(4) + struct.pack(">III", 0, 0, timescale) + bytes(8))
    hdlr = _box(b"hdlr", bytes(8) + handler + bytes(12) + b"\0")
    stsd = _box(b"stsd", bytes(4) + struct.pack(">I", 1) + (entry or sample_entry()))
    stsz = _box(
        b"stsz", bytes(4) + struct.pack(">II", 0, n_frames) + bytes(4 * n_frames)
    )
    stts = _box(b"stts", bytes(4) + struct.pack(">III", 1, n_frames, delta))
    stbl = _box(b"stbl", stsd + stts + stsz)
    trak = _box(b"trak", _box(b"mdia", hdlr + _box(b"minf", stbl) + mdhd))
    body = _box(b"ftyp", b"isom" + bytes(4)) + _box(b"moov", trak)
    if fragmented:
        body += _box(b"moof", b"")
    return body + _box(b"mdat", bytes(16))


def test_video_frames_are_counted_from_the_index(tmp_path):
    path = tmp_path / "video.mp4"
    path.write_bytes(minimal_mp4(450))
    (track,) = video_tracks(path)
    assert (track.n_frames, track.fps) == (450, 30)
    assert (track.codec, track.width, track.height) == ("av01", W, H)
    assert (track.bit_depth, track.chroma) == (8, "4:2:0")
    assert fmt.video_error(track, (H, W), 30, 450) is None

    path.write_bytes(minimal_mp4(450, fragmented=True))
    with pytest.raises(ValueError, match="fragmented"):
        video_tracks(path)

    path.write_bytes(minimal_mp4(450)[:-8])
    with pytest.raises(ValueError, match="past the end"):
        video_tracks(path)

    # An empty mdhd box, the last of the moov box.
    path.write_bytes(minimal_mp4(450, mdhd=_box(b"mdhd", b"")))
    with pytest.raises(ValueError, match="a truncated mdhd box"):
        video_tracks(path)

    path.write_bytes(minimal_mp4(450, handler=b"soun"))
    assert video_tracks(path) == []


@pytest.mark.parametrize(
    ("mp4", "message"),
    [
        (
            minimal_mp4(4, entry=sample_entry(b"avc1")),
            r"is coded as avc1, not AV1 \(av01\)",
        ),
        (minimal_mp4(4, entry=sample_entry(flags=0x4C)), "is 10-bit 4:2:0, not 8-bit"),
        (minimal_mp4(4, entry=sample_entry(flags=0x00)), "is 8-bit 4:4:4, not 8-bit"),
        (minimal_mp4(4, entry=sample_entry(flags=0x1C)), "is 8-bit 4:0:0, not 8-bit"),
        (
            minimal_mp4(4, entry=sample_entry(width=H, height=W)),
            "is 96 x 64 pixels, not the camera's 64 x 96",
        ),
        (minimal_mp4(4, delta=1024), "does not run at a constant 30 fps"),
        (minimal_mp4(5), "holds 5 frames, not 4"),
    ],
)
def test_a_video_that_is_not_the_cameras_is_found_from_its_index(
    tmp_path, mp4, message
):
    path = tmp_path / "video.mp4"
    path.write_bytes(mp4)
    (track,) = video_tracks(path)
    assert re.match(message, fmt.video_error(track, (H, W), 30, 4))


def _fake_ffmpeg(directory, script):
    path = directory / "ffmpeg"
    path.write_text(f"#!/bin/sh\n{script}\n")
    path.chmod(0o755)
    return str(path)


@POSIX
@pytest.mark.parametrize(
    ("script", "message"),
    [
        ("head -c 100 /dev/zero", "decoded to 0 frames, its index has 4"),
        ("head -c 80000 /dev/zero", "decoded to more frames than the 4 of its index"),
        ("echo 'no AV1 decoder' >&2; exit 3", r"\(exit 3\): no AV1 decoder"),
    ],
)
def test_read_video_checks_what_ffmpeg_decodes(tmp_path, script, message):
    path = tmp_path / "video.mp4"
    path.write_bytes(minimal_mp4(4))
    ffmpeg = _fake_ffmpeg(tmp_path, script)
    with pytest.raises(RuntimeError, match=message):
        read_video(path, (H, W), ffmpeg)


def test_read_video_checks_the_size_and_count_and_names_the_decoder(tmp_path):
    path = tmp_path / "video.mp4"
    path.write_bytes(minimal_mp4(4))
    with pytest.raises(ValueError, match="is 64 x 96 pixels, not 96 x 64"):
        read_video(path, (W, H))
    with pytest.raises(ValueError, match="holds 4 frames, not 5"):
        read_video(path, n_frames=5)
    with pytest.raises(
        RuntimeError, match="reading RGB needs an ffmpeg .* AV1 decoder"
    ):
        read_video(path, ffmpeg=str(tmp_path / "no-ffmpeg"))


@POSIX
@pytest.mark.parametrize(
    ("listing", "message"),
    [
        (" V....D libx264  libx264 H.264 (codec h264)", "has no libsvtav1 encoder"),
        (" V..... libsvtav1  something else (codec h264)", "not an AV1 encoder"),
    ],
)
def test_the_recorder_checks_that_ffmpeg_has_the_encoder(tmp_path, listing, message):
    ffmpeg = _fake_ffmpeg(
        tmp_path, f"printf 'Encoders:\\n V..... = Video\\n ------\\n{listing}\\n'"
    )
    with pytest.raises(RuntimeError, match=message):
        GraspDataRecorder(tmp_path / "root", make_spec(fmt.CHANNELS), ffmpeg=ffmpeg)


@needs_av1
def test_rgb_is_written_as_av1_video(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec(channels=fmt.CHANNELS))
    frames = write_episode(recorder, n_frames=4)
    files = recorder.files("apple", "ep-0")
    sources = np.array(
        [[colour(f["sim_step"], c) for c in fmt.CAMERAS] for f in frames], dtype=float
    )
    for c, camera in enumerate(fmt.CAMERAS):
        (track,) = video_tracks(files.videos[camera])
        assert (track.codec, track.width, track.height) == ("av01", W, H)
        assert (track.bit_depth, track.chroma) == (8, "4:2:0")
        assert (track.n_frames, track.fps) == (4, 30)
        decoded = read_video(files.videos[camera], (H, W), n_frames=4)
        assert decoded.shape == (4, H, W, 3)
        assert decoded.flags.writeable
        # Each frame is the solid colour of its own frame and camera, to AV1's loss.
        means = decoded.reshape(4, -1, 3).mean(axis=1)
        for i, mean in enumerate(means):
            distance = np.abs(sources - mean).max(axis=-1)
            assert np.unravel_index(distance.argmin(), distance.shape) == (i, c)
            assert distance[i, c] < 8
    data = read_frames(
        files.frames, columns=["rgb.front.video_frame", "rgb.front.sim_step"]
    )
    assert data["rgb.front.video_frame"].tolist() == [0, 1, 2, 3]
    assert data["rgb.front.sim_step"].tolist() == [2, 4, 6, 8]
    meta = episode_metadata(files.frames)
    assert meta["video"] == VideoSpec().record()
    assert meta["ffmpeg_version"] == video_module.ffmpeg_version(shutil.which("ffmpeg"))

    metadata = recorder.finalize()
    assert metadata["video"] == {
        **VideoSpec().record(),
        "ffmpeg_versions": [meta["ffmpeg_version"]],
    }
    assert sum(f["kind"] == "video" for f in metadata["files"]) == 2
    assert metadata["counts"]["n_files"] == 4


@needs_av1
def test_an_rgb_image_of_another_size_is_refused(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec(channels=fmt.CHANNELS))
    with open_episode(recorder) as episode:
        frame = make_frame(2, recorder.spec)
        frame["rgb.front.image"] = np.zeros((W, H, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match=r"rgb.front.image must be \(64, 96, 3\)"):
            episode.add_frame(frame)
        episode.abort()
    assert files_under(tmp_path) == []


@needs_av1
def test_read_video_refuses_a_damaged_video(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec(channels=fmt.CHANNELS))
    write_episode(recorder, n_frames=4)
    path = recorder.files("apple", "ep-0").videos["front"]
    data = bytearray(path.read_bytes())
    # Inverting the first OBU header sets its forbidden bit.
    start = data.find(b"mdat") + 4
    data[start : start + 16] = bytes(b ^ 0xFF for b in data[start : start + 16])
    path.write_bytes(bytes(data))
    with pytest.raises(RuntimeError, match="could not decode"):
        read_video(path, (H, W))


@needs_av1
@pytest.mark.parametrize(
    ("mp4", "message"),
    [
        (minimal_mp4(4, entry=sample_entry(b"avc1")), "is coded as avc1"),
        (
            minimal_mp4(4, entry=sample_entry(width=64, height=64)),
            "is 64 x 64 pixels, not the camera's 64 x 96",
        ),
        (minimal_mp4(3), "holds 3 frames, not 4"),
        (minimal_mp4(4, handler=b"soun"), "holds 0 video tracks, not one"),
        (
            minimal_mp4(4, mdhd=_box(b"mdhd", b"")),
            "cannot be read: a truncated mdhd box",
        ),
    ],
)
def test_finalize_refuses_a_video_that_is_not_the_cameras(tmp_path, mp4, message):
    recorder = GraspDataRecorder(tmp_path, make_spec(channels=fmt.CHANNELS))
    write_episode(recorder, n_frames=4)
    recorder.files("apple", "ep-0").videos["wrist"].write_bytes(mp4)
    with pytest.raises(DatasetError, match=f"videos/wrist/apple/ep-0.mp4: {message}"):
        recorder.finalize()


@needs_av1
def test_finalize_reports_metadata_that_does_not_describe_a_video(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec(channels=fmt.CHANNELS))
    write_episode(recorder, n_frames=2)
    _footers(tmp_path, lambda f: {**f, "fps": "30"})
    with pytest.raises(
        DatasetError,
        match="videos/front/apple/ep-0.mp4: its metadata does not describe it: TypeError",
    ):
        recorder.finalize()


@needs_av1
@POSIX
def test_episodes_from_two_ffmpeg_builds_share_one_video_record(tmp_path):
    spec = make_spec(channels=fmt.CHANNELS)
    first = GraspDataRecorder(tmp_path / "root", spec)
    write_episode(first, uid="ep-0", n_frames=2)
    real = shutil.which("ffmpeg")
    other = _fake_ffmpeg(
        tmp_path,
        'if [ "$2" = "-version" ]; then echo "ffmpeg version 0.0-test"; exit 0; fi\n'
        f'exec "{real}" "$@"',
    )
    second = GraspDataRecorder(tmp_path / "root", spec, ffmpeg=other)
    write_episode(second, uid="ep-1", object_type="banana", n_frames=2)
    metadata = first.finalize()
    assert metadata["video"]["ffmpeg_versions"] == sorted(
        ["ffmpeg version 0.0-test", video_module.ffmpeg_version(real)]
    )

    third = GraspDataRecorder(
        tmp_path / "root", make_spec(fmt.CHANNELS, video=VideoSpec(crf=40))
    )
    write_episode(third, uid="ep-2", object_type="cherry", n_frames=2)
    with pytest.raises(DatasetError, match="episodes disagree on video"):
        first.finalize()


@needs_av1
def test_an_encoder_that_fails_to_start_stops_those_started(tmp_path, monkeypatch):
    recorder = GraspDataRecorder(tmp_path, make_spec(channels=fmt.CHANNELS))
    started = []
    popen = subprocess.Popen

    def fail_the_second(*args, **kwargs):
        if started:
            raise OSError(24, "Too many open files")
        started.append(popen(*args, **kwargs))
        return started[-1]

    monkeypatch.setattr(video_module.subprocess, "Popen", fail_the_second)
    with (
        pytest.raises(OSError, match="Too many open files"),
        open_episode(recorder) as episode,
    ):
        episode.add_frame(make_frame(2, recorder.spec))
    assert len(started) == 1
    assert started[0].poll() is not None
    assert files_under(tmp_path) == []
