# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import json
import shutil
import struct
import subprocess

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("yaml")

import pyarrow.parquet as pq

from physgraspgen.dataset import (
    DatasetError,
    DatasetSpec,
    GraspDataRecorder,
    episode_metadata,
    read_frames,
    read_metadata,
    read_trajectory,
    read_video,
    video_tracks,
)
from physgraspgen.dataset import check_dataset as fmt

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
    assert meta["format_version"] == fmt.FORMAT_VERSION
    assert (meta["n_frames"], meta["n_steps"], meta["n_tets"]) == (5, 10, T)
    assert (meta["facts"], meta["task"]) == (FACTS, TASK)
    assert meta["video"] is None
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
        r"determinant is not \+1",
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
    ({"finite": False}, "object.centroid must be NaN, or None, where finite is false"),
    (
        {"finite": False, "object.centroid": None, "object.bbox_min": [np.nan] * 3},
        "object.bbox_max must be NaN, or None, where finite is false",
    ),
    ({"robot.arm_q": np.full(6, np.nan)}, "robot.arm_q must be finite"),
    ({"robot.gripper_width_m": np.inf}, "robot.gripper_width_m must be finite"),
    ({"robot.grip_contact_width_m": -np.inf}, "must be finite or NaN"),
    ({"force.gripper_force_mag": np.nan}, "force.gripper_force_mag must be finite"),
    ({"force.gripper_force_mag": None}, "force.gripper_force_mag must be numeric"),
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
    with open_episode(recorder) as episode:
        episode.add_frame(make_frame(2, recorder.spec))
        episode.add_step(make_step(1))
        episode.add_step(make_step(2))
        nan = {name: np.full(3, np.nan) for name in fmt.NOT_FINITE_COLUMNS}
        episode.add_step({**make_step(3, finite=False), **nan})
    trajectory = read_trajectory(recorder.files("apple", "ep-0").trajectory)
    assert trajectory["finite"].tolist() == [True, True, False]
    assert np.isnan(trajectory["object.bbox_min"][2]).all()
    assert np.isnan(trajectory["robot.grip_contact_width_m"]).all()


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


# --- finalize --------------------------------------------------------------------


def test_finalize_lists_and_hashes_every_file(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    write_episode(recorder, uid="ep-0", object_type="apple")
    write_episode(recorder, uid="ep-1", object_type="banana", n_frames=3)
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "check_criteria.yaml").write_text("scale: {min_frames: 1}\n")

    metadata = recorder.finalize(workers=2)
    assert metadata == read_metadata(tmp_path)
    assert metadata["counts"]["n_frames"] == 8
    assert metadata["counts"]["n_episodes"] == 2
    assert metadata["counts"]["n_object_types"] == 2
    assert metadata["object_types"]["banana"] == {
        "n_episodes": 1,
        "n_frames": 3,
        "assets": ["banana_01"],
    }
    assert {f["kind"] for f in metadata["files"]} == {"frames", "trajectory"}
    lines = (tmp_path / "SHA256SUMS").read_text().splitlines()
    assert len(lines) == len(metadata["files"]) + 1
    for line in lines:
        digest, rel = line.split("  ")
        assert hashlib.sha256((tmp_path / rel).read_bytes()).hexdigest() == digest


def test_finalize_refuses_what_the_format_does_not_describe(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    write_episode(recorder)
    (tmp_path / "data" / "apple" / "ep-9.parquet.part").write_bytes(b"")
    with pytest.raises(DatasetError, match="not part of the format"):
        recorder.finalize()
    assert not (tmp_path / "metadata.yaml").exists()


def test_finalize_refuses_episodes_of_another_dataset(tmp_path):
    write_episode(GraspDataRecorder(tmp_path, make_spec()), uid="ep-0")
    other = make_spec(version="0.0.2")
    write_episode(GraspDataRecorder(tmp_path, other), uid="ep-1")
    with pytest.raises(DatasetError, match="disagree on dataset_version"):
        GraspDataRecorder(tmp_path, other).finalize()


# --- video ----------------------------------------------------------------------


@needs_av1
def test_rgb_is_written_as_av1_video(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec(channels=fmt.CHANNELS))
    frames = write_episode(recorder, n_frames=4)
    files = recorder.files("apple", "ep-0")
    for camera, path in files.videos.items():
        (track,) = video_tracks(path)
        assert (track.n_frames, track.fps) == (4, 30)
        decoded = read_video(path, (H, W))
        assert decoded.shape == (4, H, W, 3)
        error = np.abs(
            decoded.astype(int) - frames[2][f"rgb.{camera}.image"].astype(int)
        )
        assert error.mean() < 128
    data = read_frames(
        files.frames, columns=["rgb.front.video_frame", "rgb.front.sim_step"]
    )
    assert data["rgb.front.video_frame"].tolist() == [0, 1, 2, 3]
    metadata = recorder.finalize()
    assert metadata["video"]["codec"] == "av1"
    assert sum(f["kind"] == "video" for f in metadata["files"]) == 2


def _box(kind, payload):
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def minimal_mp4(n_frames, timescale=15360, delta=512, fragmented=False):
    mdhd = _box(b"mdhd", bytes(4) + struct.pack(">III", 0, 0, timescale) + bytes(8))
    hdlr = _box(b"hdlr", bytes(8) + b"vide" + bytes(12) + b"\0")
    stsz = _box(
        b"stsz", bytes(4) + struct.pack(">II", 0, n_frames) + bytes(4 * n_frames)
    )
    stts = _box(b"stts", bytes(4) + struct.pack(">III", 1, n_frames, delta))
    stbl = _box(b"stbl", stts + stsz)
    trak = _box(b"trak", _box(b"mdia", mdhd + hdlr + _box(b"minf", stbl)))
    body = _box(b"ftyp", b"isom" + bytes(4)) + _box(b"moov", trak)
    if fragmented:
        body += _box(b"moof", b"")
    return body + _box(b"mdat", bytes(16))


def test_video_frames_are_counted_from_the_index(tmp_path):
    path = tmp_path / "video.mp4"
    path.write_bytes(minimal_mp4(450))
    (track,) = video_tracks(path)
    assert (track.n_frames, track.fps) == (450, 30)

    path.write_bytes(minimal_mp4(450, fragmented=True))
    with pytest.raises(ValueError, match="fragmented"):
        video_tracks(path)

    path.write_bytes(minimal_mp4(450)[:-8])
    with pytest.raises(ValueError, match="past the end"):
        video_tracks(path)
