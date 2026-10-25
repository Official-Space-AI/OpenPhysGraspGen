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
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("yaml")

import mp4_index
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from dataset_helpers import (
    DROP,
    FACTS,
    FORCE_MAGNITUDES,
    NO_RGB,
    NON_FINITE,
    POSIX,
    TASK,
    H,
    T,
    W,
    _box,
    _fake_ffmpeg,
    colour,
    edit,
    fake_av1,
    fake_encoder,
    files_under,
    fill,
    intrinsics,
    make_frame,
    make_spec,
    make_step,
    minimal_mp4,
    needs_av1,
    nested,
    open_episode,
    rewrite,
    sample_entry,
    world_to_camera,
    write_episode,
)

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
from openphysgraspgen.dataset import finalize as finalize_module
from openphysgraspgen.dataset import video as video_module
from openphysgraspgen.dataset import writer as writer_module

# --- writing and reading -----------------------------------------------------------


def test_an_episode_reads_back_as_written(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    frames = write_episode(recorder, source_n_frames=10)
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
    assert trajectory["sim_time_s"].tolist() == pytest.approx(
        [s / 60 for s in range(1, 11)]
    )
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
    assert (meta["source_n_frames"], meta["source_cut_at_row"]) == (10, -1)
    assert (meta["facts"], meta["task"]) == (FACTS, TASK)
    assert (meta["video"], meta["ffmpeg_version"], meta["video_sha256"]) == (
        None,
        None,
        None,
    )
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
        assert fmt.text_error(data["robot.phase"][i]) is None
    trajectory = read_trajectory(files.trajectory)
    previous = None
    for i in range(len(trajectory["sim_step"])):
        row = {name: values[i] for name, values in trajectory.items()}
        assert fmt.trajectory_row_error(row, previous) is None
        assert fmt.text_error(row["robot.phase"]) is None
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
    (lambda f: {"robot.phase": ""}, "robot.phase must be a nonempty string"),
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
    # Where finite is false, the object's state is NaN and the reached state and force
    # may be; commands and setpoints stay finite, and infinity is never allowed.
    (
        {**NON_FINITE, "force.gripper_force_mag": np.inf},
        "force.gripper_force_mag must be finite or NaN",
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
    ({"robot.phase": None}, "robot.phase must be a nonempty string"),
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


def test_a_blown_up_step_may_lack_the_reached_state_and_force(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    unstable = {name: None for name in fmt.UNSTABLE_MAY_BE_NAN}
    with open_episode(recorder, source_n_frames=3) as episode:
        episode.add_frame(make_frame(2, recorder.spec))
        episode.add_step(make_step(1))
        episode.add_step(make_step(2))
        with pytest.raises(ValueError, match="robot.ee_pose_cmd must be finite"):
            episode.add_step(
                {**make_step(3, finite=False), **unstable, "robot.ee_pose_cmd": None}
            )
        with pytest.raises(ValueError, match="robot.arm_q must be finite or NaN"):
            episode.add_step(
                {**make_step(3, finite=False), "robot.arm_q": np.full(6, np.inf)}
            )
        episode.add_step({**make_step(3, finite=False), **unstable})
    trajectory = read_trajectory(recorder.files("apple", "ep-0").trajectory)
    for name in fmt.UNSTABLE_MAY_BE_NAN:
        assert np.isnan(trajectory[name][2]).all(), name
        assert np.isfinite(trajectory[name][:2]).all(), name


def test_the_reached_state_must_be_finite_on_a_finite_step(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    with open_episode(recorder) as episode:
        episode.add_frame(make_frame(2, recorder.spec))
        episode.add_step(make_step(1))
        with pytest.raises(ValueError, match="robot.ee_pose must be finite"):
            episode.add_step({**make_step(2), "robot.ee_pose": None})
        episode.abort()


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        ({"uid": "../x"}, ValueError, "episode_uid must match"),
        ({"object_type": 5}, TypeError, "object_type must be a string"),
        ({"object_asset": 5}, TypeError, "object_asset must be a string"),
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
        # A name every file system a copy lands on can hold.
        ({"uid": "CON"}, ValueError, "episode_uid must not be a device name"),
        ({"uid": "nul.ep"}, ValueError, "episode_uid must not be a device name"),
        ({"object_type": "LPT9"}, ValueError, "object_type must not be a device name"),
        ({"object_type": "pear."}, ValueError, "object_type must not end in a dot"),
        ({"uid": "e" * 101}, ValueError, "episode_uid must be at most 100 characters"),
        ({"object_asset": " "}, ValueError, "object_asset must not be empty"),
        ({"n_tets": 2.0}, TypeError, "cannot be interpreted as an integer"),
        (
            {"facts": {**FACTS, "material": {"E_pa": 10**400}}},
            ValueError,
            r"levels of nesting: is not JSON: 1000000000.* is out of the int64",
        ),
        # The footer is level 1 and facts level 2: 31 more levels make 33. On some
        # Pythons json.dumps itself gives up on the deeper one.
        *(
            (
                {"facts": {**FACTS, "history": nested(depth)}},
                ValueError,
                "levels of nesting: is not JSON: it nests deeper than 32 levels$",
            )
            for depth in (31, 1200)
        ),
        ({"header": {"seed": -(2**63) - 1}}, ValueError, "out of the int64 range"),
        ({"task": ["lift"]}, TypeError, "task must be a mapping or None"),
        ({"header": "v1"}, TypeError, "header must be a mapping or None"),
        (
            {"generation_provenance": "abc123"},
            TypeError,
            "generation_provenance must be a mapping or None",
        ),
        ({"export_provenance": [1]}, TypeError, "export_provenance must be a mapping"),
        ({"source_n_frames": -1}, ValueError, "source_n_frames must not be negative"),
        (
            {"source_cut_at_row": -2},
            ValueError,
            "source_cut_at_row must be a row or -1",
        ),
    ],
)
def test_episode_arguments_are_checked(tmp_path, change, error, message):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    with pytest.raises(error, match=message):
        open_episode(recorder, **change)
    assert files_under(tmp_path) == []


def test_an_episodes_metadata_may_nest_32_levels(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    facts = {**FACTS, "history": nested(30)}
    write_episode(recorder, facts=facts)
    files = recorder.files("apple", "ep-0")
    for path in (files.frames, files.trajectory):
        assert episode_metadata(path)["facts"] == facts


def test_an_episode_states_a_source_count_of_at_least_its_steps(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    with pytest.raises(
        ValueError,
        match="metadata source_n_frames is 9, not null or an integer of at least "
        "n_steps, 10",
    ):
        write_episode(recorder, source_n_frames=9)
    assert files_under(tmp_path) == []
    write_episode(recorder, source_n_frames=10)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("robot.arm_q", np.linspace(0.0, 1.0, 6) + 2.5),
        ("robot.phase", "hold"),
        ("robot.waypoint", 7),
        ("object.centroid", np.array([0.0, 0.0, 0.03])),
        ("robot.grip_contact_width_m", 0.04),
        ("force.gripper_force_mag", 2.0),
    ],
)
def test_a_frame_holds_the_trajectorys_values_at_its_step(tmp_path, column, value):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    message = f"{column} differs between the frame and the trajectory at this step"
    with open_episode(recorder) as episode:
        # The step first: the frame is refused.
        episode.add_step(make_step(1))
        episode.add_step(make_step(2))
        with pytest.raises(ValueError, match=f"frame 0: {message}"):
            episode.add_frame({**make_frame(2, recorder.spec), column: value})
        episode.add_frame(make_frame(2, recorder.spec))
        # The frame first: the step is refused.
        episode.add_frame({**make_frame(4, recorder.spec), column: value})
        episode.add_step(make_step(3))
        with pytest.raises(ValueError, match=f"step 3: {message}"):
            episode.add_step(make_step(4))
        episode.add_step({**make_step(4), column: value})
    data = read_frames(recorder.files("apple", "ep-0").frames)
    assert data["sim_step"].tolist() == [2, 4]


def test_an_episode_states_the_generators_frame_count(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    with pytest.raises(TypeError, match="source_n_frames"):
        recorder.episode(
            episode_uid="ep-0",
            object_type="apple",
            object_asset="apple_01",
            n_tets=T,
            label="success",
            facts=FACTS,
        )
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
        (
            {"image_size": {"front": (2**16, 2**15)}},
            "front image size must hold at most 2,147,483,647 pixels",
        ),
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


@pytest.mark.parametrize(
    "change",
    [
        {"fps": 30.0},
        {"recorded_fps": True},
        {"image_size": {"front": (64.0, W)}},
        {"image_size": {"front": (H, "96")}},
    ],
)
def test_rates_and_image_sizes_are_integers(change):
    kwargs = {"fps": 30, "recorded_fps": 60, "image_size": {"front": (H, W)}, **change}
    with pytest.raises(TypeError, match="must be an integer"):
        DatasetSpec(name="unit-test", version="0.0.1", channels=NO_RGB, **kwargs)
    spec = DatasetSpec(
        name="unit-test",
        version="0.0.1",
        fps=np.int64(30),
        recorded_fps=60,
        image_size={"front": (np.int32(H), W)},
        channels=NO_RGB,
    )
    assert (type(spec.fps), type(spec.image_size["front"][0])) == (int, int)


@pytest.mark.parametrize("change", [{"crf": 30.5}, {"gop": True}, {"preset": "8"}])
def test_video_settings_are_integers(change):
    with pytest.raises(TypeError, match="must be an integer"):
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


@pytest.mark.parametrize("published", [False, True])
def test_a_writer_whose_episode_was_taken_over_leaves_it_alone(tmp_path, published):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    old = open_episode(recorder, label="fail")
    new = open_episode(recorder, overwrite=True)
    if published:
        # The claim's path is then gone, renamed by the writer that took it.
        fill(new)
        new.close()
    fill(old)
    with pytest.raises(RuntimeError, match="another writer took episode ep-0 over"):
        old.close()
    if not published:
        fill(new)
        new.close()
    files = recorder.files("apple", "ep-0")
    assert episode_metadata(files.frames)["label"] == "success"
    assert files_under(tmp_path) == [
        "data/apple/ep-0.parquet",
        "trajectories/apple/ep-0.parquet",
    ]


class PlaceholderIds:
    """The os module, but an empty file has one placeholder id, as FAT32 and exFAT
    report it on macOS until the file holds data."""

    def __getattr__(self, name):
        return getattr(os, name)

    @staticmethod
    def _id(result):
        if result.st_size:
            return result
        fields = list(result)
        fields[1] = 2**64 - 8
        return os.stat_result(fields)

    def stat(self, path):
        return self._id(os.stat(path))

    def fstat(self, fd):
        return self._id(os.fstat(fd))


def test_a_claim_is_its_open_file_not_the_id_it_was_created_with(tmp_path, monkeypatch):
    monkeypatch.setattr(writer_module, "os", PlaceholderIds())
    recorder = GraspDataRecorder(tmp_path, make_spec())
    # One frame: the claim stays empty until close() writes it.
    write_episode(recorder, uid="ep-0", n_frames=1)
    old = open_episode(recorder, "ep-1", label="fail")
    new = open_episode(recorder, "ep-1", overwrite=True)
    fill(old, 1)
    with pytest.raises(RuntimeError, match="another writer took episode ep-1 over"):
        old.close()
    fill(new, 1)
    new.close()
    assert files_under(tmp_path) == [
        "data/apple/ep-0.parquet",
        "data/apple/ep-1.parquet",
        "trajectories/apple/ep-0.parquet",
        "trajectories/apple/ep-1.parquet",
    ]
    assert (
        episode_metadata(recorder.files("apple", "ep-1").frames)["label"] == "success"
    )


def test_a_writer_taken_over_while_closing_publishes_nothing(tmp_path, monkeypatch):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    old = open_episode(recorder, label="fail")
    fill(old)
    taker = []
    check = writer_module._check_encodings

    def take_over_after_the_frames(path, options):
        check(path, options)
        taker.append(open_episode(recorder, overwrite=True))

    monkeypatch.setattr(writer_module, "_check_encodings", take_over_after_the_frames)
    with pytest.raises(RuntimeError, match="another writer took episode ep-0 over"):
        old.close()
    monkeypatch.undo()
    files = recorder.files("apple", "ep-0")
    assert not files.frames.exists() and not files.trajectory.exists()
    (new,) = taker
    fill(new)
    new.close()
    assert episode_metadata(files.frames)["label"] == "success"
    assert episode_metadata(files.trajectory)["label"] == "success"


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
    generation = {"generator": "1.0.0", "commit": "abc123"}
    write_episode(
        recorder,
        uid="ep-0",
        object_type="apple",
        generation_provenance=generation,
        export_provenance={"exporter": "0.1.0"},
    )
    write_episode(
        recorder,
        uid="ep-1",
        object_type="banana",
        n_frames=3,
        generation_provenance=generation,
        export_provenance={"exporter": "0.2.0"},
    )
    # The tool files of an earlier run are replaced.
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "check_criteria.yaml").write_text("scale: {min_frames: 1}\n")
    (tmp_path / "tools" / "check_dataset.py.part").write_text("half written\n")
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
    # The dataset ships the check, this package's own module, and its criteria.
    shipped = {
        "tools/check_dataset.py": Path(fmt.__file__).read_bytes(),
        "tools/check_criteria.yaml": Path(fmt.__file__)
        .with_name("check_criteria.yaml")
        .read_bytes(),
    }
    assert metadata["tools"] == [
        {
            "path": rel,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
        for rel, data in shipped.items()
    ]
    for rel, data in shipped.items():
        assert (tmp_path / rel).read_bytes() == data
    assert metadata["provenance"] == {
        "generation": [generation],
        "export": [{"exporter": "0.1.0"}, {"exporter": "0.2.0"}],
    }
    lines = (tmp_path / "SHA256SUMS").read_text().splitlines()
    assert len(lines) == len(metadata["files"]) + 2
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
    again = recorder.finalize()
    assert again["counts"] == metadata["counts"]
    assert again["tools"] == metadata["tools"]
    assert sorted(p.name for p in (tmp_path / "tools").iterdir()) == [
        "check_criteria.yaml",
        "check_dataset.py",
    ]


def test_finalize_ships_the_criteria_it_is_given(tmp_path):
    recorder = GraspDataRecorder(tmp_path / "root", make_spec())
    write_episode(recorder)
    criteria = tmp_path / "criteria.yaml"
    text = (Path(fmt.__file__).with_name("check_criteria.yaml")).read_text()
    criteria.write_text(text.replace("min_frames: 1000000", "min_frames: 5"))
    metadata = recorder.finalize(criteria=criteria)
    shipped = tmp_path / "root/tools/check_criteria.yaml"
    assert shipped.read_bytes() == criteria.read_bytes()
    assert (
        metadata["tools"][1]["sha256"]
        == hashlib.sha256(criteria.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda text: text.replace("min_frames: 1000000", "min_frames: 0"),
            r"the criteria's \[scale\] min_frames is 0, not an integer of at least 1",
        ),
        (
            lambda text: text.replace("bins: 8", "bins: 8.0"),
            r"the criteria's \[diversity\] bins is 8.0, not an integer",
        ),
        (
            lambda text: text.replace("  max_categories: 30\n", ""),
            r"the criteria's \[diversity\] lacks \['max_categories'\]",
        ),
        (
            lambda text: text + "scale:\n  min_frames: 1\n",
            "repeats the key 'scale'",
        ),
        (lambda text: "- a list\n", "is not a YAML mapping"),
        (lambda text: "scale: [\n", "is not valid YAML"),
    ],
)
def test_finalize_refuses_criteria_the_check_cannot_apply(tmp_path, change, message):
    recorder = GraspDataRecorder(tmp_path / "root", make_spec())
    write_episode(recorder)
    criteria = tmp_path / "criteria.yaml"
    criteria.write_text(
        change(Path(fmt.__file__).with_name("check_criteria.yaml").read_text())
    )
    with pytest.raises(DatasetError, match=message):
        recorder.finalize(criteria=criteria)
    assert files_under(tmp_path / "root") == [
        "data/apple/ep-0.parquet",
        "trajectories/apple/ep-0.parquet",
    ]
    with pytest.raises(DatasetError, match="no tool files"):
        recorder.finalize(criteria=tmp_path / "missing.yaml")


def test_finalize_refuses_facts_the_criteria_require(tmp_path):
    recorder = GraspDataRecorder(tmp_path / "root", make_spec())
    write_episode(recorder)
    criteria = tmp_path / "criteria.yaml"
    text = Path(fmt.__file__).with_name("check_criteria.yaml").read_text()
    criteria.write_text(text.replace("lighting, camera]", "lighting, camera, texture]"))
    with pytest.raises(
        DatasetError,
        match=r"data/apple/ep-0.parquet: its facts lack \['texture'\], which the "
        "criteria require",
    ):
        recorder.finalize(criteria=criteria)


def test_finalize_refuses_a_folder_the_hub_cannot_hold(tmp_path, monkeypatch):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    write_episode(recorder, uid="ep-0")
    write_episode(recorder, uid="ep-1")
    monkeypatch.setattr(finalize_module, "MAX_FOLDER_FILES", 1)
    with pytest.raises(
        DatasetError, match="data/apple holds 2 files; the Hub allows 1 in a folder"
    ):
        recorder.finalize()
    # A folder of exactly the limit is held.
    monkeypatch.setattr(finalize_module, "MAX_FOLDER_FILES", 2)
    assert recorder.finalize()["counts"]["n_files"] == 4


def test_finalize_states_no_classes_without_masks(tmp_path):
    channels = ("depth", "force", "stress", "deformation_gradient")
    recorder = GraspDataRecorder(tmp_path, make_spec(channels=channels))
    write_episode(recorder)
    metadata = recorder.finalize()
    assert metadata["conventions"]["segmentation_classes"] is None
    assert metadata["channels"] == list(channels)


def test_finalize_writes_metadata_the_check_reads_from_the_deepest_footer(tmp_path):
    # metadata.yaml holds a provenance record three levels below where a footer holds
    # it: a footer of 32 levels makes a metadata.yaml of 34, within the 64 it may have.
    recorder = GraspDataRecorder(tmp_path, make_spec())
    provenance = nested(31)
    write_episode(recorder, generation_provenance=provenance)
    recorder.finalize()
    metadata, _ = fmt.read_metadata(tmp_path)
    assert metadata["provenance"]["generation"] == [provenance]


def test_finalize_writes_a_value_held_twice_in_full():
    # PyYAML writes a value held twice as an alias, which the check refuses.
    shared = {"a": [1, 2]}
    held_twice = {"x": shared, "y": shared}
    with pytest.raises(fmt.CheckError, match="uses the alias"):
        fmt.load_yaml(yaml.safe_dump(held_twice).encode(), "metadata")
    text = yaml.dump(held_twice, Dumper=finalize_module._Dumper)
    assert fmt.load_yaml(text.encode(), "metadata") == held_twice


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
        lambda root: (
            (root / "tools").mkdir() or (root / "tools/notes.txt").write_text("")
        ),
        "tools/notes.txt is not part of the format",
    ),
    (
        lambda root: (root / "data/apple/.DS_Store").write_bytes(b""),
        "data/apple/.DS_Store is not part of the format",
    ),
    (
        lambda root: shutil.copy(
            root / "data/apple/ep-0.parquet", root / "data/apple/nul.parquet"
        ),
        (
            "data/apple/nul.parquet: is not where the format puts a frames file "
            r"\(data/<object_type>/<episode_uid>.parquet\): its episode id 'nul' "
            "must not be a device name"
        ),
    ),
    (
        lambda root: (root / "trajectories/banana/ep-1.parquet").unlink(),
        "data/banana/ep-1.parquet: its trajectory trajectories/banana/ep-1.parquet is not on disk",
    ),
    (
        lambda root: shutil.copy(
            root / "trajectories/banana/ep-1.parquet",
            root / "trajectories/banana/ep-7.parquet",
        ),
        "trajectories/banana/ep-7.parquet: belongs to no episode: data/banana/ep-7.parquet is not on disk",
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
        "data/apple/ep-0.parquet: format_version is '1.0', not '2.0'",
    ),
    (
        lambda root: rewrite(
            root / "data/apple/ep-0.parquet",
            lambda t, f: (t, {k: v for k, v in f.items() if k != "video"}),
        ),
        r"data/apple/ep-0.parquet: metadata keys \['video'\] are missing",
    ),
    (
        lambda root: shutil.copy(
            root / "data/apple/ep-0.parquet", root / "data/apple/ep-5.parquet"
        ),
        "data/apple/ep-5.parquet: holds episode 'ep-0' of 'apple', not 'ep-5' of 'apple'",
    ),
    (
        lambda root: write_episode(
            GraspDataRecorder(root, make_spec()), uid="ep-0", object_type="cherry"
        ),
        "data/cherry/ep-0.parquet: episode ep-0 is also the episode of data/apple/ep-0.parquet",
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
        "data/apple/ep-0.parquet: metadata facts.lighting holds no value but null",
    ),
    (
        lambda root: shutil.copy(
            root / "trajectories/banana/ep-1.parquet",
            root / "trajectories/apple/ep-0.parquet",
        ),
        (
            "trajectories/apple/ep-0.parquet: its physgraspgen metadata is not that of "
            r"data/apple/ep-0.parquet \(it differs in \['episode_uid', 'n_frames', "
            r"'n_steps', 'object_asset', 'object_type'\]\)"
        ),
    ),
    (
        _swap_trajectory_of_an_earlier_write,
        r"is not that of data/apple/ep-0.parquet \(it differs in \['label'\]\)",
    ),
    (
        lambda root: rewrite(root / "data/apple/ep-0.parquet", _drop_last_frame),
        "data/apple/ep-0.parquet: metadata n_frames is 5 but the file has 4 rows",
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
        "data/apple/ep-0.parquet: metadata object_asset is 5, not a nonempty string",
    ),
    (
        lambda root: _footers(root, lambda f: {**f, "image_size": {"front": {}}}),
        "data/apple/ep-0.parquet: metadata image_size is {'front': {}}, not the height",
    ),
    (
        lambda root: _footers(root, lambda f: {**f, "label": "won"}),
        "data/apple/ep-0.parquet: metadata label is 'won', not one of success, fracture",
    ),
    (
        lambda root: _footers(root, lambda f: {**f, "n_tets": "5"}),
        "data/apple/ep-0.parquet: metadata n_tets is '5', not an integer of at least 1",
    ),
    (
        lambda root: _footers(root, lambda f: {**f, "n_steps": 0}),
        "data/apple/ep-0.parquet: metadata n_steps is 0, not an integer of at least 1",
    ),
    (
        lambda root: _footers(root, lambda f: {**f, "fps": True}),
        "data/apple/ep-0.parquet: metadata fps is True, not an integer of at least 1",
    ),
    (
        lambda root: _footers(root, lambda f: {**f, "fps": 7}),
        "data/apple/ep-0.parquet: metadata fps 7 does not divide recorded_fps 60",
    ),
    (
        lambda root: _footers(root, lambda f: {**f, "ffmpeg_version": "ffmpeg 8"}),
        "metadata ffmpeg_version is 'ffmpeg 8', not null without rgb",
    ),
    (
        lambda root: _footers(
            root,
            lambda f: {
                **f,
                "channels": ["rgb", *f["channels"]],
                "video": VideoSpec().record(),
                "ffmpeg_version": None,
            },
        ),
        "metadata ffmpeg_version is None, not a nonempty string with rgb",
    ),
    (
        lambda root: _footers(
            root,
            lambda f: {
                **f,
                "channels": ["rgb", *f["channels"]],
                "video": "av1",
                "ffmpeg_version": "ffmpeg 8",
            },
        ),
        "data/apple/ep-0.parquet: metadata video is 'av1', not the settings of an encoding",
    ),
    (
        lambda root: rewrite(root / "data/apple/ep-0.parquet", lambda t, f: (t, [f])),
        "data/apple/ep-0.parquet: its physgraspgen metadata is not a JSON object",
    ),
    (
        lambda root: rewrite(
            root / "trajectories/apple/ep-0.parquet", lambda t, f: (t, [])
        ),
        "trajectories/apple/ep-0.parquet: its physgraspgen metadata is not a JSON object",
    ),
    (
        lambda root: rewrite(
            root / "data/apple/ep-0.parquet",
            _field_metadata("camera.front.intrinsics", {"shape": "[9]"}),
        ),
        (
            r"column camera.front.intrinsics has the field metadata "
            r"\{'shape': '\[9\]'\}, not \{'shape': '\[3, 3\]'\}"
        ),
    ),
    (
        lambda root: rewrite(
            root / "data/apple/ep-0.parquet",
            _field_metadata("depth.wrist.depth_m", None),
        ),
        r"column depth.wrist.depth_m has the field metadata \{\}, not \{'shape': '\[64, 96\]'\}",
    ),
    (
        _damage_a_page,
        (
            "data/apple/ep-0.parquet: cannot be read: OSError: could not verify page "
            "integrity, CRC checksum verification failed"
        ),
    ),
    (
        lambda root: rewrite(
            root / "data/apple/ep-0.parquet",
            lambda t, f: (t.append_column("extra", pa.array([0] * t.num_rows)), f),
        ),
        r"data/apple/ep-0.parquet: columns \['extra'\] are not in the format",
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
        "trajectories/apple/ep-0.parquet: column robot.waypoint is int64, not int32",
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
    (tmp_path / "tools").mkdir()
    for name in (
        "metadata.yaml.part",
        "SHA256SUMS.part",
        ".gitattributes.part",
        "tools/check_dataset.py.part",
        "tools/check_criteria.yaml.part",
    ):
        (tmp_path / name).write_text("half written\n")
    recorder.finalize()
    assert files_under(tmp_path) == [
        ".gitattributes",
        "SHA256SUMS",
        "data/apple/ep-0.parquet",
        "metadata.yaml",
        "tools/check_criteria.yaml",
        "tools/check_dataset.py",
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
        else "data/Apple/ep-1.parquet: holds episode 'ep-1' of 'apple', not 'ep-1' of "
        "'Apple'"
    )
    with pytest.raises(DatasetError, match=message):
        recorder.finalize()


@pytest.mark.parametrize(
    ("doi", "url", "message"),
    [
        ("https://doi.org/10.5281/zenodo.1234567", None, "doi must be a bare DOI"),
        ("zenodo.1234567", None, "doi must be a bare DOI"),
        ("10.1000.10/abc", None, "registrant code of 4 to 9 digits"),
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

    # Two runs of one duration are one constant rate.
    path.write_bytes(minimal_mp4(4, runs=[(2, 512), (2, 512)]))
    (track,) = video_tracks(path)
    assert (track.n_frames, track.fps) == (4, 30)


@pytest.mark.parametrize(
    ("mp4", "message"),
    [
        (minimal_mp4(4, descriptions=2), "a video track with 2 sample descriptions"),
        (
            minimal_mp4(4, entry=sample_entry(av1c=False)),
            "an av01 sample entry without an av1C box",
        ),
        (minimal_mp4(4, runs=[(3, 512)]), "the time-to-sample table and the sample"),
    ],
)
def test_an_index_the_format_does_not_describe_cannot_be_read(tmp_path, mp4, message):
    path = tmp_path / "video.mp4"
    path.write_bytes(mp4)
    with pytest.raises(ValueError, match=message):
        video_tracks(path)


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
        # The first duration alone would give 30 fps.
        (
            minimal_mp4(4, runs=[(3, 512), (1, 1024)]),
            "does not run at a constant 30 fps",
        ),
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


@pytest.mark.parametrize(
    ("mp4", "message"),
    [
        # mdhd's duration is not the samples' total: a player cuts or pads the video.
        (
            minimal_mp4(4, duration=3 * 512),
            "its media lasts 1536 units, its frames 2048",
        ),
        # One edit at media time 0, rate 1, that lasts as long as every frame.
        (minimal_mp4(4, edits=[(66, 0, 1, 0)]), "its edit list does not play every"),
        (minimal_mp4(4, edits=[(134, 512, 1, 0)]), "its edit list does not play every"),
        (minimal_mp4(4, edits=[(134, 0, 2, 0)]), "its edit list does not play every"),
        (minimal_mp4(4, edits=[(134, 0, 1, 1)]), "its edit list does not play every"),
        (
            minimal_mp4(4, edits=[(17, -1, 1, 0), (134, 0, 1, 0)]),
            "its edit list does not play every",
        ),
        (minimal_mp4(4, edits=[]), "its edit list does not play every"),
        (minimal_mp4(4, ctts=[0, 512, 0, 0]), r"shows its frames out of their stored"),
    ],
)
def test_a_video_shows_every_frame_in_order(tmp_path, mp4, message):
    path = tmp_path / "video.mp4"
    path.write_bytes(mp4)
    (track,) = video_tracks(path)
    (error,) = fmt.video_errors(track, (H, W), 30, 4)
    assert re.match(message, error)
    assert fmt.video_error(track, (H, W), 30, 4) == error


@pytest.mark.parametrize(
    "mp4",
    [
        minimal_mp4(4),
        minimal_mp4(4, edits=None),
        minimal_mp4(4, edits=[(500, 0, 1, 0)]),
        minimal_mp4(4, ctts=[0, 0, 0, 0]),
        minimal_mp4(4, mdhd=mp4_index.mdhd(15360, 2048, version=1)),
        minimal_mp4(4, movie_timescale=600, edits=[(80, 0, 1, 0)]),
    ],
)
def test_a_video_index_as_muxers_write_it_is_the_cameras(tmp_path, mp4):
    path = tmp_path / "video.mp4"
    path.write_bytes(mp4)
    (track,) = video_tracks(path)
    assert fmt.video_errors(track, (H, W), 30, 4) == []


def _without(data, kind):
    """data with every box of this kind renamed, so that no reader finds it."""
    return data.replace(kind, b"free")


@pytest.mark.parametrize(
    ("mp4", "message"),
    [
        (_without(minimal_mp4(4), b"mvhd"), "a moov box without mvhd"),
        (
            minimal_mp4(4, mvhd_box=mp4_index.full_box(b"mvhd", 0, bytes(8))),
            "a truncated mvhd box",
        ),
        (
            minimal_mp4(
                4,
                edts_box=mp4_index.box(
                    b"edts",
                    mp4_index.full_box(b"elst", 0, struct.pack(">I", 2) + bytes(12)),
                ),
            ),
            "a truncated elst box",
        ),
        (
            minimal_mp4(4, mdhd=mp4_index.full_box(b"mdhd", 1, bytes(24))),
            "a truncated mdhd box",
        ),
        (
            minimal_mp4(
                4,
                ctts_box=mp4_index.box(
                    b"ctts", bytes(4) + struct.pack(">I", 2) + bytes(8)
                ),
            ),
            "a truncated ctts box",
        ),
        (_without(minimal_mp4(4), b"stts"), "without a time-to-sample table"),
        (_without(minimal_mp4(4), b"stsz"), "without a sample size table"),
        (minimal_mp4(4, moov_boxes=2), "two moov boxes"),
        (b"", "no moov box"),
    ],
)
def test_an_index_box_that_is_not_whole_cannot_be_read(tmp_path, mp4, message):
    path = tmp_path / "video.mp4"
    path.write_bytes(mp4)
    with pytest.raises(ValueError, match=message):
        video_tracks(path)


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
    path.write_bytes(minimal_mp4(4, handler=b"soun"))
    with pytest.raises(ValueError, match="has 0 video tracks, not one"):
        read_video(path)
    path.write_bytes(minimal_mp4(4))
    with pytest.raises(
        RuntimeError, match="reading RGB needs an ffmpeg .* AV1 decoder"
    ):
        read_video(path, ffmpeg=str(tmp_path / "no-ffmpeg"))


@POSIX
def test_read_video_leaves_the_callers_input_alone(tmp_path):
    path = tmp_path / "video.mp4"
    path.write_bytes(minimal_mp4(4))
    # An ffmpeg that keeps its arguments and all it reads, then decodes black frames.
    ffmpeg = _fake_ffmpeg(
        tmp_path,
        f'echo "$@" > "$0.args"; cat > "$0.input"; head -c {4 * H * W * 3} /dev/zero',
    )
    # The caller's standard input: a pipe that holds a line.
    read, write = os.pipe()
    os.write(write, b"the caller's input\n")
    os.close(write)
    saved = os.dup(0)
    os.dup2(read, 0)
    try:
        frames = read_video(path, (H, W), ffmpeg)
    finally:
        os.dup2(saved, 0)
        os.close(saved)
    try:
        left = os.read(read, 100)
    finally:
        os.close(read)
    assert frames.shape == (4, H, W, 3) and not frames.any()
    assert left == b"the caller's input\n"
    assert (tmp_path / "ffmpeg.input").read_bytes() == b""
    assert "-nostdin" in (tmp_path / "ffmpeg.args").read_text().split()


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


@POSIX
@pytest.mark.parametrize(
    ("video", "message"),
    [
        (minimal_mp4(3, entry=sample_entry(b"avc1")), "ep-0.mp4.part is coded as avc1"),
        (minimal_mp4(2), "ep-0.mp4.part holds 2 frames, not 3"),
        (None, r"ffmpeg failed on .*ep-0.mp4.part \(exit 1\): cannot encode"),
    ],
)
def test_the_writer_checks_the_video_ffmpeg_wrote(tmp_path, video, message):
    ffmpeg = fake_encoder(tmp_path / "bin", video)
    root = tmp_path / "root"
    recorder = GraspDataRecorder(root, make_spec(fmt.CHANNELS), ffmpeg=ffmpeg)
    with pytest.raises(RuntimeError, match=message):
        write_episode(recorder, n_frames=3)
    assert files_under(root) == []


@POSIX
def test_a_frame_that_one_camera_did_not_take_fails_the_episode(tmp_path, monkeypatch):
    ffmpeg = fake_encoder(tmp_path / "bin", minimal_mp4(2))
    root = tmp_path / "root"
    recorder = GraspDataRecorder(root, make_spec(fmt.CHANNELS), ffmpeg=ffmpeg)
    write = video_module.VideoEncoder.write
    failed = []

    def wrist_fails_once(encoder, image):
        if encoder.path.parts[-3] == "wrist" and not failed:
            failed.append(encoder)
            raise RuntimeError("ffmpeg stopped")
        write(encoder, image)

    monkeypatch.setattr(video_module.VideoEncoder, "write", wrist_fails_once)
    episode = open_episode(recorder)
    with pytest.raises(RuntimeError, match="ffmpeg stopped"):
        episode.add_frame(make_frame(2, recorder.spec))
    # The front video took that frame; the episode did not.
    episode.add_frame(make_frame(2, recorder.spec))
    for step in (1, 2):
        episode.add_step(make_step(step))
    with pytest.raises(
        RuntimeError, match="the front video holds 2 frames, the episode 1"
    ):
        episode.close()
    assert files_under(root) == []


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
        # ffmpeg's edit list plays every frame from the first, in order.
        assert track.duration == track.samples_duration == 4 * 512
        assert track.edits is not None and track.edits[0][1:] == (0, 1, 0)
        assert fmt.video_errors(track, (H, W), 30, 4) == []
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
    assert meta["video_sha256"] == {
        c: hashlib.sha256(files.videos[c].read_bytes()).hexdigest() for c in fmt.CAMERAS
    }

    metadata = recorder.finalize()
    assert metadata["video"] == {
        **VideoSpec().record(),
        "ffmpeg_versions": [meta["ffmpeg_version"]],
    }
    assert sum(f["kind"] == "video" for f in metadata["files"]) == 2
    assert metadata["counts"]["n_files"] == 4


@POSIX
def test_the_footer_holds_each_videos_sha256(tmp_path):
    recorder = GraspDataRecorder(
        tmp_path / "root", make_spec(fmt.CHANNELS), ffmpeg=fake_av1(tmp_path / "bin")
    )
    write_episode(recorder, uid="ep-0", n_frames=3)
    write_episode(recorder, uid="ep-1", n_frames=3, label="fail", frame=_darker)
    digests = []
    for uid in ("ep-0", "ep-1"):
        files = recorder.files("apple", uid)
        meta = episode_metadata(files.frames)
        assert meta["video_sha256"] == {
            c: hashlib.sha256(files.videos[c].read_bytes()).hexdigest()
            for c in fmt.CAMERAS
        }
        assert episode_metadata(files.trajectory) == meta
        digests.append(meta["video_sha256"])
    # Two episodes' videos differ, so the digest ties each video to its episode.
    assert digests[0]["front"] != digests[1]["front"]
    recorder.finalize()


def _darker(step, spec, n_contacts=2):
    frame = make_frame(step, spec, n_contacts)
    for camera in spec.cameras:
        frame[f"rgb.{camera}.image"] //= 2
    return frame


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b'{"a": NaN}', "is not JSON: NaN is not a number"),
        (b'{"a": -Infinity}', "is not JSON: -Infinity is not a number"),
        (b'{"a": 1e400}', "is not JSON: 1e400 is out of the float64 range"),
        (b'{"a": 9223372036854775808}', "is out of the int64 range"),
        (b'{"a": -9223372036854775809}', "is out of the int64 range"),
        (b'{"a": ' + b"9" * 5000 + b"}", r"9999\.\.\. \(5,000 characters\) is out"),
        (b'{"a": 1, "a": 2}', "the key 'a' is repeated in an object"),
        (b'{"a": {"b": 1, "b": 1}}', "the key 'b' is repeated in an object"),
        (b"[1, 2]", "is not a JSON object"),
        (b'{"a": "\xff"}', "is not UTF-8"),
        (b"\xef\xbb\xbf{}", "is not JSON"),
        pytest.param(
            b"[" * 100000,
            "is not JSON: it nests deeper than 32 levels",
            id="100000-levels",
        ),
        pytest.param(
            b'{"a": ' + b"[" * 32 + b"]" * 32 + b"}",
            "it nests deeper than 32 levels",
            id="33-levels",
        ),
        pytest.param(
            b'{"a": ' + b"[" * 2000 + b"}",
            "it nests deeper than 32 levels",
            id="2000-levels-unclosed",
        ),
    ],
)
def test_the_footer_is_strict_json(raw, message):
    with pytest.raises(ValueError, match=message):
        fmt.parse_footer(raw)


def test_the_footer_nests_at_most_32_levels():
    # Brackets and escaped quotes inside a string are no levels.
    value = fmt.parse_footer('{"a": ' + "[" * 31 + '"[\\"{"' + "]" * 31 + "}")["a"]
    for _ in range(31):
        (value,) = value
    assert value == '["{'


def test_the_footer_keeps_every_number_it_can_hold():
    footer = fmt.parse_footer(
        b'{"i": 9223372036854775807, "j": -9223372036854775808, "f": 1.7e308}'
    )
    assert footer == {"i": 2**63 - 1, "j": -(2**63), "f": 1.7e308}


@POSIX
def test_an_rgb_image_of_another_size_is_refused(tmp_path):
    ffmpeg = fake_encoder(tmp_path / "bin", minimal_mp4(1))
    root = tmp_path / "root"
    recorder = GraspDataRecorder(root, make_spec(fmt.CHANNELS), ffmpeg=ffmpeg)
    with open_episode(recorder) as episode:
        frame = make_frame(2, recorder.spec)
        frame["rgb.front.image"] = np.zeros((W, H, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match=r"rgb.front.image must be \(64, 96, 3\)"):
            episode.add_frame(frame)
        episode.abort()
    assert files_under(root) == []


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


@POSIX
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
            "its MP4 index cannot be read: a truncated mdhd box",
        ),
        (
            minimal_mp4(4, descriptions=2),
            "its MP4 index cannot be read: a video track with 2 sample descriptions",
        ),
        (
            minimal_mp4(4, entry=sample_entry(av1c=False)),
            "its MP4 index cannot be read: an av01 sample entry without an av1C box",
        ),
        (
            minimal_mp4(4, runs=[(3, 512), (1, 1024)]),
            "does not run at a constant 30 fps",
        ),
        (minimal_mp4(4, edits=[(66, 0, 1, 0)]), "its edit list does not play every"),
        # The same index around other frames: not the video the episode was written with.
        (minimal_mp4(4, payload=b"other frames"), "is not the video its episode was"),
    ],
)
def test_finalize_refuses_a_video_that_is_not_the_cameras(tmp_path, mp4, message):
    ffmpeg = fake_encoder(tmp_path / "bin", minimal_mp4(4))
    recorder = GraspDataRecorder(
        tmp_path / "root", make_spec(channels=fmt.CHANNELS), ffmpeg=ffmpeg
    )
    write_episode(recorder, n_frames=4)
    recorder.files("apple", "ep-0").videos["wrist"].write_bytes(mp4)
    with pytest.raises(DatasetError, match=f"videos/wrist/apple/ep-0.mp4: {message}"):
        recorder.finalize()


@POSIX
def test_finalize_reports_metadata_that_does_not_describe_a_video(tmp_path):
    # Without depth and masks, only the videos read the image size.
    channels = ("rgb", "force", "stress", "deformation_gradient")
    ffmpeg = fake_encoder(tmp_path / "bin", minimal_mp4(2))
    root = tmp_path / "root"
    recorder = GraspDataRecorder(root, make_spec(channels=channels), ffmpeg=ffmpeg)
    write_episode(recorder, n_frames=2)
    _footers(root, lambda f: {**f, "image_size": {}})
    with pytest.raises(
        DatasetError,
        match=r"data/apple/ep-0.parquet: metadata image_size is \{\}, not the height",
    ):
        recorder.finalize()


@POSIX
def test_episodes_from_two_ffmpeg_builds_share_one_video_record(tmp_path):
    root = tmp_path / "root"
    spec = make_spec(channels=fmt.CHANNELS)
    for uid, object_type, version in (
        ("ep-0", "apple", "ffmpeg version 8.1"),
        ("ep-1", "banana", "ffmpeg version 7.1"),
    ):
        ffmpeg = fake_encoder(tmp_path / uid, minimal_mp4(2), version)
        recorder = GraspDataRecorder(root, spec, ffmpeg=ffmpeg)
        write_episode(recorder, uid=uid, object_type=object_type, n_frames=2)
    metadata = recorder.finalize()
    assert metadata["video"] == {
        **VideoSpec().record(),
        "ffmpeg_versions": ["ffmpeg version 7.1", "ffmpeg version 8.1"],
    }
    assert sum(f["kind"] == "video" for f in metadata["files"]) == 4

    third = GraspDataRecorder(
        root, make_spec(fmt.CHANNELS, video=VideoSpec(crf=40)), ffmpeg=ffmpeg
    )
    write_episode(third, uid="ep-2", object_type="cherry", n_frames=2)
    with pytest.raises(DatasetError, match="episodes disagree on video"):
        recorder.finalize()


@POSIX
def test_an_encoder_that_fails_to_start_stops_those_started(tmp_path, monkeypatch):
    ffmpeg = fake_encoder(tmp_path / "bin", minimal_mp4(1))
    root = tmp_path / "root"
    recorder = GraspDataRecorder(root, make_spec(channels=fmt.CHANNELS), ffmpeg=ffmpeg)
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
    assert files_under(root) == []
