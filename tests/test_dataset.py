# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
import hashlib
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

H, W = 64, 64
T = 5
NO_RGB = tuple(c for c in fmt.CHANNELS if c != "rgb")


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


def make_spec(channels=NO_RGB):
    return DatasetSpec(
        name="unit-test",
        version="0.0.1",
        fps=30,
        recorded_fps=60,
        image_size={"front": (H, W), "wrist": (H, W)},
        channels=channels,
        frames_per_row_group=2,
    )


def make_frame(step, rng, spec, n_contacts=2):
    frame = {"sim_step": step}
    for camera in spec.cameras:
        if "rgb" in spec.channels:
            frame[f"rgb.{camera}.image"] = rng.integers(
                0, 256, (H, W, 3), dtype=np.uint8
            )
        if "depth" in spec.channels:
            frame[f"depth.{camera}.depth_m"] = rng.random((H, W), dtype=np.float32)
        if "segmentation" in spec.channels:
            frame[f"segmentation.{camera}.mask"] = rng.integers(
                0, 6, (H, W), dtype=np.uint8
            )
        frame[f"camera.{camera}.intrinsics"] = np.array(
            [[50.0, 0.0, 31.5], [0.0, 50.0, 31.5], [0.0, 0.0, 1.0]]
        )
        frame[f"camera.{camera}.world_to_camera"] = np.eye(4)
    if "force" in spec.channels:
        frame["force.contact_force"] = rng.random((n_contacts, 3))
        frame["force.contact_point_world"] = rng.random((n_contacts, 3))
        frame["force.contact_particle"] = np.arange(n_contacts)
        frame["force.contact_is_ground"] = np.zeros(n_contacts, dtype=bool)
        frame["force.contact_is_robot"] = np.ones(n_contacts, dtype=bool)
        frame["force.gripper_net_force"] = rng.random(3)
        frame["force.ground_net_force"] = rng.random(3)
        frame["force.gripper_force_mag"] = 1.5
        frame["force.ground_force_mag"] = 0.5
    if "stress" in spec.channels:
        von_mises = rng.random(T).astype(np.float32)
        von_mises[0] = np.nan
        frame["stress.von_mises"] = von_mises
        frame["stress.valid"] = np.arange(T) > 0
        frame["stress.volume_ratio"] = np.ones(T)
    if "deformation_gradient" in spec.channels:
        frame["deformation_gradient.F"] = np.broadcast_to(np.eye(3), (T, 3, 3))
    frame.update(state(step, rng))
    frame.update(
        {
            "robot.grip_contact_width_m": np.nan,
            "object.n_soft_contacts": n_contacts,
            "object.n_rigid_contacts": 0,
        }
    )
    return frame


def state(step, rng):
    return {
        "robot.ee_pose_cmd": np.array([0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]),
        "robot.ee_pose": np.array([0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]),
        "robot.arm_q_setpoint": rng.random(6),
        "robot.arm_q": rng.random(6),
        "robot.gripper_width_cmd_m": 0.05,
        "robot.gripper_width_setpoint_m": 0.05,
        "robot.gripper_width_m": 0.05,
        "robot.waypoint": step // 4,
        "robot.phase": "lift",
        "object.centroid": np.array([0.0, 0.0, 0.01 * step]),
        "object.bbox_min": np.array([-0.02, -0.02, 0.01 * step - 0.02]),
        "object.bbox_max": np.array([0.02, 0.02, 0.01 * step + 0.02]),
        "object.n_robot_contacts": 2,
    }


def make_step(step, rng, finite=True):
    row = state(step, rng)
    row.update(
        {
            "sim_step": step,
            "finite": finite,
            "robot.grip_contact_width_m": 0.04,
            "force.gripper_force_mag": 1.5,
            "force.ground_force_mag": 0.5,
        }
    )
    if not finite:
        row.update(
            {"object.centroid": None, "object.bbox_min": None, "object.bbox_max": None}
        )
    return row


def write_episode(recorder, uid="ep-0", object_type="apple", n_frames=5, **kwargs):
    rng = np.random.default_rng(0)
    frames = [
        make_frame(2 * (i + 1), rng, recorder.spec, n_contacts=i % 3)
        for i in range(n_frames)
    ]
    with recorder.episode(
        episode_uid=uid,
        object_type=object_type,
        object_asset=f"{object_type}_01",
        n_tets=T,
        label="success",
        facts={"material": {"E_pa": 1.0e5}},
        task={"name": "lift_hold_place"},
        **kwargs,
    ) as episode:
        for frame in frames:
            episode.add_frame(frame)
        for step in range(1, 2 * n_frames + 1):
            episode.add_step(make_step(step, rng, finite=step != 3))
    return frames


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
    assert data["depth.front.depth_m"].shape == (5, H, W)
    np.testing.assert_array_equal(
        data["depth.wrist.depth_m"][3], frames[3]["depth.wrist.depth_m"]
    )
    np.testing.assert_array_equal(
        data["segmentation.front.mask"][1], frames[1]["segmentation.front.mask"]
    )
    assert data["camera.front.intrinsics"].shape == (5, 3, 3)
    assert data["deformation_gradient.F"].shape == (5, T, 3, 3)
    assert np.isnan(data["stress.von_mises"][:, 0]).all()
    assert data["force.n_contacts"].tolist() == [0, 1, 2, 0, 1]
    assert data["force.contact_offset"].tolist() == [0, 0, 1, 3, 3, 4]
    np.testing.assert_allclose(
        data["force.contact_force"][1:3], frames[2]["force.contact_force"], rtol=1e-6
    )
    assert data["robot.phase"].tolist() == ["lift"] * 5
    for layer in ("depth.front", "force", "stress", "deformation_gradient"):
        assert data[f"{layer}.sim_step"].tolist() == data["sim_step"].tolist()

    trajectory = read_trajectory(files.trajectory)
    assert trajectory["sim_step"].tolist() == list(range(1, 11))
    assert np.isnan(trajectory["object.centroid"][2]).all()
    assert trajectory["object.centroid"][3, 2] == pytest.approx(0.04)

    meta = episode_metadata(files.frames)
    assert meta["format_version"] == fmt.FORMAT_VERSION
    assert (meta["n_frames"], meta["n_steps"], meta["n_tets"]) == (5, 10, T)
    assert meta["task"] == {"name": "lift_hold_place"}
    assert episode_metadata(files.trajectory) == meta


def test_columns_can_be_read_alone(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    write_episode(recorder)
    data = read_frames(
        recorder.files("apple", "ep-0").frames, columns=["object.centroid"]
    )
    assert list(data) == ["object.centroid"]
    assert data["object.centroid"].shape == (5, 3)


def test_float_columns_are_byte_stream_split(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    write_episode(recorder)
    metadata = pq.read_metadata(recorder.files("apple", "ep-0").frames)
    columns = metadata.row_group(0)
    encodings = {
        columns.column(i).path_in_schema: columns.column(i).encodings
        for i in range(columns.num_columns)
    }
    assert "BYTE_STREAM_SPLIT" in encodings["depth.front.depth_m.list.element"]
    assert metadata.num_row_groups == 3


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"unknown": 1}, "unknown"),
        ({"sim_step": 4}, "does not increase"),
        ({"sim_step": 7}, "fps grid"),
        ({"force.sim_step": 4}, "another step"),
        ({"robot.arm_q": np.zeros(5)}, "shape"),
        ({"robot.waypoint": 1.5}, "integers"),
        (
            {"segmentation.front.mask": np.full((H, W), 6, dtype=np.uint8)},
            "segmentation",
        ),
    ],
)
def test_a_bad_frame_is_refused_and_the_episode_goes_on(tmp_path, change, message):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    rng = np.random.default_rng(1)
    with recorder.episode(
        episode_uid="ep-0",
        object_type="apple",
        object_asset="apple_01",
        n_tets=T,
        label="fail",
        facts={},
    ) as episode:
        episode.add_frame(make_frame(4, rng, recorder.spec))
        with pytest.raises(ValueError, match=message):
            episode.add_frame({**make_frame(6, rng, recorder.spec), **change})
        episode.add_frame(make_frame(6, rng, recorder.spec))
        for step in range(1, 7):
            episode.add_step(make_step(step, rng))
    assert read_frames(recorder.files("apple", "ep-0").frames)["sim_step"].tolist() == [
        4,
        6,
    ]


def test_a_failed_episode_leaves_nothing(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    rng = np.random.default_rng(2)
    with (
        pytest.raises(RuntimeError, match="interrupted"),
        recorder.episode(
            episode_uid="ep-0",
            object_type="apple",
            object_asset="apple_01",
            n_tets=T,
            label="fail",
            facts={},
        ) as episode,
    ):
        for i in range(3):
            episode.add_frame(make_frame(2 * (i + 1), rng, recorder.spec))
        raise RuntimeError("interrupted")
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


def test_the_trajectory_must_cover_every_frame(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    rng = np.random.default_rng(3)
    episode = recorder.episode(
        episode_uid="ep-0",
        object_type="apple",
        object_asset="apple_01",
        n_tets=T,
        label="fail",
        facts={},
    )
    episode.add_frame(make_frame(2, rng, recorder.spec))
    episode.add_frame(make_frame(4, rng, recorder.spec))
    episode.add_step(make_step(1, rng))
    episode.add_step(make_step(2, rng))
    with pytest.raises(ValueError, match="lacks frame steps"):
        episode.close()
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


def test_an_episode_is_not_written_twice(tmp_path):
    recorder = GraspDataRecorder(tmp_path, make_spec())
    write_episode(recorder)
    with pytest.raises(FileExistsError):
        write_episode(recorder)
    write_episode(recorder, overwrite=True)


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
    other = DatasetSpec(
        name="unit-test",
        version="0.0.2",
        fps=30,
        recorded_fps=60,
        image_size={"front": (H, W), "wrist": (H, W)},
        channels=NO_RGB,
    )
    write_episode(GraspDataRecorder(tmp_path, other), uid="ep-1")
    with pytest.raises(DatasetError, match="disagree on dataset_version"):
        finalize = GraspDataRecorder(tmp_path, other).finalize
        finalize()


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
