# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import json
import operator
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import check_dataset as fmt
from .finalize import finalize_dataset
from .video import VideoEncoder, VideoSpec, check_encoder, ffmpeg_version, find_ffmpeg

if TYPE_CHECKING:
    from typing_extensions import Self

# The contact arrays a frame is given, and the field of force.contacts each fills.
CONTACT_INPUTS = {
    "force.contact_force": "force",
    "force.contact_point_world": "point_world",
    "force.contact_particle": "particle",
    "force.contact_is_ground": "is_ground",
    "force.contact_is_robot": "is_robot",
}


@dataclass(frozen=True)
class DatasetSpec:
    """What every episode of one dataset shares."""

    name: str
    version: str
    fps: int
    recorded_fps: int
    image_size: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    channels: tuple[str, ...] = fmt.CHANNELS
    license: str = "CC-BY-4.0"
    video: VideoSpec = field(default_factory=VideoSpec)
    frames_per_row_group: int = 32
    zstd_level: int = 3

    def __post_init__(self) -> None:
        for key in ("name", "version", "license"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a nonempty string")
        for key in ("fps", "recorded_fps"):
            object.__setattr__(self, key, _integer(key, getattr(self, key)))
        if self.fps < 1 or self.recorded_fps < 1 or self.recorded_fps % self.fps:
            raise ValueError("fps must be positive and divide recorded_fps")
        unknown = sorted(set(self.channels) - set(fmt.CHANNELS))
        if unknown:
            raise ValueError(f"unknown channels {unknown}")
        unknown = sorted(set(self.image_size) - set(fmt.CAMERAS))
        if unknown:
            raise ValueError(f"unknown cameras {unknown}; the format has {fmt.CAMERAS}")
        channels = tuple(c for c in fmt.CHANNELS if c in self.channels)
        sizes = {}
        for camera in fmt.CAMERAS:
            if camera in self.image_size:
                height, width = (
                    _integer(f"{camera} image size", v) for v in self.image_size[camera]
                )
                if height < 1 or width < 1:
                    raise ValueError(f"{camera} image size must be positive")
                if height * width > fmt.MAX_PIXELS:
                    raise ValueError(
                        f"{camera} image size must hold at most {fmt.MAX_PIXELS:,} pixels"
                    )
                if "rgb" in channels and (height % 2 or width % 2):
                    raise ValueError(
                        f"{camera} video frames need an even width and height"
                    )
                sizes[camera] = (height, width)
        if any(c in channels for c in fmt.CAMERA_CHANNELS) and not sizes:
            raise ValueError(
                "camera channels need the image size of at least one camera"
            )
        if self.frames_per_row_group < 1:
            raise ValueError("frames_per_row_group must be at least 1")
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "image_size", sizes)

    @property
    def cameras(self) -> tuple[str, ...]:
        if not any(c in self.channels for c in fmt.CAMERA_CHANNELS):
            return ()
        return tuple(self.image_size)

    @property
    def video_cameras(self) -> tuple[str, ...]:
        return self.cameras if "rgb" in self.channels else ()

    @property
    def stride(self) -> int:
        return self.recorded_fps // self.fps


@dataclass(frozen=True)
class EpisodeFiles:
    frames: Path
    trajectory: Path
    videos: dict[str, Path]
    n_frames: int = 0
    n_steps: int = 0


def episode_files(
    root: str | Path, object_type: str, episode_uid: str, cameras: tuple[str, ...]
) -> EpisodeFiles:
    root = Path(root)
    name = f"{episode_uid}{fmt.PARQUET_SUFFIX}"
    return EpisodeFiles(
        frames=root / fmt.DATA_DIR / object_type / name,
        trajectory=root / fmt.TRAJECTORY_DIR / object_type / name,
        videos={
            camera: root
            / fmt.VIDEO_DIR
            / camera
            / object_type
            / f"{episode_uid}{fmt.VIDEO_SUFFIX}"
            for camera in cameras
        },
    )


def _part(path: Path) -> Path:
    return path.with_name(path.name + ".part")


def _leaves(schema: pa.Schema) -> list[tuple[str, str]]:
    """The Parquet leaf columns of a schema: (path, physical type)."""
    sink = pa.BufferOutputStream()
    pq.write_table(schema.empty_table(), sink)
    parquet = pq.ParquetFile(pa.BufferReader(sink.getvalue())).schema
    return [
        (parquet.column(i).path, parquet.column(i).physical_type)
        for i in range(len(parquet))
    ]


def _writer_options(schema: pa.Schema, spec: DatasetSpec) -> dict[str, Any]:
    leaves = _leaves(schema)
    scalars = [
        f.name
        for f in schema
        if not pa.types.is_list(f.type) and not pa.types.is_fixed_size_list(f.type)
    ]
    masks = {f.name for f in schema if f.name.endswith(".mask")}
    return {
        "compression": "zstd",
        "compression_level": spec.zstd_level,
        # Dictionary pages inflate wide float columns. Text repeats, and Parquet has no
        # 8-bit type: a mask pixel is a 4-byte value unless its few classes are coded.
        "use_dictionary": [f.name for f in schema if pa.types.is_string(f.type)]
        + [path for path, _ in leaves if path.split(".list.")[0] in masks],
        "use_byte_stream_split": [
            path for path, kind in leaves if kind in ("FLOAT", "DOUBLE")
        ],
        "write_statistics": scalars,
        "write_page_checksum": True,
    }


def _check_encodings(path: Path, options: dict[str, Any]) -> None:
    # A byte-stream-split request that names no leaf is ignored without a warning.
    wanted = set(options["use_byte_stream_split"])
    metadata = pq.read_metadata(path)
    for group in range(metadata.num_row_groups):
        row_group = metadata.row_group(group)
        for i in range(row_group.num_columns):
            column = row_group.column(i)
            if (
                column.path_in_schema in wanted
                and "BYTE_STREAM_SPLIT" not in column.encodings
            ):
                raise RuntimeError(
                    f"{path}: {column.path_in_schema} is not byte-stream split"
                )


def _integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    return int(value)


def _dumps(value: Any, what: str) -> str:
    """value as the compact JSON the check reads back: no NaN or infinity, integers
    that fit int64, at most fmt.MAX_NESTING levels of objects and arrays."""
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        fmt.parse_footer(text)
    except (TypeError, ValueError, RecursionError) as exc:
        # json.dumps raises RecursionError on a value deeper than this Python encodes.
        problem = (
            f"is not JSON: it nests deeper than {fmt.MAX_NESTING} levels"
            if isinstance(exc, RecursionError)
            else exc
        )
        raise ValueError(
            f"{what} must be JSON without NaN, infinity, integers beyond int64 or more "
            f"than {fmt.MAX_NESTING} levels of nesting: {problem}"
        ) from exc
    return text


def _cast(name: str, array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """A copy of array in dtype; refused when a value does not fit."""
    if dtype.kind == "f":
        if array.dtype.kind not in "fiu":
            raise ValueError(f"{name} must be numeric")
        with np.errstate(over="ignore"):
            result = np.array(array, dtype=dtype)
        if (
            array.dtype.kind == "f"
            and array.dtype.itemsize > dtype.itemsize
            and (np.isinf(result) & np.isfinite(array)).any()
        ):
            raise ValueError(f"{name} does not fit {dtype}")
        return result
    if dtype.kind == "b":
        if array.dtype.kind == "b" or (
            array.dtype.kind in "iu" and np.isin(array, (0, 1)).all()
        ):
            return np.array(array, dtype=bool)
        raise ValueError(f"{name} must be boolean")
    if array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be integers")
    info = np.iinfo(dtype)
    if array.size and (array.min() < info.min or array.max() > info.max):
        raise ValueError(f"{name} does not fit {dtype}")
    return np.array(array, dtype=dtype)


def _value(name: str, value: Any, dtype: str, expected: tuple[int, ...] | None) -> Any:
    """A column's value as stored: a Python scalar, a string or a flat array copy.

    None stands for NaN in a float column that a value rule covers, and the rule decides
    whether NaN is allowed there.
    """
    if dtype == "string":
        problem = fmt.text_error(value)
        if problem:
            raise ValueError(f"{name} {problem}")
        return value
    if (
        value is None
        and dtype.startswith("float")
        and name.split(".", 1)[0] in fmt.FINITE_GROUPS
    ):
        value = np.full(expected or (), np.nan)
    array = np.asarray(value)
    if expected is None:
        if array.shape != ():
            raise ValueError(f"{name} must be one value, got shape {array.shape}")
        return _cast(name, array, np.dtype(dtype)).item()
    if array.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {array.shape}")
    return _cast(name, array, np.dtype(dtype)).reshape(-1)


def _arrow(
    values: list[Any], dtype: str, shape: object, type_: pa.DataType
) -> pa.Array:
    if dtype == "string":
        return pa.array(values, type=type_)
    if shape == ():
        return pa.array(np.asarray(values, dtype=np.dtype(dtype)), type=type_)
    flat = pa.array(np.concatenate(values))
    if pa.types.is_fixed_size_list(type_):
        return pa.FixedSizeListArray.from_arrays(flat, type_.list_size)
    offsets = np.zeros(len(values) + 1, dtype=np.int32)
    np.cumsum([len(v) for v in values], out=offsets[1:])
    return pa.ListArray.from_arrays(pa.array(offsets), flat)


def _claim(files: EpisodeFiles, uid: str, overwrite: bool) -> IO[bytes]:
    """Create the episode's frames .part file, which only one writer can create."""
    targets = [files.frames, files.trajectory, *files.videos.values()]
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
    claim = _part(files.frames)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(claim, flags, 0o666)
    except FileExistsError:
        if not overwrite:
            raise FileExistsError(
                f"episode {uid} is being written, or its writer stopped without "
                f"removing {claim}; overwrite=True replaces it"
            ) from None
        claim.unlink(missing_ok=True)
        fd = os.open(claim, flags, 0o666)
    stream = os.fdopen(fd, "wb")
    existing = [p for p in targets if p.exists()]
    if existing and not overwrite:
        stream.close()
        claim.unlink()
        raise FileExistsError(f"episode {uid} is already written: {existing[0]}")
    # Holding the claim, any other .part file of the episode is a leftover.
    for path in targets[1:]:
        _part(path).unlink(missing_ok=True)
    return stream


class EpisodeWriter:
    """Writes one episode: call add_frame and add_step in order, then close.

    The writer claims the episode by creating its frames .part file, so a second writer
    of the same episode is refused until this one closes or aborts.
    """

    def __init__(
        self,
        root: Path,
        spec: DatasetSpec,
        info: dict[str, Any],
        *,
        ffmpeg: str | None,
        ffmpeg_version: str | None,
        overwrite: bool,
    ) -> None:
        self.spec = spec
        self.info = info
        self.n_tets = info["n_tets"]
        self.files = episode_files(
            root, info["object_type"], info["episode_uid"], spec.video_cameras
        )
        self._columns = fmt.frame_columns(spec.channels, spec.cameras)
        self._schema = fmt.frame_schema(spec.channels, spec.cameras, spec.image_size)
        self._options = _writer_options(self._schema, spec)
        self._trajectory_schema = fmt.trajectory_schema()
        self._layers = [
            name.rsplit(".", 1)[0]
            for name in self._columns
            if name.endswith(".sim_step")
        ]
        computed = {
            "episode_uid",
            "frame_index",
            "sim_time_s",
            "force.n_contacts",
            "force.contacts",
        }
        computed |= {f"{layer}.sim_step" for layer in self._layers}
        computed |= {f"rgb.{c}.video_frame" for c in spec.cameras}
        self._required = set(self._columns) - computed
        if "force" in spec.channels:
            self._required |= set(CONTACT_INPUTS)
        if "rgb" in spec.channels:
            self._required |= {f"rgb.{c}.image" for c in spec.cameras}
        self._optional = {f"{layer}.sim_step" for layer in self._layers}

        self._buffer: dict[str, list[Any]] = {name: [] for name in self._columns}
        self._contacts: dict[str, list[np.ndarray]] = {
            key: [] for key in CONTACT_INPUTS
        }
        self._steps: list[dict[str, Any]] = []
        self._frame_steps: list[int] = []
        # The shared values of each frame whose step the trajectory does not hold yet.
        self._pending: dict[int, dict[str, Any]] = {}
        self._shared = [n for n in fmt.SHARED_COLUMNS if n in self._columns]
        self._writer: pq.ParquetWriter | None = None
        self._ffmpeg = ffmpeg
        self._ffmpeg_version = ffmpeg_version
        self._encoders: dict[str, VideoEncoder] = {}
        self._done = False
        self._claim = _claim(self.files, info["episode_uid"], overwrite)
        # The claim's file id once closed, which only Windows does before publishing.
        self._claim_id: tuple[int, int] | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, kind: object, error: object, trace: object) -> None:
        if self._done:
            return
        if kind is None:
            self.close()
        else:
            self.abort()

    @property
    def n_frames(self) -> int:
        return len(self._frame_steps)

    def _owns_claim(self, path: Path | None = None) -> bool:
        """Whether path, the frames .part file by default, is this writer's claim.

        The open claim is compared with the file at path, not with an id taken when it
        was created: FAT32 and exFAT give an empty file a placeholder id that changes
        once the file holds data. On POSIX the claim stays open until this is no longer
        asked, so that its file id cannot be given to another file.
        """
        try:
            found = os.stat(path or _part(self.files.frames))
        except FileNotFoundError:
            return False
        if self._claim.closed:
            return (found.st_dev, found.st_ino) == self._claim_id
        return os.path.samestat(os.fstat(self._claim.fileno()), found)

    def _check_claim(self, path: Path | None = None) -> None:
        if not self._owns_claim(path):
            raise RuntimeError(
                f"another writer took episode {self.info['episode_uid']} over"
            )

    def _expected_shape(self, name: str, shape: object) -> tuple[int, ...] | None:
        if shape == ():
            return None
        if shape == "image":
            return self.spec.image_size[fmt.column_camera(name)]
        if isinstance(shape, tuple) and shape[0] == "tets":
            return (self.n_tets,) if shape[1] == 1 else (self.n_tets, 3, 3)
        return shape

    def add_frame(self, frame: Mapping[str, Any]) -> None:
        """One published frame. Frames are consecutive on the fps grid: each one's
        sim_step is the previous one's plus recorded_fps / fps. Its robot and object
        values and force magnitudes must equal the trajectory's at its step, whichever
        of the two is added first.

        camera.<camera>.world_to_camera must be rigid to within 1e-6, which a rotation
        computed in float32 can miss: compute it in float64.
        """
        if self._done:
            raise RuntimeError("the episode is closed")
        keys = set(frame)
        missing = sorted(self._required - keys)
        unknown = sorted(keys - self._required - self._optional)
        if missing or unknown:
            raise ValueError(
                f"frame {self.n_frames}: missing {missing}, unknown {unknown}"
            )
        step = _value("sim_step", frame["sim_step"], "int64", None)
        steps = {"sim_step": step}
        for layer in self._layers:
            name = f"{layer}.sim_step"
            steps[name] = _value(name, frame.get(name, step), "int64", None)
        problem = fmt.frame_steps_error(
            steps,
            self._frame_steps[-1] if self._frame_steps else None,
            self.spec.fps,
            self.spec.recorded_fps,
        )
        if problem:
            raise ValueError(f"frame {self.n_frames}: {problem}")

        row: dict[str, Any] = {
            "episode_uid": self.info["episode_uid"],
            "frame_index": self.n_frames,
            "sim_step": step,
            "sim_time_s": step / self.spec.recorded_fps,
        }
        for layer in self._layers:
            row[f"{layer}.sim_step"] = step
        for camera in self.spec.cameras:
            if "rgb" in self.spec.channels:
                row[f"rgb.{camera}.video_frame"] = self.n_frames
        contacts = {}
        if "force" in self.spec.channels:
            for key, field_name in CONTACT_INPUTS.items():
                dtype, shape = fmt.CONTACT_FIELDS[field_name]
                array = np.asarray(frame[key])
                n = array.shape[0] if array.ndim else -1
                if array.shape != (n, *shape) or n < 0:
                    raise ValueError(
                        f"{key} must have shape (M, *{shape}), got {array.shape}"
                    )
                contacts[key] = _cast(key, array, np.dtype(dtype))
            counts = {len(a) for a in contacts.values()}
            if len(counts) != 1:
                raise ValueError(
                    f"frame {self.n_frames}: the contact arrays differ in length"
                )
            row["force.n_contacts"] = counts.pop()
        for name, (dtype, shape) in self._columns.items():
            if name in row or shape == "contacts":
                continue
            row[name] = _value(
                name, frame[name], dtype, self._expected_shape(name, shape)
            )
        self._check_values(row, contacts)
        shared = {name: row[name] for name in self._shared}
        if step <= len(self._steps):
            problem = fmt.state_error(shared, self._steps[step - 1])
            if problem:
                raise ValueError(f"frame {self.n_frames}: {problem}")
        images = {}
        for camera in self.files.videos:
            image = np.asarray(frame[f"rgb.{camera}.image"])
            height, width = self.spec.image_size[camera]
            if image.shape != (height, width, 3) or image.dtype != np.uint8:
                raise ValueError(
                    f"rgb.{camera}.image must be ({height}, {width}, 3) uint8"
                )
            images[camera] = image

        if images:
            if self._ffmpeg is None:
                raise RuntimeError("writing RGB needs an ffmpeg executable")
            # One at a time, so that abort stops every encoder that has started.
            for camera, path in self.files.videos.items():
                if camera not in self._encoders:
                    self._encoders[camera] = VideoEncoder(
                        _part(path),
                        self.spec.image_size[camera],
                        self.spec.fps,
                        self.spec.video,
                        self._ffmpeg,
                    )
        for camera, image in images.items():
            self._encoders[camera].write(image)
        for name, value in row.items():
            self._buffer[name].append(value)
        for key, array in contacts.items():
            self._contacts[key].append(array)
        self._frame_steps.append(step)
        if step > len(self._steps):
            self._pending[step] = shared
        if len(self._frame_steps) % self.spec.frames_per_row_group == 0:
            self._flush()

    def _check_values(self, row: dict[str, Any], contacts: dict[str, Any]) -> None:
        problems = [
            (name, fmt.finite_error(name, value))
            for name, value in [*row.items(), *contacts.items()]
        ]
        for camera in self.spec.cameras:
            for name, error in (
                (f"segmentation.{camera}.mask", fmt.mask_error),
                (f"camera.{camera}.intrinsics", fmt.intrinsics_error),
                (f"camera.{camera}.world_to_camera", fmt.world_to_camera_error),
            ):
                if name in row:
                    problems.append((name, error(row[name])))
        if "stress" in self.spec.channels:
            problem = fmt.stress_error(row["stress.von_mises"], row["stress.valid"])
            problems.append(("stress.von_mises", problem))
        for name, problem in problems:
            if problem:
                raise ValueError(f"frame {self.n_frames}: {name} {problem}")

    def add_step(self, step: Mapping[str, Any]) -> None:
        """One row of the trajectory: every recorded step from step 1, in order.

        object.centroid, object.bbox_min and object.bbox_max are NaN, or None, exactly
        where finite is false; there the reached robot state and the force magnitudes may
        be NaN too. robot.grip_contact_width_m may be NaN; every other float value is
        finite.
        """
        if self._done:
            raise RuntimeError("the episode is closed")
        where = f"step {len(self._steps)}"
        names = set(fmt.TRAJECTORY_COLUMNS) - {"episode_uid", "sim_time_s"}
        keys = set(step)
        if keys != names:
            raise ValueError(
                f"{where}: missing {sorted(names - keys)}, "
                f"unknown {sorted(keys - names)}"
            )
        sim_step = _value("sim_step", step["sim_step"], "int64", None)
        row = {
            "episode_uid": self.info["episode_uid"],
            "sim_step": sim_step,
            "sim_time_s": sim_step / self.spec.recorded_fps,
            "finite": bool(_value("finite", step["finite"], "bool", None)),
        }
        for name, (dtype, shape) in fmt.TRAJECTORY_COLUMNS.items():
            if name in row:
                continue
            row[name] = _value(
                name, step[name], dtype, self._expected_shape(name, shape)
            )
        problem = fmt.trajectory_row_error(
            row, self._steps[-1]["sim_step"] if self._steps else None
        )
        if sim_step in self._pending:
            problem = problem or fmt.state_error(self._pending[sim_step], row)
        if problem:
            raise ValueError(f"{where}: {problem}")
        self._steps.append(row)
        self._pending.pop(sim_step, None)

    def _flush(self) -> None:
        n = len(self._buffer["sim_step"])
        if n == 0:
            return
        arrays = []
        for field_ in self._schema:
            dtype, shape = self._columns[field_.name]
            if shape == "contacts":
                arrays.append(self._contact_array(field_.type))
            else:
                arrays.append(
                    _arrow(self._buffer[field_.name], dtype, shape, field_.type)
                )
        table = pa.Table.from_arrays(arrays, schema=self._schema)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self._claim, self._schema, **self._options)
        self._writer.write_table(table, row_group_size=n)
        for values in self._buffer.values():
            values.clear()
        for values in self._contacts.values():
            values.clear()

    def _contact_array(self, type_: pa.DataType) -> pa.Array:
        struct_type = type_.value_type
        children = []
        for key, field_name in CONTACT_INPUTS.items():
            dtype, shape = fmt.CONTACT_FIELDS[field_name]
            flat = np.concatenate([a.reshape(-1) for a in self._contacts[key]])
            values = pa.array(flat.astype(np.dtype(dtype), copy=False))
            if shape:
                values = pa.FixedSizeListArray.from_arrays(values, shape[0])
            children.append(values)
        structs = pa.StructArray.from_arrays(children, fields=list(struct_type))
        offsets = np.zeros(
            len(self._contacts["force.contact_force"]) + 1, dtype=np.int32
        )
        np.cumsum(
            [len(a) for a in self._contacts["force.contact_force"]], out=offsets[1:]
        )
        return pa.ListArray.from_arrays(pa.array(offsets), structs, type=type_)

    def _metadata(self, digests: dict[str, str]) -> dict[str, bytes]:
        info = dict(self.info)
        video = self.spec.video.record() if self.files.videos else None
        info.update(
            {
                "format": fmt.FORMAT,
                "format_version": fmt.FORMAT_VERSION,
                "n_frames": self.n_frames,
                "n_steps": len(self._steps),
                "fps": self.spec.fps,
                "recorded_fps": self.spec.recorded_fps,
                "channels": list(self.spec.channels),
                "cameras": list(self.spec.cameras),
                "image_size": {
                    c: {"height": h, "width": w}
                    for c, (h, w) in self.spec.image_size.items()
                    if c in self.spec.cameras
                },
                "license": self.spec.license,
                "dataset_name": self.spec.name,
                "dataset_version": self.spec.version,
                "segmentation_classes": list(fmt.SEGMENTATION_CLASSES)
                if "segmentation" in self.spec.channels
                else None,
                "video": video,
                "ffmpeg_version": None
                if video is None
                else self._ffmpeg_version or "unknown",
                # Each video's sha256 ties it to this episode by content.
                "video_sha256": None if video is None else digests,
            }
        )
        problems = fmt.footer_errors(
            info, info["object_type"], info["episode_uid"], self.n_frames
        )
        if problems:
            raise ValueError(
                "the episode's footer breaks the format: " + "; ".join(problems)
            )
        return {fmt.METADATA_KEY: _dumps(info, "the episode metadata")}

    def close(self) -> EpisodeFiles:
        if self._done:
            raise RuntimeError("the episode is closed")
        try:
            if not self._frame_steps:
                raise ValueError("an episode needs at least one frame")
            unrecorded = [s for s in self._frame_steps if s > len(self._steps)]
            if unrecorded:
                raise ValueError(
                    f"the trajectory lacks frame steps, first {unrecorded[0]}"
                )
            self._flush()
            for camera, encoder in self._encoders.items():
                encoder.close()
                if encoder.n_frames != self.n_frames:
                    raise RuntimeError(
                        f"the {camera} video holds {encoder.n_frames} frames, "
                        f"the episode {self.n_frames}"
                    )
            digests = {
                camera: fmt.sha256_file(_part(path))[0]
                for camera, path in self.files.videos.items()
            }
            metadata = self._metadata(digests)
            assert self._writer is not None
            self._writer.add_key_value_metadata(metadata)
            self._writer.close()
            self._writer = None
            self._claim.flush()
            # Checked once the claim holds data, which gives it its own id on FAT32.
            self._check_claim()
            if os.name == "nt":
                # Windows renames no open file, and does not soon reuse a file id.
                stat = os.fstat(self._claim.fileno())
                self._claim_id = (stat.st_dev, stat.st_ino)
                self._claim.close()
            _check_encodings(_part(self.files.frames), self._options)
            trajectory = pa.Table.from_pylist(
                self._steps, schema=self._trajectory_schema
            )
            options = _writer_options(self._trajectory_schema, self.spec)
            with pq.ParquetWriter(
                _part(self.files.trajectory), self._trajectory_schema, **options
            ) as writer:
                writer.write_table(trajectory)
                writer.add_key_value_metadata(metadata)
            self._check_claim()
        except BaseException:
            self.abort()
            raise
        # If a rename fails, the episode stays claimed and its .part files tell why.
        self._done = True
        try:
            self._publish()
        finally:
            self._claim.close()
        return EpisodeFiles(
            self.files.frames,
            self.files.trajectory,
            dict(self.files.videos),
            self.n_frames,
            len(self._steps),
        )

    def _publish(self) -> None:
        """Rename the files into place, the frames file last and only while it is this
        writer's claim: until it is renamed, the episode stays claimed."""
        claim = _part(self.files.frames)
        # No other writer uses this name, so the file found there cannot change.
        mine = claim.with_name(f"{claim.name}.{os.urandom(4).hex()}")
        try:
            for path in (self.files.trajectory, *self.files.videos.values()):
                os.replace(_part(path), path)
            os.replace(claim, mine)
        except FileNotFoundError:
            # A writer that takes the episode over removes its other .part files.
            self._check_claim()
            raise
        try:
            self._check_claim(mine)
        except RuntimeError:
            os.replace(mine, claim)
            raise
        os.replace(mine, self.files.frames)

    def abort(self) -> None:
        """Stop without writing; nothing of the episode is left on disk.

        A writer whose episode another writer took over (overwrite=True) leaves the
        files alone.
        """
        self._done = True
        if self._writer is not None:
            with contextlib.suppress(OSError, pa.ArrowException):
                self._writer.close()
            self._writer = None
        for encoder in self._encoders.values():
            encoder.abort()
        owned = self._owns_claim()
        with contextlib.suppress(OSError):
            self._claim.close()
        if not owned:
            return
        for path in (
            self.files.trajectory,
            *self.files.videos.values(),
            self.files.frames,
        ):
            _part(path).unlink(missing_ok=True)


class GraspDataRecorder:
    """Records grasp episodes in the public dataset format.

    Several recorders, in several processes, can write episodes of one dataset into
    one root; finalize writes ``metadata.yaml``, ``SHA256SUMS`` and ``.gitattributes``
    once all are written.
    """

    def __init__(
        self, root: str | Path, spec: DatasetSpec, *, ffmpeg: str = "ffmpeg"
    ) -> None:
        self.root = Path(root)
        self.spec = spec
        self._ffmpeg = None
        self._ffmpeg_version = None
        if "rgb" in spec.channels:
            self._ffmpeg = find_ffmpeg(ffmpeg)
            check_encoder(self._ffmpeg, spec.video.encoder)
            self._ffmpeg_version = ffmpeg_version(self._ffmpeg)

    def files(self, object_type: str, episode_uid: str) -> EpisodeFiles:
        return episode_files(
            self.root, object_type, episode_uid, self.spec.video_cameras
        )

    def episode(
        self,
        *,
        episode_uid: str,
        object_type: str,
        object_asset: str,
        n_tets: int,
        label: str,
        facts: Mapping[str, Any],
        source_n_frames: int | None,
        task: Mapping[str, Any] | None = None,
        header: Mapping[str, Any] | None = None,
        generation_provenance: Mapping[str, Any] | None = None,
        export_provenance: Mapping[str, Any] | None = None,
        source_cut_at_row: int = -1,
        overwrite: bool = False,
    ) -> EpisodeWriter:
        """Start writing an episode.

        episode_uid and object_type become path components: each matches
        [A-Za-z0-9][A-Za-z0-9._-]*, has at most 100 characters, does not end in a dot
        and is no device name Windows reserves (CON, NUL, COM1, ...). facts holds each
        of the kinds material, placement, lighting and camera as a mapping with at least
        one value that is not null, and its integers fit int64. source_n_frames is the
        generator's frame count, which n_steps equals when the trajectory holds every
        step it recorded; it is required, None only when the count is unknown, and at
        least n_steps. task is free-form until the PG-T1 scorer defines its keys; task,
        header and the provenance records are mappings or None.
        """
        for key, value in (("episode_uid", episode_uid), ("object_type", object_type)):
            if not isinstance(value, str):
                raise TypeError(f"{key} must be a string")
            problem = fmt.name_error(value)
            if problem:
                raise ValueError(f"{key} {problem}: {value!r}")
        if not isinstance(object_asset, str):
            raise TypeError("object_asset must be a string")
        if not object_asset.strip():
            raise ValueError("object_asset must not be empty")
        if label not in fmt.LABELS:
            raise ValueError(f"label must be one of {fmt.LABELS}")
        if operator.index(n_tets) < 1:
            raise ValueError("n_tets must be at least 1")
        if not isinstance(facts, Mapping):
            raise TypeError("facts must be a mapping")
        problem = fmt.facts_error(facts)
        if problem:
            raise ValueError(problem)
        for key, value in (
            ("task", task),
            ("header", header),
            ("generation_provenance", generation_provenance),
            ("export_provenance", export_provenance),
        ):
            if value is not None and not isinstance(value, Mapping):
                raise TypeError(f"{key} must be a mapping or None")
        if source_n_frames is not None and operator.index(source_n_frames) < 0:
            raise ValueError("source_n_frames must not be negative")
        if operator.index(source_cut_at_row) < -1:
            raise ValueError("source_cut_at_row must be a row or -1")
        info = {
            "episode_uid": episode_uid,
            "object_type": object_type,
            "object_asset": object_asset,
            "n_tets": operator.index(n_tets),
            "label": label,
            "facts": dict(facts),
            "task": None if task is None else dict(task),
            "header": None if header is None else dict(header),
            "generation_provenance": None
            if generation_provenance is None
            else dict(generation_provenance),
            "export_provenance": None
            if export_provenance is None
            else dict(export_provenance),
            "source_n_frames": None
            if source_n_frames is None
            else operator.index(source_n_frames),
            "source_cut_at_row": operator.index(source_cut_at_row),
        }
        # A copy, so that the caller can reuse or change its mappings.
        info = json.loads(_dumps(info, "the episode metadata"))
        return EpisodeWriter(
            self.root,
            self.spec,
            info,
            ffmpeg=self._ffmpeg,
            ffmpeg_version=self._ffmpeg_version,
            overwrite=overwrite,
        )

    def finalize(
        self,
        *,
        doi: str | None = None,
        download_url: str | None = None,
        workers: int = 1,
        criteria: str | Path | None = None,
        check_seed: int | None = None,
    ) -> dict[str, Any]:
        return finalize_dataset(
            self.root,
            doi=doi,
            download_url=download_url,
            workers=workers,
            criteria=criteria,
            check_seed=check_seed,
        )
