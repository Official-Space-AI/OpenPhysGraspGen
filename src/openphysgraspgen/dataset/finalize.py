# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from . import check_dataset as fmt


class DatasetError(ValueError):
    pass


# What reading a damaged or foreign file can raise.
READ_ERRORS = (OSError, ValueError, struct.error, pa.ArrowException)

# What every episode of a dataset must state alike.
SHARED_KEYS = (
    "format",
    "format_version",
    "dataset_name",
    "dataset_version",
    "license",
    "fps",
    "recorded_fps",
    "channels",
    "cameras",
    "image_size",
    "segmentation_classes",
    "video",
)
# Files at the root that are not data and are not hashed.
ROOT_FILES = (fmt.METADATA_NAME, fmt.CHECKSUMS_NAME, "README.md", ".gitattributes")
SKIPPED_DIRS = (".git", ".cache")


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as stream:
        while chunk := stream.read(1 << 22):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _footer(path: Path) -> dict[str, Any]:
    metadata = pq.read_metadata(path).metadata or {}
    raw = metadata.get(fmt.METADATA_KEY.encode())
    if raw is None:
        raise DatasetError(f"{path} carries no {fmt.METADATA_KEY} metadata")
    return json.loads(raw)


def _write_atomic(path: Path, text: str) -> None:
    part = path.with_name(path.name + ".part")
    part.write_text(text, encoding="utf-8")
    os.replace(part, path)


def finalize_dataset(
    root: str | Path,
    *,
    doi: str | None = None,
    download_url: str | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Write metadata.yaml and SHA256SUMS for the episodes under root."""
    root = Path(root)
    frames: dict[str, Path] = {}
    trajectories: dict[str, Path] = {}
    videos: dict[str, Path] = {}
    tools: list[Path] = []
    errors: list[str] = []
    for directory, subdirs, names in os.walk(root):
        here = Path(directory)
        if here == root:
            subdirs[:] = [d for d in subdirs if d not in SKIPPED_DIRS]
        for name in names:
            path = here / name
            rel = path.relative_to(root).as_posix()
            parts = rel.split("/")
            if len(parts) == 1 and name in ROOT_FILES:
                continue
            if len(parts) == 2 and parts[0] == fmt.TOOLS_DIR:
                tools.append(path)
            elif (
                len(parts) == 3
                and parts[0] == fmt.DATA_DIR
                and name.endswith(fmt.PARQUET_SUFFIX)
            ):
                frames[rel] = path
            elif (
                len(parts) == 3
                and parts[0] == fmt.TRAJECTORY_DIR
                and name.endswith(fmt.PARQUET_SUFFIX)
            ):
                trajectories[rel] = path
            elif (
                len(parts) == 4
                and parts[0] == fmt.VIDEO_DIR
                and name.endswith(fmt.VIDEO_SUFFIX)
            ):
                videos[rel] = path
            else:
                errors.append(f"{rel} is not part of the format")
    if not frames and not errors:
        errors.append("no episodes")

    shared: dict[str, set[str]] = {key: set() for key in SHARED_KEYS}
    episodes = {}
    expected_files: dict[str, tuple[str, dict[str, Any]]] = {}
    generation: dict[str, Any] = {}
    export: dict[str, Any] = {}
    for rel, path in sorted(frames.items()):
        _, object_type, name = rel.split("/")
        uid = name[: -len(fmt.PARQUET_SUFFIX)]
        try:
            meta = _footer(path)
        except READ_ERRORS as exc:
            errors.append(f"{rel}: {exc}")
            continue
        if (
            meta.get("format") != fmt.FORMAT
            or meta.get("format_version") != fmt.FORMAT_VERSION
        ):
            errors.append(f"{rel} is not {fmt.FORMAT} {fmt.FORMAT_VERSION}")
            continue
        if meta.get("episode_uid") != uid or meta.get("object_type") != object_type:
            errors.append(
                f"{rel} holds episode {meta.get('episode_uid')} of {meta.get('object_type')}"
            )
            continue
        if uid in episodes:
            errors.append(f"episode {uid} is under two object types")
            continue
        episodes[uid] = meta
        for key in SHARED_KEYS:
            shared[key].add(json.dumps(meta.get(key), sort_keys=True))
        for record, store in (
            (meta.get("generation_provenance"), generation),
            (meta.get("export_provenance"), export),
        ):
            if record is not None:
                store.setdefault(json.dumps(record, sort_keys=True), record)
        trajectory = f"{fmt.TRAJECTORY_DIR}/{object_type}/{name}"
        expected_files[rel] = ("frames", meta)
        expected_files[trajectory] = ("trajectory", meta)
        if "rgb" in meta.get("channels", []):
            for camera in meta.get("cameras", []):
                expected_files[
                    f"{fmt.VIDEO_DIR}/{camera}/{object_type}/{uid}{fmt.VIDEO_SUFFIX}"
                ] = ("video", meta)

    for key, values in shared.items():
        if len(values) > 1:
            errors.append(f"episodes disagree on {key}: {sorted(values)}")
    present = {**frames, **trajectories, **videos}
    for rel in sorted(set(present) - set(expected_files)):
        errors.append(f"{rel} belongs to no episode")
    for rel in sorted(set(expected_files) - set(present)):
        errors.append(f"{rel} is missing")

    entries = []
    for rel, (kind, meta) in sorted(expected_files.items()):
        if rel not in present:
            continue
        path = present[rel]
        entry: dict[str, Any] = {
            "path": rel,
            "kind": kind,
            "episode_uid": meta["episode_uid"],
            "object_type": meta["object_type"],
        }
        try:
            if kind == "trajectory":
                own = _footer(path)
                if own.get("episode_uid") != meta["episode_uid"]:
                    raise DatasetError("holds another episode")
                rows = pq.read_metadata(path).num_rows
                if rows != own.get("n_steps"):
                    raise DatasetError(
                        f"holds {rows} steps, its metadata says {own.get('n_steps')}"
                    )
                entry["n_steps"] = rows
            elif kind == "frames":
                rows = pq.read_metadata(path).num_rows
                if rows != meta.get("n_frames"):
                    raise DatasetError(
                        f"holds {rows} frames, its metadata says {meta.get('n_frames')}"
                    )
                entry["n_frames"] = rows
            else:
                tracks = fmt.video_tracks(path)
                if len(tracks) != 1 or tracks[0].n_frames != meta.get("n_frames"):
                    raise DatasetError(
                        "does not hold one video track of the episode's frames"
                    )
                entry["n_frames"] = tracks[0].n_frames
        except READ_ERRORS as exc:
            errors.append(f"{rel}: {exc}")
            continue
        entries.append((entry, path))
    if errors:
        raise DatasetError(
            "cannot finalize the dataset:\n" + "\n".join(f"  {e}" for e in errors)
        )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        digests = list(pool.map(_sha256, [path for _, path in entries] + sorted(tools)))
    files = []
    for (entry, _), (sha, size) in zip(entries, digests):
        files.append({**entry, "sha256": sha, "bytes": size})
    tool_entries = [
        {"path": path.relative_to(root).as_posix(), "sha256": sha, "bytes": size}
        for path, (sha, size) in zip(sorted(tools), digests[len(entries) :])
    ]

    first = next(iter(episodes.values()))
    object_types: dict[str, dict[str, Any]] = {}
    for meta in episodes.values():
        entry = object_types.setdefault(
            meta["object_type"], {"n_episodes": 0, "n_frames": 0, "assets": set()}
        )
        entry["n_episodes"] += 1
        entry["n_frames"] += meta["n_frames"]
        entry["assets"].add(meta["object_asset"])
    for entry in object_types.values():
        entry["assets"] = sorted(entry["assets"])
    metadata = {
        "format": fmt.FORMAT,
        "format_version": fmt.FORMAT_VERSION,
        "name": first["dataset_name"],
        "version": first["dataset_version"],
        "license": first["license"],
        "doi": doi,
        "download_url": download_url,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fps": first["fps"],
        "recorded_fps": first["recorded_fps"],
        "channels": first["channels"],
        "cameras": first["cameras"],
        "image_size": first["image_size"],
        "video": first["video"],
        "counts": {
            "n_frames": sum(m["n_frames"] for m in episodes.values()),
            "n_episodes": len(episodes),
            "n_files": len(files),
            "n_object_types": len(object_types),
            "n_object_assets": len({m["object_asset"] for m in episodes.values()}),
            "n_channels": len(first["channels"]),
        },
        "object_types": dict(sorted(object_types.items())),
        "files": files,
        "tools": tool_entries,
        "conventions": fmt.CONVENTIONS,
        "provenance": {
            "generation": list(generation.values()),
            "export": list(export.values()),
        },
    }
    _write_atomic(
        root / fmt.METADATA_NAME,
        yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True, width=100),
    )
    sums = sorted((f["path"], f["sha256"]) for f in [*files, *tool_entries])
    _write_atomic(
        root / fmt.CHECKSUMS_NAME, "".join(f"{sha}  {rel}\n" for rel, sha in sums)
    )
    return metadata
