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
READ_ERRORS = (OSError, ValueError, IndexError, struct.error, pa.ArrowException)

# What every episode of a dataset must state alike. video is the encoder settings; the
# ffmpeg build that ran them may differ between episodes.
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
GITATTRIBUTES_NAME = ".gitattributes"
# The files finalize writes, each through a .part file of its own.
WRITTEN = (fmt.METADATA_NAME, fmt.CHECKSUMS_NAME, GITATTRIBUTES_NAME)
# Files at the root that are not data and are not hashed.
ROOT_FILES = (*WRITTEN, "README.md")
SKIPPED_DIRS = (".git", ".cache")
# A git checkout must keep the hashed text files byte for byte (core.autocrlf), and the
# episode files go through Git LFS, as the Hugging Face Hub stores them.
GITATTRIBUTES = """\
/metadata.yaml -text
/SHA256SUMS -text
/tools/** -text
*.parquet filter=lfs diff=lfs merge=lfs -text
*.mp4 filter=lfs diff=lfs merge=lfs -text
"""


def card_front_matter(license: str = "CC-BY-4.0") -> str:
    """The YAML front matter that the dataset card, README.md, starts with.

    It declares the frames and the trajectories as two configs, because the Hugging
    Face Hub reads one table per config and the two have different columns. finalize
    does not write the card; an exporter puts this text at its top.
    """
    header = {
        "license": license.lower(),
        "configs": [
            {
                "config_name": "frames",
                "data_files": f"{fmt.DATA_DIR}/*/*{fmt.PARQUET_SUFFIX}",
                "default": True,
            },
            {
                "config_name": "trajectories",
                "data_files": f"{fmt.TRAJECTORY_DIR}/*/*{fmt.PARQUET_SUFFIX}",
            },
        ],
    }
    return f"---\n{yaml.safe_dump(header, sort_keys=False)}---\n"


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as stream:
        while chunk := stream.read(1 << 22):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _footer(metadata: pq.FileMetaData) -> dict[str, Any]:
    raw = (metadata.metadata or {}).get(fmt.METADATA_KEY.encode())
    if raw is None:
        raise DatasetError(f"carries no {fmt.METADATA_KEY} metadata")
    footer = json.loads(raw)
    if not isinstance(footer, dict):
        raise DatasetError(f"its {fmt.METADATA_KEY} metadata is not a mapping")
    return footer


def _refuse(errors: list[str]) -> None:
    if errors:
        raise DatasetError(
            "cannot finalize the dataset:\n" + "\n".join(f"  {e}" for e in errors)
        )


def _write_atomic(path: Path, text: str) -> None:
    part = path.with_name(path.name + ".part")
    part.write_text(text, encoding="utf-8")
    os.replace(part, path)


def _frame_schema(meta: dict[str, Any]) -> pa.Schema:
    sizes = {c: (s["height"], s["width"]) for c, s in meta["image_size"].items()}
    return fmt.frame_schema(tuple(meta["channels"]), tuple(meta["cameras"]), sizes)


def _read_parquet(path: Path, schema: pa.Schema) -> tuple[pq.FileMetaData, int]:
    """Read every column of every row group, which decodes every page and checks its
    checksum; returns the footer and the row count."""
    with pq.ParquetFile(path, page_checksum_verification=True) as parquet:
        found = parquet.schema_arrow
        if not found.equals(schema, check_metadata=False):
            raise DatasetError("its columns are not the format's")
        # A column's shape is in its field metadata, which equals() leaves out.
        differ = [f.name for f, g in zip(found, schema) if f.metadata != g.metadata]
        if differ:
            raise DatasetError(f"the field metadata of {differ} is not the format's")
        for group in range(parquet.num_row_groups):
            parquet.read_row_group(group)
        return parquet.metadata, parquet.metadata.num_rows


def _check_file(path: Path, kind: str, meta: dict[str, Any], camera: str | None) -> int:
    """Read an episode's file in full; returns its row or video frame count."""
    if kind == "frames":
        _, rows = _read_parquet(path, _frame_schema(meta))
        if rows != meta["n_frames"]:
            raise DatasetError(
                f"holds {rows} frames, its metadata says {meta['n_frames']}"
            )
        return rows
    if kind == "trajectory":
        metadata, rows = _read_parquet(path, fmt.trajectory_schema())
        own = _footer(metadata)
        if own != meta:
            keys = sorted(k for k in {*own, *meta} if own.get(k) != meta.get(k))
            raise DatasetError(f"its metadata differs from the frames file's in {keys}")
        if rows != meta["n_steps"]:
            raise DatasetError(
                f"holds {rows} steps, its metadata says {meta['n_steps']}"
            )
        return rows
    tracks = fmt.video_tracks(path)
    if len(tracks) != 1:
        raise DatasetError(f"holds {len(tracks)} video tracks, not one")
    size = meta["image_size"][camera]
    problem = fmt.video_error(
        tracks[0], (size["height"], size["width"]), meta["fps"], meta["n_frames"]
    )
    if problem:
        raise DatasetError(problem)
    return tracks[0].n_frames


def _verify(
    path: Path, kind: str, meta: dict[str, Any], camera: str | None
) -> tuple[str | None, int, str, int]:
    """Read a file in full, then hash it: (what is wrong, count, sha256, bytes)."""
    try:
        count = _check_file(path, kind, meta, camera)
        sha, size = _sha256(path)
    except DatasetError as exc:
        return str(exc), 0, "", 0
    except READ_ERRORS as exc:
        return f"cannot be read: {exc}", 0, "", 0
    except (KeyError, TypeError) as exc:
        return f"its metadata does not describe it: {exc!r}", 0, "", 0
    return None, count, sha, size


def _episode_error(rel: str, meta: dict[str, Any]) -> str | None:
    _, object_type, name = rel.split("/")
    uid = name[: -len(fmt.PARQUET_SUFFIX)]
    if (
        meta.get("format") != fmt.FORMAT
        or meta.get("format_version") != fmt.FORMAT_VERSION
    ):
        return f"{rel} is not {fmt.FORMAT} {fmt.FORMAT_VERSION}"
    lacking = [key for key in fmt.EPISODE_KEYS if key not in meta]
    if lacking:
        return f"{rel}: its metadata lacks {lacking}"
    if meta["episode_uid"] != uid or meta["object_type"] != object_type:
        return f"{rel} holds episode {meta['episode_uid']} of {meta['object_type']}"
    if not isinstance(meta["object_asset"], str) or not meta["object_asset"]:
        return f"{rel}: its object_asset is not a nonempty string"
    try:
        _frame_schema(meta)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        return f"{rel}: its metadata does not describe its columns: {exc!r}"
    problem = fmt.facts_error(meta["facts"])
    if problem:
        return f"{rel}: {problem}"
    return None


def finalize_dataset(
    root: str | Path,
    *,
    doi: str | None = None,
    download_url: str | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Check the episodes under root, then write metadata.yaml, SHA256SUMS and
    .gitattributes.

    Each Parquet file is read in full and each video's index parsed, in `workers`
    threads, before the file is hashed. doi is a bare DOI (10.<registrant>/<suffix>)
    and download_url an https URL.
    """
    root = Path(root)
    frames: dict[str, Path] = {}
    trajectories: dict[str, Path] = {}
    videos: dict[str, Path] = {}
    tools: list[Path] = []
    errors: list[str] = []
    if doi is not None and not (
        isinstance(doi, str) and fmt.DOI_PATTERN.fullmatch(doi)
    ):
        errors.append(
            f"doi must be a bare DOI, 10.<registrant>/<suffix> without a URL: {doi!r}"
        )
    if download_url is not None and not (
        isinstance(download_url, str) and fmt.URL_PATTERN.fullmatch(download_url)
    ):
        errors.append(f"download_url must be an https:// URL: {download_url!r}")
    for directory, subdirs, names in os.walk(root):
        here = Path(directory)
        if here == root:
            subdirs[:] = [d for d in subdirs if d not in SKIPPED_DIRS]
        for name in names:
            path = here / name
            rel = path.relative_to(root).as_posix()
            parts = rel.split("/")
            # A finalize that was stopped may leave its .part files, which it replaces.
            if len(parts) == 1 and (
                name in ROOT_FILES or name.removesuffix(".part") in WRITTEN
            ):
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
    expected_files: dict[str, tuple[str, dict[str, Any], str | None]] = {}
    generation: dict[str, Any] = {}
    export: dict[str, Any] = {}
    for rel, path in sorted(frames.items()):
        try:
            meta = _footer(pq.read_metadata(path))
        except READ_ERRORS as exc:
            errors.append(f"{rel}: {exc}")
            continue
        problem = _episode_error(rel, meta)
        if problem:
            errors.append(problem)
            continue
        uid, object_type = meta["episode_uid"], meta["object_type"]
        if uid in episodes:
            errors.append(f"episode {uid} is under two object types")
            continue
        episodes[uid] = meta
        for key in SHARED_KEYS:
            shared[key].add(json.dumps(meta[key], sort_keys=True))
        for record, store in (
            (meta["generation_provenance"], generation),
            (meta["export_provenance"], export),
        ):
            if record is not None:
                store.setdefault(json.dumps(record, sort_keys=True), record)
        name = f"{uid}{fmt.PARQUET_SUFFIX}"
        expected_files[rel] = ("frames", meta, None)
        expected_files[f"{fmt.TRAJECTORY_DIR}/{object_type}/{name}"] = (
            "trajectory",
            meta,
            None,
        )
        if "rgb" in meta["channels"]:
            for camera in meta["cameras"]:
                expected_files[
                    f"{fmt.VIDEO_DIR}/{camera}/{object_type}/{uid}{fmt.VIDEO_SUFFIX}"
                ] = ("video", meta, camera)

    for key, values in shared.items():
        if len(values) > 1:
            errors.append(f"episodes disagree on {key}: {sorted(values)}")
    spellings: dict[str, set[str]] = {}
    filed: dict[str, set[str]] = {}
    for meta in episodes.values():
        key = meta["object_type"].strip().lower()
        spellings.setdefault(key, set()).add(meta["object_type"])
        filed.setdefault(meta["object_asset"], set()).add(meta["object_type"])
    for names in spellings.values():
        if len(names) > 1:
            errors.append(f"object types {sorted(names)} are one type in lower case")
    for asset, types in sorted(filed.items()):
        if len(types) > 1:
            errors.append(f"object asset {asset} is under object types {sorted(types)}")
    present = {**frames, **trajectories, **videos}
    for rel in sorted(set(present) - set(expected_files)):
        errors.append(f"{rel} belongs to no episode")
    for rel in sorted(set(expected_files) - set(present)):
        errors.append(f"{rel} is missing")
    _refuse(errors)

    checked = sorted(expected_files)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(
            pool.map(
                lambda rel: _verify(present[rel], *expected_files[rel]),
                checked,
            )
        )
        tool_digests = list(pool.map(_sha256, sorted(tools)))
    files = []
    for rel, (problem, count, sha, size) in zip(checked, results):
        if problem:
            errors.append(f"{rel}: {problem}")
            continue
        kind, meta, _ = expected_files[rel]
        entry = {
            "path": rel,
            "kind": kind,
            "episode_uid": meta["episode_uid"],
            "object_type": meta["object_type"],
        }
        entry["n_steps" if kind == "trajectory" else "n_frames"] = count
        files.append({**entry, "sha256": sha, "bytes": size})
    _refuse(errors)
    tool_entries = [
        {"path": path.relative_to(root).as_posix(), "sha256": sha, "bytes": size}
        for path, (sha, size) in zip(sorted(tools), tool_digests)
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
    video = first["video"]
    if video is not None:
        builds = {meta["ffmpeg_version"] for meta in episodes.values()}
        video = {**video, "ffmpeg_versions": sorted(builds)}
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
        "video": video,
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
        "conventions": {
            **fmt.CONVENTIONS,
            "segmentation_classes": first["segmentation_classes"],
        },
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
    _write_atomic(root / GITATTRIBUTES_NAME, GITATTRIBUTES)
    return metadata
