# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import check_dataset as fmt


class DatasetError(ValueError):
    pass


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
# The Hugging Face Hub holds at most this many files in a folder.
MAX_FOLDER_FILES = 10_000
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


class _Dumper(yaml.SafeDumper):
    """Writes a value as often as metadata.yaml holds it, never as an alias, which the
    check refuses."""

    def ignore_aliases(self, data: Any) -> bool:
        return True


def _refuse(errors: list[str]) -> None:
    if errors:
        raise DatasetError(
            "cannot finalize the dataset:\n" + "\n".join(f"  {e}" for e in errors)
        )


def _write_atomic(path: Path, data: str | bytes) -> None:
    part = path.with_name(path.name + ".part")
    if isinstance(data, str):
        data = data.encode("utf-8")
    part.write_bytes(data)
    os.replace(part, path)


def _tools(criteria: str | Path | None) -> tuple[dict[str, bytes], dict[str, Any]]:
    """The bytes of the two tool files every dataset ships: this package's check module
    itself, and the criteria file, the packaged one unless criteria names another. Both
    are refused when the check could not run with them."""
    source = (
        Path(fmt.__file__).with_name(fmt.CRITERIA_NAME)
        if criteria is None
        else Path(criteria)
    )
    try:
        script = Path(fmt.__file__).read_bytes()
        data = source.read_bytes()
    except OSError as exc:
        raise DatasetError(
            f"cannot finalize the dataset: no tool files: {exc}"
        ) from exc
    try:
        values = fmt.load_yaml(data, f"criteria file {source}")
        if not isinstance(values, dict):
            raise fmt.CheckError(f"criteria file {source} is not a YAML mapping")
        fmt.validate_criteria(values)
    except fmt.CheckError as exc:
        raise DatasetError(f"cannot finalize the dataset: {exc}") from None
    return dict(zip(fmt.TOOL_FILES, (script, data))), values


def _links(root: Path) -> list[str]:
    """Every symbolic link under root, whether it names a file, a folder or nothing,
    outside the folders finalize skips. os.walk lists a linked folder but does not
    enter it."""
    links = []
    for directory, subdirs, names in os.walk(root):
        here = Path(directory)
        if here == root:
            subdirs[:] = [d for d in subdirs if d not in SKIPPED_DIRS]
        for name in (*subdirs, *names):
            if (here / name).is_symlink():
                links.append((here / name).relative_to(root).as_posix())
    return sorted(links)


def _published_check(root: Path, seed: int, workers: int) -> dict[str, Any]:
    """Run the dataset's own tools/check_dataset.py on it with this interpreter, and
    return its exit status and RESULT payload; a check that could not run is refused."""
    root = root.resolve()
    done = subprocess.run(
        [
            sys.executable,
            str(root / fmt.TOOL_FILES[0]),
            str(root),
            "--seed",
            str(seed),
            "--workers",
            str(max(1, workers)),
        ],
        stdout=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=False,
    )
    lines = done.stdout.strip().splitlines()
    payload = None
    if lines and lines[-1].startswith("RESULT "):
        with contextlib.suppress(ValueError):
            payload = json.loads(lines[-1].removeprefix("RESULT "))
    if done.returncode not in (0, 1) or not isinstance(payload, dict):
        reason = payload.get("reason") if isinstance(payload, dict) else None
        raise DatasetError(
            f"the dataset check could not run (exit status {done.returncode}): "
            f"{reason or 'it printed no RESULT line'}"
        )
    return {"exit_status": done.returncode, **payload}


def finalize_dataset(
    root: str | Path,
    *,
    doi: str | None = None,
    download_url: str | None = None,
    workers: int = 1,
    criteria: str | Path | None = None,
    check_seed: int | None = None,
) -> dict[str, Any]:
    """Check the episodes under root with the dataset check's own rules, then write the
    check and its criteria into tools/, then .gitattributes, metadata.yaml and
    SHA256SUMS.

    Every Parquet file is read in full, every page checksum verified and every value
    rule applied, and every video's index read, in `workers` threads; then each file is
    hashed. A symbolic link under root, to a file or a folder, is refused, not followed.
    tools/check_dataset.py is the package's check_dataset.py byte for byte, and
    tools/check_criteria.yaml the packaged acceptance values, or the file criteria names.
    doi is a bare DOI (10.<registrant>/<suffix>, a registrant code of 4 to 9 digits)
    and download_url an https URL. Nothing is written when the dataset is refused.

    With check_seed, the tools/check_dataset.py written then checks the dataset with
    that seed, run by this interpreter, and the metadata returned holds `check`: its
    exit status and RESULT payload. A failed check is returned as such, and the check
    fails until doi and download_url are given; one that could not run raises
    DatasetError.
    """
    root = Path(root)
    errors: list[str] = []
    if doi is not None and not (
        isinstance(doi, str) and fmt.DOI_PATTERN.fullmatch(doi)
    ):
        errors.append(
            "doi must be a bare DOI, 10.<registrant>/<suffix> with a registrant code "
            f"of 4 to 9 digits and no URL: {doi!r}"
        )
    if download_url is not None and not (
        isinstance(download_url, str) and fmt.URL_PATTERN.fullmatch(download_url)
    ):
        errors.append(f"download_url must be an https:// URL: {download_url!r}")
    if check_seed is not None and (
        not isinstance(check_seed, int) or isinstance(check_seed, bool)
    ):
        errors.append(f"check_seed must be an integer: {check_seed!r}")
    tools, values = _tools(criteria)
    # An upload publishes a link as the link or as what it names, as the tool decides,
    # and what it names may lie outside the root: finalize follows none.
    links = _links(root)
    if links:
        _refuse(errors + [f"{rel} is a symbolic link" for rel in links])
    rels, others = fmt.find_data_files(root)
    errors += [f"{rel} is not part of the format" for rel in others]
    data_dirs = (fmt.DATA_DIR, fmt.TRAJECTORY_DIR, fmt.VIDEO_DIR)
    for directory, subdirs, names in os.walk(root):
        here = Path(directory)
        if here == root:
            subdirs[:] = [d for d in subdirs if d not in (*SKIPPED_DIRS, *data_dirs)]
        for name in names:
            rel = (here / name).relative_to(root).as_posix()
            # A finalize that was stopped may leave its .part files, which it replaces.
            written = name.removesuffix(".part")
            if rel.count("/") == 0 and (name in ROOT_FILES or written in WRITTEN):
                continue
            # The tool files are replaced, and nothing else lies in tools/.
            if f"{fmt.TOOLS_DIR}/{written}" in fmt.TOOL_FILES and rel.count("/") == 1:
                continue
            errors.append(f"{rel} is not part of the format")
    if not any(rel.startswith(f"{fmt.DATA_DIR}/") for rel in rels) and not errors:
        errors.append("no episodes")
    # The Hugging Face Hub refuses a folder of more files.
    for folder, count in Counter(rel.rsplit("/", 1)[0] for rel in rels).items():
        if count > MAX_FOLDER_FILES:
            errors.append(
                f"{folder} holds {count:,} files; the Hub allows {MAX_FOLDER_FILES:,} "
                "in a folder"
            )
    _refuse(errors)

    files = [fmt.data_file(rel) for rel in rels]
    episodes = fmt.read_files(root, files, workers=workers)
    errors += [f"{rec.rel}: {error}" for rec in files for error in rec.errors]
    _refuse(errors)

    metas = {e.frames.id.episode_uid: e.frames.footer for e in episodes.values()}
    shared: dict[str, set[str]] = {key: set() for key in SHARED_KEYS}
    kinds = values["diversity"]["required_fact_kinds"]
    for episode in episodes.values():
        meta = episode.frames.footer
        for key in SHARED_KEYS:
            shared[key].add(json.dumps(meta[key], sort_keys=True))
        lacking = [k for k in kinds if not fmt.holds_a_value(meta["facts"].get(k))]
        if lacking:
            errors.append(
                f"{episode.frames.rel}: its facts lack {lacking}, which the criteria "
                "require"
            )
    for key, found in shared.items():
        if len(found) > 1:
            errors.append(f"episodes disagree on {key}: {sorted(found)}")
    spellings: dict[str, set[str]] = {}
    filed: dict[str, set[str]] = {}
    for meta in metas.values():
        key = meta["object_type"].strip().lower()
        spellings.setdefault(key, set()).add(meta["object_type"])
        filed.setdefault(meta["object_asset"], set()).add(meta["object_type"])
    for names in spellings.values():
        if len(names) > 1:
            errors.append(f"object types {sorted(names)} are one type in lower case")
    for asset, types in sorted(filed.items()):
        if len(types) > 1:
            errors.append(f"object asset {asset} is under object types {sorted(types)}")
    _refuse(errors)

    hashes, stats = fmt.hash_files(root, rels, workers)
    errors += [f"{rel} cannot be read" for rel in rels if hashes.get(rel) is None]
    _refuse(errors)
    files_out = []
    for rec in files:
        entry = {
            "path": rec.rel,
            "kind": rec.kind,
            "episode_uid": rec.id.episode_uid,
            "object_type": rec.id.object_type,
        }
        count = rec.track.n_frames if rec.kind == "video" else rec.n_rows
        entry["n_steps" if rec.kind == "trajectory" else "n_frames"] = count
        files_out.append(
            {**entry, "sha256": hashes[rec.rel], "bytes": stats[rec.rel][0]}
        )
    tool_entries = [
        {"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        for rel, data in tools.items()
    ]

    first = next(iter(metas.values()))
    object_types: dict[str, dict[str, Any]] = {}
    for meta in metas.values():
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
        builds = {meta["ffmpeg_version"] for meta in metas.values()}
        video = {**video, "ffmpeg_versions": sorted(builds)}
    generation: dict[str, Any] = {}
    export: dict[str, Any] = {}
    for meta in metas.values():
        for record, store in (
            (meta["generation_provenance"], generation),
            (meta["export_provenance"], export),
        ):
            if record is not None:
                store.setdefault(json.dumps(record, sort_keys=True), record)
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
            "n_frames": sum(m["n_frames"] for m in metas.values()),
            "n_episodes": len(metas),
            "n_files": len(files_out),
            "n_object_types": len(object_types),
            "n_object_assets": len({m["object_asset"] for m in metas.values()}),
            "n_channels": len(first["channels"]),
        },
        "object_types": dict(sorted(object_types.items())),
        "files": files_out,
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
    (root / fmt.TOOLS_DIR).mkdir(exist_ok=True)
    for rel, data in tools.items():
        _write_atomic(root / rel, data)
    _write_atomic(root / GITATTRIBUTES_NAME, GITATTRIBUTES)
    _write_atomic(
        root / fmt.METADATA_NAME,
        yaml.dump(
            metadata, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100
        ),
    )
    sums = sorted((f["path"], f["sha256"]) for f in [*files_out, *tool_entries])
    _write_atomic(
        root / fmt.CHECKSUMS_NAME, "".join(f"{sha}  {rel}\n" for rel, sha in sums)
    )
    if check_seed is not None:
        metadata["check"] = _published_check(root, check_seed, workers)
    return metadata
