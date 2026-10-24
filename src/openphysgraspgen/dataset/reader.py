# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from . import check_dataset as fmt
from .writer import CONTACT_INPUTS, MATRIX_SHAPES


def read_metadata(root: str | Path) -> dict[str, Any]:
    with open(Path(root) / fmt.METADATA_NAME, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def episode_metadata(path: str | Path) -> dict[str, Any]:
    """The metadata an episode's frames or trajectory file carries in its footer."""
    metadata = pq.read_metadata(path).metadata or {}
    raw = metadata.get(fmt.METADATA_KEY.encode())
    if raw is None:
        raise ValueError(f"{path} carries no {fmt.METADATA_KEY} metadata")
    return json.loads(raw)


def _numpy(
    array: pa.Array, dtype: str, shape: object, name: str, meta: dict
) -> np.ndarray:
    n = len(array)
    if dtype == "string":
        return np.asarray(array.to_pylist(), dtype=object)
    if shape == ():
        return array.to_numpy(zero_copy_only=False)
    size = array.type.list_size if pa.types.is_fixed_size_list(array.type) else None
    if size is not None:
        values = array.values.slice(array.offset * size, n * size)
        result = values.to_numpy(zero_copy_only=False).reshape(n, size)
    else:
        result = array.flatten().to_numpy(zero_copy_only=False).reshape(n, -1)
    if shape == "image":
        size_of = meta["image_size"][fmt.column_camera(name)]
        return result.reshape(n, size_of["height"], size_of["width"])
    if isinstance(shape, tuple) and shape[0] == "tets":
        return (
            result.reshape(n, meta["n_tets"])
            if shape[1] == 1
            else result.reshape(n, -1, 3, 3)
        )
    matrix = MATRIX_SHAPES.get(name.rsplit(".", 1)[1])
    return result.reshape(n, *matrix) if matrix else result


def _contacts(array: pa.Array) -> dict[str, np.ndarray]:
    offsets = np.asarray(array.offsets)
    flat = array.flatten()
    result = {"force.contact_offset": (offsets - offsets[0]).astype(np.int64)}
    for key, field_name in CONTACT_INPUTS.items():
        child = flat.field(field_name)
        shape = fmt.CONTACT_FIELDS[field_name][1]
        if shape:
            values = child.values.slice(child.offset * shape[0], len(child) * shape[0])
            result[key] = values.to_numpy(zero_copy_only=False).reshape(
                len(child), *shape
            )
        else:
            result[key] = child.to_numpy(zero_copy_only=False)
    return result


def _read(
    path: str | Path,
    spec: dict[str, tuple[str, object]],
    columns: Sequence[str] | None,
) -> dict[str, np.ndarray]:
    meta = episode_metadata(path)
    table = pq.read_table(path, columns=None if columns is None else list(columns))
    result: dict[str, np.ndarray] = {}
    for name in table.column_names:
        dtype, shape = spec[name]
        array = table.column(name).combine_chunks()
        if shape == "contacts":
            result.update(_contacts(array))
        else:
            result[name] = _numpy(array, dtype, shape, name, meta)
    return result


def read_frames(
    path: str | Path, columns: Sequence[str] | None = None
) -> dict[str, np.ndarray]:
    """An episode's frames as arrays with their shapes restored.

    Images are (N, H, W), per-tetrahedron values (N, T) and the deformation gradient
    (N, T, 3, 3). ``force.contacts`` becomes ``force.contact_offset`` and one flat
    array per contact field, as the writer takes them.
    """
    meta = episode_metadata(path)
    spec = fmt.frame_columns(tuple(meta["channels"]), tuple(meta["cameras"]))
    return _read(path, spec, columns)


def read_trajectory(
    path: str | Path, columns: Sequence[str] | None = None
) -> dict[str, np.ndarray]:
    """An episode's trajectory, one row per recorded step."""
    return _read(path, fmt.TRAJECTORY_COLUMNS, columns)
