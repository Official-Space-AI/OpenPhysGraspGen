# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
"""The dataset check, tools/check_dataset.py, on synthetic datasets, damaged ones too."""

import ast
import errno
import hashlib
import json
import os
import shutil
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
from dataset_helpers import (
    DOI,
    EPISODES,
    N_FRAMES,
    POSIX,
    SMALL,
    URL,
    build_dataset,
    criteria_text,
    fake_av1,
    needs_av1,
    nested,
    scene_frame,
)
from mp4_index import minimal_mp4, sample_entry

from openphysgraspgen.dataset import DatasetError, finalize_dataset
from openphysgraspgen.dataset import check_dataset as fmt

SEED = 7
SCRIPT = Path(fmt.__file__)
CRITERIA = "tools/check_criteria.yaml"
# The files of the dataset the tests build: 4 episodes of 3 types, 2 cameras.
N_TYPES, N_EPISODES = 3, len(EPISODES)
N_DATA_FILES = N_EPISODES * (2 + len(fmt.CAMERAS))
APPLE = "data/apple/ep-0.parquet"
APPLE_TRAJECTORY = "trajectories/apple/ep-0.parquet"
APPLE_FRONT = "videos/front/apple/ep-0.mp4"
APPLE_WRIST = "videos/wrist/apple/ep-0.mp4"


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    if sys.platform == "win32":
        pytest.skip("the fake ffmpeg is a shell script")
    tmp = tmp_path_factory.mktemp("base")
    build_dataset(tmp / "dataset", ffmpeg=fake_av1(tmp / "bin"))
    return tmp / "dataset"


@pytest.fixture
def root(base, tmp_path):
    target = tmp_path / "dataset"
    shutil.copytree(base, target, symlinks=True)
    return target


def check(root, **kwargs):
    criteria, digest = fmt.load_criteria(root / CRITERIA)
    options = {
        "seed": SEED,
        "criteria_path": root / CRITERIA,
        "criteria_sha256": digest,
        "workers": 2,
        **kwargs,
    }
    return fmt.run_checks(root, criteria, **options)


def errors(report, rel):
    """A file's errors, as the scale item lists them."""
    found = [
        e for e in report["items"]["scale"]["files_with_errors"] if e["file"] == rel
    ]
    return found[0]["errors"] if found else []


def mismatches(report):
    return [m["what"] for m in report["items"]["metadata"]["mismatches"]]


def read_meta(root):
    return yaml.safe_load((root / "metadata.yaml").read_text())


def write_meta(root, metadata):
    text = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True, width=100)
    (root / "metadata.yaml").write_text(text)


def restamp(root, rel, **fields):
    """List a changed file's sha256 and size, so it passes the gate and reaches the
    rule under test."""
    data = (root / rel).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    metadata = read_meta(root)
    for entry in metadata["files"]:
        if entry["path"] == rel:
            entry.update(sha256=digest, bytes=len(data), **fields)
    write_meta(root, metadata)
    lines = (root / "SHA256SUMS").read_text().splitlines()
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{digest}  {rel}\n" if line.endswith(f"  {rel}") else f"{line}\n"
            for line in lines
        )
    )


def compact(footer):
    return json.dumps(footer, sort_keys=True, separators=(",", ":")).encode()


def rewrite(root, rel, table=None, footer=None):
    """Rewrite a Parquet file with table(table) as its rows and footer as its footer
    (bytes, or a mapping written as the writer writes it), then restamp it."""
    path = root / rel
    parquet = pq.ParquetFile(path)
    rows = parquet.read()
    raw = parquet.metadata.metadata[fmt.METADATA_KEY.encode()]
    if table is not None:
        rows = table(rows)
    if footer is not None:
        raw = footer if isinstance(footer, bytes) else compact(footer)
    rows = rows.replace_schema_metadata({fmt.METADATA_KEY: raw})
    pq.write_table(rows, path, row_group_size=2)
    restamp(root, rel, **({} if table is None else {"n_frames": rows.num_rows}))


def footer_of(root, rel):
    return fmt.parse_footer(pq.read_metadata(root / rel).metadata[b"physgraspgen"])


def set_footer(root, change, object_type="apple", uid="ep-0"):
    """Change an episode's footer in both its Parquet files alike."""
    footer = change(footer_of(root, f"data/{object_type}/{uid}.parquet"))
    for kind in ("data", "trajectories"):
        rewrite(root, f"{kind}/{object_type}/{uid}.parquet", footer=footer)


def cell(name, row, value):
    """A table change: column name holds value at row, or value(the old one) when it
    is callable."""

    def change(table):
        i = table.schema.get_field_index(name)
        values = table.column(name).to_pylist()
        values[row] = value(values[row]) if callable(value) else value
        field = table.schema.field(i)
        return table.set_column(i, field, pa.array(values, type=field.type))

    return change


def list_file(root, rel, **entry):
    """List a file in metadata.yaml and SHA256SUMS, as finalize lists one."""
    data = (root / rel).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    metadata = read_meta(root)
    metadata["files"].append(
        {"path": rel, **entry, "sha256": digest, "bytes": len(data)}
    )
    write_meta(root, metadata)
    with open(root / "SHA256SUMS", "a") as stream:
        stream.write(f"{digest}  {rel}\n")


def copy_episode(root, uid, object_type, new_type, new_uid=None):
    """Copy an episode's files under another object type (and id), with its footer
    and episode_uid column changed to match, each file listed."""
    new_uid = new_uid or uid
    footer = footer_of(root, f"data/{object_type}/{uid}.parquet")
    footer = {**footer, "object_type": new_type, "episode_uid": new_uid}
    for kind in ("data", "trajectories"):
        source = root / f"{kind}/{object_type}/{uid}.parquet"
        rel = f"{kind}/{new_type}/{new_uid}.parquet"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(source)
        table = table.set_column(
            0,
            table.schema.field(0),
            pa.array([new_uid] * table.num_rows, type=pa.string()),
        )
        table = table.replace_schema_metadata({fmt.METADATA_KEY: compact(footer)})
        pq.write_table(table, root / rel, row_group_size=2)
        count = {"n_frames" if kind == "data" else "n_steps": table.num_rows}
        list_file(
            root,
            rel,
            kind="frames" if kind == "data" else "trajectory",
            episode_uid=new_uid,
            object_type=new_type,
            **count,
        )
    for camera in fmt.CAMERAS:
        rel = f"videos/{camera}/{new_type}/{new_uid}.mp4"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(root / f"videos/{camera}/{object_type}/{uid}.mp4", root / rel)
        list_file(
            root,
            rel,
            kind="video",
            episode_uid=new_uid,
            object_type=new_type,
            n_frames=footer["n_frames"],
        )


def column(name, kind=None, values=None, nullable=False, metadata="keep"):
    """A table change: column name retyped, refilled or with other field metadata."""

    def change(table):
        i = table.schema.get_field_index(name)
        field = table.schema.field(i)
        array = table.column(name)
        if values is not None:
            array = pa.array(values(array.to_pylist()), type=kind or field.type)
        elif kind is not None:
            array = array.cast(kind)
        new = pa.field(
            name,
            array.type,
            nullable=nullable,
            metadata=field.metadata if metadata == "keep" else metadata,
        )
        return table.set_column(i, new, array)

    return change


def replace_video(root, rel, data):
    """Put other bytes in a video and in its episode's footer, so only the rule under
    test speaks."""
    (root / rel).write_bytes(data)
    _, camera, object_type, name = rel.split("/")
    uid = name.removesuffix(".mp4")
    digest = hashlib.sha256(data).hexdigest()
    set_footer(
        root,
        lambda f: {**f, "video_sha256": {**f["video_sha256"], camera: digest}},
        object_type,
        uid,
    )
    restamp(root, rel)


def run(root, *args, flags=("-I",), script=None, env=None, cwd=None):
    """The shipped script run as a tester runs it: exit status, stdout, stderr and the
    RESULT payload."""
    script = script or root / "tools/check_dataset.py"
    done = subprocess.run(
        [sys.executable, *flags, str(script), *(str(a) for a in args)],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        check=False,
    )
    lines = done.stdout.strip().splitlines()
    payload = None
    if lines and lines[-1].startswith("RESULT "):
        payload = json.loads(lines[-1][len("RESULT ") :])
    return done.returncode, done.stdout, done.stderr, payload


def only_pyarrow_numpy_yaml(tmp_path):
    """A folder of links to NumPy, PyArrow and PyYAML alone, for PYTHONPATH under -S."""
    links = tmp_path / "links"
    links.mkdir()
    site = Path(np.__file__).resolve().parents[1]
    for name in ("numpy", "pyarrow", "yaml", "_yaml", "numpy.libs", "pyarrow.libs"):
        if (site / name).exists():
            (links / name).symlink_to(site / name)
    return links


# --- a complete dataset -----------------------------------------------------------


def worked_count(n_types, n_episodes, n_files, rgb, checksums):
    """The metadata item's comparisons on a clean dataset: format and version, five
    counts, channels and their count, conventions, three per type, eleven per frames
    file, the ffmpeg builds with rgb, provenance, per listed file six (five without
    sha256), per tool two (one), and SHA256SUMS: its paths, then each path."""
    per_file, per_tool = (6, 2) if checksums else (5, 1)
    sums = 1 + n_files + 2
    return (
        2
        + 8
        + 3 * n_types
        + 11 * n_episodes
        + rgb
        + 1
        + per_file * n_files
        + 2 * per_tool
        + sums
    )


def test_a_complete_dataset_passes_every_item(root):
    report = check(root)
    assert report["pass"]
    items = report["items"]
    assert list(items) == list(fmt.ITEMS)
    assert all(items[item]["pass"] for item in fmt.ITEMS)
    scale = items["scale"]
    assert (scale["n_frames"], scale["n_episodes"], scale["n_files"]) == (
        N_FRAMES,
        N_EPISODES,
        N_DATA_FILES,
    )
    assert scale["per_object_type"] == {
        "apple": {"n_episodes": 2, "n_frames": 5},
        "banana": {"n_episodes": 1, "n_frames": 3},
        "cherry": {"n_episodes": 1, "n_frames": 2},
    }
    metadata = items["metadata"]
    assert metadata["n_mismatches"] == 0
    assert metadata["n_checked"] == worked_count(
        N_TYPES, N_EPISODES, N_DATA_FILES, 1, True
    )
    assert metadata["spec"]["channels_in_every_episode"] == list(fmt.CHANNELS)
    assert metadata["spec"]["cameras_in_every_episode"] == list(fmt.CAMERAS)
    consistency = items["consistency"]
    assert (
        consistency["steps"]["n_frames"] == consistency["steps"]["n_pass"] == N_FRAMES
    )
    projection = consistency["projection"]
    assert (projection["n_candidates"], projection["n_sampled"]) == (N_FRAMES, 5)
    assert projection["statuses"] == {"pass": 5}
    for camera in fmt.CAMERAS:
        stats = projection["per_camera"][camera]
        assert stats["n_frames_in_view"] == 5
        assert stats["distance_px"]["max"] < 1
        assert stats["landing_classes"] == {"object": 10}
        # Every point lies on the object's near face, which the depth image shows.
        assert stats["depth_residual_m"]["max"] == 0.0
    diversity = items["diversity"]
    assert diversity["n_object_types"] == N_TYPES
    assert diversity["labels"] == {
        "success": {"n_episodes": 4, "n_frames": 10, "frame_fraction": 1.0}
    }
    assert diversity["tables"]["material.model"]["kind"] == "categorical"
    assert diversity["tables"]["material.E_pa"]["kind"] == "numeric"
    assert report["published_tools"]["pass"]
    assert report["ignored_files"] == {"n": 0, "paths": []}


def test_without_checksums_every_file_is_read_and_metadata_fails(root):
    report = check(root, checksums=False)
    assert not report["pass"]
    assert not report["items"]["metadata"]["pass"]
    assert report["items"]["metadata"]["n_mismatches"] == 0
    assert report["items"]["metadata"]["n_checked"] == worked_count(
        N_TYPES, N_EPISODES, N_DATA_FILES, 1, False
    )
    for item in ("scale", "consistency", "diversity"):
        assert report["items"][item]["pass"]
    assert "(sha256 not checked)" in fmt.format_table(report)


@pytest.mark.parametrize(
    ("item", "values"),
    [
        ("scale", {"min_frames": N_FRAMES + 1}),
        ("metadata", {"license": "CC0-1.0"}),
        ("consistency", {"min_contact_frames": N_FRAMES + 1}),
        ("diversity", {"min_object_types": N_TYPES + 1}),
    ],
)
def test_each_item_that_fails_fails_the_run(root, capsys, item, values):
    # Published criteria that this one item does not meet.
    republish(root, CRITERIA, criteria_text(**{**SMALL, **values}).encode())
    code, out, _, payload = main(capsys, root, "--seed", SEED)
    assert code == 1
    assert [i for i in fmt.ITEMS if not payload[i]["pass"]] == [item]
    assert payload["criteria"]["matches_published"] is True
    assert payload["script"]["matches_published"] is True
    assert payload["pass"] is False
    assert out.splitlines()[-2] == "Overall: FAIL"


@POSIX
def test_the_shipped_script_passes_the_dataset_as_a_tester_runs_it(root, tmp_path):
    assert (root / "tools/check_dataset.py").read_bytes() == SCRIPT.read_bytes()
    code, out, err, payload = run(root, root, "--seed", SEED, "--json", tmp_path / "r")
    assert code == 0, out + err
    lines = out.splitlines()
    assert lines[0] == "physgraspgen-public 2.0 dataset check"
    assert lines[1] == f"root: {root.resolve()}"
    assert lines[-3:-1] == ["", "Overall: PASS"]
    assert [line.split()[0] for line in lines[:-3] if line.endswith(" PASS")] == [
        "scale",
        "metadata",
        "spec",
        "consistency",
        "diversity",
    ]
    progress = [line.split(":")[0] for line in err.splitlines()]
    assert progress == ["hashing", "reading", "projecting"]
    assert err.splitlines()[1].startswith(
        f"reading: {N_DATA_FILES}/{N_DATA_FILES} files"
    )
    assert payload["pass"] is True
    assert payload["items_run"] == list(fmt.ITEMS)
    assert payload["scale"] == {
        "pass": True,
        "n_frames": N_FRAMES,
        "n_episodes": N_EPISODES,
        "n_files": N_DATA_FILES,
        "n_files_with_errors": 0,
    }
    assert payload["metadata"]["spec"]["n_channels"] == 6
    assert type(payload["consistency"]["projection"]["n_candidates"]) is int
    assert payload["consistency"]["projection_tolerance_provenance"].startswith(
        "placeholder: "
    )
    for key in ("criteria", "script"):
        assert payload[key]["matches_published"] is True
    digest = hashlib.sha256((root / "metadata.yaml").read_bytes()).hexdigest()
    assert payload["metadata_yaml"] == {"sha256": digest}

    def strict(text):
        raise ValueError(f"{text} in a strict JSON report")

    report = json.loads((tmp_path / "r").read_text(), parse_constant=strict)
    assert report["pass"] is True
    assert list(report) == [
        "pass",
        "root",
        "format",
        "format_version",
        "items_run",
        "items_not_run",
        "published_tools",
        "dataset",
        "environment",
        "ignored_files",
        "items",
        "criteria",
    ]
    assert report["criteria"]["scale"] == {"min_frames": N_FRAMES}
    assert report["environment"]["pyarrow"] == pa.__version__


@POSIX
def test_the_shipped_script_needs_only_numpy_pyarrow_and_pyyaml(root, tmp_path):
    env = {"PYTHONPATH": str(only_pyarrow_numpy_yaml(tmp_path)), "PATH": ""}
    code, _, err, payload = run(root, root, "--seed", SEED, flags=("-S",), env=env)
    assert code == 0, err
    assert payload["pass"] is True
    # The same interpreter cannot import the package or the test tools that way.
    probe = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import importlib.util as u; print([m for m in ('openphysgraspgen', "
                "'pytest', 'numpy', 'pyarrow', 'yaml') if u.find_spec(m)])"
            ),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert probe.stdout.strip() == "['numpy', 'pyarrow', 'yaml']"


@POSIX
def test_modules_beside_the_shipped_script_are_not_imported(root):
    for name in ("json.py", "__future__.py", "yaml.py", "hashlib.py"):
        (root / "tools" / name).write_text(
            "raise SystemExit('a module in tools ran')\n"
        )
    (root / "tools" / "numpy").mkdir()
    (root / "tools" / "numpy" / "__init__.py").write_text("raise SystemExit(3)\n")
    for flags in ((), ("-S",), ("-I",)):
        env = None
        if flags == ("-S",):
            env = {"PYTHONPATH": os.pathsep.join(sys.path[1:]), "PATH": ""}
        code, _out, err, payload = run(root, root, "--seed", SEED, flags=flags, env=env)
        assert code == 0, err
        assert payload["pass"] is True
    report = check(root)
    assert report["ignored_files"]["paths"] == [
        "tools/__future__.py",
        "tools/hashlib.py",
        "tools/json.py",
        "tools/numpy/__init__.py",
        "tools/yaml.py",
    ]
    assert report["pass"]


# --- the criteria -----------------------------------------------------------------


def criteria_file(tmp_path, text):
    path = tmp_path / "criteria.yaml"
    path.write_text(text)
    return path


# The acceptance values every dataset ships, with their types: a change is deliberate.
PUBLISHED = {
    "scale": {"min_frames": 1_000_000},
    "metadata": {
        "max_mismatches": 0,
        "required_channels": [
            "rgb",
            "depth",
            "force",
            "stress",
            "deformation_gradient",
            "segmentation",
        ],
        "required_cameras": ["front", "wrist"],
        "license": "CC-BY-4.0",
    },
    "consistency": {
        "min_pass_fraction": 1.0,
        "layers": ["rgb", "depth", "force", "stress"],
        "sample_frames": 1000,
        "min_contact_frames": 1000,
        "projection_radius_px": 4.0,
        "min_point_fraction": 1.0,
        "projection_tolerance_provenance": (
            "placeholder: one draw per camera rig measured pressed contacts up to 3.4 px "
            "from the object; to be measured on format 2.0 data and agreed"
        ),
        "distance_reach_px": 16.0,
    },
    "diversity": {
        "min_object_types": 10,
        "required_fact_kinds": ["material", "placement", "lighting", "camera"],
        "bins": 8,
        "log_bins_min_ratio": 100.0,
        "max_categories": 30,
    },
}


def test_the_package_ships_the_published_acceptance_values():
    criteria, _ = fmt.load_criteria(SCRIPT.with_name("check_criteria.yaml"))
    # As JSON text, 1 is not 1.0.
    assert json.dumps(criteria, sort_keys=True) == json.dumps(PUBLISHED, sort_keys=True)


BAD_CRITERIA = [
    (
        {"min_frames": "true"},
        "[scale] min_frames is True, not an integer of at least 1",
    ),
    ({"min_frames": "26.9"}, "[scale] min_frames is 26.9, not an integer"),
    ({"min_frames": "1e6"}, "[scale] min_frames is '1e6', not an integer"),
    ({"min_frames": "0"}, "[scale] min_frames is 0, not an integer of at least 1"),
    ({"max_mismatches": "-1"}, "[metadata] max_mismatches is -1, not an integer"),
    (
        {"required_channels": "[rgb, rgb]"},
        "[metadata] required_channels is ['rgb', 'rgb'], not a list of distinct channels",
    ),
    (
        {"required_channels": "[smell]"},
        "[metadata] required_channels is ['smell'], not a list of distinct channels",
    ),
    ({"required_cameras": "[]"}, "[metadata] required_cameras is [], not a nonempty"),
    ({"required_cameras": "[side]"}, "[metadata] required_cameras is ['side'], not a"),
    ({"license": "''"}, "[metadata] license is '', not a nonempty string"),
    (
        {"min_pass_fraction": "1.5"},
        "[consistency] min_pass_fraction is 1.5, not a number",
    ),
    ({"min_pass_fraction": ".nan"}, "[consistency] min_pass_fraction is nan, not a"),
    # Beyond the float range, which the check computes in.
    ({"min_pass_fraction": "1" + "0" * 400}, "[consistency] min_pass_fraction is 1000"),
    ({"layers": "[]"}, "[consistency] layers is [], not a nonempty list"),
    ({"layers": "rgb"}, "[consistency] layers is 'rgb', not a nonempty list"),
    ({"sample_frames": "0"}, "[consistency] sample_frames is 0, not an integer"),
    ({"min_contact_frames": "-1"}, "[consistency] min_contact_frames is -1, not an"),
    ({"projection_radius_px": "0"}, "[consistency] projection_radius_px is 0, not a"),
    (
        {"projection_radius_px": ".inf"},
        "[consistency] projection_radius_px is inf, not",
    ),
    ({"min_point_fraction": "-0.5"}, "[consistency] min_point_fraction is -0.5, not a"),
    ({"distance_reach_px": "-1.0"}, "[consistency] distance_reach_px is -1.0, not a"),
    (
        {"projection_tolerance_provenance": "' '"},
        "[consistency] projection_tolerance_provenance is ' ', not a nonempty string",
    ),
    ({"min_object_types": "false"}, "[diversity] min_object_types is False, not an"),
    (
        {"required_fact_kinds": "[material, '']"},
        "[diversity] required_fact_kinds is ['material', ''], not a list of distinct",
    ),
    ({"bins": "8.0"}, "[diversity] bins is 8.0, not an integer of at least 1"),
    ({"log_bins_min_ratio": "0.5"}, "[diversity] log_bins_min_ratio is 0.5, not a"),
    ({"max_categories": "'30'"}, "[diversity] max_categories is '30', not an integer"),
]


@pytest.mark.parametrize(("values", "message"), BAD_CRITERIA)
def test_criteria_values_are_refused_before_any_file_is_read(
    root, tmp_path, values, message, monkeypatch
):
    path = criteria_file(tmp_path, criteria_text(**{**SMALL, **values}))
    criteria, _ = fmt.load_criteria(path)
    hashed = []
    monkeypatch.setattr(fmt, "hash_files", lambda *a, **k: hashed.append(a))
    with pytest.raises(fmt.CheckError) as raised:
        fmt.run_checks(root, criteria, seed=SEED, criteria_path=path)
    assert str(raised.value).startswith("the criteria's " + message)
    assert not hashed


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("- a list\n", "is not a YAML mapping"),
        ("scale: [\n", "is not valid YAML"),
        ("scale: {min_frames: 1}\nscale: {min_frames: 2}\n", "repeats the key 'scale'"),
        (
            "scale:\n  min_frames: 1\n  min_frames: 2\n",
            "repeats the key 'min_frames' (line 3)",
        ),
        ("scale: !!python/object:os.system {}\n", "is not valid YAML"),
        ("\xff", "is not valid YAML"),
    ],
)
def test_a_criteria_file_that_is_not_one_is_refused(tmp_path, text, message):
    path = tmp_path / "criteria.yaml"
    path.write_bytes(text.encode("latin-1"))
    with pytest.raises(fmt.CheckError, match=f"criteria file {path} .*") as raised:
        fmt.load_criteria(path)
    assert message in str(raised.value)
    with pytest.raises(fmt.CheckError, match="does not exist \\(pass --criteria\\)"):
        fmt.load_criteria(tmp_path / "missing.yaml")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c: c.pop("scale"), "the criteria have no [scale] values"),
        (lambda c: c.update(scale=[1]), "the criteria have no [scale] values"),
        (
            lambda c: c["consistency"].pop("layers"),
            "the criteria's [consistency] lacks ['layers']",
        ),
        (
            lambda c: c.update(metadata="none"),
            "the criteria have no [metadata] values",
        ),
    ],
)
def test_criteria_without_a_value_an_item_reads_are_refused(root, change, message):
    criteria, _ = fmt.load_criteria(root / CRITERIA)
    change(criteria)
    with pytest.raises(fmt.CheckError) as raised:
        fmt.run_checks(root, criteria, seed=SEED)
    assert str(raised.value) == message


def test_an_item_reads_only_its_own_criteria(root):
    criteria, _ = fmt.load_criteria(root / CRITERIA)
    criteria["diversity"]["bins"] = 0
    criteria["unknown"] = {"anything": 1}
    report = fmt.run_checks(root, criteria, items=("scale",))
    assert report["items_run"] == ["scale"]
    assert report["items"]["scale"]["pass"]
    # The consistency item's camera floor is read from [metadata] even when only it runs.
    criteria["metadata"] = "front"
    with pytest.raises(fmt.CheckError, match=r"the criteria's \[metadata\] is not a"):
        fmt.run_checks(root, criteria, items=("consistency",), seed=SEED)
    criteria["metadata"] = {"required_cameras": ["side"]}
    with pytest.raises(fmt.CheckError, match=r"\[metadata\] required_cameras is"):
        fmt.run_checks(root, criteria, items=("consistency",), seed=SEED)


# --- the command line ---------------------------------------------------------------


def main(capsys, *args):
    code = fmt.main([str(a) for a in args])
    out, err = capsys.readouterr()
    lines = out.strip().splitlines()
    payload = (
        json.loads(lines[-1][7:]) if lines and lines[-1].startswith("RESULT ") else None
    )
    return code, out, err, payload


def refused(capsys, *args):
    code, out, err, payload = main(capsys, *args)
    assert code == 2, out + err
    assert out.count("\n") == 1, "a refusal prints no table"
    assert set(payload) == {"pass", "reason"} and payload["pass"] is False
    assert err.splitlines()[-1] == f"check_dataset: {payload['reason']}"
    return payload["reason"], err


@pytest.mark.parametrize(
    ("args", "reason"),
    [
        (["--seed", "abc"], "argument --seed: invalid int value: 'abc'"),
        (["--sample", "0", "--seed", "1"], "argument --sample: 0 is below 1"),
        (["--workers", "x"], "argument --workers: invalid int value: 'x'"),
        (["--seed", "1", "--colour"], "unrecognized arguments: --colour"),
        (["--seed", "1", "second"], "unrecognized arguments: second"),
        (["--only", " , "], "--only names no item; the items are scale, metadata, "),
        ([], "the consistency item draws its samples with the tester's seed: pass "),
        (["--only", "scale,size"], "unknown item(s) ['size']; known: ['scale', "),
    ],
)
def test_a_usage_error_ends_with_a_result_line(root, capsys, args, reason):
    message, err = refused(capsys, root, *args)
    assert message.startswith(reason)
    if reason.startswith(("argument", "unrecognized")):
        assert err.startswith("usage: ")


def test_a_missing_root_is_a_usage_error(capsys):
    message, _ = refused(capsys)
    assert message == "the following arguments are required: root"


def test_help_prints_the_options_and_runs_nothing(capsys):
    with pytest.raises(SystemExit) as stopped:
        fmt.main(["--help"])
    assert stopped.value.code == 0
    out = capsys.readouterr().out
    assert "RESULT" not in out
    for option in ("--criteria", "--only", "--workers", "--no-checksums", "--seed"):
        assert option in out
    assert "exit status: 0" in " ".join(out.split())


def test_the_report_path_is_checked_before_the_run(root, tmp_path, capsys):
    folder = tmp_path / "folder"
    folder.mkdir()
    message, _ = refused(capsys, root, "--seed", 1, "--json", folder)
    assert message == f"--json {folder} is a folder, not a file"
    message, _ = refused(capsys, root, "--seed", 1, "--json", tmp_path / "no/r.json")
    assert message == f"the folder of --json {tmp_path / 'no/r.json'} does not exist"
    if os.geteuid() != 0:
        folder.chmod(0o500)
        try:
            message, _ = refused(capsys, root, "--seed", 1, "--json", folder / "r")
            assert (
                message == f"the folder of --json {folder / 'r'} cannot be written to"
            )
        finally:
            folder.chmod(0o700)


def test_a_root_that_is_not_a_dataset_is_refused(root, tmp_path, capsys):
    criteria = ["--criteria", root / CRITERIA, "--seed", 1]
    message, _ = refused(capsys, root / "metadata.yaml", *criteria)
    assert message.startswith("dataset root ") and message.endswith(
        " is not a directory"
    )
    message, _ = refused(capsys, tmp_path, *criteria)
    assert message.endswith(
        "metadata.yaml does not exist; the dataset root is the folder that holds it"
    )
    (tmp_path / "metadata.yaml").write_text("files: []\nfiles: []\n")
    message, _ = refused(capsys, tmp_path, *criteria)
    assert message.endswith("repeats the key 'files' (line 2)")
    (tmp_path / "metadata.yaml").write_text("- 1\n")
    message, _ = refused(capsys, tmp_path, *criteria)
    assert message.endswith("metadata.yaml is not a YAML mapping")
    message, _ = refused(capsys, tmp_path, "--seed", 1)
    assert message == (
        f"criteria file {tmp_path / CRITERIA} does not exist (pass --criteria)"
    )


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (KeyboardInterrupt(), "the check was interrupted"),
        (RuntimeError("disk on fire"), "the check stopped: RuntimeError: disk on fire"),
        (
            OSError(errno.EMFILE, "Too many open files"),
            "the check stopped: OSError: [Errno 24] Too many open files",
        ),
    ],
)
def test_a_run_that_cannot_finish_ends_with_a_result_line(
    root, capsys, monkeypatch, error, reason
):
    def stop(*args, **kwargs):
        raise error

    monkeypatch.setattr(fmt, "hash_files", stop)
    message, _ = refused(capsys, root, "--seed", 1)
    assert message == reason


@pytest.mark.parametrize("stage", ["format_table", "summary", "jsonable"])
def test_a_report_that_cannot_be_built_is_a_refusal(root, capsys, monkeypatch, stage):
    def stop(*args, **kwargs):
        raise ValueError("cannot")

    monkeypatch.setattr(fmt, stage, stop)
    message, _ = refused(capsys, root, "--seed", 1, "--json", root.parent / "r.json")
    assert message == "the check stopped: ValueError: cannot"


def test_the_verdict_is_printed_before_the_report_is_written(
    root, tmp_path, capsys, monkeypatch
):
    target = tmp_path / "report.json"
    write = Path.write_text

    def fail(path, *args, **kwargs):
        if path == target:
            raise OSError("disk full")
        return write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail)
    code, out, err, payload = main(capsys, root, "--seed", SEED, "--json", target)
    assert code == 2
    assert payload["pass"] is True
    assert out.splitlines()[-2] == "Overall: PASS"
    assert err.splitlines()[-1] == (
        f"check_dataset: the report could not be written to {target}: disk full"
    )


def test_only_runs_the_items_named_in_their_order(root, capsys):
    code, out, _, payload = main(capsys, root, "--only", " diversity , scale ,scale")
    assert code == 0
    assert payload["items_run"] == ["scale", "diversity"]
    assert "items run: scale, diversity (not run: metadata, consistency)" in out
    assert out.splitlines()[-2] == (
        "Overall: PASS for the items run; not run: metadata, consistency"
    )
    assert set(payload) == {
        "pass",
        "items_run",
        "criteria",
        "script",
        "metadata_yaml",
        "scale",
        "diversity",
    }


def test_no_checksums_warns_and_fails(root, capsys):
    code, _out, err, payload = main(capsys, root, "--seed", SEED, "--no-checksums")
    assert code == 1
    assert err.splitlines()[0] == (
        "check_dataset: --no-checksums reads every file without comparing its sha256; "
        "a damaged file is then read as it is, and the metadata item cannot pass"
    )
    assert "hashing:" not in err
    assert payload["metadata"]["pass"] is False
    assert payload["scale"]["pass"] is True


def test_the_environment_is_checked(monkeypatch):
    monkeypatch.setattr(fmt.pa, "__version__", "16.1.0")
    with pytest.raises(fmt.CheckError, match="the check needs PyArrow 17.0 or newer; "):
        fmt.check_environment()
    monkeypatch.undo()
    monkeypatch.setattr(fmt.np, "__version__", "1.25.2")
    with pytest.raises(fmt.CheckError, match="needs NumPy 1.26 or newer; this is 1.25"):
        fmt.check_environment()
    monkeypatch.undo()
    monkeypatch.setattr(fmt, "_IMPORT_ERROR", ImportError("No module named 'pyarrow'"))
    with pytest.raises(fmt.CheckError, match="NumPy, PyArrow and PyYAML are required"):
        fmt.check_environment()


def test_a_missing_package_names_what_python_leaves_out(root, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    env = {"PYTHONPATH": str(empty), "PATH": ""}
    required = (
        "NumPy, PyArrow and PyYAML are required: pip install numpy pyarrow pyyaml "
        "(No module named 'numpy')"
    )
    code, _, _, payload = run(root, root, "--seed", SEED, flags=("-S",), env=env)
    assert (code, payload["reason"]) == (2, required)
    # -s, which -I implies, leaves out the packages pip install --user installed.
    code, _, _, payload = run(root, root, "--seed", SEED, flags=("-s", "-S"), env=env)
    assert (code, payload["reason"]) == (
        2,
        required + "; python -I and -s leave out packages that pip install --user "
        "installed: install them in a virtual environment, or run the script without "
        "-I or -s",
    )


# --- the files the check finds ------------------------------------------------------


@pytest.mark.parametrize(
    ("rel", "message"),
    [
        ("data/x.parquet", "is not where the format puts a frames file "),
        ("data/apple/sub/ep-0.parquet", "is not where the format puts a frames file "),
        (
            "data/apple/ep 0.parquet",
            "its episode id 'ep 0' must match [A-Za-z0-9][A-Za-z0-9._-]*",
        ),
        ("data/CON/ep-7.parquet", "its object type 'CON' must not be a device name"),
        ("data/apple/ep-7..parquet", "its episode id 'ep-7.' must not end in a dot"),
        ("trajectories/apple.parquet", "is not where the format puts a trajectory"),
        ("videos/side/apple/ep-0.mp4", "its camera 'side' is not one of the format's"),
        ("videos/apple/ep-0.mp4", "is not where the format puts a video file "),
    ],
)
def test_a_data_file_where_the_format_puts_none_is_misplaced(root, rel, message):
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(root / APPLE, root / rel)
    report = check(root)
    found = errors(report, rel)
    assert len(found) == 1 and message in found[0]
    assert not report["items"]["scale"]["pass"]
    assert f"files: {rel} is on disk but not listed" in mismatches(report)
    assert report["items"]["scale"]["n_frames"] == N_FRAMES


def test_files_that_are_not_data_files_are_ignored_and_reported(root):
    for rel in (
        "data/apple/ep-9.parquet.part",
        "data/apple/._ep-0.parquet",
        "data/.DS_Store",
        "data/apple/ep-5.PARQUET",
        "data/.hidden/ep-0.parquet",
        "videos/front/apple/ep-0.mov",
        "tools/notes.txt",
        ".cache/huggingface/x",
        "README.md",
        ".gitattributes",
    ):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"\0")
    report = check(root)
    assert report["pass"]
    assert report["ignored_files"]["paths"] == [
        "data/.DS_Store",
        "data/.hidden/ep-0.parquet",
        "data/apple/._ep-0.parquet",
        "data/apple/ep-5.PARQUET",
        "data/apple/ep-9.parquet.part",
        "tools/notes.txt",
        "videos/front/apple/ep-0.mov",
    ]
    assert "ignored: 7 files in data/, trajectories/, videos/ or tools/" in (
        fmt.format_table(report)
    )


@POSIX
def test_links_count_as_the_files_they_name(root, tmp_path):
    # A link to a file is the file; a dangling link and a linked folder are not.
    moved = tmp_path / "elsewhere.mp4"
    shutil.move(root / APPLE_FRONT, moved)
    (root / APPLE_FRONT).symlink_to(moved)
    (root / "data/apple/ep-9.parquet").symlink_to(tmp_path / "missing.parquet")
    (tmp_path / "folder").mkdir()
    shutil.copy(root / APPLE, tmp_path / "folder/ep-8.parquet")
    (root / "data/linked").symlink_to(tmp_path / "folder", target_is_directory=True)
    report = check(root)
    assert report["pass"]
    assert report["ignored_files"]["paths"] == [
        "data/apple/ep-9.parquet",
        "data/linked/",
    ]


@pytest.mark.parametrize(
    "path",
    [
        "/etc/hosts",
        "../outside.parquet",
        "data/../metadata.yaml",
        "data/./apple/ep-0.parquet",
        "data//apple/ep-0.parquet",
        "data\\apple\\ep-0.parquet",
        "README.md",
        "data/apple/ep-0.parquet.part",
        "tools/../SHA256SUMS",
    ],
)
def test_a_listed_path_the_format_does_not_define_is_never_touched(
    root, monkeypatch, path
):
    metadata = read_meta(root)
    metadata["files"].append({"path": path, "sha256": "0" * 64, "bytes": 1})
    metadata["tools"].append({"path": path, "sha256": "0" * 64, "bytes": 1})
    write_meta(root, metadata)
    touched = []
    hash_one = fmt.sha256_file
    monkeypatch.setattr(
        fmt, "sha256_file", lambda p: touched.append(Path(p)) or hash_one(p)
    )
    report = check(root)
    touched = [p for p in touched if p.resolve() != SCRIPT.resolve()]
    assert touched and all(root.resolve() in p.resolve().parents for p in touched)
    assert not any(p.name in ("hosts", "outside.parquet") for p in touched)
    i, j = len(metadata["files"]) - 1, len(metadata["tools"]) - 1
    assert (
        mismatches(report)[-20:].count(
            f"files[{i}] path {path!r} is not a path the format defines"
        )
        + mismatches(report).count(
            f"tools[{j}] path {path!r} is not a path the format defines"
        )
        == 2
    )


def test_a_list_that_is_not_one_is_a_mismatch(root):
    metadata = read_meta(root)
    metadata["files"] = metadata["files"] + [metadata["files"][0], "x", {"bytes": 1}]
    metadata["tools"] = {"path": "tools/check_dataset.py"}
    write_meta(root, metadata)
    found = mismatches(check(root))
    n = N_DATA_FILES
    assert f"files: {APPLE} is listed twice" in found
    assert f"files[{n + 1}] is not a mapping with a path" in found
    assert f"files[{n + 2}] is not a mapping with a path" in found
    assert "tools is not a list" in found
    assert "tools: tools/check_dataset.py is not listed" in found


# --- the hash gate --------------------------------------------------------------------


def flip(path, at=-100):
    data = bytearray(path.read_bytes())
    data[at] ^= 0x01
    path.write_bytes(bytes(data))


@pytest.mark.parametrize(
    "rel", [APPLE, APPLE_TRAJECTORY, APPLE_FRONT, "videos/wrist/banana/ep-2.mp4"]
)
def test_a_damaged_file_is_never_opened(root, rel):
    flip(root / rel)
    report = check(root)
    assert errors(report, rel)[0] == (
        "not opened: its sha256 is not the one metadata.yaml states"
    )
    assert not report["pass"]
    assert not report["items"]["scale"]["pass"]
    assert f"{rel}: sha256" in mismatches(report)
    steps = report["items"]["consistency"]["steps"]
    if rel == APPLE:
        # Its stated frames stay in the count, and fail it.
        assert steps["files_without_frame_count"] == [
            {
                "file": APPLE,
                "frames_stated": 3,
                "reasons": [
                    "not opened: its sha256 is not the one metadata.yaml states"
                ],
            }
        ]
        assert report["items"]["diversity"]["files_not_read"] == [APPLE]
        assert steps["n_frames"] == N_FRAMES
    elif rel.startswith("videos/"):
        # The frames of its episode have no video to show their RGB.
        camera = rel.split("/")[1]
        failed = steps["failed_by_reason"]
        assert any(r.startswith(f"rgb.{camera} has no video") for r in failed)


def test_a_file_metadata_yaml_does_not_list_is_never_opened(root):
    shutil.copy(root / APPLE, root / "data/apple/ep-9.parquet")
    report = check(root)
    assert errors(report, "data/apple/ep-9.parquet") == [
        "not opened: metadata.yaml does not list it",
        "its trajectory trajectories/apple/ep-9.parquet is not on disk",
    ]
    assert report["items"]["diversity"]["files_not_read"] == ["data/apple/ep-9.parquet"]


def test_a_stated_digest_is_compared_in_lower_case(root):
    metadata = read_meta(root)
    for entry in metadata["files"]:
        entry["sha256"] = entry["sha256"].upper()
    write_meta(root, metadata)
    report = check(root)
    assert report["items"]["scale"]["pass"]
    # SHA256SUMS states them in lower case, as metadata.yaml does once lowered.
    assert not any(m.startswith("SHA256SUMS") for m in mismatches(report))


@pytest.mark.parametrize("stated", [None, "missing"])
def test_a_file_without_a_stated_digest_is_never_opened(root, stated):
    metadata = read_meta(root)
    entry = next(e for e in metadata["files"] if e["path"] == APPLE)
    if stated is None:
        entry["sha256"] = None
    else:
        del entry["sha256"]
    write_meta(root, metadata)
    report = check(root)
    assert errors(report, APPLE)[0] == (
        "not opened: its sha256 is not the one metadata.yaml states"
    )


def test_a_git_lfs_pointer_is_named(root):
    (root / APPLE).write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n"
    )
    report = check(root)
    assert errors(report, APPLE)[0] == (
        "not opened: its sha256 is not the one metadata.yaml states; it is a Git LFS "
        "pointer, not the file (fetch it with git lfs pull)"
    )


def test_without_checksums_a_damaged_file_is_read_as_it_is(root):
    path = root / APPLE
    data = bytearray(path.read_bytes())
    # The first page of the first column: its checksum no longer holds.
    group = pq.read_metadata(path).row_group(0).column(0)
    data[group.data_page_offset + group.total_compressed_size // 2] ^= 0x01
    path.write_bytes(bytes(data))
    report = check(root, checksums=False)
    (error,) = [e for e in errors(report, APPLE) if "cannot be read" in e]
    assert error.startswith("cannot be read: OSError: ")
    # The page's checksum, not a decoder that happens to fail, finds the damage.
    assert "CRC checksum verification failed" in error


def test_a_file_changed_after_it_was_hashed_is_not_judged(root, monkeypatch):
    hash_files = fmt.hash_files

    def hash_then_change(root_, rels, workers, progress=None):
        out = hash_files(root_, rels, workers, progress)
        stat = os.stat(root_ / APPLE)
        os.utime(root_ / APPLE, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
        return out

    monkeypatch.setattr(fmt, "hash_files", hash_then_change)
    report = check(root)
    assert errors(report, APPLE) == [
        "changed while the check ran: it is not the file that was hashed"
    ]


def test_running_out_of_file_descriptors_stops_the_run(root, monkeypatch):
    def exhausted(*args, **kwargs):
        raise OSError(errno.EMFILE, "Too many open files")

    monkeypatch.setattr(fmt.pq, "ParquetFile", exhausted)
    with pytest.raises(fmt.CheckError, match="the check ran out of file descriptors "):
        check(root)


@POSIX
def test_the_check_holds_no_file_open_between_reads(tmp_path):
    resource = pytest.importorskip("resource")
    root = tmp_path / "dataset"
    episodes = [("apple", f"ep-{i}", 1) for i in range(40)]
    build_dataset(
        root,
        ffmpeg=fake_av1(tmp_path / "bin"),
        episodes=episodes,
        criteria=criteria_text(**{**SMALL, "min_frames": 40, "min_object_types": 1}),
    )

    def few_descriptors():
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

    done = subprocess.run(
        [
            sys.executable,
            "-I",
            str(root / "tools/check_dataset.py"),
            str(root),
            "--seed",
            "1",
            "--workers",
            "1",
        ],
        capture_output=True,
        text=True,
        preexec_fn=few_descriptors,
        check=False,
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]


# --- a frames file --------------------------------------------------------------------

# apple's ep-0: 3 frames at steps 2, 4 and 6, row groups of 2 frames, 3 contacts a
# frame, 5 tetrahedra, both cameras at 16 x 24.

FOOTERS = [
    (
        lambda f: {k: v for k, v in f.items() if k != "facts"},
        "metadata keys ['facts'] are missing",
    ),
    (
        lambda f: {**f, "colour": "red"},
        "metadata keys ['colour'] are not in the format",
    ),
    (
        lambda f: {**f, "format": "physgraspgen"},
        "format is 'physgraspgen', not 'physgraspgen-public'",
    ),
    (lambda f: {**f, "format_version": 2.0}, "format_version is 2.0, not '2.0'"),
    (
        lambda f: {**f, "episode_uid": "ep-9"},
        "holds episode 'ep-9' of 'apple', not 'ep-0' of 'apple'",
    ),
    (
        lambda f: {**f, "object_asset": " "},
        "metadata object_asset is ' ', not a nonempty string",
    ),
    (
        lambda f: {**f, "object_asset": ["x"]},
        "metadata object_asset is ['x'], not a nonempty string",
    ),
    (
        lambda f: {**f, "object_asset": {"a": 1}},
        "metadata object_asset is {'a': 1}, not a nonempty string",
    ),
    (
        lambda f: {**f, "label": "won"},
        "metadata label is 'won', not one of success, fracture, fail, unknown",
    ),
    (lambda f: {**f, "n_frames": 4}, "metadata n_frames is 4 but the file has 3 rows"),
    (
        lambda f: {**f, "n_frames": 3.0},
        "metadata n_frames is 3.0, not an integer of at least 1",
    ),
    (lambda f: {**f, "n_tets": True}, "metadata n_tets is True, not an integer"),
    (lambda f: {**f, "fps": 7}, "metadata fps 7 does not divide recorded_fps 60"),
    (
        lambda f: {**f, "channels": f["channels"][::-1]},
        (
            "not a list of channels from rgb, depth, force, stress, deformation_gradient, "
            "segmentation in that order"
        ),
    ),
    (
        lambda f: {**f, "cameras": ["wrist", "front"]},
        "metadata cameras is ['wrist', 'front'], not a list of cameras",
    ),
    (
        lambda f: {**f, "image_size": {**f["image_size"], "front": [16, 24]}},
        "metadata image_size is",
    ),
    (
        lambda f: {
            **f,
            "image_size": {**f["image_size"], "front": {"height": 16, "width": 25}},
        },
        "positive integers and even of at most 2,147,483,647 pixels",
    ),
    (
        lambda f: {**f, "segmentation_classes": f["segmentation_classes"][::-1]},
        "metadata segmentation_classes is ['prop', ",
    ),
    (
        lambda f: {**f, "video": {**f["video"], "crf": 0}},
        "allows: crf must be an integer from 1 to 63",
    ),
    (
        lambda f: {**f, "video": {**f["video"], "encoder": "libaom-av1"}},
        "allows: must be encoded by libsvtav1",
    ),
    (
        lambda f: {**f, "ffmpeg_version": ""},
        "metadata ffmpeg_version is '', not a nonempty string with rgb",
    ),
    (
        lambda f: {**f, "ffmpeg_version": ["x"]},
        "metadata ffmpeg_version is ['x'], not a nonempty string with rgb",
    ),
    (
        lambda f: {**f, "ffmpeg_version": {"a": 1}},
        "metadata ffmpeg_version is {'a': 1}, not a nonempty string with rgb",
    ),
    (
        lambda f: {
            **f,
            "video_sha256": {k: v.upper() for k, v in f["video_sha256"].items()},
        },
        "not the sha256 of each camera's video, in lower-case hex",
    ),
    (
        lambda f: {**f, "facts": {**f["facts"], "camera": {"rig": None}}},
        "metadata facts.camera holds no value but null",
    ),
    (
        lambda f: {**f, "task": ["lift"]},
        "metadata task is ['lift'], not a mapping or null",
    ),
    (
        lambda f: {**f, "source_n_frames": 5},
        "metadata source_n_frames is 5, not null or an integer of at least n_steps, 6",
    ),
    (
        lambda f: {**f, "source_cut_at_row": -2},
        "metadata source_cut_at_row is -2, not an integer of at least -1",
    ),
]


@pytest.mark.parametrize(("change", "message"), FOOTERS)
def test_a_footer_the_writer_would_not_write_is_found(root, change, message):
    set_footer(root, change)
    report = check(root)
    found = errors(report, APPLE)
    assert any(message in e for e in found), found
    assert not report["items"]["scale"]["pass"]
    # Its trajectory carries the same footer, which is held to the rules through it.
    assert errors(report, APPLE_TRAJECTORY) == []


def drop_footer(root):
    table = pq.read_table(root / APPLE).replace_schema_metadata(None)
    pq.write_table(table, root / APPLE, row_group_size=2)
    restamp(root, APPLE)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (None, "carries no physgraspgen metadata"),
        (b'{"a": "\xff"}', "its physgraspgen metadata is not UTF-8: "),
        (b"{", "its physgraspgen metadata is not JSON: Expecting property name"),
        (b'{"n_tets": NaN}', "its physgraspgen metadata is not JSON: NaN is not a"),
        (b'{"n_tets": 1e999}', "is not JSON: 1e999 is out of the float64 range"),
        (b'{"n_tets": 99999999999999999999}', "is out of the int64 range"),
        (b'{"a": 1, "a": 1}', "is not JSON: the key 'a' is repeated in an object"),
        (b"[]", "its physgraspgen metadata is not a JSON object"),
    ],
)
def test_a_footer_that_is_not_strict_json_is_found(root, raw, message):
    if raw is None:
        drop_footer(root)
    else:
        rewrite(root, APPLE, footer=raw)
    report = check(root)
    found = errors(report, APPLE)
    assert message in found[0], found
    # Without a footer the file states no columns to check, and no cameras' videos.
    assert not any(e.startswith("column") for e in found)
    assert report["items"]["diversity"]["files_not_read"] == [APPLE]


@pytest.mark.parametrize(("depth", "ok"), [(30, True), (31, False), (1200, False)])
def test_a_footer_nests_at_most_32_levels_on_any_python(root, capsys, depth, ok):
    # Written as text: a Python may not encode so deep a value. The footer is level 1,
    # facts level 2, so facts.material nested 30 levels makes 32.
    footer = footer_of(root, APPLE)
    raw = compact({**footer, "facts": {**footer["facts"], "material": "deep"}})
    raw = raw.replace(b'"deep"', b'{"a":' * depth + b"1.0" + b"}" * depth)
    for rel in (APPLE, APPLE_TRAJECTORY):
        rewrite(root, rel, footer=raw)
    # Too deep is the file's finding: the run completes, and fails.
    code, _, _, payload = main(capsys, root, "--seed", SEED)
    assert (code, payload["pass"]) == ((0, True) if ok else (1, False))
    report = check(root)
    too_deep = "its physgraspgen metadata is not JSON: it nests deeper than 32 levels"
    for rel in (APPLE, APPLE_TRAJECTORY):
        assert errors(report, rel) == ([] if ok else [too_deep])
    tables = report["items"]["diversity"]["tables"]
    assert ("material." + ".".join(["a"] * 30) in tables) is ok


def drop(name):
    return lambda t: t.remove_column(t.schema.get_field_index(name))


COLUMNS = [
    (drop("source_frame_index"), "columns ['source_frame_index'] are missing"),
    (
        lambda t: t.append_column("extra", pa.array([0] * t.num_rows)),
        "columns ['extra'] are not in the format",
    ),
    (
        lambda t: t.select([*t.column_names[1:], t.column_names[0]]),
        "its columns are not in the format's order",
    ),
    (column("robot.waypoint", pa.int64()), "column robot.waypoint is int64, not int32"),
    (
        column("camera.front.intrinsics", pa.list_(pa.float32(), 9)),
        (
            "column camera.front.intrinsics is fixed_size_list<element: float>[9], not "
            "fixed_size_list<item: double>[9]"
        ),
    ),
    (column("robot.waypoint", nullable=True), "column robot.waypoint may hold nulls"),
    (
        column("camera.front.intrinsics", metadata={"shape": "[9]"}),
        (
            "column camera.front.intrinsics has the field metadata {'shape': '[9]'}, "
            "not {'shape': '[3, 3]'}"
        ),
    ),
    (
        column("depth.wrist.depth_m", metadata=None),
        "column depth.wrist.depth_m has the field metadata {}, not {'shape': '[16, 24]'}",
    ),
    (lambda t: t.slice(0, 0), "holds no frame; an episode has at least one"),
]


@pytest.mark.parametrize(("change", "message"), COLUMNS)
def test_columns_the_format_does_not_define_are_found(root, change, message):
    rewrite(root, APPLE, table=change)
    found = errors(check(root), APPLE)
    assert any(e.startswith(message) for e in found), found


def test_a_file_without_its_arrow_schema_is_named(root):
    table = pq.read_table(root / APPLE)
    raw = pq.read_metadata(root / APPLE).metadata[b"physgraspgen"]
    with pq.ParquetWriter(root / APPLE, table.schema, store_schema=False) as writer:
        writer.write_table(table, row_group_size=2)
        writer.add_key_value_metadata({fmt.METADATA_KEY: raw})
    restamp(root, APPLE)
    found = errors(check(root), APPLE)
    assert found[0] == (
        "its footer holds no ARROW:schema entry, which states each column's type and "
        "shape"
    )
    assert any(e.startswith("column camera.front.intrinsics is list<") for e in found)


def at(index, value):
    """A value change: element index of a list replaced."""

    def change(values):
        values = list(values)
        values[index] = value
        return values

    return change


def contact(i, key, value):
    def change(contacts):
        contacts = [dict(c) for c in contacts]
        contacts[i][key] = value
        return contacts

    return change


nan, inf = float("nan"), float("inf")

ROWS = [
    (cell("episode_uid", 1, "ep-9"), "frame 1: episode_uid is 'ep-9', not 'ep-0'"),
    (cell("frame_index", 2, 5), "frame 2: frame_index is 5, not 2"),
    (
        cell("sim_step", 0, 0),
        (
            "frame 0: sim_step 0 is not on the 30 fps grid, a positive multiple of 2 (and "
            "1 more frames)"
        ),
    ),
    (
        cell("sim_step", 1, 2),
        "frame 1: sim_step 2 does not follow 2 on the 30 fps grid, which goes on at 4",
    ),
    (
        cell("sim_step", 2, 7),
        "frame 2: sim_step 7 does not follow 4 on the 30 fps grid",
    ),
    (
        cell("sim_step", 2, 8),
        "frame 2: sim_step 8 does not follow 4 on the 30 fps grid, which goes on at 6",
    ),
    (cell("force.sim_step", 1, 6), "frame 1: force was captured at another step"),
    (
        cell("segmentation.wrist.sim_step", 0, 4),
        "frame 0: segmentation.wrist was captured at another step",
    ),
    (
        cell("sim_time_s", 1, 0.07),
        "frame 1: sim_time_s is 0.07, not sim_step / recorded_fps (0.06666666666666667)",
    ),
    (
        cell("rgb.front.video_frame", 2, 0),
        "frame 2: rgb.front.video_frame is 0, not the frame's index 2",
    ),
    (
        cell("force.n_contacts", 0, 5),
        "frame 0: force.n_contacts is 5, not the 3 entries of force.contacts",
    ),
    (
        cell("stress.von_mises", 0, lambda v: v[:-1]),
        "frame 0: stress.von_mises holds 4 values, not n_tets x 1 = 5",
    ),
    (
        cell("deformation_gradient.F", 1, lambda v: v[:36]),
        "frame 1: deformation_gradient.F holds 36 values, not n_tets x 9 = 45",
    ),
    (
        cell("camera.front.intrinsics", 0, at(0, None)),
        "column camera.front.intrinsics holds nulls",
    ),
    (
        cell("force.contacts", 1, contact(0, "force", [None, 0.0, 1.0])),
        "column force.contacts holds nulls",
    ),
    (cell("robot.ee_pose", 1, at(3, nan)), "frame 1: robot.ee_pose must be finite"),
    (
        cell("robot.grip_contact_width_m", 0, inf),
        "frame 0: robot.grip_contact_width_m must be finite or NaN",
    ),
    (
        cell("force.gripper_net_force", 2, at(0, -inf)),
        "frame 2: force.gripper_net_force must be finite",
    ),
    (
        cell("force.contacts", 0, contact(1, "force", [nan, 0.0, 1.0])),
        "frame 0: force.contacts.force must be finite",
    ),
    (
        cell("force.contacts", 2, contact(2, "point_world", [0.0, inf, 1.0])),
        "frame 2: force.contacts.point_world must be finite",
    ),
    (
        cell("deformation_gradient.F", 1, at(44, inf)),
        "frame 1: deformation_gradient.F must be finite",
    ),
    (
        cell("segmentation.front.mask", 0, at(7, 9)),
        "frame 0: segmentation.front.mask holds a value outside the segmentation classes",
    ),
    (
        cell("camera.wrist.intrinsics", 2, at(1, 0.5)),
        (
            "frame 2: camera.wrist.intrinsics is not a pinhole K: K[0][1] (skew) and K[1][0] "
            "must be 0"
        ),
    ),
    (
        cell("camera.front.world_to_camera", 1, at(0, 2.0)),
        (
            "frame 1: camera.front.world_to_camera is not a rigid transform: its rotation is "
            "not orthonormal"
        ),
    ),
    (
        cell("stress.von_mises", 0, lambda v: [1.0 if np.isnan(x) else x for x in v]),
        (
            "frame 0: stress.von_mises must be NaN exactly where stress.valid is false, and "
            "finite elsewhere"
        ),
    ),
    (cell("robot.phase", 2, ""), "frame 2: robot.phase must be a nonempty string"),
]


@pytest.mark.parametrize(("change", "message"), ROWS)
def test_a_frame_that_breaks_a_value_rule_is_found(root, change, message):
    rewrite(root, APPLE, table=change)
    found = errors(check(root), APPLE)
    assert any(e.startswith(message) for e in found), found


def test_the_result_does_not_depend_on_how_many_frames_are_read_at_once(
    root, monkeypatch
):
    # About 16 MiB of values a read: 5 frames at 480 x 640, a whole file here.
    assert fmt.frames_per_read({"front": (480, 640), "wrist": (480, 640)}, 3000) == 5
    rewrite(root, APPLE, table=cell("robot.ee_pose", 2, at(3, nan)))
    rewrite(
        root,
        "data/banana/ep-2.parquet",
        table=cell("segmentation.front.mask", 1, at(7, 9)),
    )
    whole = check(root, sample=N_FRAMES)
    assert errors(whole, APPLE) == ["frame 2: robot.ee_pose must be finite"]
    monkeypatch.setattr(fmt, "BYTES_PER_READ", 1)
    assert fmt.frames_per_read({"front": (16, 24), "wrist": (16, 24)}, 5) == 1
    one = check(root, sample=N_FRAMES)
    assert one["items"] == whole["items"]


def test_a_rule_broken_on_several_frames_is_counted(root):
    rewrite(
        root,
        APPLE,
        table=column(
            "robot.ee_pose", values=lambda rows: [at(3, nan)(r) for r in rows]
        ),
    )
    assert errors(check(root), APPLE) == [
        "frame 0: robot.ee_pose must be finite (and 2 more frames)"
    ]


def test_a_files_findings_come_in_the_rules_order(root):
    set_footer(root, lambda f: {**f, "label": "won", "n_tets": 0})
    rewrite(
        root,
        APPLE,
        table=lambda t: cell("robot.ee_pose", 0, at(3, nan))(
            t.append_column("extra", pa.array([0] * t.num_rows))
        ),
    )
    (root / APPLE_TRAJECTORY).unlink()
    found = errors(check(root), APPLE)
    # The footer's rules, the columns, then the episode's; the values are not read.
    assert found == [
        "metadata label is 'won', not one of success, fracture, fail, unknown",
        "metadata n_tets is 0, not an integer of at least 1",
        "columns ['extra'] are not in the format",
        f"its trajectory {APPLE_TRAJECTORY} is not on disk",
    ]


# --- a trajectory ---------------------------------------------------------------------

# apple's ep-0 records steps 1 to 6; step 3 is not finite.

TRAJECTORIES = [
    (
        lambda root: rewrite(
            root, APPLE_TRAJECTORY, footer={**footer_of(root, APPLE), "label": "fail"}
        ),
        f"its physgraspgen metadata is not that of {APPLE} (it differs in ['label'])",
    ),
    (
        lambda root: rewrite(
            root, APPLE_TRAJECTORY, footer=json.dumps(footer_of(root, APPLE)).encode()
        ),
        f"its physgraspgen metadata is not that of {APPLE}",
    ),
    (
        lambda root: rewrite(
            root, APPLE_TRAJECTORY, table=column("robot.waypoint", pa.int64())
        ),
        "column robot.waypoint is int64, not int32",
    ),
    (
        lambda root: rewrite(root, APPLE_TRAJECTORY, table=lambda t: t.slice(0, 5)),
        "holds 5 steps, its metadata says 6",
    ),
    (
        lambda root: rewrite(root, APPLE_TRAJECTORY, table=cell("sim_step", 0, 2)),
        (
            "row 0: sim_step 2 is not 1; the trajectory holds every recorded step from "
            "step 1 (and 1 more rows)"
        ),
    ),
    (
        lambda root: rewrite(root, APPLE_TRAJECTORY, table=cell("sim_step", 3, 5)),
        (
            "row 3: sim_step 5 is not 4; the trajectory holds every recorded step from "
            "step 1 (and 1 more rows)"
        ),
    ),
    (
        lambda root: rewrite(
            root, APPLE_TRAJECTORY, table=cell("object.centroid", 0, at(1, nan))
        ),
        "row 0: object.centroid must be finite where finite is true",
    ),
    (
        lambda root: rewrite(
            root, APPLE_TRAJECTORY, table=cell("object.bbox_max", 2, [0.0, 0.0, 0.0])
        ),
        "row 2: object.bbox_max must be NaN where finite is false",
    ),
    (
        lambda root: rewrite(
            root, APPLE_TRAJECTORY, table=cell("robot.arm_q_setpoint", 2, at(0, nan))
        ),
        "row 2: robot.arm_q_setpoint must be finite",
    ),
    (
        lambda root: rewrite(root, APPLE_TRAJECTORY, table=cell("sim_time_s", 1, 0.5)),
        "row 1: sim_time_s is 0.5, not sim_step / recorded_fps (0.03333333333333333)",
    ),
    (
        lambda root: rewrite(root, APPLE_TRAJECTORY, table=cell("robot.phase", 4, "")),
        "row 4: robot.phase must be a nonempty string",
    ),
    (
        lambda root: rewrite(
            root, APPLE_TRAJECTORY, table=cell("robot.arm_q", 3, at(0, 9.0))
        ),
        (
            f"frame 1 of {APPLE}: robot.arm_q differs between the frame and the trajectory "
            "at this step"
        ),
    ),
    (
        lambda root: rewrite(
            root, APPLE_TRAJECTORY, table=cell("episode_uid", 5, "ep-1")
        ),
        "row 5: episode_uid is 'ep-1', not 'ep-0'",
    ),
    (
        lambda root: rewrite(
            root, APPLE_TRAJECTORY, table=cell("object.centroid", 1, at(0, None))
        ),
        "column object.centroid holds nulls",
    ),
]


@pytest.mark.parametrize(("damage", "message"), TRAJECTORIES)
def test_a_trajectory_the_writer_would_not_write_is_found(root, damage, message):
    damage(root)
    report = check(root)
    assert message in errors(report, APPLE_TRAJECTORY)
    assert not report["items"]["scale"]["pass"]
    # Its frames are still judged layer by layer, and pass.
    steps = report["items"]["consistency"]["steps"]
    assert steps["n_pass"] == steps["n_frames"] == N_FRAMES


def test_a_trajectorys_rows_are_judged_whatever_else_its_footer_breaks(root):
    set_footer(root, lambda f: {**f, "label": "bogus"})
    rewrite(root, APPLE_TRAJECTORY, table=cell("sim_time_s", 1, 0.5))
    assert errors(check(root), APPLE_TRAJECTORY) == [
        "row 1: sim_time_s is 0.5, not sim_step / recorded_fps (0.03333333333333333)"
    ]


def test_a_trajectory_must_hold_every_frames_step(root):
    set_footer(root, lambda f: {**f, "n_steps": 4})
    rewrite(root, APPLE_TRAJECTORY, table=lambda t: t.slice(0, 4))
    assert errors(check(root), APPLE_TRAJECTORY) == [
        f"lacks the step 6 of a frame of {APPLE}"
    ]


def test_a_missing_trajectory_fails_its_episode(root):
    (root / APPLE_TRAJECTORY).unlink()
    report = check(root)
    assert errors(report, APPLE) == [
        f"its trajectory {APPLE_TRAJECTORY} is not on disk"
    ]
    assert f"files: {APPLE_TRAJECTORY} is listed but not on disk" in mismatches(report)
    assert report["items"]["scale"]["n_frames"] == N_FRAMES - 3


# --- a video --------------------------------------------------------------------------


def video(n_frames=3, width=24, height=16, **kwargs):
    entry = kwargs.pop("entry", None) or sample_entry(width=width, height=height)
    return minimal_mp4(n_frames, entry=entry, **kwargs)


VIDEOS = [
    (b"not an mp4 file", "its MP4 index cannot be read: "),
    (video()[:-8], "its MP4 index cannot be read: box b'mdat' runs past the end of"),
    (video(fragmented=True), "its MP4 index cannot be read: fragmented MP4 is not"),
    (video(moov_boxes=2), "its MP4 index cannot be read: two moov boxes"),
    (video(handler=b"soun"), "holds 0 video tracks, not one"),
    (video(tracks=2), "holds 2 video tracks, not one"),
    (
        video(descriptions=2),
        "its MP4 index cannot be read: a video track with 2 sample",
    ),
    (video(entry=sample_entry(b"avc1", 24, 16)), "is coded as avc1, not AV1 (av01)"),
    (
        video(entry=sample_entry(width=24, height=16, flags=0x4C)),
        "is 10-bit 4:2:0, not",
    ),
    (video(entry=sample_entry(width=24, height=16, flags=0x00)), "is 8-bit 4:4:4, not"),
    (video(width=16, height=24), "is 24 x 16 pixels, not the camera's 16 x 24"),
    (video(4), "holds 4 frames, not 3"),
    (video(2), "holds 2 frames, not 3"),
    (video(runs=[(2, 512), (1, 1024)]), "does not run at a constant 30 fps"),
    (video(delta=513), "does not run at a constant 30 fps"),
    (video(edits=[(100, 512, 1, 0)]), "its edit list does not play every frame from"),
    (
        video(edits=[(17, -1, 1, 0), (100, 0, 1, 0)]),
        "its edit list does not play every frame from",
    ),
    (video(duration=1024), "its media lasts 1024 units, its frames 1536"),
    (video(ctts=[0, 512, 0]), "shows its frames out of their stored order (ctts)"),
]


@pytest.mark.parametrize(("data", "message"), VIDEOS)
def test_a_video_that_is_not_its_cameras_is_found_from_its_index(root, data, message):
    replace_video(root, APPLE_WRIST, data)
    report = check(root)
    found = errors(report, APPLE_WRIST)
    assert any(e.startswith(message) for e in found), found
    # Its episode's frames have no RGB of the wrist camera.
    failed = report["items"]["consistency"]["steps"]["failed_by_reason"]
    reason = f"rgb.wrist has no video of the episode's frames: {APPLE_WRIST} {found[0]}"
    assert failed[reason] == 3


@pytest.mark.parametrize("edits", [None, [(1000, 0, 1, 0)]])
def test_a_video_index_as_muxers_write_it_passes(root, edits):
    replace_video(root, APPLE_WRIST, video(edits=edits, payload=b"other frames"))
    assert check(root)["pass"]


def test_a_video_filed_under_another_camera_is_found(root):
    # The front video has as many frames, the same size and rate: only its bytes differ.
    shutil.copy(root / APPLE_FRONT, root / APPLE_WRIST)
    restamp(root, APPLE_WRIST)
    report = check(root)
    assert errors(report, APPLE_WRIST) == [
        "is not the video its episode was written with"
    ]
    assert not report["items"]["scale"]["pass"]


@pytest.mark.parametrize(
    ("data", "stated", "message"),
    [
        (video(2), True, "holds 2 frames, not 3"),
        (
            video(width=48, height=32),
            True,
            "is 32 x 48 pixels, not the camera's 16 x 24",
        ),
        (video(payload=b"other frames"), False, "is not the video its episode was"),
    ],
    ids=["short", "other-size", "other-bytes"],
)
def test_a_video_is_held_to_its_episode_whatever_else_its_footer_breaks(
    root, data, stated, message
):
    if stated:
        replace_video(root, APPLE_FRONT, data)
    else:
        (root / APPLE_FRONT).write_bytes(data)
        restamp(root, APPLE_FRONT)
    # A footer error that has nothing to do with the video.
    set_footer(root, lambda f: {**f, "label": "bogus"})
    report = check(root)
    assert errors(report, APPLE)[0].startswith("metadata label is 'bogus'")
    (found,) = errors(report, APPLE_FRONT)
    assert found.startswith(message)
    # So the episode's frames have no RGB of the front camera, and the episode no rgb.
    failed = report["items"]["consistency"]["steps"]["failed_by_reason"]
    reason = f"rgb.front has no video of the episode's frames: {APPLE_FRONT} {found}"
    assert failed == {reason: 3}
    assert not report["items"]["consistency"]["pass"]
    spec = report["items"]["metadata"]["spec"]
    assert spec["missing_channels"] == ["rgb"]
    assert f"{APPLE}: channels" in mismatches(report)


@needs_av1
def test_a_dataset_written_with_svt_av1_passes(tmp_path):
    root = tmp_path / "dataset"
    build_dataset(root, ffmpeg=shutil.which("ffmpeg"), episodes=EPISODES[:3])
    criteria = criteria_text(**{**SMALL, "min_frames": 8, "min_object_types": 2})
    (root / CRITERIA).write_text(criteria)
    metadata = read_meta(root)
    data = criteria.encode()
    metadata["tools"][1].update(
        sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)
    )
    write_meta(root, metadata)
    sums = (root / "SHA256SUMS").read_text().splitlines()
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(data).hexdigest()}  {CRITERIA}\n"
            if line.endswith(CRITERIA)
            else f"{line}\n"
            for line in sums
        )
    )
    code, out, err, payload = run(root, root, "--seed", SEED)
    assert code == 0, out + err
    assert payload["consistency"]["projection"]["n_pass"] == 5


# --- episodes -------------------------------------------------------------------------


def test_a_missing_video_fails_its_episode(root):
    (root / APPLE_WRIST).unlink()
    report = check(root)
    assert errors(report, APPLE) == [f"its video {APPLE_WRIST} is not on disk"]
    steps = report["items"]["consistency"]["steps"]
    reason = (
        f"rgb.wrist has no video of the episode's frames: {APPLE_WRIST} is not on disk"
    )
    assert steps["failed_by_reason"][reason] == 3
    assert report["items"]["metadata"]["spec"]["missing_channels"] == ["rgb"]


@pytest.mark.parametrize(
    ("source", "rel"),
    [
        (APPLE_TRAJECTORY, "trajectories/apple/ep-9.parquet"),
        (APPLE_FRONT, "videos/front/apple/ep-9.mp4"),
    ],
)
def test_a_file_of_no_episode_is_found(root, source, rel):
    shutil.copy(root / source, root / rel)
    kind = "trajectory" if rel.startswith("trajectories") else "video"
    list_file(root, rel, kind=kind, episode_uid="ep-9", object_type="apple")
    found = errors(check(root), rel)
    assert found[-1] == "belongs to no episode: data/apple/ep-9.parquet is not on disk"
    if kind == "trajectory":
        # Without its frames file, its own footer is held to the rules.
        assert found[0] == "holds episode 'ep-0' of 'apple', not 'ep-9' of 'apple'"


def test_one_episode_under_two_types_is_found(root):
    copy_episode(root, "ep-0", "apple", "cherry")
    report = check(root)
    assert errors(report, "data/cherry/ep-0.parquet") == [
        f"episode ep-0 is also the episode of {APPLE}"
    ]
    assert errors(report, APPLE) == []
    diversity = report["items"]["diversity"]
    assert diversity["episodes_in_several_files"] == {
        "ep-0": [APPLE, "data/cherry/ep-0.parquet"]
    }
    assert diversity["assets_under_several_types"] == {"apple_01": ["apple", "cherry"]}
    assert not diversity["pass"]


@POSIX
def test_a_video_its_episode_does_not_record_is_found(tmp_path):
    root = tmp_path / "front"
    build_dataset(
        root,
        ffmpeg=fake_av1(tmp_path / "bin"),
        episodes=EPISODES[:1],
        cameras=("front",),
    )
    rel = "videos/wrist/apple/ep-0.mp4"
    (root / rel).parent.mkdir(parents=True)
    shutil.copy(root / APPLE_FRONT, root / rel)
    list_file(root, rel, kind="video", episode_uid="ep-0", object_type="apple")
    found = errors(check(root), rel)
    assert found == ["is not a video of its episode, whose cameras are ['front']"]

    root = tmp_path / "no-rgb"
    channels = tuple(c for c in fmt.CHANNELS if c != "rgb")
    build_dataset(root, episodes=EPISODES[:1], channels=channels)
    (root / APPLE_FRONT).parent.mkdir(parents=True)
    shutil.copy(tmp_path / "front" / APPLE_FRONT, root / APPLE_FRONT)
    list_file(root, APPLE_FRONT, kind="video", episode_uid="ep-0", object_type="apple")
    assert errors(check(root), APPLE_FRONT) == [
        "is not a video of its episode, which records no rgb"
    ]


def test_paths_a_case_insensitive_disk_holds_as_one_are_found(root, tmp_path):
    probe = tmp_path / "Probe"
    probe.mkdir()
    if (tmp_path / "probe").exists():
        pytest.skip("this file system is case-insensitive")
    copy_episode(root, "ep-0", "apple", "Apple")
    report = check(root)
    assert errors(report, "data/apple/ep-0.parquet")[-1] == (
        "differs from data/Apple/ep-0.parquet only in case; a case-insensitive file "
        "system holds one of them"
    )
    assert report["items"]["diversity"]["label_collisions"] == {
        "apple": ["Apple", "apple"]
    }


# --- scale ----------------------------------------------------------------------------


def test_scale_needs_its_minimum_of_frames(root, tmp_path):
    path = criteria_file(
        tmp_path, criteria_text(**{**SMALL, "min_frames": N_FRAMES + 1})
    )
    criteria, _ = fmt.load_criteria(path)
    report = fmt.run_checks(root, criteria, items=("scale",))
    scale = report["items"]["scale"]
    assert not scale["pass"]
    assert (scale["n_frames"], scale["n_files_with_errors"]) == (N_FRAMES, 0)
    assert scale["criterion"] == {"min_frames": N_FRAMES + 1}


def test_scale_counts_only_episodes_whose_files_are_all_clean(root):
    flip(root / APPLE_WRIST)
    scale = check(root)["items"]["scale"]
    assert not scale["pass"]
    assert scale["n_frames"] == N_FRAMES - 3
    assert (scale["n_episodes"], scale["n_episodes_with_errors"]) == (3, 1)
    assert scale["n_frames_in_episodes_with_errors"] == 3
    assert scale["per_object_type"]["apple"] == {"n_episodes": 1, "n_frames": 2}
    assert scale["files_with_errors"] == [
        {
            "file": APPLE_WRIST,
            "kind": "video",
            "errors": ["not opened: its sha256 is not the one metadata.yaml states"],
        }
    ]


def test_scale_lists_fifty_files_and_the_report_the_rest(root):
    for i in range(51):
        shutil.copy(root / APPLE, root / f"data/apple/ep-{100 + i}.parquet")
    report = check(root)
    assert len(report["items"]["scale"]["files_with_errors"]) == 51
    table = fmt.format_table(report)
    assert "scale: files with errors (51, 1 more in the JSON report)" in table
    listing = table.split("scale: files with errors")[1].split("\n\n")[0]
    assert listing.count('\n  {"file": ') == 50


def test_values_json_cannot_hold_are_written_as_strict_json(root, tmp_path, capsys):
    metadata = read_meta(root)
    metadata["counts"]["n_frames"] = float("nan")
    metadata["counts"]["n_episodes"] = float("-inf")
    write_meta(root, metadata)
    target = tmp_path / "report.json"
    code, out, _, _payload = main(capsys, root, "--seed", SEED, "--json", target)
    assert code == 1

    def strict(text):
        raise ValueError(text)

    report = json.loads(target.read_text(), parse_constant=strict)
    found = report["items"]["metadata"]["mismatches"][:2]
    assert [m["metadata"] for m in found] == ["NaN", "-Infinity"]
    assert '"metadata": "NaN"' in out


def test_a_dataset_without_frames_fails_scale(root):
    shutil.rmtree(root / "data")
    report = check(root)
    assert report["items"]["scale"]["n_frames"] == 0
    assert not report["items"]["scale"]["pass"]
    assert not report["items"]["metadata"]["pass"]


# --- metadata -------------------------------------------------------------------------


def entry(path):
    return lambda m: next(e for e in m["files"] if e["path"] == path)


METADATA = [
    (lambda m: m.update(format="physgraspgen"), ["format"]),
    (
        lambda m: m.update(format_version=2.0),
        ["format_version"]
        + [
            f"{f}: format_version"
            for f in sorted(f"data/{t}/{u}.parquet" for t, u, _ in EPISODES)
        ],
    ),
    (
        lambda m: m["counts"].update(n_frames=N_FRAMES + 1),
        ["counts.n_frames"],
    ),
    (lambda m: m["counts"].update(n_frames=float(N_FRAMES)), ["counts.n_frames"]),
    (lambda m: m["counts"].update(n_episodes=True), ["counts.n_episodes"]),
    (lambda m: m["counts"].update(n_files=N_DATA_FILES + 2), ["counts.n_files"]),
    (lambda m: m["counts"].update(n_object_types=4), ["counts.n_object_types"]),
    (lambda m: m["counts"].update(n_object_assets=2), ["counts.n_object_assets"]),
    (lambda m: m["counts"].update(n_channels=5), ["counts.n_channels"]),
    (
        lambda m: m.update(counts=None),
        [
            "counts.n_frames",
            "counts.n_episodes",
            "counts.n_files",
            "counts.n_object_types",
            "counts.n_object_assets",
            "counts.n_channels",
        ],
    ),
    (
        lambda m: m.update(channels=m["channels"][::-1]),
        ["channels"]
        + [
            f"{f}: channels"
            for f in sorted(f"data/{t}/{u}.parquet" for t, u, _ in EPISODES)
        ],
    ),
    (lambda m: m["conventions"].update(units="imperial"), ["conventions"]),
    (
        lambda m: m["object_types"]["apple"].update(n_frames=6),
        ["object_types.apple.n_frames"],
    ),
    (
        lambda m: m["object_types"]["apple"].update(assets=["apple_02"]),
        ["object_types.apple.assets"],
    ),
    (
        lambda m: m["object_types"].update(
            durian={"n_episodes": 1, "n_frames": 3, "assets": ["durian_01"]}
        ),
        [
            "object_types.durian.n_episodes",
            "object_types.durian.n_frames",
            "object_types.durian.assets",
        ],
    ),
    (
        lambda m: m["object_types"].update({1: {"n_episodes": 1}}),
        ["object_types: 1 is not a type name"],
    ),
    (
        lambda m: m.update(name="other"),
        [
            f"{f}: dataset_name"
            for f in sorted(f"data/{t}/{u}.parquet" for t, u, _ in EPISODES)
        ],
    ),
    (
        lambda m: m["video"].update(ffmpeg_versions=["ffmpeg 9"]),
        ["video.ffmpeg_versions"],
    ),
    (lambda m: m["provenance"].update(generation=[{"git": "abc"}]), ["provenance"]),
    (lambda m: entry(APPLE)(m).update(kind="video"), [f"{APPLE}: kind"]),
    (lambda m: entry(APPLE)(m).update(bytes=1), [f"{APPLE}: bytes"]),
    (lambda m: entry(APPLE)(m).update(n_frames=4), [f"{APPLE}: n_frames"]),
    (
        lambda m: entry(APPLE_TRAJECTORY)(m).update(n_steps=7),
        [f"{APPLE_TRAJECTORY}: n_steps"],
    ),
    (lambda m: entry(APPLE_FRONT)(m).update(n_frames=2), [f"{APPLE_FRONT}: n_frames"]),
    (lambda m: entry(APPLE)(m).update(object_type="pear"), [f"{APPLE}: object_type"]),
    (lambda m: entry(APPLE)(m).update(episode_uid="ep-9"), [f"{APPLE}: episode_uid"]),
    (
        lambda m: m["files"].append(
            dict(entry(APPLE)(m), path="data/apple/ep-9.parquet")
        ),
        [
            "files: data/apple/ep-9.parquet is listed but not on disk",
            "SHA256SUMS: paths",
        ],
    ),
    (
        lambda m: m.update(tools=m["tools"][:1]),
        [f"tools: {CRITERIA} is not listed", "SHA256SUMS: paths"],
    ),
    (
        lambda m: m["tools"].append(dict(m["tools"][0], path="tools/extra.py")),
        ["tools: tools/extra.py is not a tool of the format", "SHA256SUMS: paths"],
    ),
    (lambda m: m["tools"][1].update(bytes=0), [f"{CRITERIA}: bytes"]),
]


@pytest.mark.parametrize(("change", "found"), METADATA)
def test_each_disagreement_with_metadata_yaml_is_one_mismatch(root, change, found):
    metadata = read_meta(root)
    change(metadata)
    write_meta(root, metadata)
    report = check(root)
    assert mismatches(report) == found
    assert not report["items"]["metadata"]["pass"]


def test_mismatches_come_in_one_order(root):
    metadata = read_meta(root)
    metadata["format"] = "x"
    metadata["counts"]["n_files"] = 0
    entry(APPLE)(metadata).update(bytes=0)
    metadata["files"].append(
        dict(entry(APPLE)(metadata), path="data/apple/ep-9.parquet")
    )
    metadata["files"] = [e for e in metadata["files"] if e["path"] != APPLE_FRONT]
    metadata["files"].append({"path": "/etc/hosts"})
    metadata["tools"] = metadata["tools"][:1]
    write_meta(root, metadata)
    # The values, the list's problems, its entries in path order, the files it does
    # not list, the tools, then SHA256SUMS. The unlisted video is not opened, so its
    # episode lacks rgb.
    assert mismatches(check(root)) == [
        "format",
        "counts.n_files",
        f"{APPLE}: channels",
        f"files[{N_DATA_FILES}] path '/etc/hosts' is not a path the format defines",
        f"{APPLE}: bytes",
        "files: data/apple/ep-9.parquet is listed but not on disk",
        f"files: {APPLE_FRONT} is on disk but not listed",
        f"tools: {CRITERIA} is not listed",
        "SHA256SUMS: paths",
    ]


@pytest.mark.parametrize(("n", "passes"), [(1, True), (2, False)])
def test_metadata_allows_its_maximum_of_mismatches(root, tmp_path, n, passes):
    metadata = read_meta(root)
    metadata["counts"]["n_frames"] += 1
    if n == 2:
        metadata["counts"]["n_episodes"] += 1
    write_meta(root, metadata)
    path = criteria_file(tmp_path, criteria_text(**SMALL, max_mismatches=1))
    criteria, _ = fmt.load_criteria(path)
    result = fmt.run_checks(root, criteria, items=("metadata",))["items"]["metadata"]
    assert (result["n_mismatches"], result["pass"]) == (n, passes)


def test_sha256sums_is_compared_with_metadata_yaml(root):
    sums = (root / "SHA256SUMS").read_text().splitlines()
    first = sums[0].split("  ")[1]
    lines = [
        sums[0].upper().replace(first.upper(), first),
        sums[1].replace("  ", " *"),
        "not a line",
        "",
        "0" * 64 + "  " + first,
        *sums[2:-1],
    ]
    (root / "SHA256SUMS").write_bytes("\r\n".join(lines).encode())
    found = mismatches(check(root))
    assert found == [
        "SHA256SUMS line 3 is not '<sha256>  <path>'",
        f"SHA256SUMS: {first} is listed twice",
        "SHA256SUMS: paths",
        f"SHA256SUMS: {first}",
    ]
    (root / "SHA256SUMS").unlink()
    assert mismatches(check(root)) == ["SHA256SUMS is missing"]


def test_a_tool_listed_but_missing_is_a_mismatch(root):
    criteria, _ = fmt.load_criteria(root / CRITERIA)
    (root / CRITERIA).unlink()
    report = fmt.run_checks(root, criteria, seed=SEED)
    assert f"tools: {CRITERIA} is listed but missing" in mismatches(report)
    assert not report["published_tools"]["pass"]


@pytest.mark.parametrize("value", [["x"], {"a": 1}])
@pytest.mark.parametrize("key", ["object_asset", "ffmpeg_version"])
def test_a_footer_value_that_is_no_string_fails_its_file_not_the_run(
    root, capsys, key, value
):
    # banana's one episode: no other file names its asset.
    set_footer(root, lambda f: {**f, key: value}, "banana", "ep-2")
    code, out, _, payload = main(capsys, root, "--seed", SEED)
    assert code == 1
    assert payload["scale"]["n_files_with_errors"] == 1
    assert '{"file": "data/banana/ep-2.parquet", "kind": "frames"' in out
    # The other episodes were encoded by the same ffmpeg build.
    assert mismatches(check(root)) == (
        ["counts.n_object_assets", "object_types.banana.assets"]
        if key == "object_asset"
        else []
    )


def test_mismatches_of_an_unread_file_hold_no_none(root):
    flip(root / APPLE)
    report = check(root)
    counted = report["items"]["metadata"]["counted"]
    # Its frames still count (its row count is unread), its type and id by its path.
    assert counted["n_frames"] == N_FRAMES - 3
    assert counted["n_episodes"] == N_EPISODES
    assert counted["n_object_types"] == N_TYPES
    assert counted["n_object_assets"] == N_TYPES
    assert "null" not in json.dumps(counted)


@pytest.mark.parametrize(
    ("doi", "ok"),
    [
        ("10.5281/zenodo.1234567", True),
        ("10.1000.10/abc", False),
        ("10.123/x", False),
        ("10.1234567890/x", False),
        ("doi:10.5281/x", False),
        ("https://doi.org/10.5281/x", False),
        ("10.5281/zenodo.1\n", False),
        ("10.5281/zenodo 1", False),
        ("10.١٢٣٤/x", False),
        (None, False),
    ],
)
def test_the_doi_is_a_bare_ascii_doi(root, doi, ok):
    metadata = read_meta(root)
    metadata["doi"] = doi
    write_meta(root, metadata)
    spec = check(root, items=("metadata",))["items"]["metadata"]["spec"]
    assert spec["checks"]["doi"] is ok
    assert spec["pass"] is ok


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        (URL, True),
        ("https://", False),
        ("http://example.org/dataset", False),
        ("HTTPS://example.org", False),
        ("https://example.org/a b", False),
        ("https://éxample.org", False),
        (None, False),
    ],
)
def test_the_download_url_is_an_https_url(root, url, ok):
    metadata = read_meta(root)
    metadata["download_url"] = url
    write_meta(root, metadata)
    spec = check(root, items=("metadata",))["items"]["metadata"]["spec"]
    assert spec["checks"]["download_url"] is ok


def test_the_specification_needs_every_channel_camera_and_the_licence(root, tmp_path):
    path = criteria_file(tmp_path, criteria_text(**SMALL, license="CC0-1.0"))
    criteria, _ = fmt.load_criteria(path)
    spec = fmt.run_checks(root, criteria, items=("metadata",))["items"]["metadata"]
    assert spec["spec"]["checks"]["license"] is False
    assert "failed: license" in fmt.format_table(
        {
            **fmt.run_checks(root, criteria, items=("metadata",)),
            "items": {"metadata": spec},
        }
    )


@POSIX
def test_the_specification_needs_every_required_camera(tmp_path):
    root = tmp_path / "dataset"
    build_dataset(root, ffmpeg=fake_av1(tmp_path / "bin"), cameras=("front",))
    metadata = check(root, items=("metadata",))["items"]["metadata"]
    assert metadata["n_mismatches"] == 0
    spec = metadata["spec"]
    assert spec["cameras_in_every_episode"] == ["front"]
    assert spec["missing_cameras"] == ["wrist"]
    assert [k for k, ok in spec["checks"].items() if not ok] == ["cameras"]
    assert not metadata["pass"]


def test_a_dataset_without_masks_lacks_a_channel_of_the_specification(tmp_path):
    root = tmp_path / "dataset"
    channels = ("rgb", "depth", "force", "stress", "deformation_gradient")
    build_dataset(root, ffmpeg=fake_av1(tmp_path / "bin"), channels=channels)
    report = check(root)
    spec = report["items"]["metadata"]["spec"]
    assert spec["missing_channels"] == ["segmentation"]
    assert not spec["pass"]
    assert report["items"]["metadata"]["n_mismatches"] == 0
    # Without masks the projection has nothing to land on.
    unavailable = report["items"]["consistency"]["projection"]["unavailable_files"]
    assert unavailable[APPLE][:2] == [
        "segmentation.front.mask",
        "segmentation.front.sim_step",
    ]
    assert "  spec (in metadata)  5 of 6 GT channels (missing segmentation)" in (
        fmt.format_table(report)
    )


# --- the step part --------------------------------------------------------------------


def test_a_layer_recorded_at_another_step_fails_its_frame(root):
    rewrite(root, APPLE, table=cell("depth.wrist.sim_step", 1, 2))
    steps = check(root)["items"]["consistency"]["steps"]
    assert not steps["pass"]
    assert (steps["n_frames"], steps["n_pass"]) == (N_FRAMES, N_FRAMES - 1)
    assert steps["failed_frames"] == [
        {
            "file": APPLE,
            "frame": 1,
            "reasons": ["depth.wrist was recorded at another step"],
        }
    ]


@pytest.mark.parametrize(("fraction", "passes"), [(0.9, True), (0.95, False)])
def test_each_consistency_part_needs_its_fraction_of_frames(
    root, tmp_path, fraction, passes
):
    # Frame 1 has a depth image of another step and frame 0 a mask of another step, so
    # nine of ten frames pass each part; masks are not a published layer.
    rewrite(root, APPLE, table=cell("depth.wrist.sim_step", 1, 2))
    rewrite(root, APPLE, table=cell("segmentation.wrist.sim_step", 0, 4))
    values = {**SMALL, "sample_frames": N_FRAMES, "min_pass_fraction": fraction}
    criteria, _ = fmt.load_criteria(criteria_file(tmp_path, criteria_text(**values)))
    result = fmt.run_checks(root, criteria, items=("consistency",), seed=SEED)
    steps, proj = result["items"]["consistency"]["steps"], projection(result)
    assert (steps["n_pass"], steps["n_frames"]) == (N_FRAMES - 1, N_FRAMES)
    assert proj["statuses"] == {"mask_step": 1, "pass": N_FRAMES - 1}
    assert (steps["pass"], proj["pass"]) == (passes, passes)


def test_a_layer_without_its_own_steps_fails_every_frame(root):
    rewrite(root, APPLE, table=column("force.sim_step", pa.float64()))
    rewrite(
        root,
        "data/banana/ep-2.parquet",
        table=drop("stress.sim_step"),
    )
    steps = check(root)["items"]["consistency"]["steps"]
    assert steps["failed_by_reason"] == {
        "force has no step for each frame": 3,
        "stress has no step for each frame": 3,
    }


def test_a_frame_out_of_order_fails_the_step_part(root):
    rewrite(root, APPLE, table=cell("sim_step", 0, 0))
    steps = check(root)["items"]["consistency"]["steps"]
    reasons = steps["failed_frames"][0]["reasons"]
    assert reasons[0] == "its step is below 1 or not later than the previous frame's"
    assert "rgb.front was recorded at another step" in reasons


def test_a_step_column_that_is_not_one_fails_every_frame(root):
    rewrite(root, APPLE, table=column("sim_step", pa.int32()))
    steps = check(root)["items"]["consistency"]["steps"]
    assert steps["failed_by_reason"] == {"sim_step has no step for each frame": 3}


def test_the_video_frame_index_is_compared(root):
    rewrite(root, APPLE, table=cell("rgb.front.video_frame", 2, 1))
    steps = check(root)["items"]["consistency"]["steps"]
    assert steps["failed_frames"] == [
        {
            "file": APPLE,
            "frame": 2,
            "reasons": ["rgb.front is not the frame's video frame"],
        }
    ]


def test_a_file_that_drops_a_camera_fails_its_frames(root):
    metadata = read_meta(root)
    metadata["cameras"] = ["front"]
    write_meta(root, metadata)
    # The criteria's cameras are the floor: metadata.yaml cannot narrow the check.
    consistency = check(root)["items"]["consistency"]
    assert consistency["cameras"] == ["front", "wrist"]
    assert consistency["steps"]["pass"]


def test_layers_can_be_any_channels(root, tmp_path):
    layers = "[segmentation, deformation_gradient]"
    path = criteria_file(tmp_path, criteria_text(**SMALL, layers=layers))
    criteria, _ = fmt.load_criteria(path)
    rewrite(root, APPLE, table=cell("segmentation.front.sim_step", 0, 4))
    steps = fmt.run_checks(root, criteria, items=("consistency",), seed=SEED)
    steps = steps["items"]["consistency"]["steps"]
    assert steps["layers"] == ["segmentation", "deformation_gradient"]
    assert steps["failed_by_reason"] == {
        "segmentation.front was recorded at another step": 1
    }


@pytest.mark.parametrize(
    ("stated", "counted"),
    [
        (4, 4),
        (-3, 0),
        ("4", 0),
        (None, 0),
        (True, 0),
        pytest.param(10**400, 10**400, id="beyond-float"),
    ],
)
def test_a_file_without_a_frame_count_keeps_its_stated_frames(
    root, capsys, stated, counted
):
    flip(root / APPLE)
    metadata = read_meta(root)
    entry(APPLE)(metadata)["n_frames"] = stated
    write_meta(root, metadata)
    steps = check(root)["items"]["consistency"]["steps"]
    n_frames = N_FRAMES - 3 + counted
    assert steps["n_frames"] == n_frames
    assert steps["n_pass"] == N_FRAMES - 3
    assert steps["files_without_frame_count"][0]["frames_stated"] == counted
    assert steps["failed_by_reason"]["files without a frame count, or missing"] == 1
    assert not steps["pass"]
    # A count beyond the float range is reported too.
    code, out, _, payload = main(capsys, root, "--seed", SEED)
    assert code == 1
    assert payload["consistency"]["steps"]["n_frames"] == n_frames
    pct = 100 * (N_FRAMES - 3) / n_frames
    assert f"steps {N_FRAMES - 3:,}/{n_frames:,} frames ({pct:.2f} %)" in out


def test_a_listed_frames_file_missing_from_disk_keeps_its_frames(root):
    (root / APPLE).unlink()
    steps = check(root)["items"]["consistency"]["steps"]
    assert steps["files_without_frame_count"] == [
        {
            "file": APPLE,
            "frames_stated": 3,
            "reasons": ["metadata.yaml lists it, but it is not on disk"],
        }
    ]
    assert steps["n_frames"] == N_FRAMES


def test_the_sample_key_is_fixed_by_its_inputs():
    def key(seed, purpose, rel, i):
        return hashlib.sha256(f"{seed}\0{purpose}\0{rel}\0{i}".encode()).digest()[:8]

    assert key(7, "steps", "data/apple/ep-0.parquet", 3).hex() == "ff0e417863d98223"
    assert (
        key(7, "projection", "data/apple/ep-0.parquet", 3).hex() == "774de6267adf4e2c"
    )
    assert key(-5, "steps", "data/banana/ep-1.parquet", 0).hex() == "a2f47dd87324d2b7"
    frames = [("data/apple/ep-0.parquet", i) for i in range(5)] + [
        ("data/banana/ep-1.parquet", i) for i in range(3)
    ]
    assert fmt.sample_frames(frames, 3, 7, "steps") == [
        ("data/apple/ep-0.parquet", 4),
        ("data/banana/ep-1.parquet", 0),
        ("data/banana/ep-1.parquet", 1),
    ]
    assert fmt.sample_frames(reversed(frames), 3, 7, "steps") == fmt.sample_frames(
        frames, 3, 7, "steps"
    )
    assert fmt.sample_frames(frames, 100, 7, "steps") == sorted(frames)


def test_one_seed_draws_the_same_frames_however_the_files_are_read(root):
    one = check(root, workers=1)["items"]["consistency"]
    many = check(root, workers=8)["items"]["consistency"]
    assert one["steps"]["sample"] == many["steps"]["sample"]
    assert one["projection"] == many["projection"]
    other = check(root, seed=8)["items"]["consistency"]["projection"]
    assert other["seed"] == 8


# --- the projection -------------------------------------------------------------------


@pytest.fixture(scope="module")
def scenes(tmp_path_factory):
    """Build a dataset for a frame builder once, the scene's fake ffmpeg shared."""
    if sys.platform == "win32":
        pytest.skip("the fake ffmpeg is a shell script")
    tmp = tmp_path_factory.mktemp("scenes")
    ffmpeg = fake_av1(tmp / "bin")
    built = {}

    def build(name, frame, **kwargs):
        if name not in built:
            build_dataset(tmp / name, ffmpeg=ffmpeg, frame=frame, **kwargs)
            built[name] = tmp / name
        return built[name]

    return build


def projection(report):
    return report["items"]["consistency"]["projection"]


def moved(offset):
    return lambda step, spec, n=None: scene_frame(step, spec, offset=offset)


def test_points_beside_the_object_fail_the_projection(scenes):
    # 0.5 m at 1 m is 10 px in the front camera: 3.5 and 5.5 px past the object.
    root = scenes("off", moved(0.5))
    report = check(root)
    proj = projection(report)
    assert proj["statuses"] == {"off_object": 5}
    assert proj["failed_frames"][0]["reason"] == (
        "pressed contact points lie more than 4 px from the object"
    )
    front = proj["per_camera"]["front"]
    assert front["landing_classes"] == {"background": 10}
    # From (22.5, 7.9) and (20.5, 7.9) to the centre of pixel (8, 17).
    assert front["distance_px"]["max"] == pytest.approx(np.hypot(5.5, 0.1))
    assert front["distance_px"]["median"] == pytest.approx(
        (np.hypot(5.5, 0.1) + np.hypot(3.5, 0.1)) / 2
    )
    assert not proj["pass"] and not report["items"]["consistency"]["pass"]
    assert report["items"]["consistency"]["steps"]["pass"]


@pytest.mark.parametrize(
    ("values", "status"),
    [
        # From the front camera the points lie 3.5 and 5.5 px from the object, from
        # the wrist camera 2.8 and 4.6 px.
        ({"projection_radius_px": 5.0}, "off_object"),
        ({"projection_radius_px": 6.0}, "pass"),
        ({"projection_radius_px": 6.0, "distance_reach_px": 2.0}, "pass"),
        ({"min_point_fraction": 0.5}, "pass"),
        ({"min_point_fraction": 0.6}, "off_object"),
    ],
)
def test_the_projection_applies_the_tolerance_it_is_given(
    scenes, tmp_path, values, status
):
    root = scenes("off", moved(0.5))
    criteria, _ = fmt.load_criteria(
        criteria_file(tmp_path, criteria_text(**{**SMALL, **values}))
    )
    report = fmt.run_checks(root, criteria, items=("consistency",), seed=SEED)
    assert projection(report)["statuses"] == {status: 5}
    # Distances are measured out to the radius at least.
    reach = max(
        values.get("distance_reach_px", 16.0), values.get("projection_radius_px", 4.0)
    )
    assert report["items"]["consistency"]["criterion"]["distance_reach_px"] == reach


def test_a_point_at_the_radius_is_on_the_object(scenes, tmp_path):
    root = scenes("off", moved(0.5))
    farthest = projection(check(root))["per_camera"]["front"]["distance_px"]["max"]
    for radius, status in (
        (farthest, "pass"),
        (float(np.nextafter(farthest, 0.0)), "off_object"),
    ):
        text = criteria_text(**SMALL, projection_radius_px=repr(radius))
        criteria, _ = fmt.load_criteria(criteria_file(tmp_path, text))
        assert criteria["consistency"]["projection_radius_px"] == radius
        report = fmt.run_checks(root, criteria, items=("consistency",), seed=SEED)
        assert projection(report)["statuses"] == {status: 5}


def test_points_behind_the_cameras_are_not_in_view(scenes):
    def behind(step, spec, n=None):
        frame = scene_frame(step, spec)
        frame["force.contact_point_world"] = frame["force.contact_point_world"] * [
            1,
            1,
            -1,
        ]
        return frame

    proj = projection(check(scenes("behind", behind)))
    assert proj["statuses"] == {"not_in_view": 5}
    assert proj["n_points_in_view"] == 0
    assert proj["cameras_without_points"] == [
        "front sees no pressed contact point in the 5 sampled frames",
        "wrist sees no pressed contact point in the 5 sampled frames",
    ]


def test_a_camera_that_sees_no_point_fails_the_projection(scenes):
    def wrist_away(step, spec, n=None):
        frame = scene_frame(step, spec)
        frame["camera.wrist.world_to_camera"] = np.eye(4)
        frame["camera.wrist.world_to_camera"][0, 3] = 5.0
        return frame

    proj = projection(check(scenes("wrist-away", wrist_away)))
    # Every frame passes on the front camera alone, which is not each camera.
    assert proj["statuses"] == {"pass": 5}
    assert proj["cameras_without_points"] == [
        "wrist sees no pressed contact point in the 5 sampled frames"
    ]
    assert not proj["pass"]


def test_two_cameras_that_disagree_fail_the_frame(scenes):
    def wrist_off(step, spec, n=None):
        frame = scene_frame(step, spec)
        frame["camera.wrist.world_to_camera"] = np.eye(4)
        frame["camera.wrist.world_to_camera"][0, 3] = 0.6
        return frame

    proj = projection(check(scenes("wrist-off", wrist_off)))
    assert proj["statuses"] == {"off_object": 5}
    cameras = proj["failed_frames"][0]["cameras"]
    assert cameras["front"]["n_on_object"] == cameras["front"]["n_in_view"] == 2
    assert cameras["wrist"]["n_on_object"] == 0


def test_contacts_that_press_nothing_are_no_candidates(scenes, tmp_path):
    def light(step, spec, n=None):
        return scene_frame(step, spec, pressed=step > 4)

    root = scenes("light", light)
    proj = projection(check(root))
    # Only the frames after step 4 press: 1 of apple's ep-0, 1 of banana's ep-2.
    assert proj["n_candidates"] == 2
    assert not proj["pass"]
    criteria = criteria_text(**{**SMALL, "min_contact_frames": 2})
    path = criteria_file(tmp_path, criteria)
    values, _digest = fmt.load_criteria(path)
    proj = projection(fmt.run_checks(root, values, items=("consistency",), seed=SEED))
    assert (proj["n_candidates"], proj["n_sampled"], proj["n_pass"]) == (2, 2, 2)


def test_contacts_without_force_or_off_the_robot_press_nothing(root):
    def no_force(contacts):
        contacts = [dict(c) for c in contacts]
        for c in contacts:
            c["force"] = [0.0, 0.0, 0.0]
        contacts[0]["is_robot"] = False
        return contacts

    for row in range(3):
        rewrite(root, APPLE, table=cell("force.contacts", row, no_force))
    assert projection(check(root))["n_candidates"] == N_FRAMES - 3
    # A force that is not a number is not 0: that contact presses.
    rewrite(
        root, APPLE, table=cell("force.contacts", 1, contact(1, "force", [nan, 0, 0]))
    )
    report = check(root)
    assert projection(report)["n_candidates"] == N_FRAMES - 2
    assert "frame 1: force.contacts.force must be finite" in errors(report, APPLE)


def test_the_masks_step_and_the_camera_matrices_are_checked_per_frame(root):
    rewrite(root, APPLE, table=cell("segmentation.wrist.sim_step", 0, 4))
    rewrite(root, APPLE, table=cell("camera.front.intrinsics", 1, at(1, 0.5)))
    rewrite(
        root,
        "data/apple/ep-1.parquet",
        table=cell("camera.wrist.world_to_camera", 0, at(15, 2.0)),
    )
    criteria = criteria_text(**{**SMALL, "sample_frames": N_FRAMES})
    path = criteria_file(root.parent, criteria)
    values, _ = fmt.load_criteria(path)
    proj = projection(fmt.run_checks(root, values, items=("consistency",), seed=SEED))
    reasons = {(f["file"], f["frame"]): f["reason"] for f in proj["failed_frames"]}
    assert reasons == {
        (APPLE, 0): "segmentation.wrist is of step 4, not the frame's 2",
        (APPLE, 1): "front: camera.front.intrinsics is not a pinhole K: K[0][1] (skew) "
        "and K[1][0] must be 0",
        ("data/apple/ep-1.parquet", 0): "wrist: camera.wrist.world_to_camera is not a "
        "rigid transform: its last row must be 0 0 0 1",
    }
    assert proj["statuses"] == {"camera_matrices": 2, "mask_step": 1, "pass": 7}


def test_classes_that_do_not_name_the_object_make_a_file_unavailable(root):
    set_footer(root, lambda f: {**f, "segmentation_classes": ["background", "fruit"]})
    proj = projection(check(root))
    assert proj["unavailable_files"][APPLE] == ["segmentation_classes naming 'object'"]
    assert not proj["pass"]


def test_the_object_is_found_by_its_class_name(root):
    # The classes put the object at value 2; its masks say so too.
    classes = ["background", "arm", "object", "gripper", "support", "prop"]
    set_footer(root, lambda f: {**f, "segmentation_classes": classes})
    swap = {0: 0, 1: 2, 3: 3}
    for camera in fmt.CAMERAS:
        name = f"segmentation.{camera}.mask"
        for row in range(3):
            rewrite(root, APPLE, table=cell(name, row, lambda m: [swap[v] for v in m]))
    report = check(root, sample=N_FRAMES)
    assert (
        "metadata segmentation_classes is ['background', 'arm', 'object'"
        in (errors(report, APPLE)[0])
    )
    proj = projection(report)
    assert proj["statuses"] == {"pass": N_FRAMES}
    assert proj["per_camera"]["front"]["landing_classes"] == {"object": 2 * N_FRAMES}


@pytest.mark.parametrize(("sample", "passes"), [(4, False), (10, True), (50, True)])
def test_a_sample_below_the_criteria_passes_only_with_every_candidate(
    root, sample, passes
):
    proj = projection(check(root, sample=sample))
    assert proj["sample_override"] == sample
    assert proj["n_sampled"] == min(sample, N_FRAMES)
    assert proj["pass"] is passes


def test_the_projection_needs_its_minimum_of_candidates(root, tmp_path):
    path = criteria_file(tmp_path, criteria_text(**{**SMALL, "min_contact_frames": 11}))
    criteria, _ = fmt.load_criteria(path)
    proj = projection(fmt.run_checks(root, criteria, items=("consistency",), seed=SEED))
    assert (proj["n_candidates"], proj["n_pass"]) == (N_FRAMES, 5)
    assert not proj["pass"]
    # The minimum counts the candidates, not the sample drawn from them.
    values = {**SMALL, "sample_frames": 5, "min_contact_frames": 8}
    criteria, _ = fmt.load_criteria(criteria_file(tmp_path, criteria_text(**values)))
    proj = projection(fmt.run_checks(root, criteria, items=("consistency",), seed=SEED))
    assert (proj["n_candidates"], proj["n_sampled"]) == (N_FRAMES, 5)
    assert proj["pass"]


def test_a_depth_column_of_another_shape_leaves_out_the_residual(root):
    def short(table):
        i = table.schema.get_field_index("depth.front.depth_m")
        values = [v[:-1] for v in table.column(i).to_pylist()]
        kind = pa.list_(pa.float32(), 16 * 24 - 1)
        return table.set_column(
            i,
            pa.field("depth.front.depth_m", kind, nullable=False),
            pa.array(values, kind),
        )

    rewrite(root, APPLE, table=short)
    report = check(root)
    proj = projection(report)
    assert proj["statuses"] == {"pass": 5}
    assert "depth.front.depth_m" in " ".join(errors(report, APPLE))


def test_a_frames_file_whose_read_fails_is_unavailable(root, monkeypatch):
    pressed = fmt.pressed_frames

    def fail_for_apple(contacts):
        if len(contacts) == 3 and fail_for_apple.calls == 0:
            fail_for_apple.calls += 1
            raise OSError("a read failed")
        return pressed(contacts)

    fail_for_apple.calls = 0
    monkeypatch.setattr(fmt, "pressed_frames", fail_for_apple)
    proj = projection(check(root))
    assert list(proj["unavailable_files"].values()) == [
        ["cannot be read: OSError: a read failed"]
    ]
    assert not proj["pass"]


# --- diversity ------------------------------------------------------------------------


def typed(rel, footer, n_rows=3):
    object_type, name = rel.split("/")[1:]
    rec = fmt.data_file(rel)
    rec.opened, rec.n_rows = True, n_rows
    rec.footer = {
        "object_type": object_type,
        "episode_uid": name.removesuffix(".parquet"),
        "object_asset": f"{object_type}_01",
        "label": "success",
        "facts": {k: {"x": 1} for k in fmt.FACT_KINDS},
        **footer,
    }
    return rec


def diversity(records, **criteria):
    values = {
        "min_object_types": 1,
        "required_fact_kinds": list(fmt.FACT_KINDS),
        "bins": 4,
        "log_bins_min_ratio": 100.0,
        "max_categories": 2,
        **criteria,
    }
    ds = fmt.Dataset(Path("."), records, {}, {}, {}, [], {}, [], True)
    return fmt.check_diversity(ds, values)


def test_diversity_counts_object_types_against_its_minimum():
    records = [typed(f"data/t{i}/ep-{i}.parquet", {}) for i in range(10)]
    assert diversity(records, min_object_types=10)["pass"]
    assert not diversity(records[:9], min_object_types=10)["pass"]
    assert diversity(records)["object_types"]["t0"] == {
        "n_episodes": 1,
        "n_frames": 3,
        "frame_fraction": 0.1,
    }


def test_diversity_finds_types_assets_and_episodes_that_are_one():
    result = diversity(
        [
            typed("data/Apple/ep-0.parquet", {}),
            typed("data/apple/ep-1.parquet", {"object_asset": "shared"}),
            typed("data/pear/ep-2.parquet", {"object_asset": "shared"}),
            typed("data/pear/ep-0.parquet", {}),
            typed("data/plum/ep-4.parquet", {"object_type": "Plum"}),
            typed("data/fig/ep-5.parquet", {"episode_uid": 5}),
        ]
    )
    assert result["label_collisions"] == {"apple": ["Apple", "apple"]}
    assert result["assets_under_several_types"] == {"shared": ["apple", "pear"]}
    assert result["episodes_in_several_files"] == {
        "ep-0": ["data/Apple/ep-0.parquet", "data/pear/ep-0.parquet"]
    }
    assert result["files_without_type"] == ["data/plum/ep-4.parquet"]
    assert result["n_object_types"] == 4
    assert not result["pass"]


@pytest.mark.parametrize(
    ("facts", "missing"),
    [
        (
            {"placement": {"x": 1}, "lighting": {"x": 1}, "camera": {"x": 1}},
            ["material"],
        ),
        ({"material": "steel"}, ["material"]),
        ({"material": {}}, ["material"]),
        ({"material": {"E": None, "nu": None}}, ["material"]),
        ({"material": {"yaw": {}}}, ["material"]),
        ({"material": {"yaw": {"deg": None}}}, ["material"]),
    ],
)
def test_diversity_needs_every_required_kind_of_facts(facts, missing):
    full = {k: {"x": 1} for k in fmt.FACT_KINDS if k != "material"}
    result = diversity([typed("data/apple/ep-0.parquet", {"facts": {**full, **facts}})])
    assert result["files_missing_facts"] == [
        {"file": "data/apple/ep-0.parquet", "missing": missing}
    ]
    assert not result["pass"]
    assert diversity(
        [typed("data/apple/ep-0.parquet", {"facts": {**full, **facts}})],
        required_fact_kinds=["camera"],
    )["pass"]


def test_diversity_tabulates_every_variable_of_the_facts():
    values = [1.0, 1.0, 1.0]
    records = [
        typed(
            f"data/t{i}/ep-{i}.parquet",
            {
                "facts": {
                    "material": {"E": [1.0, 99.0, 50.5][i], "same": values[i]},
                    "placement": {
                        "code": ["1", 1, None][i],
                        "on": [True, False, True][i],
                    },
                    "lighting": {"lux": [1.0, 10.0, 200.0][i]},
                    "camera": {"rig": ["a", "b", "c"][i]},
                }
            },
            n_rows=i + 1,
        )
        for i in range(3)
    ]
    tables = diversity(records)["tables"]
    assert tables["material.same"]["rows"] == [
        {"bin": "[1, 1]", "n_episodes": 3, "n_frames": 6, "frame_fraction": 1.0}
    ]
    assert [r["n_episodes"] for r in tables["material.E"]["rows"]] == [1, 0, 1, 1]
    assert tables["material.E"]["rows"][0]["bin"] == "[1, 25.5]"
    # 200 / 1 is at least 100: log-spaced bins.
    assert tables["lighting.lux"]["rows"][0]["bin"] == "[1, 3.7606]"
    # "1" meets 1, as their JSON texts are one; ties fall to the text.
    code = tables["placement.code"]
    assert code["kind"] == "categorical"
    assert [(r["value"], r["n_episodes"]) for r in code["rows"]] == [
        ("1", 2),
        ("null", 1),
    ]
    rig = tables["camera.rig"]["rows"]
    assert [r["value"] for r in rig] == ["c", "b", "other (1 values)"]
    assert tables["placement.on"]["kind"] == "categorical"
    assert diversity(records)["labels"]["success"]["n_episodes"] == 3


def test_diversity_reports_files_not_read(root):
    flip(root / APPLE)
    result = check(root)["items"]["diversity"]
    assert result["files_not_read"] == [APPLE]
    assert not result["pass"]


def test_diversity_walks_facts_of_any_depth():
    # Footers nest at most 32 levels, and nothing walks one by recursion either.
    facts = {k: {"x": 1} for k in fmt.FACT_KINDS}
    facts["material"] = nested(5000)
    result = diversity([typed("data/apple/ep-0.parquet", {"facts": facts})])
    assert result["pass"]
    assert ".".join(["material"] + ["a"] * 5000) in result["tables"]
    assert fmt.holds_a_value(nested(5000))
    assert not fmt.holds_a_value(nested(5000, None))


# --- the published tools --------------------------------------------------------------


def republish(root, rel, data):
    """Put other bytes in a tool file and list them in metadata.yaml and SHA256SUMS."""
    (root / rel).write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    metadata = read_meta(root)
    for tool in metadata["tools"]:
        if tool["path"] == rel:
            tool.update(sha256=digest, bytes=len(data))
    write_meta(root, metadata)
    lines = (root / "SHA256SUMS").read_text().splitlines()
    (root / "SHA256SUMS").write_text(
        "".join(f"{digest}  {rel}\n" if l.endswith(rel) else f"{l}\n" for l in lines)
    )


def test_a_run_passes_only_with_the_published_script(root):
    republish(root, "tools/check_dataset.py", SCRIPT.read_bytes() + b"# edited\n")
    report = check(root)
    assert all(report["items"][item]["pass"] for item in fmt.ITEMS)
    assert not report["published_tools"]["script"]["matches_published"]
    assert not report["pass"]
    assert fmt.format_table(report).endswith(
        "Overall: FAIL (the criteria or the script are not the ones metadata.yaml "
        "publishes)"
    )


@POSIX
def test_the_published_script_passes_wherever_it_lies(root, tmp_path):
    copy = tmp_path / "elsewhere" / "check.py"
    copy.parent.mkdir()
    shutil.copy(SCRIPT, copy)
    code, _, err, payload = run(root, root, "--seed", SEED, script=copy)
    assert code == 0, err
    assert payload["script"]["matches_published"] is True
    # An edited copy, run in its place, does not pass.
    copy.write_bytes(SCRIPT.read_bytes() + b"\n")
    code, out, _, payload = run(root, root, "--seed", SEED, script=copy)
    assert code == 1
    assert payload["script"]["matches_published"] is False
    assert "script: " in out and "the published one: NO" in out


def test_a_run_passes_only_with_the_published_criteria(root, tmp_path):
    weaker = criteria_file(tmp_path, criteria_text(**{**SMALL, "min_frames": 1}))
    criteria, digest = fmt.load_criteria(weaker)
    report = fmt.run_checks(
        root, criteria, seed=SEED, criteria_path=weaker, criteria_sha256=digest
    )
    assert all(report["items"][item]["pass"] for item in fmt.ITEMS)
    assert not report["published_tools"]["criteria"]["matches_published"]
    assert not report["pass"]
    # The same values in other bytes are other criteria.
    same = criteria_file(tmp_path, (root / CRITERIA).read_text() + "\n")
    criteria, digest = fmt.load_criteria(same)
    report = fmt.run_checks(
        root, criteria, seed=SEED, criteria_path=same, criteria_sha256=digest
    )
    assert not report["published_tools"]["pass"]
    # A byte-identical copy elsewhere is the published file.
    copy = tmp_path / "copy.yaml"
    shutil.copy(root / CRITERIA, copy)
    criteria, digest = fmt.load_criteria(copy)
    report = fmt.run_checks(
        root, criteria, seed=SEED, criteria_path=copy, criteria_sha256=digest
    )
    assert report["pass"]


def test_criteria_given_as_values_must_be_the_datasets_own(root):
    criteria, _ = fmt.load_criteria(root / CRITERIA)
    assert fmt.run_checks(root, criteria, seed=SEED)["pass"]
    criteria["scale"]["min_frames"] = 1
    report = fmt.run_checks(root, criteria, seed=SEED)
    assert not report["published_tools"]["criteria"]["matches_published"]
    assert not report["pass"]


def test_criteria_given_with_a_path_alone_must_be_that_files(root):
    path = root / CRITERIA
    criteria, digest = fmt.load_criteria(path)
    report = fmt.run_checks(root, criteria, seed=SEED, criteria_path=path)
    assert report["published_tools"]["criteria"]["sha256"] == digest
    assert report["pass"]
    criteria["scale"]["min_frames"] = 1
    criteria["diversity"]["min_object_types"] = 1
    report = fmt.run_checks(root, criteria, seed=SEED, criteria_path=path)
    assert all(report["items"][item]["pass"] for item in fmt.ITEMS)
    # The values applied were read from no bytes the file holds.
    assert report["published_tools"]["criteria"]["sha256"] is None
    assert not report["published_tools"]["criteria"]["matches_published"]
    assert not report["pass"]


def test_a_tool_that_is_not_listed_fails_the_run(root):
    metadata = read_meta(root)
    metadata["tools"] = [t for t in metadata["tools"] if t["path"] != CRITERIA]
    write_meta(root, metadata)
    report = check(root)
    assert report["published_tools"]["criteria"]["published_sha256"] is None
    assert not report["pass"]


# --- the shipped script ---------------------------------------------------------------


def test_the_shipped_script_imports_only_numpy_pyarrow_and_pyyaml():
    source = SCRIPT.read_text()
    tree = ast.parse(source, feature_version=(3, 8))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "a relative import"
            modules.add(node.module.split(".")[0])
    assert modules - set(sys.stdlib_module_names) == {"numpy", "pyarrow", "yaml"}
    assert "__future__" not in modules
    # The path fix and the version test come before every other import.
    first = [type(node).__name__ for node in tree.body[:4]]
    assert first == ["Expr", "Import", "If", "If"]
    assert tree.body[1].names[0].name == "sys"
    assert source.splitlines()[-2:] == [
        'if __name__ == "__main__":',
        "    sys.exit(main())",
    ]


@pytest.mark.parametrize("name", ["check_dataset.py", "check_criteria.yaml"])
def test_the_shipped_files_have_fixed_bytes(name):
    data = SCRIPT.with_name(name).read_bytes()
    assert b"\r" not in data
    assert data.endswith(b"\n")
    attributes = (Path(__file__).parents[1] / ".gitattributes").read_text()
    assert f"src/openphysgraspgen/dataset/{name} -text" in attributes


def _old_python():
    for name in ("python3.9", "python3.8", "/usr/bin/python3"):
        path = shutil.which(name)
        if path is None:
            continue
        version = subprocess.run(
            [path, "-c", "import sys; print(sys.version_info[:2] < (3, 10))"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if version == "True":
            return path
    return None


@pytest.mark.skipif(_old_python() is None, reason="needs a Python older than 3.10")
def test_an_old_python_is_refused_with_a_result_line(root):
    done = subprocess.run(
        [_old_python(), "-I", str(root / "tools/check_dataset.py"), str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 2
    payload = json.loads(done.stdout.splitlines()[-1][len("RESULT ") :])
    assert payload["reason"].startswith("the check needs Python 3.10 or newer; this is")


# --- finalize runs the check ----------------------------------------------------------


@POSIX
def test_finalize_runs_the_check_it_publishes(tmp_path):
    root = tmp_path / "dataset"
    build_dataset(root, ffmpeg=fake_av1(tmp_path / "bin"), doi=None, download_url=None)
    criteria = tmp_path / "dataset-criteria.yaml"
    result = finalize_dataset(root, criteria=criteria, check_seed=SEED)["check"]
    # Without its DOI and download URL the dataset fails its specification.
    assert (result["exit_status"], result["pass"]) == (1, False)
    assert result["metadata"]["spec"]["pass"] is False
    assert all(result[item]["pass"] for item in ("scale", "consistency", "diversity"))
    assert result["script"]["matches_published"] is True
    assert result["criteria"]["matches_published"] is True
    metadata = finalize_dataset(
        root,
        doi=DOI,
        download_url=URL,
        criteria=criteria,
        check_seed=SEED,
        workers=2,
    )
    assert (metadata["check"]["exit_status"], metadata["check"]["pass"]) == (0, True)
    assert metadata["check"]["consistency"]["projection"]["seed"] == SEED
    assert "check" not in read_meta(root)


def test_finalize_refuses_a_check_that_could_not_run(root, monkeypatch):
    with pytest.raises(DatasetError, match="check_seed must be an integer: '7'"):
        finalize_dataset(root, criteria=root / CRITERIA, check_seed="7")

    def refused(args, **kwargs):
        stdout = 'RESULT {"pass": false, "reason": "no NumPy"}\n'
        return subprocess.CompletedProcess(args, 2, stdout=stdout)

    monkeypatch.setattr(subprocess, "run", refused)
    with pytest.raises(
        DatasetError,
        match=r"the dataset check could not run \(exit status 2\): no NumPy",
    ):
        finalize_dataset(root, criteria=root / CRITERIA, check_seed=SEED)

    def crashed(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout="Traceback\n")

    monkeypatch.setattr(subprocess, "run", crashed)
    with pytest.raises(DatasetError, match="exit status 1\\): it printed no RESULT"):
        finalize_dataset(root, criteria=root / CRITERIA, check_seed=SEED)
