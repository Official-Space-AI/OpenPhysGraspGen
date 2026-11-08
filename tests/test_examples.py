# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
import os
import pathlib
import subprocess
import sys

import pytest

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parent.parent / "examples"
EXAMPLES = sorted(EXAMPLES_DIR.glob("*.py"))


def test_examples_exist():
    assert len(EXAMPLES) > 0


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_example_runs(script):
    env = os.environ.copy()
    src_dir = str(pathlib.Path(__file__).resolve().parent.parent / "src")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = src_dir
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    if script.name == "constitutive_law.py":
        output_val = float(result.stdout.strip())
        assert 1.3e5 <= output_val <= 1.5e5
