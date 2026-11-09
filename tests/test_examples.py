# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
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
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    if script.name == "constitutive_law.py":
        output_val = float(result.stdout.strip())
        assert 1.3e5 <= output_val <= 1.5e5
