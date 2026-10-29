# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
import importlib.metadata
import math

import pytest

import openphysgraspgen
from openphysgraspgen import run_pipeline
from openphysgraspgen.pipeline import main


def test_version_matches_the_installed_metadata():
    assert openphysgraspgen.__version__ == importlib.metadata.version(
        "openphysgraspgen"
    )


def test_version_flag_prints_the_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert (
        capsys.readouterr().out.strip()
        == f"openphysgraspgen {openphysgraspgen.__version__}"
    )


def test_reference_pipeline_runs_end_to_end():
    result = run_pipeline()
    assert result["training_samples"] > 0
    assert 0 <= result["fracture_samples"] <= result["training_samples"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["estimate_allowed"], bool)
    assert all(math.isfinite(v) for v in result["calibration_parameters"])


def test_higher_fracture_threshold_never_adds_fracture_samples():
    strict = run_pipeline(fracture_threshold=0.05)["fracture_samples"]
    loose = run_pipeline(fracture_threshold=10.0)["fracture_samples"]
    assert loose <= strict


@pytest.mark.parametrize("threshold", [-0.1, float("nan"), float("inf")])
def test_invalid_fracture_threshold_is_rejected(threshold):
    with pytest.raises(ValueError):
        run_pipeline(fracture_threshold=threshold)
