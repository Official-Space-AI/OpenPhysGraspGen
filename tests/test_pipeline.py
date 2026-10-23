import math

import pytest

from physgraspgen import run_pipeline


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
