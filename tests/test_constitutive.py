# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from physgraspgen import (
    MooneyRivlinParams,
    OgdenParams,
    YeohParams,
    cauchy_stress,
    strain_energy_density,
    von_mises_stress,
)

LAWS = [
    YeohParams(c10=0.12e6, c20=0.01e6, c30=0.0, nu=0.45),
    MooneyRivlinParams(c10=0.10e6, c01=0.02e6, nu=0.45),
    OgdenParams(mu=(0.2e6, 0.01e6, -0.005e6), alpha=(1.3, 5.0, -2.0), nu=0.45),
]


@pytest.mark.parametrize("law", LAWS, ids=lambda law: type(law).__name__)
def test_undeformed_state_is_stress_free(law):
    assert np.allclose(cauchy_stress(np.eye(3), law), 0.0, atol=1e-6)
    assert strain_energy_density(np.eye(3), law) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("law", LAWS, ids=lambda law: type(law).__name__)
def test_small_simple_shear_recovers_shear_modulus(law):
    gamma = 1e-5
    gradient = np.eye(3)
    gradient[0, 1] = gamma
    stress = cauchy_stress(gradient, law)
    assert stress[0, 1] / gamma == pytest.approx(law.mu0, rel=1e-3)


@pytest.mark.parametrize("law", LAWS, ids=lambda law: type(law).__name__)
def test_stress_is_symmetric(law):
    gradient = np.array([[1.1, 0.05, 0.0], [0.02, 0.95, 0.01], [0.0, 0.03, 0.97]])
    stress = cauchy_stress(gradient, law)
    assert np.allclose(stress, stress.T, atol=1e-6 * np.abs(stress).max())


@pytest.mark.parametrize("law", LAWS, ids=lambda law: type(law).__name__)
def test_inverted_element_gives_nan_stress(law):
    stress = cauchy_stress(np.diag([1.0, 1.0, -1.0]), law)
    assert np.isnan(stress).all()


def test_von_mises_of_uniaxial_stress_is_its_magnitude():
    assert von_mises_stress(np.diag([-3.0, 0.0, 0.0])) == pytest.approx(3.0)


def test_von_mises_ignores_hydrostatic_pressure():
    assert von_mises_stress(5.0 * np.eye(3)) == pytest.approx(0.0, abs=1e-12)


def test_von_mises_rejects_non_3x3():
    with pytest.raises(ValueError):
        von_mises_stress(np.eye(2))


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        MooneyRivlinParams(c10=0.0, c01=0.1, nu=0.45)
    with pytest.raises(ValueError):
        OgdenParams(mu=(1.0, 0.0, 0.0), alpha=(0.0, 1.0, 1.0), nu=0.45)
