# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
"""Compare hyperelastic constitutive laws under uniaxial stretch."""

import numpy as np

from openphysgraspgen import (
    MooneyRivlinParams,
    OgdenParams,
    YeohParams,
    cauchy_stress,
    von_mises_stress,
)

# Coefficients of the same order as the Yeoh example; illustrative, not measured.
materials = [
    ("Yeoh", YeohParams(c10=0.12e6, c20=0.01e6, c30=0.0, nu=0.45)),
    ("Mooney-Rivlin", MooneyRivlinParams(c10=0.10e6, c01=0.02e6, nu=0.45)),
    (
        "Ogden",
        OgdenParams(mu=(0.2e6, 0.01e6, -0.005e6), alpha=(1.3, 5.0, -2.0), nu=0.45),
    ),
]

F = np.diag([1.2, 0.92, 0.92])  # uniaxial stretch

for name, params in materials:
    sigma = cauchy_stress(F, params)
    stress_vm = von_mises_stress(sigma)
    print(f"{name}: {stress_vm:.4e} Pa")
