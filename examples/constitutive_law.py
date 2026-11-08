# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from openphysgraspgen import YeohParams, cauchy_stress, von_mises_stress

silicone = YeohParams(c10=0.12e6, c20=0.01e6, c30=0.0, nu=0.45)
F = np.diag([1.2, 0.92, 0.92])  # uniaxial stretch
sigma = cauchy_stress(F, silicone)  # 3x3 Cauchy stress [Pa]
print(von_mises_stress(sigma))  # about 1.4e5 Pa
