from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .formulations import CalibrationModel
from .records import FloatArray, RobotObservation, SafetyDecision, StateEstimate, vector


def assess_estimate(
    observation: RobotObservation,
    estimate: StateEstimate,
    physics: CalibrationModel,
    distribution_to_material: Callable[[FloatArray], FloatArray],
    agreement: Callable[[FloatArray, FloatArray], float],
    confidence_threshold: float,
) -> SafetyDecision:
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between zero and one")
    material = vector(
        distribution_to_material(estimate.material_distribution),
        "estimated_material",
    )
    recalculated_stress = vector(
        physics.internal_stress(observation.boundary_conditions, material),
        "recalculated_stress",
    )
    if recalculated_stress.shape != estimate.internal_stress.shape:
        raise ValueError("recalculated stress must match the estimated stress shape")
    confidence = float(agreement(estimate.internal_stress, recalculated_stress))
    if not np.isfinite(confidence):
        raise ValueError("agreement must return a finite value")
    confidence = float(np.clip(confidence, 0.0, 1.0))
    allowed = bool(confidence >= float(confidence_threshold))
    return SafetyDecision(
        confidence=confidence,
        estimate_allowed=allowed,
        action_scale_limit=1.0 if allowed else 0.0,
    )
