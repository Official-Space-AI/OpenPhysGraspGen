from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np

from .formulations import CalibrationModel, ProductionModel
from .records import (
    FloatArray,
    MaterialPair,
    MeasuredObjectData,
    ProbeScene,
    TrainingSample,
    vector,
)


def _select_parameter(
    candidates: Iterable[FloatArray],
    target: FloatArray,
    response: Callable[[FloatArray], FloatArray],
) -> FloatArray:
    expected = vector(target, "target_response")
    best_candidate: FloatArray | None = None
    best_error = float("inf")
    for raw_candidate in candidates:
        candidate = vector(raw_candidate, "candidate")
        candidate_response = vector(response(candidate), "candidate_response")
        if candidate_response.shape != expected.shape:
            raise ValueError("candidate response must match the target response shape")
        with np.errstate(over="ignore", invalid="ignore"):
            error = float(np.mean((candidate_response - expected) ** 2))
        if not np.isfinite(error):
            raise ValueError("candidate response error must be finite")
        if error < best_error:
            best_candidate = candidate
            best_error = error
    if best_candidate is None:
        raise ValueError("at least one candidate is required")
    return best_candidate.copy()


def calibrate_material(
    data: MeasuredObjectData,
    formulation: CalibrationModel,
    candidates: Iterable[FloatArray],
) -> FloatArray:
    return _select_parameter(
        candidates,
        data.mechanical_response,
        lambda candidate: formulation.measured_response(data, candidate),
    )


def transfer_parameter(
    calibration_parameters: FloatArray,
    calibration_formulation: CalibrationModel,
    production_formulation: ProductionModel,
    probes: Sequence[ProbeScene],
    candidates: Iterable[FloatArray],
) -> FloatArray:
    if not probes:
        raise ValueError("at least one probe scene is required")
    target = np.concatenate(
        [
            calibration_formulation.probe_response(probe, calibration_parameters)
            for probe in probes
        ]
    )
    return _select_parameter(
        candidates,
        target,
        lambda candidate: np.concatenate(
            [
                production_formulation.probe_response(probe, candidate)
                for probe in probes
            ]
        ),
    )


def build_training_data(
    calibration_formulation: CalibrationModel,
    production_formulation: ProductionModel,
    materials: Sequence[MaterialPair],
    actions: Sequence[FloatArray],
    fracture_rule: Callable[[FloatArray], bool],
) -> list[TrainingSample]:
    samples: list[TrainingSample] = []
    for material in materials:
        for raw_action in actions:
            action = vector(raw_action, "action")
            observation = production_formulation.manipulate(
                material.production_parameters, action
            )
            stress = calibration_formulation.internal_stress(
                observation.boundary_conditions,
                material.calibration_parameters,
            )
            samples.append(
                TrainingSample(
                    observation=observation,
                    action=action,
                    material_distribution=vector(
                        material.distribution_parameters, "distribution_parameters"
                    ),
                    internal_stress=vector(stress, "internal_stress"),
                    fracture=bool(fracture_rule(stress)),
                )
            )
    if not samples:
        raise ValueError("training data cannot be empty")
    return samples
