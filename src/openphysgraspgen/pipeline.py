from __future__ import annotations

import argparse
import json

import numpy as np

from .confidence import assess_estimate
from .estimation import LinearJointEstimator
from .formulations import CalibrationFormulation, ProductionFormulation
from .generation import build_training_data, calibrate_material, transfer_parameter
from .records import MaterialPair, MeasuredObjectData, ProbeScene


def run_pipeline(
    fracture_threshold: float = 0.25,
    confidence_threshold: float = 0.5,
) -> dict[str, object]:
    if not np.isfinite(fracture_threshold) or fracture_threshold < 0.0:
        raise ValueError("fracture_threshold must be finite and nonnegative")
    calibration = CalibrationFormulation()
    production = ProductionFormulation()
    measured = MeasuredObjectData(
        structure=np.ones(3),
        applied_input=np.array([0.0, 0.2, 0.4, 0.6]),
        mechanical_response=np.array([0.0, 0.4, 0.8, 1.2]),
    )
    calibration_candidates = [np.array([value]) for value in (1.0, 2.0, 3.0)]
    calibration_parameters = calibrate_material(
        measured,
        calibration,
        calibration_candidates,
    )
    probes = [ProbeScene(np.array([0.2, 0.5]))]
    production_candidates = [
        np.array([value]) for value in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    ]
    production_parameters = transfer_parameter(
        calibration_parameters,
        calibration,
        production,
        probes,
        production_candidates,
    )
    materials = []
    for value in np.linspace(1.0, 3.0, 9):
        material_parameters = np.array([value])
        materials.append(
            MaterialPair(
                calibration_parameters=material_parameters,
                production_parameters=transfer_parameter(
                    material_parameters,
                    calibration,
                    production,
                    probes,
                    production_candidates,
                ),
                distribution_parameters=np.array([value, 0.1 * value]),
            )
        )
    actions = [np.array([value]) for value in np.linspace(-0.6, 0.6, 9)]
    samples = build_training_data(
        calibration,
        production,
        materials,
        actions,
        fracture_rule=lambda stress: bool(np.max(np.abs(stress)) > fracture_threshold),
    )
    estimator = LinearJointEstimator().fit(samples)
    observation = production.manipulate(production_parameters, np.array([0.3]))
    estimate = estimator.predict(observation)
    decision = assess_estimate(
        observation,
        estimate,
        calibration,
        distribution_to_material=lambda distribution: distribution[:1],
        agreement=lambda first, second: (
            1.0 / (1.0 + float(np.mean(np.abs(first - second))))
        ),
        confidence_threshold=confidence_threshold,
    )
    return {
        "calibration_parameters": calibration_parameters.tolist(),
        "production_parameters": production_parameters.tolist(),
        "training_samples": len(samples),
        "fracture_samples": sum(sample.fracture for sample in samples),
        "estimated_material_distribution": estimate.material_distribution.tolist(),
        "estimated_internal_stress": estimate.internal_stress.tolist(),
        "confidence": decision.confidence,
        "estimate_allowed": decision.estimate_allowed,
        "action_scale_limit": decision.action_scale_limit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fracture-threshold", type=float, default=0.25)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    args = parser.parse_args()
    print(
        json.dumps(
            run_pipeline(args.fracture_threshold, args.confidence_threshold),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
