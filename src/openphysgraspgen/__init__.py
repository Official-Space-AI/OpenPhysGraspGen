from .estimation import LinearJointEstimator
from .generation import build_training_data, calibrate_material, transfer_parameter
from .records import (
    MaterialPair,
    MeasuredObjectData,
    ProbeScene,
    RobotObservation,
    SafetyDecision,
    StateEstimate,
    TrainingSample,
)


def run_pipeline(
    fracture_threshold: float = 0.25,
    confidence_threshold: float = 0.5,
) -> dict[str, object]:
    from .pipeline import run_pipeline as execute

    return execute(fracture_threshold, confidence_threshold)


__all__ = [
    "LinearJointEstimator",
    "MaterialPair",
    "MeasuredObjectData",
    "ProbeScene",
    "RobotObservation",
    "SafetyDecision",
    "StateEstimate",
    "TrainingSample",
    "build_training_data",
    "calibrate_material",
    "run_pipeline",
    "transfer_parameter",
]
