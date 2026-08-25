from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def vector(value: ArrayLike, name: str) -> FloatArray:
    result = np.array(value, dtype=float, copy=True).reshape(-1)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MeasuredObjectData:
    structure: FloatArray
    applied_input: FloatArray
    mechanical_response: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "structure", vector(self.structure, "structure"))
        object.__setattr__(
            self, "applied_input", vector(self.applied_input, "applied_input")
        )
        object.__setattr__(
            self,
            "mechanical_response",
            vector(self.mechanical_response, "mechanical_response"),
        )
        if self.applied_input.shape != self.mechanical_response.shape:
            raise ValueError(
                "applied_input and mechanical_response must have the same shape"
            )


@dataclass(frozen=True)
class ProbeScene:
    applied_input: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "applied_input", vector(self.applied_input, "applied_input")
        )


@dataclass(frozen=True)
class RobotObservation:
    depth: FloatArray
    contact_force: FloatArray
    proprioception: FloatArray
    boundary_conditions: FloatArray

    def __post_init__(self) -> None:
        for field_name in (
            "depth",
            "contact_force",
            "proprioception",
            "boundary_conditions",
        ):
            object.__setattr__(
                self, field_name, vector(getattr(self, field_name), field_name)
            )

    def feature_vector(self) -> FloatArray:
        return np.concatenate((self.depth, self.contact_force, self.proprioception))


@dataclass(frozen=True)
class MaterialPair:
    calibration_parameters: FloatArray
    production_parameters: FloatArray
    distribution_parameters: FloatArray

    def __post_init__(self) -> None:
        for field_name in (
            "calibration_parameters",
            "production_parameters",
            "distribution_parameters",
        ):
            object.__setattr__(
                self, field_name, vector(getattr(self, field_name), field_name)
            )


@dataclass(frozen=True)
class TrainingSample:
    observation: RobotObservation
    action: FloatArray
    material_distribution: FloatArray
    internal_stress: FloatArray
    fracture: bool

    def __post_init__(self) -> None:
        if not isinstance(self.observation, RobotObservation):
            raise TypeError("observation must be a RobotObservation")
        for field_name in ("action", "material_distribution", "internal_stress"):
            object.__setattr__(
                self, field_name, vector(getattr(self, field_name), field_name)
            )
        object.__setattr__(self, "fracture", bool(self.fracture))


@dataclass(frozen=True)
class StateEstimate:
    material_distribution: FloatArray
    internal_stress: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "material_distribution",
            vector(self.material_distribution, "material_distribution"),
        )
        object.__setattr__(
            self,
            "internal_stress",
            vector(self.internal_stress, "internal_stress"),
        )


@dataclass(frozen=True)
class SafetyDecision:
    confidence: float
    estimate_allowed: bool
    action_scale_limit: float
