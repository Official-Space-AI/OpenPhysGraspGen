from __future__ import annotations

from typing import Protocol

import numpy as np

from .records import (
    FloatArray,
    MeasuredObjectData,
    ProbeScene,
    RobotObservation,
    vector,
)


def _scalar(value: FloatArray, name: str) -> float:
    prepared = vector(value, name)
    if prepared.size != 1:
        raise ValueError(f"{name} must contain exactly one value")
    return float(prepared[0])


class CalibrationModel(Protocol):
    def measured_response(
        self, data: MeasuredObjectData, material_parameters: FloatArray
    ) -> FloatArray: ...

    def probe_response(
        self, scene: ProbeScene, material_parameters: FloatArray
    ) -> FloatArray: ...

    def internal_stress(
        self, boundary_conditions: FloatArray, material_parameters: FloatArray
    ) -> FloatArray: ...


class ProductionModel(Protocol):
    def probe_response(
        self, scene: ProbeScene, material_parameters: FloatArray
    ) -> FloatArray: ...

    def manipulate(
        self, material_parameters: FloatArray, action: FloatArray
    ) -> RobotObservation: ...


class CalibrationFormulation:
    def measured_response(
        self, data: MeasuredObjectData, material_parameters: FloatArray
    ) -> FloatArray:
        scale = float(np.mean(np.abs(data.structure)))
        parameter = _scalar(material_parameters, "material_parameters")
        return parameter * scale * data.applied_input

    def probe_response(
        self, scene: ProbeScene, material_parameters: FloatArray
    ) -> FloatArray:
        return _scalar(material_parameters, "material_parameters") * scene.applied_input

    def internal_stress(
        self, boundary_conditions: FloatArray, material_parameters: FloatArray
    ) -> FloatArray:
        return _scalar(material_parameters, "material_parameters") * vector(
            boundary_conditions, "boundary_conditions"
        )


class ProductionFormulation:
    def probe_response(
        self, scene: ProbeScene, material_parameters: FloatArray
    ) -> FloatArray:
        applied = scene.applied_input
        parameter = _scalar(material_parameters, "material_parameters")
        return parameter * applied * (1.0 + np.abs(applied))

    def manipulate(
        self, material_parameters: FloatArray, action: FloatArray
    ) -> RobotObservation:
        parameter = _scalar(material_parameters, "material_parameters")
        command = _scalar(action, "action")
        boundary = np.array([command / (1.0 + abs(parameter))], dtype=float)
        return RobotObservation(
            depth=np.array([1.0 - min(abs(boundary[0]), 1.0)], dtype=float),
            contact_force=np.array([parameter * abs(boundary[0])], dtype=float),
            proprioception=np.array([command], dtype=float),
            boundary_conditions=boundary,
        )
