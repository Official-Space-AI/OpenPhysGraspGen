from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

StressTensor = NDArray[np.float64]


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _poisson_ratio(value: float) -> float:
    result = _finite(value, "nu")
    if not -1.0 < result < 0.5:
        raise ValueError("nu must be between -1 and 0.5")
    return result


def _gradient(value: ArrayLike) -> StressTensor:
    result = np.array(value, dtype=float, copy=True)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError("deformation_gradient must be a finite 3 by 3 matrix")
    return result


def _volume_ratio(gradient: StressTensor) -> float:
    return float(np.linalg.det(gradient))


def _invalid_stress() -> StressTensor:
    return np.full((3, 3), np.nan, dtype=float)


class _HyperelasticParams:
    nu: float

    @property
    def mu0(self) -> float:
        raise NotImplementedError

    @property
    def k_bulk(self) -> float:
        return 2.0 * self.mu0 * (1.0 + self.nu) / (3.0 * (1.0 - 2.0 * self.nu))

    def _validate_common(self) -> None:
        if not self.mu0 > 0.0 or not np.isfinite(self.mu0):
            raise ValueError(
                "the small-strain shear modulus must be finite and positive"
            )


@dataclass(frozen=True)
class YeohParams(_HyperelasticParams):
    c10: float
    c20: float
    c30: float
    nu: float

    def __post_init__(self) -> None:
        for name in ("c10", "c20", "c30"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        object.__setattr__(self, "nu", _poisson_ratio(self.nu))
        self._validate_common()

    @property
    def mu0(self) -> float:
        return 2.0 * self.c10

    def strain_energy_density(self, deformation_gradient: ArrayLike) -> float:
        gradient = _gradient(deformation_gradient)
        volume_ratio = _volume_ratio(gradient)
        if volume_ratio <= 0.0 or not np.isfinite(volume_ratio):
            return float("nan")
        first_invariant = float(np.sum(gradient * gradient))
        excess = volume_ratio ** (-2.0 / 3.0) * first_invariant - 3.0
        isochoric = self.c10 * excess + self.c20 * excess**2 + self.c30 * excess**3
        return float(isochoric + 0.5 * self.k_bulk * (volume_ratio - 1.0) ** 2)

    def cauchy_stress(self, deformation_gradient: ArrayLike) -> StressTensor:
        gradient = _gradient(deformation_gradient)
        volume_ratio = _volume_ratio(gradient)
        if volume_ratio <= 0.0 or not np.isfinite(volume_ratio):
            return _invalid_stress()
        first_invariant = float(np.sum(gradient * gradient))
        scale = volume_ratio ** (-2.0 / 3.0)
        reduced_invariant = scale * first_invariant
        excess = reduced_invariant - 3.0
        derivative = self.c10 + 2.0 * self.c20 * excess + 3.0 * self.c30 * excess**2
        inverse_transpose = np.linalg.inv(gradient).T
        first_piola = (
            derivative
            * (
                2.0 * scale * gradient
                - (2.0 / 3.0) * reduced_invariant * inverse_transpose
            )
            + self.k_bulk * (volume_ratio - 1.0) * volume_ratio * inverse_transpose
        )
        return (first_piola @ gradient.T) / volume_ratio


@dataclass(frozen=True)
class MooneyRivlinParams(_HyperelasticParams):
    c10: float
    c01: float
    nu: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "c10", _finite(self.c10, "c10"))
        object.__setattr__(self, "c01", _finite(self.c01, "c01"))
        object.__setattr__(self, "nu", _poisson_ratio(self.nu))
        if self.c10 <= 0.0 or self.c01 < 0.0:
            raise ValueError("c10 must be positive and c01 must be nonnegative")
        self._validate_common()

    @property
    def mu0(self) -> float:
        return 2.0 * (self.c10 + self.c01)

    def _invariants(self, gradient: StressTensor) -> tuple[float, float, float]:
        volume_ratio = _volume_ratio(gradient)
        right_cauchy_green = gradient.T @ gradient
        first = float(np.trace(right_cauchy_green))
        second = 0.5 * (
            first**2 - float(np.trace(right_cauchy_green @ right_cauchy_green))
        )
        return volume_ratio, first, second

    def strain_energy_density(self, deformation_gradient: ArrayLike) -> float:
        gradient = _gradient(deformation_gradient)
        volume_ratio, first, second = self._invariants(gradient)
        if volume_ratio <= 0.0 or not np.isfinite(volume_ratio):
            return float("nan")
        reduced_first = volume_ratio ** (-2.0 / 3.0) * first
        reduced_second = volume_ratio ** (-4.0 / 3.0) * second
        isochoric = self.c10 * (reduced_first - 3.0) + self.c01 * (reduced_second - 3.0)
        return float(isochoric + 0.5 * self.k_bulk * (volume_ratio - 1.0) ** 2)

    def cauchy_stress(self, deformation_gradient: ArrayLike) -> StressTensor:
        gradient = _gradient(deformation_gradient)
        volume_ratio, first, second = self._invariants(gradient)
        if volume_ratio <= 0.0 or not np.isfinite(volume_ratio):
            return _invalid_stress()
        right_cauchy_green = gradient.T @ gradient
        scale_first = volume_ratio ** (-2.0 / 3.0)
        scale_second = volume_ratio ** (-4.0 / 3.0)
        reduced_first = scale_first * first
        reduced_second = scale_second * second
        inverse_transpose = np.linalg.inv(gradient).T
        first_piola = self.c10 * (
            2.0 * scale_first * gradient
            - (2.0 / 3.0) * reduced_first * inverse_transpose
        ) + self.c01 * (
            2.0 * scale_second * (first * gradient - gradient @ right_cauchy_green)
            - (4.0 / 3.0) * reduced_second * inverse_transpose
        )
        first_piola += (
            self.k_bulk * (volume_ratio - 1.0) * volume_ratio * inverse_transpose
        )
        return (first_piola @ gradient.T) / volume_ratio


@dataclass(frozen=True)
class OgdenParams(_HyperelasticParams):
    mu: tuple[float, float, float]
    alpha: tuple[float, float, float]
    nu: float

    def __post_init__(self) -> None:
        mu = tuple(_finite(value, "mu") for value in self.mu)
        alpha = tuple(_finite(value, "alpha") for value in self.alpha)
        if len(mu) != 3 or len(alpha) != 3:
            raise ValueError("mu and alpha must each contain three values")
        if any(value == 0.0 for value in alpha):
            raise ValueError("alpha values must be nonzero")
        if any(m != 0.0 and m * a <= 0.0 for m, a in zip(mu, alpha)):
            raise ValueError("each active Ogden term must satisfy mu times alpha > 0")
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "nu", _poisson_ratio(self.nu))
        self._validate_common()

    @property
    def mu0(self) -> float:
        return 0.5 * sum(m * a for m, a in zip(self.mu, self.alpha))

    def _principal_values(
        self, gradient: StressTensor
    ) -> tuple[float, StressTensor, NDArray[np.float64]]:
        volume_ratio = _volume_ratio(gradient)
        left_basis, stretches, _ = np.linalg.svd(gradient)
        return volume_ratio, left_basis, stretches

    def strain_energy_density(self, deformation_gradient: ArrayLike) -> float:
        gradient = _gradient(deformation_gradient)
        volume_ratio, _, stretches = self._principal_values(gradient)
        if volume_ratio <= 0.0 or not np.isfinite(volume_ratio):
            return float("nan")
        reduced_stretches = volume_ratio ** (-1.0 / 3.0) * stretches
        isochoric = sum(
            (m / a) * (float(np.sum(reduced_stretches**a)) - 3.0)
            for m, a in zip(self.mu, self.alpha)
        )
        return float(isochoric + 0.5 * self.k_bulk * (volume_ratio - 1.0) ** 2)

    def cauchy_stress(self, deformation_gradient: ArrayLike) -> StressTensor:
        gradient = _gradient(deformation_gradient)
        volume_ratio, left_basis, stretches = self._principal_values(gradient)
        if volume_ratio <= 0.0 or not np.isfinite(volume_ratio):
            return _invalid_stress()
        reduced_stretches = volume_ratio ** (-1.0 / 3.0) * stretches
        principal_terms = np.array(
            [
                sum(m * stretch**a for m, a in zip(self.mu, self.alpha))
                for stretch in reduced_stretches
            ],
            dtype=float,
        )
        principal_stress = (
            principal_terms - float(np.mean(principal_terms))
        ) / volume_ratio + self.k_bulk * (volume_ratio - 1.0)
        return left_basis @ np.diag(principal_stress) @ left_basis.T


ConstitutiveParameters = YeohParams | MooneyRivlinParams | OgdenParams
LABEL_LAWS = ("yeoh", "mooney_rivlin", "ogden")


def cauchy_stress(
    deformation_gradient: ArrayLike,
    parameters: ConstitutiveParameters,
) -> StressTensor:
    return parameters.cauchy_stress(deformation_gradient)


def strain_energy_density(
    deformation_gradient: ArrayLike,
    parameters: ConstitutiveParameters,
) -> float:
    return parameters.strain_energy_density(deformation_gradient)


def von_mises_stress(stress: ArrayLike) -> float:
    tensor = np.asarray(stress, dtype=float)
    if tensor.shape != (3, 3):
        raise ValueError("stress must be a 3 by 3 matrix")
    deviator = tensor - np.trace(tensor) * np.eye(3) / 3.0
    return float(np.sqrt(1.5 * np.sum(deviator * deviator)))
