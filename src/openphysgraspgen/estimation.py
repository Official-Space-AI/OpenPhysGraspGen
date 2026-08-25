from __future__ import annotations

import numpy as np

from .records import RobotObservation, StateEstimate, TrainingSample


class LinearJointEstimator:
    def __init__(self) -> None:
        self._weights: np.ndarray | None = None
        self._material_width = 0
        self._feature_width = 0

    def fit(self, samples: list[TrainingSample]) -> LinearJointEstimator:
        if not samples:
            raise ValueError("samples cannot be empty")
        feature_widths = {
            sample.observation.feature_vector().size for sample in samples
        }
        material_widths = {sample.material_distribution.size for sample in samples}
        stress_widths = {sample.internal_stress.size for sample in samples}
        if len(feature_widths) != 1:
            raise ValueError("all observations must have the same feature width")
        if len(material_widths) != 1 or len(stress_widths) != 1:
            raise ValueError("all training targets must have matching component widths")
        features = np.stack([sample.observation.feature_vector() for sample in samples])
        design = np.column_stack((np.ones(len(samples)), features))
        targets = np.stack(
            [
                np.concatenate((sample.material_distribution, sample.internal_stress))
                for sample in samples
            ]
        )
        material_width = samples[0].material_distribution.size
        weights = np.linalg.lstsq(design, targets, rcond=None)[0]
        if not np.all(np.isfinite(weights)):
            raise ValueError("training produced non-finite weights")
        self._material_width = material_width
        self._feature_width = features.shape[1]
        self._weights = weights
        return self

    def predict(self, observation: RobotObservation) -> StateEstimate:
        if self._weights is None:
            raise RuntimeError("fit must be called before predict")
        observation_feature = observation.feature_vector()
        if observation_feature.size != self._feature_width:
            raise ValueError(
                "observation feature width does not match the fitted model"
            )
        feature = np.concatenate(([1.0], observation_feature))
        prediction = np.asarray(feature @ self._weights, dtype=float)
        return StateEstimate(
            material_distribution=prediction[: self._material_width],
            internal_stress=prediction[self._material_width :],
        )
