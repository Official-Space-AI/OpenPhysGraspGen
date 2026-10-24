from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .layers.base import SensorLayer


class SensorRecorder:
    def __init__(self, layers: Sequence[SensorLayer]) -> None:
        prepared = tuple(layers)
        names = [layer.name for layer in prepared]
        if len(names) != len(set(names)):
            raise ValueError("sensor layer names must be unique")
        self.layers = prepared

    def record_frame(self, sim_state: Any) -> dict[str, Any]:
        return {layer.name: layer.capture(sim_state) for layer in self.layers}
