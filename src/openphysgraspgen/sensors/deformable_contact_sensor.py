from __future__ import annotations

from typing import Any

from .force_label import ForceLabel
from .layers.force import ForceLayer


class DeformableContactSensor(ForceLayer):
    def __init__(self, contact_provenance: str) -> None:
        super().__init__(contact_provenance)

    def capture(self, sim_state: Any) -> ForceLabel:
        return super().capture(sim_state)

    def __repr__(self) -> str:
        return (
            f"DeformableContactSensor(contact_provenance={self.contact_provenance!r})"
        )
