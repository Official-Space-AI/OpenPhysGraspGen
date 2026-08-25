from __future__ import annotations

from typing import Any

from ..force_label import ForceLabel, force_from_contacts
from .base import SensorLayer


class ForceLayer(SensorLayer):
    name = "force"

    def __init__(self, contact_provenance: str) -> None:
        if not isinstance(contact_provenance, str) or not contact_provenance.strip():
            raise ValueError("contact_provenance must be a nonempty string")
        self.contact_provenance = contact_provenance

    def capture(self, sim_state: Any) -> ForceLabel:
        return force_from_contacts(sim_state, self.contact_provenance)
