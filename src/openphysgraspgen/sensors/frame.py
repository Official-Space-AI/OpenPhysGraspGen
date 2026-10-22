from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SimFrame:
    model: Any
    particle_q: Any
    contacts: Any
    body_q: Any
    n_robot_shapes: int
    # slice objects are unhashable before Python 3.12, so they need a default factory.
    soft_slice: slice = field(default_factory=lambda: slice(None))
    sim_time: float = 0.0
    waypoint: int = -1
    frame: int = -1
    state: Any = None
