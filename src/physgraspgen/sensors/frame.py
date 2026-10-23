from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SimFrame:
    model: Any
    particle_q: Any
    contacts: Any
    body_q: Any
    n_robot_shapes: int
    soft_slice: slice = slice(None)
    sim_time: float = 0.0
    waypoint: int = -1
    frame: int = -1
    state: Any = None
