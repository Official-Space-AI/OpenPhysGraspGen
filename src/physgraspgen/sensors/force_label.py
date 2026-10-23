from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ForceLabel:
    contact_force: NDArray[np.float64]
    contact_particle: NDArray[np.int64]
    contact_is_ground: NDArray[np.bool_]
    gripper_net_force: NDArray[np.float64]
    ground_net_force: NDArray[np.float64]
    gripper_force_mag: float
    ground_force_mag: float
    n_contacts: int
    method: str
    contact_provenance: str


def _array(value: Any, dtype: type) -> np.ndarray:
    getter = getattr(value, "numpy", None)
    source = getter() if callable(getter) else value
    return np.asarray(source, dtype=dtype)


def _quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quaternion[:, :3]
    scalar = quaternion[:, 3:4]
    doubled_cross = 2.0 * np.cross(xyz, vector)
    return vector + scalar * doubled_cross + np.cross(xyz, doubled_cross)


def _transform_points(
    body_transform: np.ndarray,
    body_index: np.ndarray,
    local_point: np.ndarray,
) -> np.ndarray:
    result = local_point.astype(np.float64).copy()
    attached = body_index >= 0
    if attached.any():
        transform = body_transform[body_index[attached]]
        result[attached] = transform[:, :3] + _quat_rotate(
            transform[:, 3:7], local_point[attached]
        )
    return result


def force_from_contacts(sim_state: Any, contact_provenance: str) -> ForceLabel:
    if not isinstance(contact_provenance, str) or not contact_provenance.strip():
        raise ValueError("contact_provenance must be a nonempty string")
    model = sim_state.model
    contacts = sim_state.contacts
    shape_body = _array(model.shape_body, int).reshape(-1)
    robot_shapes = int(sim_state.n_robot_shapes)
    if robot_shapes < 0 or robot_shapes > shape_body.size:
        raise ValueError("n_robot_shapes is outside the shape array")
    if shape_body.size - robot_shapes > 1:
        raise ValueError("at most one non-robot ground shape is supported")

    particles = _array(contacts.soft_contact_particle, int).reshape(-1)
    capacity = particles.size
    count_values = _array(contacts.soft_contact_count, int).reshape(-1)
    if count_values.size != 1:
        raise ValueError("soft_contact_count must contain one value")
    count = min(max(int(count_values[0]), 0), capacity)
    if count == 0:
        return ForceLabel(
            contact_force=np.zeros((0, 3), dtype=float),
            contact_particle=np.zeros(0, dtype=int),
            contact_is_ground=np.zeros(0, dtype=bool),
            gripper_net_force=np.zeros(3, dtype=float),
            ground_net_force=np.zeros(3, dtype=float),
            gripper_force_mag=0.0,
            ground_force_mag=0.0,
            n_contacts=0,
            method="vbd-penalty-normal",
            contact_provenance=contact_provenance,
        )

    particle = particles[:count]
    shape = _array(contacts.soft_contact_shape, int).reshape(-1)[:count]
    body_position = _array(contacts.soft_contact_body_pos, float)[:count]
    normal = _array(contacts.soft_contact_normal, float)[:count]
    particle_q = np.asarray(sim_state.particle_q, dtype=float)[particle]
    body_q = np.asarray(sim_state.body_q, dtype=float)
    contact_point = _transform_points(body_q, shape_body[shape], body_position)

    radius = _array(model.particle_radius, float).reshape(-1)[particle]
    margin = _array(model.shape_margin, float).reshape(-1)[shape]
    shape_stiffness = _array(model.shape_material_ke, float).reshape(-1)[shape]
    stiffness = 0.5 * (float(model.soft_contact_ke) + shape_stiffness)
    penetration = np.maximum(
        radius + margin - np.sum(normal * (particle_q - contact_point), axis=1),
        0.0,
    )
    magnitude = penetration * stiffness
    contact_force = normal * magnitude[:, None]
    is_ground = shape >= robot_shapes
    is_gripper = ~is_ground

    return ForceLabel(
        contact_force=contact_force,
        contact_particle=particle,
        contact_is_ground=is_ground,
        gripper_net_force=(
            contact_force[is_gripper].sum(axis=0)
            if is_gripper.any()
            else np.zeros(3, dtype=float)
        ),
        ground_net_force=(
            contact_force[is_ground].sum(axis=0)
            if is_ground.any()
            else np.zeros(3, dtype=float)
        ),
        gripper_force_mag=float(magnitude[is_gripper].sum()),
        ground_force_mag=float(magnitude[is_ground].sum()),
        n_contacts=count,
        method="vbd-penalty-normal",
        contact_provenance=contact_provenance,
    )
