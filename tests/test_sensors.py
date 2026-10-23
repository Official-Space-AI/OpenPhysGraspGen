from types import SimpleNamespace

import numpy as np
import pytest

from openphysgraspgen import DeformableContactSensor, SensorRecorder, SimFrame

RADIUS = 0.01
CONTACT_KE = 1.0e4
SHAPE_KE = 3.0e4
STIFFNESS = 0.5 * (CONTACT_KE + SHAPE_KE)


def make_frame(count=2):
    # Shape 0 is a gripper pad on body 0 at z = 0.05; shape 1 is the ground.
    model = SimpleNamespace(
        shape_body=np.array([0, -1]),
        particle_radius=np.full(2, RADIUS),
        shape_margin=np.zeros(2),
        shape_material_ke=np.full(2, SHAPE_KE),
        soft_contact_ke=CONTACT_KE,
    )
    contacts = SimpleNamespace(
        soft_contact_count=np.array([count]),
        soft_contact_particle=np.array([0, 1]),
        soft_contact_shape=np.array([0, 1]),
        soft_contact_body_pos=np.zeros((2, 3)),
        soft_contact_normal=np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]),
    )
    return SimFrame(
        model=model,
        particle_q=np.array([[0.0, 0.0, 0.042], [0.0, 0.0, 0.009]]),
        contacts=contacts,
        body_q=np.array([[0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0]]),
        n_robot_shapes=1,
    )


def test_penalty_force_is_split_between_gripper_and_ground():
    label = DeformableContactSensor("unit-test").capture(make_frame())
    # Gripper particle is 8 mm below the pad: penetration 10 - 8 = 2 mm.
    # Ground particle is 9 mm above the ground: penetration 10 - 9 = 1 mm.
    assert label.n_contacts == 2
    assert label.gripper_force_mag == pytest.approx(0.002 * STIFFNESS)
    assert label.ground_force_mag == pytest.approx(0.001 * STIFFNESS)
    assert np.allclose(label.gripper_net_force, [0.0, 0.0, -0.002 * STIFFNESS])
    assert np.allclose(label.ground_net_force, [0.0, 0.0, 0.001 * STIFFNESS])
    assert label.contact_is_ground.tolist() == [False, True]
    assert label.contact_provenance == "unit-test"


def test_no_contacts_gives_empty_label():
    label = DeformableContactSensor("unit-test").capture(make_frame(count=0))
    assert label.n_contacts == 0
    assert label.contact_force.shape == (0, 3)
    assert label.gripper_force_mag == 0.0


def test_recorder_returns_one_entry_per_layer():
    recorder = SensorRecorder([DeformableContactSensor("unit-test")])
    assert set(recorder.record_frame(make_frame())) == {"force"}


def test_recorder_rejects_duplicate_layer_names():
    with pytest.raises(ValueError):
        SensorRecorder([DeformableContactSensor("a"), DeformableContactSensor("b")])


def test_sensor_requires_provenance():
    with pytest.raises(ValueError):
        DeformableContactSensor("  ")
