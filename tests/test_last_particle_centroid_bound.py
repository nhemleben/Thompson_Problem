import math
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np


sys.path.append(str(Path(__file__).resolve().parents[1]))

import last_particle_centroid_bound as lpcb


def test_cartesian_to_spherical_axes_and_quadrants():
    theta, phi = lpcb.cartesian_to_spherical(np.array([1.0, 0.0, 0.0]))
    assert theta == 0.0
    assert phi == math.pi / 2.0

    theta, phi = lpcb.cartesian_to_spherical(np.array([0.0, 1.0, 0.0]))
    assert theta == math.pi / 2.0
    assert phi == math.pi / 2.0

    theta, phi = lpcb.cartesian_to_spherical(np.array([-1.0, 0.0, 0.0]))
    assert theta == math.pi
    assert phi == math.pi / 2.0

    theta, phi = lpcb.cartesian_to_spherical(np.array([0.0, 0.0, 1.0]))
    assert theta == 0.0
    assert phi == 0.0


def test_last_particle_cap_returns_all_sphere_when_sum_vanishes():
    points = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]

    with patch.object(lpcb, "thompson_energy_last_particle", return_value=1.0):
        cap = lpcb.last_particle_cap(points)

    assert cap["all_sphere"] is True
    assert cap["center_theta"] is None
    assert cap["center_phi"] is None
    assert cap["angular_radius"] == math.pi


def test_last_particle_cap_returns_empty_when_lower_dot_bound_is_too_small():
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    with patch.object(lpcb, "thompson_energy_last_particle", return_value=0.1):
        cap = lpcb.last_particle_cap(points)

    assert cap == {"empty": True}


def test_last_particle_cap_returns_center_and_radius_for_regular_case():
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )

    with patch.object(lpcb, "thompson_energy_last_particle", return_value=1.0):
        cap = lpcb.last_particle_cap(points)

    assert cap["center_theta"] == math.pi
    assert cap["center_phi"] == math.pi / 2.0
    np.testing.assert_allclose(cap["center_vector"], np.array([-1.0, 0.0, 0.0]))
    assert cap["S_norm"] == 1.0
    assert cap["dot_bound"] == 0.0
    assert cap["angular_radius"] == math.pi / 2.0


def test_last_particle_cap_uses_supplied_energy_bounds():
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )

    with patch.object(lpcb, "thompson_energy_last_particle", side_effect=AssertionError("should not be called")):
        cap = lpcb.last_particle_cap(points, U_n=4.0, E_prev=3.0)

    assert cap["center_theta"] == math.pi
    assert cap["center_phi"] == math.pi / 2.0
    assert cap["angular_radius"] == math.pi / 2.0


def test_cap_to_theta_phi_bounds_handles_regular_and_polar_cases():
    theta_min, theta_max, phi_min, phi_max = lpcb.cap_to_theta_phi_bounds(
        center_theta=math.pi,
        center_phi=math.pi / 2.0,
        alpha=math.pi / 6.0,
    )

    assert theta_min == math.pi - math.pi / 6.0
    assert theta_max == math.pi + math.pi / 6.0
    np.testing.assert_allclose(phi_min, math.pi / 3.0)
    np.testing.assert_allclose(phi_max, 2.0 * math.pi / 3.0)

    theta_min, theta_max, phi_min, phi_max = lpcb.cap_to_theta_phi_bounds(
        center_theta=1.23,
        center_phi=0.0,
        alpha=0.25,
    )

    assert theta_min == 0.0
    assert theta_max == 2.0 * math.pi
    assert phi_min == 0.0
    assert phi_max == 0.25


def test_cap_to_theta_phi_bounds_expands_to_full_theta_when_cap_is_wide():
    theta_min, theta_max, phi_min, phi_max = lpcb.cap_to_theta_phi_bounds(
        center_theta=0.4,
        center_phi=math.pi / 3.0,
        alpha=math.pi / 2.0,
    )

    assert theta_min == 0.0
    assert theta_max == 2.0 * math.pi
    assert phi_min == 0.0
    np.testing.assert_allclose(phi_max, 5.0 * math.pi / 6.0)