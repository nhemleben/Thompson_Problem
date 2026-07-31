import math
import sys
from pathlib import Path

from numpy import pi
from known_optimal import KNOWN_SOLUTIONS, cartesian_to_spherical, spherical_configuration

sys.path.append(str(Path(__file__).resolve().parents[1]))

from exsclusion_zone import (
    _candidate_to_flat,
    _free_variable_mask,
    _variable_bounds,
    build_taylor_model,
    check_local_minimum,
    prove_exclusion_zone,
    _set_iv_dps,
)


def test_n2_antipodal():
    candidate = spherical_configuration(KNOWN_SOLUTIONS[2], antipodal=True)

    n = 2
    _set_iv_dps(40)
    model = build_taylor_model(n, "antipodal")

    center_flat = _candidate_to_flat(candidate)
    free_mask = _free_variable_mask(n, "antipodal")
    free_indices = [i for i, is_free in enumerate(free_mask) if is_free]
    lo_bounds, hi_bounds = _variable_bounds(n, "antipodal")

    local_min = check_local_minimum(
        model=model,
        center_flat=center_flat,
        free_indices=free_indices,
        gradient_tol=1e-10,
        min_eig_tol=1e-12,
    )

    assert local_min.ok, local_min.reason
    assert local_min.max_abs_gradient < 1e-10
    assert local_min.min_eig_hessian > 1e-12

    zone = prove_exclusion_zone(
        model=model,
        center_flat=center_flat,
        free_mask=free_mask,
        lo_bounds=lo_bounds,
        hi_bounds=hi_bounds,
        initial_h=1e-4,
        growth=2.0,
        tol=1e-12,
        max_iter=50,
        validate_ldlt=True,
    )

    assert zone.proved
    assert zone.half_width > 0.0
    assert zone.attempts > 0
    assert zone.one_sided_lower_boundary_warning
    assert len(zone.one_sided_positive_indices) > 0


def test_n3_not_local_min():
    # Thomson n=3 exact solution in this coordinate gauge.
    candidate = [
        (0.0, 0.0),
        (0.0,  math.pi / 3.0),
        (0.0, 2.0 * math.pi / 3.0),
    ]

    n = 3
    _set_iv_dps(40)
    model = build_taylor_model(n)

    center_flat = _candidate_to_flat(candidate)
    free_mask = _free_variable_mask(n)
    free_indices = [i for i, is_free in enumerate(free_mask) if is_free]
    lo_bounds, hi_bounds = _variable_bounds(n)

    local_min = check_local_minimum(
        model=model,
        center_flat=center_flat,
        free_indices=free_indices,
        gradient_tol=1e-10,
        min_eig_tol=1e-12,
    )

    assert not local_min.ok, local_min.reason
    assert local_min.max_abs_gradient > 1e-10
    assert local_min.min_eig_hessian < 1e-12


def test_n3_equilateral_triangle_local_minimum_and_boundary_zone_behavior_chart_failure():
    # Thomson n=3 exact solution in this coordinate gauge.
    candidate = [
        (0.0, 0.0),
        (0.0, 2.0 * math.pi / 3.0),
        (0.0, 4.0 * math.pi / 3.0),
    ]

    n = 3
    _set_iv_dps(40)
    model = build_taylor_model(n)

    center_flat = _candidate_to_flat(candidate)
    free_mask = _free_variable_mask(n)
    free_indices = [i for i, is_free in enumerate(free_mask) if is_free]
    lo_bounds, hi_bounds = _variable_bounds(n)

    local_min = check_local_minimum(
        model=model,
        center_flat=center_flat,
        free_indices=free_indices,
        gradient_tol=1e-10,
        min_eig_tol=1e-12,
    )

    assert local_min.ok, local_min.reason
    assert local_min.max_abs_gradient < 1e-10
    assert local_min.min_eig_hessian > 1e-12

    # Current exclusion-zone implementation requires a centered box in all free
    # variables. For this exact representative, theta_3 = pi lies on the chart
    # boundary, so a nonzero centered half-width cannot exist.
    zone = prove_exclusion_zone(
        model=model,
        center_flat=center_flat,
        free_mask=free_mask,
        lo_bounds=lo_bounds,
        hi_bounds=hi_bounds,
        initial_h=1e-4,
        growth=2.0,
        tol=1e-12,
        max_iter=50,
        validate_ldlt=True,
    )

    assert not zone.proved
    assert zone.half_width == 0.0
    assert zone.attempts == 0


def test_n3_equilateral_triangle_local_minimum_and_boundary_zone_behavior():
    # Thomson n=3 exact solution in this coordinate gauge.
    acosthird = math.acos(-1/3)
    candidate = [
        (0.0, 0.0),
        ( 0.0, 2*math.pi/3),
        ( math.pi, 2*math.pi/3),
    ]

    n = 3
    _set_iv_dps(40)
    model = build_taylor_model(n)

    center_flat = _candidate_to_flat(candidate)
    free_mask = _free_variable_mask(n)
    free_indices = [i for i, is_free in enumerate(free_mask) if is_free]
    lo_bounds, hi_bounds = _variable_bounds(n)

    local_min = check_local_minimum(
        model=model,
        center_flat=center_flat,
        free_indices=free_indices,
        gradient_tol=1e-10,
        min_eig_tol=1e-12,
    )

    assert local_min.ok, local_min.reason
    assert local_min.max_abs_gradient < 1e-10
    assert local_min.min_eig_hessian > 1e-12

    zone = prove_exclusion_zone(
        model=model,
        center_flat=center_flat,
        free_mask=free_mask,
        lo_bounds=lo_bounds,
        hi_bounds=hi_bounds,
        initial_h=1e-4,
        growth=2.0,
        tol=1e-12,
        max_iter=50,
        validate_ldlt=True,
    )

    assert zone.proved
    assert zone.half_width > 0.0
    assert zone.attempts > 0




def known_configurations_test(candidate,n, intial_cell_mode):
    _set_iv_dps(40)
    model = build_taylor_model(n, intial_cell_mode)

    center_flat = _candidate_to_flat(candidate)
    free_mask = _free_variable_mask(n, intial_cell_mode)
    free_indices = [i for i, is_free in enumerate(free_mask) if is_free]
    lo_bounds, hi_bounds = _variable_bounds(n, intial_cell_mode)

    local_min = check_local_minimum(
        model=model,
        center_flat=center_flat,
        free_indices=free_indices,
        gradient_tol=1e-10,
        min_eig_tol=1e-12,
    )

    assert local_min.ok, local_min.reason
    assert local_min.max_abs_gradient < 1e-10
    assert local_min.min_eig_hessian > 1e-12

    zone = prove_exclusion_zone(
        model=model,
        center_flat=center_flat,
        free_mask=free_mask,
        lo_bounds=lo_bounds,
        hi_bounds=hi_bounds,
        initial_h=1e-4,
        growth=2.0,
        tol=1e-12,
        max_iter=50,
        validate_ldlt=True,
    )

    assert zone.proved
    assert zone.half_width > 0.0
    assert zone.attempts > 0


def test_n2():
    n = 2
    intial_cell_mode = 'antipodal'
    antipodal = True
    candidate = spherical_configuration(KNOWN_SOLUTIONS[n], antipodal=antipodal)
    known_configurations_test(candidate,n,intial_cell_mode)

def test_n4():
    n = 4
    intial_cell_mode = 'non-antipodal'
    candidate = spherical_configuration(KNOWN_SOLUTIONS[n], antipodal=(intial_cell_mode=='antipodal'))
    known_configurations_test(candidate,n,intial_cell_mode)

def test_n5():
    n = 5
    intial_cell_mode = 'non-antipodal'
    candidate = spherical_configuration(KNOWN_SOLUTIONS[n], antipodal=(intial_cell_mode=='antipodal'))
    known_configurations_test(candidate,n,intial_cell_mode)


def test_n6():
    n = 6
    intial_cell_mode = 'antipodal'
    candidate = spherical_configuration(KNOWN_SOLUTIONS[n], antipodal=(intial_cell_mode=='antipodal'))
    known_configurations_test(candidate,n,intial_cell_mode)

def test_n7():
    n = 7
    intial_cell_mode = 'antipodal'
    candidate = spherical_configuration(KNOWN_SOLUTIONS[n], antipodal=(intial_cell_mode=='antipodal'))
    known_configurations_test(candidate,n,intial_cell_mode)
