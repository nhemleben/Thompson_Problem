import math
import sys
from pathlib import Path

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
    candidate = [
        (0.0, 0.0),
        ( 0.0, 2.0 * math.pi / 3.0),
        ( 0.0, 4.0 * math.pi / 3.0),
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

    assert zone.proved
    assert zone.half_width > 0.0
    assert zone.attempts > 0
