import argparse
import ast
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from intvalpy import Interval

from LDL_interval import interval_ldlt, validate_interval_ldlt_result
from inital_part import initial_cell
from search import _cell_charts
from taylor_search import build_taylor_model as _build_taylor_model_base
from taylor_search import _set_iv_dps, _wrap_model_for_charts


def build_taylor_model(n: int, initial_cell_mode: str = "non-antipodal"):
    """Build a Taylor model and wrap it for the coordinate charts of the chosen mode."""

    base_model = _build_taylor_model_base(int(n))
    root = initial_cell(int(n), mode=initial_cell_mode)
    charts = _cell_charts(root)
    return _wrap_model_for_charts(base_model, charts)


@dataclass
class LocalMinCheckResult:
    ok: bool
    max_abs_gradient: float
    min_eig_hessian: float
    reason: str


@dataclass
class ExclusionZoneResult:
    proved: bool
    half_width: float
    attempts: int
    one_sided_lower_boundary_warning: bool = False
    one_sided_positive_indices: tuple[int, ...] = ()


def _parse_candidate(candidate_text: str) -> list[tuple[float, float]]:
    text = candidate_text.strip()

    try:
        obj = ast.literal_eval(text)
    except Exception:
        safe_globals = {"__builtins__": {}}
        safe_locals = {"np": np, "math": math, "pi": math.pi}
        obj = eval(text, safe_globals, safe_locals)  # noqa: S307

    if not isinstance(obj, (list, tuple)):
        raise ValueError("Candidate must be a list/tuple of (theta, phi) pairs")

    candidate: list[tuple[float, float]] = []
    for i, item in enumerate(obj):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Candidate item {i} must be a pair (theta, phi)")

        theta = float(item[0])
        phi = float(item[1])
        candidate.append((theta, phi))

    return candidate


def _candidate_to_flat(candidate: list[tuple[float, float]]) -> np.ndarray:
    flat: list[float] = []
    for theta, phi in candidate:
        flat.append(float(theta))
        flat.append(float(phi))
    return np.asarray(flat, dtype=float)


def _free_variable_mask(n: int, initial_cell_mode: str = "non-antipodal") -> list[bool]:
    root = initial_cell(n, mode=initial_cell_mode)
    mask: list[bool] = []
    for particle in root.particle_ranges:
        fixed_values = particle.fixed
        if isinstance(fixed_values, bool):
            fixed_values = [fixed_values] * len(particle.bounds)

        for fixed in fixed_values:
            mask.append(not bool(fixed))
    return mask


def _variable_bounds(n: int, initial_cell_mode: str = "non-antipodal") -> tuple[np.ndarray, np.ndarray]:
    root = initial_cell(n, mode=initial_cell_mode)
    lo: list[float] = []
    hi: list[float] = []

    for particle in root.particle_ranges:
        lo.append(float(particle.bounds[0].lo))
        hi.append(float(particle.bounds[0].hi))
        lo.append(float(particle.bounds[1].lo))
        hi.append(float(particle.bounds[1].hi))

    return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)


def _iv_from_bounds(lo: float, hi: float) -> Any:
    from mpmath import iv

    return iv.mpf((float(lo), float(hi)))


def _to_float_mid(value: Any) -> float:
    if hasattr(value, "a") and hasattr(value, "b"):
        return 0.5 * (float(value.a) + float(value.b))
    return float(value)


def _to_intvalpy(value: Any) -> Interval:
    if hasattr(value, "a") and hasattr(value, "b"):
        return Interval(float(value.a), float(value.b))
    v = float(value)
    return Interval(v, v)


def _build_interval_variables(
    center_flat: np.ndarray,
    free_mask: list[bool],
    h: float,
    lo_bounds: np.ndarray,
    hi_bounds: np.ndarray,
    one_sided_positive_indices: set[int] | None = None,
) -> list[Any]:
    intervals: list[Any] = []
    one_sided_positive_indices = one_sided_positive_indices or set()

    for i, x in enumerate(center_flat):
        if free_mask[i]:
            if i in one_sided_positive_indices:
                lo = max(lo_bounds[i], x)
                hi = min(hi_bounds[i], x + h)
            else:
                lo = max(lo_bounds[i], x - h)
                hi = min(hi_bounds[i], x + h)

            if hi < lo:
                return []
            intervals.append(_iv_from_bounds(lo, hi))
        else:
            intervals.append(_iv_from_bounds(x, x))

    return intervals


def _extract_free_interval_hessian(hessian: np.ndarray, free_indices: list[int]) -> list[list[Interval]]:
    matrix: list[list[Interval]] = []

    for i in free_indices:
        row: list[Interval] = []
        for j in free_indices:
            row.append(_to_intvalpy(hessian[i, j]))
        matrix.append(row)

    return matrix


def _point_hessian_numeric(hessian: np.ndarray, free_indices: list[int]) -> np.ndarray:
    n = len(free_indices)
    H = np.zeros((n, n), dtype=float)

    for r, i in enumerate(free_indices):
        for c, j in enumerate(free_indices):
            H[r, c] = _to_float_mid(hessian[i, j])

    H = 0.5 * (H + H.T)
    return H


def check_local_minimum(
    model,
    center_flat: np.ndarray,
    free_indices: list[int],
    gradient_tol: float,
    min_eig_tol: float,
) -> LocalMinCheckResult:
    gradient = model.gradient(center_flat)
    grad_free = np.asarray([_to_float_mid(gradient[i]) for i in free_indices], dtype=float)
    max_abs_gradient = float(np.max(np.abs(grad_free))) if grad_free.size else 0.0

    point_hessian = model.hessian(center_flat)
    H = _point_hessian_numeric(point_hessian, free_indices)
    eigenvalues = np.linalg.eigvalsh(H)
    min_eig = float(np.min(eigenvalues)) if eigenvalues.size else 0.0

    if max_abs_gradient > gradient_tol:
        return LocalMinCheckResult(
            ok=False,
            max_abs_gradient=max_abs_gradient,
            min_eig_hessian=min_eig,
            reason=(
                f"Gradient stationarity check failed: max |grad_free|={max_abs_gradient:.3e} "
                f"> tolerance={gradient_tol:.3e}"
            ),
        )

    if min_eig <= min_eig_tol:
        return LocalMinCheckResult(
            ok=False,
            max_abs_gradient=max_abs_gradient,
            min_eig_hessian=min_eig,
            reason=(
                f"Point Hessian is not strictly positive: min eigenvalue={min_eig:.3e} "
                f"<= tolerance={min_eig_tol:.3e}"
            ),
        )

    return LocalMinCheckResult(
        ok=True,
        max_abs_gradient=max_abs_gradient,
        min_eig_hessian=min_eig,
        reason="Candidate passed local-minimum checks",
    )


def _hessian_box_is_pd(model, interval_variables: list[Any], free_indices: list[int], validate: bool) -> bool:
    H_interval = model.hessian(interval_variables)
    H_free = _extract_free_interval_hessian(H_interval, free_indices)

    ldlt_result = interval_ldlt(H_free)
    if ldlt_result is None:
        return False

    if validate:
        ok, _ = validate_interval_ldlt_result(H_free, ldlt_result)
        return ok

    return True


def prove_exclusion_zone(
    model,
    center_flat: np.ndarray,
    free_mask: list[bool],
    lo_bounds: np.ndarray,
    hi_bounds: np.ndarray,
    initial_h: float,
    growth: float,
    tol: float,
    max_iter: int,
    validate_ldlt: bool,
) -> ExclusionZoneResult:
    free_indices = [i for i, free in enumerate(free_mask) if free]
    if not free_indices:
        return ExclusionZoneResult(proved=False, half_width=0.0, attempts=0)

    distances_to_boundary: list[float] = []
    one_sided_positive_indices: set[int] = set()
    boundary_eps = max(float(tol), 1e-14)

    for i in free_indices:
        x = center_flat[i]
        left = x - lo_bounds[i]
        right = hi_bounds[i] - x

        # If the candidate sits on the lower boundary, permit a one-sided
        # positive-direction interval [x, x+h] for this coordinate.
        if left <= boundary_eps and right > boundary_eps:
            one_sided_positive_indices.add(i)
            distances_to_boundary.append(right)
        else:
            # Otherwise keep centered-box semantics.
            distances_to_boundary.append(min(left, right))

    warning_one_sided = bool(one_sided_positive_indices)

    max_reachable_h = max(0.0, min(distances_to_boundary))
    if max_reachable_h <= 0.0:
        return ExclusionZoneResult(
            proved=False,
            half_width=0.0,
            attempts=0,
            one_sided_lower_boundary_warning=warning_one_sided,
            one_sided_positive_indices=tuple(sorted(one_sided_positive_indices)),
        )

    h = min(initial_h, max_reachable_h)
    attempts = 0

    while h > tol:
        attempts += 1
        interval_variables = _build_interval_variables(
            center_flat,
            free_mask,
            h,
            lo_bounds,
            hi_bounds,
            one_sided_positive_indices,
        )
        if not interval_variables:
            break
        if _hessian_box_is_pd(model, interval_variables, free_indices, validate_ldlt):
            break
        h *= 0.5
    else:
        return ExclusionZoneResult(
            proved=False,
            half_width=0.0,
            attempts=attempts,
            one_sided_lower_boundary_warning=warning_one_sided,
            one_sided_positive_indices=tuple(sorted(one_sided_positive_indices)),
        )

    h_low = h
    h_high = min(max_reachable_h, h * growth)

    while h_high < max_reachable_h:
        attempts += 1
        interval_variables = _build_interval_variables(
            center_flat,
            free_mask,
            h_high,
            lo_bounds,
            hi_bounds,
            one_sided_positive_indices,
        )
        if not interval_variables:
            break
        if _hessian_box_is_pd(model, interval_variables, free_indices, validate_ldlt):
            h_low = h_high
            h_high = min(max_reachable_h, h_high * growth)
        else:
            break

    for _ in range(max_iter):
        if h_high - h_low <= tol:
            break

        h_mid = 0.5 * (h_low + h_high)
        attempts += 1
        interval_variables = _build_interval_variables(
            center_flat,
            free_mask,
            h_mid,
            lo_bounds,
            hi_bounds,
            one_sided_positive_indices,
        )
        if not interval_variables:
            break

        if _hessian_box_is_pd(model, interval_variables, free_indices, validate_ldlt):
            h_low = h_mid
        else:
            h_high = h_mid

    return ExclusionZoneResult(
        proved=True,
        half_width=h_low,
        attempts=attempts,
        one_sided_lower_boundary_warning=warning_one_sided,
        one_sided_positive_indices=tuple(sorted(one_sided_positive_indices)),
    )


def _format_zone(
    candidate: list[tuple[float, float]],
    free_mask: list[bool],
    h: float,
    one_sided_positive_indices: set[int] | None = None,
) -> str:
    lines = []
    flat = _candidate_to_flat(candidate)
    one_sided_positive_indices = one_sided_positive_indices or set()

    for p in range(len(candidate)):
        theta_idx = 2 * p
        phi_idx = theta_idx + 1

        theta = flat[theta_idx]
        phi = flat[phi_idx]

        if free_mask[theta_idx]:
            if theta_idx in one_sided_positive_indices:
                theta_text = f"[{theta:.12g}, {theta + h:.12g}]"
            else:
                theta_text = f"[{theta - h:.12g}, {theta + h:.12g}]"
        else:
            theta_text = f"fixed {theta:.12g}"

        if free_mask[phi_idx]:
            if phi_idx in one_sided_positive_indices:
                phi_text = f"[{phi:.12g}, {phi + h:.12g}]"
            else:
                phi_text = f"[{phi - h:.12g}, {phi + h:.12g}]"
        else:
            phi_text = f"fixed {phi:.12g}"

        lines.append(f"  p{p}: theta {theta_text}, phi {phi_text}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Given a candidate Thomson configuration, check local minimality and "
            "prove an exclusion zone via interval LDLT positivity of the Hessian."
        )
    )
    parser.add_argument("--n", type=int, required=True, help="Number of particles")
    parser.add_argument(
        "--candidate",
        type=str,
        required=True,
        help=(
            "Candidate as Python/JSON-like list of (theta,phi) pairs, "
            "e.g. '[(0,0),(0,2.08),(2.94,2.08)]'"
        ),
    )
    parser.add_argument("--iv-dps", type=int, default=50, help="Interval precision")
    parser.add_argument(
        "--initial-cell-mode",
        type=str,
        default="non-antipodal",
        choices=["antipodal", "non-antipodal"],
        help="Initial-cell chart mode used to interpret candidate coordinates",
    )
    parser.add_argument(
        "--gradient-tol",
        type=float,
        default=1e-8,
        help="Tolerance for max absolute free-variable gradient",
    )
    parser.add_argument(
        "--min-eig-tol",
        type=float,
        default=1e-10,
        help="Strict positivity tolerance for Hessian eigenvalues at the candidate",
    )
    parser.add_argument("--h0", type=float, default=1e-4, help="Initial half-width")
    parser.add_argument(
        "--growth",
        type=float,
        default=2.0,
        help="Growth factor while expanding a valid zone",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-12,
        help="Binary-search tolerance for zone half-width",
    )
    parser.add_argument("--max-iter", type=int, default=80, help="Maximum bisection steps")
    parser.add_argument(
        "--validate-ldlt",
        action="store_true",
        help="Run structural LDLT validation after successful decomposition",
    )
    args = parser.parse_args()

    candidate = _parse_candidate(args.candidate)
    if len(candidate) != args.n:
        raise ValueError(
            f"Candidate length ({len(candidate)}) does not match n ({args.n})"
        )

    _set_iv_dps(int(args.iv_dps))
    model = build_taylor_model(int(args.n), initial_cell_mode=str(args.initial_cell_mode))

    center_flat = _candidate_to_flat(candidate)
    free_mask = _free_variable_mask(int(args.n), initial_cell_mode=str(args.initial_cell_mode))
    free_indices = [i for i, free in enumerate(free_mask) if free]
    lo_bounds, hi_bounds = _variable_bounds(int(args.n), initial_cell_mode=str(args.initial_cell_mode))

    local_min = check_local_minimum(
        model=model,
        center_flat=center_flat,
        free_indices=free_indices,
        gradient_tol=float(args.gradient_tol),
        min_eig_tol=float(args.min_eig_tol),
    )

    print("Local Minimum Check")
    print(f"  ok: {local_min.ok}")
    print(f"  max |grad_free|: {local_min.max_abs_gradient:.6e}")
    print(f"  min eig(H_free): {local_min.min_eig_hessian:.6e}")
    print(f"  reason: {local_min.reason}")

    if not local_min.ok:
        print("\nNo exclusion zone attempted because local-minimum checks failed.")
        return

    zone = prove_exclusion_zone(
        model=model,
        center_flat=center_flat,
        free_mask=free_mask,
        lo_bounds=lo_bounds,
        hi_bounds=hi_bounds,
        initial_h=float(args.h0),
        growth=float(args.growth),
        tol=float(args.tol),
        max_iter=int(args.max_iter),
        validate_ldlt=bool(args.validate_ldlt),
    )

    print("\nExclusion Zone Result")
    print(f"  proved: {zone.proved}")
    print(f"  half-width h: {zone.half_width:.12e}")
    print(f"  LDLT checks: {zone.attempts}")
    if zone.one_sided_lower_boundary_warning:
        print(
            "  warning: one-sided positive-direction zone used for free variables "
            f"on lower boundary indices {list(zone.one_sided_positive_indices)}"
        )

    if zone.proved and zone.half_width > 0.0:
        print("\nCertified parameter box (free variables vary, fixed variables remain fixed):")
        print(
            _format_zone(
                candidate,
                free_mask,
                zone.half_width,
                set(zone.one_sided_positive_indices),
            )
        )
        print(
            "\nInterpretation: the interval Hessian is positive definite in this box, "
            "so the energy is strictly convex there. Combined with stationarity at the "
            "candidate, this certifies a unique local minimizer within that region."
        )
    else:
        print(
            "\nCould not certify a positive-definite Hessian box around the candidate "
            "with current settings."
        )


if __name__ == "__main__":
    main()
