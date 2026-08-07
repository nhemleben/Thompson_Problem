from __future__ import annotations

from typing import Any

import numpy as np
from mpmath import iv
from intvalpy import Interval

from LDL_interval import interval_ldlt, validate_interval_ldlt_result


def _iv_from_bounds(lo: float, hi: float):
	return iv.mpf((float(lo), float(hi)))  # type: ignore[arg-type]


def _to_intvalpy(value: Any) -> Any:
	if hasattr(value, "a") and hasattr(value, "b"):
		return Interval(float(value.a), float(value.b))

	v = float(value)
	return Interval(v, v)


def _free_indices_from_cell(cell) -> list[int]:
	free_indices: list[int] = []
	for particle_index, particle_range in enumerate(cell.particle_ranges):
		for local_dim, is_fixed in enumerate(particle_range.fixed):
			if not is_fixed:
				free_indices.append(2 * particle_index + local_dim)
	return free_indices


def _cell_intervals(cell) -> list[Any]:
	intervals: list[Any] = []
	for particle_range in cell.particle_ranges:
		for bounds in particle_range.bounds:
			intervals.append(_iv_from_bounds(bounds.lo, bounds.hi))
	return intervals


def _zone_interval_for_index(
	center_flat: np.ndarray,
	index: int,
	half_width: float,
	one_sided_positive_indices: set[int],
):
	center_value = float(center_flat[index])
	if index in one_sided_positive_indices:
		return center_value, center_value + float(half_width)
	return center_value - float(half_width), center_value + float(half_width)


def cell_fully_inside_exclusion_zone(
	cell,
	center_flat: np.ndarray,
	free_mask: list[bool],
	half_width: float,
	one_sided_positive_indices: set[int],
	epsilon: float = 1e-15,
) -> bool:
	for index, is_free in enumerate(free_mask):
		if not is_free:
			continue

		zone_lo, zone_hi = _zone_interval_for_index(
			center_flat,
			index,
			half_width,
			one_sided_positive_indices,
		)
		particle_index = index // 2
		local_dim = index % 2
		bounds = cell.particle_ranges[particle_index].bounds[local_dim]
		cell_lo = float(bounds.lo)
		cell_hi = float(bounds.hi)

		if cell_lo < zone_lo - epsilon or cell_hi > zone_hi + epsilon:
			return False

	return True


def cell_borders_exclusion_zone(
	cell,
	center_flat: np.ndarray,
	free_mask: list[bool],
	half_width: float,
	one_sided_positive_indices: set[int],
	epsilon: float = 1e-15,
) -> bool:
	if cell_fully_inside_exclusion_zone(
		cell,
		center_flat,
		free_mask,
		half_width,
		one_sided_positive_indices,
		epsilon=epsilon,
	):
		return False

	for index, is_free in enumerate(free_mask):
		if not is_free:
			continue

		zone_lo, zone_hi = _zone_interval_for_index(
			center_flat,
			index,
			half_width,
			one_sided_positive_indices,
		)
		particle_index = index // 2
		local_dim = index % 2
		bounds = cell.particle_ranges[particle_index].bounds[local_dim]
		cell_lo = float(bounds.lo)
		cell_hi = float(bounds.hi)

		if cell_hi < zone_lo - epsilon or cell_lo > zone_hi + epsilon:
			return False

	return True


def cell_hessian_is_positive_definite(model, cell, validate_ldlt: bool = True) -> bool:
	interval_variables = _cell_intervals(cell)
	free_indices = _free_indices_from_cell(cell)

	if not free_indices:
		return False

	hessian = model.hessian(interval_variables)
	free_hessian: list[list[Any]] = []
	for i in free_indices:
		row: list[Any] = []
		for j in free_indices:
			row.append(_to_intvalpy(hessian[i, j]))
		free_hessian.append(row)

	ldlt_result = interval_ldlt(free_hessian)
	if ldlt_result is None:
		return False

	if validate_ldlt:
		ok, _ = validate_interval_ldlt_result(free_hessian, ldlt_result)
		return ok

	return True


def positive_definite_boundary_expansion(
	cells,
	model,
	center_flat: np.ndarray,
	free_mask: list[bool],
	half_width: float,
	one_sided_positive_indices: set[int],
	validate_ldlt: bool = True,
	epsilon: float = 1e-15,
):
	kept_cells = []
	pruned_cells = []
	events = []

	for index, cell in enumerate(cells):
		if not cell_borders_exclusion_zone(
			cell,
			center_flat,
			free_mask,
			half_width,
			one_sided_positive_indices,
			epsilon=epsilon,
		):
			kept_cells.append(cell)
			continue

		if cell_hessian_is_positive_definite(model, cell, validate_ldlt=validate_ldlt):
			pruned_cells.append(cell)
			events.append(
				{
					"cell_index": index,
					"reason": "positive definite boundary expansion",
				}
			)
		else:
			kept_cells.append(cell)

	return kept_cells, pruned_cells, events