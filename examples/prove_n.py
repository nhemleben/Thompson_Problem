import argparse
import csv
import math
import multiprocessing as mp
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

from exsclusion_zone import (
	_candidate_to_flat,
	_free_variable_mask,
	_variable_bounds,
	build_taylor_model as build_zone_model,
	check_local_minimum,
	prove_exclusion_zone,
)
from inital_part import initial_cell
from known_optimal import KNOWN_SOLUTIONS, spherical_configuration
from partition import split_with_index
from search import (
	_cell_charts,
	_center_config,
	_min_separation_state_from_cell,
	_min_separation_state_from_parent,
	_ordered_theta_possible,
)
from taylor_search import (
	_cell_taylor_metadata,
	_cascading_taylor_lower_bound,
	_cell_parameter_volume,
	_set_iv_dps,
	TaylorModel,
	_wrap_model_for_charts,
	build_taylor_model,
)
from bound import d_min as d_min_bound
from energy import thompson_energy
from exsclusion_zone_depth_check import positive_definite_boundary_expansion


_WORKER_MODEL = None
_WORKER_LIPSHITZ_ONLY = False


def _lipshitz_only_for_depth(
	depth: int,
	lipshitz_only: bool,
	lipshitz_only_depth_bound: int | None,
) -> bool:
	if lipshitz_only_depth_bound is not None:
		return depth <= lipshitz_only_depth_bound

	return bool(lipshitz_only)


def _child_ordered_theta_possible_from_parent(
	parent_cell,
	child_cell,
	split_particle_index,
	epsilon=1e-15,
):
	if split_particle_index is None:
		return _ordered_theta_possible(child_cell, epsilon=epsilon)

	# Ordering constraints apply only to particles from index 2 onward.
	if split_particle_index < 2:
		return True

	parent_theta = parent_cell.particle_ranges[split_particle_index].bounds[0]
	child_theta = child_cell.particle_ranges[split_particle_index].bounds[0]

	# If theta bounds did not change, ordering feasibility is inherited.
	if parent_theta.lo == child_theta.lo and parent_theta.hi == child_theta.hi:
		return True

	return _ordered_theta_possible(child_cell, epsilon=epsilon)

def _fixed_indices_from_cell(cell) -> tuple[int, ...]:
	fixed_indices: list[int] = []
	for particle_index, particle_range in enumerate(cell.particle_ranges):
		for local_dim, is_fixed in enumerate(particle_range.fixed):
			if is_fixed:
				fixed_indices.append(2 * particle_index + local_dim)
	return tuple(fixed_indices)


def _wrap_model_with_fixed_gradient_zeroing(base_model: TaylorModel, cell) -> TaylorModel:
	fixed_indices = _fixed_indices_from_cell(cell)

	def energy(values):
		return base_model.energy(values)

	def gradient(values):
		gradient_values = np.asarray(base_model.gradient(values), dtype=object).reshape(-1).copy()
		for index in fixed_indices:
			gradient_values[index] = 0.0
		return gradient_values

	def hessian(values):
		return base_model.hessian(values)

	def third_derivative(values):
		return base_model.third_derivative(values)

	return TaylorModel(
		energy=energy,
		gradient=gradient,
		hessian=hessian,
		third_derivative=third_derivative,
		third_derivative_usable=base_model.third_derivative_usable,
	)


def _init_lb_worker(n: int, iv_dps: int, initial_cell_mode: str, lipshitz_only: bool):
	global _WORKER_MODEL, _WORKER_LIPSHITZ_ONLY
	_set_iv_dps(int(iv_dps))
	root = initial_cell(int(n), mode=str(initial_cell_mode))
	charts = _cell_charts(root)
	base_model = build_taylor_model(int(n))
	chart_model = _wrap_model_for_charts(base_model, charts)
	_WORKER_MODEL = _wrap_model_with_fixed_gradient_zeroing(chart_model, root)
	_WORKER_LIPSHITZ_ONLY = bool(lipshitz_only)


def _worker_cell_lb(task):
	global _WORKER_MODEL, _WORKER_LIPSHITZ_ONLY
	if len(task) == 2:
		cell, best_known = task
		lipshitz_only = _WORKER_LIPSHITZ_ONLY
	else:
		cell, best_known, lipshitz_only = task
	if _WORKER_MODEL is None:
		raise RuntimeError("Lower-bound worker model not initialized")
	lb, _, _ = _cascading_taylor_lower_bound(
		cell,
		_WORKER_MODEL,
		best_energy=float(best_known),
		lipshitz_only=bool(lipshitz_only),
	)
	return lb


def _worker_cell_lb_batch(task):
	global _WORKER_MODEL, _WORKER_LIPSHITZ_ONLY
	if len(task) == 2:
		cells, best_known = task
		lipshitz_only = _WORKER_LIPSHITZ_ONLY
	else:
		cells, best_known, lipshitz_only = task
	if _WORKER_MODEL is None:
		raise RuntimeError("Lower-bound worker model not initialized")

	lbs = []
	for cell in cells:
		lb, _, _ = _cascading_taylor_lower_bound(
			cell,
			_WORKER_MODEL,
			best_energy=float(best_known),
			lipshitz_only=bool(lipshitz_only),
		)
		lbs.append(lb)

	return lbs


def _evaluate_frontier_lbs(
	cells: list,
	model,
	best_known: float,
	pool=None,
	metadata=None,
	lipshitz_only: bool = False,
):
	if not cells:
		return []

	if pool is None or len(cells) == 1:
		lbs = []
		for index, cell in enumerate(cells):
			precomputed = None
			if metadata is not None:
				precomputed = metadata[index]
			lb, _, _ = _cascading_taylor_lower_bound(
				cell,
				model,
				best_energy=best_known,
				precomputed=precomputed,
				lipshitz_only=lipshitz_only,
			)
			lbs.append(lb)
		return lbs

	tasks = [(cell, best_known) for cell in cells]
	worker_count = max(1, int(getattr(pool, "_processes", 1)))
	batch_size = max(1, len(tasks) // (worker_count * 4))
	batches = [
		(cells[start:start + batch_size], best_known, bool(lipshitz_only))
		for start in range(0, len(cells), batch_size)
	]
	batch_lbs = pool.map(_worker_cell_lb_batch, batches, chunksize=1)
	return [lb for lbs in batch_lbs for lb in lbs]


def _zone_interval_for_index(
	center_flat: np.ndarray,
	index: int,
	half_width: float,
	one_sided_positive_indices: set[int],
) -> tuple[float, float]:
	center_value = float(center_flat[index])
	if index in one_sided_positive_indices:
		return center_value, center_value + float(half_width)
	return center_value - float(half_width), center_value + float(half_width)


def _zone_bounds_for_free_indices(
	center_flat: np.ndarray,
	free_indices: list[int],
	half_width: float,
	one_sided_positive_indices: set[int],
) -> list[tuple[int, float, float]]:
	zone_bounds: list[tuple[int, float, float]] = []
	for index in free_indices:
		zone_lo, zone_hi = _zone_interval_for_index(
			center_flat,
			index,
			half_width,
			one_sided_positive_indices,
		)
		zone_bounds.append((index, zone_lo, zone_hi))
	return zone_bounds


def _cell_fully_inside_exclusion_zone(
	cell,
	zone_bounds: list[tuple[int, float, float]],
) -> bool:
	for index, zone_lo, zone_hi in zone_bounds:
		particle_index = index // 2
		local_dim = index % 2
		bounds = cell.particle_ranges[particle_index].bounds[local_dim]
		cell_lo = float(bounds.lo)
		cell_hi = float(bounds.hi)

		if cell_lo < zone_lo or cell_hi > zone_hi:
			return False

	return True


def _known_candidate_for_mode(n, initial_cell_mode: str) -> list[tuple[float, float]]:
	antipodal = str(initial_cell_mode).strip().lower() == "antipodal"
	return spherical_configuration(KNOWN_SOLUTIONS[n], antipodal=antipodal)


def _build_exclusion_zone(n: int, initial_cell_mode: str, iv_dps: int):
	_set_iv_dps(iv_dps)
	candidate = _known_candidate_for_mode(n, initial_cell_mode)
	model = build_zone_model(n, initial_cell_mode)
	center_flat = _candidate_to_flat(candidate)
	free_mask = _free_variable_mask(n, initial_cell_mode)
	free_indices = [i for i, flag in enumerate(free_mask) if flag]
	lo_bounds, hi_bounds = _variable_bounds(n, initial_cell_mode)

	local_min = check_local_minimum(
		model=model,
		center_flat=center_flat,
		free_indices=free_indices,
		gradient_tol=1e-10,
		min_eig_tol=1e-12,
	)
	if not local_min.ok:
		raise RuntimeError(f"Known n={n} candidate failed local minimum check: {local_min.reason}")

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
	if not zone.proved:
		raise RuntimeError(f"Failed to prove exclusion zone around known n={n} candidate")

	return candidate, center_flat, free_mask, zone


def _depth_report(depth: int, cells: list, best_energy: float):
	volume = sum(_cell_parameter_volume(cell) for cell in cells)
	print(
		f"depth={depth:2d} | active_cells={len(cells):8d} | total_volume={volume:.12e} | best={best_energy:.12e}"
	)


def _flatten_center_config(center_config: list[tuple[float, float]]) -> np.ndarray:
	flat: list[float] = []
	for theta, phi in center_config:
		flat.append(float(theta))
		flat.append(float(phi))
	return np.asarray(flat, dtype=float)


def _center_row_fields(center_config: list[tuple[float, float]]) -> dict[str, float]:
	fields: dict[str, float] = {}
	for index, (theta, phi) in enumerate(center_config):
		fields[f"theta_{index}"] = float(theta)
		fields[f"phi_{index}"] = float(phi)
	return fields


def _interval_midpoint(value) -> float:
	if hasattr(value, "a") and hasattr(value, "b"):
		return 0.5 * (float(value.a) + float(value.b))
	return float(value)


def _write_surviving_cells_csv(
	path: Path,
	depth: int,
	cells: list,
	model,
	charts: list[str],
	include_gradient: bool,
) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)

	if not cells:
		with path.open("w", newline="") as handle:
			writer = csv.writer(handle)
			writer.writerow(["depth", "cell_index", "volume", "center_energy"])
		return path

	fieldnames = ["depth", "cell_index"]
	for particle_index in range(len(cells[0].particle_ranges)):
		fieldnames.append(f"theta_{particle_index}")
		fieldnames.append(f"phi_{particle_index}")
	fieldnames.append("center_energy")
	fieldnames.append("volume")
	if include_gradient:
		fieldnames.append("gradient_norm")
		for component_index in range(2 * len(cells[0].particle_ranges)):
			fieldnames.append(f"gradient_{component_index}")

	with path.open("w", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()

		for cell_index, cell in enumerate(cells):
			center_config = _center_config(cell)
			center_flat = _flatten_center_config(center_config)
			gradient_mid = None
			if include_gradient:
				gradient = np.asarray(model.gradient(center_flat), dtype=object).reshape(-1)
				gradient_mid = np.asarray([_interval_midpoint(component) for component in gradient], dtype=float)
			center_energy = float(thompson_energy(center_config, charts=charts))
			row = {
				"depth": depth,
				"cell_index": cell_index,
				"center_energy": center_energy,
				"volume": _cell_parameter_volume(cell),
			}
			if include_gradient and gradient_mid is not None:
				row["gradient_norm"] = float(np.linalg.norm(gradient_mid))
			row.update(_center_row_fields(center_config))
			if include_gradient and gradient_mid is not None:
				for component_index, component_value in enumerate(gradient_mid):
					row[f"gradient_{component_index}"] = float(component_value)
			writer.writerow(row)

	return path


def _write_active_cells_checkpoint(
	path: Path,
	*,
	n: int,
	depth: int,
	search_depth: int,
	initial_cell_mode: str,
	iv_dps: int,
	use_min_separation: bool,
	d_min: float | None,
	alpha_min: float | None,
	poss_def_boundary_check_level: int | None,
	parallel_workers: int,
	best_known: float,
	candidate: list[tuple[float, float]],
	zone_half_width: float,
	one_sided_positive_indices: set[int],
	active_cells: list,
) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	payload = {
		"version": 1,
		"created_utc": datetime.now(timezone.utc).isoformat(),
		"state": {
			"n": int(n),
			"depth": int(depth),
			"search_depth": int(search_depth),
			"initial_cell_mode": str(initial_cell_mode),
			"iv_dps": int(iv_dps),
			"use_min_separation": bool(use_min_separation),
			"d_min": None if d_min is None else float(d_min),
			"alpha_min": None if alpha_min is None else float(alpha_min),
			"poss_def_boundary_check_level": poss_def_boundary_check_level,
			"parallel_workers": int(parallel_workers),
			"best_known": float(best_known),
			"candidate": [(float(theta), float(phi)) for theta, phi in candidate],
			"zone_half_width": float(zone_half_width),
			"one_sided_positive_indices": sorted(int(i) for i in one_sided_positive_indices),
		},
		"active_cells": active_cells,
	}

	with path.open("wb") as handle:
		pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return path


def run_depth_limited_proof(
	n: int,
	search_depth: int,
	iv_dps: int,
	use_min_separation: bool,
	lipshitz_only: bool,
	lipshitz_only_depth_bound: int | None,
	d_min: float | None,
	alpha_min: float | None,
	poss_def_boundary_check_level: int | None,
	initial_cell_mode: str,
	parallel_workers: int,
	csv_path: str | None,
	csv_include_gradient: bool,
	use_min_sep_depth_limit: int | None,
):
	candidate, center_flat, free_mask, zone = _build_exclusion_zone(
		n=n,
		initial_cell_mode=initial_cell_mode,
		iv_dps=iv_dps,
	)

	_set_iv_dps(iv_dps)
	root = initial_cell(n, mode=initial_cell_mode)
	charts = _cell_charts(root)
	best_known = float(thompson_energy(candidate, charts=charts))
	if not math.isfinite(best_known):
		raise RuntimeError(
			"Known candidate energy is non-finite. Check candidate/chart alignment "
			f"for mode={initial_cell_mode}."
		)
	one_sided_positive_indices = set(zone.one_sided_positive_indices)
	free_indices = [i for i, is_free in enumerate(free_mask) if is_free]
	zone_bounds = _zone_bounds_for_free_indices(
		center_flat,
		free_indices,
		zone.half_width,
		one_sided_positive_indices,
	)
	model = _wrap_model_with_fixed_gradient_zeroing(
		_wrap_model_for_charts(build_taylor_model(n), charts),
		root,
	)
	use_min_separation = use_min_separation or (d_min is not None) or (alpha_min is not None)
	if use_min_separation and d_min is None:
		d_min = float(d_min_bound(n))
	if poss_def_boundary_check_level is not None and int(poss_def_boundary_check_level) <= 0:
		raise ValueError("--poss-def-boundary-check-level must be a positive integer")
	if lipshitz_only_depth_bound is not None and int(lipshitz_only_depth_bound) < 0:
		raise ValueError("--lipshitz-only-depth-bound must be >= 0")
	boundary_pd_period = int(poss_def_boundary_check_level) if poss_def_boundary_check_level is not None else None
	min_sep_cos_alpha = float(np.cos(alpha_min)) if alpha_min is not None else None
	min_sep_d_sq = float(d_min) * float(d_min) if d_min is not None else None
	pool = None
	worker_count = max(1, int(parallel_workers))
	if worker_count > 1:
		pool = mp.Pool(
			processes=worker_count,
			initializer=_init_lb_worker,
			initargs=(int(n), int(iv_dps), str(initial_cell_mode), bool(lipshitz_only)),
		)

	frontier = [(root, None)]
	pruned_by_zone = 0
	pruned_by_min_sep = 0
	pruned_by_boundary_pd = 0
	pruned_by_bound = 0
	boundary_pd_done = False

	print("Known-candidate certification")
	print(f"  n: {n}")
	print(f"  initial_cell_mode: {initial_cell_mode}")
	print(f"  known energy: {best_known:.12e}")
	print(f"  exclusion half-width: {zone.half_width:.12e}")
	print(f"  exclusion attempts: {zone.attempts}")
	print(f"  lb workers: {worker_count}")
	print(f"  use_min_separation: {use_min_separation}")
	print(f"  lipshitz_only: {lipshitz_only}")
	print(f"  lipshitz_only_depth_bound: {lipshitz_only_depth_bound}")
	if d_min is not None:
		print(f"  d_min: {d_min:.12e}")
	if alpha_min is not None:
		print(f"  alpha_min: {alpha_min:.12e}")
	print(f"  poss_def_boundary_check_level: {poss_def_boundary_check_level}")
	print(f"  csv_include_gradient: {csv_include_gradient}")
	if zone.one_sided_lower_boundary_warning:
		print("  warning: one-sided lower-boundary exclusion used")
	print("")
	print("Depth-by-depth branch/prune report")
	depth_csv_path = (
		Path(csv_path)
		if csv_path
		else Path(__file__).resolve().parent / "csvs" / f"prove_n_survivors_n{n}_{initial_cell_mode}_depth{search_depth}.csv"
	)

	try:
		for depth in range(search_depth + 1):
			if not frontier:
				print(f"terminated early at depth={depth}: all cells died")
				break

			zone_survivors = []
			for cell, pair_state in frontier:
				if _cell_fully_inside_exclusion_zone(
					cell,
					zone_bounds,
				):
					pruned_by_zone += 1
					continue
				zone_survivors.append((cell, pair_state))

			min_sep_survivors = zone_survivors
			if not use_min_separation or (use_min_sep_depth_limit is not None and depth > use_min_sep_depth_limit):
				pass
			else:
				min_sep_survivors = []
				for cell, pair_state in zone_survivors:
					if pair_state is None:
						min_sep_ok, child_state = _min_separation_state_from_cell(
							cell,
							d_min=d_min,
							alpha_min=alpha_min,
							cos_alpha_min=min_sep_cos_alpha,
							d_min_sq=min_sep_d_sq,
						)
					else:
						min_sep_ok, child_state = True, pair_state
					if not min_sep_ok:
						pruned_by_min_sep += 1
						continue
					min_sep_survivors.append((cell, child_state))

			min_sep_cells = [cell for cell, _ in min_sep_survivors]

			cell_metadata = None
			if pool is None:
				cell_metadata = [_cell_taylor_metadata(cell) for cell in min_sep_cells]

			depth_lipshitz_only = _lipshitz_only_for_depth(
				depth,
				lipshitz_only,
				lipshitz_only_depth_bound,
			)

			lbs = _evaluate_frontier_lbs(
				min_sep_cells,
				model,
				best_known,
				pool=pool,
				metadata=cell_metadata,
				lipshitz_only=depth_lipshitz_only,
			)

			active_at_depth = []
			for (cell, pair_state), lb in zip(min_sep_survivors, lbs):
				if lb is None or lb >= best_known:
					pruned_by_bound += 1
					continue
				active_at_depth.append((cell, pair_state))

			active_cells = [cell for cell, _ in active_at_depth]

			if (
				not boundary_pd_done
				and boundary_pd_period is not None
				and depth % boundary_pd_period == 0
				and active_cells
			):
				state_by_cell_id = {id(cell): pair_state for cell, pair_state in active_at_depth}
				kept_after_pd, pruned_pd_cells, pd_events = positive_definite_boundary_expansion(
					active_cells,
					model,
					center_flat,
					free_mask,
					zone.half_width,
					one_sided_positive_indices,
					validate_ldlt=True,
				)
				if pruned_pd_cells:
					for event in pd_events:
						print(
							f"positive definite boundary expansion termination: depth={depth}, "
							f"cell_index={event['cell_index']}"
						)
					pruned_by_boundary_pd += len(pruned_pd_cells)
					active_at_depth = [
						(cell, state_by_cell_id.get(id(cell)))
						for cell in kept_after_pd
					]
					active_cells = [cell for cell, _ in active_at_depth]
				boundary_pd_done = True

			_depth_report(depth, active_cells, best_known)

			if depth == search_depth:
				print(f"terminated at depth limit: search_depth={search_depth}")
				written_csv = _write_surviving_cells_csv(
					depth_csv_path,
					depth,
					active_cells,
					model,
					charts,
					csv_include_gradient,
				)
				checkpoint_path = written_csv.with_suffix(".pkl")
				written_checkpoint = _write_active_cells_checkpoint(
					checkpoint_path,
					n=n,
					depth=depth,
					search_depth=search_depth,
					initial_cell_mode=initial_cell_mode,
					iv_dps=iv_dps,
					use_min_separation=use_min_separation,
					d_min=d_min,
					alpha_min=alpha_min,
					poss_def_boundary_check_level=poss_def_boundary_check_level,
					parallel_workers=parallel_workers,
					best_known=best_known,
					candidate=candidate,
					zone_half_width=zone.half_width,
					one_sided_positive_indices=one_sided_positive_indices,
					active_cells=active_cells,
				)
				print(f"wrote surviving-cell csv: {written_csv}")
				print(f"wrote active-cell checkpoint: {written_checkpoint}")
				frontier = active_at_depth
				break

			next_frontier = []
			for cell, pair_state in active_at_depth:
				children, split_particle_index = split_with_index(cell)
				if not children:
					continue
				for child in children:
					if not _child_ordered_theta_possible_from_parent(
						cell,
						child,
						split_particle_index,
					):
						continue
					child_state = pair_state
					if use_min_separation:
						child_ok, child_state = _min_separation_state_from_parent(
							parent_state=pair_state,
							parent_cell=cell,
							child_cell=child,
							split_particle_index=split_particle_index,
							d_min=d_min,
							alpha_min=alpha_min,
							cos_alpha_min=min_sep_cos_alpha,
							d_min_sq=min_sep_d_sq,
						)
						if not child_ok:
							pruned_by_min_sep += 1
							continue
					next_frontier.append((child, child_state))

			frontier = next_frontier
		else:
			frontier = []
	finally:
		if pool is not None:
			pool.close()
			pool.join()

	print("")
	print("Summary")
	print(f"  remaining_active_cells: {len(frontier)}")
	print(f"  zone_pruned_cells: {pruned_by_zone}")
	print(f"  min_separation_pruned_cells: {pruned_by_min_sep}")
	print(f"  positive_definite_boundary_pruned_cells: {pruned_by_boundary_pd}")
	print(f"  bound_pruned_cells: {pruned_by_bound}")
	print(f"  final_remaining_volume: {sum(_cell_parameter_volume(cell) for cell, _ in frontier):.12e}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Prove optimality of the known n Thomson configuration by certifying "
			"an exclusion zone and running depth-limited Taylor branch/prune outside it."
		)
	)
	parser.add_argument("--n", type=int, default=3, help="Number of particles")
	parser.add_argument("--search-depth", type=int, default=10, help="Depth limit for branch/prune")
	parser.add_argument("--iv-dps", type=int, default=50, help="Interval precision")
	parser.add_argument("--use-min-separation", action="store_true", help="Enable geometric min-separation pruning")
	parser.add_argument(
		"--use-min-sep-depth-limit",
		type=int,
		default=None,
		help="Use min-separation pruning only through this depth (inclusive), then disable it",
	)
	parser.add_argument(
		"--lipshitz-only",
		action="store_true",
		help="Use only the Lipschitz stage in cascading Taylor lower bounds",
	)
	parser.add_argument(
		"--lipshitz-only-depth-bound",
		type=int,
		default=None,
		help="Use only Lipschitz stage through this depth (inclusive), then disable it",
	)
	parser.add_argument("--d-min", type=float, default=None, help="Optional explicit minimum pair distance")
	parser.add_argument("--alpha-min", type=float, default=None, help="Optional explicit minimum angular separation (radians)")
	parser.add_argument(
		"--poss-def-boundary-check-level",
		type=int,
		default=None,
		help="Depth level at which to prune border cells whose interval Hessian is positive definite",
	)
	parser.add_argument(
		"--parallel-workers",
		type=int,
		default=mp.cpu_count(),
		help="Number of worker processes for lower-bound evaluation (1 disables parallelism)",
	)
	parser.add_argument(
		"--initial-cell-mode",
		type=str,
		default="non-antipodal",
		choices=["non-antipodal", "antipodal"],
		help="Initial-cell chart mode",
	)
	parser.add_argument(
		"--csv-path",
		type=str,
		default=None,
		help="Optional CSV output path for surviving cells when the search stops at the depth limit",
	)
	parser.add_argument(
		"--csv-include-gradient",
		action="store_true",
		help="Include gradient norm/components in the output CSV (expensive for large frontiers)",
	)
	args = parser.parse_args()

	run_depth_limited_proof(
		n=int(args.n),
		search_depth=int(args.search_depth),
		use_min_separation=args.use_min_separation,
		lipshitz_only=bool(args.lipshitz_only),
		lipshitz_only_depth_bound=args.lipshitz_only_depth_bound,
		iv_dps=int(args.iv_dps),
		d_min=args.d_min,
		alpha_min=args.alpha_min,
		poss_def_boundary_check_level=args.poss_def_boundary_check_level,
		initial_cell_mode=str(args.initial_cell_mode),
		parallel_workers=int(args.parallel_workers),
		csv_path=args.csv_path,
		csv_include_gradient=bool(args.csv_include_gradient),
		use_min_sep_depth_limit=args.use_min_sep_depth_limit,
	)


if __name__ == "__main__":
	main()





