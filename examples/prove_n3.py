import argparse
import math
import multiprocessing as mp
import sys
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
from partition import split
from search import _cell_charts, _ordered_theta_possible
from taylor_search import (
	_cascading_taylor_lower_bound,
	_cell_parameter_volume,
	_set_iv_dps,
	_wrap_model_for_charts,
	build_taylor_model,
)
from energy import thompson_energy


_WORKER_MODEL = None


def _init_lb_worker(n: int, iv_dps: int, initial_cell_mode: str):
	global _WORKER_MODEL
	_set_iv_dps(int(iv_dps))
	root = initial_cell(int(n), mode=str(initial_cell_mode))
	charts = _cell_charts(root)
	base_model = build_taylor_model(int(n))
	_WORKER_MODEL = _wrap_model_for_charts(base_model, charts)


def _worker_cell_lb(task):
	global _WORKER_MODEL
	cell, best_known = task
	if _WORKER_MODEL is None:
		raise RuntimeError("Lower-bound worker model not initialized")
	lb, _, _ = _cascading_taylor_lower_bound(
		cell,
		_WORKER_MODEL,
		best_energy=float(best_known),
	)
	return lb


def _evaluate_frontier_lbs(cells: list, model, best_known: float, pool=None):
	if not cells:
		return []

	if pool is None or len(cells) == 1:
		lbs = []
		for cell in cells:
			lb, _, _ = _cascading_taylor_lower_bound(
				cell,
				model,
				best_energy=best_known,
			)
			lbs.append(lb)
		return lbs

	tasks = [(cell, best_known) for cell in cells]
	return pool.map(_worker_cell_lb, tasks)


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


def _cell_dimension_bounds(cell, index: int) -> tuple[float, float]:
	particle_index = index // 2
	local_dim = index % 2
	bounds = cell.particle_ranges[particle_index].bounds[local_dim]
	return float(bounds.lo), float(bounds.hi)


def _cell_fully_inside_exclusion_zone(
	cell,
	center_flat: np.ndarray,
	free_mask: list[bool],
	half_width: float,
	one_sided_positive_indices: set[int],
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
		cell_lo, cell_hi = _cell_dimension_bounds(cell, index)

		if cell_lo < zone_lo or cell_hi > zone_hi:
			return False

	return True


def _known_candidate_for_mode(initial_cell_mode: str) -> list[tuple[float, float]]:
	antipodal = str(initial_cell_mode).strip().lower() == "antipodal"
	return spherical_configuration(KNOWN_SOLUTIONS[3], antipodal=antipodal)


def _build_exclusion_zone(n: int, initial_cell_mode: str, iv_dps: int):
	if n != 3:
		raise ValueError("This proof driver currently targets n=3 only")

	_set_iv_dps(iv_dps)
	candidate = _known_candidate_for_mode(initial_cell_mode)
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
		raise RuntimeError(f"Known n=3 candidate failed local minimum check: {local_min.reason}")

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
		raise RuntimeError("Failed to prove exclusion zone around known n=3 candidate")

	return candidate, center_flat, free_mask, zone


def _depth_report(depth: int, cells: list, best_energy: float):
	volume = sum(_cell_parameter_volume(cell) for cell in cells)
	print(
		f"depth={depth:2d} | active_cells={len(cells):8d} | total_volume={volume:.12e} | best={best_energy:.12e}"
	)


def run_depth_limited_proof(
	n: int,
	search_depth: int,
	iv_dps: int,
	initial_cell_mode: str,
	parallel_workers: int,
):
	candidate, center_flat, free_mask, zone = _build_exclusion_zone(
		n=n,
		initial_cell_mode=initial_cell_mode,
		iv_dps=iv_dps,
	)
	best_known = float(thompson_energy(candidate))
	one_sided_positive_indices = set(zone.one_sided_positive_indices)

	_set_iv_dps(iv_dps)
	root = initial_cell(n, mode=initial_cell_mode)
	charts = _cell_charts(root)
	model = _wrap_model_for_charts(build_taylor_model(n), charts)
	pool = None
	worker_count = max(1, int(parallel_workers))
	if worker_count > 1:
		pool = mp.Pool(
			processes=worker_count,
			initializer=_init_lb_worker,
			initargs=(int(n), int(iv_dps), str(initial_cell_mode)),
		)

	frontier = [root]
	pruned_by_zone = 0
	pruned_by_bound = 0

	print("Known-candidate certification")
	print(f"  n: {n}")
	print(f"  initial_cell_mode: {initial_cell_mode}")
	print(f"  known energy: {best_known:.12e}")
	print(f"  exclusion half-width: {zone.half_width:.12e}")
	print(f"  exclusion attempts: {zone.attempts}")
	print(f"  lb workers: {worker_count}")
	if zone.one_sided_lower_boundary_warning:
		print("  warning: one-sided lower-boundary exclusion used")
	print("")
	print("Depth-by-depth branch/prune report")

	try:
		for depth in range(search_depth + 1):
			if not frontier:
				print(f"terminated early at depth={depth}: all cells died")
				break

			zone_survivors = []
			for cell in frontier:
				if _cell_fully_inside_exclusion_zone(
					cell,
					center_flat,
					free_mask,
					zone.half_width,
					one_sided_positive_indices,
				):
					pruned_by_zone += 1
					continue
				zone_survivors.append(cell)

			lbs = _evaluate_frontier_lbs(
				zone_survivors,
				model,
				best_known,
				pool=pool,
			)

			active_at_depth = []
			for cell, lb in zip(zone_survivors, lbs):
				if lb is None or lb >= best_known:
					pruned_by_bound += 1
					continue
				active_at_depth.append(cell)

			_depth_report(depth, active_at_depth, best_known)

			if depth == search_depth:
				print(f"terminated at depth limit: search_depth={search_depth}")
				frontier = active_at_depth
				break

			next_frontier = []
			for cell in active_at_depth:
				children = split(cell)
				if not children:
					continue
				next_frontier.extend(
					child
					for child in children
					if _ordered_theta_possible(child)
				)

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
	print(f"  bound_pruned_cells: {pruned_by_bound}")
	print(f"  final_remaining_volume: {sum(_cell_parameter_volume(cell) for cell in frontier):.12e}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Prove optimality of the known n=3 Thomson configuration by certifying "
			"an exclusion zone and running depth-limited Taylor branch/prune outside it."
		)
	)
	parser.add_argument("--n", type=int, default=3, help="Number of particles (currently only n=3 supported)")
	parser.add_argument("--search-depth", type=int, default=10, help="Depth limit for branch/prune")
	parser.add_argument("--iv-dps", type=int, default=50, help="Interval precision")
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
	args = parser.parse_args()

	run_depth_limited_proof(
		n=int(args.n),
		search_depth=int(args.search_depth),
		iv_dps=int(args.iv_dps),
		initial_cell_mode=str(args.initial_cell_mode),
		parallel_workers=int(args.parallel_workers),
	)


if __name__ == "__main__":
	main()





