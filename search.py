import heapq
from itertools import count
import time
import multiprocessing as mp
import math 
import numpy as np

from visualizations.global_visualize import draw_global_search
from partition import *
from partition import _find_split_axis, _split_on_axis
from bound import *
from energy import *
from geometry import spherical_to_cart
from inital_part import initial_cell
from visualizations import visualize_parameter_mesh
from visualizations import visualize_final_minimum


def _ordered_theta_possible(cell, epsilon=1e-15):

    if len(cell.particle_ranges) <= 3:
        return True

    theta_bounds = [
        pr.bounds[0]
        for pr in cell.particle_ranges[2:]
    ]

    current_theta = theta_bounds[0].lo
    if current_theta > theta_bounds[0].hi:
        return False

    for bounds in theta_bounds[1:]:
        current_theta = max(bounds.lo, current_theta + epsilon)
        if current_theta > bounds.hi:
            return False

    return True


def _ordered_theta_center(config, epsilon=1e-12):

    if len(config) <= 3:
        return True

    thetas = [theta for theta, _ in config[2:]]
    return all(thetas[i] + epsilon < thetas[i + 1] for i in range(len(thetas) - 1))


def _center_config(cell):

    config=[]

    for theta,phi in (pr.bounds for pr in cell.particle_ranges):

        tc=(theta.lo+theta.hi)/2
        pc=(phi.lo+phi.hi)/2

        config.append(
            (tc,pc)
        )

    return config


def _cell_max_side_length(cell):

    max_side = 0.0

    for particle_range in cell.particle_ranges:
        for dim, bounds in enumerate(particle_range.bounds):
            if particle_range.fixed[dim]:
                continue

            side = bounds.hi - bounds.lo
            if side > max_side:
                max_side = side

    return max_side


def _build_even_mesh(root, side_length=0.1, cell_filter=None):

    if side_length <= 0:
        return [root]

    ready = []
    frontier = [root]
    tolerance = 1e-15

    while frontier:
        cell = frontier.pop()

        if cell_filter is not None and not cell_filter(cell):
            continue

        split_axis = _find_split_axis(cell)

        if split_axis is None:
            ready.append(cell)
            continue

        axis_i, axis_j = split_axis
        axis_bounds = cell.particle_ranges[axis_i].bounds[axis_j]
        axis_side = axis_bounds.hi - axis_bounds.lo

        if axis_side <= side_length + tolerance:
            ready.append(cell)
            continue

        children = _split_on_axis(cell, axis_i, axis_j)

        if not children:
            ready.append(cell)
            continue

        frontier.extend(children)

    return ready


def _respects_min_separation(config, d_min=None, alpha_min=None, epsilon=1e-15):

    if d_min is None and alpha_min is None:
        return True

    cartesian_points = []
    cos_alpha_min = None
    if alpha_min is not None:
        cos_alpha_min = np.cos(alpha_min)

    for theta, phi in config:
        point = spherical_to_cart(theta, phi)

        for prior_point in cartesian_points:
            if d_min is not None:
                if np.linalg.norm(point - prior_point) + epsilon < d_min:
                    return False

            if cos_alpha_min is not None:
                if float(np.dot(point, prior_point)) > cos_alpha_min + epsilon:
                    return False

        cartesian_points.append(point)

    return True


def _min_separation_cell_possible(
    cell,
    d_min=None,
    alpha_min=None,
    epsilon=1e-15,
    cos_alpha_min=None,
    d_min_sq=None,
):

    if d_min is None and alpha_min is None:
        return True

    if cos_alpha_min is None and alpha_min is not None:
        cos_alpha_min = float(np.cos(alpha_min))

    if d_min_sq is None and d_min is not None:
        d_min_sq = float(d_min) * float(d_min)

    bounds = [particle_range.bounds for particle_range in cell.particle_ranges]

    for i in range(len(bounds)):
        b1 = bounds[i]
        for j in range(i + 1, len(bounds)):
            b2 = bounds[j]
            if not min_separation_pair_possible(
                b1,
                b2,
                d_min_sq=d_min_sq,
                cos_alpha_min=cos_alpha_min,
                epsilon=epsilon,
            ):
                return False

    return True


def _pair_key(i, j):

    if i < j:
        return (i, j)
    return (j, i)


def _min_separation_state_from_cell(
    cell,
    d_min=None,
    alpha_min=None,
    epsilon=1e-15,
    cos_alpha_min=None,
    d_min_sq=None,
):

    if d_min is None and alpha_min is None:
        return True, None

    if cos_alpha_min is None and alpha_min is not None:
        cos_alpha_min = float(np.cos(alpha_min))

    if d_min_sq is None and d_min is not None:
        d_min_sq = float(d_min) * float(d_min)

    bounds = [particle_range.bounds for particle_range in cell.particle_ranges]
    pair_state = {}

    for i in range(len(bounds)):
        b1 = bounds[i]
        for j in range(i + 1, len(bounds)):
            b2 = bounds[j]
            feasible = min_separation_pair_possible(
                b1,
                b2,
                d_min_sq=d_min_sq,
                cos_alpha_min=cos_alpha_min,
                epsilon=epsilon,
            )
            pair_state[(i, j)] = feasible

            if not feasible:
                return False, pair_state

    return True, pair_state


def _min_separation_state_from_parent(
    parent_state,
    parent_cell,
    child_cell,
    split_particle_index,
    d_min=None,
    alpha_min=None,
    epsilon=1e-15,
    cos_alpha_min=None,
    d_min_sq=None,
):

    if d_min is None and alpha_min is None:
        return True, None

    if split_particle_index is None or parent_state is None:
        return _min_separation_state_from_cell(
            child_cell,
            d_min=d_min,
            alpha_min=alpha_min,
            epsilon=epsilon,
            cos_alpha_min=cos_alpha_min,
            d_min_sq=d_min_sq,
        )

    child_state = dict(parent_state)
    child_bounds = [particle_range.bounds for particle_range in child_cell.particle_ranges]

    i = split_particle_index
    for k in range(len(child_bounds)):
        if k == i:
            continue

        key = _pair_key(i, k)
        feasible = min_separation_pair_possible(
            child_bounds[i],
            child_bounds[k],
            d_min_sq=d_min_sq,
            cos_alpha_min=cos_alpha_min,
            epsilon=epsilon,
        )
        child_state[key] = feasible

        if not feasible:
            return False, child_state

    return True, child_state


def search(
    n,
    target_depth=12,
    visualize_search=False,
    visualize_all_particles=False,
    visualize_mesh = False,
    show_progress=True,
    progress_update_every=10000,
    parallel_child_bounds=False,
    parallel_workers=mp.cpu_count(),
    parallel_batch_size=32,
    visualize_final=True,
    d_min=None,
    alpha_min=None,
    initial_mesh_side_length=0.1,
):

    root=initial_cell(n)
    tie_breaker=count()

    queue=[]
    active_cells = []
    bounds = []

    best=float("inf")
    best_config=None

    use_min_separation = (d_min is not None) or (alpha_min is not None)
    min_sep_cos_alpha = float(np.cos(alpha_min)) if alpha_min is not None else None
    min_sep_d_sq = float(d_min) * float(d_min) if d_min is not None else None

    processed_nodes = 0
    #Naive 2^depth and then discard n! that don't obey symetry arguments
    estimated_total_nodes = ((2 ** (target_depth + 1)) - 1 ) / math.factorial(n-2)
    progress_line_width = 0
    start_time = time.perf_counter()

    def _print_progress_line(current_nodes):
        nonlocal progress_line_width
        percent = min(100.0, (current_nodes / estimated_total_nodes) * 100)
        elapsed = time.perf_counter() - start_time
        if current_nodes > 0:
            seconds_per_1000 = (elapsed / current_nodes) * 1000
            rate_text = f"{seconds_per_1000:.3f}s/1000 nodes"
        else:
            rate_text = "n/a s/1000 nodes"
        line = (
            f"Progress: {percent:.1f}% "
            f"({current_nodes}/{estimated_total_nodes} estimated nodes, {rate_text})"
        )
        progress_line_width = max(progress_line_width, len(line))
        print(line.ljust(progress_line_width), end="\r", flush=True)

    if show_progress:
        _print_progress_line(0)

    pool = None
    if parallel_child_bounds:
        pool = mp.Pool(processes=parallel_workers)

    try:
        initial_cells = _build_even_mesh(
            root,
            side_length=initial_mesh_side_length,
            cell_filter=_ordered_theta_possible,
        )

        if parallel_child_bounds and len(initial_cells) > 1:
            if pool is None:
                initial_lbs = [energy_lower_bound(cell) for cell in initial_cells]
            else:
                initial_lbs = pool.map(energy_lower_bound, initial_cells)
        else:
            initial_lbs = [energy_lower_bound(cell) for cell in initial_cells]

        for cell, lb_value in zip(initial_cells, initial_lbs):
            min_sep_state = None
            if use_min_separation:
                min_sep_ok, min_sep_state = _min_separation_state_from_cell(
                    cell,
                    d_min=d_min,
                    alpha_min=alpha_min,
                    cos_alpha_min=min_sep_cos_alpha,
                    d_min_sq=min_sep_d_sq,
                )
                if not min_sep_ok:
                    continue

            heapq.heappush(
                queue,
                (
                    lb_value,
                    next(tie_breaker),
                    cell,
                    min_sep_state,
                )
            )

        use_batched_parallel = parallel_child_bounds and parallel_batch_size > 1

        while queue:

            if use_batched_parallel:
                batch_count = min(parallel_batch_size, len(queue))
                frontier = [heapq.heappop(queue) for _ in range(batch_count)]

                pending_children = []
                pending_children_states = []
                pending_tasks = []

                for lb, _, cell, parent_min_sep_state in frontier:
                    processed_nodes += 1

                    if show_progress and (
                        processed_nodes % progress_update_every == 0
                    ):
                        _print_progress_line(processed_nodes)

                    if visualize_search:
                        active_cells.append(cell)
                        bounds.append(lb)

                    if lb>=best:
                        continue

                    if not _ordered_theta_possible(cell):
                        continue

                    if use_min_separation and parent_min_sep_state is None:
                        min_sep_ok, parent_min_sep_state = _min_separation_state_from_cell(
                            cell,
                            d_min=d_min,
                            alpha_min=alpha_min,
                            cos_alpha_min=min_sep_cos_alpha,
                            d_min_sq=min_sep_d_sq,
                        )
                        if not min_sep_ok:
                            continue

                    config=_center_config(cell)

                    center_feasible = True
                    if use_min_separation:
                        center_feasible = _respects_min_separation(
                            config,
                            d_min=d_min,
                            alpha_min=alpha_min,
                        )

                    if not center_feasible:
                        E = float("inf")
                    else:
                        E=thompson_energy(config)

                    if E<best:

                        if not _ordered_theta_center(config):
                            continue

                        best=E
                        best_config=config

                        if show_progress:
                            print()
                        print( "new", best)

                    if cell.depth < target_depth:

                        children, split_particle_index = split_with_index(cell)
                        children = [child for child in children if _ordered_theta_possible(child)]

                        if not children:
                            continue

                        feasible_children = []
                        feasible_states = []

                        for child in children:
                            if use_min_separation:
                                child_ok, child_state = _min_separation_state_from_parent(
                                    parent_min_sep_state,
                                    cell,
                                    child,
                                    split_particle_index,
                                    d_min=d_min,
                                    alpha_min=alpha_min,
                                    cos_alpha_min=min_sep_cos_alpha,
                                    d_min_sq=min_sep_d_sq,
                                )
                                if not child_ok:
                                    continue
                            else:
                                child_state = None

                            feasible_children.append(child)
                            feasible_states.append(child_state)

                        if not feasible_children:
                            continue

                        tasks = build_child_lb_tasks(
                            cell,
                            lb,
                            feasible_children,
                            split_particle_index
                        )

                        if not tasks:
                            continue

                        pending_children.extend(feasible_children)
                        pending_children_states.extend(feasible_states)
                        pending_tasks.extend(tasks)

                if pending_tasks:
                    child_lbs = evaluate_child_lb_tasks(pending_tasks, pool=pool)

                    for child, child_lb, child_state in zip(pending_children, child_lbs, pending_children_states):

                        heapq.heappush(
                            queue,
                            (
                            child_lb,
                            next(tie_breaker),
                            child,
                            child_state,
                            )
                        )
            else:
                lb,_,cell,parent_min_sep_state=heapq.heappop(queue)
                processed_nodes += 1

                if show_progress and (
                    processed_nodes % progress_update_every == 0 or not queue
                ):
                    _print_progress_line(processed_nodes)

                if visualize_search:
                    active_cells.append(cell)
                    bounds.append(lb)

                if lb>=best:
                    continue

                if not _ordered_theta_possible(cell):
                    continue

                if use_min_separation and parent_min_sep_state is None:
                    min_sep_ok, parent_min_sep_state = _min_separation_state_from_cell(
                        cell,
                        d_min=d_min,
                        alpha_min=alpha_min,
                        cos_alpha_min=min_sep_cos_alpha,
                        d_min_sq=min_sep_d_sq,
                    )
                    if not min_sep_ok:
                        continue

                # test center point
                config=_center_config(cell)

                center_feasible = True
                if use_min_separation:
                    center_feasible = _respects_min_separation(
                        config,
                        d_min=d_min,
                        alpha_min=alpha_min,
                    )

                if not center_feasible:
                    E = float("inf")
                else:
                    E=thompson_energy(config)


                if E<best:

                    if not _ordered_theta_center(config):
                        continue

                    best=E
                    best_config=config

                    if show_progress:
                        print()
                    print( "new", best)


                if cell.depth < target_depth:

                    children, split_particle_index = split_with_index(cell)
                    children = [child for child in children if _ordered_theta_possible(child)]

                    if not children:
                        continue

                    feasible_children = []
                    feasible_states = []

                    for child in children:
                        if use_min_separation:
                            child_ok, child_state = _min_separation_state_from_parent(
                                parent_min_sep_state,
                                cell,
                                child,
                                split_particle_index,
                                d_min=d_min,
                                alpha_min=alpha_min,
                                cos_alpha_min=min_sep_cos_alpha,
                                d_min_sq=min_sep_d_sq,
                            )
                            if not child_ok:
                                continue
                        else:
                            child_state = None

                        feasible_children.append(child)
                        feasible_states.append(child_state)

                    if not feasible_children:
                        continue

                    child_lbs = energy_lower_bound_children(
                        cell,
                        lb,
                        feasible_children,
                        split_particle_index,
                        pool=pool
                    )

                    for child, child_lb, child_state in zip(feasible_children, child_lbs, feasible_states):

                        heapq.heappush(
                            queue,
                            (
                            child_lb,
                            next(tie_breaker),
                            child,
                            child_state,
                            )
                        )

            if show_progress and not queue:
                _print_progress_line(processed_nodes)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if visualize_search:
        if visualize_all_particles:
            for particle in range(n):
                draw_global_search(
                    active_cells,
                    bounds,
                    particle=particle
                )

    elapsed_total = time.perf_counter() - start_time
    print('Total elapsed time:', elapsed_total)
    if processed_nodes > 0:
        average_seconds_per_node = elapsed_total / processed_nodes
        print(
            f"Average time: {average_seconds_per_node:.6f}s/node "
            f"over {processed_nodes} nodes"
        )
    else:
        print("Average time: n/a s/node over 0 nodes")

    if visualize_final and best_config is not None:
        visualize_final_minimum.plot_final_minimum(best_config, best)
    if visualize_mesh:
        visualize_parameter_mesh.visualize_parameter_mesh(
            active_cells,
            particle_indexes=range(n),
            lower_bounds=bounds
        )

        #print(active_cells)
        #print(bounds)

    if show_progress:
        elapsed = time.perf_counter() - start_time
        if processed_nodes > 0:
            seconds_per_1000 = (elapsed / processed_nodes) * 1000
            rate_text = f"{seconds_per_1000:.3f}s/1000 nodes"
        else:
            rate_text = "n/a s/1000 nodes"

        final_line = (
            f"Progress: 100.0% (processed {processed_nodes} nodes, {rate_text})"
        )
        progress_line_width = max(progress_line_width, len(final_line))
        print(final_line.ljust(progress_line_width))

    return best,best_config