import heapq
from itertools import count

from matplotlib.pyplot import draw

from global_visualize import draw_global_search
from partition import *
from bound import *
from energy import *
from inital_part import initial_cell
import visualize_parameter_mesh


def search(n,target_depth=12, visualize_search=False):


    root=initial_cell(n)
    tie_breaker=count()

    queue=[]
    active_cells = []
    bounds = []

    heapq.heappush(
        queue,
        (
            energy_lower_bound(root),
            next(tie_breaker),
            root
        )
    )


    best=float("inf")
    best_config=None


    while queue:

        lb,_,cell=heapq.heappop(queue)

        if visualize_search:
            active_cells.append(cell)
            bounds.append(lb)

        if lb>=best:
            continue


        # test center point

        config=[]

        for t,p in cell.bounds:

            tc=(t[0]+t[1])/2
            pc=(p[0]+p[1])/2

            config.append(
                (tc,pc)
            )


        E=thompson_energy(config)


        if E<best:

            best=E
            best_config=config

            print( "new", best)


        if cell.depth < target_depth:

            for child in split(cell):

                heapq.heappush(
                    queue,
                    (
                    energy_lower_bound(child),
                    next(tie_breaker),
                    child
                    )
                )

    if visualize_search:
        draw_global_search(
            active_cells,
            bounds,
            particle=0
        )
        visualize_parameter_mesh.visualize_parameter_mesh(
            active_cells,
            particles=range(n),
            lower_bounds=bounds
        )

        print(active_cells)
        print(bounds)

    return best,best_config