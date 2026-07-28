import heapq
from itertools import count

from partition import *
from bound import *
from energy import *
from inital_part import initial_cell


def search(n,target_depth=12):


    root=initial_cell(n)
    tie_breaker=count()

    queue=[]

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


    return best,best_config