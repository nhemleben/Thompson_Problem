from dataclasses import dataclass


@dataclass
class Cell:

    bounds:list
    depth:int=0


def split(cell):

    """
    Bisect largest angular interval, skip infinitly small ones ande give warning
    """

    largest=None
    size=-1

    for i,bounds in enumerate(cell.bounds):

        for j in range(2):

            s=bounds[j][1]-bounds[j][0]

            if s<1e-8:
                print(f"Warning: infinitesimally small interval at cell {i}, dimension {j}")
                continue

            if s>size:
                size=s
                largest=(i,j)


    i,j=largest

    new=[]

    b=cell.bounds.copy()

    lo,hi=b[i][j]

    mid=(lo+hi)/2


    b1=[x[:] for x in b]
    b2=[x[:] for x in b]


    b1[i][j]=[lo,mid]
    b2[i][j]=[mid,hi]


    return [
        Cell(b1,cell.depth+1),
        Cell(b2,cell.depth+1)
    ]