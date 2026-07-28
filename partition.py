from dataclasses import dataclass


@dataclass
class Cell:

    bounds:list
    depth:int=0


def split(cell):

    """
    Bisect largest angular interval
    """

    largest=None
    size=-1

    for i,b in enumerate(cell.bounds):

        for j in range(2):

            s=b[j][1]-b[j][0]

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