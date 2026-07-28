from dataclasses import dataclass, field


@dataclass
class Bounds:
    lo: float
    hi: float

@dataclass
class Particle_Ranges:
    bounds:list[Bounds]
    fixed: list[bool] | bool = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.fixed, bool):
            self.fixed = [self.fixed] * len(self.bounds)
        elif not self.fixed:
            self.fixed = [False] * len(self.bounds)

@dataclass
class Cell:
    particle_ranges:list[Particle_Ranges]
    depth:int=0


def split(cell):

    """
    Bisect largest angular interval, skip infinitly small ones ande give warning
    """

    largest=None
    size=-1

    for i,particle_range in enumerate(cell.particle_ranges):
        for j in range(2):
            if particle_range.fixed[j]: #skip fixed particles and dimensions
                continue

            s=particle_range.bounds[j].hi-particle_range.bounds[j].lo
            if s ==0:
                print(f"Warning: zero interval at cell {i}, dimension {j}")
                continue

            if s<1e-15:
                print(f"Warning: infinitesimally small interval at cell {i}, dimension {j}")
                continue

            if s>size:
                size=s
                largest=(i,j)


    if largest is None:
        return []

    i,j=largest

    b=cell.particle_ranges.copy()

    lo,hi=b[i].bounds[j].lo,b[i].bounds[j].hi

    mid=(lo+hi)/2


    b1=[[Bounds(bb.lo, bb.hi) for bb in x.bounds] for x in cell.particle_ranges]
    b2=[[Bounds(bb.lo, bb.hi) for bb in x.bounds] for x in cell.particle_ranges]


    b1[i][j]=Bounds(lo,mid)
    b2[i][j]=Bounds(mid,hi)


    return [
        Cell(
            [ Particle_Ranges( bounds=x, fixed=cell.particle_ranges[k].fixed.copy())
                for k, x in enumerate(b1) ],
            cell.depth+1
        ),
        Cell(
            [ Particle_Ranges( bounds=x, fixed=cell.particle_ranges[k].fixed.copy())
                for k, x in enumerate(b2) ],
            cell.depth+1
        )
    ]