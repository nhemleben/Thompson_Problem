from dataclasses import dataclass, field


@dataclass
class Bounds:
    lo: float
    hi: float

@dataclass
class Particle_Ranges:
    bounds:list[Bounds]
    fixed: list[bool] | bool = field(default_factory=list)
    chart: str = "standard"

    def __post_init__(self):
        if isinstance(self.fixed, bool):
            self.fixed = [self.fixed] * len(self.bounds)
        elif not self.fixed:
            self.fixed = [False] * len(self.bounds)

@dataclass
class Cell:
    particle_ranges:list[Particle_Ranges]
    depth:int=0


def _find_split_axis(cell):

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


    return largest


def _split_on_axis(cell, i, j):

    target_particle = cell.particle_ranges[i]
    lo,hi=target_particle.bounds[j].lo,target_particle.bounds[j].hi

    mid=(lo+hi)/2

    left_particle_ranges = list(cell.particle_ranges)
    right_particle_ranges = list(cell.particle_ranges)

    left_bounds = list(target_particle.bounds)
    right_bounds = list(target_particle.bounds)

    left_bounds[j]=Bounds(lo,mid)
    right_bounds[j]=Bounds(mid,hi)

    left_particle_ranges[i]=Particle_Ranges(
        bounds=left_bounds,
        fixed=target_particle.fixed.copy(),
        chart=target_particle.chart,
    )
    right_particle_ranges[i]=Particle_Ranges(
        bounds=right_bounds,
        fixed=target_particle.fixed.copy(),
        chart=target_particle.chart,
    )


    children = [
        Cell(
            left_particle_ranges,
            cell.depth+1
        ),
        Cell(
            right_particle_ranges,
            cell.depth+1
        )
    ]

    return children


def split(cell):

    largest = _find_split_axis(cell)

    if largest is None:
        return []

    i, j = largest
    return _split_on_axis(cell, i, j)


def split_with_index(cell):

    largest = _find_split_axis(cell)

    if largest is None:
        return [], None

    i, j = largest
    return _split_on_axis(cell, i, j), i