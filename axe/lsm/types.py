from dataclasses import dataclass
import enum
from typing import Tuple


class Policy(enum.Enum):
    Tiering = 0
    Leveling = 1
    Classic = 2
    Kapacity = 3
    QHybrid = 4
    Fluid = 5


@dataclass()
class System():
    entry_size: int = 8192
    selectivity: float = 4e-7
    entries_per_page: int = 4
    num_entries: int = 1_000_000_000
    mem_budget: float = 10.0
    phi: float = 1.0  # Read/Write asymmetry coefficient


@dataclass(kw_only=True)
class LSMDesign:
    bits_per_elem: float
    size_ratio: float
    policy: Policy
    kapacity: Tuple[float, ...]

    def __post_init__(self):
        if self.policy in (Policy.Tiering, Policy.Leveling, Policy.Classic):
            assert len(self.kapacity) == 0
        elif self.policy == Policy.QHybrid:
            assert len(self.kapacity) == 1
        elif self.policy == Policy.Fluid:
            assert len(self.kapacity) == 2


@dataclass()
class LSMBounds:
    max_considered_levels: int = 20
    bits_per_elem_range: Tuple[float, float] = (1, 10)
    size_ratio_range: Tuple[int, int] = (2, 31)
    page_sizes: Tuple = (4, 8, 16)
    entry_sizes: Tuple = (1024, 2048, 4096, 8192)
    memory_budget_range: Tuple[float, float] = (5.0, 20.0)
    selectivity_range: Tuple[float, float] = (1e-7, 1e-9)
    elements_range: Tuple[int, int] = (100000000, 1000000000)


@dataclass(frozen=True)
class Workload:
    z0: float = 0.25
    z1: float = 0.25
    q: float = 0.25
    w: float = 0.25
