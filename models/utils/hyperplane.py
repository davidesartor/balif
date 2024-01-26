from dataclasses import dataclass
from typing import Optional
from typing_extensions import Self
import numpy as np
import random
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Hyperplane:
    normal: NDArray[np.float64]
    intercept: np.float64 = np.float64(0)
    components: int | None = None
    
    @property
    def dim(self) -> int:
        return self.normal.shape[0]

    def distance(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.dot(data, self.normal) - self.intercept

    @classmethod
    def random_splitting_plane(cls, data: NDArray[np.float64], components: int | None = None) -> Self:
        normal = random_vector(data.shape[1], non_zero=components)
        distances = np.dot(data, normal)
        intercept = np.float64(random.uniform(min(distances), max(distances)))
        return cls(normal, intercept, components)
    

def random_vector(dims: int, non_zero: Optional[int] = None, unit_norm=True) -> NDArray[np.float64]:
    """Generate a random vector from a unit ball with a given number of non-zero components."""
    normals = np.zeros(dims, dtype=np.float64)
    if non_zero is None or non_zero == dims:
        for idx in range(dims):
            normals[idx] = random.gauss(0, 1)
    elif non_zero == 1:
        normals[random.randrange(dims)] = 1
        unit_norm = False
    else:
        for idx in random.sample(range(dims), non_zero):
            normals[idx] = random.gauss(0, 1)
        
    if unit_norm:
        normals = normals / np.linalg.norm(normals)
    return normals