from numba import njit
from numpy.typing import NDArray
import numpy as np

@njit
def apply(
    data: NDArray[np.float64],
    hyperplane_normals: NDArray[np.float64],
    hyperplanes_intercepts: NDArray[np.float64],
):
    """Find the paths in the tree for each data point."""
    paths = []
    for datapoint in data:
        path = []
        node_idx = 0
        while node_idx<len(hyperplane_normals):
            path.append(node_idx)
            normal = hyperplane_normals[node_idx]
            intercept = hyperplanes_intercepts[node_idx]
            if datapoint @ normal - intercept <= 0:
                node_idx = 2*node_idx + 1
            else:
                node_idx = 2*node_idx + 2
        paths.append(path)
    return np.array(paths)
