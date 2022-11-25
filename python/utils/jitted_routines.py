import numba
from numpy.typing import NDArray
import numpy as np


@numba.njit
def isolation_node_idxs(data, hyperplane_normals, hyperplane_intercepts, node_sizes):
    res = np.zeros(len(data), dtype=np.int32)
    for i in range(len(data)):
        node_id = 0
        while 2 * node_id + 2 < len(hyperplane_normals) and node_sizes[node_id] > 1:
            # compute distance to hyperplane
            dist = -hyperplane_intercepts[node_id]
            for j in range(len(hyperplane_normals[node_id])):
                dist += data[i, j] * hyperplane_normals[node_id][j]

            # go left or right
            if dist <= 0:
                node_id = 2 * node_id + 1
            else:
                node_id = 2 * node_id + 2
        res[i] = node_id
    return res


@numba.njit
def leaf_node_idxs(data, hyperplane_normals, hyperplane_intercepts):
    res = np.zeros(len(data), dtype=np.int32)
    for i in range(len(data)):
        node_id = 0
        while 2 * node_id + 2 < len(hyperplane_normals):
            # compute distance to hyperplane
            dist = -hyperplane_intercepts[node_id]
            for j in range(len(hyperplane_normals[node_id])):
                dist += data[i, j] * hyperplane_normals[node_id][j]

            # go left or right
            if dist <= 0:
                node_id = 2 * node_id + 1
            else:
                node_id = 2 * node_id + 2
        res[i] = node_id
    return res
