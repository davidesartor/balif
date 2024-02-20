from functools import cache
import numpy as np


def depth(node_idx: int) -> int:
    return np.frexp(node_idx + 1)[1] - 1


def left_child_idx(parent_idx: int) -> int:
    return 2 * parent_idx + 1


def right_child_idx(parent_idx) -> int:
    return 2 * parent_idx + 2


def path_to(node_idx: int) -> list[int]:
    """Indices of nodes on the path from the root to the current node (both included)."""
    path = [node_idx]
    while node_idx > 0:
        node_idx = (node_idx - 1) // 2
        path.append(node_idx)
    return path[::-1]


@cache
def expected_isolation_depth(n: int) -> float:
    """Compute the expected isolation depth for a node with n data points."""
    if n <= 1:
        return 0.0
    euler_mascheroni_constant = 0.5772156649
    nth_harmonic_number = np.log(n - 1) + euler_mascheroni_constant
    return 2 * nth_harmonic_number - 2 * (n - 1) / n
