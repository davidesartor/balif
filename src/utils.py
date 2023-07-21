from __future__ import annotations
from typing import Optional

from functools import cache

import numpy as np
from numpy.typing import NDArray
    
@cache
def expected_isolation_depth(n: int) -> float:
    """Compute the expected isolation depth for a node with n data points."""
    if n <= 1:
        return 0.0
    euler_mascheroni_constant = 0.5772156649
    nth_harmonic_number = np.log(n - 1) + euler_mascheroni_constant
    return 2 * nth_harmonic_number - 2 * (n - 1) / n

def random_vector(dims: int, non_zero: Optional[int] = None) -> NDArray[np.float64]:
    """Generate a random vector from a unit ball with a given number of non-zero components."""
    if dims <= 0:
        raise ValueError("number of dimensions must be positive")
    if non_zero is None:
        non_zero = dims
    elif non_zero <= 0:
        raise ValueError("number of non zero components must be positive")

    normals: NDArray[np.float32] = np.zeros(dims, dtype=np.float64)
    selected = np.random.choice(np.arange(dims), size=non_zero, replace=False)
    normals[selected] = np.random.randn(int(non_zero))
    return normals / np.linalg.norm(normals)
