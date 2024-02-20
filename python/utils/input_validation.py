from typing import Any, Sequence
from numpy.typing import NDArray
import numpy as np


def validate_dtype(field_name: str, value: Any, expected_dtype: type) -> NDArray[Any]:
    if not isinstance(value, np.ndarray):
        raise TypeError(
            f"Expected {field_name} to be of type np.ndarray, got {value} of type {type(value)}"
        )
    try:
        value = value.astype(expected_dtype)
    except TypeError:
        raise TypeError(
            f"Could not convert {field_name} of type {value.dtype} to {expected_dtype}"
        )
    return value


def validate_array_dimension(
    field_name: str, value: NDArray[Any], expected_ndim: Sequence[int]
) -> None:
    if value.ndim not in expected_ndim:
        raise ValueError(
            f"Supported number of dimensions for {field_name} array of shape {value.shape} are {expected_ndim}, got {value.ndim}"
        )


def validate_array_shape(
    field_name: str, value: NDArray[Any], expected_shape: Sequence[ellipsis | int]
) -> None:
    error_message = f"Supported shape for {field_name} array is {expected_shape} but got {value.shape}"

    # if there are no ellipsis in expected_shape, check that shapes match exactly
    if not any(isinstance(dim, type(...)) for dim in expected_shape):
        if expected_shape != value.shape:
            raise ValueError(error_message)
    else:
        # if trailing dimensions are ellipsis, check that leading dimensions match
        if isinstance(expected_shape[-1], type(...)):
            if expected_shape[:-1] != value.shape[:len(expected_shape) - 1]:
                raise ValueError(error_message)
        # if leading dimensions are ellipsis, check that trailing dimensions match
        if isinstance(expected_shape[0], type(...)):
            if expected_shape[1:] != value.shape[-len(expected_shape) + 1:]:
                raise ValueError(error_message)
