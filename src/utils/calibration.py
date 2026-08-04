from __future__ import annotations

import numpy as np


def matrix4_from_flat(values: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size != 16:
        raise ValueError("Expected 16 values to create a 4x4 matrix.")
    return array.reshape(4, 4)
