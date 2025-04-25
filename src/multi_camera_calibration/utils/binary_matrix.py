import numpy as np


def find_largest_rectangle(matrix: np.ndarray) -> np.ndarray | None:
    """Find the largest rectangle in a binary matrix.

    :param matrix: The binary matrix.
    :return: The corners of the largest rectangle (tl, tr, br, bl).
    """
    if not np.any(matrix):
        return None

    rows, cols = matrix.shape
    max_area = 0
    result = None

    for top in range(rows):
        for left in range(cols):
            if matrix[top, left] == 0:
                continue

            for bottom in range(top, rows):
                for right in range(left, cols):
                    if matrix[bottom, right] == 0:
                        continue

                    if matrix[top, right] == 0:
                        continue

                    if matrix[bottom, left] == 0:
                        continue

                    area = (bottom - top + 1) * (right - left + 1)
                    if area > max_area:
                        max_area = area
                        result = (
                            (top, left),
                            (top, right),
                            (bottom, right),
                            (bottom, left)
                        )

    return np.array(result)
