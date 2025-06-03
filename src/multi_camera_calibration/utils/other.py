from typing import TYPE_CHECKING

import numpy as np

from src.multi_camera_calibration.utils.types import Coordinate, Shape

if TYPE_CHECKING:
    from src.grid import CalibrationGrid


def get_border_of_points(points: np.ndarray) -> tuple[float, float, float, float]:
    """Get the border of the points.

    :param points: The points.
    :return: The border of the points.
    """
    min_x = np.min(points[:, 0])
    min_y = np.min(points[:, 1])
    max_x = np.max(points[:, 0])
    max_y = np.max(points[:, 1])

    return min_x, min_y, max_x, max_y


def find_intersection(
        line1: tuple[Coordinate, Coordinate],
        line2: tuple[Coordinate, Coordinate],
        segments: bool = True
) -> Coordinate | None:
    """Find the intersection between two lines.

    :param line1: The first line.
    :param line2: The second line.
    :param segments: Whether the lines are segments or infinite lines.
    :return: The intersection between the two lines, if it exists.
    """
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]

    denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denominator == 0:
        return None

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator

    px = x1 + ua * (x2 - x1)
    py = y1 + ua * (y2 - y1)

    if segments and not (0 <= ua <= 1 and 0 <= ub <= 1):
        return None

    return px, py


def find_offsets(grids: list["CalibrationGrid"], shapes: list[Shape], ref_idx: int) -> np.ndarray:
    """Find the offsets for the images.

    :param grids: The grids of the ChArUco boards after warping.
    :param shapes: The shapes of the warped images.
    :param ref_idx: The index of the reference image.
    :return: The offsets for the images.
    """
    if len(grids) != len(shapes):
        raise ValueError("The number of grids and shapes must be the same")

    max_x = [np.max(grid.grid[:, :, 0][np.nonzero(grid.grid[:, :, 0])]) for grid in grids]
    leftmost_idx = np.argmax(max_x)
    rightmost_idx = np.argmin(max_x)

    offsets = np.zeros((len(grids), 2), dtype=np.float32)
    ref_points = grids[leftmost_idx].flatten()

    for i in range(len(grids)):
        if i == leftmost_idx:
            continue

        for p1, p2 in zip(grids[i].flatten(), ref_points):
            if not np.all(p1) or not np.all(p2):
                continue

            offsets[i] = p2 - p1

    # Normalize the offsets
    ref_y = offsets[ref_idx][1]
    offsets[:, 1] -= ref_y

    # Center the reference image
    dist_to_left = offsets[ref_idx][0]
    dist_to_right = offsets[rightmost_idx][0] + shapes[rightmost_idx][1] - (shapes[ref_idx][1] + dist_to_left)

    diff = dist_to_right - dist_to_left
    offsets[:, 0] += diff

    return offsets
