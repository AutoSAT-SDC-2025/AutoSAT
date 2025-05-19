import cv2
import numpy as np

from src.multi_camera_calibration.utils.binary_matrix import find_largest_rectangle
from src.multi_camera_calibration.utils.other import find_intersection


class CalibrationGrid:
    """Represents a ChArUco board as a grid.

    Attributes
    ----------
        grid: The grid representing the ChArUco board.

    """

    grid: np.ndarray

    def __init__(self, grid: np.ndarray):
        """Initialize the grid.

        :param grid: The grid representing the ChArUco board.
        """
        self.grid = grid

    def find_corners(self) -> tuple[np.ndarray, tuple[int, int]]:
        """Find the corners of the ChArUco board in the grid.

        :return: The corners and the shape of the detected board (tl, tr, br, bl).
        """
        binary_matrix = np.any(self.grid, axis=2).astype(np.uint8)

        corner_indices = find_largest_rectangle(binary_matrix)
        if corner_indices is None:
            raise ValueError("No rectangle found in the ChArUco board")

        rect_w = int(np.linalg.norm(corner_indices[0] - corner_indices[1]))
        rect_h = int(np.linalg.norm(corner_indices[1] - corner_indices[2]))
        shape = (rect_w, rect_h)

        return self.grid[corner_indices[:, 0], corner_indices[:, 1]], shape

    def find_homography(self, other: "CalibrationGrid") -> np.ndarray:
        """Find the homography matrix between two grids.

        :param other: The other grid to find the homography with.
        :return: The homography matrix.
        """
        flat_self = self.flatten()
        flat_other = other.flatten()

        points_self = flat_self[np.any(flat_self, axis=1) & np.any(flat_other, axis=1)]
        points_other = flat_other[np.any(flat_self, axis=1) & np.any(flat_other, axis=1)]
        if len(points_other) < 4:
            raise ValueError("Not enough points to find the homography")

        return cv2.findHomography(points_self, points_other)[0]

    def find_intersections(self, vertical: bool = False) -> float:
        """Calculate the intersections of the ChArUco board.

        :param vertical: Whether to use the vertical lines.
        :return: The intersections of the ChArUco board.
        """
        # Get the grid of the ChArUco board
        grid = self.grid
        if vertical:
            grid = grid.transpose(1, 0, 2)

        # Find the lines of the ChArUco board
        lines = [[point for point in row if np.any(point)] for row in grid]
        lines = [(line[0], line[-1]) for line in lines if len(line) > 1]

        # Find the intersections of the lines
        intersections = [find_intersection(lines[i], lines[j], False)
                         for i in range(len(lines) - 1)
                         for j in range(i + 1, len(lines))]
        intersections = [point for point in intersections if point is not None]
        intersections = np.array(intersections)

        if len(intersections) == 0:
            return np.nan

        return np.median(intersections, axis=0)[1]

    def flatten(self) -> np.ndarray:
        """Flatten the grid to a 2D array.

        :return: The flattened grid.
        """
        return self.grid.reshape(-1, 2)

    def get_center(self) -> tuple[float, float]:
        """Get the center of the ChArUco board.

        :return: The center of the ChArUco board.
        """
        return np.mean(self.grid, axis=(0, 1))

    def is_empty(self) -> bool:
        """Check if the grid is empty.

        :return: Whether the grid is empty.
        """
        return not np.any(self.grid)

    def merge(self, other: "CalibrationGrid") -> "CalibrationGrid":
        """Merge another grid with this grid.

        :param other: The grid to merge with this grid.
        :return: The merged grid.
        """
        self.grid = np.where(other.grid, other.grid, self.grid)
        return self

    def offset(self, offset: tuple[float, float]) -> "CalibrationGrid":
        """Offset the grid by a value.

        :param offset: The offset to apply to the grid.
        :return: The offset grid.
        """
        grid = self.grid.copy()
        grid[np.any(self.grid, axis=2)] += offset

        return CalibrationGrid(grid)

    def transform(self, matrix: np.ndarray) -> "CalibrationGrid":
        """Transform the grid with a perspective matrix.

        :param matrix: The perspective matrix to transform the grid with.
        :return: The transformed grid.
        """
        src = self.grid.reshape(-1, 1, 2)

        dst = cv2.perspectiveTransform(src, matrix).reshape(self.grid.shape)
        dst[np.all(self.grid == 0, axis=2)] = 0

        return CalibrationGrid(dst)

    @classmethod
    def empty(cls, shape: tuple[int, int]) -> "CalibrationGrid":
        """Create an empty grid with a shape.

        :param shape: The shape of the grid (w, h).
        :return: The empty grid.
        """
        w, h = np.subtract(shape, 1)
        grid = np.zeros((h, w, 2), dtype=np.float32)

        return cls(grid)

    @classmethod
    def from_corners(cls, shape: tuple[int, int], corners: np.ndarray, ids: np.ndarray) -> "CalibrationGrid":
        """Convert corners and IDs to a grid.

        :param shape: The shape of the board (w, h).
        :param corners: An array of corners.
        :param ids: An array of IDs for each corner.
        :return: The grid of corners.
        """
        w, h = np.subtract(shape, 1)
        grid = np.zeros((h, w, 2), dtype=np.float32)

        rows = ids[:, 0] // w
        cols = ids[:, 0] % w

        grid[rows, cols] = corners[:, 0]
        return cls(grid)

    @classmethod
    def from_shape(cls, shape: tuple[int, int], length: float) -> "CalibrationGrid":
        """Create a flat grid with the shape of the board.

        :param shape: The shape of the board (w, h).
        :param length: The length of a single square.
        :return: The grid of the board.
        """
        w, h = np.subtract(shape, 1)
        grid = np.indices((w, h), dtype=np.float32).transpose(2, 1, 0)
        grid *= length

        return cls(grid)

    def __str__(self) -> str:
        """Return a string representation of the ChArUco board."""
        grid_str = ""
        for row in self.grid:
            for point in row:
                if np.any(point):
                    grid_str += "█"
                else:
                    grid_str += "░"
            grid_str += "\n"
        return grid_str

    def __repr__(self) -> str:
        """Return a string representation of the ChArUco board."""
        return f"CalibrationGrid({self.grid.shape[1]}, {self.grid.shape[0]})"
