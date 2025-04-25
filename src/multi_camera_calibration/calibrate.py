import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.multi_camera_calibration.grid import CalibrationGrid
from src.multi_camera_calibration.pattern import CalibrationPattern
from src.multi_camera_calibration.utils.other import find_intersection, find_offsets
from src.multi_camera_calibration.utils.transform import get_transformed_corners, get_transformed_shape
from src.multi_camera_calibration.utils.types import RelShape

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Calibrator:
    """A class for calibrating multi-camera systems using ChArUco boards.

    Attributes
    ----------
        images: The images containing the ChArUco boards.
        matrices: The homography matrices for stitching the images.
        offsets: The offsets for each image.
        pattern: The ChArUco pattern used for calibration.
        ref_idx: The index of the reference image.
        shape_stitched: The shape of the stitched image.
        topdown_matrix: The top-down transformation matrix.

    """

    images: list[np.ndarray]
    matrices: np.ndarray | None
    offsets: np.ndarray | None
    pattern: CalibrationPattern
    ref_idx: int | None
    shape_stitched: RelShape | None
    topdown_matrix: np.ndarray | None

    _combined_grid: CalibrationGrid
    _dst_grids: list[CalibrationGrid]
    _src_grids: list[CalibrationGrid]
    _vanishing_line: float = 0.0
    _vanishing_line_offset: float

    def __init__(
            self,
            images: list[np.ndarray],
            pattern: CalibrationPattern,
            ref_idx: int = None,
            vanishing_line_offset: float = 0.05
    ) -> None:
        """Initialize the calibrator.

        :param images: The images containing the ChArUco boards.
        :param pattern: The ChArUco pattern used for calibration.
        :param ref_idx: The index of the reference image (will be determined automatically if not provided).
        :param vanishing_line_offset: The offset for the vanishing line (default is 0.05).
        """
        if len(images) < 2:
            raise ValueError("At least two images are required for calibration")

        if not all(image.shape == images[0].shape for image in images):
            raise ValueError("All images must have the same dimensions")

        if ref_idx is not None and (ref_idx < 0 or ref_idx >= len(images)):
            raise ValueError("The reference index must be within the bounds of the images")

        if not -1 <= vanishing_line_offset <= 1:
            raise ValueError("The vanishing line offset must be within -1.0 and 1.0")

        self.images = images
        self.pattern = pattern
        self.ref_idx = ref_idx

        self._combined_grid = CalibrationGrid.empty(pattern.get_shape())
        self._src_grids = []
        self._dst_grids = []
        self._vanishing_line_offset = vanishing_line_offset

    def calibrate(self) -> None:
        """Perform the calibration process."""
        self._detect_grids()
        self._determine_reference_image()
        self._compute_homography_matrices()
        self._merge_grids()
        self._calculate_vanishing_line()
        self._calculate_offsets()
        self._determine_stitched_shape()
        self._compute_topdown_matrix()
        self._adjust_topdown_matrix()
        self._normalize()

    def save(
            self,
            save_dir: Path | str,
            filename: str = "latest.npz",
            keep_history: bool = True,
            overwrite: bool = False
    ) -> Path:
        """Save the calibration data to a file.

        :param save_dir: The folder to save the calibration data to.
        :param filename: The name of the file to save the calibration data to.
        :param keep_history: Whether to keep a history of the calibration data.
        :param overwrite: Whether to overwrite the existing file.
        :return: The path to the saved file.
        """
        if self.matrices is None:
            raise ValueError("Cannot save calibration data before calibrating")

        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        primary_path = save_dir / filename

        if primary_path.exists() and not overwrite:
            raise FileExistsError(
                f"The file {primary_path} already exists. Use a different filename or set 'overwrite=True'"
            )

        arrays = dict(
            matrices=self.matrices,
            offsets=self.offsets,
            ref_idx=self.ref_idx,
            shape_original=self.images[self.ref_idx].shape[:2][::-1],
            shape_stitched=self.shape_stitched,
            topdown_matrix=self.topdown_matrix
        )

        np.savez(primary_path, **arrays)
        if not keep_history:
            return primary_path

        history_dir = save_dir / "history"
        history_dir.mkdir(exist_ok=True, parents=True)

        history_name = datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".npz"
        history_path = history_dir / history_name

        np.savez(history_path, **arrays)
        return history_path

    def _adjust_topdown_matrix(self) -> None:
        """Adjust the top-down matrix to correct for rotation errors."""
        h, w = self.images[0].shape[:2]

        # Find the corners of the stitched image
        src_corners = np.zeros((4 * len(self.images), 1, 2), dtype=np.float32)
        for i in range(len(self.images)):
            dst_points = np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype=np.float32)
            if i != self.ref_idx:
                dst_points = cv2.perspectiveTransform(dst_points, self.matrices[i])

            src_corners[i * 4:i * 4 + 4] = dst_points + self.offsets[i]

        # Find the new location of the corners
        dst_corners = cv2.perspectiveTransform(src_corners, self.topdown_matrix)

        src_corners = src_corners.reshape(-1, 2)
        dst_corners = dst_corners.reshape(-1, 2)

        # Calculate the centroids
        src_centroid = np.mean(src_corners, axis=0)
        dst_centroid = np.mean(dst_corners, axis=0)

        src_centered = src_corners - src_centroid
        dst_centered = dst_corners - dst_centroid

        # Find the angle of the top-down matrix
        cov_matrix = src_centered.T @ dst_centered
        u, s, vh = np.linalg.svd(cov_matrix)
        rmat = vh.T @ u.T

        angle = -np.arctan2(rmat[1, 0], rmat[0, 0])

        # Adjust the top-down matrix
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        self.topdown_matrix = rotation_matrix @ self.topdown_matrix

    def _calculate_offsets(self) -> None:
        """Calculate offsets for each image."""
        h, w = self.images[0].shape[:2]
        shapes = []
        for i in range(len(self.images)):
            if i == self.ref_idx:
                shapes.append((w, h))
                continue

            shapes.append(get_transformed_shape(self.matrices[i], (w, h)))

        self.offsets = find_offsets(self._dst_grids, shapes, self.ref_idx) - [0, self._vanishing_line]

    def _calculate_vanishing_line(self) -> None:
        """Calculate the vanishing line for the stitched image."""
        h_line = self._combined_grid.find_intersections(False)
        v_line = self._combined_grid.find_intersections(True)

        vanishing_line = min(h_line, v_line)
        if 0 < vanishing_line < self.images[0].shape[0]:
            self._vanishing_line = vanishing_line
            self._vanishing_line *= 1 + self._vanishing_line_offset

    def _compute_homography_matrices(self) -> None:
        """Compute homography matrices for stitching the images."""
        ref_grid = self._src_grids[self.ref_idx]

        self.matrices = np.zeros((len(self.images), 3, 3))
        for i, grid in enumerate(self._src_grids):
            if i == self.ref_idx:
                self.matrices[i] = np.eye(3)
                continue

            self.matrices[i] = grid.find_homography(ref_grid)

    def _compute_topdown_matrix(self) -> None:
        """Compute the top-down transformation matrix."""
        length = self.pattern.board.getSquareLength()

        # Calculate the top-down matrix
        src_flat = self._combined_grid.flatten()
        dst_grid = CalibrationGrid.from_shape(self.pattern.get_shape(), length)
        dst_flat = dst_grid.flatten()

        mask = np.any(src_flat, axis=1)
        src_flat = src_flat[mask] + self.offsets[self.ref_idx]
        dst_flat = dst_flat[mask]

        self.topdown_matrix = cv2.findHomography(src_flat, dst_flat)[0]

    def _detect_grids(self) -> None:
        """Detect the ChArUco boards in the images."""
        shape = self.pattern.get_shape()

        self._src_grids = []
        for i, image in enumerate(self.images):
            corners, ids = self.pattern.detect(image)
            if corners is None:
                raise ValueError(f"No ChArUco board detected in image {i}")

            grid = CalibrationGrid.from_corners(shape, corners, ids)
            self._src_grids.append(grid)

            logger.debug(f"Detected ChArUco board in image {i}:\n{grid}")

    def _determine_reference_image(self) -> None:
        """Determine the reference image."""
        if self.ref_idx is not None:
            return

        center = np.array([self.images[0].shape[1] / 2, self.images[0].shape[0]])
        idx = np.argmin([
            np.linalg.norm(grid.get_center() - center)
            for grid in self._src_grids
        ])

        self.ref_idx = int(idx)
        logger.debug(f"No reference index provided, using image {self.ref_idx} as reference")

    def _determine_stitched_shape(self) -> None:
        """Determine the shape of the stitched image."""
        h, w = self.images[0].shape[:2]
        height = h - self._vanishing_line

        # Calculate the corners of each part of the stitched image.
        corners = np.zeros((len(self.images), 4, 2), dtype=np.float32)
        for i in range(len(self.images)):
            offset = self.offsets[i]
            if i == self.ref_idx:
                corners[i] = np.array([
                    [offset[0], offset[1]],
                    [offset[0] + w, offset[1]],
                    [offset[0] + w, offset[1] + height],
                    [offset[0], offset[1] + height]
                ])
                continue

            src_points = np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype=np.float32)
            dst_points = cv2.perspectiveTransform(src_points, self.matrices[i])
            corners[i] = dst_points[:, 0] + offset

        # Get the leftmost and rightmost corners
        leftmost_idx = np.argmin([np.min(corners[i][:, 0]) for i in range(len(corners))])
        rightmost_idx = np.argmax([np.max(corners[i][:, 0]) for i in range(len(corners))])

        leftmost = corners[leftmost_idx]
        rightmost = corners[rightmost_idx]

        left_line = (leftmost[0], leftmost[1])
        right_line = (rightmost[0], rightmost[1])

        # Find the point where the top edge intersects with the bottom of the stitched image.
        min_x = 0.0
        max_x = float(rightmost[1][0])

        y_line = ((min_x, height), (max_x, height))

        left_intersection = find_intersection(left_line, y_line)
        right_intersection = find_intersection(right_line, y_line)

        # Calculate the new width of the image.
        if left_intersection is not None:
            min_x = left_intersection[0]
            self.offsets -= [min_x, 0]

        if right_intersection is not None:
            max_x = right_intersection[0]

        self.shape_stitched = (max_x - min_x, height)

    def _merge_grids(self) -> None:
        """Merge transformed grids into a combined grid."""
        self._combined_grid = CalibrationGrid.empty(self.pattern.get_shape())
        self._dst_grids = []

        for i in range(len(self.images)):
            grid = self._src_grids[i]
            if i == self.ref_idx:
                self._combined_grid.merge(grid)
                self._dst_grids.append(grid)
                continue

            h, w = self.images[i].shape[:2]
            min_x, min_y, _, _ = get_transformed_corners(self.matrices[i], (w, h))

            dst_grid = grid.transform(self.matrices[i])
            self._combined_grid.merge(dst_grid)

            dst_grid = dst_grid.offset((-min_x, -min_y))
            self._dst_grids.append(dst_grid)

            adjusted_matrix = np.array([
                [1, 0, -min_x],
                [0, 1, -min_y],
                [0, 0, 1]
            ])
            self.matrices[i] = adjusted_matrix @ self.matrices[i]

    def _normalize(self) -> None:
        """Normalize the calibration data."""
        h, w = self.images[0].shape[:2]
        scale_matrix = np.array([
            [w, 0, 0],
            [0, h, 0],
            [0, 0, 1]
        ])
        scale_matrix_inv = np.linalg.inv(scale_matrix)

        self.matrices = scale_matrix_inv @ self.matrices @ scale_matrix
        self.topdown_matrix = scale_matrix_inv @ self.topdown_matrix @ scale_matrix

        self.offsets /= [w, h]
        self.shape_stitched = (
            self.shape_stitched[0] / w,
            self.shape_stitched[1] / h
        )
