import logging

import cv2
import numpy as np

from src.multi_camera_calibration.utils.other import find_intersection, get_border_of_points
from src.multi_camera_calibration.utils.transform import stitch_images
from src.multi_camera_calibration.utils.types import Shape

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class RenderDistance:
    """A class to represent the render distance for the topdown image.

    Attributes
    ----------
        front: The distance in meters to the front of the camera.
        side: The distance in meters to the sides of the camera.

    """

    front: float
    side: float

    def __init__(self, front: float, sides: float) -> None:
        """Initialize the render distance configuration.

        :param front: The distance in meters to the front of the camera.
        :param sides: The distance in meters to the sides of the camera.
        """
        if front <= 0 or sides <= 0:
            raise ValueError("Render distance must be positive")

        self.front = front
        self.side = sides


class CalibrationData:
    """A class for transforming images of multi-camera systems to a topdown view.

    Attributes
    ----------
        input_shape: The shape of the input images (width, height).
        matrices: The homography matrices for stitching the images.
        offsets: The offsets for each image.
        output_shape: The maximum allowed shape of the top-down image (width, height).
        pixels_per_meter: The amount of pixels per meter in the top-down image.
        ref_idx: The index of the reference image.
        render_distance: The render distance for the topdown image.
        shapes: The shapes of the images after warping.
        shape_stitched: The shape of the stitched image.
        shape_topdown: The shape of the top-down image.
        topdown_matrix: The top-down transformation matrix.

    """

    input_shape: Shape
    matrices: np.ndarray
    offsets: np.ndarray
    output_shape: Shape
    pixels_per_meter: float
    ref_idx: int
    render_distance: RenderDistance
    shapes: np.ndarray
    shape_stitched: Shape
    shape_topdown: Shape
    topdown_matrix: np.ndarray

    _shape_original: Shape

    def __init__(self, path: str, input_shape: Shape, output_shape: Shape, render_distance: RenderDistance) -> None:
        """Load calibration data from a file.

        :param path: The path to the file.
        :param input_shape: The shape of the input images (width, height).
        :param output_shape: The maximum allowed shape of the topdown image (width, height).
        :param render_distance: The render distance for the topdown image.
        """
        if input_shape[0] <= 0 or input_shape[1] <= 0:
            raise ValueError("Input shape must be positive")

        if output_shape[0] <= 0 or output_shape[1] <= 0:
            raise ValueError("Output shape must be positive")

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.render_distance = render_distance

        # Load the calibration data
        data = np.load(path)

        self.matrices = data["matrices"]
        self.offsets = data["offsets"]
        self.ref_idx = int(data["ref_idx"])
        self.shape_stitched = tuple(data["shape_stitched"])
        self.topdown_matrix = data["topdown_matrix"]
        self._shape_original = tuple(data["shape_original"])

        if self._shape_original[0] / self._shape_original[1] != self.input_shape[0] / self.input_shape[1]:
            raise ValueError(
                "The aspect ratio of the input images does not match the aspect ratio of the original images."
            )

        # Adjust the calibration data
        self._denormalize()
        self._adjust_region_of_interest()

    def get_distance(self, pixels: int) -> float:
        """Get the distance in meters from pixels.

        :param pixels: The amount of pixels.
        :return: The distance in meters.
        """
        return pixels / self.pixels_per_meter

    def get_distance_to_point(self, x: int, y: int, shape: Shape) -> float:
        """Get the distance in meters from a point in the reference image.

        :param x: The x-coordinate in the reference image.
        :param y: The y-coordinate in the reference image.
        :param shape: The shape of the image (width, height).
        :return: The distance in meters.
        """
        x, y = self.transform_point(x, y, shape)
        return self.get_distance_to_transformed_point(x, y)

    def get_distance_to_transformed_point(self, x: int, y: int) -> float:
        """Get the distance in meters from a point in the topdown image.

        :param x: The x-coordinate in the topdown image.
        :param y: The y-coordinate in the topdown image.
        :return: The distance in meters.
        """
        center = self.shape_topdown[0] // 2, self.shape_topdown[1]
        dist = np.linalg.norm(np.array([x, y]) - np.array(center))

        return self.get_distance(int(dist))

    def get_distance_to_y(self, x: int, y: int, shape: Shape) -> float:
        """Get the distance in meters from a point in the reference image, not considering the x-coordinate.

        :param x: The x-coordinate in the reference image.
        :param y: The y-coordinate in the reference image.
        :param shape: The shape of the image (width, height).
        :return: The distance in meters.
        """
        x, y = self.transform_point(x, y, shape)

        dist = int(self.shape_topdown[1] - y)
        return self.get_distance(dist)

    def get_pixels(self, meters: float) -> int:
        """Get the amount of pixels from a distance in meters.

        :param meters: The distance in meters.
        :return: The amount of pixels.
        """
        return int(meters * self.pixels_per_meter)

    def stitch(self, images: list[np.ndarray]) -> np.ndarray:
        """Stitch the images together.

        :param images: The images to stitch.
        :return: The stitched image.
        """
        shape = self.shape_stitched[::-1]
        if len(images[0].shape) == 3:
            shape += (3,)

        stitched = np.zeros(shape, dtype=np.uint8)
        for i, image in enumerate(images):
            if i != self.ref_idx:
                stitched = self._stitch_image(stitched, image, i)

        return self._stitch_image(stitched, images[self.ref_idx], self.ref_idx)

    def transform(self, images: list[np.ndarray]) -> np.ndarray:
        """Transform the images to a topdown view.

        :param images: The images to transform.
        :return: The topdown image.
        """
        stitched = self.stitch(images)
        return cv2.warpPerspective(
            stitched,
            self.topdown_matrix,
            self.shape_topdown,
            flags=cv2.INTER_NEAREST
        )

    def transform_point(self, x: int, y: int, shape: Shape) -> tuple[int, int]:
        """Transform a point to a topdown view.

        :param x: The x-coordinate of the point.
        :param y: The y-coordinate of the point.
        :param shape: The shape of the image (width, height).
        :return: The transformed point.
        """
        x *= self.input_shape[0] / shape[0]
        y *= self.input_shape[1] / shape[1]

        x += self.offsets[self.ref_idx][0]
        y += self.offsets[self.ref_idx][1]

        src_point = np.array([[[x, y]]], dtype=np.float32)
        dst_point = cv2.perspectiveTransform(src_point, self.topdown_matrix)

        return int(dst_point[0][0][0]), int(dst_point[0][0][1])

    def _adjust_region_of_interest(self) -> None:
        """Adjust the calibration data to selected region of interest."""
        self.shapes = np.zeros((len(self.matrices), 2), dtype=np.int32)
        w, h = self.input_shape

        # Find the corners of the stitched image
        src_corners = np.zeros((4 * len(self.matrices), 1, 2), dtype=np.float32)
        for i in range(len(self.matrices)):
            dst_points = np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype=np.float32)
            self.shapes[i] = w, h

            if i != self.ref_idx:
                dst_points = cv2.perspectiveTransform(dst_points, self.matrices[i])
                min_x, min_y, max_x, max_y = get_border_of_points(dst_points[:, 0])

                width = int(max_x - min_x)
                height = int(max_y - min_y)

                self.shapes[i] = width, height

            src_corners[i * 4:i * 4 + 4] = dst_points + self.offsets[i]

        # Find the new location of the corners
        src_corners = np.clip(src_corners, 0, None)
        dst_corners = cv2.perspectiveTransform(src_corners, self.topdown_matrix)

        min_x, min_y, max_x, max_y = get_border_of_points(dst_corners[:, 0])

        # Find the new center of the image
        src_center = np.array([[[w // 2, h]]], dtype=np.float32) + self.offsets[self.ref_idx]
        dst_center = cv2.perspectiveTransform(src_center, self.topdown_matrix)[0][0]

        cx = float(dst_center[0])
        cy = float(dst_center[1])

        self.pixels_per_meter = self.input_shape[0] / self._shape_original[0]

        # Update the render distance
        render_side = self.render_distance.side * self.pixels_per_meter
        render_front = self.render_distance.front * self.pixels_per_meter

        min_x = max(min_x, cx - render_side)
        max_x = min(max_x, cx + render_side)
        min_y = max(min_y, cy - render_front)
        max_y = cy

        # Limit the size of the image and center it
        dist_to_left = cx - min_x
        dist_to_right = max_x - cx
        diff = dist_to_left - dist_to_right

        if dist_to_left < render_side:
            real_dist = dist_to_left / self.pixels_per_meter
            logger.warning(
                f"The distance to the left edge ({real_dist:.2f}m) is smaller than the configured render distance ({self.render_distance.side:.2f}m)."
            )

        if dist_to_right < render_side:
            real_dist = dist_to_right / self.pixels_per_meter
            logger.warning(
                f"The distance to the right edge ({real_dist:.2f}m) is smaller than the configured render distance ({self.render_distance.side:.2f}m)."
            )

        min_x = max(min_x, min_x + diff)
        max_x = min(max_x, max_x + diff)

        # Find the new top of the image
        lines = [dst_corners[i * 4:i * 4 + 2, 0] for i in range(len(self.matrices))]
        min_x_line = ((min_x, min_y), (min_x, max_y))
        max_x_line = ((max_x, min_y), (max_x, max_y))

        intersections = [find_intersection(lines[i], min_x_line) for i in range(len(lines))]
        intersections += [find_intersection(lines[i], max_x_line) for i in range(len(lines))]
        intersections = [point for point in intersections if point is not None]

        if len(intersections) > 0:
            min_y = max([point[1] for point in intersections])

            real_dist = (max_y - min_y) / self.pixels_per_meter
            logger.warning(
                f"The distance to the top of the image ({real_dist:.2f}m) is smaller than the configured render distance ({self.render_distance.front:.2f}m)."
            )

        # Calculate the new shape of the image
        width = max_x - min_x
        height = max_y - min_y

        scale_factor = min(
            self.output_shape[0] / width,
            self.output_shape[1] / height
        )

        min_x *= scale_factor
        min_y *= scale_factor
        self.shape_topdown = (
            int(width * scale_factor),
            int(height * scale_factor)
        )

        # Adjust the top-down matrix
        adjusted_matrix = np.array([
            [scale_factor, 0, -min_x],
            [0, scale_factor, -min_y],
            [0, 0, 1]
        ])
        self.topdown_matrix = adjusted_matrix @ self.topdown_matrix

    def _denormalize(self) -> None:
        """Denormalize the calibration data."""
        w, h = self.input_shape
        scale_matrix = np.array([
            [w, 0, 0],
            [0, h, 0],
            [0, 0, 1]
        ])
        scale_matrix_inv = np.linalg.inv(scale_matrix)

        self.matrices = scale_matrix @ self.matrices @ scale_matrix_inv
        self.topdown_matrix = scale_matrix @ self.topdown_matrix @ scale_matrix_inv

        self.offsets = (self.offsets * self.input_shape).astype(int)
        self.shape_stitched = (
            int(self.shape_stitched[0] * w),
            int(self.shape_stitched[1] * h)
        )

    def _stitch_image(self, stitched: np.ndarray, image: np.ndarray, idx: int) -> np.ndarray:
        """Stitch an image to the stitched image.

        :param stitched: The stitched image.
        :param image: The image to stitch.
        :param idx: The index of the camera.
        :return: The stitched image.
        """
        if image.shape[:2] != self.input_shape[::-1]:
            image = cv2.resize(image, self.input_shape, interpolation=cv2.INTER_NEAREST)

        warped = self._warp_image(image, idx)
        return stitch_images(stitched, warped, self.offsets[idx])

    def _warp_image(self, image: np.ndarray, idx: int) -> np.ndarray:
        """Warp an image using a perspective matrix.

        :param image: The image to warp.
        :param idx: The camera to select the configuration for.
        :return: The warped image.
        """
        if idx == self.ref_idx:
            return image

        return cv2.warpPerspective(image, self.matrices[idx], self.shapes[idx], flags=cv2.INTER_NEAREST)
