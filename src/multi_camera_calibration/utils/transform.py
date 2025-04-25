import cv2
import numpy as np

from src.multi_camera_calibration.utils.other import get_border_of_points
from src.multi_camera_calibration.utils.types import Shape


def get_transformed_corners(matrix: np.ndarray, shape: Shape) -> tuple[float, float, float, float]:
    """Get the transformed corners of the image.

    :param matrix: The perspective matrix.
    :param shape: The shape of the image (width, height).
    :return: The transformed corners of the image.
    """
    w, h = shape
    src_points = np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype=np.float32)
    dst_points = cv2.perspectiveTransform(src_points, matrix)

    return get_border_of_points(dst_points[:, 0])

def get_transformed_shape(matrix: np.ndarray, shape: Shape) -> Shape:
    """Get the transformed shape of the image.

    :param matrix: The perspective matrix.
    :param shape: The shape of the image (width, height).
    :return: The transformed shape of the image.
    """
    min_x, min_y, max_x, max_y = get_transformed_corners(matrix, shape)

    width = int(max_x - min_x)
    height = int(max_y - min_y)

    return height, width

def stitch_images(base_image: np.ndarray, new_image: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Merge a new image onto a base image with the given offset.

    :param base_image: The base image onto which the new image will be merged.
    :param new_image: The new image to merge onto the base image.
    :param offset: The offset for the new image.
    :return: The merged image.
    """
    # Calculate dimensions for merging
    new_height, new_width = new_image.shape[:2]
    base_height, base_width = base_image.shape[:2]
    offset_x, offset_y = offset

    # Calculate the region of interest (ROI) for merging
    roi_top = max(offset_y, 0)
    roi_bottom = min(offset_y + new_height, base_height)
    roi_left = max(offset_x, 0)
    roi_right = min(offset_x + new_width, base_width)

    # Calculate the cropped region of the new image
    crop_top = roi_top - offset_y
    crop_bottom = crop_top + (roi_bottom - roi_top)
    crop_left = roi_left - offset_x
    crop_right = crop_left + (roi_right - roi_left)

    image = new_image[crop_top:crop_bottom, crop_left:crop_right]
    image_mask = image != 0

    # Merge the images
    base_image[roi_top:roi_bottom, roi_left:roi_right][image_mask] = image[image_mask]
    return base_image
