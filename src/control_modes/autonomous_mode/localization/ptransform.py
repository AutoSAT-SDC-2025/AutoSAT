import numpy as np
import cv2 as cv

def get_perspective_matrix(src_points: np.array):

    # src_points = np.array(
    #     [
    #         [200, 300],  # Top-left corner
    #         [590, 300],  # Top-right corner
    #         [847, 425],  # Bottom-right corner
    #         [0, 425],  # Bottom-left corner
    #     ],
    #     dtype="float32",
    # )

    # Calculate the width and height of the transformed image
    width_top = np.linalg.norm(src_points[0] - src_points[1])
    width_bottom = np.linalg.norm(src_points[3] - src_points[2])
    width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(src_points[0] - src_points[3])
    height_right = np.linalg.norm(src_points[1] - src_points[2])
    height = int(max(height_left, height_right))
    height = height*2.7

    # Define destination points for the top-down view
    dst_points = np.array(
        [
            [0, 0],  # Top-left
            [width, 0],  # Top-right
            [width, height],  # Bottom-right
            [0, height],  # Bottom-left
        ],
        dtype="float32",
    )

    print(dst_points)

    matrix = cv.getPerspectiveTransform(src_points, dst_points)
    return matrix
