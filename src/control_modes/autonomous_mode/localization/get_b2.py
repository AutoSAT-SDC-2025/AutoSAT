import numpy as np
import cv2 as cv
from src.multi_camera_calibration import CalibrationData, RenderDistance

LineDetectionDims = {
    'width': 720,
    'height': 720
}

data = CalibrationData(
    path="assets/calibration/latest.npz",
    input_shape=(1920, 1080),
    output_shape=(LineDetectionDims['width'], LineDetectionDims['height']),
    render_distance=RenderDistance(
        front=12.0,
        sides=6.0
    )
)

img = cv.imread("/home/thimo/Git/SDC/first.png")#, cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (1920, 1080))

shape = data.shape_stitched[::-1]
if len(img.shape) == 3:
    shape += (3,)

stitched = np.zeros(shape, dtype=np.uint8)
data._stitch_image(stitched, img, data.ref_idx)
A = np.array([[1,0,data.offsets[1,0]],[0,1,data.offsets[1,1]], [0,0,1]])
# img = cv.warpAffine(img, np.array([[1,0,data.offsets[1,0]],[0,1,data.offsets[1,1]]], dtype=np.float32), (stitched.shape[1], stitched.shape[0]))
# img2 = cv.warpPerspective(stitched, data.topdown_matrix, data.shape_topdown)
img2 = cv.warpPerspective(img, data.topdown_matrix@A, data.shape_topdown)
B = data.topdown_matrix@A

bottom_left = np.array([0, 1080,1])
bottom_right = np.array([1920, 1080,1])
bottom_left = B@bottom_left
bottom_right = B@bottom_right
bottom_left = bottom_left / bottom_left[2]
bottom_right = bottom_right / bottom_right[2]
print(bottom_left)
print(bottom_right)
dx = bottom_right[0] - bottom_left[0]

destination_width = 847
# destination_height = 285
destination_height = 847

ratio = destination_width / dx
height = destination_height / ratio
top_left = np.array([bottom_left[0], bottom_left[1]-height, 1])
src_points = np.array([[bottom_left[0], bottom_left[1]], [bottom_right[0], bottom_right[1]], [top_left[0], top_left[1]]], dtype=np.float32)
print(src_points)
dst_points = np.array([[0, destination_height],[destination_width, destination_height],[0,0]], dtype=np.float32)
A2 = cv.getAffineTransform(src_points, dst_points)
# img3 = cv.warpAffine(img2, A2, (destination_width,destination_height))
# A2 = np.vcat([A2, [0,0,1]])
A2 = np.vstack([A2,[0,0,1]])
B = A2@B
img3 = cv.warpPerspective(img, B, (destination_width, destination_height))
print(A2)
cv.imshow("final", img3)
cv.waitKey(0)
np.save("new_b.npy", B)
