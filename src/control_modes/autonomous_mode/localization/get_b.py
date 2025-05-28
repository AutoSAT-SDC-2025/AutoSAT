import numpy as np
import cv2 as cv


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

data = np.load("/home/thimo/Seafile/Git/AutoSAT/assets/calibration/latest.npz")
data = np.load("/home/thimo/Downloads/latest.npz")
# print(data["ref_idx"])
# print(data["offsets"])
# print(data["matrices"])
# print(data["shape_stitched"])
shape = np.array([0,0], dtype=np.int64)
shape[0] = int(data["shape_stitched"][0] * 1920)
shape[1] = int(data["shape_stitched"][1] * 1080)


topdown_matrix = data["topdown_matrix"]
offsets = data["offsets"]
h = 1080
w = 1920
offsets[:,0] = offsets[:,0]*w
offsets[:,1] = offsets[:,1]*h
offsets = np.array(offsets, dtype=np.int32)
print(offsets)

A = np.array([[1,0,offsets[1,0]],[0,1,offsets[1,1]]], dtype=np.float32)

img = cv.imread("/home/thimo/Git/SDC/first.png")#, cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (1920, 1080))
# w, h, _ = img.shape
# h, w, _ = img.shape

# print(w,h)
scale_matrix = np.array([
    [w, 0, 0],
    [0, h, 0],
    [0, 0, 1]
])
adjusted_matrix = np.array([
    [1,0,-1000],
    [0,1,-1000],
    [0,0,1]
])
scale_matrix_inv = np.linalg.inv(scale_matrix)
topdown_matrix = scale_matrix @ topdown_matrix @ scale_matrix_inv
topdown_matrix = adjusted_matrix @ topdown_matrix
print(topdown_matrix)
# print(shape)
# print(offsets[1])

# if len(img.shape) == 3:
    # shape += (3,)
# stitched = np.zeros(shape, dtype=np.uint8)

# img = stitch_images(stitched, img, offsets[1])
# print(A)

# img = cv.warpAffine(img, A, dsize=(shape[0], shape[1]))
B = np.eye(3)
img2 = cv.warpPerspective(img, topdown_matrix, (50000,50000))
img2 = cv.resize(img2, (1000,1000))
# img2 = cv.warpPerspective(img, B, (5000,5000), flags=cv.INTER_NEAREST)
cv.imshow("result", img2)
cv.imshow("img", img)
cv.waitKey(0)
print(data["shape_topdown"])