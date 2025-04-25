import cv2

from src.multi_camera_calibration import CalibrationPattern, Calibrator

# Load the images
left_image = cv2.imread("./images/left.png")
center_image = cv2.imread("./images/center.png")
right_image = cv2.imread("./images/right.png")

# Create the calibration pattern
pattern = CalibrationPattern(10, 8, 0.115, 0.086, cv2.aruco.DICT_4X4_100)

# Calibrate the cameras
calibrator = Calibrator([left_image, center_image, right_image], pattern)
calibrator.calibrate()

# Save the results
calibrator.save("./calibration", keep_history=False)
