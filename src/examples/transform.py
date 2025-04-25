import cv2

from src.multi_camera_calibration import CalibrationData, RenderDistance

# Load the calibration data
data = CalibrationData(
    path="C:\\Users\\Gebruiker\\PycharmProjects\\AutoSAT\\src\\examples\\calibration\\latest.npz",
    input_shape=(1920, 1080),
    output_shape=(720, 720),
    render_distance=RenderDistance(
        front=12.0,
        sides=6.0
    )
)

# Load the images
left_image = cv2.imread("C:\\Users\\Gebruiker\\PycharmProjects\\AutoSAT\\src\\examples\\images\\left.png")
center_image = cv2.imread("C:\\Users\\Gebruiker\\PycharmProjects\\AutoSAT\\src\\examples\\images\\center.png")
right_image = cv2.imread("C:\\Users\\Gebruiker\\PycharmProjects\\AutoSAT\\src\\examples\\images\\right.png")

# Transform the images
stitched = data.stitch([left_image, center_image, right_image])
topdown = data.transform([left_image, center_image, right_image])

# Display the images
cv2.imshow("Stitched", cv2.resize(stitched, (720, 720 * stitched.shape[0] // stitched.shape[1])))
cv2.imshow("Top-Down", topdown)
cv2.waitKey(0)
cv2.destroyAllWindows()
