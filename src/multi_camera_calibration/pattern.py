import cv2
import numpy as np


class CalibrationPattern:
    """Represents a calibration pattern.

    Attributes
    ----------
        board: The calibration board.
        detector: The calibration pattern detector.
        height: The height of the calibration pattern.
        width: The width of the calibration pattern.

    """

    board: cv2.aruco.CharucoBoard
    detector: cv2.aruco.CharucoDetector
    height: int
    width: int

    def __init__(
            self,
            width: int,
            height: int,
            square_length: float,
            marker_length: float,
            aruco_dict: int
    ):
        """Initialize the calibration pattern.

        :param width: The width of the calibration pattern.
        :param height: The height of the calibration pattern.
        :param square_length: The length of the squares on the calibration pattern.
        :param marker_length: The length of the markers on the calibration pattern.
        :param aruco_dict: The dictionary for the calibration pattern.
        """
        self.width = width
        self.height = height

        # Initialize the ChArUco board and detector
        dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.board = cv2.aruco.CharucoBoard(
            self.get_shape(),
            square_length,
            marker_length,
            dictionary
        )

        self.detector = cv2.aruco.CharucoDetector(
            self.board,
            cv2.aruco.CharucoParameters(),
            cv2.aruco.DetectorParameters()
        )

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Detect the ChArUco board in an image.

        :param image: The image to detect the ChArUco board in.
        :return: The corners and ids of the detected ChArUco board.
        """
        charuco_corners, charuco_ids, _, _ = self.detector.detectBoard(image)
        return charuco_corners, charuco_ids

    def get_shape(self) -> tuple[int, int]:
        """Get the shape of the ChArUco board.

        :return: The shape of the ChArUco board.
        """
        return self.width, self.height
