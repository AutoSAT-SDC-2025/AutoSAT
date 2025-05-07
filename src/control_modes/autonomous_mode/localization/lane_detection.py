from typing import Any
import cv2 as cv
import numpy as np

class LaneDetector():
    def __init__(self, perspective_matrix) -> None:
        self.pmatrix = perspective_matrix
    
    def __call__(self, img: np.array) -> Any:
        width, height = (img.shape[1]+1200, img.shape[0]+500)
        img = cv.warpPerspective(img, self.pmatrix, (width, height))
        img = cv.GaussianBlur(img, (5,5), 0)
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        
        lower = np.array([0, 0, 170])    # Lower bound of road color in HSV
        upper = np.array([180, 255, 255]) # Upper bound of road color in HSV
    
        mask = cv.inRange(hsv, lower, upper)
        return mask
