#from os import wait
from typing import Any
import cv2 as cv
import numpy as np
import configparser
from pathlib import Path

class LaneDetector():
    def __init__(self) -> None:
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[3]  # adjust this number as needed
        config = configparser.ConfigParser()
        config.read(project_root/"config"/"config.ini")
        self.pmatrix = np.load(config["LaneDetection"]["transformation"])

    
    def __call__(self, img: np.array) -> Any:
        width, height = (img.shape[1]+1200, img.shape[0]+500)
        img = cv.warpPerspective(img, self.pmatrix, (width, height))
        img = cv.GaussianBlur(img, (5,5), 0)
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        
        lower = np.array([0, 0, 170])    # Lower bound of road color in HSV
        upper = np.array([180, 255, 255]) # Upper bound of road color in HSV
    
        mask = cv.inRange(hsv, lower, upper)
        return mask
