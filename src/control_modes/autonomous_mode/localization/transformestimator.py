import numpy as np
import cv2 as cv

class TransformEstimator:
    def __init__(self, threshold = 25, pixel_threshold=30, scale=0.5):
        self.threshold = threshold
        self.pixel_threshold = pixel_threshold
        self.distance = 753
        self.scale = 1/scale

    def estimate_transformation(self, original_points, transformed_points):
        
        if len(original_points) < self.threshold:
            return None
        
        affinemtx, inliers = cv.estimateAffinePartial2D(original_points, transformed_points)
        
        R = affinemtx[:2, :2]
        t = affinemtx[:, 2:]
        t = t*self.scale

        if abs(t[0][0]) + abs(t[1][0]) < self.pixel_threshold:
            return None, None, None

        return R, t, inliers

class TransformAngleEstimator:
    def __init__(self, distance=1900, threshold = 25, pixel_threshold=30, scale=0.5):
        self.threshold = threshold
        self.pixel_threshold = pixel_threshold
        self.circumferance = 2*np.pi*distance
        self.distance = distance
        self.A = np.identity(3)
        self.scale = 1/scale

    def translation_to_rotation(self, translation, distance):
        theta = translation[0]/distance

        R = np.array([
            [np.cos(theta)[0], -np.sin(theta)[0]],
            [np.sin(theta)[0], np.cos(theta)[0]]
        ])
        return R
        

    def estimate_transformation(self, original_points, transformed_points):
        
        if len(original_points) < self.threshold:
            return None, None, None
        
        affinemtx, inliers = cv.estimateAffinePartial2D(original_points, transformed_points)
        inliers = np.reshape(inliers, newshape=(len(inliers))).astype(bool)

        dx = affinemtx[0,2]*self.scale
        dy = affinemtx[1,2]*self.scale
        affinemtx[0,2] = dx
        affinemtx[1,2] = dy
        
        t = np.array([[dx], [dy]])
        
        R = self.translation_to_rotation(t, self.distance)

        if abs(t[0][0]) + abs(t[1][0]) < self.pixel_threshold:
            # print("didnt make pixel threshold")
            return (None, None, None)

        self.A = affinemtx

        return R, t, inliers
