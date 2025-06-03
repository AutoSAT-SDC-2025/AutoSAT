from os import wait
import cv2 as cv
import numpy as np
from .keypointmatcher import StarKeyPointMatcher
from .transformestimator import TransformAngleEstimator
from .particlefilter import ParticleFilter
from .mapper import Mapper
import multiprocessing as mp 
from .lane_detection import LaneDetector
import configparser
from pathlib import Path

class Localizer:
    def __init__(self, **kwargs):
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[3]  # adjust this number as needed
        config = configparser.ConfigParser()
        config.read(project_root/"config"/"config.ini")
        self.perspective_matrix = np.load(project_root/config["Localizer"]["transformation"])
        self.width = int(config["Localizer"]["width"])
        self.height = int(config["Localizer"]["height"])
        self.scale = float(config["Localizer"]["scale"])
        self.distance = int(config["Localizer"]["distance"])
        
        self.x = float(config["Main"]["start_x"])
        self.y = float(config["Main"]["start_y"])
        self.theta = float(config["Main"]["start_theta"])
        
        self.mapper = Mapper()
        self.x = self.x/self.mapper.scale
        self.y = self.y/self.mapper.scale

        self.particle_filter = ParticleFilter()
        # self.set_start_location(x, y, theta)
        self.matcher = StarKeyPointMatcher(width=self.width//self.scale, height=self.height//self.scale)
        self.transform_estimator = TransformAngleEstimator(pixel_threshold=30, distance=self.distance, scale=self.scale)

        self.rotation = np.eye(2, dtype=np.float32)
        self.initialized = False
        self.method = 1

        vars(self).update(kwargs)
    
    def set_start_location(self, x, y, theta, lane):
        pf = ParticleFilter()
        pf.amount = 2000
        pf.std = [100, 100, 0.1]
        print(x, y, theta)
        pf.spawn_new_particles(x, y, theta)
        pf._update_particles(0,0,0, std=pf.std)
        particle, _, _ = pf.find_location(lane)
        x = particle[0]
        y = particle[1]
        theta = particle[2]
        self.set_location(x, y, theta)
        print(x, y, theta)
        self.particle_filter.spawn_new_particles(x, y, theta)

    def set_location(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self._x = self.x - self.distance * np.sin(theta)
        self._y = self.y + self.distance * np.cos(theta)
        self.rotation = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

    def get_transformation(self, img):
        keypoints1, des1 = self.matcher.find_keypoints(img)
        keypoints2, des2 = (self.kp, self.des)
        match_objects = self.matcher.match_keypoints(des1, des2)

        src_points = np.array([keypoints1[match_object.queryIdx].pt for match_object in match_objects], dtype=np.float32)
        dst_points = np.array([keypoints2[match_object.trainIdx].pt for match_object in match_objects], dtype=np.float32)
        
        rot, translation, inliers = self.transform_estimator.estimate_transformation(dst_points, src_points)
        if translation is None:
            return None
        else:
            self.kp = keypoints1
            self.des = des1
            return rot, translation, inliers

    def update_location(self, rot, translation):
        theta = -(np.arctan2(rot[1, 0], rot[0, 0]))
        rot = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        self.rot = rot
        self.rotation = rot @ self.rotation
        self.theta = (np.arctan2(self.rotation[1, 0], self.rotation[0, 0]))

        translation[1][0] = translation[1][0]*1.1
        self.translation = translation

        if self.method == 0:
            translation[0][0] = -translation[0][0] # positive x means to the left
            translation = self.rotation @ translation
            self._x += translation[0][0]
            self._y += translation[1][0]
            self.x = self._x + self.distance * np.sin(self.theta)
            self.y = self._y - self.distance * np.cos(self.theta)

        elif self.method == 1:
            translation[0][0] = 0 # positive x means to the left
            translation = self.rotation @ translation
            self.x += translation[0][0]
            self.y += translation[1][0]
            self._x = self.x - self.distance * np.sin(self.theta)
            self._y = self.y + self.distance * np.cos(self.theta)

    def update(self, img: np.ndarray, lane=None):
        if self.initialized is not True:
            img = self.preprocess(img)
            self.kp, self.des = self.matcher.find_keypoints(img)
            self.translation = None
            self.initialized = True
            if lane is not None:
                self.set_start_location(self.x, self.y, self.theta, lane)
            return
        x_old = self.x
        y_old = self.y
        theta_old = self.theta
        img = self.preprocess(img)
        if (result := self.get_transformation(img)) is None:
            return
        rot, translation, inliers = result
        self.update_location(rot, translation)
        dx = self.x - x_old
        dy = self.y - y_old
        dtheta = self.theta - theta_old

        # if lane is None, don't use the map to localize
        if lane is None:
            return
        # self.particle_filter.update(lane, dx, dy, dtheta)
        self.particle_filter.update(lane, translation[1][0], dtheta)
        self.set_location(self.particle_filter.x, self.particle_filter.y, self.particle_filter.theta)
            
    def preprocess(self, img: np.array):
        img = cv.warpPerspective(img, self.perspective_matrix, (self.width, self.height))
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.GaussianBlur(img,(5,5),0)
        # img = cv.GaussianBlur(img,(15,15),0)
        img = cv.resize(img, None, fx=self.scale, fy=self.scale, interpolation= cv.INTER_LINEAR)
        cv.imshow("Localization topdown view", img)
        return img

def localization_worker(shared):
    localizer = Localizer()
    lane_detector = LaneDetector()
    img = None
    while True:
        if not np.array_equal(shared.img, img):
            img = shared.img.copy()
            frame = cv.resize(img, (848, 480))
            lane = lane_detector(frame)
            cv.imshow("lane", lane)
            localizer.update(img, lane)
            shared.x = localizer.x
            shared.y = localizer.y
            shared.theta = localizer.theta


if __name__ == "__main__":
    from pathlib import Path

    # Assuming this script is somewhere inside the project
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[3]  # adjust this number as needed
    print(project_root)
    import time
    mg = mp.Manager()
    shared = mg.Namespace()
    shared.img = None
    shared.x = None
    shared.y = None
    shared.theta = None
    localization_process = mp.Process(target=localization_worker, args=(shared,))
    localization_process.start()
    time.sleep(1)
