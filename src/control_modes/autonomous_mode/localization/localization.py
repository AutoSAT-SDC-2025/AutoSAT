import cv2 as cv
import numpy as np
from keypointmatcher import StarKeyPointMatcher
from transformestimator import TransformAngleEstimator
from particlefilter import ParticleFilter
from mapper import Mapper

class Localizer:
    def __init__(self, perspective_matrix, init_img: np.ndarray, transform_estimator=None, distance=1730, **kwargs):
        self.perspective_matrix = perspective_matrix
        self.width = 847
        self.height = 285
        self.scale = 0.5
        self.distance = distance

        self._x = 0 # camera location
        self._y = 0 # camera location
        self.x = 0 # car location
        self.y = 0 # car location
        self.translation = None
        self.theta = 0
        self._theta = 0

        map = cv.imread("../var/map.png", cv.IMREAD_GRAYSCALE)
        
        self.mapper = Mapper(scale=0.0483398, map=map)
        self.particle_filter = ParticleFilter(5640*(1/self.mapper.scale),350*(1/self.mapper.scale),np.pi/2,10,50,0.05, 100)

        if transform_estimator is None:
            self.transform_estimator = TransformAngleEstimator(pixel_threshold=30, distance=self.distance, scale=self.scale)
        else:
            self.transform_estimator = transform_estimator

        self.rotation = np.eye(2, dtype=np.float32)
        init_img = self.preprocess(init_img)
        
        self.matcher = StarKeyPointMatcher(width=self.width//self.scale, height=self.height//self.scale)
        self.kp, self.des = self.matcher.find_keypoints(init_img)

        vars(self).update(kwargs)
    
    def set_start_location(self, x, y, theta):
        self.set_location(x, y, theta)
        self.particle_filter.spawn_new_particles(self.x, self.y, theta)

    def set_location(self, x, y, theta):
        self.x = x
        self.y = y
        self._theta = theta
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
        theta = (np.arctan2(rot[1, 0], rot[0, 0]))
        rot = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        self.rot = rot
        self.theta = theta
        self.rotation = rot @ self.rotation
        self._theta = (np.arctan2(self.rotation[1, 0], self.rotation[0, 0]))

        self.translation = translation
        translation[0][0] = -translation[0][0] # positive x means to the left
        # translation[0][0] = 0 # positive x means to the left

        translation = self.rotation @ translation
        
        self._x += translation[0][0]
        self._y += translation[1][0]
        self.x = self._x + self.distance * np.sin(self._theta)
        self.y = self._y - self.distance * np.cos(self._theta)
        # print(translation)
        # self.x += translation[0][0]
        # self.y += translation[1][0]
        # self._x = self.x - self.distance * np.sin(self._theta)
        # self._y = self.y + self.distance * np.cos(self._theta)

    # def map_lookup(self, x, y, theta, dx, dy, dtheta):
    #     self.particle_filter.update(x, y, theta, dx, dy, dtheta)
    #     for particle in self.particle_filter.particles:
    #         position = (particle[0],particle[1])
    #         rotation = np.array([[np.cos(particle[2]), -np.sin(particle[2])], [np.sin(particle[2]), np.cos(particle[2])]])
    #         img = self.mapper.get_sight(position, rotation)
    #         cv.imshow("map img", img)
    #     pass

    def update(self, img: np.ndarray, lane=None):
        x_old = self.x
        y_old = self.y
        theta_old = self._theta
        img = self.preprocess(img)
        if (result := self.get_transformation(img)) is None:
            return
        rot, translation, inliers = result
        self.update_location(rot, translation)
        dx = self.x - x_old
        dy = self.y - y_old
        dtheta = self._theta - theta_old

        # if lane is None, don't use the map to localize
        if lane is None:
            return
        self.particle_filter.update(lane, dx, dy, dtheta)
        self.set_location(self.particle_filter.x, self.particle_filter.y, self.particle_filter.theta)
                
            
    def preprocess(self, img: np.array):
        img = cv.warpPerspective(img, self.perspective_matrix, (self.width, self.height))
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.GaussianBlur(img,(5,5),0)
        img = cv.resize(img, None, fx=self.scale, fy=self.scale, interpolation= cv.INTER_LINEAR)
        return img
