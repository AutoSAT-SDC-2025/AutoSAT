import cv2 as cv
import numpy as np
from utils import rotate_img, apply_affine
import configparser

class Mapper:
    def __init__(self) -> None:
        config = configparser.ConfigParser()
        config.read("config.ini")
        map = cv.imread(config["Mapper"]["map"], cv.IMREAD_GRAYSCALE)
        self.map = map
        self.offset = np.array([5000,0])
        self.pose_map_list = []
        self.scale = float(config["Mapper"]["scale"])
        self.distance = int(config["Localizer"]["distance"]) * self.scale
        self.width = 98
        self.height = 47
    
    def update(self, map, position, rotation, mask):
        # Flip X,Y coordinates
        # map = cv.rotate(map, cv.ROTATE_180)
        # rotation = np.linalg.inv(rotation)
        rotation = np.array([[-1,0],[0,-1]])@np.linalg.inv(rotation)
        map, T = rotate_img(map, rotation)
        mask, _ = rotate_img(mask, rotation)
        mask = mask.astype(np.bool_)
        cv.imshow("map input", cv.resize(map, None, fx=0.25, fy=0.25))
        height = map.shape[0] # y
        width = map.shape[1] # x
        # assume top left corner is at the location for simplicity
        x, y = position
        x = -x
        position = (x+int(T[0,2]), y+int(T[1,2]))
        x, y = self.get_index(*position)
        right = max(-self.map.shape[1] + x + width, 0)
        bottom = max(-self.map.shape[0] + y + height, 0)
        left = max(-x,0)
        top = max(-y, 0)
        self.expand_map(top, bottom, left, right)
        x, y = self.get_index(*position)
        print("T", T)
        self.map[y:y+height, x:x+width][mask] = map[mask]
    
    
    def update_car_location(self, map, position, rotation, mask):
        map = cv.resize(map, None, fx=self.scale, fy=self.scale)
        mask = cv.resize(mask, None, fx=self.scale, fy=self.scale)
        x, y = position
        x = int(x*self.scale)
        y = int(y*self.scale)
        position = (x, y)
        self._update_car_location(map, position, rotation, mask)
        
        
    def _update_car_location(self, map, position, rotation, mask):
        map = cv.rotate(map, cv.ROTATE_180)
        width = map.shape[1]
        rotation = np.linalg.inv(rotation)
        t = rotation @ np.array([[-width//2],[self.distance]])
        affine = np.hstack([rotation, t])
        map, T = apply_affine(map, affine)
        mask, _ = rotate_img(mask, rotation)
        mask = mask.astype(np.bool_)
        height = map.shape[0] # y
        width = map.shape[1] # x
        x, y = position
        x = -x
        position = (x+int(T[0,2]), y+int(T[1,2]))
        x, y = self.get_index(*position)
        right = max(-self.map.shape[1] + x + width, 0)
        bottom = max(-self.map.shape[0] + y + height, 0)
        left = max(-x,0)
        top = max(-y, 0)
        self.expand_map(top, bottom, left, right)
        x, y = self.get_index(*position)
        print("T", T)
        self.map[y:y+height, x:x+width][mask] = map[mask]

    def get_sight(self, position, angle):
        # get real width
        x, y = position
        x *= self.scale
        y *= self.scale
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotation_inv = np.linalg.inv(rotation)
        t = rotation @ np.array([[self.width//2], [-self.distance]])
        t[0,0] += -x
        t[1,0] += -y
        affine = np.hstack([np.identity(2), t])
        affine = np.vstack([affine , [0,0,1]])
        temp = rotation_inv
        rotation = np.identity(3)
        rotation[:2,:2] = temp
        affine = rotation @ affine
        img = cv.warpAffine(self.map, affine[:2,:], (self.width,self.height))
        return cv.flip(img, -1)
        # return img

    def add_pose(self, map, pose):
        self.pose_map_list.append((map, pose))
    
    def generate_map(self):
        self.map = np.zeros((10,10), dtype=np.uint8)
        for pose_map in self.pose_map_list:
            map, pose = pose_map
            self.update(map, pose)
    
    def expand_map(self, top=0, bottom=0, left=0, right=0):
        self.offset[1] += top
        self.offset[0] += left
        self.map = np.pad(self.map, ((top, bottom), (left, right)))
        
    def get_index(self, x, y):
        return x+self.offset[0], y+self.offset[1]

    def get_map_with_car(self, x, y, theta, scale=0.25):
        # scale = scale*self.scale
        map_with_car = cv.resize(self.map, None, fx=scale, fy=scale)
        map_with_car = cv.cvtColor(map_with_car, cv.COLOR_GRAY2BGR)
        start_point = (int(x*scale*self.scale), int(y*scale*self.scale))
        x, y = start_point
        length = self.distance*scale
        end_point = (int(x-np.sin(theta)*length), int(y+np.cos(theta)*length))
        color = (0, 2550, 0)
        thickness = int(9*scale)
        return cv.arrowedLine(map_with_car, start_point, end_point, color, thickness, tipLength=0.5)
