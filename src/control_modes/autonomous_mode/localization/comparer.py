import cv2 as cv
import numpy as np
from scipy.spatial import KDTree
from collections import OrderedDict
from functools import wraps

def skeletonize(lane):
    skeleton = np.zeros(lane.shape, np.uint8)
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
    _, lane = cv.threshold(lane, 127, 255, cv.THRESH_BINARY)
    done = False
    while not done:
        eroded = cv.erode(lane, element)
        temp = cv.dilate(eroded, element)
        temp = cv.subtract(lane, temp)
        skeleton = cv.bitwise_or(skeleton, temp)
        lane = eroded.copy()

        if cv.countNonZero(lane) == 0:
            done = True
    return skeleton

def id_lru_cache(maxsize=128, arg_index=1):
    """
    Decorator for methods that caches results based on the id() of a specific argument.
    By default, uses the second argument (arg_index=1), assuming the first is `self`.
    """
    def decorator(func):
        cache = OrderedDict()

        @wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) <= arg_index:
                raise ValueError(f"Expected at least {arg_index + 1} positional arguments")

            obj = args[arg_index]
            key = id(obj)

            if key in cache:
                cache.move_to_end(key)
                return cache[key]

            result = func(*args, **kwargs)
            cache[key] = result
            cache.move_to_end(key)

            if len(cache) > maxsize:
                cache.popitem(last=False)

            return result

        return wrapper
    return decorator

class Comparitor:
    def __init__(self) -> None:
        winsize = (128, 64)
        blocksize = (8, 8)
        blockstride = (4, 4)
        cellsize = (4,4)
        nbins = 9
        self.hog = cv.HOGDescriptor(winsize, blocksize, blockstride, cellsize, nbins)

    def __call__(self, img1, img2):
        features1 = self.get_hog_features(img1)
        features2 = self.get_hog_features(img2)
        dist = np.linalg.norm(features1 - features2)
        return dist

    # @id_lru_cache(maxsize=4)
    def get_hog_features(self, img):
        img = cv.resize(img, (128,64))
        _, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        features = self.hog.compute(img)
        return features

    def get_distance_features(self, features1, features2):
        return np.linalg.norm(features1 - features2)

    def trust_score(self, img):
        """Return a score between 0 and 1"""
        print(np.sum(img))
        return min(np.sum(img)/113636, 1)

    
class KDTreeComparitor:
    def __init__(self):
        pass

    def __call__(self, img1, img2):
        img1_sk = skeletonize(img1)
        img2_sk = skeletonize(img2)
        list1 = np.column_stack(np.where(img1_sk == 255))
        list2 = np.column_stack(np.where(img2_sk == 255))
        
        # Build a KD-Tree for list2
        tree = KDTree(list2)
        
        total_distance = 0
        for point in list1:
            # Find the nearest neighbor in list2 using the KD-Tree
            distance, _ = tree.query(point)
            total_distance += distance
        
        # Return the average distance (dissimilarity score)
        return total_distance / ((len(list1)+len(list2))//2)
