from abc import ABC, abstractmethod
import cv2
import numpy as np

class KeypointMatcher(ABC):
    """Abstract base class for keypoint matching in OpenCV."""
    
    @abstractmethod
    def find_keypoints(self, img):
        """Match descriptors from two images."""
        pass

class OrbKeyPointMatcher(KeypointMatcher):
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    def find_keypoints(self, img):
        """
        Find keypoint matches between two images using ORB

        Parameters:
            img (numpy.ndarray): The first image.

        Returns:
            Tuple[cv2.KeyPoint, cv2.Descriptors]: The keypoints and their corresponding descriptors from the image.
        """
        kp, des = self.orb.detectAndCompute(img, None)
        return kp, des

    def match_keypoints(self, des1, des2):
        """
        Match keypoints between two images using BFMatcher
        Parameters:
           des1 (numpy.ndarray): The descriptors from the first image.
           des2 (numpy.ndarray): The descriptors from the second image.
           Returns:
           List[cv2.DMatch]: A list of matches between two images.
        """
        matches = self.matcher.match(des1, des2)
        return matches

    def draw_matches(self, img1, kp1, img2, kp2, matches):
        """
        Draw the matched keypoints between two images
        Parameters:
          img1  (numpy.ndarray): The first image.
          kp1 (cv2.KeyPoint): The keypoints from the first image.
          img2  (numpy.ndarray): The second image.
          kp2  (cv2.KeyPoint): The keypoints from the second image.
          matches (List[cv2.DMatch]): The matches between the two images.
          Returns:
          numpy.ndarray: The image with the matched keypoints drawn on it.
        """
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:50],None,flags=2)
        return img3
    
class StarKeyPointMatcher(KeypointMatcher):
    def __init__(self, width, height):
        """"For Ranger, we extract up to 1250 keypoint objects with CenSurE (in OpenCV called StarDetector), with a maximum patch size of 14, a response threshold of 0, a projected line threshold value of 29, a binarized threshold value of 22, and a non-maximum suppression size of 2. Keypoint objects are described using the rotation invariant 64-byte BRIEF descriptor. For localization, Ranger selects the closest reference image to the prior and matches its features with the query image features, using the OpenCV brute force matcher with Hamming norm and cross check. Subsequently, the camera pose is estimated in a RANSAC procedure. If there are at least 25 RANSAC inliers, the localization terminates; otherwise, matching and pose estimation are repeated with the next closest reference image. In case that none of the considered reference images satisfies the condition, we use the attempt that had the most RANSAC inliers."""
        self.detector = cv2.xfeatures2d.StarDetector.create(maxSize=14, responseThreshold=0, lineThresholdProjected=29, lineThresholdBinarized=22, suppressNonmaxSize=2)

        self.descriptor = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.width = int(width)
        self.height = int(height)

    def find_keypoints(self, img):
        """
        Find keypoint matches between two images using ORB

        Parameters:
            img (numpy.ndarray): The first image.

        Returns:
            Tuple[cv2.KeyPoint, cv2.Descriptors]: The keypoints and their corresponding descriptors from the image.
        """
        kp = self.detector.detect(img, None)
        kp, des = self.descriptor.compute(img, kp)
        return kp, des

    def match_keypoints(self, des1, des2):
        """
        Match keypoints between two images using BFMatcher
        Parameters:
           des1 (numpy.ndarray): The descriptors from the first image.
           des2 (numpy.ndarray): The descriptors from the second image.
           Returns:
           List[cv2.DMatch]: A list of matches between two images.
        """
        matches = self.matcher.match(des1, des2)
        return matches

    def draw_matches(self, img1, kp1, img2, kp2, matches, max_matches=100):
        """
        Draw the matched keypoints between two images
        Parameters:
          img1  (numpy.ndarray): The first image.
          kp1 (cv2.KeyPoint): The keypoints from the first image.
          img2  (numpy.ndarray): The second image.
          kp2  (cv2.KeyPoint): The keypoints from the second image.
          matches (List[cv2.DMatch]): The matches between the two images.
          Returns:
          numpy.ndarray: The image with the matched keypoints drawn on it.
        """
        if len(matches) > max_matches:
            # matches = matches[:max_matches]
            matches = np.random.choice(matches, size=max_matches, replace=False)
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,flags=2)
        return img3
    
class SiftKeyPointMatcher(KeypointMatcher):
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()

    def find_keypoints(self, img):
        """
        Find keypoint matches between two images using ORB

        Parameters:
            img (numpy.ndarray): The first image.

        Returns:
            Tuple[cv2.KeyPoint, cv2.Descriptors]: The keypoints and their corresponding descriptors from the image.
        """
        kp, des = self.sift.detectAndCompute(img, None)
        return kp, des

    def match_keypoints(self, des1, des2):
        """
        Match keypoints between two images using BFMatcher
        Parameters:
           des1 (numpy.ndarray): The descriptors from the first image.
           des2 (numpy.ndarray): The descriptors from the second image.
           Returns:
           List[cv2.DMatch]: A list of matches between two images.
        """
        matches = self.matcher.match(des1, des2)
        return matches

    def draw_matches(self, img1, kp1, img2, kp2, matches):
        """
        Draw the matched keypoints between two images
        Parameters:
          img1  (numpy.ndarray): The first image.
          kp1 (cv2.KeyPoint): The keypoints from the first image.
          img2  (numpy.ndarray): The second image.
          kp2  (cv2.KeyPoint): The keypoints from the second image.
          matches (List[cv2.DMatch]): The matches between the two images.
          Returns:
          numpy.ndarray: The image with the matched keypoints drawn on it.
        """
        if len(matches) > 100:
            matches = matches[:100]
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,flags=2)
        return img3
    
if __name__ == "__main__":
    detector = StarKeyPointMatcher()
