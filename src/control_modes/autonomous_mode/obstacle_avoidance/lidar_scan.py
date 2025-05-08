from rplidar import RPLidar
from math import floor
import math
import matplotlib.pyplot as plt
import numpy as np

class LidarScans:
    def __init__(self, port='COM3'):
        self.lidar = RPLidar(port)
        self.focal_length = 540
        self.image_height = 1080
        self.image_width = 1920

    def iter_scans(self):
        lidar_scan = []
        try:
            for new_scan, _, angle, distance in self.lidar.iter_measures():
                if new_scan:
                    if len(lidar_scan) > 5:
                        yield lidar_scan
                    lidar_scan = []

                lidar_scan.append((angle, distance))

        except KeyboardInterrupt:
            print("Stopping Lidar")

        finally:
            self.lidar.stop()
            self.lidar.disconnect()

    def coordinate_conversion(self, angle, distance):
        radians = math.radians(angle)
        x = (distance / 1000.0) * math.cos(radians)
        y = (distance / 1000.0) * math.sin(radians)
        return x, y

    def homogeneous_coordinates(self, x, y):
        k = np.array([
            [self.focal_length, 0, self.image_width / 2],
            [0, self.focal_length, self.image_height / 2],
            [0, 0, 1]
        ])

        t = np.array([[0], [0.3], [0.5]])
        R = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]])

        extrinsics = np.hstack((R, t))

        homogeneous_point = np.array([x, y, 0, 1])
        projection_matrix = np.dot(k, extrinsics)
        projected_2d_homogeneous = np.dot(projection_matrix, homogeneous_point)

        u = projected_2d_homogeneous[0] / projected_2d_homogeneous[2]
        v = projected_2d_homogeneous[1] / projected_2d_homogeneous[2]
        return u, v

if __name__ == "__main__":
    main()