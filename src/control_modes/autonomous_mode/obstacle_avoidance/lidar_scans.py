import numpy as np
import matplotlib.pyplot as plt

from rplidar import RPLidar
from math import floor

def iter_scans(lidar):
    """ Generator to yield LIDAR scans. """
    scan_list = []
    for new_scan, quality, angle, distance in lidar.iter_measures():
        if new_scan:
            if len(scan_list) > 5:
                yield scan_list  # Yield scan data if it has enough points
            scan_list = []  # Reset for new scan

        scan_list.append((quality, angle, distance))

def coordinate_conversion(angle, distance):
    """Convert polar coordinates (angle, distance) to rotated and mirrored Cartesian (x, y).
       - Rotate -90° to place 0° at bottom
       - Flip x-axis to correct left/right
    """
    angle_rad = np.radians(angle - 90)
    x = -distance * np.cos(angle_rad)  # Negate x to flip left/right
    y = distance * np.sin(angle_rad)
    return x, y

def determine_closest(scan):
    """Find and print the closest object distance from the scan data."""
    scan_data = np.full(360, np.inf)

    for _, angle, distance in scan:
        angle_idx = min(359, floor(angle))

        if distance < 150:  # Ignore very close objects
            scan_data[angle_idx] = np.inf
        else:
            scan_data[angle_idx] = distance

    closest_distance = np.min(scan_data)
    print(f"Closest object distance: {closest_distance:.2f} mm")

def detect_objects(lidar):
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-6000, 6000)
    ax.set_ylim(-6000, 6000)
    ax.set_title("LIDAR Scan - Real-Time Points")

    scatter = ax.scatter([], [], s=2)

    try:
        for scan in iter_scans(lidar):
            x_points = []
            y_points = []

            for _, angle, distance in scan:
                if distance > 150:  # Filter out very close points
                    x, y = coordinate_conversion(angle, distance)
                    x_points.append(x)
                    y_points.append(y)

            scatter.set_offsets(np.c_[x_points, y_points])
            plt.draw()
            plt.pause(0.05)

            determine_closest(scan)

    except KeyboardInterrupt:
        print("Stopping LIDAR")

    finally:
        lidar.stop()
        lidar.disconnect()

if __name__ == "__main__":
    lidar = RPLidar("COM3")  #Set the correct COM port
    detect_objects(lidar)