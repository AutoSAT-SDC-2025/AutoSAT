# line_processing.py
import numpy as np
from shapely.geometry import LineString
from itertools import combinations

from src.control_modes.autonomous_mode.line_detection.utils import getColorMask, getRoiMask, filterContours, filterWhite
import cv2

def clusterLines(lines, th_dis, th_ang):
    """
    Groups similar lines into clusters based on spatial proximity and angle similarity.

    Args:
        lines: List of lines, each in the format [[[x1, y1, x2, y2]], ...] as returned by HoughLinesP.
        th_dis: Distance threshold (in pixels) for clustering lines together.
        th_ang: Angle threshold (in degrees) for clustering lines together.

    Returns:
        clusters: List of clusters, each a list of lines that are close and similarly oriented.
    """
    if lines is None:
        return 0  # No lines to cluster

    cluster_total = 0  # Total number of clusters found
    cluster_id = np.zeros(len(lines), dtype=int)  # Cluster assignment for each line

    # Compare every pair of lines to determine if they should be clustered
    for i, j in combinations(enumerate(lines), 2):
        x1i, y1i, x2i, y2i = i[1][0]
        l1 = LineString([(x1i, y1i), (x2i, y2i)])  # Convert to shapely LineString for distance calculation
        x1j, y1j, x2j, y2j = j[1][0]
        l2 = LineString([(x1j, y1j), (x2j, y2j)])

        distance = l1.distance(l2)  # Minimum distance between the two lines

        # Calculate the angle of each line in degrees
        linepar1 = np.polyfit((x1i, x2i), (y1i, y2i), 1)
        angdeg1 = (180/np.pi) * np.arctan(linepar1[0])
        linepar2 = np.polyfit((x1j, x2j), (y1j, y2j), 1)
        angdeg2 = (180/np.pi) * np.arctan(linepar2[0])
        angdif = abs(angdeg1 - angdeg2)  # Absolute difference in angle

        # If both distance and angle are within thresholds, cluster the lines together
        if distance < th_dis and angdif < th_ang:
            if cluster_id[i[0]] == 0 and cluster_id[j[0]] == 0:
                cluster_total += 1
                cluster_id[i[0]] = cluster_total
                cluster_id[j[0]] = cluster_total
            elif cluster_id[j[0]] == 0:
                cluster_id[j[0]] = cluster_id[i[0]]

    # Assign unclustered lines to their own new clusters
    for count, id in enumerate(cluster_id):
        if id == 0:
            cluster_total += 1
            cluster_id[count] = cluster_total

    # Adjust cluster IDs to be zero-based
    cluster_id = cluster_id - 1

    # Group lines by their cluster assignment
    clusters = [[] for _ in range(cluster_total)]
    for i, line in enumerate(lines):
        clusters[cluster_id[i]].append(line)

    return clusters

def combineLines(lines):
    """
    Combines a cluster of similar lines into a single representative line.

    Args:
        lines: List of lines (each as [[x1, y1, x2, y2]]) belonging to the same cluster.

    Returns:
        line_new: A single line [x1, y1, x2, y2] representing the cluster.
    """
    x1a = []
    x2a = []
    y1a = []
    y2a = []
    angles = []

    # Collect endpoints and angles for all lines in the cluster
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x1a.append(x1)
        x2a.append(x2)
        y1a.append(y1)
        y2a.append(y2)
        linepar = np.polyfit((x1, x2), (y1, y2), 1)
        angdeg = (180/np.pi) * np.arctan(linepar[0])
        angles.append(angdeg)

    ang = np.mean(angles)  # Average angle of the cluster
    x1 = min(x1a)
    x2 = max(x2a)

    # For positive angles, use min/max y accordingly to preserve orientation
    if ang > 0:
        y1 = min(y1a)
        y2 = max(y2a)
    else:
        y1 = max(y1a)
        y2 = min(y2a)

    line_new = np.array([x1, y1, x2, y2])
    return line_new

def splitLines(lines):
    """
    Splits lines into left and right groups based on their angle.

    Args:
        lines: List of lines, each as [x1, y1, x2, y2].

    Returns:
        llines: Lines with negative angle (left lane lines).
        rlines: Lines with positive angle (right lane lines).
    """
    llines = []
    rlines = []
    for line in lines:
        x1, y1, x2, y2 = line
        linepar = np.polyfit((x1, x2), (y1, y2), 1)
        angle = (180/np.pi) * np.arctan(linepar[0])
        if angle > 5:
            rlines.append(line)
        if angle < -5:
            llines.append(line)
    return llines, rlines

def getLines(img, scale, height, width):
    """
    Detects line segments in the input image using color, region of interest, and edge filtering.

    Args:
        img: Input BGR image (numpy array).
        scale: Scaling factor for region of interest and Hough transform parameters.
        height: Image height (for filtering).
        width: Image width (for filtering).

    Returns:
        lines: Detected lines as returned by cv2.HoughLinesP, or None if no lines found.
    """
    sigma = 5  # Gaussian blur kernel size
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (sigma, sigma), 0)
    edges = cv2.Canny(blur, 50, 150)  # Edge detection

    # Convert to HSV for color masking
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurhsv = cv2.GaussianBlur(imghsv, (sigma, sigma), 0)
    maskColor = getColorMask(imghsv)  # Mask for lane colors (e.g., white/yellow)
    blurMaskColor = getColorMask(blurhsv)  # Mask on blurred HSV

    maskRoi = getRoiMask(img, scale)  # Mask for region of interest (e.g., road area)
    img_masked = cv2.bitwise_and(edges, maskRoi)  # Apply ROI mask to edges
    img_masked = cv2.bitwise_and(img_masked, maskColor)  # Further mask by color

    img_masked = filterWhite(img_masked, height, width)  # Remove non-white pixels (if lanes are white)
    img_masked = filterContours(img_masked, scale)  # Remove small or irrelevant contours

    # Dilate to connect broken line segments
    dilfactor = 2
    dilationkernel = np.ones((dilfactor, dilfactor), np.uint8)
    img_masked = cv2.dilate(img_masked, dilationkernel, iterations=1)

    # Probabilistic Hough Transform to detect line segments
    lines = cv2.HoughLinesP(
        img_masked,
        cv2.HOUGH_PROBABILISTIC,
        np.pi/180,
        70,
        maxLineGap=int(scale * 10),
        minLineLength=int(scale * 20)
    )
    return lines