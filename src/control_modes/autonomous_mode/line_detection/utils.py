import cv2
import numpy as np
from PIL import Image, ImageDraw


def getColorMask(imghsv):
    """
    Generate a binary mask isolating bright (typically white) regions in an HSV image.
    This is useful for lane detection, as lane markings are often white or bright.

    Args:
        imghsv (np.ndarray): Input image in HSV color space.

    Returns:
        np.ndarray: Dilated binary mask where white/bright regions are 255, others are 0.
    """
    sigma = 3  # Kernel size for Gaussian blur to reduce noise.
    # Define HSV range for white/bright colors.
    lower_range = (0, 0, 160)
    upper_range = (255, 255, 255)
    # Blur the image to smooth out noise and small artifacts.
    blurhsv = cv2.GaussianBlur(imghsv, (sigma, sigma), 0)
    # Create a mask where pixels within the specified HSV range are white (255).
    maskhsv = cv2.inRange(blurhsv, lower_range, upper_range)

    # Further dilate the mask to connect fragmented regions and fill small gaps.
    dilfactor = 4
    dilationkernel = np.ones((dilfactor, dilfactor), np.uint8)
    maskhsvdil = cv2.dilate(maskhsv, dilationkernel, iterations=2)
    return maskhsvdil


def getRoiMask(img, scale):
    """
    Create a binary mask for the region of interest (ROI) in the image.
    The ROI is typically a trapezoid covering the road area where lane lines are expected.

    Args:
        img (np.ndarray): Input image (BGR or grayscale).
        scale (float): Scaling factor to adjust the ROI size for different image resolutions.

    Returns:
        np.ndarray: Binary mask with ROI filled as white (255), rest as black (0).
    """
    width = img.shape[1]
    height = img.shape[0]
    mid = width / 2  # Horizontal center of the image

    # Define the width of the mask at the bottom and top (trapezoid shape)
    maskwd = 0
    maskwu = mid
    # Height of the top edge of the ROI (from the bottom)
    maskh = 220 * scale

    # Define the four corners of the trapezoidal ROI polygon
    polygon = [
        (maskwd, height),  # Bottom-left
        (mid - maskwu, maskh),  # Top-left
        (mid + maskwu, maskh),  # Top-right
        (width - maskwd, height)  # Bottom-right
    ]
    # Create a blank mask and draw the ROI polygon filled with white (255)
    imgmask = Image.new('L', (width, height), 0)
    ImageDraw.Draw(imgmask).polygon(polygon, outline=1, fill=255)
    mask = np.array(imgmask)
    return mask


def filterContours(img, scale):
    """
    Remove small or irrelevant contours from a binary image, keeping only large, likely lane-related regions.

    Args:
        img (np.ndarray): Input binary image (e.g., edge or color mask).
        scale (float): Scaling factor for contour size thresholds.

    Returns:
        np.ndarray: Masked image with small contours removed.
    """
    # Dilate to connect close regions and fill small holes.
    dilfactor = 2
    dilationkernel = np.ones((dilfactor, dilfactor), np.uint8)
    img_dil = cv2.dilate(img, dilationkernel, iterations=1)

    # Find all external contours in the dilated image.
    contours = cv2.findContours(img_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    # Sort contours by area, largest first (not strictly necessary here).
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Create a mask initialized to all white (keep everything by default).
    mask = np.ones(img.shape[:2], dtype="uint8") * 255
    for cnt in contours:
        # Get bounding rectangle for each contour.
        x1, y1, w, h = cv2.boundingRect(cnt)
        # Enforce minimum width and height for the bounding box.
        w = max(w, int(20 * scale))
        h = max(h, int(20 * scale))
        rect = img_dil[y1:y1 + h, x1:x1 + w]

        # Threshold for small contours (remove if both width and height are below this).
        th = int(scale * 50)
        if w < th and h < th:
            # Draw the small contour as black (remove) on the mask.
            cv2.drawContours(mask, [cnt], -1, 0, -1)

    # Apply the mask to the input image, removing small contours.
    img_masked = cv2.bitwise_and(img, img, mask=mask)
    return img_masked


def filterWhite(img_masked, height, width):
    """
    Remove regions from the mask that are too bright (likely not lane lines) using a sliding window approach.
    This helps suppress large white areas (e.g., sunlight, reflections) that could confuse lane detection.

    Args:
        img_masked (np.ndarray): Input binary mask (lane candidates).
        height (int): Image height.
        width (int): Image width.

    Returns:
        np.ndarray: Masked image with overly bright regions suppressed.
    """
    window_size = 24  # Size of the sliding window (pixels)
    hstart = 240  # Start row for windowing (skip top of image)
    sidemargin = 4  # Margin to avoid windows at the extreme left/right edges
    mask = np.ones_like(img_masked)
    mask = mask * 255  # Start with all white (keep everything)

    # Slide window over the lower part of the image
    for row in range(hstart, height - window_size + 1, window_size):
        for col in range(sidemargin, width - window_size - sidemargin + 1, window_size):
            window = img_masked[row:row + window_size, col:col + window_size]
            density = np.mean(window)  # Average pixel value in the window
            if density > 45:
                # If the window is too bright, suppress it (set to black)
                mask[row:row + window_size, col:col + window_size] = 0

    # Apply the mask to the input image, removing overly bright regions
    img_masked = cv2.bitwise_and(img_masked, mask)
    return img_masked