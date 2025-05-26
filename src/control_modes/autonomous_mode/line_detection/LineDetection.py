import cv2
import numpy as np

from src.control_modes.autonomous_mode.line_detection.LineProcessor import clusterLines, combineLines, getLines



enable_debug = False

class LineFollowingNavigation:
    def __init__(self, width=848, height=480, scale=1):
        self.width = width
        self.height = height
        self.scale = scale

    def newLines(self, lines):
        processed_lines = []
        if lines is not None:
            clusters = clusterLines(lines, int(self.scale * 10), 15)
            for cluster in clusters:
                combined_line = combineLines(cluster)
                processed_lines.append(combined_line)
            return processed_lines
        return 0

    def splitLines(self, lines):
        """Split lines into left and right with simplified logic."""
        if not lines:
            return [], []
            
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            # Simply divide based on position relative to center
            mid_x = (x1 + x2) / 2
            
            if mid_x < self.width / 2:
                left_lines.append(line)
            else:
                right_lines.append(line)
                    
        return left_lines, right_lines

    def longestLine(self, lines):
        max_length = 0
        longest_line = None
        for line in lines:
            x1, y1, x2, y2 = line
            length = np.sqrt((abs(x2 - x1)) ** 2 + (abs(y2 - y1)) ** 2)
            if length > max_length:
                max_length = length
                longest_line = line
        return longest_line

    def findTarget(self, left_lines, right_lines, horizon_height, img, left_weight=1, right_weight=1, weight_factor=1, bias=0, draw=1):
        visuals = []

        if not left_lines and not right_lines:
            return False, visuals

        elif not right_lines:
            longest_left_line = self.longestLine(left_lines)
            x1l, y1l, x2l, y2l = longest_left_line

            line_params_left = np.polyfit((x1l, x2l), (y1l, y2l), 1)
            horizon_x_left = round((horizon_height - line_params_left[1]) / line_params_left[0])

            visuals.append({
                'type': 'line',
                'start': (x1l, y1l),
                'end': (x2l, y2l),
                'color': (50, 200, 200),
                'thickness': 3
            })
            visuals.append({
                'type': 'circle',
                'center': (horizon_x_left, horizon_height),
                'radius': 3,
                'color': (50, 200, 200),
                'thickness': -1
            })

            target = horizon_x_left

        elif not left_lines:
            longest_right_line = self.longestLine(right_lines)
            x1r, y1r, x2r, y2r = longest_right_line

            line_params_right = np.polyfit((x1r, x2r), (y1r, y2r), 1)
            horizon_x_right = round((horizon_height - line_params_right[1]) / line_params_right[0])

            visuals.append({
                'type': 'line',
                'start': (x1r, y1r),
                'end': (x2r, y2r),
                'color': (100, 200, 200),
                'thickness': 3
            })
            visuals.append({
                'type': 'circle',
                'center': (horizon_x_right, horizon_height),
                'radius': 3,
                'color': (100, 200, 200),
                'thickness': -1
            })

            target = horizon_x_right

        else:
            longest_left_line = self.longestLine(left_lines)
            longest_right_line = self.longestLine(right_lines)

            x1r, y1r, x2r, y2r = longest_right_line
            x1l, y1l, x2l, y2l = longest_left_line

            line_params_right = np.polyfit((x1r, x2r), (y1r, y2r), 1)
            horizon_x_right = round((horizon_height - line_params_right[1]) / line_params_right[0])

            line_params_left = np.polyfit((x1l, x2l), (y1l, y2l), 1)
            horizon_x_left = round((horizon_height - line_params_left[1]) / line_params_left[0])

            left_line_height = line_params_left[1]
            right_line_height = line_params_right[0] * self.width + line_params_right[1]

            intersection_x = (line_params_right[1] - line_params_left[1]) / (line_params_left[0] - line_params_right[0])
            intersection_y = intersection_x * line_params_left[0] + line_params_left[1]

            left_weight = max(left_weight, 0.01)
            right_weight = max(right_weight, 0.01)

            target = ((horizon_x_left + horizon_x_right) / 2) + (left_line_height - right_line_height) * weight_factor + bias

            visuals.extend([
                {
                    'type': 'line',
                    'start': (x1r, y1r),
                    'end': (x2r, y2r),
                    'color': (100, 200, 200),
                    'thickness': 3
                },
                {
                    'type': 'line',
                    'start': (x1l, y1l),
                    'end': (x2l, y2l),
                    'color': (50, 200, 200),
                    'thickness': 3
                },
                {
                    'type': 'circle',
                    'center': (round(intersection_x), round(intersection_y)),
                    'radius': 3,
                    'color': (210, 200, 200),
                    'thickness': -1
                },
                {
                    'type': 'circle',
                    'center': (horizon_x_right, horizon_height),
                    'radius': 3,
                    'color': (100, 200, 200),
                    'thickness': -1
                },
                {
                    'type': 'circle',
                    'center': (horizon_x_left, horizon_height),
                    'radius': 3,
                    'color': (50, 200, 200),
                    'thickness': -1
                },
                {
                    'type': 'circle',
                    'center': (int(target), horizon_height),
                    'radius': 3,
                    'color': (180, 200, 200),
                    'thickness': -1
                }
            ])

        # Add center reference point
        visuals.append({
            'type': 'circle',
            'center': (int(self.width / 2), horizon_height),
            'radius': 3,
            'color': (0, 0, 255),
            'thickness': -1
        })

        return target, visuals

    def processFrame(self, img, horizon_height=280, weight_factor=1, bias=0):
        """Process a frame to find the line following target with improved wall filtering."""
        if img.shape[1] != self.width or img.shape[0] != self.height:
            img = cv2.resize(img, (self.width, self.height))

        # Handle glare in the image
        img = self.detect_and_handle_glare(img)
        
        visuals = []
        
        # Quick preprocessing - reduce resolution for faster processing
        small_img = cv2.resize(img, (self.width // 2, self.height // 2))
        
        # Apply a tighter mask to focus on the road region only
        road_mask = np.zeros_like(small_img[:,:,0])
        center_width = int(small_img.shape[1] * 0.6)  # Focus on central 60%
        x_start = (small_img.shape[1] - center_width) // 2
        x_end = x_start + center_width
        road_mask[:, x_start:x_end] = 255
        
        # Display road mask
        if enable_debug: cv2.imshow("1. Road Mask", road_mask)
        
        # Apply mask to small image
        small_masked = cv2.bitwise_and(small_img, small_img, mask=road_mask)
        
        # Display masked image
        if enable_debug: cv2.imshow("2. Masked Image", small_masked)
        
        # Fast grayscale conversion
        gray = cv2.cvtColor(small_masked, cv2.COLOR_BGR2GRAY)
        
        # Display grayscale image
        if enable_debug: cv2.imshow("3. Grayscale", gray)
        
        # Apply direct threshold to isolate white lines - with higher threshold to exclude walls
        _, binary = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)  # Increased from 180 to 190
        
        # Display thresholded image
        if enable_debug: cv2.imshow("4. Thresholded", binary)
        
        # Focus only on bottom part of image
        roi_height = int(small_img.shape[0] * 0.5)  # Increased from 0.4 to 0.5
        binary_roi = binary.copy()
        binary_roi[:small_img.shape[0] - roi_height, :] = 0
        
        # Display ROI image
        if enable_debug: cv2.imshow("5. ROI", binary_roi)
        
        # Quick dilation to connect broken lines
        kernel = np.ones((3, 3), np.uint8)
        binary_dilated = cv2.dilate(binary_roi, kernel, iterations=1)
        
        # Display dilated image
        if enable_debug: cv2.imshow("6. Dilated", binary_dilated)
        
        # Direct Hough line detection on binary image
        lines = cv2.HoughLinesP(
            binary_dilated,
            rho=1,
            theta=np.pi/180,
            threshold=30,
            minLineLength=25,  # Increased from 20 to 25
            maxLineGap=10
        )
        
        # Create visualization of detected lines
        line_vis = cv2.cvtColor(binary_dilated, cv2.COLOR_GRAY2BGR)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Display lines on binary image
        if enable_debug: cv2.imshow("7. Detected Lines", line_vis)

        # Scale lines back to original size
        scaled_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                scaled_lines.append([x1*2, y1*2, x2*2, y2*2])

        if scaled_lines:
            # LESS AGGRESSIVE filtering of lines to exclude walls
            filtered_lines = []
            
            # Create debug visualization for angles
            debug_vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            for line in scaled_lines:
                x1, y1, x2, y2 = line
                
                # Skip very short lines - less restrictive
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if length < 30:  # Reduced from 40 to 30
                    continue
                    
                # Only consider lines with appropriate angles
                dx = x2 - x1
                dy = y2 - y1
                if abs(dx) < 1:  # Avoid division by zero
                    continue
                    
                # Calculate angle
                angle = abs(np.arctan2(dy, dx) * 180 / np.pi)
                
                # Color code by angle in debug vis
                if angle < 60:
                    color = (0, 0, 255)  # Red for angles < 60
                elif angle > 120:
                    color = (255, 0, 0)  # Blue for angles > 120
                else:
                    color = (0, 255, 0)  # Green for angles 60-120
                    
                cv2.line(debug_vis, (x1, y1), (x2, y2), color, 2)
                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                cv2.putText(debug_vis, f"{angle:.0f}", (mid_x, mid_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Wider angle range to include more potential lane lines
                if 60 < angle < 120:  # Wider range (was 70-110)
                    # Less restrictive edge filtering
                    mid_x = (x1 + x2) / 2
                    distance_from_edge = min(mid_x, self.width - mid_x)
                    
                    # Allow lines closer to edges
                    if distance_from_edge > (self.width * 0.05):  # Reduced from 10% to 5%
                        filtered_lines.append(line)
    
        # Display debug visualization for angles
        if enable_debug: cv2.imshow("8. Line Angles", debug_vis)
        
        # Create visualization of filtered lines
        filtered_vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for line in filtered_lines:
            x1, y1, x2, y2 = line
            cv2.line(filtered_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display filtered lines
        if enable_debug: cv2.imshow("9. Filtered Lines", filtered_vis)
        
        # Split lines into left and right
        left_lines = []
        right_lines = []
        
        for line in filtered_lines:
            x1, y1, x2, y2 = line
            mid_x = (x1 + x2) / 2
            
            if mid_x < self.width / 2:
                left_lines.append(line)
            else:
                right_lines.append(line)
        
        # Find target
        if left_lines or right_lines:
            target, target_visuals = self.findTarget(
                left_lines, right_lines, horizon_height, img,
                weight_factor=weight_factor, bias=bias
            )
            visuals.extend(target_visuals)
            return target, visuals

        return None, visuals

    def calculateSteeringAngle(self, target, mid_point=None):
        """Calculate steering angle with reduced sensitivity for smoother control."""
        if target is None:
            return 0.0

        if mid_point is None:
            mid_point = self.width / 2

        # Calculate offset from center
        offset = target - mid_point
        
        # Add a dead zone to reduce sensitivity to small movements
        dead_zone = self.width * 0.05  # 5% dead zone
        if abs(offset) < dead_zone:
            return 0.0
        elif offset > 0:
            offset -= dead_zone
        else:
            offset += dead_zone
            
        # Use non-linear response for smoother steering
        normalized_offset = offset / (self.width / 2)
        steering_factor = (normalized_offset**3 * 0.7) + (normalized_offset * 0.3)
        
        # Convert to steering angle
        max_angle = 30.0
        steering_angle = steering_factor * max_angle
        
        # Limit to max angle
        steering_angle = max(min(steering_angle, max_angle), -max_angle)
        
        return steering_angle

    def calculateSpeed(self, steering_angle, base_speed=100):
        """Calculate speed

        Parameters:
        steering_angle: Current steering angle in degrees
        base_speed: Base speed value

        Returns:
        float: Adjusted speed value
        """
        # Reduce speed in turns
        angle_factor = 1.0 - (abs(steering_angle) / 30.0) * 0.5
        speed = base_speed * angle_factor

        return max(speed, base_speed * 0.5)

    def setResolution(self, width, height):
        self.width = width
        self.height = height

    def run(self, img, base_speed=100, draw=1):
        """Run the line following algorithm on a frame.



        Parameters:
        img: Input image frame from camera/video
        base_speed: Base speed value
        draw: Flag to enable/disable visualization

        Returns:
        tuple: (steering_angle, speed, visualization image if draw=1)
        """
        # Process the frame to find target

        #We only want to use the bottom-half of the image.
      #  img = img[int(self.height / 2):, :]


        target, visuals = self.processFrame(img)
        if visuals is None:
            visuals = []

        # Calculate steering angle
        steering_angle = self.calculateSteeringAngle(target)

        # Calculate speed
        speed = self.calculateSpeed(steering_angle, base_speed)

        # Add steering and speed info to visualization
        if target is not None:
            text = f"Steering: {steering_angle:.1f} | Speed: {speed:.1f}"

            visuals.append({
                'type': 'text',
                'text': text,
                'position': (10, 30),
                'font': 'FONT_HERSHEY_SIMPLEX',
                'font_scale': 0.7,
                'color': (0, 0, 255),
                'thickness': 2
            })

            center_x = int(self.width / 2)
            center_y = self.height - 50
            endpoint_x = center_x + int(steering_angle * 2)

            visuals.append({
                'type': 'line',
                'start': (center_x, center_y),
                'end': (endpoint_x, center_y - 30),
                'color': (0, 255, 0),
                'thickness': 3
            })

        return steering_angle, speed, visuals

    def process(self, frame, base_speed=100, draw=1):
        """Process a single frame and return the results."""
        steering_angle, speed, visuals = self.run(frame, base_speed, draw)
        return steering_angle, speed, visuals

    def filterLinesByDistance(self, lines):
        """A faster implementation to filter lines by distance."""
        if not lines:
            return []
        
        # Filter by position, prioritizing lines closer to expected lane positions
        lane_width = self.width * 0.25  # Approximate lane width
        left_lane_x = self.width / 2 - lane_width / 2
        right_lane_x = self.width / 2 + lane_width / 2
        
        scored_lines = []
        for line in lines:
            x1, y1, x2, y2 = line
            mid_x = (x1 + x2) / 2
            
            # Calculate score based on distance to expected lane position
            left_distance = abs(mid_x - left_lane_x)
            right_distance = abs(mid_x - right_lane_x)
            min_distance = min(left_distance, right_distance)
            
            # Also consider line length - longer is better
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Combined score (lower is better)
            score = min_distance / length
            
            scored_lines.append((line, score))
        
        # Sort by score
        scored_lines.sort(key=lambda x: x[1])
        
        # Return the best lines, up to 4
        return [line for line, _ in scored_lines[:4]]

    def distanceToPosition(self, line, position):
        """Calculate minimum distance from a line to a position."""
        x1, y1, x2, y2 = line
        
        # Calculate distances from both endpoints to position
        dist1 = np.sqrt((x1 - position[0])**2 + (y1 - position[1])**2)
        dist2 = np.sqrt((x2 - position[0])**2 + (y2 - position[1])**2)
        
        # Return the minimum distance
        return min(dist1, dist2)

    def getCustomLines(self, img, binary_mask):
        """A much faster line detection implementation."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        
        # Combine with mask
        masked = cv2.bitwise_and(binary, binary_mask)
        
        # Fast Hough transform
        lines = cv2.HoughLinesP(
            masked,
            rho=1,
            theta=np.pi/180,
            threshold=20,
            minLineLength=int(self.scale * 25),
            maxLineGap=int(self.scale * 15)
        )
        
        if lines is None:
            return None
        
        # Simple filtering
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Skip horizontal lines
            if abs(y2 - y1) < 10:
                continue
                
            filtered_lines.append(line)
            
        return filtered_lines if filtered_lines else None

    def detect_and_handle_glare(self, img):
        """Detect and reduce glare in the image for better line detection."""
        # Convert to HSV for better glare detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Glare typically has high V (brightness) values
        _, _, v = cv2.split(hsv)
        
        # Threshold to find bright areas (potential glare)
        _, glare_mask = cv2.threshold(v, 220, 255, cv2.THRESH_BINARY)
        
        # Dilate to ensure we capture all glare regions
        kernel = np.ones((5, 5), np.uint8)
        glare_mask = cv2.dilate(glare_mask, kernel, iterations=1)
        
        # Calculate the percentage of the image affected by glare
        glare_percentage = (np.sum(glare_mask) / 255) / (img.shape[0] * img.shape[1]) * 100
        
        # If significant glare is detected, apply correction
        if glare_percentage > 5:
            # Create a visualization of detected glare
            glare_vis = img.copy()
            glare_vis[glare_mask > 0] = [0, 0, 255]  # Mark glare areas in red
            if enable_debug: cv2.imshow("Glare Detection", glare_vis)
            
            # Apply adaptive correction based on glare severity
            if glare_percentage > 15:
                # Create mask for non-glare regions (inverse of glare mask)
                non_glare_mask = cv2.bitwise_not(glare_mask)
                
                # Apply a light blur to reduce the effect of glare
                blurred = cv2.GaussianBlur(img, (15, 15), 0)
                
                # Create result by taking original image in non-glare regions
                # and blurred image in glare regions
                result = img.copy()
                result[glare_mask > 0] = blurred[glare_mask > 0]
                
                # Adjust contrast in glare regions to improve line detection
                glare_region = result[glare_mask > 0]
                if glare_region.size > 0:
                    # Apply contrast reduction only to glare regions
                    adjusted = cv2.addWeighted(
                        glare_region, 0.6, np.zeros_like(glare_region), 0, 30
                    )
                    result[glare_mask > 0] = adjusted
            
                return result
            elif glare_percentage > 10:
                # For moderate glare, just reduce contrast in glare regions
                result = img.copy()
                glare_region = result[glare_mask > 0]
                if glare_region.size > 0:
                    adjusted = cv2.addWeighted(
                        glare_region, 0.8, np.zeros_like(glare_region), 0, 20
                    )
                    result[glare_mask > 0] = adjusted
            
                return result
    
        # If no significant glare, return original image
        return img
