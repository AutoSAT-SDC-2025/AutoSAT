import cv2
import numpy as np

from src.control_modes.autonomous_mode.line_detection.LineProcessor import clusterLines, combineLines, getLines


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
        left_lines = []
        right_lines = []
        for line in lines:
            x1, y1, x2, y2 = line
            line_params = np.polyfit((x1, x2), (y1, y2), 1)
            angle = (180 / np.pi) * np.arctan(line_params[0])
            if angle > 5:
                right_lines.append(line)
            if angle < -5:
                left_lines.append(line)
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
        """Process a frame to find the line following target.

        Parameters:
        img (ndarray): Input image frame
        horizon_height (int): Height of the horizon line for target calculation
        weight_factor (float): Weight factor for line calculation
        bias (float): Bias adjustment for target position

        Returns:
        tuple: (target position, visualization image if draw=1)
        """
        # Resize image if needed
        if img.shape[1] != self.width or img.shape[0] != self.height:
            img = cv2.resize(img, (self.width, self.height))

        # Detect lines in the image
        lines = getLines(img, self.scale, self.height, self.width)

        # Process detected lines
        if lines is not None:
            # Get the new processed lines
            processed_lines = self.newLines(lines)

            # Split into left and right lines
            if processed_lines:
                left_lines, right_lines = self.splitLines(processed_lines)

                # Find target based on detected lines
                return self.findTarget(left_lines, right_lines, horizon_height, img, weight_factor=weight_factor, bias=bias)

        # No valid lines detected
        return None, img

    def calculateSteeringAngle(self, target, mid_point=None):
        """Calculate steering angle based on target position.

        Parameters:
        target: Target x-position
        mid_point: Middle point of the frame (if None, use self.width/2)

        Returns:
        float: Steering angle in degrees (-30 to 30)
        """
        if target is None:
            return 0.0

        if mid_point is None:
            mid_point = self.width / 2

        # Calculate offset from center
        offset = target - mid_point

        # Convert to steering angle (scale to range -30 to 30 degrees)
        max_angle = 30.0
        steering_angle = (offset / (self.width / 2)) * max_angle

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
