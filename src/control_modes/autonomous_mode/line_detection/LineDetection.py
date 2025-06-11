import cv2
import numpy as np
from src.control_modes.autonomous_mode.line_detection.LineProcessor import clusterLines, combineLines, getLines

enable_debug = True


class LineFollowingNavigation:
    def __init__(self, width=848, height=480, scale=1, mode='normal'):
        self.width = width
        self.height = height
        self.scale = scale
        self.mode = mode  # 'normal', 'left_parallel', 'right_parallel'
        # Memory for line tracking
        self.prev_left_line = None
        self.prev_right_line = None
        self.crosswalk_detected = False
        self.crosswalk_confidence = 0
        self.left_min_bound = 0
        self.left_max_bound = 250
        self.right_min_bound = self.width - 250
        self.right_max_bound = self.width
        self.frame_count = 0



    def set_mode(self, mode):
        """Set the driving mode."""
        if mode in ['normal', 'left_parallel', 'right_parallel']:
            self.mode = mode
        else:
            raise ValueError("Mode must be 'normal', 'left_parallel', or 'right_parallel'")

    def detect_white_lines(self, img):
        """Simple and effective white line detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        _, binary = cv2.threshold(blurred, 170, 255, cv2.THRESH_BINARY)

        height, width = img.shape[:2]
        mask = np.zeros_like(binary)

        # Trapezoid ROI for road area - taller and wider at the top
        roi_vertices = np.array([
            [0, height],  # Bottom left
            [width // 4, height // 3],  # Top left (wider and higher)
            [3 * width // 4, height // 3],  # Top right (wider and higher)
            [width, height]  # Bottom right
        ], dtype=np.int32)

        cv2.fillPoly(mask, [roi_vertices], 255)
        masked = cv2.bitwise_and(binary, mask)

        # Apply morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.dilate(cleaned, kernel, iterations=1)

        if enable_debug:
            cv2.imshow("1. Gray", gray)
            cv2.imshow("2. Binary", binary)
            cv2.imshow("3. ROI Mask", mask)
            cv2.imshow("4. Masked", masked)
            cv2.imshow("5. Cleaned", cleaned)

        return cleaned

    def detect_lines(self, binary_img):
        """Detect lines using Hough transform."""
        lines = cv2.HoughLinesP(
            binary_img,
            rho=1,
            theta=np.pi / 180,
            threshold=20,  # Lower threshold to catch more lines
            minLineLength=40,  # Minimum line length
            maxLineGap=20  # Maximum gap between line segments
        )

        return lines

    def filter_lines(self, lines, img_shape):
        """Filter lines by angle and position, excluding crosswalk lines."""
        if lines is None:
            return [], []

        height, width = img_shape[:2]
        left_lines = []
        right_lines = []

        # First, detect if there's a crosswalk
        is_crosswalk, crosswalk_conf = self.detect_crosswalk(lines, img_shape)
        self.crosswalk_detected = is_crosswalk
        self.crosswalk_confidence = crosswalk_conf

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate line angle
            if x2 - x1 == 0:  # Vertical line
                angle = 90.0
            else:
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi


            # Enhanced filtering to exclude crosswalk lines
            # Skip vertical lines (crosswalk stripes) - more aggressive filtering
            if abs(angle) > 85:  # Skip lines that are too vertical (crosswalk stripes)
                continue

            # Filter by angle - lane lines should be diagonal but not too steep
            if abs(angle) < 5:  # Skip nearly horizontal lines
                continue

            # Calculate line length
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length < 30:  # Skip very short lines
                continue

            # Determine if line is on left or right side
            center_x = (x1 + x2) / 2

            if center_x < width / 2 and angle < 0:  # Left side, negative slope (line going up-left to down-right)
                left_lines.append([x1, y1, x2, y2])
            elif center_x > width / 2 and angle > 0:  # Right side, positive slope (line going up-right to down-left)
                right_lines.append([x1, y1, x2, y2])

        print(f"Returning {len(left_lines)} left lines and {len(right_lines)} right lines.")
        return left_lines, right_lines

    """def get_center_x(self, frame):
        binary = self.detect_white_lines(frame)
        lines = self.detect_lines(binary)
        _, _, center_x = self.filter_lines(lines, frame.shape)
        return center_x"""

    def get_best_line(self, lines):
        """Get the longest/best line from a group."""
        if not lines:
            return None

        best_line = None
        max_length = 0

        for line in lines:
            x1, y1, x2, y2 = line
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if length > max_length:
                max_length = length
                best_line = line

        return best_line

    def extrapolate_line(self, line, img_height, horizon_y=200):
        """Extrapolate line to full height."""
        if line is None:
            return None

        x1, y1, x2, y2 = line

        # Calculate line slope and intercept
        if x2 - x1 == 0:
            return line  # Vertical line, return as is

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # Calculate x coordinates at horizon and bottom
        x_bottom = int((img_height - intercept) / slope)
        x_horizon = int((horizon_y - intercept) / slope)

        return [x_horizon, horizon_y, x_bottom, img_height]

    def detect_clustered_lines(self, lines, img_shape):
        """Detect clusters of lines and create thick purple lines for likely paths."""
        if lines is None or len(lines) == 0:
            return []

        height, width = img_shape[:2]
        clustered_lines = []

        # Group lines by proximity and angle similarity (for dotted lines)
        line_groups = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate line properties
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            # Find if this line belongs to an existing group (for connecting dotted lines)
            found_group = False
            for group in line_groups:
                # Check if line is close to group and has similar angle
                group_center_x = np.mean([((l[0] + l[2]) / 2) for l in group])
                group_center_y = np.mean([((l[1] + l[3]) / 2) for l in group])
                group_angle = np.mean([np.arctan2(l[3] - l[1], l[2] - l[0]) * 180 / np.pi for l in group])

                # Tighter thresholds for connecting dotted lines
                distance_threshold = 30  # pixels (closer together)
                angle_threshold = 15  # degrees (more similar angles)

                distance = np.sqrt((center_x - group_center_x) ** 2 + (center_y - group_center_y) ** 2)
                angle_diff = abs(angle - group_angle)

                # Also check if lines are roughly aligned (for dotted line detection)
                if distance < distance_threshold and angle_diff < angle_threshold:
                    group.append([x1, y1, x2, y2])
                    found_group = True
                    break

            if not found_group:
                line_groups.append([[x1, y1, x2, y2]])

        # Create purple lines for groups with multiple lines (dotted lines or dense clusters)
        for group in line_groups:
            if len(group) >= 2:  # Changed from 3 to 2 for dotted line detection
                # Sort lines by position to connect them properly
                group.sort(key=lambda l: (l[1] + l[3]) / 2)  # Sort by average y position

                # For dotted lines, connect the endpoints
                if len(group) >= 2:
                    first_line = group[0]
                    last_line = group[-1]

                    # Find the endpoints that are furthest apart
                    points = [
                        (first_line[0], first_line[1]), (first_line[2], first_line[3]),
                        (last_line[0], last_line[1]), (last_line[2], last_line[3])
                    ]

                    # Find top-most and bottom-most points
                    top_point = min(points, key=lambda p: p[1])
                    bottom_point = max(points, key=lambda p: p[1])

                    # Create a line connecting the extremes
                    clustered_lines.append([int(top_point[0]), int(top_point[1]),
                                            int(bottom_point[0]), int(bottom_point[1])])

                # Alternative: if you want to fit through all points (for dense clusters)
                elif len(group) >= 4:  # For very dense clusters, use line fitting
                    all_points = []
                    for line in group:
                        all_points.extend([(line[0], line[1]), (line[2], line[3])])

                    if len(all_points) >= 2:
                        x_coords = [p[0] for p in all_points]
                        y_coords = [p[1] for p in all_points]

                        # Fit line: y = mx + b
                        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
                        m, b = np.linalg.lstsq(A, y_coords, rcond=None)[0]

                        # Calculate endpoints
                        y_min = min(y_coords)
                        y_max = max(y_coords)
                        x_min = (y_min - b) / m if m != 0 else min(x_coords)
                        x_max = (y_max - b) / m if m != 0 else max(x_coords)

                        # Clamp to image bounds
                        x_min = max(0, min(x_min, width))
                        x_max = max(0, min(x_max, width))

                        clustered_lines.append([int(x_min), int(y_min), int(x_max), int(y_max)])

        return clustered_lines

    def find_point_of_interception(self, lines, img_height=None):
        """
        Returns a list of (x, y) points where each line intercepts the bottom of the frame (y = img_height).
        Accepts a single line [x1, y1, x2, y2] or a list of such lines.
        """
        if lines is None or len(lines) == 0:
            return []

        # Normalize to list of lines
        if isinstance(lines[0], (int, float)):
            lines = [lines]

        if img_height is None:
            img_height = self.height

        interception_points = []
        for line in lines:
            x1, y1, x2, y2 = line
            if y2 == y1:
                continue  # Skip horizontal lines
            m = (y2 - y1) / (x2 - x1) if x2 != x1 else None
            if m is not None:
                b = y1 - m * x1
                x_at_bottom = int((img_height - b) / m)
                interception_points.append((x_at_bottom, img_height))
            else:
                interception_points.append((x1, img_height))
        return interception_points

    def get_lines_in_window(self, intercepting, lines, lower_bound, upper_bound):
        """
        Returns the lines whose interception point at the bottom of the frame
        is within [lower_bound, upper_bound] (inclusive).
        """
        result = []
        for point, line in zip(intercepting, lines):
            x, _ = point
            if lower_bound <= x <= upper_bound:
                result.append(line)
        return result


    def processFrame(self, img, horizon_height=280, weight_factor=1, bias=0):
        """Main processing function."""
        if img.shape[1] != self.width or img.shape[0] != self.height:
            img = cv2.resize(img, (self.width, self.height))

        self.frame_count += 1
        visuals = []

        windowed_frame = img.copy()

        # Step 1: Detect white lines
        binary = self.detect_white_lines(img)

        # Step 2: Detect lines using Hough transform
        raw_lines = self.detect_lines(binary)

        # Step 3: Detect clustered lines (purple lines)
        clustered_lines = self.detect_clustered_lines(raw_lines, img.shape)

        # Step 4: Filter and separate left/right lines
        left_lines, right_lines = self.filter_lines(raw_lines, img.shape)

        # Step 5: Get best lines

        intercepts_left = self.find_point_of_interception(left_lines)
        intercepts_right = self.find_point_of_interception(right_lines)

        left_lines = self.get_lines_in_window(intercepts_left, left_lines, 0, 350)
        right_lines = self.get_lines_in_window(intercepts_right, right_lines, self.width - 300, self.width)

        best_left = self.get_best_line(left_lines)
        best_right = self.get_best_line(right_lines)

        # draw over the new windowed image copy
        if enable_debug:
            line_img = windowed_frame
            if best_left is not None:
                x1, y1, x2, y2 = best_left
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 4)
            if best_right is not None:
                x1, y1, x2, y2 = best_right
                cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 4)
            if enable_debug:
                cv2.imshow("5. Best Lines", line_img)

        # Step 6: Extrapolate lines
        if best_left is not None:
            best_left = self.extrapolate_line(best_left, self.height, horizon_height)
        if best_right is not None:
            best_right = self.extrapolate_line(best_right, self.height, horizon_height)

        if enable_debug:
            # Show all detected lines
            line_img = img.copy()
            if raw_lines is not None:
                for line in raw_lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 255), 1)  # Yellow
                    # draw text saying each coord


            # Show clustered lines (purple)
            for line in clustered_lines:
                x1, y1, x2, y2 = line
                cv2.line(line_img, (x1, y1), (x2, y2), (128, 0, 128), 3)  # Purple, thick

            # Show filtered lines
            for line in left_lines:
                x1, y1, x2, y2 = line
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green
            for line in right_lines:
                x1, y1, x2, y2 = line
                cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Red

            # Show best lines
            if best_left is not None:
                x1, y1, x2, y2 = best_left
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 4)  # Thick green
            if best_right is not None:
                x1, y1, x2, y2 = best_right
                cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 4)  # Thick red

            cv2.imshow("6. Lines Detected", line_img)

        # Find target
        final_left = [best_left] if best_left is not None else []
        final_right = [best_right] if best_right is not None else []

        if final_left or final_right:
            target, target_visuals = self.findTarget(
                final_left, final_right, horizon_height, img,
                weight_factor=weight_factor, bias=bias
            )
            visuals.extend(target_visuals)
            return target, visuals

        return None, visuals

    def findTarget(self, left_lines, right_lines, horizon_height, img, left_weight=1, right_weight=1, weight_factor=1,
                   bias=0, draw=1):
        visuals = []
        # print("Finding target...")

        # If crosswalk is detected, prioritize going straight
        if self.crosswalk_detected:
            # print(f"Crosswalk detected - maintaining straight course")
            target = self.width / 2  # Go straight through crosswalk

            visuals.append({
                'type': 'text',
                'text': f"CROSSWALK - GO STRAIGHT ({self.crosswalk_confidence}%)",
                'position': (10, 60),
                'font': 'FONT_HERSHEY_SIMPLEX',
                'font_scale': 0.7,
                'color': (0, 0, 255),
                'thickness': 2
            })

            visuals.append({
                'type': 'circle',
                'center': (int(target), horizon_height),
                'radius': 8,
                'color': (0, 165, 255),  # Orange circle for crosswalk mode
                'thickness': -1
            })

            return target, visuals

        # Rest of the original findTarget method...
        if not left_lines and not right_lines:
            # print("No left || or right detected")
            return None, visuals

        elif not right_lines:
            # Only left line detected
            x1l, y1l, x2l, y2l = left_lines[0]
            target = x1l + 200  # Estimate right side based on typical lane width

            visuals.append({
                'type': 'line',
                'start': (x1l, y1l),
                'end': (x2l, y2l),
                'color': (0, 255, 0),
                'thickness': 3
            })

        elif not left_lines:
            # Only right line detected
            x1r, y1r, x2r, y2r = right_lines[0]
            target = x1r - 200  # Estimate left side based on typical lane width

            visuals.append({
                'type': 'line',
                'start': (x1r, y1r),
                'end': (x2r, y2r),
                'color': (255, 0, 0),
                'thickness': 3
            })

        else:
            # Both lines detected
            x1l, y1l, x2l, y2l = left_lines[0]
            x1r, y1r, x2r, y2r = right_lines[0]

            # print(f"Target: Left Line: ({x1l}, {y1l}) to ({x2l}, {y2l}), Right Line: ({x1r}, {y1r}) to ({x2r}, {y2r})")
            # Target is midpoint between the two lines at horizon
            target = (x1l + x1r) / 2

            visuals.extend([
                {
                    'type': 'line',
                    'start': (x1l, y1l),
                    'end': (x2l, y2l),
                    'color': (0, 255, 0),
                    'thickness': 3
                },
                {
                    'type': 'line',
                    'start': (x1r, y1r),
                    'end': (x2r, y2r),
                    'color': (255, 0, 0),
                    'thickness': 3
                }
            ])

        # Add target point
        if target is not None:
            visuals.append({
                'type': 'circle',
                'center': (int(target), horizon_height),
                'radius': 5,
                'color': (0, 255, 255),
                'thickness': -1
            })

        # Add center reference point
        visuals.append({
            'type': 'circle',
            'center': (int(self.width / 2), horizon_height),
            'radius': 5,
            'color': (0, 0, 255),
            'thickness': -1
        })

        return target, visuals

    def calculateSteeringAngle(self, target, left_lines=None, right_lines=None, mid_point=None):
        """Calculate steering angle based on detected lines, target, and driving mode."""
        if mid_point is None:
            mid_point = self.width / 2

        max_angle = 30.0

        # Mode-specific steering calculations
        if self.mode == 'left_parallel':
            # Force left lane paralleling - only use left line if available
            if left_lines and len(left_lines) > 0:
                x1, y1, x2, y2 = left_lines[0]
                line_angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                steering_angle = line_angle * 0.8

                # Maintain distance from left line
                line_center_x = (x1 + x2) / 2
                distance_from_line = line_center_x - mid_point
                distance_correction = distance_from_line * 0.08
                steering_angle += distance_correction
            else:
                # No left line found, try to estimate or go straight
                steering_angle = 0.0

        elif self.mode == 'right_parallel':
            # Force right lane paralleling - only use right line if available
            if right_lines and len(right_lines) > 0:
                x1, y1, x2, y2 = right_lines[0]
                line_angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                steering_angle = line_angle * 0.8

                # Maintain distance from right line
                line_center_x = (x1 + x2) / 2
                distance_from_line = line_center_x - mid_point
                distance_correction = distance_from_line * 0.08
                steering_angle += distance_correction
            else:
                # No right line found, try to estimate or go straight
                steering_angle = 0.0

        else:  # normal mode
            # Original logic for normal mode
            if left_lines and right_lines and len(left_lines) > 0 and len(right_lines) > 0:
                offset = target - mid_point if target is not None else 0.0
                steering_angle = (offset / (self.width / 2)) * max_angle

            elif left_lines and len(left_lines) > 0 and (not right_lines or len(right_lines) == 0):
                x1, y1, x2, y2 = left_lines[0]
                line_angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                steering_angle = line_angle * 0.8

                line_center_x = (x1 + x2) / 2
                distance_from_line = line_center_x - mid_point
                distance_correction = distance_from_line * 0.08
                steering_angle += distance_correction

            elif right_lines and len(right_lines) > 0 and (not left_lines or len(left_lines) == 0):
                x1, y1, x2, y2 = right_lines[0]
                line_angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                steering_angle = line_angle * 0.8

                line_center_x = (x1 + x2) / 2
                distance_from_line = line_center_x - mid_point
                distance_correction = distance_from_line * 0.08
                steering_angle += distance_correction

            elif target is not None:
                offset = target - mid_point
                steering_angle = (offset / (self.width / 2)) * max_angle
            else:
                # print("No lines detected and no target provided, defaulting to straight.")
                return 0.0

        # Limit steering angle
        steering_angle = max(min(steering_angle, max_angle), -max_angle)
        # print(f"Steering angle calculated: {steering_angle:.2f} degrees")
        return steering_angle

    def calculateSpeed(self, steering_angle, base_speed=100):
        """Calculate speed based on steering angle."""
        angle_factor = 1.0 - (abs(steering_angle) / 30.0) * 0.3
        speed = base_speed * angle_factor
        return max(speed, base_speed * 0.6)

    def run(self, img, base_speed=100, draw=1):
        """Run the line following algorithm on a frame."""
        target, visuals = self.processFrame(img)
        if visuals is None:
            visuals = []

        # Get the detected lines for steering calculation
        binary = self.detect_white_lines(img)
        raw_lines = self.detect_lines(binary)

        left_lines, right_lines = self.filter_lines(raw_lines, img.shape)

        # Get best lines
        best_left = self.get_best_line(left_lines)
        best_right = self.get_best_line(right_lines)

        final_left = [best_left] if best_left is not None else []
        final_right = [best_right] if best_right is not None else []

        # Add line visuals
        if draw:
            # Add all detected raw lines (thin yellow)
            if raw_lines is not None:
                for line in raw_lines:
                    x1, y1, x2, y2 = line[0]
                    visuals.append({
                        'type': 'line',
                        'start': (x1, y1),
                        'end': (x2, y2),
                        'color': (0, 255, 255),  # Yellow
                        'thickness': 1
                    })

            # Add filtered left lines (green)
            for line in left_lines:
                x1, y1, x2, y2 = line
                visuals.append({
                    'type': 'line',
                    'start': (x1, y1),
                    'end': (x2, y2),
                    'color': (0, 255, 0),  # Green
                    'thickness': 2
                })

            # Add filtered right lines (red)
            for line in right_lines:
                x1, y1, x2, y2 = line
                visuals.append({
                    'type': 'line',
                    'start': (x1, y1),
                    'end': (x2, y2),
                    'color': (255, 0, 0),  # Red
                    'thickness': 2
                })

            # Add best lines (thicker)
            if best_left is not None:
                x1, y1, x2, y2 = best_left
                visuals.append({
                    'type': 'line',
                    'start': (x1, y1),
                    'end': (x2, y2),
                    'color': (0, 255, 0),  # Bright green
                    'thickness': 4
                })

            if best_right is not None:
                x1, y1, x2, y2 = best_right
                visuals.append({
                    'type': 'line',
                    'start': (x1, y1),
                    'end': (x2, y2),
                    'color': (255, 0, 0),  # Bright red
                    'thickness': 4
                })

        steering_angle = self.calculateSteeringAngle(target, final_left, final_right)
        speed = self.calculateSpeed(steering_angle, base_speed)


        # Prepare visuals for display
        print("Final steering angle:", steering_angle)

        _mode = self.mode if self.mode else 'normal'
        _target = target if target is not None else 0.0
        _steering_angle = steering_angle if steering_angle is not None else 0.0
        _speed = speed if speed is not None else 0.0

        rad_steer = _steering_angle * np.pi / 180
        print("Steering angle in radians:", rad_steer)
        rad_steer = max(min(rad_steer, 1.25), -1.25)
        text = (f"Mode: {_mode}, target: {_target:.2f}, "
                f"steering angle: {rad_steer}, speed: {_speed:.2f}")
        if target is not None:
            visuals.append({
                'type': 'text',
                'text': text,
                'position': (10, 30),
                'font': 'FONT_HERSHEY_SIMPLEX',
                'font_scale': 0.7,
                'color': (255, 255, 255),
                'thickness': 2
            })
        else:
            visuals.append({
                'type': 'text',
                'text': text,
                'position': (10, 30),
                'font': 'FONT_HERSHEY_SIMPLEX',
                'font_scale': 0.7,
                'color': (255, 255, 255),
                'thickness': 2
            })

        return steering_angle, speed, visuals

    def detect_crosswalk(self, lines, img_shape):
        """Detect crosswalk patterns and return confidence level."""
        if lines is None:
            return False, 0

        height, width = img_shape[:2]
        vertical_lines = []
        crosswalk_confidence = 0

        # Look for vertical or near-vertical lines
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate angle from horizontal
            if x2 - x1 == 0:
                angle = 90  # Perfectly vertical
            else:
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # Check if line is vertical (80-90 degrees from horizontal)
            if angle > 80:
                line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                # Only consider lines that are reasonably long (potential crosswalk stripes)
                if line_length > 30:
                    vertical_lines.append({
                        'line': [x1, y1, x2, y2],
                        'length': line_length,
                        'center_x': (x1 + x2) / 2,
                        'center_y': (y1 + y2) / 2
                    })

        # Analyze vertical lines for crosswalk pattern
        if len(vertical_lines) >= 2:  # Need at least 2 vertical lines
            # Sort by x position
            vertical_lines.sort(key=lambda x: x['center_x'])

            # Check for evenly spaced vertical lines (crosswalk pattern)
            spacings = []
            for i in range(1, len(vertical_lines)):
                spacing = vertical_lines[i]['center_x'] - vertical_lines[i - 1]['center_x']
                spacings.append(spacing)

            if len(spacings) >= 1:
                # Check if spacings are relatively consistent (crosswalk stripes)
                mean_spacing = np.mean(spacings)
                spacing_variance = np.var(spacings) if len(spacings) > 1 else 0

                # Crosswalk stripes are typically 30-150 pixels apart depending on distance
                if 30 < mean_spacing < 150 and spacing_variance < 400:
                    crosswalk_confidence = min(len(vertical_lines) * 20, 100)  # Max 100%

                    # Additional checks for higher confidence
                    if len(vertical_lines) >= 3:  # 3 or more stripes
                        crosswalk_confidence += 20

                    # Check if lines span significant portion of width
                    total_width_coverage = max([l['center_x'] for l in vertical_lines]) - min(
                        [l['center_x'] for l in vertical_lines])
                    if total_width_coverage > width * 0.3:  # Covers at least 30% of width
                        crosswalk_confidence += 15

                    crosswalk_confidence = min(crosswalk_confidence, 100)

        is_crosswalk = crosswalk_confidence > 40  # Threshold for crosswalk detection


        return is_crosswalk, crosswalk_confidence


    def process(self, frame, base_speed=100, draw=1):
        """Process a single frame and return the results."""
        # print(f"Width: {self.width}, Height: {self.height}, Scale: {self.scale}, Mode: {self.mode}")
        steering_angle, speed, visuals = self.run(frame, base_speed, draw)
        return steering_angle, speed, visuals
