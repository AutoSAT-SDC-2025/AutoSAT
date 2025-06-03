# src/control_modes/autonomous_mode/navigation/modes/Checkpoint.py

from src.control_modes.autonomous_mode.navigation.INavigator import INavigator
import math
import asyncio
import numpy as np


class Checkpoint(INavigator):
    def __init__(self, can_controller, localization_system=None):
        self.can_controller = can_controller
        self.localization_system = localization_system
        self.checkpoints = [(10, 10), (20, 20), (30, 30)]  # placeholders; need actual track coords.
        self.current_checkpoint = None
        self.reached_radius = 1.5
        self.running = True

        # Performance tuning parameters
        self.max_steering_angle = 30.0
        self.base_speed = 0.4
        self.min_speed = 0.15
        self.max_speed = 0.6

        # PID controller for steering
        self.steering_kp = 1.2
        self.steering_ki = 0.05
        self.steering_kd = 0.3
        self.steering_error_sum = 0
        self.previous_steering_error = 0

        # Look-ahead parameters
        self.look_ahead_distance = 3.0
        self.min_look_ahead = 1.5
        self.max_look_ahead = 5.0

        # Smoothing
        self.steering_history = []
        self.max_steering_history = 3

    def normalize_angle(self, angle):
        """Normalize angle to [-180, 180] range."""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def compute_steering_pid(self, angle_error, dt=0.1):
        """Compute steering using PID controller."""
        # Proportional term
        p_term = self.steering_kp * angle_error

        # Integral term (with windup protection)
        self.steering_error_sum += angle_error * dt
        self.steering_error_sum = max(-50, min(50, self.steering_error_sum))  # Clamp integral
        i_term = self.steering_ki * self.steering_error_sum

        # Derivative term
        d_term = self.steering_kd * (angle_error - self.previous_steering_error) / dt
        self.previous_steering_error = angle_error

        # Combined PID output
        steering = p_term + i_term + d_term

        # Apply limits
        steering = max(-self.max_steering_angle, min(self.max_steering_angle, steering))

        return steering

    def smooth_steering(self, steering):
        """Apply smoothing to steering commands."""
        self.steering_history.append(steering)
        if len(self.steering_history) > self.max_steering_history:
            self.steering_history.pop(0)

        # Weighted average (more weight to recent values)
        weights = np.linspace(0.5, 1.0, len(self.steering_history))
        weights = weights / np.sum(weights)

        return np.average(self.steering_history, weights=weights)

    def compute_adaptive_speed(self, angle_error, distance_to_target):
        """Compute adaptive speed based on steering angle and distance."""
        # Reduce speed for sharp turns
        angle_factor = 1.0 - (abs(angle_error) / 90.0) * 0.5
        angle_factor = max(0.3, angle_factor)

        # Reduce speed when approaching target
        distance_factor = 1.0
        if distance_to_target < 5.0:
            distance_factor = max(0.5, distance_to_target / 5.0)

        # Combine factors
        speed = self.base_speed * angle_factor * distance_factor
        speed = max(self.min_speed, min(self.max_speed, speed))

        return speed

    def compute_look_ahead_point(self, current_pos, current_heading, destination):
        """Compute look-ahead point for smoother navigation."""
        distance_to_target = math.dist(current_pos, destination)

        # Adaptive look-ahead distance based on speed and distance
        look_ahead = min(self.max_look_ahead, max(self.min_look_ahead, distance_to_target * 0.3))

        # If we're close to the target, look directly at it
        if distance_to_target < look_ahead:
            return destination

        # Calculate direction vector to target
        dx = destination[0] - current_pos[0]
        dy = destination[1] - current_pos[1]

        # Normalize direction vector
        length = math.sqrt(dx * dx + dy * dy)
        if length == 0:
            return destination

        dx_norm = dx / length
        dy_norm = dy / length

        # Calculate look-ahead point
        look_ahead_x = current_pos[0] + dx_norm * look_ahead
        look_ahead_y = current_pos[1] + dy_norm * look_ahead

        return (look_ahead_x, look_ahead_y)

    def compute_steering(self, current_pos, destination, current_heading_degrees):
        """Compute steering angle with improved algorithm."""
        # Use look-ahead point for smoother navigation
        look_ahead_point = self.compute_look_ahead_point(current_pos, current_heading_degrees, destination)

        # Calculate desired direction
        dx = look_ahead_point[0] - current_pos[0]
        dy = look_ahead_point[1] - current_pos[1]

        # Handle case where we're at the target
        if abs(dx) < 0.01 and abs(dy) < 0.01:
            return 0.0

        # Calculate desired heading
        desired_angle_rad = math.atan2(dy, dx)
        desired_angle_deg = math.degrees(desired_angle_rad)

        # Calculate angle error
        angle_error = desired_angle_deg - current_heading_degrees
        angle_error = self.normalize_angle(angle_error)

        # Use PID controller for steering
        steering = self.compute_steering_pid(angle_error)

        # Apply smoothing
        steering = self.smooth_steering(steering)

        return steering

    def inRadius(self, posA, posB):
        """Check if position A is within radius of position B."""
        return math.dist(posA, posB) <= self.reached_radius

    def getOwnPosition(self):
        """Get current position from localization system."""
        if self.localization_system:
            try:
                x, y, rotation = self.localization_system.get_position()
                return (x, y), rotation
            except Exception as e:
                print(f"[NAV] Localization error: {e}")
                return (0, 0), 0
        else:
            # Fallback to default values
            return (0, 0), 0

    def navigateTo(self, current_pos, current_heading, destination):
        """Navigate to destination with improved control."""
        # Calculate distance to target
        distance_to_target = math.dist(current_pos, destination)

        # Compute steering
        steering = self.compute_steering(current_pos, destination, current_heading)

        # Compute adaptive throttle
        angle_error = self.normalize_angle(
            math.degrees(math.atan2(destination[1] - current_pos[1],
                                    destination[0] - current_pos[0])) - current_heading
        )
        throttle = self.compute_adaptive_speed(angle_error, distance_to_target)

        # Apply controls
        print(f"[NAV] Pos: ({current_pos[0]:.2f}, {current_pos[1]:.2f}), "
              f"Heading: {current_heading:.1f}°, "
              f"Steering: {steering:.2f}°, Throttle: {throttle:.2f} "
              f"→ {destination} (dist: {distance_to_target:.2f}m)")

        self.can_controller.set_steering(steering)
        self.can_controller.set_throttle(throttle)
        self.can_controller.set_break(0)

    def add_checkpoint(self, x, y):
        """Add a new checkpoint to the list."""
        self.checkpoints.append((x, y))
        print(f"[NAV] Added checkpoint: ({x}, {y})")

    def clear_checkpoints(self):
        """Clear all checkpoints."""
        self.checkpoints.clear()
        self.current_checkpoint = None
        print("[NAV] All checkpoints cleared.")

    def get_progress(self):
        """Get navigation progress information."""
        total_checkpoints = len(self.checkpoints) + (1 if self.current_checkpoint else 0)
        completed = total_checkpoints - len(self.checkpoints) - (1 if self.current_checkpoint else 0)

        return {
            'total_checkpoints': total_checkpoints,
            'completed_checkpoints': completed,
            'current_checkpoint': self.current_checkpoint,
            'remaining_checkpoints': len(self.checkpoints)
        }

    async def start(self):
        """Start checkpoint navigation with improved control."""
        if not self.checkpoints:
            print("[NAV] No checkpoints available.")
            return

        self.current_checkpoint = self.checkpoints.pop(0)
        print(f"[NAV] Starting navigation to checkpoint: {self.current_checkpoint}")

        while self.current_checkpoint and self.running:
            try:
                # Get current position and heading
                current_position, current_heading = self.getOwnPosition()

                # Navigate to current checkpoint
                self.navigateTo(current_position, current_heading, self.current_checkpoint)

                # Check if we've reached the checkpoint
                if self.inRadius(current_position, self.current_checkpoint):
                    print(f"[NAV] ✓ Reached checkpoint: {self.current_checkpoint}")

                    if self.checkpoints:
                        self.current_checkpoint = self.checkpoints.pop(0)
                        print(f"[NAV] → Next checkpoint: {self.current_checkpoint}")

                        # Reset PID controller for new target
                        self.steering_error_sum = 0
                        self.previous_steering_error = 0
                        self.steering_history.clear()
                    else:
                        self.current_checkpoint = None
                        print("[NAV] ✓ All checkpoints reached! Mission complete.")

                        # Stop the vehicle
                        self.can_controller.set_throttle(0)
                        self.can_controller.set_break(100)
                        self.can_controller.set_steering(0)
                        break

                await asyncio.sleep(0.1)  # 10 Hz control loop

            except Exception as e:
                print(f"[NAV] Navigation error: {e}")
                await asyncio.sleep(0.5)  # Longer delay on error

    async def stop(self) -> None:
        """Stop navigation gracefully."""
        print("[NAV] Stopping navigation...")
        self.running = False

        # Bring vehicle to a stop
        self.can_controller.set_throttle(0)
        self.can_controller.set_break(100)
        self.can_controller.set_steering(0)

        print("[NAV] Navigation stopped.")
