# src/control_modes/autonomous_mode/navigation/modes/Checkpoint.py

from src.control_modes.autonomous_mode.navigation.INavigator import INavigator
import math
import asyncio

class Checkpoint(INavigator):
    def __init__(self, can_controller):
        self.can_controller = can_controller
        self.checkpoints = [(10, 10), (20, 20), (30, 30)] # placeholders; need actual track coords.
        self.current_checkpoint = None
        self.reached_radius = 1.5
        self.running = True

    def compute_steering(self, current_pos, destination, current_heading_degrees):
        dx = destination[0] - current_pos[0]
        dy = destination[1] - current_pos[1]

        desired_angle_rad = math.atan2(dy, dx)
        desired_angle_deg = math.degrees(desired_angle_rad)
        angle_error = desired_angle_deg - current_heading_degrees
        angle_error = (angle_error + 180) % 360 - 180

        max_steering_angle = 30 # todo: change this to hunter or cart's actual radius.
        steering = max(-max_steering_angle, min(max_steering_angle, angle_error))

        return steering

    def inRadius(self, posA, posB):
        return math.dist(posA, posB) <= self.reached_radius

    def getOwnPosition(self):
        # TODO: Replace with GPS or localization system
        return 0, 0

    def navigateTo(self, current_pos, destination):
        current_heading = 0 # adjust based on gps/localization system

        steering = self.compute_steering(current_pos, destination, current_heading)
        throttle = 0.3  # Constant forward speed for now - ignore obstacles/traffic signs

        print(f"[NAV] Steering: {steering:.2f}°, Throttle: {throttle:.2f} → to {destination}")
        self.can_controller.set_steering(steering)
        self.can_controller.set_throttle(throttle)
        self.can_controller.set_break(0)

    async def start(self):
        if not self.checkpoints:
            print("No checkpoints available.")
            return

        self.current_checkpoint = self.checkpoints.pop(0)
        print(f"Starting navigation to checkpoint: {self.current_checkpoint}")

        while self.current_checkpoint and self.running:
            current_position = self.getOwnPosition()
            self.navigateTo(current_position, self.current_checkpoint)

            if self.inRadius(current_position, self.current_checkpoint):
                print(f"Reached checkpoint: {self.current_checkpoint}")
                if self.checkpoints:
                    self.current_checkpoint = self.checkpoints.pop(0)
                    print(f"Next checkpoint: {self.current_checkpoint}")
                else:
                    self.current_checkpoint = None
                    print("All checkpoints reached.")
                    self.can_controller.set_throttle(0)
                    self.can_controller.set_break(100)

            await asyncio.sleep(0.1)

    async def stop(self) -> None:
        print("Stopping navigation.")
        self.running = False
        self.can_controller.set_throttle(0)
        self.can_controller.set_break(100)
        self.can_controller.set_steering(0)
