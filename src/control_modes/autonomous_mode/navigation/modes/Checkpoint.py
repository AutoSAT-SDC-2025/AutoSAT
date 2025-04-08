# src/control_modes/autonomous_mode/navigation/modes/Checkpoint.py
import asyncio

from src.control_modes.autonomous_mode.navigation.modes.INavigator import INavigator


class Checkpoint(INavigator):
    def __init__(self):
        self.checkpoints = [(10, 10), (20, 20), (30, 30)]  # Fake checkpoints
        self.current_checkpoint = None

    def inRadius(self, posA, ):
        pass

    def getOwnPosition(self):
        # Implement GPS (later)
        return 0, 0

    async def start(self):
        if self.checkpoints:
            self.current_checkpoint = self.checkpoints.pop(0)
            print(f"Starting navigation to checkpoint: {self.current_checkpoint}")
            # Add logic to navigate to the current checkpoint
            while self.current_checkpoint:
                current_position = self.getOwnPosition()
                if current_position == self.current_checkpoint:
                    print(f"Reached checkpoint: {self.current_checkpoint}")
                    if self.checkpoints:
                        self.current_checkpoint = self.checkpoints.pop(0)
                        print(f"Next checkpoint: {self.current_checkpoint}")
                    else:
                        self.current_checkpoint = None
                        print("All checkpoints reached.")
        else:
            print("No checkpoints available.")

    async def stop(self) -> None:
        self.current_checkpoint = None
        print("Stopping navigation.")
