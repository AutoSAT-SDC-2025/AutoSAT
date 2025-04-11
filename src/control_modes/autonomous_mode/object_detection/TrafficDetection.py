import re

class TrafficManager:
    def __init__(self):
        self.red_light_threshold = 4.5  # Distance threshold to consider traffic light as "red"
        self.speed_limit = 10

    def extract_speed_from_label(self, class_label: str) -> int | None:
        """Extracts the speed limit from a class label like 'Speed-limit-20km-h'."""
        match = re.search(r'(\d+)\s*km[- ]?h', class_label.lower())
        if match:
            return int(match.group(1))
        return None

    def process_traffic_signals(self, detections):
        saw_red_light = False
        new_speed_limits = []

        for det in detections:
            label = det['class']
            if label == 'Traffic Light Red' and det['distance'] < self.red_light_threshold:
                saw_red_light = True

            elif label.startswith('Speed-limit'):
                speed = self.extract_speed_from_label(label)
                if speed is not None:
                    new_speed_limits.append(speed)

        if new_speed_limits:
            self.speed_limit = min(new_speed_limits)

        return {'red_light': saw_red_light, 'speed_limit': self.speed_limit}
