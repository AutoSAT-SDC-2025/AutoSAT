class TrafficManager:
    def __init__(self):
        # Distance threshold (in meters) to consider a red traffic light as relevant for stopping
        self.red_light_threshold = 4.5
        # Default speed limit (in km/h); will be updated if a lower limit is detected
        self.speed_limit = 10

    def process_traffic_signals(self, detections):
        """
        Analyze detected objects to determine traffic light and speed limit state.

        Args:
            detections (list): List of detection dicts, each with 'class' and 'distance' keys.

        Returns:
            dict: {
                'red_light': True if a red light is detected within threshold,
                'speed_limit': lowest detected speed limit (default 10)
            }
        """
        saw_red_light = False
        for det in detections:
            # Check for red traffic light within the stopping threshold
            if det['class'] == 'Traffic Light Red' and det['distance'] < self.red_light_threshold:
                saw_red_light = True
            # Update speed limit if a lower one is detected
            elif det['class'].startswith('Speed-limit'):
                speed = int(det['class'].split('-')[2].replace('km', ''))
                self.speed_limit = min(self.speed_limit, speed)
        return {'red_light': saw_red_light, 'speed_limit': self.speed_limit}