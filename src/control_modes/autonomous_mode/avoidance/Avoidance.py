import logging
import cv2

class Avoidance:
    def __init__(self):
        self.car_seen_counter = 0
        self.car_on_left = False

    def process(self, frame, detections):
        car_detected = False
        frame_center = frame.shape[1] // 2

        for det in detections:
            if det['label'] == 'car':
                x_center = (det['bbox'][0] + det['bbox'][2]) / 2
                car_detected = True
                self.car_on_left = x_center < frame_center
                break

        if car_detected:
            self.car_seen_counter = 5
        else:
            self.car_seen_counter = max(0, self.car_seen_counter - 1)

        draw_instructions = []

        if self.car_seen_counter > 0:
            bias = 0.3
            avoidance_steering = -bias if self.car_on_left else bias
            speed_scale = 0.85

            height, width = frame.shape[:2]
            arrow_start = (width // 2, height - 20)
            arrow_end = (int(arrow_start[0] + 100 * avoidance_steering), arrow_start[1] - 50)

            draw_instructions.append({
                'type': 'arrow',
                'start': arrow_start,
                'end': arrow_end,
                'color': (0, 0, 255),
                'thickness': 3
            })

            draw_instructions.append({
                'type': 'text',
                'text': f"Avoiding car on {'left' if self.car_on_left else 'right'} side",
                'org': (30, 30),
                'font': 'FONT_HERSHEY_SIMPLEX',
                'font_scale': 0.7,
                'color': (0, 0, 255),
                'thickness': 2
            })

            return avoidance_steering, speed_scale, draw_instructions

        return 0.0, 1.0, []
