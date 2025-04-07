from src.object_detection.detection import ObjectDetection
class PedestrianHandler:
    def __init__(self):
        self.car_distance_threshold = 2
        self.previous_detection = {}
        self.detector = ObjectDetection()

    def get_position(self, detections, width):
        current_position = {}
        for det in detections:
            x_center = (det['bbox'][0] + det['bbox'][2])/2
            obj_id = det['class']
            
            if x_center < (width / 2):
                position = 'Left'
            else:
                position = 'Right'

            current_position[obj_id] = position

        return current_position

    def get_direction(self, detections):
        directions = {}

        for det in detections:
            x_center = (det['bbox'][0] + det['bbox'][2]) / 2
            obj_id = det['class']  

            if obj_id in self.previous_detection:
                prev_x_center = self.previous_detection[obj_id]['x_center']

                if x_center > prev_x_center + 5:
                    direction = 'Right'
                elif x_center < prev_x_center - 5:
                    direction = 'Left'
                else:
                    direction = 'Stationary'
            else:
                direction = 'Unknown'  

            directions[obj_id] = direction  
            self.previous_detection[obj_id] = {'x_center': x_center}  

        return directions
            
    def get_safe_pos(self, detections, directions):
        safe_positions = {}
        for det in detections:
            obj_id = det['class']
            x_center = (det['bbox'][0] + det['bbox'][2]) / 2
            direction = directions.get(obj_id, "Unknown")
            if direction == 'Stationary' and (x_center < 50 or x_center > 800):
                print("Pedestrian is in a safe zone")
                safe_positions[obj_id] = True
            else:
                print("Pedestrian is in an unsafe zone")
                safe_positions[obj_id] = False
        return safe_positions

    def main(self, detections):
        pedestrian_passed = False

        #current_position = self.get_position(detections, width)
        directions = self.get_direction(detections)
        safe_positions = self.get_safe_pos(detections, directions)
        
        for det in detections:
            obj_id = det['class']
            x1, y1, x2, y2 = det['bbox']
            distance = self.detector.estimate_distance(self, x1, y1, x2, y2)

            if obj_id == 'person' and distance < self.car_distance_threshold:
                safe_position = safe_positions.get(obj_id, False)
                direction = directions.get(obj_id, "Unknown")

                if direction != 'Stationary' and not safe_position:
                    return False 
                elif direction == 'Stationary' and safe_position:
                    pedestrian_passed = True 

        return pedestrian_passed