from src.object_detection.detection import ObjectDetection
from src.obstacle_avoidance.lidar_scan import LidarScans

class VehicleHandler:
    def __init__(self):
        self.car_distance_threshold = 3
        self.overtake_state = 'Driving state'

    def overtake_vehicle(self, saw_vehicle):
        self.overtake_state = 'None'
        if saw_vehicle == True:
            self.overtake_state = 'Turning left'
            return self.overtake_state
        else:
            self.overtake_state = 'Driving straight'
            return self.overtake_state
    
    def free_area(self, scan):
        front_sector = LidarScans.scan_area(scan, min_angle=260, max_angle=280)
        
        for angle, distance in front_sector:
            if distance < self.clear_distance_threshold:
                return False 
        return True
    
    def vehicle_passed(self, overtake_state, scan):
        if overtake_state == 'Driving straight' and self.free_area(scan):
            return True
        return False
    
    def process_vehicle(self, detections):
        saw_vehicle = False
        for det in detections:
            if det['class'] == 'car' and det['distance'] < self.car_distance_threshold:
                saw_vehicle = True
                

        return {'Vehicle seen': saw_vehicle}