from rplidar import RPLidar

class LidarScans:
    def iter_scans(lidar):
        scan_list = []
        for new_scan, _, angle, distance in lidar.iter_measures():
            if new_scan:
                if len(scan_list) > 5:
                    yield scan_list  
                scan_list = []  

            scan_list.append((angle, distance))

    def scan_area(scan, min_angle = 260, max_angle = 280):
        filtered = []
        for angle, distance in scan:
            angle = angle % 360

            if min_angle < max_angle:
                if min_angle <= angle <= max_angle:
                    filtered.append((angle, distance))
            else:
                if angle >= min_angle or angle <= max_angle:
                    filtered.append((angle, distance))

        return filtered