from .localization import Localizer
from .mapper import Mapper
from .lane_detection import LaneDetector
import numpy as np
import cv2 as cv
import time
from pathlib import Path

start_frame = 1119
start_frame = 1552
start_frame = 0
map_scale = 0.0483398

x, y = (5640*(1/map_scale),350*(1/map_scale))
theta = np.pi/2

mapping = False

# matrix = np.load("../v
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]  # adjust this number as needed
K = np.load(project_root/"var/camera/middle.npy")
# cap = cv.VideoCapture("/home/thimo/Seafile/Git/SDC/data/recording 29-03-2024 08-56-36.mp4")
cap = cv.VideoCapture("/home/thimo/Seafile/Git/AutoSAT/src/control_modes/autonomous_mode/localization/output.mp4")

ret, prev_frame = cap.read()

A = np.identity(3)
A[0,2] = 500
A[1,2] = 900
B = A@np.load("/home/thimo/Git/AutoSAT/src/var/B_small.npy")
np.save("lane_detection_B.npy", B)
lanedetector = LaneDetector()
if mapping is True:
    mapper = Mapper(scale=map_scale)

i = start_frame - 1
cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

if not ret:
    print("Error: Unable to read video")
    cap.release()
    exit()
ret, frame = cap.read()
frame = cv.undistort(frame, K, None)
# print(frame.shape)
# exit()
localizer = Localizer()
localizer.set_start_location(x, y, theta)
gframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
# mask = cv.warpPerspective(np.ones_like(gframe), B, dsize=(frame.shape[1]+1200, frame.shape[0]+500))


while cap.isOpened():
    i += 1
    ret, frame = cap.read()
    if not ret:
        break
    if i == start_frame:
        continue
    
    
    start_time = time.perf_counter()
    # frame = cv.undistort(frame, K, None)
    frame2 = cv.resize(frame, (848, 480))
    cv.imshow("frame", frame)
    lane = lanedetector(frame2)
    lane = cv.resize(lane, (128, 64))
    localizer.update(frame, lane)
    end_time = time.perf_counter()
    print("S: ", end_time - start_time)
    if localizer.translation is None:
        continue
    print("[dx, dy]: ", localizer.translation)
    # cv.imshow("Frame", frame)
    print(f"[x, y]: [{localizer.x}, {localizer.y}]")
    print(f"[px, py]: [{localizer.particle_filter.x}, {localizer.particle_filter.y}]")
    print(f"[_x, _y]: [{localizer._x}, {localizer._y}]")
    cv.imshow("lane", lane)
    map_input = cv.resize(localizer.mapper.get_sight((localizer.x, localizer.y), localizer.theta), (128, 64))
    cv.imshow("map_input", map_input)
    cv.imshow("map", localizer.mapper.get_map_with_car(localizer.x,localizer.y,localizer.theta))

    if mapping is True:
        mapper.update_car_location(lane, (int(localizer.x), int(localizer.y)), localizer.rotation, mask)
        scale = min(1080/mapper.map.shape[0], 1080/mapper.map.shape[1])
        x, y = mapper.get_index(-localizer.x*map_scale, localizer.y*map_scale)
        resized_map = cv.resize(mapper.map.astype(np.uint8),None, fx=scale, fy=scale) 
        resized_map = cv.cvtColor(resized_map, cv.COLOR_GRAY2BGR)
        resized_map = cv.circle(resized_map, (int(x*scale), int(y*scale)), radius=2, color=(255,0,0), thickness=2)
        x, y = mapper.get_index(-localizer._x*map_scale, localizer._y*map_scale)
        resized_map = cv.circle(resized_map, (int(x*scale), int(y*scale)), radius=2, color=(0,0,255), thickness=2)
        cv.imshow("map", resized_map)
    R = localizer.rotation
    rotation = np.arctan2(R[1, 0], R[0, 0])
    print("Theta: ", rotation)
    print("Frame_count: ", i)
    print("-"*10)
    if cv.waitKey(0) & 0xFF == ord('q'):
        cv.imwrite("data/lane.png", lane)
        cv.imwrite("data/map_input.png", map_input)
        cv.imwrite("data/lane_input.png", lane)
        break
cv.destroyAllWindows()
if mapping is True:
    scale = min(1080/mapper.map.shape[0], 1080/mapper.map.shape[1])
    cv.imshow("map", cv.resize(mapper.map,None, fx=scale, fy=scale))
    cv.waitKey(0)
    cap.release()
