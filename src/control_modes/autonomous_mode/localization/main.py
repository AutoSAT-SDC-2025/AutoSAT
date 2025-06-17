from .localization import Localizer
from .mapper import Mapper
from .lane_detection import LaneDetector
import numpy as np
import cv2 as cv
import time
from pathlib import Path
from .evaluation import get_angle

start_frame = 1119
start_frame = 1552
start_frame = 500
map_scale = 0.0483398

frame_angles = [(555,848), (1449, 1523), (1721, 1794),
                (1893, 2004), (2028, 2090), (2100, 2285)]
map_angles = [np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2]

x, y = ((5440)*(1/map_scale),350*(1/map_scale))
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
# localizer.set_start_location(x, y, theta)
gframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
errors = []
particle_x_std_list = []
particle_y_std_list = []
particle_theta_std_list = []
pos_theta_list = []
frames_list = []

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
    true_angle = get_angle(lane)
    print("Ground truth angle: ", true_angle)

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
    # cv.imshow("lane", lane)
    map_input = cv.resize(localizer.mapper.get_sight((localizer.x, localizer.y), localizer.theta), (128, 64))
    # cv.imshow("map_input", map_input)
    # cv.imshow("map", localizer.mapper.get_map_with_car(localizer.x,localizer.y,localizer.theta))
    map_img = localizer.mapper.get_map_with_car(localizer.x, localizer.y, localizer.theta)
    x_list = localizer.particle_filter.particles[:,0]
    y_list = localizer.particle_filter.particles[:,1]
    theta_list = localizer.particle_filter.particles[:,2]
    
    particle_x_std_list.append(np.std(np.array(x_list)))
    particle_y_std_list.append(np.std(np.array(y_list)))
    particle_theta_std_list.append(np.std(np.array(theta_list)))
    
    map_img = localizer.mapper.get_map_with_particles_and_car(x_list, y_list, theta_list, localizer.x, localizer.y, localizer.theta)
    h, w, _ = map_img.shape
    canvas = np.zeros((h*2, w, 3), dtype=np.uint8)
    canvas[:h, :w] = map_img
    
    lane_rgb = cv.cvtColor(lane, cv.COLOR_GRAY2BGR)
    lane_rgb = cv.resize(lane_rgb, None, fx=2, fy=2)
    h_lane, w_lane, _ = lane_rgb.shape
    canvas[h:h+h_lane, :w_lane] = lane_rgb
    
    map_input_rgb = cv.cvtColor(map_input, cv.COLOR_GRAY2BGR)
    map_input_rgb = cv.resize(map_input_rgb, None, fx=2, fy=2)
    h_map, w_map, _ = map_input_rgb.shape
    canvas[h:h+h_map, w_lane:w_lane+w_map] = map_input_rgb

    frame_canvas = cv.resize(frame, None, fx=0.2, fy=0.2)
    frame_h, frame_w, _ = frame_canvas.shape
    canvas[h:h+frame_h, w_lane+w_map:w_lane+w_map+frame_w] = frame_canvas
    
    cv.imshow("canvas", canvas)
    
    # cv.imshow("particles", localizer.mapper.get_map_with_particles(x_list, y_list, theta_list))

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
    pos_theta_list.append(rotation)
    frames_list.append(i)

    # Evaluation
    for k, frame_range in enumerate(frame_angles):
        num_frame_1, num_frame_2 = frame_range
        if num_frame_1 < i < num_frame_2:
            print("error: ", rotation-(map_angles[k]-true_angle))
            errors.append(rotation-(map_angles[k]-true_angle))

    
    
    
    print("Frame_count: ", i)
    print("-"*10)
    if cv.waitKey(0) & 0xFF == ord('q'):
        # cv.imwrite("data/lane.png", lane)
        # cv.imwrite("data/map_input.png", map_input)
        # cv.imwrite("data/lane_input.png", lane)
        # cv.imwrite("data/topdown1.png", localizer.preprocess(prev_frame))
        # cv.imwrite("data/topdown2.png", localizer.preprocess(frame))
        cv.imwrite(f"data/particle{i}.png", map_img)
        # break
    prev_frame = frame
errors = np.array(errors)
np.save("errors.npy", errors)
np.save("particles.npy", [particle_x_std_list, particle_y_std_list, particle_theta_std_list])
np.save("theta.npy", pos_theta_list)
np.save("frames.npy", frames_list)
cv.destroyAllWindows()
if mapping is True:
    scale = min(1080/mapper.map.shape[0], 1080/mapper.map.shape[1])
    cv.imshow("map", cv.resize(mapper.map,None, fx=scale, fy=scale))
    cv.waitKey(0)
    cap.release()
