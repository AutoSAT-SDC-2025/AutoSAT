import logging
import sys
import cv2
import os
import time
import can
import struct
from queue import Queue
from datetime import datetime
from collections import deque
from typing import Dict
import platform
from pathlib import Path
import threading

import numpy as np

from src.misc import print_can_messages, setup_listeners

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from constants import CAN_MSG_SENDING_SPEED, height, width, scale
from can_listener import CanListener
from image_worker import ImageWorker
from can_worker import CanWorker, initialize_can
from camera_utils import initialize_cameras, getHorizon
from line_detection import getLines, newLines, splitLines, findTarget
from object_detection import initialize, traffic_object_detection, adjust_throttle
from ..line_detection.LineDetection import LineFollowingNavigation
from ....util.Render import Renderer


from src.can_interface.can_factory import select_can_controller_creator, create_can_controller
from src.can_interface.bus_connection import connect_to_can_interface, disconnect_from_can_interface
from src.car_variables import CarType, KartGearBox, HunterControlMode, HunterFeedbackCanIDs, KartFeedbackCanIDs


def main():
    """
    Main loop of the self-driving car, populates the frame queue and writes to the self-driving car its steering and
    throttle, it also initializes everything that is needed.
    """
    # Initialize CAN bus
    # bus = initialize_can()
    bus = connect_to_can_interface(0)
    car_type = CarType.hunter
    can_creator = select_can_controller_creator(car_type)
    can_controller = create_can_controller(can_creator, bus)
    can_controller.start()
    # setup_listeners(can_controller, car_type)
    line_detection = LineFollowingNavigation(width=width, height=height)
    renderer = Renderer()
    # Initialize cameras
    cameras = initialize_cameras()
    front_camera = cameras["front"]

    print('Creating folders...', file=sys.stderr)
    log_root = "logs"
    os.makedirs(log_root, exist_ok=True)

    recording_folder_name = "recording " + datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    recording_folder = os.path.join(log_root, recording_folder_name)
    os.makedirs(recording_folder, exist_ok=True)
    for subdir in cameras.keys():
        os.makedirs(os.path.join(recording_folder, subdir), exist_ok=True)

    # Initialize CAN listener and workers
    can_listener = CanListener(bus)
    can_listener.start_listening()
    image_queue = Queue()
    image_worker = ImageWorker(image_queue, recording_folder)
    image_worker.start()
    can_worker = CanWorker(Queue(), recording_folder)
    can_worker.start()

    print('Recording...', file=sys.stderr)
    frames: Dict[str, cv2.Mat] = dict()

    # Object detection initialization
    weights_path = 'src/control_modes/autonomous_mode/old_twente_code/v5_model.pt'
    output_directory_base = 'logs/detection_frames'
    model, device, gui_available = initialize(weights_path, output_directory_base)
    print("GUI available: ", gui_available)

    # Shared state and queues
    shared_state = {
        "spotted_red_light": False,
        "Speed limit": 10,
        "Initial Person Position": "None",
        "Current Person Position": "None",
        "Car Spotted": False
    }

    MAX_CAR_SPEED = 20
    queue_maxsize = 3
    state_queue = deque(maxlen=queue_maxsize)
    frame_queue = deque(maxlen=queue_maxsize)
    throttle_queue = deque(maxlen=1)
    object_stop_event = threading.Event()

    # Initialize threads
    frame_processing_thread = threading.Thread(
        target=traffic_object_detection, args=(frame_queue, state_queue, model, device, object_stop_event)
    )
    throttle_adjustment_thread = threading.Thread(
        target=adjust_throttle, args=(state_queue, throttle_queue, MAX_CAR_SPEED)
    )

    frame_processing_thread.start()
    throttle_adjustment_thread.start()
    print("Object detection threads started...")

    try:
        # # Define CAN messages
        # brake_msg = can.Message(arbitration_id=0x110, is_extended_id=False, data=[0, 0, 0, 0, 0, 0, 0, 0])
        # brake_task = bus.send_periodic(brake_msg, CAN_MSG_SENDING_SPEED)
        # steering_msg = can.Message(arbitration_id=0x220, is_extended_id=False, data=[0, 0, 0, 0, 0, 0, 0, 0])
        # steering_task = bus.send_periodic(steering_msg, CAN_MSG_SENDING_SPEED)
        # throttle_msg = can.Message(arbitration_id=0x330, is_extended_id=False, data=[0, 0, 0, 0, 0, 0, 0, 0])
        # throttle_task = bus.send_periodic(throttle_msg, CAN_MSG_SENDING_SPEED)
        #
        # time.sleep(2)

        # Start running
        start_time = time.time()
        frame_count = 0

        # Horizon detection
        ret, frame = front_camera.read()
        if not ret or frame is None:
            print("[ERROR] Failed to read frame! Substituting black frame...")
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
        hx, hy = getHorizon(frame)
        print("Horizon found at", hy)
        countL = 0
        countR = 0
        countMax = 3

        #overtaking initialization
        car_passed = False
        t0 = 0
        tprev = 0
        distance_driven = 0
        turnsens = 0.2
        throttle_index = 0
        car_spotted = False
        try:
            while True:
                # Recording part
                ok_count = 0
                values = can_listener.get_new_values()
                timestamp = time.time()

                # First grab all camera frames
                for camera in cameras.values():
                    camera.grab()

                # Then retrieve them
                for side, camera in cameras.items():
                    ok, frames[side] = camera.retrieve()
                    ok_count += ok

                if ok_count == len(cameras):
                    for side, frame in frames.items():
                        image_worker.put((timestamp, side, frame))
                    can_worker.put((timestamp, values))

                # Get camera data
                _, frame = front_camera.read()
                # add frame to frame queue (in cv2 format)
                frame_queue.append(frame)

                if len(throttle_queue) != 0:
                    throttle_state = throttle_queue.pop()
                    throttle_index = throttle_state['throttle']
                    car_spotted = throttle_state['car in range']
                    kill_object_detection = throttle_state['kill object detection']

                throttle_msg_data=[throttle_index, 0, 1, 0, 0, 0, 0, 0]

                if car_type is CarType.hunter:
                    # await can_controller.set_throttle(throttle_index)
                    can_controller.set_control_mode(HunterControlMode.command_mode)
                    can_controller.set_parking_mode(False)
                    can_controller.set_steering_and_throttle(0.0,throttle_index*10)
                elif car_type is CarType.kart:
                    can_controller.set_kart_gearbox(KartGearBox.forward)
                    can_controller.set_throttle(throttle_index)
                # throttle_msg.data = [throttle_index, 0, 1, 0, 0, 0, 0, 0]

                # Steering part
                # lines = getLines(frame)
                # if lines is not None:
                #     lines = newLines(lines)
                #     llines, rlines = splitLines(lines)
                #
                #     wl = 1
                #     wr = 1
                #
                #     if car_spotted and not car_passed:
                #         print(".")
                #         if t0 == 0:
                #             t0 = timestamp
                #             tprev = timestamp
                #             print("CAR SPOTTED")
                #         tnow = timestamp - t0
                #         speed = int(struct.unpack(">H", bytearray(values["speed_sensor"][:2]))[0]) if values["speed_sensor"] else 0
                #         speed = int(speed / 36)
                #         distance_change = speed * (tnow - tprev)
                #         distance_driven += distance_change
                #         tprev = tnow
                #
                #         if distance_driven < 5:
                #             target = findTarget(llines, rlines, hy, frame, wl, wr, weight=0, bias=-400, draw=0)
                #             print("Going Left")
                #         elif distance_driven < 12:
                #             target = findTarget(llines, rlines, hy, frame, wl, wr, weight=1, bias=0, draw=0)
                #             print("Going Straight")
                #         elif distance_driven < 17:
                #             target = findTarget(llines, rlines, hy, frame, wl, wr, weight=0, bias=400, draw=0)
                #             print("Going Right")
                #         else:
                #             print("Overtaking Completed")
                #             car_passed = True
                #     else:
                #         target = findTarget(llines, rlines, hy, frame, wl, wr, weight=1, bias=0, draw=0)
                #
                #     if target is False:
                #         print("ERROR, NO LINES FOUND")
                #         throttle_msg_data = [1, 0, 1, 0, 0, 0, 0, 0]
                #         steer_angle = 0
                #     else:
                #         Error = target - width / 2
                #         if Error > 0:
                #             steer_angle = min(Error / (width / 2), 1.05)
                #         else:
                #             steer_angle = max(Error / (width / 2), -1.05)
                # else:
                #     print("ERROR, NO LINES FOUND")
                #     throttle_msg_data = [1, 0, 1, 0, 0, 0, 0, 0]
                #     steer_angle = 0
                renderer.clear()
                steer_angle, speed, line_visuals = line_detection.process(frame)
                renderer.add_drawings(line_visuals)
                renderer.draw(frame)

                # steering_msg.data = list(bytearray(struct.pack("f", float(steer_angle)))) + [0] * 4
                # steering_task.modify_data(steering_msg)
                # throttle_task.modify_data(throttle_msg)
                if car_type is CarType.hunter:
                    # await can_controller.set_throttle(throttle_msg_data[0])
                    # await can_controller.set_steering(steer_angle)
                    can_controller.set_steering_and_throttle(steer_angle*5760, throttle_msg_data[0]*20)
                    # logging.debug(f"Sent Hunter CAN message: Steering={steer_angle * 5760}, Throttle={throttle_msg_data[0] * 20}")
                elif car_type is CarType.kart:
                    can_controller.set_kart_gearbox(KartGearBox.forward)
                    can_controller.set_throttle(throttle_msg_data[0])
                    can_controller.set_steering(steer_angle)
                    # logging.debug(f"Sent Kart CAN message: Steering={steer_angle}, Throttle={throttle_msg_data[0]}")

                # if throttle_msg.data[0] == 0:
                #     brake_msg.data = [50, 0, 1, 0, 0, 0, 0, 0]
                #     brake_task.modify_data(brake_msg)
                # else:
                #     brake_msg.data = [0, 0, 1, 0, 0, 0, 0, 0]
                #     brake_task.modify_data(brake_msg)

                if throttle_msg_data[0] == 0:
                    brake_msg_data = [50, 0, 1, 0, 0, 0, 0, 0]
                else:
                    brake_msg_data = [0, 0, 1, 0, 0, 0, 0, 0]

                if car_type is CarType.hunter:
                    # await can_controller.set_throttle(-0.5*throttle_msg_data[0])
                    can_controller.set_parking_mode(False)
                elif car_type is CarType.kart:
                    can_controller.set_break(brake_msg_data[0])


                frame_count += 1
        except KeyboardInterrupt:
            pass

    finally:
        can_controller.set_control_mode(HunterControlMode.idle_mode)
        end_time = time.time()
        time_diff = end_time - start_time
        print(f'Time elapsed: {time_diff:.2f}s')
        print(f'Frames processed: {frame_count}')
        print(f'FPS: {frame_count / time_diff:.2f}')
        print('Stopping...', file=sys.stderr)

        object_stop_event.set()

        frame_processing_thread.join()
        throttle_adjustment_thread.join()

        can_listener.stop_listening()

        for camera in cameras.values():
            camera.release()

        image_worker.stop()
        can_worker.stop()

        # brake_task.stop()
        # steering_task.stop()
        # throttle_task.stop()


if __name__ == '__main__':
    main()