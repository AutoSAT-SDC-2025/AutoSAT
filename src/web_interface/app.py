from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import sys
import pathlib
import threading
import asyncio
import cv2
import base64
import logging
from contextlib import asynccontextmanager
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from src.car_variables import CarType, CameraResolution
from src.control_modes.manual_mode.manual_mode import ManualMode
from src.control_modes.autonomous_mode.autonomous_mode import AutonomousMode
from src.camera.camera_controller import CameraController
from src.misc import setup_listeners

from src.data_logger.logger_manager import DataLoggerManager

from .websocket_manager import can_ws_manager

class CameraManager:
    def __init__(self):
        self.active = False
        self.camera_controller = None
        self.camera_thread = None
        self.frame = None
        self.clients = []
        self.frame_task = None
        self.view_mode = "front"
        try:
            self.camera_controller = CameraController()
            self.camera_controller.enable_cameras()
            self.camera_controller.setup_cameras()
        except Exception as e:
            logger.error(f"Error initializing camera controller: {e}")
            self.camera_controller = None

    def start(self):
        if self.active:
            return False

        try:
            self.active = True
            self.camera_thread = threading.Thread(target=self._camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()

            logger.info("CameraController started successfully")
            return True

        except Exception as e:
            logger.error(f"Error starting CameraController: {e}")
            return False

    def stop(self):
        self.active = False
        if self.camera_controller:
            try:
                self.camera_controller.disable_cameras()
                self.frame = None
                logger.info("CameraController stopped")
                return True
            except Exception as e:
                logger.error(f"Error stopping camera controller: {e}")
        return False

    def set_view_mode(self, mode):
        """Set the camera view mode: front, left, right, topdown, stitched"""
        if mode in ["front", "left", "right", "topdown", "stitched"]:
            self.view_mode = mode
            return True
        return False

    def _camera_loop(self):
        """Thread function to continuously read frames based on view mode"""
        while self.active and self.camera_controller is not None:
            try:
                self.camera_controller.capture_camera_frames()
                
                if self.view_mode == "front":
                    self.frame = self.camera_controller.get_front_view()
                elif self.view_mode == "left":
                    self.frame = self.camera_controller.get_left_view()
                elif self.view_mode == "right":
                    self.frame = self.camera_controller.get_right_view()
                elif self.view_mode == "topdown":
                    self.frame = self.camera_controller.get_top_down_view()
                elif self.view_mode == "stitched":
                    self.frame = self.camera_controller.get_stitched_image()

                if self.frame is not None and (self.frame.shape[1] > CameraResolution.WIDTH or self.frame.shape[0] > CameraResolution.HEIGHT):
                    self.frame = cv2.resize(self.frame, (CameraResolution.WIDTH, CameraResolution.HEIGHT))
                
            except Exception as e:
                logger.error(f"Error in camera loop: {e}")
                time.sleep(0.1)

    async def send_frame_to_clients(self):
        """Send the latest frame to all connected clients"""
        while True:
            if self.active and self.frame is not None and self.clients:
                try:
                    _, buffer = cv2.imencode('.jpg', self.frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')

                    for client in list(self.clients):
                        try:
                            await client.send_json({
                                "type": "frame",
                                "data": frame_base64,
                                "view_mode": self.view_mode
                            })
                        except Exception:
                            if client in self.clients:
                                self.clients.remove(client)
                except Exception as e:
                    logger.error(f"Error sending frame: {e}")
            await asyncio.sleep(0.033)  # ~30 FPS

    def add_client(self, websocket):
        self.clients.append(websocket)

    def remove_client(self, websocket):
        if websocket in self.clients:
            self.clients.remove(websocket)

class ControlManager:
    def __init__(self, camera_controller = None):
        self.mode = None
        self.car_type = None
        self.controller = None
        self.controller_thread = None
        self.running = False
        self.camera_controller = camera_controller
        if self.camera_controller is None:
            try:
                self.camera_controller = CameraController()
                self.camera_controller.enable_cameras()
                self.camera_controller.setup_cameras()
            except Exception as e:
                logging.error(f"Error initializing camera controller: {e}")
                self.camera_controller = None

    def set_mode(self, mode):
        """Set the control mode"""
        if mode not in ['manual', 'autonomous']:
            return False

        if self.running:
            self.stop_controller()

        self.mode = mode
        logger.info(f"Mode set to {mode}")
        return True

    def set_car_type(self, car_type_str):
        """Set the car type"""
        if car_type_str == 'kart':
            self.car_type = CarType.kart
        elif car_type_str == 'hunter':
            self.car_type = CarType.hunter
        else:
            return False

        if self.running:
            self.stop_controller()

        logger.info(f"Car type set to {car_type_str}")
        return True

    def start_controller(self):
        """Start the selected control mode"""
        if self.running:
            return False

        if not self.mode or self.car_type is None:
            logger.error("Cannot start controller: mode or car type not set")
            return False

        try:
            if self.mode == 'manual':
                logger.info("Starting manual mode...")
                self.controller = ManualMode(self.car_type)
            else:
                logger.info("Starting autonomous mode...")
                self.controller = AutonomousMode(self.car_type, self.camera_controller)

            setup_listeners(self.controller.can_controller, self.car_type)
            logger.info(f"CAN listeners registered for {self.car_type} controller")

            self.running = True
            self.controller_thread = threading.Thread(target=self._run_controller)
            self.controller_thread.daemon = True
            self.controller_thread.start()

            return True
        except Exception as e:
            logger.error(f"Error starting controller: {e}")
            self.running = False
            self.controller = None
            return False

    def _run_controller(self):
        """Thread function to run the controller"""
        try:
            self.controller.start()
        except Exception as e:
            logger.error(f"Error in controller thread: {e}")
        finally:
            self.running = False
            self.controller = None

    def stop_controller(self):
        """Stop the current controller"""
        if not self.running or not self.controller:
            return False

        try:
            self.controller.stop()
            self.camera_controller.disable_cameras()
            self.running = False
            self.controller = None
            self.camera_controller = None
            logger.info("Controller stopped")
            logger.info("CameraController stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping controller: {e}")
            return False

    def get_status(self):
        """Get the current status of the control manager"""
        return {
            'mode': self.mode,
            'car_type': 'kart' if self.car_type == CarType.kart else 'hunter' if self.car_type == CarType.hunter else None,
            'running': self.running
        }

camera_manager = CameraManager()
camera_manager.start()
control_manager = ControlManager(camera_manager.camera_controller)
data_logger_manager = DataLoggerManager(camera_manager.camera_controller)

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    logger.info("Starting up camera service")
    camera_manager.frame_task = asyncio.create_task(camera_manager.send_frame_to_clients())

    yield

    logger.info("Shutting down services")
    camera_manager.stop()
    if camera_manager.frame_task:
        camera_manager.frame_task.cancel()
        try:
            await camera_manager.frame_task
        except asyncio.CancelledError:
            pass

    if control_manager.running:
        control_manager.stop_controller()

app = FastAPI(title="AutoSAT Control Panel", lifespan=lifespan)


templates = Jinja2Templates(directory=str(pathlib.Path(__file__).parent / "templates"))
static_dir = pathlib.Path(__file__).parent / "static"
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/mode")
async def set_mode(request: Request):
    data = await request.json()
    mode = data.get("mode")
    success = control_manager.set_mode(mode)

    return JSONResponse({
        "success": success,
        "message": f"Mode set to {mode}" if success else "Invalid mode",
        "status": control_manager.get_status()
    })

@app.post("/api/car-type")
async def set_car_type(request: Request):
    data = await request.json()
    car_type = data.get("car_type")
    success = control_manager.set_car_type(car_type)

    return JSONResponse({
        "success": success,
        "message": f"Car type set to {car_type}" if success else "Invalid car type",
        "status": control_manager.get_status()
    })

@app.api_route("/api/start", methods=["GET", "POST"])
async def start_controller(request: Request):
    try:
        if request.method == "POST":
            data = await request.json()
            if data and isinstance(data, dict):
                # Update mode if provided
                if "mode" in data:
                    control_manager.set_mode(data["mode"])
                
                # Update car type if provided
                if "car_type" in data:
                    control_manager.set_car_type(data["car_type"])
    except Exception as e:
        logger.error(f"Error processing start request data: {e}")

    success = control_manager.start_controller()
    
    return JSONResponse({
        "success": success,
        "message": "Controller started" if success else "Failed to start controller",
        "status": control_manager.get_status()
    })

@app.api_route("/api/stop", methods=["GET", "POST"])
async def stop_controller():
    success = control_manager.stop_controller()
    
    return JSONResponse({
        "success": success,
        "message": "Controller stopped" if success else "Failed to stop controller",
        "status": control_manager.get_status()
    })

@app.get("/api/status")
async def get_status():
    return JSONResponse({
        "success": True,
        "status": control_manager.get_status()
    })

@app.post("/api/camera-view")
async def set_camera_view(request: Request):
    data = await request.json()
    view_mode = data.get("view_mode")
    success = camera_manager.set_view_mode(view_mode)

    return JSONResponse({
        "success": success,
        "message": f"Camera view set to {view_mode}" if success else f"Invalid camera view: {view_mode}",
        "current_view": camera_manager.view_mode
    })

@app.websocket("/ws/camera")
async def websocket_camera_endpoint(websocket: WebSocket):
    await websocket.accept()

    if not camera_manager.active:
        camera_manager.start()

    camera_manager.add_client(websocket)

    if camera_manager.frame is not None:
        try:
            _, buffer = cv2.imencode('.jpg', camera_manager.frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            await websocket.send_json({
                "type": "frame",
                "data": frame_base64,
                "view_mode": camera_manager.view_mode
            })
        except Exception as e:
            logger.error(f"Error sending initial frame: {e}")

    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
            elif data.startswith("view:"):
                view_mode = data.split(":")[1]
                if camera_manager.set_view_mode(view_mode):
                    await websocket.send_json({
                        "type": "view_changed",
                        "view_mode": camera_manager.view_mode
                    })
    except WebSocketDisconnect:
        camera_manager.remove_client(websocket)
        if not camera_manager.clients:
            camera_manager.stop()

@app.websocket("/ws/can")
async def websocket_can_endpoint(websocket: WebSocket):
    await websocket.accept()
    can_ws_manager.add_client(websocket)
    logger.info("New CAN WebSocket client connected")
    
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        can_ws_manager.remove_client(websocket)
        logger.info("CAN WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Error in CAN WebSocket: {e}")
        can_ws_manager.remove_client(websocket)

@app.post("/api/logger/start")
async def start_logger():
    if not data_logger_manager:
        logger.error("Data logger manager not initialized")
        return JSONResponse({
            "success": False,
            "message": "Data logger not available",
            "status": {"enabled": False, "log_dir": None}
        })

    try:
        logger.info("Attempting to start data logger...")
        data_logger_manager.enable_logger()
        logger.info(f"Data logger started successfully. Log dir: {data_logger_manager.log_dir}")
        return JSONResponse({
            "success": True,
            "message": "Data logger started",
            "status": {"enabled": True, "log_dir": data_logger_manager.log_dir}
        })
    except Exception as e:
        logger.error(f"Error starting data logger: {e}")
        return JSONResponse({
            "success": False,
            "message": f"Error starting data logger: {e}",
            "status": {"enabled": False, "log_dir": None}
        })


@app.post("/api/logger/stop")
async def stop_logger():
    if not data_logger_manager:
        return JSONResponse({
            "success": False,
            "message": "Data logger not available",
            "status": {"enabled": False, "log_dir": None}
        })

    try:
        data_logger_manager.disable_logger()
        return JSONResponse({
            "success": True,
            "message": "Data logger stopped",
            "status": {"enabled": False, "log_dir": None}
        })
    except Exception as e:
        logger.error(f"Error stopping data logger: {e}")
        return JSONResponse({
            "success": False,
            "message": f"Error stopping data logger: {e}",
            "status": {"enabled": data_logger_manager.enabled, "log_dir": data_logger_manager.log_dir}
        })


@app.api_route("/api/logger/status", methods=["GET", "POST"])
async def get_logger_status():
    if not data_logger_manager:
        return JSONResponse({
            "success": False,
            "message": "Data logger not available",
            "status": {"enabled": False, "log_dir": None}
        })

    return JSONResponse({
        "success": True,
        "status": {
            "enabled": data_logger_manager.enabled,
            "log_dir": data_logger_manager.log_dir
        }
    })