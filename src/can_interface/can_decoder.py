import struct
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, ClassVar, Union
from can import Message
from ..car_variables import HunterFeedbackCanIDs, KartFeedbackCanIDs
from ..web_interface.websocket_manager import sync_broadcast_can_json

@dataclass
class CanMessageData:
    type: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {key: getattr(self, key) for key in self.__annotations__}


@dataclass
class HunterMovementData(CanMessageData):
    speed: float = 0.0
    steering: float = 0.0
    type: str = field(default="hunter_movement")

    def __str__(self) -> str:
        return f"Speed: {self.speed:.2f} m/s | Steering: {self.steering:.2f} rad"


@dataclass
class HunterStatusData(CanMessageData):
    body_status: str = "Unknown"
    control_mode: str = "Unknown"
    brake_status: str = "Unknown"
    type: str = field(default="hunter_status")

    def __str__(self) -> str:
        return f"Body: {self.body_status} | Control: {self.control_mode} | Brake: {self.brake_status}"


@dataclass
class KartSteeringData(CanMessageData):
    steering_raw: int = 0
    type: str = field(default="kart_steering")

    def __str__(self) -> str:
        return f"Steering: {self.steering_raw}"


@dataclass
class KartBreakingData(CanMessageData):
    current_pot: int = 0
    target_pot: int = 0
    direction: str = "Unknown"
    speed: int = 0
    error: str = "Unknown"
    type: str = field(default="kart_breaking")

    def __str__(self) -> str:
        return f"Current: {self.current_pot} | Target: {self.target_pot} | Direction: {self.direction} | Speed: {self.speed} | Status: {self.error}"


@dataclass
class KartThrottleData(CanMessageData):
    throttle_voltage: int = 0
    braking: str = "Not Braking"
    gear: str = "N"
    state: str = "Active"
    type: str = field(default="kart_throttle")

    def __str__(self) -> str:
        return f"Throttle: {self.throttle_voltage} | Braking: {self.braking} | Gear: {self.gear} | State: {self.state}"


@dataclass
class KartSpeedData(CanMessageData):
    speed: float = 0.0
    speed_hmh: int = 0
    type: str = field(default="kart_speed")

    def __str__(self) -> str:
        return f"Speed: {self.speed:.2f} m/s"


@dataclass
class DecodedMessage:
    timestamp: str
    id: str
    raw_id: int
    hex_data: str
    type: str
    data: Union[HunterMovementData, HunterStatusData, KartSteeringData,
                KartBreakingData, KartThrottleData, KartSpeedData, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "timestamp": self.timestamp,
            "type": self.type,
            "id": self.id,
            "data": self.data.to_dict() if isinstance(self.data, CanMessageData) else {},
        }
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def __str__(self) -> str:
        data_str = str(self.data) if isinstance(self.data, CanMessageData) else f"Raw: {self.hex_data}"
        return f"[{self.timestamp}] ID: {self.id} | {data_str}"


class CanDecoder:
    """Decodes CAN messages based on the encoding used in controller classes"""

    BODY_STATUS_MAP: ClassVar[Dict[int, str]] = {
        0x00: "Normal",
        0x01: "Warning",
        0x10: "Error"
    }

    CONTROL_MODE_MAP: ClassVar[Dict[int, str]] = {
        0x00: "Idle Mode",
        0x01: "Command Mode",
        0x02: "Remote Mode"
    }

    GEAR_MAP: ClassVar[Dict[int, str]] = {
        0: "N",
        1: "F",
        2: "R"
    }

    @staticmethod
    def decode_message(message: Message) -> DecodedMessage:
        """Decode CAN message into structured data for easy serialization"""
        timestamp = datetime.fromtimestamp(message.timestamp).strftime('%H:%M:%S.%f')[:-3]
        message_id = message.arbitration_id
        hex_data = ' '.join([f"{b:02X}" for b in message.data])

        data: Union[CanMessageData, Dict[str, Any]] = {}
        msg_type = "unknown"

        try:
            if message_id == HunterFeedbackCanIDs.movement_feedback.value:
                if len(message.data) >= 8:
                    speed_raw = struct.unpack('>h', bytes(message.data[0:2]))[0]
                    speed_mps = speed_raw / 1000.0
                    steering_raw = struct.unpack('>h', bytes(message.data[6:8]))[0]
                    steering_radians = steering_raw / 1000.0

                    data = HunterMovementData(
                        speed=round(speed_mps, 2),
                        steering=round(steering_radians, 2)
                    )
                    msg_type = "hunter_movement"

            elif message_id == HunterFeedbackCanIDs.status_feedback.value:
                if len(message.data) >= 7:
                    body_status = message.data[0]
                    body_status_str = CanDecoder.BODY_STATUS_MAP.get(body_status, "Unknown")
                    control_mode = message.data[1]
                    control_mode_str = CanDecoder.CONTROL_MODE_MAP.get(control_mode, "Unknown")
                    brake_status = message.data[6]
                    brake_status_str = "Engaged" if brake_status == 0x01 else "Disengaged"

                    data = HunterStatusData(
                        body_status=body_status_str,
                        control_mode=control_mode_str,
                        brake_status=brake_status_str
                    )
                    msg_type = "hunter_status"

            elif message_id == KartFeedbackCanIDs.steering_sensor.value:
                if len(message.data) >= 2:
                    steering_raw = (message.data[0] << 8 | message.data[1])

                    data = KartSteeringData(steering_raw=steering_raw)
                    msg_type = "kart_steering"

            elif message_id == KartFeedbackCanIDs.breaking_sensor.value:
                if len(message.data) >= 7:
                    current_pot = (message.data[0] << 8 | message.data[1])
                    target_pot = (message.data[2] << 8 | message.data[3])
                    direction_raw = message.data[4]
                    direction = "Extending" if direction_raw == 1 else "Retracting"
                    speed = message.data[5]
                    error = "Error" if message.data[6] == 1 else "No Error"

                    data = KartBreakingData(
                        current_pot=current_pot,
                        target_pot=target_pot,
                        direction=direction,
                        speed=speed,
                        error=error
                    )
                    msg_type = "kart_breaking"

            elif message_id == KartFeedbackCanIDs.internal_throttle.value:
                if len(message.data) >= 1:
                    throttle_voltage = message.data[0]
                    braking_raw = int(message.data[1] == 1) if len(message.data) > 1 else 0
                    gear_val = message.data[2] if len(message.data) > 2 else 0
                    idle_raw = int(message.data[3] == 1) if len(message.data) > 3 else 0

                    braking = "Braking" if braking_raw else "Not Braking"
                    gear = CanDecoder.GEAR_MAP.get(gear_val, f"Unknown({gear_val})")
                    idle = "Idle" if idle_raw else "Active"

                    data = KartThrottleData(
                        throttle_voltage=throttle_voltage,
                        braking=braking,
                        gear=gear,
                        state=idle
                    )
                    msg_type = "kart_throttle"

            elif message_id == KartFeedbackCanIDs.speed_sensor.value:
                if len(message.data) >= 2:
                    speed_hmh = (message.data[0] << 8) | message.data[1]
                    speed_mps = speed_hmh / 36.0

                    data = KartSpeedData(
                        speed=round(speed_mps, 2),
                        speed_hmh=speed_hmh
                    )
                    msg_type = "kart_speed"

        except (IndexError, struct.error):
            pass

        return DecodedMessage(
            timestamp=timestamp,
            id=f"0x{message_id:03X}",
            raw_id=message_id,
            hex_data=hex_data,
            type=msg_type,
            data=data
        )


def broadcast_can_message(message: Message) -> str:
    """Process a CAN message and return a WebSocket-ready JSON string"""
    decoded = CanDecoder.decode_message(message)
    sync_broadcast_can_json(decoded.to_json())

def print_can_messages(message: Message) -> None:
    # decoded = CanDecoder.decode_message(message)
    # print(decoded)
    pass