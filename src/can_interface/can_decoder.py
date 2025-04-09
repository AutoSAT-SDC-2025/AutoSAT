import struct
from datetime import datetime
from can import Message
from ..car_variables import HunterFeedbackCanIDs, KartFeedbackCanIDs

class CanDecoder:
    """Decodes CAN messages based on the encoding used in controller classes"""
    
    @staticmethod
    def decode_message(message: Message) -> str:
        """Main decoder function that handles all message types"""
        timestamp = datetime.fromtimestamp(message.timestamp).strftime('%H:%M:%S.%f')[:-3]
        message_id = message.arbitration_id
        
        hex_data = ' '.join([f"{b:02X}" for b in message.data])
        decoded_data = "Unknown message format"

        if message_id == HunterFeedbackCanIDs.movement_feedback.value:
            speed_raw = struct.unpack('>h', bytes(message.data[0:2]))[0]
            speed_mps = speed_raw / 1000.0

            steering_raw = struct.unpack('>h', bytes(message.data[6:8]))[0]
            steering_radians = steering_raw / 1000.0
                
            decoded_data = f"Speed: {speed_mps:.2f} m/s | Steering: {steering_radians:.2f} rad"
        
        elif message_id == HunterFeedbackCanIDs.status_feedback.value:

            body_status = message.data[0]
            body_status_str = "Unknown"
            if body_status == 0x00:
                body_status_str = "Normal"
            elif body_status == 0x01:
                body_status_str = "Warning"
            elif body_status == 0x10:
                body_status_str = "Error"

            control_mode = message.data[1]
            control_mode_str = "Unknown"
            if control_mode == 0x00:
                control_mode_str = "Idle Mode"
            elif control_mode == 0x01:
                control_mode_str = "Command Mode"
            elif control_mode == 0x02:
                control_mode_str = "Remote Mode"

            brake_status = message.data[6]
            brake_status_str = "Disengaged"
            if brake_status == 0x01:
                brake_status_str = "Engaged"

            decoded_data = f"Body: {body_status_str} | Control: {control_mode_str} | Brake: {brake_status_str})"

        elif message_id == KartFeedbackCanIDs.steering_sensor.value:
            steering_raw = (message.data[0] << 8 | message.data[1])
            decoded_data = f"Steering: {steering_raw} raw counts"

        elif message_id == KartFeedbackCanIDs.breaking_sensor.value:
            current_pot = (message.data[0] << 8 | message.data[1])
            target_pot = (message.data[2] << 8 | message.data[3])
            direction = "Extending" if message.data[4] == 1 else "Retracting"
            speed = message.data[5]
            error = "Error" if message.data[6] == 1 else "No Error"
            decoded_data = f"Current potentiometer {current_pot} | Target potentiometer {target_pot} | Direction: {direction} | Speed: {speed} | Status: {error}"

        elif message_id == KartFeedbackCanIDs.internal_throttle.value:
            throttle_voltage = message.data[0]
            braking = "Braking" if len(message.data) > 1 and message.data[1] == 1 else "Not Braking"
            gear_val = message.data[2]
            gear = {0: "N", 1: "F", 2: "R"}.get(gear_val, f"Unknown({gear_val})")
            idle = "Idle" if len(message.data) > 3 and message.data[3] == 1 else "Active"
            decoded_data = f"Throttle Voltage: {throttle_voltage} | Braking: {braking} | Gear: {gear} | State: {idle}"

        elif message_id == KartFeedbackCanIDs.speed_sensor.value:
            speed_hmh = (message.data[0] << 8) | message.data[1]
            speed_mps = speed_hmh / 36.0
            decoded_data = f"Speed: {speed_mps:.2f} m/s ({speed_hmh} hm/h)"

        if decoded_data == "Unknown message format":
            decoded_data = f"Raw: {hex_data}"

        return f"[{timestamp}] ID: 0x{message_id:03X} | {decoded_data}"# | Data: {hex_data}"


def print_can_messages(message: Message) -> None:
    """
    Decodes and prints CAN messages in a human-readable format
    To be used as a listener callback in setup_listeners
    """
    decoded = CanDecoder.decode_message(message)
    print(decoded, flush=True)