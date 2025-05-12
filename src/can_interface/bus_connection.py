import logging
import can
import os


def connect_to_can_interface(bus_id: int) -> can.ThreadSafeBus:
    bus_channel = "can" + str(bus_id)
    bitrate = 500000
    if os.system(f"ip link show {bus_channel}") == 0:
        os.system(f"sudo ip link set {bus_channel} type can bitrate {bitrate}")
        os.system(f"sudo ip link set {bus_channel} up")
        logging.debug(f"Initialized physical CAN interface: {bus_channel}")
        return can.ThreadSafeBus(interface='socketcan', channel=bus_channel, bitrate=bitrate)
    logging.error(
        f'Interface {bus_channel} not available. Error when initialising.\nCreating a virtual interface instead.')
    os.system("sudo ip link add dev vcan0 type vcan")
    os.system("sudo ip link set dev vcan0 up")
    logging.debug("Initialized virtual CAN interface: vcan0")
    return can.ThreadSafeBus(interface='virtual', channel='vcan0', bitrate=bitrate, receive_own_messages=True)


def disconnect_from_can_interface(can_bus: can.ThreadSafeBus) -> None:
    logging.debug("Shutting down CAN interface.")
    can_bus.shutdown()
