import subprocess
import can
import os

def connect_to_can_interface(bus_id : int) -> can.ThreadSafeBus:
        bus_channel = "can" + str(bus_id)
        bitrate = 500000
        if os.system(f"ip link show {bus_channel}") == 0:
                os.system(f"sudo ip link set {bus_channel} type can bitrate {bitrate}")
                os.system(f"sudo ip link set {bus_channel} up")
                return can.ThreadSafeBus(interface='socketcan', channel=bus_channel, bitrate=bitrate)
        print(f'Interface {bus_channel} not available. Error when initialising.\nCreating a virtual interface instead.')
        return can.ThreadSafeBus(interface='virtual',channel='vcan0', receive_own_messages=True)

def disconnect_from_can_interface(can_bus: can.ThreadSafeBus) -> None:
        can_bus.shutdown()