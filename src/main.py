from time import process_time_ns

from src.gamepad import Gamepad


def main():
    gamepad = Gamepad.Xbox360()
    gamepad.startBackgroundUpdates()
    while gamepad.isConnected():
        # Wait for the next event
        eventType, control, value = gamepad.getNextEvent()

        # Determine the type
        if eventType == 'BUTTON':
            print(f'{control} {value}')

        elif eventType == 'AXIS':
            print(f'{control} {value}')
    gamepad.disconnect()

if __name__ == "__main__":
    main()