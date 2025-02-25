from src.gamepad import Gamepad


def main():
    gamepad = Gamepad.XboxONE()
    gamepad.startBackgroundUpdates()
    gamepad.disconnect()

if __name__ == "__main__":
    main()