# AutoSAT
The code for the RDW Self Driving Challenge 2025 developed by the Saxion team

## Getting bluetooth to work with the controller

[First step](https://wiki.archlinux.org/title/Gamepad#Xbox_Wireless_Controller_/_Xbox_One_Wireless_Controller) is plugging in the controller into a Windows 10/11 pc and downloading the [Xbox Accessories application](https://apps.microsoft.com/store/detail/xbox-accessories/9NBLGGH30XJ3?hl=en-us&gl=us) through the Microsoft Store to update the firmware of the controller.

Next step is installing a driver for it. We will use [xpadneo](https://github.com/atar-axis/xpadneo/). To install it follow the instructions from the repo:
* `git clone https://github.com/atar-axis/xpadneo.git`
* `cd xpadneo`
* `sudo ./install.sh`

Then to connect the controller:
* `sudo bluetoothctl`
* `scan on`
* Hold the pair button
* Find the address of the controller
* `scan off`
* `pair controller_mac` Replace controller_mac with the address you got from scan
* `trust controller_mac`
* `connect controller_mac`
* `exit`

Now everything should be connected. If it doesn't work you need to remove the controller and try again.

To test that everything works install the joystick package `sudo apt-get install joystick` and test the controller `jstest /dev/input/js0`.