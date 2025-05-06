o# AutoSAT
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

In case rumble breaks again this one file needs to modified like in [here](https://github.com/atar-axis/xpadneo/commit/4a3a623b5facca8184e9070317fea03adc3a9e8f), but keep an eye out for any issues on the main page of [xpadneo](https://github.com/atar-axis/xpadneo/).

### Set permissions for current user

To get access to the camera feeds you need to add yourself to the video group.

Run ``sudo usermod -a -G video <username>`` and replace username with your username.

To get access to the controller you need to add yourself to the input group.

Run ``sudo usermod -a -G input <username>`` and replace username with your username.

To manage network interfaces you need to add yourself to the netdev group.

Run ``sudo usermod -a -G netdev <username>`` and replace username with your username.

Log out and log back in and the permissions will be set.