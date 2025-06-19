# AutoSAT
The code for the RDW Self Driving Challenge 2025 developed by the Saxion team.

## How to Use the Program

### Environment Setup

First, install [pyenv](https://github.com/pyenv/pyenv) and [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) by following the instructions for your OS.

Try running ``pyenv doctor`` to see if the installation was successful.

Then create the python environment:

```bash
pyenv install 3.12
pyenv virtualenv 3.12 autosat-env
```
And activate it with ``pyenv activate autosat-env``.

Then install the requirements with ``pip install -r requirements.txt``.

Finally start the program with ``python -m src.main`` and navigate to http://0.0.0.0:8000 or http://device-ip:8000.

## Remote Working Tools

For working on the vehicle remotely:

### Tailscale
- **What it does**: Creates a secure network between your laptop and the vehicle computer
- **Why it's useful**: You don't need to be on the same Wi-Fi network
- **Quality of life**: Really useful if you change connections often because you don't need to search for the IP of the computer
- **Bonus**: You can also send files over Tailscale using Taildrop
- **How to use**: Install on both devices, connect to your tailnet, then use Tailscale IPs to access the web interface

### NoMachine
- **What it does**: Lets you control the vehicle computer like you're sitting in front of it
- **Why it's useful**: Full desktop access - keyboard, mouse, screen sharing. Much more responsive than X11 forwarding through SSH
- **Note**: Optional, but helpful for full remote control
- **How to use**: Install on the vehicle computer and your laptop, then connect remotely
- **Headless fix**: If the device doesn't have a monitor plugged in and NoMachine won't connect, try:
  ```bash
  sudo systemctl stop display-manager
  sudo /etc/NX/nxserver --restart
  ```
This fixes an [old bug](https://forum.nomachine.com/topic/connection-fails-on-headless-client#post-21783) with headless systems.

### Why use both?
Tailscale gets you connected from anywhere, NoMachine gives you full control. Together you can work on the vehicle from your couch, debug issues without being physically there, and access the AutoSAT interface whether you're local or remote.

## How to Use the Program

### Environment Setup

First, install [pyenv](https://github.com/pyenv/pyenv) and [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) by following the instructions for your OS.

Try running ``pyenv doctor`` to see if the installation was successful.

Then create the python environment:

```bash
pyenv install 3.12
pyenv virtualenv 3.12 autosat-env
pyenv activate autosat-env
```

Then install the requirements with ``pip install -r requirements.txt``

Finally start the program with ``python -m src.main`` and navigate to http://0.0.0.0:8000 or http://device-ip:8000.

## Remote Working Tools

For working on the vehicle remotely:

### Tailscale
- **What it does**: Creates a secure network between your laptop and the vehicle computer
- **Why it's useful**: You don't need to be on the same Wi-Fi network
- **Quality of life**: Really useful if you change connections often because you don't need to search for the IP of the computer
- **Bonus**: You can also send files over Tailscale using Taildrop
- **How to use**: Install on both devices, connect to your tailnet, then use Tailscale IPs to access the web interface

### NoMachine
- **What it does**: Lets you control the vehicle computer like you're sitting in front of it
- **Why it's useful**: Full desktop access - keyboard, mouse, screen sharing. Much more responsive than X11 forwarding through SSH
- **Note**: Optional, but helpful for full remote control
- **How to use**: Install on the vehicle computer and your laptop, then connect remotely
- **Headless fix**: If the device doesn't have a monitor plugged in and NoMachine won't connect, try:
  ```bash
  sudo systemctl stop display-manager
  sudo /etc/NX/nxserver --restart
  ```
This fixes an [old bug](https://forum.nomachine.com/topic/connection-fails-on-headless-client#post-21783) with headless systems.

### Why use both?
Tailscale gets you connected from anywhere, NoMachine gives you full control. Together you can work on the vehicle from your couch, debug issues without being physically there, and access the AutoSAT interface whether you're local or remote.

## Getting bluetooth to work with the controller

[First step](https://wiki.archlinux.org/title/Gamepad#Xbox_Wireless_Controller_/_Xbox_One_Wireless_Controller) is plugging in the controller into a Windows 10/11 pc and downloading the [Xbox Accessories application](https://apps.microsoft.com/store/detail/xbox-accessories/9NBLGGH30XJ3?hl=en-us&gl=us) through the Microsoft Store to update the firmware of the controller.

The next step is installing a driver for it. We will use [xpadneo](https://github.com/atar-axis/xpadneo/). To install it, follow the instructions from the repo:
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

Now everything should be connected. If it doesn't work, you need to remove the controller and try again.

To test that everything works install the joystick package `sudo apt-get install joystick` and test the controller `jstest /dev/input/js0`.

In case rumble breaks again, this one file needs to be modified like in [here](https://github.com/atar-axis/xpadneo/commit/4a3a623b5facca8184e9070317fea03adc3a9e8f), but keep an eye out for any issues on the main page of [xpadneo](https://github.com/atar-axis/xpadneo/).

### Set permissions for current user

To get access to the camera feeds, you need to add yourself to the video group.

Run ``sudo usermod -a -G video <username>`` and replace username with your username.

To get access to the controller, you need to add yourself to the input group.

Run ``sudo usermod -a -G input <username>`` and replace username with your username.

To manage network interfaces, you need to add yourself to the netdev group.

Run ``sudo usermod -a -G netdev <username>`` and replace username with your username.

Log out and log back in, and the permissions will be set.