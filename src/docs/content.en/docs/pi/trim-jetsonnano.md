---
title: Reducing Jetson Nano OS for Server
weight: 1
---

# Reducing Jetson Nano OS for Server

The Nvidia Jetson Nano is a Single Board Computer (SBC) with a scaled-down Nvidia GPU (Tegra X1). I have the 2GB version, the smallest available. The OS that Nvidia forces you to use comes with a full-blown desktop environment, which chews through the 2GB of RAM pretty easily - not leaving as much room as I'd like for other things.

Consequently, this page is to document the steps to trim down the OS to save disk space and RAM - adding to steps documented elsewhere.

After installing the OS, hooking up periferals, inserting the flashed SD card, and setting up the Jetson, you can run the following:
```bash {style=tango,linenos=false}
sudo systemctl stop gdm3
sudo systemctl disable gdm3
sudo sytemctl set-default multi-user.target
sudo reboot
sudo apt remove --purge -y \
	ubuntu-desktop \            # meta-package for Ubuntu desktop
	gnome* \                    # packages to create desktop GUI
	libreoffice* \              # Microsoft office equivalent
	nautilus \                  # GUI file exporer
	thunderbird \               # GUI email app
	chromium-browser            # crap browser
sudo apt autoremove -y          # remove packages no longer needed
sudo apt autoclean              # removes cached package files
sudo apt update                 
sudo apt install \
	ubuntu-server \             # meta-package for Ubuntu server
	firefox                     # a better browser
sudo apt upgrade -y
```