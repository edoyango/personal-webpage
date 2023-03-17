---
title: CUDA + NVHPC on WSL
weight: 1
---

# Installing CUDA + NVHPC on WSL

This page describes the steps to setup CUDA and NVHPC within the WSL2 container (Windows 10) - avoiding the need for dual-boot or a separate Linux PC. Note that WSL2 must not have been installed when beginning these steps.


1. Install the latest Windows CUDA graphics driver
2. Install WSL2
    * open PowerShell as administrator
    * Make sure to update WSL kernel to latest version wsl --update
        * if accidentally rolled back, follow the instructions here
        * then wsl --update again
    * check which Linux flavours are available with wsl --list --online
    * install the desired flavour by wsl --install -d <flavour name>
    * start WSL with wsl, or opening the WSL application from the Windows search bar
3. `sudo apt update && sudo apt upgrade -y`
4. Close and restart WSL
5. `sudo apt update && sudo apt upgrade -y && sudo apt autoremove -y`
6. Install CUDA for WSL
    * check which CUDA version is compatible with the desired version NVHPC kit here
    * select the correct CUDA version here
    * Select the right setup: Linux -> x86_64 -> WSL-Ubuntu -> 2.0 -> dev (local)
    * before beginning install, delete old GPG key sudo apt-key del 7fa2af80
    * Perform the install with code below:
```bash {style=tango}
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda-repo-wsl-ubuntu-11-7-local_11.7.1-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-7-local_11.7.1-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```
7. Install NVHPC
    * Download the desired NVHPC kit
    * `tar xzvf nvhpc*.tar.gz`
    * `sudo nvhpc*/install`

# Uninstalling NVHPC and CUDA

```bash {style=tango}
sudo rm /opt/nvidia/hpc_sdk
sudo apt-get purge -y cuda && sudo apt-get autoremove -y
sudo rm /usr/share/keyrings/cuda-*-keyring.gpg
sudo dpdkg -P cuda-repo-wsl-ubuntu-*-*-local_*amd64.deb
```

You can then perform the installation steps again for the desired NVHPC-CUDA combination