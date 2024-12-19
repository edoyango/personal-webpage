---
title: Setting up Orange Pi 5 Pro for YOLOv5
weight: 1
---

# Setting up Orange Pi 5 Pro for YOLOv5

* orange pi 5 pro 8GB with official power supply (25W - 5V, 5A), 4-sheet copper cooler with fan.
* Debian server image with latest kernel https://drive.google.com/drive/folders/1F2uc8v_EQnvsNrevDihwoymOJlFgM-dZ, imaged with balena etcher. No headless setup available.
* Install latest RKNN binaries ([library](https://github.com/airockchip/rknn-toolkit2/tree/master/rknpu2/runtime/Linux/librknn_api/aarch64) to `/usr/lib` and then symlink to `/usr/lib64`, and [exe/scripts](https://github.com/airockchip/rknn-toolkit2/tree/master/rknpu2/runtime/Linux/rknn_server/aarch64/usr/bin) to /usr/local/bin.
* git clone [rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo) and [RockChip's YOLOv5 fork](https://github.com/airockchip/yolov5).
* [rockchip ffmpeg](https://github.com/nyanmisaka/ffmpeg-rockchip)
