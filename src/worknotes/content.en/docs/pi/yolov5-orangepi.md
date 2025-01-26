---
title: Combining the Orange Pi 5 Pro and YOLOv5 for bird detection
weight: 1
---

# Combining the Orange Pi 5 Pro and YOLOv5 for bird detection

I started this project to record and identify the birds that we saw on our verandah throughout the day, as well as an
excuse to learn about object detection. I aimed at using YOLO mainly because the `ultralytics` package is very easy to
use and the performance-accuracy tradeoff was good. 

When I first got started, it was quite easy to play with `ultralytics` and its latest iterations of YOLO on my laptop,
but my main constraint was that I didn't have any machines that were a good (and affordable) candidate to process the
videos. This was mainly because inference on GPUs was fairly resource intensive and we needed the GPUs for other daily
resource-intensive use cases like gaming, work, or image/video editing. 

So at first, I was recording webcam videos of our verandah, uploading them to google drive, downloading them to the HPC
I had at work, and then processing them using the under-utilized GPUs on the HPC. This worked pretty well - except that
I changed job, and no longer have access to free and available GPU compute. So what was I to do!

I knew that the [RK1](https://turingpi.com/product/turing-rk1) chips existed which used the RK3588 chip, which had an
NPU built into it. From searching the web, I discovered that the RockChip NPU supported many object detection 
frameworks - including all the recent YOLO frameworks (see the 
[RKNN model zoo](https://github.com/airockchip/rknn_model_zoo))! My original expectation was that I would buy three RK1
modules - two to perform object detection and the third for any peripheral needs like work management, post-processing,
file upload/download etc. So, **The Orange pi 5 Pro was originally intended for testing before I got multiple RK1 
modules, but turns out the lone SBC was enough!** This turned out to be because the NPU has 3 cores, so I can
perform inference on multiple video feeds, and the hardware accelerated FFMPEG encoding is very fast!

Here's an example GIF of the output (compressed in a few ways):

![Example output of inference output](/worknotes/imgs/sample.gif)

Here I've documented the setup steps for future reference.

## Requirements

The core objective was to **perform real-time object detection on two webcam video feeds**. In a little more detail, 
this meant I needed the setup to be able to:

1. Perform inference on the video feeds at a resolution of 704x396 @ 10fps using a medium sized YOLO model
2. save and encode/compress bird-containing segments of the video feed

704x396 resolution and medium model were chosen by experimenting and finding a balance between speed and accuracy. The 
cameras could record at over 30 FPS, but 10 FPS was chosen as it reduced storage space significantly (videos were being 
kept for future model training).

On an NVIDIA A30 GPU, the object-detection alone usually took roughly 2h to process 20h of video feed (2 webcams 
recording for 10h).

## Hardware

### webcams

I used two of the same [2MP webcam with zoom](https://www.amazon.com.au/dp/B0B6QV5SVW) so I could get a good view of the 
platforms we had setup on the verandah for the birds. Using regular non-zoom webcams (e.g. what you would use for 
meetings), meant the birds were way too small on the feed and it was challenging for the model, as well as to label 
images. Sometimes the birds were only a handful of pixels!

### Orange Pi 5 Pro

I chose the Orange Pi 5 Pro mainly because:

* I didn't need many peripherals as it was acting as a server (just needed USB ports)
* Was cheaper than the Ultra/Max
* Had faster memory than the normal Orange Pi 5

I got the 8GB model as I guesstimated I needed roughly 4.5GB for inference. I arrived at this number by adding together 
the CPU RAM and GPU RAM usage while doing inference with the ultralytics package.

I bought the Orange Pi 5 Pro itself on Taobao while I was in China, which had roughly the same sticker price as if I had 
bought it off Aliexpress from Australia, but I wouldn't have to pay shipping fees or import taxes. If you're buying it 
outside of China, Aliexpress prices seemed to be much better than Amazon.

I bought the Orange Pi 5 Pro with the charger as it required a minimum of 5A and 5V which wasn't that easy to find 
cheaply.

To keep the Orange Pi cool, I bought a [copper plate cooler with fan (4-fans)](https://www.aliexpress.com/item/1005001866650684.html) 
as it didn't take much for the Pi to overheat and throttle itself.

### Storage

I'm using a 32GB SD Card that I had from previous Raspberry Pi projects and the SD card stores most of the root 
partition. I also repurposed an old external hard drive to store the recorded video data.

### A note about training

Training cannot be done on the NPU, so you'll need to have seperate hardware to perform the model training.

## Software Setup

### Setup

To start off, I used [Balena Etcher](https://etcher.balena.io/) to flash the SD card with the [official server image](https://drive.google.com/drive/folders/1KnmBQ3Z0M_5snRC24LjhKb8_tKbcfOkw).
Unfortunately, unlike the Raspberry Pi, I couldn't find a way to pre-configure the WiFi headless, so I had to do it with 
the screen and keyboard plugged in. Note that the default user and password is `orangepi`. The image does come with the 
`orangepi-config` utility which makes connecting to the WiFi *a little* easier.

Once your internet is setup and you've confirmed you can 

### Update drivers

Once the Pi was setup and running, I needed to update the RKNN library by downloading the latest binary (which for some 
reason, is stored on [their GitHub](https://github.com/airockchip/rknn-toolkit2/blob/master/rknpu2/runtime/Linux/librknn_api/aarch64/)).

```bash {style=tango,linenos=false}
# clear out old drivers
sudo find /usr -name librknrrt.so -delete

# download the library directly to /usr/lib
sudo wget https://raw.githubusercontent.com/airockchip/rknn-toolkit2/refs/heads/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so -O /usr/lib/librknrrt.so

# the RKNN toolkit pip package (used later) expects it to be in /usr/lib64
sudo ln -s /usr/lib/librknrrt.so /usr/lib64/librknrrt.so
```

### Optional: Install FFMPEG

I use FFMPEG to perform the video encoding and compression needed in my pipeline. If you want to use FFMPEG for encoding
using hardware acceleration, you'll need to build one from source. The instructions [here](https://github.com/nyanmisaka/ffmpeg-rockchip/wiki/Compilation)
should be sufficient to get you going.

### Test examples

To start, it's a good idea to create a virtual environment to store the required dependencies to run the Python code.

```bash {style=tango,linenos=false}
cd ~
python -m venv rknn-venv
. rknn-venv/bin/activate
pip install rknn-toolkit2
```

Then, we'll need the repo

```bash {style=tango,linenos=false}
git clone https://github.com/airockchip/rknn_model_zoo.git --depth 1
```

And now we can run the examples.

```bash {style=tango,linenos=false}
# move to the YOLOv5 example directory
cd rknn_model_zoo/examples/yolov5/python

# convert the model
python convert.py ../model/yolov5s_relu.onnx rk3588 i8 yolov5s_relu.rknn

# run the example with the converted model - this saves the image into ./result
python yolov5.py --model_path yolov5s_relu.rknn --img_save
```

If the above doesn't work, it could be that you didn't update the drivers properly, or you didn't install the 
RKNN-toolkit into your environment.

### Training a new model

Training a new model can be done by using [RKNN's customized fork of YOLOv5](https://github.com/airockchip/yolov5). I 
personally use Roboflow to label my images. If using Roboflow, make sure that the directory structure and annotations 
are compatible with the YOLOv5 format. In Roboflow, this is an option that you can choose when downloading your dataset. 
Note you cannot practically do training on the NPU, so as mentioned, you'll need seperate hardware to perform training.

The instructions below you have an NVIDIA or AMD GPU installed in the appropriate manner for your hardware.

```bash {style=tango,linenos=false}
# Get the customized YOLOv5 repository
git clone https://github.com/airockchip/yolov5.git
cd yolov5

# Install older PyTorch to stop PyTorch complaining about some things in the code
pip install 'torch<2' 'torchvision<0.15'

# Install remaining dependencies
pip install -r requirements.txt

# Train your model (assuming your dataset is my-dataset)
python train.py --weights yolov5m.pt --epochs 120 --batch 0 --data my-dataset/data.yaml
```

After training, you will need to export the model as onnx, and then convert the model to RKNN format.

```python {style=tango,linenos=false}
# Export first to ONNX
python export.py --rknpu --weight path/to/weights.pt

# go back to example python folder to use convert.py
cd ../rknn_model_zoo/examples/yolov5/python # replace with where you saved the examples
python convert.py /path/to/weights.onnx rk3588 i8 /path/to/weights.rknn
```

Note that, for some reason, the training will always try to find better anchors and perform training assuming those
anchors. You'll need to save the anchors in a txt file to be used with the example script. The steps above will print
out the anchors used in the model. Save those in `anchors.txt` in the examples folder (or wherever else that is
convenient).

### Customized inference

The example `yolov5.py` inference script is very basic and unfortunately, not written in a modular way. Hence, 
the script will require customization if you wish to repurpose it for your use case. In most cases, the `CLASSES` and 
`coco_id_list` variables will need to be changed to match your data e.g. from:

```python {style=tango,linenos=false}
CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")

coco_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
```

To:

```python {style=tango,linenos=false}
CLASSES = ("dog", "cat", "mouse") # replace this with your dataset's classes

coco_id_list = [1, 2, 3]
```

Note that the ID list starts from 1. It's not a big deal if you get the names wrong, it will just result in incorrect 
labels being written on your detected images. The script is setup to use images only, so if you want to use it on 
another media format e.g. video, you will have to customize the logic yourself.

## Other notes

### Model choice

You might be looking at the [model support table](https://github.com/airockchip/rknn_model_zoo/tree/main?tab=readme-ov-file#model-support)
and the [model performance benchmark table](https://github.com/airockchip/rknn_model_zoo/tree/main?tab=readme-ov-file#model-performance-benchmarkfps),
which show that all the newer YOLO models have usable performance (e.g., they say that yolo11m has 12.7 FPS). But, this 
only accounts for the time spent on the NPU, but doesn't include pre- and post-processing which is done on the CPU. 
Besides some image processing, post-processing also includes steps that couldn't run on the NPU, such as using the 
softmax function. **The softmax function is used by YOLOv6 and above and is the bottleneck for using these models on the 
RK3588.** So while the performance table says that `yolo11s` has 33 FPS, including the pre- and post-processing, I was 
getting around 10FPS. And note that the PyTorch softmax function already uses multiple threads, so it is challenging to 
accelerate.

### Other improvements to performance

There are two performance improvements that aren't well documented, that may be good for your use case: batch inference, 
and  multi-core inference.

#### Batch inference

When using the `convert.py` script in the example, the model only accepts singular inputs (i.e., batch size of 1). You 
can enable batches by updating the [`convert.py` script](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolov5/python/convert.py#L60).
For example:

```python {style=tango,linenos=false}
# old
ret = rknn.build(do_quantization=do_quant, dataset=DATASET_PATH)

# new
ret = rknn.build(do_quantization=do_quant, dataset=DATASET_PATH, rknn_batch_size=16)
```

This will require updating the `yolov5.py` example script too, if you wish for that to work. The [main loop](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolov5/python/yolov5.py#L236)
needs to be changed so that batches are passed to the pre-processing step, the model, and the post-processing step.

**I found that using a batch size of 16 increased throughput of inference (EXCLUDING pre- and post-processing) by 
roughly 2.5x.** While this was significant, I didn't find it necessary, as I found that `yolov5m` with `imgsz=704` was 
sufficient was achieving a performance that was satisfactory for my use case.

#### Multi-core inference

As alluded to in the `rknn_model_zoo` performance table, the RK3588 has multiple cores. By default, inference uses only
one of thoses cores, but you can use all the cores if necessary. This requires changing the [utility code in the model zoo that initializes the model](https://github.com/airockchip/rknn_model_zoo/blob/main/py_utils/rknn_executor.py#L12).

```python {style=tango,linenos=false}
# from
ret = rknn.init_runtime(target=target, device_id=device_id)

# to
ret = rknn.init_runtime(target=target, device_id=device_id, rknn.NPU_CORE_0_1_2)
```

I haven't tested this as I am performing inference on two video feeds at the same time, but there seems to be people out 
there using it ([example](https://github.com/leafqycc/rknn-multi-threaded/blob/nosigmoid/rknnpool.py)).

### Docker

To build a container that can use the NPU, the only thing that must be done is installing the driver ([example](https://github.com/edoyango/birds/blob/main/Dockerfile)). 
If hardware accelerated FFMPEG is needed, that needs to be built inside the container too.

To use the container, the container needs to be run with `--privileged` (or if using Docker Compose, 
`priveleged: true`).

### NPU monitoring

The load on the NPU can be viewed through

```bash {style=tango,linenos=false}
sudo cat /sys/kernel/debug/rknpu/load
```
``` {style=tango,linenos=false}
NPU load:  Core0: 53%, Core1: 40%, Core2:  0%,
```

which could be combined with `watch` for a dynamic feed.

## Future work

* I could potentially run larger models (or use higher resolution input) by leveraging the acceleration notes above.
    * batch processing would require that I change the main loop. It could also limit my applications e.g. in case I
    want to stream my feed(s), how frames are read/processed/written need to be thought about to ensure a smooth feed.
    * multi-core inference may handicap my ability to perform inference on two feeds at once
* Use an NVME drive instead of an external HDD. This would be more for form factor than performance, as IO isn't a
bottleneck.
