# Coral Cam

Coral Cam is a one-stop shop to protoype, compare, and benchmark state-of-the-art tflite/edgetpu models that deals with images using a live feed.
Currently, coral cam has the capabilities to deal with 4 problems:

- Classification
- Detection
- Human Pose Estimation
- Semantic Segmentation

<img src="demo.gif" width="1000">

Here you can see the magic of Coral Cam. Observe how one can easily switch between different inference types, models, as
well as how inference latency increases when the edgetpu is not in use.
The models that are used in this application are provided [here](https://github.com/google-coral/test_data), more info
are provided @ [coral/models](https://coral.ai/models).

Note:
This application has only been tested with a Debian based Linux machine with
a  [Coral USB Accelerator](https://coral.ai/products/accelerator) attached, as well as the original [Coral Dev Board](coral.ai/products/dev-board).
It has not been tested on the Mini Coral Dev Board or any other platforms.

## Getting Started

### Clone Repo along with models:

```
$ git clone --recurse-submodules https://github.com/Namburger/coral-cam.git
```

### Installs Dependencies:

```
$ python3 -m pip install eel --user
$ sudo apt install python3-tk python3-pil python3-pil.imaget
```

On the Dev Board, also install chromium:

```
$ sudo apt install chromium
```

In addition, install all tflite/libedgetpu libraries:

- [libedgetpu](https://coral.ai/docs/accelerator/get-started/#1-install-the-edge-tpu-runtime)

```
$ echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
$ curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
$ sudo apt-get update
$ sudo apt-get install libedgetpu1-std
```

- [tflite_runtime](https://www.tensorflow.org/lite/guide/python)

```
$ echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
$ curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
$ sudo apt-get update
$ sudo apt-get install python3-tflite-runtime
```

More info:
https://coral.ai/docs/accelerator/get-started/

### Update models:

```
$ git submodule init && git submodule update
```

### Run:

```
$ python3 main.py
```

