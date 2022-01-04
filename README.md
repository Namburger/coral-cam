# Coral Cam

Coral Cam is a one-stop shop for comparing/benchmarking state-of-the-art coral edgetpu models that deals with images.
Currently, coral cam has the capabilities to deal with 3 problems:

- Classification
- Detection
- Human Pose Estimation

The models that are used in this application are provided [here](https://github.com/google-coral/test_data), more info
are provided @ [coral/models](https://coral.ai/models).

Note:
This application has only been tested with a Debian based Linux machine with
a  [Coral USB Accelerator](https://coral.ai/products/accelerator) attached. It has not been tested on the Coral Dev
Board or a machine with any of the Coral PCIe Accelerators.

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

In addition, please follow this guide to
install [libedgetpu](https://coral.ai/docs/accelerator/get-started/#1-install-the-edge-tpu-runtime) as well as
the [tflite_runtime](https://coral.ai/docs/accelerator/get-started/#2-install-the-pycoral-library):
https://coral.ai/docs/accelerator/get-started/

### Update models:

```
$ git submodule init && git submodule update
```

### Run:

```
$ python3 main.py
```

### Demo:

Here you can see the magic of Coral Cam. Observe how one can easily switch between different inference types, models, as
well as how inference latency increases when the edgetpu is not in use:

<img src="demo.gif" width="777">