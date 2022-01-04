# Coral Cam

Coral Cam is a one-stop shop for comparing/benchmarking state-of-the-art coral edgetpu models that deals with images.

Currently, coral cam has the capabilities to deal with 3 problems:

- Classification
- Detections
- Pose Estimation

The models that are used in this application are provided [here](https://github.com/google-coral/test_data), more info
are provided @ [coral/models](https://coral.ai/models).

## Getting Started

# Clone Repo along with models:

```
$ git clone --recurse-submodules https://github.com/Namburger/coral-cam.git
```

# Installs Dependencies:

```
$ python3 -m pip install eel --user
$ sudo apt install python3-tk python3-pil python3-pil.imaget
```

# Update models:

```
$ git submodule init && git submodule update
```

# Run:

```
$ python3 main.py
```
