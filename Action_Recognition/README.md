# Action Recognition

## Build a first-person action classification system from VR enactments

This repository contains the modules of a pipeline. The primordial inputs are labeled recordings of actions performed in a virtual environment. The primary output is a real-time action classification system, though there are several intermediate products and by-products on the way to developing this classifier.

## Roadmap

```
[1]
 |
 |----->[3]
 |       |
 V       V
[2]<----[4]
 |
 |
 |----->[5]------+
 |               |
 |               V
 |-------+-------+
 |       |       |
 V       V       V
[6]     [8]     [7]
 |       |       |
 +-------+-------+
         |
         V
        [9]
```

## Requirements

### [Python](https://www.python.org/)
We recommend Python 3.5.2, though later versions may also be compatible.

**Ubuntu 16.04 LTS**:
```
sudo apt-get update
sudo apt-get upgrade

sudo apt-get install -y build-essential cmake gfortran git pkg-config
sudo apt-get install -y python-dev software-properties-common wget vim
sudo apt-get autoremove

sudo apt-get install python3-pip
pip3 install --upgrade pip
sudo apt install curl
curl -fsSL -o- https://bootstrap.pypa.io/pip/3.5/get-pip.py | python3.5
pip3 install --upgrade setuptools
```

**Windows**:

Download the desired version from Python's website and double-click to run the installer.

Be sure to tick the box "Add to PATH" at the bottom of the installer window.

Test your installation by calling the Python interpreter from your command line. In Windows, use `Ctrl-Z` then `Enter` to quit the Python interpreter.

Finally, make sure that Pip exists. Call the following from your command line:
```
pip -V
```

### [OpenCV](https://opencv.org/) (and other tools)

**Ubuntu 16.04 LTS**:

```
pip3 install numpy scipy scikit-learn
pip3 install metric-learn
pip3 install matplotlib
pip3 install scikit-image
pip3 install ipython
pip3 install cython
pip3 install h5py
pip3 install opencv-python
```

**Windows**:

```
pip install numpy scipy scikit-learn
pip install metric-learn
pip install matplotlib
pip install scikit-image
pip install ipython
pip install cython
pip install h5py
pip install opencv-python
```

### Python Development Tools

**Ubuntu 16.04 LTS**:
The development tools for Python should have already been handled by the call above that included
```
sudo apt-get install python-dev
```

You will need this library to compile the C code in `DTWmodule.c` into a Python library. Do this by running
```
python3.5 setup.py build
```

This creates a folder named `build`. Inside, find a file named `DTW.cpython-35m-x86_64-linux-gnu.so` or something appropriately named, given your system specs. Copy this `*.so` file into the same directory as your other Recognition scripts.

**Windows**:

???

### [TensorFlow](https://www.tensorflow.org/)

We recommend TensorFlow for the GPU, version 1.14. If you work with TensorFlow version 2, then please see below for [leekunhee](https://github.com/leekunhee)'s version of the Mask-RCNN library updated for version 2.

**Ubuntu 16.04 LTS**:
```
pip3 install tensorflow-gpu==1.14
```

**Windows**:
```
pip install tensorflow-gpu==1.14
```

### [Keras](https://keras.io/)

We recommend pairing Keras 2.2.5 with TensorFlow-GPU 1.14

**Ubuntu 16.04 LTS**:
```
pip3 install keras==2.2.5
```

**Windows**:
```
pip install keras==2.2.5
```

### [Mask-RCNN](https://github.com/matterport/Mask_RCNN)
