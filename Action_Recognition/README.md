# Action Recognition

## Build a first-person action classification system from VR enactments

This repository contains the modules of a pipeline. The primordial inputs are labeled recordings of actions performed in an environment. The primary output is a real-time action classification system, though there are several intermediate products and by-products on the way to developing this classifier.

## Roadmap

```
[1]                   1. Process enactments: establish ground-truth from recordings.
 |
 |----->[3]           2. Assemble enactments: create *.enactment files.
 |       |            3. Train a deep network to recognize objects.
 V       V            4. Use a trained network to detect objects in recordings.
[2]<----[4]
 |
 |
 |----->[5]           5. Use *.enactment files to atemporally match snippets.
 |       |
 |       V
 |<------+
 |
 |----->[6]           6. Build a database from *.enactment files.
 |       |
 |       V
 |<------+
 |
 |----->[7]           7. Compute probabilities using isotonic regression.
 |       |
 |       V
 |<------+
 |
 |----->[8]           8. Derive cutoff conditions to improve classification.
 |       |
 |       V
 |<------+
 |
 |----->[9]           9. Build and train a multi-layer perceptron (MLP) to aid classifications.
 |       |
 |       V
 |<------+
 |
 V
[10]                  10. Simulate real-time classification.
```

## Requirements

### [Python](https://www.python.org/)
We recommend Python 3.6, though later versions may also be compatible.

**Ubuntu 16.04 LTS**:
```
sudo apt-get update
sudo apt-get upgrade

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.6
sudo apt install python3.6-dev
sudo apt install python3.6-venv
wget https://bootstrap.pypa.io/get-pip.py
sudo python3.6 get-pip.py
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
pip3 install pandas
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
pip install pandas
pip install metric-learn
pip install matplotlib
pip install scikit-image
pip install ipython
pip install cython
pip install h5py
pip install opencv-python
```

### Additional Tools

**Ubuntu 16.04 LTS**:
The development tools for Python should have already been handled by the call above, `sudo apt install python3.6-dev`.

Additionally, you will need the following tools, if your system doesn't have them already:
```
sudo apt-get install -y build-essential cmake gfortran git pkg-config
sudo apt-get install -y software-properties-common wget vim
sudo apt-get autoremove
```

### [TensorFlow](https://www.tensorflow.org/)

We recommend TensorFlow for the GPU, version 2.6.

**Ubuntu 16.04 LTS**:
```
pip3 install tensorflow-gpu==2.6
```

**Windows**:
```
pip install tensorflow-gpu==2.6
```

Check your installation. Launch Python 3.6 and enter the following:
```
import tensorflow as tf
```
You want to see nothing but the interpreter prompt. If, instead, you see something like this:
```
/home/your-name-here/.local/lib/python3.6/site-packages/requests/__init__.py:104: RequestsDependencyWarning: urllib3 (1.26.7) or chardet (2.3.0)/charset_normalizer (2.0.7) doesn't match a supported version!
  RequestsDependencyWarning)
```
then quit Python and issue the following commands:
```
sudo pip3 uninstall urllib3
sudo pip3 install urllib3==1.22

sudo pip3 uninstall chardet
sudo pip3 install chardet==3.0.2
```
### Dynamic Time Warping

Included in the `classifier` module of this repository is a Python binding for an implementation of Dynamic Time Warping (DTW) written in C. DTW matches temporal snippets, allowing that instances of the same action may take more or less time to perform.

**Ubuntu 16.04 LTS**:
You will need this library to compile the C code in `DTWmodule.c` into a Python library. Do this by running
```
python3.6 setup.py build
```
This creates a folder named `build`. Inside, find a file named `DTW.cpython-36m-x86_64-linux-gnu.so` or something named appropriately, given your system specs. Copy this `*.so` file into the `classifier` directory. Then, you can throw away the `build` directory.

### Done for now.

This gives you the boilerplate; additional requirements will be explained as each module requires them.

## Data

Available upon request.
