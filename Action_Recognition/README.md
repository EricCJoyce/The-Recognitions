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

### [R](https://www.r-project.org/)

**Ubuntu 16.04 LTS**:
```
sudo apt install apt-transport-https software-properties-common
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9

sudo add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu xenial-cran35/'
sudo apt update
sudo apt install r-base
```

Check your installation by entering:
```
R --version
```

Now enter the R interpreter:
```
sudo -i R
```

Install multivariate dynamic time-warping:
```
install.packages("dtw")
```

Quit R. (No need to save the workspace image, so enter `n` when asked.)
```
q()
```

Now that you've installed the R backend, install Python 3's rpy2 library like so (obviously with your name in place of mine):
```
sudo /home/eric/.local/bin/pip3.5 install rpy2==2.9.5
```

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

### [Mask-RCNN](https://github.com/matterport/Mask_RCNN) (see also [this version](https://github.com/leekunhee/Mask_RCNN) adapted for TensorFlow version 2)
