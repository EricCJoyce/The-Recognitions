# 3 - Training a deep network to recognize objects

The ground-truth masks created by module 1 can be used to train a deep network to recognize objects. We will take advantage of a pre-trained version of [MobileNet](https://arxiv.org/abs/1704.04861), transferring the learning already achieved in its lower layers to our new task.

This module involves several steps. You will need to:
1. Prepare training and validation sets, while setting aside an untouched test set.
2. Prepare a network-training workspace.
3. Set up the TensorFlow Object Detection API.
4. Train the network.

Before starting on any of these tasks, you may need to clean up your enactments, removing color-map artifacts and objects that "leak" through gaps in the environment mesh.

## Assumptions

### Each enactment is expected to be a directory with the following structure:
```
./EnactmentName
    |
    |--- /GT                <--- Built by module 1
    |--- /Props
    |--- /Subprops
    |--- /Users
    |      |
    |      `--- /UserName
    |              |
    |              |--- Head.fvr
    |              |--- LeftHand.fvr
    |              |--- RightHand.fvr
    |              `--- /POV
    |                     |
    |                     |--- CameraIntrinsics.fvr
    |                     |--- SubpropColorMap.fvr
    |                     |--- /ColorMapCameraFrames
    |                     |--- /DepthMapCameraFrames
    |                     `--- /NormalViewCameraFrames
    |--- Labels.fvr
    `--- metadata.fvr
```
### Your working directory contains your enactments.

Each enactment will have both a directory described above and a corresponding `*_props.txt` file:
```
./MyWorkingDirectory
    |
    |--- /Enactment1
    |--- /Enactment2
   ...
    |--- /EnactmentN
    |--- Enactment1_props.txt
    |--- Enactment2_props.txt
   ...
    |
    `--- EnactmentN_props.txt
```

## 3.0 - Enactment clean-up (if applicable or desirable)

### edit_mask_artifacts.py

Use an interactive sub-program to clean up the mask artifacts. You must specify an enactment and an instance within that enactment. This script will then iterate over all masks and frames containing this instance and allow you to inspect/edit mask artifacts for that instance.

All changes are saved to an `*.editlist` file.

### refine-dataset/Makefile

This make-file builds the sub-programs called by the script above.

## 3.1 - Prepare training and validation sets

### build_object_recog_dataset.py

Build a dataset from enactments in preparation for training an object-detection network. Several objects typically appear in a single video frame. This script attempts to balance training and validation sets according to the given parameters, but must allocate frames entirely to one set or the other. We do not want to incur false false-positives!

Outputs `training-set.txt` and `validation-set.txt`

### convert_object_recog_dataset.py

Convert the training and validation set files to formats expected by the TensorFlow Object-Detection API. This means creating two folders in the current working directory: `./training/images/train` and `./training/images/test`. This script will also generate `./training/annotations/label_map.pbtxt`. This script expects to find `./training-set.txt` and `./validation-set.txt`. It also expects to find the enactments referenced by these documents. This script will copy all images into the respective folders and generate one annotation XML per image. A single XML contains all detections in its frame.

## 3.2 - Prepare training workspace
```
./MyWorkingDirectory
    |
    |--- /Enactment1
    |--- /Enactment2
   ...
    |--- /EnactmentN
    |--- Enactment1_props.txt
    |--- Enactment2_props.txt
   ...
    |
    |--- EnactmentN_props.txt
   ...
    |--- /training
    |      |
    |      |--- /annotations
    |      |--- /exported-models
    |      |--- /images
    |      |--- /models
    |      `--- /pre-trained-models
    |
    `--- EnactmentN_props.txt
```

## 3.3 - Set up the TensorFlow Object Detection API

Make sure that you have `git` and `protoc` installed:
```
git --version
protoc --version
```

`git` fetches repositories from GitHub, and `protoc` is the Protocol Buffer Compiler. Protocol buffers, or "protobufs" are a binary file format developed by Google and used by the TensorFlow TFRecord type. We will need to compile our training and validation sets into this format so that the TFOD can read them.

Make the following call to clone the TensorFlow Object-Detection API repository.
```
git clone https://github.com/tensorflow/models.git
```

Now, from your current working directory, make the following calls:
```
cd models/research
protoc object_detection/protos/*.proto --python_out=.
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools ../../models/research
cd ../..
cp object_detection/packages/tf2/setup.py .
python3.6 -m pip install .
```

You may receive some errors, which you can probably ignore. The real test of whether this API installation succeeded is the following:
```
python3.6 object_detection/builders/model_builder_tf2_test.py
```

## 3.4 - Train

## Requirements
- Python
- NumPy
- OpenCV
- TensorFlow/Keras
