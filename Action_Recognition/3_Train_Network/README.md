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

## 3.1 - Prepare training and validation sets

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

## 3.4 - Train

## Requirements
- Python
- NumPy
- OpenCV
- TensorFlow/Keras
