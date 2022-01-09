# 3 - Training a deep network to recognize objects

The ground-truth masks created by module 1 can be used to train a deep network to recognize objects. We will take advantage of a pre-trained version of [MobileNet](https://arxiv.org/abs/1704.04861), transferring the learning already achieved in its lower layers to our new task.

This module involves several steps. You will need to:
- Prepare training and validation sets, while setting aside an untouched test set.
- Prepare a network-training workspace.
- Set up the TensorFlow Object Detection API.
- Train the network.

You have the opportunity to influence our future classifier's behavior by applying certain parameters during network training. During development of this system, we identified intermediate states for objects with hinges: doors, cabinets, and machinery that could be in an opened or closed state also had an intermediate "ajar" state. We later decided that the network did not need to be taught what "ajar" was, and that it would have to decide per frame whether a recognizable object in view were opened or closed. You can drop classes from the training set via the command line when calling `train_maskrcnn.py`.

## Inputs

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

## Outputs

Once training begins, this script will save all epochs with improved (lower) validation loss.

### 

## Requirements
- Python
- NumPy
- OpenCV
- TensorFlow/Keras
