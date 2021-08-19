# 3 - Training a deep network to recognize objects

The ground-truth masks created by module 1 can be used to train a deep network to recognize objects. We will train an instance of Mask-RCNN.

You have the opportunity to influence our future classifier's behavior by applying certain parameters during network training. During development of this system, we identified intermediate states for objects with hinges: doors, cabinets, and machinery that could be in an opened or closed state also had an intermediate "ajar" state. We later decided that the network did not need to be taught what "ajar" was, and that it would have to decide per frame whether a recognizable object in view were opened or closed. You can drop classes from the training set via the command line when calling `train_maskrcnn.py`.

## Inputs

### Enactments as directories with the expected structure:
```
./EnactmentName
    |
    |--- /gt                <--- Built by module 1
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
- OpenCV
- TensorFlow
- Keras
