# 4 - Detecting objects using a trained network

This is the deep-learning version of module 1. Rather than use color-maps and object-combination rules to establish which objects are visible in each frame, we use a deep network. The products of this module are similar to those of module 1. Instead of writing a `*_groundtruth.txt` file per enactment, we write a `*_<network-name>_detections.txt` file named after the network that made the predictions.

## Inputs

### Enactments as directories with the expected structure:
```
./EnactmentName
    |
    |--- /gt           <--- Built by module 1, but not necessary for module 4
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

```
./EnactmentName
    |
    |--- /gt
    |--- /<network-name>  <--- This folder will be created and filled with detection masks
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

The created file `*_<network-name>_detections.txt` provides indices into the directory `EnactmentName/<network-name>`.

## Requirements
- Python
- OpenCV
- TensorFlow
- Keras
- Mask-RCNN
