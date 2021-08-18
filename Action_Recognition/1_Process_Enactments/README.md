# 1 - Processing Enactments

Enactments must have the expected structure:
```
./EnactmentName
    |
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
    `--- Labels.fvr
    `--- metadata.fvr
```

## Requirements
### Python
### OpenCV
### MatPlotLib
