# 2 - Assembling Enactments

Modules 1 and 4 tell us the facts: what is in each frame. These facts are established either by ground-truth (module 1's use of the color-map and rules file) or by a trained neural network (module 4).

To a degree, an `*.enactment` file is an interpretation of those facts. Objects are seen from an AR/VR user's perspective. The user's field of view is Gaussian-weighted according to the assumption that centrally-framed objects or objects near the user's hands are significant to actions being performed. These weighted signal form the basis for classifying the user's action. The particulars of this Gaussian-weighting can be controlled here. Depending on the tasks you wish to classify, you may prefer a wide and shallow Gaussian or a deep and thin Gaussian to upweight objects significant to your tasks. The hands also have their own Gaussians that apply the same effect: upweight the objects near the hands.

## Inputs

### Enactments as directories with the expected structure:
```
./EnactmentName
    |
    |--- /some_network_some_epoch  <--- Present if you have a trained network and ran module 4
    |--- /GT                       <--- Built by module 1
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

### For each enactment, the file created by module 1...
```
*_props.txt
```

### ...or the files created by module 4:
```
*_<network-name>_detections.txt
```

## Outputs

From each target enactment, this script produces an `*.enactment` file. If you called this script with the `-render` flag, then a Gaussian-weighted video `*_Gaussian-weighted.avi` of each enactment will be produced, too. These can be informative illustrations of how the Gaussian parameters affect object presence, but they are time-consuming to create.

## Requirements
- Python
- MatPlotLib
- NumPy
- OpenCV
