# 2 - Assembling Enactments

Modules 1 and 4 tell us the facts: what is in each frame. These facts are established either by ground-truth (module 1's use of the color-map and rules file) or by a trained neural network (module 4).

To a degree, an `*.enactment` file is an interpretation of those facts. Objects are seen from an AR/VR user's perspective. The user's field of view is Gaussian-weighted according to the assumption that centrally-framed objects are significant to actions being performed. The hands also have their own Gaussians to upweight objects near the hands, assuming that objects being manipulated are central to a task. These weighted signals form the basis for classifying the user's action. The particulars of this Gaussian-weighting can be controlled here. Depending on the tasks you wish to classify, you may prefer a wide and shallow Gaussian or a deep and thin Gaussian to upweight objects.

## assemble_enactment.py

There are several parameters to tune in this script. Code comments and the usage print out will tell you more.

e.g. The following call will process Enactments 11 and 12 using the default hand-sensor poses.
```
python3 assemble_enactment.py -e Enactment11 -e Enactment12 -v -render
```

e.g. The following call will process Enactment 1 using object detections by MobileNet and hand poses from ground-truth IK.
```
python3 assemble_enactment.py -e Enactment1 -suffix _ssd_mobilenet_640x640_detections.txt -handsrc .IK-bbox.handposes -v -render
```

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
