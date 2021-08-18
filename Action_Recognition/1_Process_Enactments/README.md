# 1 - Processing Enactments

## Inputs

### Enactments as directories with the expected structure:
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
    |--- Labels.fvr
    `--- metadata.fvr
```
### An optional `.rules` file:

Rules files allow us to combine, re-assign, and subject to condition all the objects seen in the enactments. The file `SubpropColorMap.fvr` assigns a unique color to every part in the virtual world. Recognizing actions involves first recognizing things with which VR users interact, so we must first group a world of parts into instances of recognizable objects.

## Outputs

The main goal of `process_enactment.py` is to produce, for each enactment, a "ground-truth" file, several pose files, and masks for all objects we wish to recognize downstream in the pipeline. An enactment's ground-truth file is a list of which objects are visible in which frames, where their masks are, and which (if any) action is being performed during that frame. Much of this information already exists in the enactment metadata; `process_enactment.py` organizes it into a single look-up table, generates masks according to the given rules, and--depending on which settings you give the script--renders several "sanity checks" like videos and 3D models (`.ply` files) that plot the recording in space.

For every call to `process_enactment.py`, expect:
```
FVR_check.log
histogram.png
```

For each enactment, expect:
```
*_groundtruth.txt
*_IKposes.bbox.shutoff.txt
*_IKposes.cntr.shutoff.txt
*_poses.full.txt
*_poses.head.txt
*_poses.shutoff.txt
*_subprops.txt
```
If you enable rendering when you call `process_enactment.py`, additionally expect for each enactment:
```
*_annotated.avi
*_annotated-avg.avi
*_annotated-bbox.avi
*_composite.avi
*_groundtruth.avi
```

### 

## Requirements
### Python
### OpenCV
### MatPlotLib
