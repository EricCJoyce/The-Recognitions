# 1 - Processing Enactments

Enactments are annotated recordings of egocentric actions.

## process_enactment.py

The main goal of `process_enactment.py` is to produce, for each enactment, a `*_props.txt` file, several pose files, and masks for all objects we wish to recognize downstream in the pipeline. An enactment's props file is a list of which objects are visible in which frames, where their masks are, and which (if any) action is being performed during that frame. Much of this information already exists in the enactment metadata; `process_enactment.py` organizes it into a single look-up table, generates masks according to the given rules, and--depending on which settings you give the script--renders several "sanity checks" like videos and 3D models (`*.ply` files) that plot the recording in space.

### Enactments are expected to be directories with the following structure:
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

### Output

After running this script, your enactment folder should look like this:
```
./EnactmentName
    |
    |--- /GT         <---- Filled with object masks, indexed in *_props.txt
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

For every call to `process_enactment.py`, expect:
```
process_enactment_<time stamp>.log
total-histogram.png
```

For each enactment, expect:
```
*_props.txt
*-histogram.png
```
This script also adds a folder, `GT`, to each enactment's directory. Inside this new directory are all the masks referenced by `*_props.txt`.

If you enable rendering when you call `process_enactment.py`, additionally expect for each enactment:
```
*_annotated.avi
*_composite.avi
*.actions.global.ply
*.actions.headframe.ply
*.skeleton.global.ply
*.skeleton.headframe.ply
```

The net effect of running this script is that color-coded parts have been (re)assigned to recognizable objects according to the rules we provided in the rules file. In each enactment, in each frame, we know what is the recognizable object, where it is within the frame, and which mask separates it from everything else in the frame.

We can use this information both to train a neural network and to create an `.enactment` file directly, according to ground-truth.

## extract_hand_poses.py

Module 3 trains a deep network to detect "props" in the egocentric scene. Module 4 applies a trained network to enactments to produce `*_detections.txt` files. These are identical to the `*_props.txt` files produced here in Module 1, except that they depend on network detection rather than ground-truth and have no knowledge of hand positions. This is by design: we allow the possibility that hand sensors are noisy, and we assume that training a network to identify hands in addition to props would be redundant. For these reasons, the expectation is that you may want to use a trained network to detect "props" and take hand data from another source.

Once you have run Modules 3 and 4 and intend to produce detection-based `*.enactment` files using Module 2, first use `extract_hand_poses.py` to extract hand poses. These can come from sensors, or from inverse-kinematic (IK) hand avatars.
