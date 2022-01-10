# 3 - Training a deep network to recognize objects

The ground-truth masks created by module 1 can be used to train a deep network to recognize objects. We will take advantage of a pre-trained version of [MobileNet](https://arxiv.org/abs/1704.04861), transferring the learning already achieved in its lower layers to our new task.

This module involves several steps. You will need to:
1. Prepare data sets and a network-training workspace.
2. Install the TensorFlow Object Detection (TFOD) API.
3. Download pre-trained models from the TensorFlow Model Zoo.
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

## 3.1 - Prepare training and validation sets, and prepare a training workspace

### build_object_recog_dataset.py

Build a dataset from enactments in preparation for training an object-detection network. Several objects typically appear in a single video frame. This script attempts to balance training and validation sets according to the given parameters, but must allocate frames entirely to one set or the other. We do not want to incur false false-positives!

Outputs `training-set.txt` and `validation-set.txt`

### convert_object_recog_dataset.py

Convert the training and validation set files to formats expected by the TensorFlow Object-Detection API. This means creating two folders in the current working directory: `./training/images/train` and `./training/images/test`. This script will also generate `./training/annotations/label_map.pbtxt`. This script expects to find `./training-set.txt` and `./validation-set.txt`. It also expects to find the enactments referenced by these documents. This script will copy all images into the respective folders and generate one annotation XML per image. A single XML contains all detections in its frame.

By the end of this process, your working directory should resemble this:

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
    |--- EnactmentN_props.txt
   ...
    |
    `--- /training
           |
           |--- /annotations
           |       |
           |       `--- label_map.pbtxt
           |--- /exported-models     <--- Initially empty
           |--- /images
           |       |
           |       |--- /train       <--- Enactment frames *copied* (not moved)
           |       |       |              from your enactments, and one *.xml per frame.
           |       |       |--- 1.png
           |       |       |--- 1.xml
           |       |       |--- 2.png
           |       |       |--- 2.xml
           |       |      ...
           |       |
           |       `--- /test        <--- Enactment frames *copied* (not moved)
           |               |              from your enactments, and one *.xml per frame.
           |               |--- 1.png
           |               |--- 1.xml
           |               |--- 2.png
           |               |--- 2.xml
           |              ...
           |
           |--- /models              <--- Initially empty
           `--- /pre-trained-models  <--- Initially empty
```

## 3.2 - Install the TensorFlow Object Detection (TFOD) API

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
```

If the call to `make` fails, open the `Makefile` in a text editor and be sure that it calls the correct version of Python. You may need to change `python` in lines 3 and 8 to `python3` or `python3.6`.

Moving on:

```
cp -r pycocotools ../..
cd ../..
cp object_detection/packages/tf2/setup.py .
python3.6 -m pip install .
```

You may receive some errors after this last command, which you can likely ignore. The real test of whether this API installation succeeded is the following:
```
python3.6 object_detection/builders/model_builder_tf2_test.py
```

Now you can convert the dataset you've prepared in the `./training` workspace to TFRecords.
The script to perform this conversion can be copied from [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records). Simply save the Python code from the yellow box as a file in your working directory named `generate_tfrecord.py`.

Then, make the following calls from your working directory:
```
python3.6 generate_tfrecord.py -x ./training/images/train -l ./training/annotations/label_map.pbtxt -o ./training/annotations/train.record
python3.6 generate_tfrecord.py -x ./training/images/test -l ./training/annotations/label_map.pbtxt -o ./training/annotations/test.record
```

## 3.3 - Download a pre-trained model from the [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

Peruse the Model Zoo and decide which pre-trained model you would like to work with. For our purposes, we will use [SSD MobileNet V2 FPNLite 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz)

From your working directory, make the following calls:
```
cd training/pre-trained-models/
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz
tar -xvf ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz
```

Now copy the extracted model's `pipeline.config` file to a directory that we'll create in `./training/models` named after the model downloaded from the zoo. Your case may vary, but here are the commands for this example:
```
cd ../..
mkdir training/models/ssd_mobilenet_640x640
cp training/pre-trained-models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/pipeline.config training/models/ssd_mobilenet_640x640/
```

## 3.4 - Train

## Requirements
- Python
- NumPy
- OpenCV
- TensorFlow/Keras
