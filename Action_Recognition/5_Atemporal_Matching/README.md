# 5 - Perform atemporal matching

Though the purpose of this pipeline is to arrive at real-time action classifier, it is helpful to run classification atemporally to establish a baseline and determine downstream parameters.

## Inputs

One or more `*.enactment` files from the same detection source, that is from ground-truth or from a trained network.

## Outputs

This script is used mainly for reporting classification performance under ideal conditions. However, it also writes two files, `conf-pred-gt_<timestamp>.txt` and `conf-pred-gt-all_<timestamp>.txt`, which can be used by module 7 to perform isotonic regression.

## Requirements

Notice that at this point in the pipeline, we have abstracted classification away from using actual enactment video frames. We will re-introduce video input when we proceed to real-time classification.
- Python
- MatPlotLib
- NumPy
- SciKit-Learn
