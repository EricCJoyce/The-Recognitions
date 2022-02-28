# 5 - Perform atemporal matching

Though the purpose of this pipeline is to arrive at real-time action classifier, it is helpful to run classification atemporally to establish a baseline and determine downstream parameters.

Notice that at this point in the pipeline, we have abstracted classification away from using actual enactment video frames. We will re-introduce video input when we proceed to real-time classification.

## atemporal.py

There are many parameters to tune in this script. Code comments for this script, its usage print-out, and the accompanying classes' comments will tell you more.

## Inputs

One or more `*.enactment` files from the same detection source, that is either from ground-truth or from a trained network.

## Outputs

This script is used mainly for reporting classification performance under ideal conditions. However, it also writes several report files:

- `confidences-all-<timestamp>.txt`
- `confidences-winners-<timestamp>.txt`
- `confusion-matrix-<timestamp>.txt`
- `results-<timestamp>.txt`
- `timing-<timestamp>.txt`

The `confidences-*.txt` files can be used by module 7 to perform isotonic regression.
