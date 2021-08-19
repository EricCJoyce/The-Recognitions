# 7 - Map matching confidence to probabilities

The classifier can always find a best match between an accumulation of incoming frames and the representative samples in its database. How ambiguous that best match is can be captured by defining a confidence in that match. A best match that is significantly better than its second-best is unambiguous, and the system should have confidence in this match. If the best few are too close to be sure, then the system should have low confidence in the best match.

The problem with this or any other definition of confidence is that its range has no upper bound. A confidence of 2.345 is better than 1.234, but how significant is this difference? Should we feel a little more confident or a lot more confident?

Isotonic regression maps the confidences collected by module 5 to the range \[0.0, 1.0\] and allows us to speak in terms of probability rather than confidence.

## Inputs

The classification files `conf-pred-gt_<timestamp>.txt` and `conf-pred-gt-all_<timestamp>.txt` created by module 5.

## Outputs

A human-readable file `isoreg.txt` or `isoreg-ALL.txt` depending on which of the input files was used. This file itemizes all the "buckets" determined by isotonic regression. These are used during classification to map a confidence score to a probability. (Again, for deployment, you would probably prefer that this be a binary file.)

## Requirements
- Python
- MatPlotLib
- NumPy
- Sci-Kit Learn
