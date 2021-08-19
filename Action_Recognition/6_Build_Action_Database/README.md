# 6 - Build a database

The deployed classifier will have an on-board database of action snippets. A real-time frame buffer will compare its contents against these representative samples and find a best match. Therefore, we want this database to be rich enough that a good match is likely, but not so full that it becomes too much to search. The script `build_db.py` therefore lets us strategically thin out the database to what are (hopefully) the most definitive samples.

## Inputs

One or more `*.enactment` files from the same detection source, that is from ground-truth or from a trained network.

## Outputs

A human-readable `*.db` database file. (For deployment, you may want to re-write this as a binary file.)

## Requirements
- Python
- MatPlotLib
- NumPy
- SciKit-Learn
- R
