import numpy as np

import time
import argparse

import util.exp as exp
import util.evaluation as evaluation
import util.visualization as viz
import util.submit as submit

# TODO
# load predictions from multiple output/<exp_name>/predictions folders
# take average of them
# convert from prob to mask
# apply RLE
# save into submission.csv
