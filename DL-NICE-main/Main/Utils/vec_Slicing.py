import numpy as np
import random


def vec_slicing(vIn, blocksize, stepsize=60, shuffle=True, splitratio=0.7):
    """
    it accepts an maximum length integer , and slices it according to the blocksize and
    stepsize. Then it returns a vector vOut
   Detailed explanation goes here
    Args:
        vIn(int): an integer, indicates the max length of the data for slicing
        blocksize(int): an integer, block size
        stepsize(int): an integer, stepsize, overlapping
        shuffle(Boolean): whether to shuffle
        splitratio(float): 0.7 by default

    Returns:training sample start idx, testing sample start idx

    """
    assert isinstance(vIn, int) and isinstance(blocksize, int) \
           and isinstance(stepsize, int), 'the inputs must be integers'

    startidx = np.arange(0, vIn - blocksize + 1, stepsize)
    startidx_len = len(startidx)
    if shuffle:
        random.shuffle(startidx)
        trainidx, testidx = np.split(startidx, [round(startidx_len * splitratio), ])
    else:
        trainidx, testidx = np.split(startidx, [round(startidx_len * splitratio), ])
    return trainidx, testidx
