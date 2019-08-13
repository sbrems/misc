import numpy as np


def bin_median(arr, smaller_by_factor=1, returnStd=False):
    '''bin an array arr by creating super pixels of size
    smaller_by_factor*smaller_by_factor and taking the median.
    Can optionally also return the standard deviation within
    the super-pixels.
    INPUTS:
    arr: 2d array
    smaller_by_factor: integer
    returnStd: bool, default False
    RETURNS binned_array OR binned_array, bin_std'''
    sub_arrs0 = []
    for i in range(smaller_by_factor):
        for j in range(smaller_by_factor):
            sub_arrs0.append(arr[i::smaller_by_factor, j::smaller_by_factor])
    sub_arrs = [
        s[:sub_arrs0[-1].shape[0] - 1, :sub_arrs0[-1].shape[1] - 1]
        for s in sub_arrs0
    ]
    if returnStd:
        # zip truncates each sub-arr at the length of the minimum length subarr.
        # this ensures every bin has the same number of datapoints, but throws
        # away data if the last bin doesn't have a full share of datapoints.
        return np.median(sub_arrs, axis=0), np.std(sub_arrs, axis=0)
    else:
        return np.median(sub_arrs, axis=0)
