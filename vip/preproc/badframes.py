#! /usr/bin/env python

"""
Module with functions for outlier frame detection.
"""

from __future__ import division

__author__ = 'C. Gomez @ ULg'
__all__ = ['cube_detect_badfr_pxstats',
           'cube_detect_badfr_ellipticipy',
           'cube_detect_badfr_correlation']

import numpy as np
import pandas as pn
from matplotlib import pyplot as plt
from photutils import detection
from astropy.stats import sigma_clip
from ..var import get_annulus
from ..conf import time_ini, timing
from ..stats import (cube_stats_aperture, cube_stats_annulus, 
                              cube_distance)


def cube_detect_badfr_pxstats(array, mode='annulus', in_radius=10, width=10, 
                              top_sigma=1.0, low_sigma=1.0, window=None, 
                              plot=True, verbose=True):             
    """ Returns the list of bad frames from a cube using the px statistics in 
    a centered annulus or circular aperture. Frames that are more than a few 
    standard deviations discrepant are rejected. Should be applied on a 
    recentered cube.
    
    Parameters
    ----------
    array : array_like 
        Input 3d array, cube.
    mode : {'annulus', 'circle'}, string optional
        Whether to take the statistics from a circle or an annulus.
    in_radius : int optional
        If mode is 'annulus' then 'in_radius' is the inner radius of the annular 
        region. If mode is 'circle' then 'in_radius' is the radius of the 
        aperture.
    width : int optional
        Size of the annulus. Ignored if mode is 'circle'.
    top_sigma : int, optional
        Top boundary for rejection.
    low_sigma : int, optional
        Lower boundary for rejection.
    window : int, optional
        Window for smoothing the median and getting the rejection statistic.
    plot : {True, False}, bool optional
        If true it plots the mean fluctuation as a function of the frames and 
        the boundaries.
    verbose : {True, False}, bool optional
        Whether to print to stdout or not.
    
    Returns
    -------
    good_index_list : array_like
        1d array of good indices.
    bad_index_list : array_like
        1d array of bad frames indices.
    
    """
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array')
    if in_radius+width > array[0].shape[0]/2.:
        msgve = 'Inner radius and annulus size are too big (out of boundaries)'
        raise ValueError(msgve)
    
    if verbose:  start_time = time_ini()
    
    n = array.shape[0]
    
    if mode=='annulus':
        mean_values,_,_,_ = cube_stats_annulus(array, inner_radius=in_radius, 
                                               size=width, full_out=True)
    elif mode=='circle':
        _,mean_values,_,_ = cube_stats_aperture(array, radius=in_radius, 
                                                full_output=True)
    else: 
        raise TypeError('Mode not recognized')
    
    if window is None:  window = int(n/3.)
    mean_smooth = pn.rolling_median(mean_values, window , center=True)
    temp = pn.Series(mean_smooth)
    temp = temp.fillna(method='backfill')
    temp = temp.fillna(method='ffill')
    mean_smooth = temp.values
    sigma = np.std(mean_values)
    bad_index_list = []
    good_index_list = []
    top_boundary = np.empty([n])
    bot_boundary = np.empty([n])
    for i in range(n):
        if mode=='annulus':
            i_mean_value = get_annulus(array[i], inner_radius=in_radius, 
                                        width=width, output_values=True).mean()
        elif mode=='circle':
            i_mean_value = mean_values[i]
        top_boundary[i] = mean_smooth[i] + top_sigma*sigma
        bot_boundary[i] = mean_smooth[i] - low_sigma*sigma
        if (i_mean_value > top_boundary[i] or i_mean_value < bot_boundary[i]):
            bad_index_list.append(i)
        else:
            good_index_list.append(i)                       

    if verbose:
        bad = len(bad_index_list)
        percent_bad_frames = (bad*100)/n
        msg1 = "Done detecting bad frames from cube: {:} out of {:} ({:.3}%)"
        print msg1.format(bad, n, percent_bad_frames) 

    if plot:
        plt.figure(figsize=(10, 6), dpi=100)
        plt.plot(mean_values, 'o', label='mean fluctuation', lw = 1.4)
        plt.plot(mean_smooth, label='smoothed median', lw = 2, ls='-', alpha=0.5)
        plt.plot(top_boundary, label='top limit', lw = 1.4, ls='-')
        plt.plot(bot_boundary, label='lower limit', lw = 1.4, ls='-')
        plt.legend(fancybox=True, framealpha=0.5)

    if verbose:  timing(start_time)

    good_index_list = np.array(good_index_list)
    bad_index_list = np.array(bad_index_list)
    
    return good_index_list, bad_index_list


def cube_detect_badfr_ellipticipy(array, fwhm, roundlo=-0.2, roundhi=0.2,
                                  verbose=True):
    """ Returns the list of bad frames  from a cube by measuring the PSF 
    ellipticity of the central source. Should be applied on a recentered cube.
    
    Parameters
    ----------
    array : array_like 
        Input 3d array, cube.
    fwhm : float
        FWHM size in pixels.
    roundlo, roundhi : float, optional
        Lower and higher bounds for the ellipticipy.
    verbose : {True, False}, bool optional
        Whether to print to stdout or not.
        
    Returns
    -------
    good_index_list : array_like
        1d array of good indices.
    bad_index_list : array_like
        1d array of bad frames indices.
    
    Notes
    -----
    From photutils.daofind documentation:
    DAOFIND calculates the object roundness using two methods.  The 'roundlo' 
    and 'roundhi' bounds are applied to both measures of roundness.  The first 
    method ('roundness1'; called 'SROUND' in DAOFIND) is based on the source 
    symmetry and is the ratio of a measure of the object's bilateral (2-fold) 
    to four-fold symmetry. The second roundness statistic ('roundness2'; called 
    'GROUND' in DAOFIND) measures the ratio of the difference in the height of
    the best fitting Gaussian function in x minus the best fitting Gaussian 
    function in y, divided by the average of the best fitting Gaussian 
    functions in x and y.  A circular source will have a zero roundness. A 
    source extended in x or y will have a negative or positive roundness, 
    respectively.
    
    """
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array')
    
    if verbose:  start_time = time_ini()
    
    n = array.shape[0]
    
    # Calculate a 2D Gaussian density enhancement kernel
    gauss2d_kernel = detection.findstars._FindObjKernel
    # Find sources in an image by convolving the image with the input kernel 
    # and selecting connected pixels above a given threshold
    find_objs = detection.findstars._findobjs
    # Find the properties of each detected source, as defined by DAOFIND
    obj_prop = detection.findstars._daofind_properties

    ff_clipped = sigma_clip(array[0], sig=4, iters=None)
    thr = ff_clipped.max()
    
    goodfr = []
    badfr = []
    for i in range(n):
        # we create a circular gaussian kernel
        kernel = gauss2d_kernel(fwhm=fwhm, ratio=1.0, theta=0.0)  
        objs = find_objs(array[i], threshold=thr, kernel=kernel, 
                         exclude_border=True)
        tbl = obj_prop(objs, threshold=thr, kernel=kernel, sky=0)
        # we mask the peak px object
        table_mask = (tbl['peak'] == tbl['peak'].max())
        tbl = tbl[table_mask]
        roun1 = tbl['roundness1'][0]
        roun2 = tbl['roundness2'][0]
        # we check the roundness
        if roun1>roundlo and roun1<roundhi and roun2>roundlo and roun2<roundhi:  
            goodfr.append(i)
        else:
            badfr.append(i)
    
    bad_index_list = np.array(badfr)
    good_index_list = np.array(goodfr)
    if verbose:
        bad = len(bad_index_list)
        percent_bad_frames = (bad*100)/n
        msg1 = "Done detecting bad frames from cube: {:} out of {:} ({:.3}%)"
        print msg1.format(bad, n, percent_bad_frames) 
    
    if verbose:  timing(start_time)
    
    return good_index_list, bad_index_list


def cube_detect_badfr_correlation(array, frame_ref, crop_size=30, dist='pearson',
                                  percentile=20, plot=True, verbose=True):
    """ Returns the list of bad frames from a cube by measuring the distance 
    (similarity) or correlation of the frames (cropped to a 30x30 subframe) 
    wrt a reference frame from the same cube. Then the distance/correlation 
    level is thresholded (percentile parameter) to find the outliers. Should be 
    applied on a recentered cube.
    
    Parameters
    ----------
    array : array_like 
        Input 3d array, cube.
    frame_ref : int
        Index of the frame that will be used as a reference. Must be of course
        a frame taken with a good wavefront quality.
    dist : {'sad','euclidean','mse','pearson','spearman'}, str optional
        One of the similarity or disimilarity measures from function 
        vip.stats.distances.cube_distance(). 
    percentile : int
        The percentage of frames that will be discarded. 
    plot : {True, False}, bool optional
        If true it plots the mean fluctuation as a function of the frames and 
        the boundaries.
    verbose : {True, False}, bool optional
        Whether to print to stdout or not.
            
    Returns
    -------
    good_index_list : array_like
        1d array of good indices.
    bad_index_list : array_like
        1d array of bad frames indices.
        
    """
    from .cosmetics import cube_crop_frames
    
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array')
    
    if verbose:  start_time = time_ini()
    
    n = array.shape[0]
    # the cube is cropped to the central area
    subarray = cube_crop_frames(array, min(crop_size, array.shape[1]), verbose=False)
    distances = cube_distance(subarray, frame_ref, 'full', dist, plot=False)
        
    if dist=='pearson' or dist=='spearman': # measures of correlation or similarity
        minval = np.min(distances[~np.isnan(distances)])
        distances = np.nan_to_num(distances)
        distances[np.where(distances==0)] = minval
        threshold = np.percentile(distances, percentile)
        indbad = np.where(distances <= threshold)
        indgood = np.where(distances > threshold)
    else:                                                   # measures of dissimilarity
        threshold = np.percentile(distances, 100-percentile)
        indbad = np.where(distances >= threshold)
        indgood = np.where(distances < threshold)
        
    bad_index_list = indbad[0]
    good_index_list = indgood[0]
    if verbose:
        bad = len(bad_index_list)
        percent_bad_frames = (bad*100)/n
        msg1 = "Done detecting bad frames from cube: {:} out of {:} ({:.3}%)"
        print msg1.format(bad, n, percent_bad_frames) 
    
    if plot:
        lista = distances
        _, ax = plt.subplots(figsize=(10, 6), dpi=100)
        x = range(len(lista))
        ax.plot(x, lista, '-', color='blue', alpha=0.3)
        if n>5000:
            ax.plot(x, lista, ',', color='blue', alpha=0.5)
        else:
            ax.plot(x, lista, '.', color='blue', alpha=0.5)
        ax.vlines(frame_ref, ymin=np.nanmin(lista), ymax=np.nanmax(lista), 
                   colors='green', linestyles='dashed', lw=2, alpha=0.8,
                   label='Reference frame '+str(frame_ref))
        ax.hlines(np.median(lista), xmin=-1, xmax=n+1, colors='purple', 
                   linestyles='solid', label='Median value')
        ax.hlines(threshold, xmin=-1, xmax=n+1, lw=2, colors='red', 
                   linestyles='dashed', label='Threshold')
        plt.xlabel('Frame number')
        if dist=='sad':
            plt.ylabel('SAD - Manhattan distance')
        elif dist=='euclidean':
            plt.ylabel('Euclidean distance')
        elif dist=='pearson':
            plt.ylabel('Pearson correlation coefficient')
        elif dist=='spearman':
            plt.ylabel('Spearman correlation coefficient')
        elif dist=='mse':
            plt.ylabel('Mean squared error')
        elif dist=='ssim':
            plt.ylabel('Structural Similarity Index')
        
        plt.xlim(xmin=-1, xmax=n+1)
        plt.minorticks_on()
        plt.legend(fancybox=True, framealpha=0.5, fontsize=12, loc='best')
        plt.grid(which='mayor')
    
    if verbose:  timing(start_time)
    
    return good_index_list, bad_index_list


