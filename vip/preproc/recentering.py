#! /usr/bin/env python

"""
Module containing functions for cubes frame registration.
"""

from __future__ import division

__author__ = 'C. Gomez @ ULg, V. Christiaens @ ULg/UChile'
__all__ = ['frame_shift',
           'frame_center_radon',
           'frame_center_satspots',
           'cube_recenter_satspots',
           'cube_recenter_radon',
           'cube_recenter_dft_upsampling',
           'cube_recenter_gauss2d_fit',
           'cube_recenter_moffat2d_fit']

import numpy as np
import warnings
import pywt
import itertools as itt
import pyprind

try:
    import cv2
    no_opencv = False
except ImportError:
    msg = "Opencv python binding are missing (consult VIP documentation for "
    msg += "Opencv installation instructions)"
    warnings.warn(msg, ImportWarning)
    no_opencv = True

from scipy.ndimage import fourier_shift
from scipy.ndimage import shift
from skimage.transform import radon
from skimage.feature import register_translation
from multiprocessing import Pool, cpu_count
#from image_registration import chi2_shift
from matplotlib import pyplot as plt
from . import approx_stellar_position
from . import frame_crop
from ..conf import time_ini, timing
from ..conf import eval_func_tuple as EFT
from ..var import (get_square, get_square_robust, frame_center,
                   get_annulus, pp_subplots, fit_2dmoffat, fit_2dgaussian)


def frame_shift(array, shift_y, shift_x, imlib='ndimage-fourier',
                interpolation='bicubic'):
    """ Shifts an 2d array by shift_y, shift_x. Boundaries are filled with zeros. 

    Parameters
    ----------
    array : array_like
        Input 2d array.
    shift_y, shift_x: float
        Shifts in x and y directions.
    imlib : {'ndimage-fourier', 'opencv', 'ndimage-interp'}, string optional
        Library or method used for performing the image shift.
    interpolation : {'bicubic', 'bilinear', 'nearneig'}, optional
        Only used in case of imlib is set to 'opencv' or 'ndimage-interp', where
        the images are shifted via interpolation. 'nneighbor' stands for
        nearest-neighbor, 'bilinear' stands for bilinear and 'bicubic' stands
        for bicubic interpolation over 4x4 pixel neighborhood. 'bicubic' is the
        default. The 'nearneig' is the fastest method and the 'bicubic' the
        slowest of the three. The 'nearneig' is the poorer option for
        interpolation of noisy astronomical images.
    
    Returns
    -------
    array_shifted : array_like
        Shifted 2d array.

    Notes
    -----
    Regarding the imlib parameter: 'ndimage-fourier', does a fourier shift
    operation and preserves better the pixel values (therefore the flux and
    photometry). 'ndimage-fourier' is used by default from VIP version 0.5.3.
    Interpolation based shift ('opencv' and 'ndimage-interp') is faster than the
    fourier shift. 'opencv' could be used when speed is critical and the flux
    preservation is not that important.

    """
    if not array.ndim == 2:
        raise TypeError ('Input array is not a frame or 2d array')
    
    image = array.copy()

    if imlib not in ['ndimage-fourier', 'ndimage-interp', 'opencv']:
        msg = 'Imlib value not recognized, try ndimage-fourier, ndimage-interp '
        msg += 'or opencv'
        raise ValueError(msg)

    if imlib=='ndimage-fourier':
        shift_val = (shift_y, shift_x)
        array_shifted = fourier_shift(np.fft.fftn(image), shift_val)
        array_shifted = np.fft.ifftn(array_shifted)
        array_shifted = array_shifted.real

    elif imlib=='ndimage-interp':
        if interpolation == 'bilinear':
            intp = 1
        elif interpolation == 'bicubic':
            intp= 3
        elif interpolation == 'nearneig':
            intp = 0
        else:
            raise TypeError('Interpolation method not recognized.')
        
        array_shifted = shift(image, (shift_y, shift_x), order=intp)
    
    elif imlib=='opencv':
        if no_opencv:
            msg = 'Opencv python bindings cannot be imported. Install opencv or '
            msg += 'set imlib to ndimage-fourier or ndimage-interp'
            raise RuntimeError(msg)

        if interpolation == 'bilinear':
            intp = cv2.INTER_LINEAR
        elif interpolation == 'bicubic':
            intp= cv2.INTER_CUBIC
        elif interpolation == 'nearneig':
            intp = cv2.INTER_NEAREST
        else:
            raise TypeError('Interpolation method not recognized.')
        
        image = np.float32(image)
        y, x = image.shape
        M = np.float32([[1,0,shift_x],[0,1,shift_y]])
        array_shifted = cv2.warpAffine(image, M, (x,y), flags=intp)
    
    return array_shifted


# TODO: expose 'imlib' parameter in the rest of functions that use frame_shift
# function

def frame_center_satspots(array, xy, subim_size=19, sigfactor=6, shift=False,
                          debug=False): 
    """ Finds the center of a frame with waffle/satellite spots (e.g. for 
    VLT/SPHERE). The method used to determine the center is by centroiding the
    4 spots via a 2d Gaussian fit and finding the intersection of the 
    lines they create (see Notes). This method is very sensitive to the SNR of 
    the satellite spots, therefore thresholding of the background pixels is 
    performed. If the results are too extreme, the debug parameter will allow to 
    see in depth what is going on with the fit (maybe you'll need to adjust the 
    sigfactor for the background pixels thresholding).  
    
    Parameters
    ----------
    array : array_like, 2d
        Image or frame.
    xy : tuple
        Tuple with coordinates X,Y of the satellite spots in this order:
        upper left, upper right, lower left, lower right. 
    subim_size : int, optional
        Size of subimage where the fitting is done.
    sigfactor : int, optional
        The background pixels will be thresholded before fitting a 2d Gaussian
        to the data using sigma clipped statistics. All values smaller than
        (MEDIAN + sigfactor*STDDEV) will be replaced by small random Gaussian 
        noise. 
    shift : {False, True}, optional 
        If True the image is shifted with bicubic interpolation. 
    debug : {False, True}, optional 
        If True debug information is printed and plotted.
    
    Returns
    -------
    shifty, shiftx 
        Shift Y,X to get to the true center.
    If shift is True then the shifted image is returned along with the shift. 
    
    Notes
    -----
    linear system:
    A1 * x + B1 * y = C1
    A2 * x + B2 * y = C2
    
    Cramer's rule - solution can be found in determinants:
    x = Dx/D
    y = Dy/D
    where D is main determinant of the system:  A1 B1
                                                A2 B2
    and Dx and Dy can be found from matrices:  C1 B1
                                               C2 B2
    and  A1 C1
         A2 C2
    C column consequently substitutes the coef. columns of x and y

    L stores our coefs A, B, C of the line equations.
    For D: L1[0] L1[1]   for Dx: L1[2] L1[1]   for Dy: L1[0] L1[2]
           L2[0] L2[1]           L2[2] L2[1]           L2[0] L2[2]
    """
    def line(p1, p2):
        """ produces coefs A, B, C of line equation by 2 points
        """
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
        return A, B, -C

    def intersection(L1, L2):
        """ finds intersection point (if any) of 2 lines provided 
        by coefs
        """
        D  = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return x,y
        else:
            return False
        
    #---------------------------------------------------------------------------
    if not array.ndim == 2:
        raise TypeError('Input array is not a frame or 2d array')
    if not len(xy) == 4:
        raise TypeError('Input waffle spot coordinates in wrong format')
    
    # If frame size is even we drop last row and last column
    if array.shape[0]%2==0:
        array = array[:-1,:].copy()
    if array.shape[1]%2==0:
        array = array[:,:-1].copy()
    
    cy, cx = frame_center(array)
    
    # Upper left
    si1, y1, x1 = get_square(array, subim_size, xy[0][1], xy[0][0], position=True)
    cent2dgx_1, cent2dgy_1 = fit_2dgaussian(si1, theta=135, crop=False, 
                                            threshold=True, sigfactor=sigfactor, 
                                            debug=debug)
    cent2dgx_1 += x1
    cent2dgy_1 += y1
    # Upper right
    si2, y2, x2 = get_square(array, subim_size, xy[1][1], xy[1][0], position=True)
    cent2dgx_2, cent2dgy_2 = fit_2dgaussian(si2, theta=45, crop=False, 
                                            threshold=True, sigfactor=sigfactor, 
                                            debug=debug)
    cent2dgx_2 += x2
    cent2dgy_2 += y2 
    #  Lower left
    si3, y3, x3 = get_square(array, subim_size, xy[2][1], xy[2][0], position=True)
    cent2dgx_3, cent2dgy_3 = fit_2dgaussian(si3, theta=45, crop=False, 
                                            threshold=True, sigfactor=sigfactor, 
                                            debug=debug)
    cent2dgx_3 += x3
    cent2dgy_3 += y3
    #  Lower right
    si4, y4, x4 = get_square(array, subim_size, xy[3][1], xy[3][0], position=True)
    cent2dgx_4, cent2dgy_4 = fit_2dgaussian(si4, theta=135, crop=False, 
                                            threshold=True, sigfactor=sigfactor, 
                                            debug=debug)
    cent2dgx_4 += x4
    cent2dgy_4 += y4
    
    if debug: 
        pp_subplots(si1, si2, si3, si4, colorb=True)
        print 'Centroids X,Y:'
        print cent2dgx_1, cent2dgy_1
        print cent2dgx_2, cent2dgy_2
        print cent2dgx_3, cent2dgy_3
        print cent2dgx_4, cent2dgy_4

    L1 = line([cent2dgx_1, cent2dgy_1], [cent2dgx_4, cent2dgy_4])
    L2 = line([cent2dgx_2, cent2dgy_2], [cent2dgx_3, cent2dgy_3])
    R = intersection(L1, L2)
    
    if R:
        shiftx = cx-R[0]
        shifty = cy-R[1]
        if debug: 
            print '\nIntersection coordinates (X,Y):', R[0], R[1], '\n'
            print 'Shifts (X,Y):', shiftx, shifty
        
        if shift:
            array_rec = frame_shift(array, shifty, shiftx)
            return array_rec, shifty, shiftx
        else:
            return shifty, shiftx
    else:
        print 'Something went wrong, no intersection found.'
        return 0


def cube_recenter_satspots(array, xy, subim_size=19, sigfactor=6, debug=False):
    """ Function analog to frame_center_satspots but for image sequences. It 
    actually will call frame_center_satspots for each image in the cube. The
    function also returns the shifted images (not recommended to use when the 
    shifts are of a few percents of a pixel) and plots the histogram of the
    shifts and calculate its statistics. This is important to assess the 
    dispersion of the star center by using artificial waffle/satellite spots
    (like those in VLT/SPHERE images) and evaluate the uncertainty of the 
    position of the center. The use of the shifted images is not recommended.   
    
    Parameters
    ----------
    array : array_like, 3d
        Input cube.
    xy : tuple
        Tuple with coordinates X,Y of the satellite spots in this order:
        upper left, upper right, lower left, lower right. 
    subim_size : int, optional
        Size of subimage where the fitting is done.
    sigfactor : int, optional
        The background pixels will be thresholded before fitting a 2d Gaussian
        to the data using sigma clipped statistics. All values smaller than
        (MEDIAN + sigfactor*STDDEV) will be replaced by small random Gaussian 
        noise. 
    debug : {False, True}, optional 
        If True debug information is printed and plotted (fit and residuals,
        intersections and shifts). This has to be used carefully as it can 
        produce too much output and plots. 
    
    Returns
    ------- 
    array_rec
        The shifted cube.
    shift_y, shift_x
        Shifts Y,X to get to the true center for each image.
    
    """    
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array')

    start_time = time_ini()

    n_frames = array.shape[0]
    shift_x = np.zeros((n_frames))
    shift_y = np.zeros((n_frames))
    array_rec = []
    
    bar = pyprind.ProgBar(n_frames, stream=1, title='Looping through frames')
    for i in range(n_frames):
        res = frame_center_satspots(array[i], xy, debug=debug, shift=True,
                                    subim_size=subim_size, sigfactor=sigfactor)
        array_rec.append(res[0])
        shift_y[i] = res[1] 
        shift_x[i] = res[2]          
        bar.update()
       
    timing(start_time)

    plt.figure(figsize=(13,4))
    plt.plot(shift_x, '.-', lw=0.5, color='green', label='Shifts X')
    plt.plot(shift_y, '.-', lw=0.5, color='blue', label='Shifts Y')
    plt.xlim(0, shift_y.shape[0]+5)
    _=plt.xticks(range(0,n_frames,5))
    plt.legend()

    print 'AVE X,Y', np.mean(shift_x), np.mean(shift_y)
    print 'MED X,Y', np.median(shift_x), np.median(shift_y)
    print 'STD X,Y', np.std(shift_x), np.std(shift_y)

    plt.figure()
    b = int(np.sqrt(n_frames))
    _ = plt.hist(shift_x, bins=b, alpha=0.5, color='green', label='Shifts X')
    _ = plt.hist(shift_y, bins=b, alpha=0.5, color='blue', label='Shifts Y')
    plt.legend()

    array_rec = np.array(array_rec) 
    return array_rec, shift_y, shift_x



def frame_center_radon(array, cropsize=101, hsize=0.4, step=0.01,
                       mask_center=None, nproc=None, satspots=False,
                       full_output=False, verbose=True, plot=True, debug=False):
    """ Finding the center of a broadband (co-added) frame with speckles and 
    satellite spots elongated towards the star (center). 
    
    The radon transform comes from scikit-image package. Takes a few seconds to
    compute one radon transform with good resolution. 
    
    Parameters
    ----------
    array : array_like
        Input 2d array or image.
    cropsize : odd int, optional
        Size in pixels of the cropped central area of the input array that will
        be used. It should be large enough to contain the satellite spots.
    hsize : float, optional
        Size of the box for the grid search. The frame is shifted to each 
        direction from the center in a hsize length with a given step.
    step : float, optional
        The step of the coordinates change.
    mask_center : None or int, optional
        If None the central area of the frame is kept. If int a centered zero 
        mask will be applied to the frame. By default the center isn't masked.
    nproc : int, optional
        Number of processes for parallel computing. If None the number of 
        processes will be set to (cpu_count()/2). 
    verbose : {True, False}, bool optional
        Whether to print to stdout some messages and info.
    plot : {True, False}, bool optional
        Whether to plot the radon cost function. 
    debug : {False, True}, bool optional
        Whether to print and plot intermediate info.
    
    Returns
    -------
    optimy, optimx : float
        Values of the Y, X coordinates of the center of the frame based on the
        radon optimization.
    If full_output is True then the radon cost function surface is returned 
    along with the optimal x and y.
        
    Notes
    -----
    The whole idea of this algorithm is based on Pueyo et al. 2014 paper: 
    http://arxiv.org/abs/1409.6388
    
    """
    from .cosmetics import frame_crop
    
    if not array.ndim==2:
        raise TypeError('Input array is not a frame or 2d array')

    if verbose:  start_time = time_ini()
    frame = array.copy()
    frame = frame_crop(frame, cropsize, verbose=False)
    listyx = np.linspace(start=-hsize, stop=hsize, num=2*hsize/step+1, 
                         endpoint=True)
    if not mask_center:
        radint = 0
    else:
        if not isinstance(mask_center, int):
            raise(TypeError('Mask_center must be either None or an integer'))
        radint = mask_center
    
    coords = [(y,x) for y in listyx for x in listyx]
    cent, _ = frame_center(frame)
           
    frame = get_annulus(frame, radint, cent-radint)
                
    if debug:
        if satspots:
            samples = 10
            theta = np.hstack((np.linspace(start=40, stop=50, num=samples, 
                                           endpoint=False), 
                               np.linspace(start=130, stop=140, num=samples, 
                                           endpoint=False),
                               np.linspace(start=220, stop=230, num=samples, 
                                           endpoint=False),
                               np.linspace(start=310, stop=320, num=samples, 
                                           endpoint=False)))             
            sinogram = radon(frame, theta=theta, circle=True)
            pp_subplots(frame, sinogram)
            print np.sum(np.abs(sinogram[cent,:]))
        else:
            theta = np.linspace(start=0., stop=360., num=cent*2, endpoint=False)
            sinogram = radon(frame, theta=theta, circle=True)
            pp_subplots(frame, sinogram)
            print np.sum(np.abs(sinogram[cent,:]))

    if not nproc:   # Hyper-threading "duplicates" the cores -> cpu_count/2
        nproc = (cpu_count()/2) 
    pool = Pool(processes=int(nproc))  
    if satspots:
        res = pool.map(EFT,itt.izip(itt.repeat(_radon_costf2), itt.repeat(frame), 
                                    itt.repeat(cent), itt.repeat(radint), coords))        
    else:
        res = pool.map(EFT,itt.izip(itt.repeat(_radon_costf), itt.repeat(frame), 
                                    itt.repeat(cent), itt.repeat(radint), coords)) 
    costf = np.array(res)
    pool.close()
        
    if verbose:  
        msg = 'Done {} radon transform calls distributed in {} processes'
        print msg.format(len(coords), int(nproc))

    cost_bound = costf.reshape(listyx.shape[0], listyx.shape[0])
    if plot:
        plt.contour(cost_bound, cmap='CMRmap', origin='lower', lw=1, hold='on')
        plt.imshow(cost_bound, cmap='CMRmap', origin='lower', 
                   interpolation='nearest')
        plt.colorbar()
        plt.grid('off')
        plt.show()
        
    #argm = np.argmax(costf) # index of 1st max in 1d cost function 'surface' 
    #optimy, optimx = coords[argm]
    
    # maxima in the 2d cost function surface
    num_max = np.where(cost_bound==cost_bound.max())[0].shape[0]
    ind_maximay, ind_maximax = np.where(cost_bound==cost_bound.max())
    argmy = ind_maximay[int(np.ceil(num_max/2))-1]
    argmx = ind_maximax[int(np.ceil(num_max/2))-1]
    y_grid = np.array(coords)[:,0].reshape(listyx.shape[0], listyx.shape[0])
    x_grid = np.array(coords)[:,1].reshape(listyx.shape[0], listyx.shape[0])
    optimy = y_grid[argmy, 0] 
    optimx = x_grid[0, argmx]  
    
    if verbose: 
        print 'Cost function max: {}'.format(costf.max())
        print 'Cost function # maxima: {}'.format(num_max)
        msg = 'Finished grid search radon optimization. Y={:.5f}, X={:.5f}'
        print msg.format(optimy, optimx)
        timing(start_time)
    
    if full_output:
        return cost_bound, optimy, optimx
    else:
        return optimy, optimx


def _radon_costf(frame, cent, radint, coords):
    """ Radon cost function used in frame_center_radon().
    """
    frame_shifted = frame_shift(frame, coords[0], coords[1])
    frame_shifted_ann = get_annulus(frame_shifted, radint, cent-radint)
    theta = np.linspace(start=0., stop=360., num=frame_shifted_ann.shape[0],    
                    endpoint=False)
    sinogram = radon(frame_shifted_ann, theta=theta, circle=True)
    costf = np.sum(np.abs(sinogram[cent,:]))
    return costf
                
                
def _radon_costf2(frame, cent, radint, coords):
    """ Radon cost function used in frame_center_radon().
    """
    frame_shifted = frame_shift(frame, coords[0], coords[1])
    frame_shifted_ann = get_annulus(frame_shifted, radint, cent-radint)
    samples = 10
    theta = np.hstack((np.linspace(start=40, stop=50, num=samples, endpoint=False), 
                   np.linspace(start=130, stop=140, num=samples, endpoint=False),
                   np.linspace(start=220, stop=230, num=samples, endpoint=False),
                   np.linspace(start=310, stop=320, num=samples, endpoint=False)))
    
    sinogram = radon(frame_shifted_ann, theta=theta, circle=True)
    costf = np.sum(np.abs(sinogram[cent,:]))
    return costf


      
def cube_recenter_radon(array, full_output=False, verbose=True, **kwargs):
    """ Recenters a cube looping through its frames and calling the 
    frame_center_radon() function. 
    
    Parameters
    ----------
    array : array_like
        Input 3d array or cube.
    full_output : {False, True}, bool optional
        If True the recentered cube is returned along with the y and x shifts.
    verbose : {True, False}, bool optional
        Whether to print timing and intermediate information to stdout.
    
    Optional parameters (keywords and values) can be passed to the     
    frame_center_radon function. 
    
    Returns
    -------
    array_rec : array_like
        Recentered cube.
    If full_output is True:
    y, x : 1d array of floats
        Shifts in y and x.
     
    """
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array')

    if verbose:  start_time = time_ini()

    n_frames = array.shape[0]
    x = np.zeros((n_frames))
    y = np.zeros((n_frames))
    array_rec = array.copy()
    
    bar = pyprind.ProgBar(n_frames, stream=1, title='Looping through frames')
    for i in range(n_frames):
        y[i], x[i] = frame_center_radon(array[i], verbose=False, plot=False, 
                                        **kwargs)
        array_rec[i] = frame_shift(array[i], y[i], x[i])
        bar.update()
        
    if verbose:  timing(start_time)

    if full_output:
        return array_rec, y, x
    else:
        return array_rec

                

def cube_recenter_dft_upsampling(array, cy_1, cx_1, negative=False, fwhm=4, 
                                 subi_size=None, upsample_factor=100,
                                 full_output=False, verbose=True,
                                 save_shifts=False, debug=False):
    """ Recenters a cube of frames using the DFT upsampling method as 
    proposed in Guizar et al. 2008 (see Notes) plus a chi^2, for determining
    automatically the upsampling factor, as implemented in the package 
    'image_registration' (see Notes).
    
    The algorithm (DFT upsampling) obtains an initial estimate of the 
    cross-correlation peak by an FFT and then refines the shift estimation by 
    upsampling the DFT only in a small neighborhood of that estimate by means 
    of a matrix-multiply DFT.
    
    Parameters
    ----------
    array : array_like
        Input cube.
    cy_1, cx_1 : int
        Coordinates of the center of the subimage for fitting a 2d Gaussian and
        centroiding the 1st frame. 
    negative : {False, True}, optional
        If True the centroiding of the 1st frames is done with a negative 
        2d Gaussian fit.   
    fwhm : float, optional
        FWHM size in pixels.
    subi_size : int or None, optional
        Size of the square subimage sides in terms of FWHM that will be used
        to centroid to frist frame. If subi_size is None then the first frame
        is assumed to be centered already.
    upsample_factor :  int optional
        Upsampling factor (default 100). Images will be registered to within
        1/upsample_factor of a pixel.
    full_output : {False, True}, bool optional
        Whether to return 2 1d arrays of shifts along with the recentered cube 
        or not.
    verbose : {True, False}, bool optional
        Whether to print to stdout the timing or not.
    save_shifts : {False, True}, bool optional
        Whether to save the shifts to a file in disk.
    debug : {False, True}, bool optional
        Whether to print to stdout the shifts or not. 
    
    Returns
    -------
    array_recentered : array_like
        The recentered cube. Frames have now odd size.
    If full_output is True:
    y, x : array_like
        1d arrays with the shifts in y and x.     
    
    Notes
    -----
    Using the implementation from skimage.feature.register_translation.
    
    Guizar-Sicairos et al. "Efficient subpixel image registration algorithms," 
    Opt. Lett. 33, 156-158 (2008). 
    The algorithm registers two images (2-D rigid translation) within a fraction 
    of a pixel specified by the user. Instead of computing a zero-padded FFT
    (fast Fourier transform), this code uses selective upsampling by a
    matrix-multiply DFT (discrete FT) to dramatically reduce computation time
    and memory without sacrificing accuracy. With this procedure all the image
    points are used to compute the upsampled cross-correlation in a very small
    neighborhood around its peak.
    
    """
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array')
    
    # If frame size is even we drop a row and a column
    if array.shape[1]%2==0:
        array = array[:,1:,:].copy()
    if array.shape[2]%2==0:
        array = array[:,:,1:].copy()
    
    if verbose:  start_time = time_ini()
    
    n_frames = array.shape[0]
    x = np.zeros((n_frames))
    y = np.zeros((n_frames))
    array_rec = array.copy()

    cy, cx = frame_center(array[0])
    # Centroiding first frame with 2d gaussian and shifting
    if subi_size is not None:
        size = int(np.round(fwhm*subi_size))
        y1, x1 = _centroid_2dg_frame(array_rec, 0, size, cy_1, cx_1, negative,
                                     debug=debug)
        x[0] = cx-x1
        y[0] = cy-y1
        array_rec[0] = frame_shift(array_rec[0], shift_y=y[0], shift_x=x[0])
        if verbose:
            print "\nShift for first frame X,Y=({:.3f},{:.3f})".format(x[0],y[0])
            print "The rest of the frames will be shifted by cross-correlation" \
                  " with the first one"
        if debug:
            pp_subplots(frame_crop(array[0], size, verbose=False),
                        frame_crop(array_rec[0], size, verbose=False),
                        grid=True, title='original / shifted 1st frame subimage')
    else:
        if verbose:
            print "It's assumed that the first frame is well centered"
            print "The rest of the frames will be shifted by cross-correlation" \
                  " with the first one"
        x[0] = cx
        y[0] = cy
    
    # Finding the shifts with DTF upsampling of each frame wrt the first
    bar = pyprind.ProgBar(n_frames, stream=1, title='Looping through frames')
    for i in range(1, n_frames):
        shift_yx, _, _ = register_translation(array_rec[0], array[i],
                                              upsample_factor=upsample_factor)
        y[i], x[i] = shift_yx
        #dx, dy, _, _ = chi2_shift(array_rec[0], array[i], upsample_factor='auto')
        #x[i] = -dx
        #y[i] = -dy
        array_rec[i] = frame_shift(array[i], shift_y=y[i], shift_x=x[i])
        bar.update()

    if debug:
        print "\nShifts in X and Y"
        for i in range(n_frames):
            print x[i], y[i]
        
    if verbose:  timing(start_time)
        
    if save_shifts: 
        np.savetxt('recent_dft_shifts.txt', np.transpose([y, x]), fmt='%f')
    if full_output:
        return array_rec, y, x
    else:
        return array_rec
  

def cube_recenter_gauss2d_fit(array, xy, fwhm=4, subi_size=5, nproc=1, 
                              full_output=False, verbose=True, save_shifts=False, 
                              offset=None, negative=False, debug=False,
                              threshold=False):
    """ Recenters the frames of a cube. The shifts are found by fitting a 2d 
    gaussian to a subimage centered at (pos_x, pos_y). This assumes the frames 
    don't have too large shifts (>5px). The frames are shifted using the 
    function frame_shift() (bicubic interpolation).
    
    Parameters
    ----------
    array : array_like
        Input cube.
    xy : tuple of int
        Coordinates of the center of the subimage.    
    fwhm : float or array_like
        FWHM size in pixels, either one value (float) that will be the same for
        the whole cube, or an array of floats with the same dimension as the 
        0th dim of array, containing the fwhm for each channel (e.g. in the case
        of an ifs cube, where the fwhm varies with wavelength)
    subi_size : int, optional
        Size of the square subimage sides in terms of FWHM.
    nproc : int or None, optional
        Number of processes (>1) for parallel computing. If 1 then it runs in 
        serial. If None the number of processes will be set to (cpu_count()/2).
    full_output : {False, True}, bool optional
        Whether to return 2 1d arrays of shifts along with the recentered cube 
        or not.
    verbose : {True, False}, bool optional
        Whether to print to stdout the timing or not.
    save_shifts : {False, True}, bool optional
        Whether to save the shifts to a file in disk.
    offset : tuple of floats, optional
        If None the region of the frames used for the 2d Gaussian fit is shifted
        to the center of the images (2d arrays). If a tuple is given it serves
        as the offset of the fitted area wrt the center of the 2d arrays.
    negative : {False, True}, optional
        If True a negative 2d Gaussian fit is performed.
    debug : {False, True}, bool optional
        If True the details of the fitting are shown. This might produce an
        extremely long output and therefore is limited to <20 frames.
        
    Returns
    -------
    array_recentered : array_like
        The recentered cube. Frames have now odd size.
    If full_output is True:
    y, x : array_like
        1d arrays with the shifts in y and x. 
    
    """    
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array')
    
    n_frames = array.shape[0]
    if isinstance(fwhm,int) or isinstance(fwhm,float):  
        fwhm_tmp = fwhm 
        fwhm = np.zeros(n_frames)
        fwhm[:] = fwhm_tmp
    subfr_sz = subi_size*fwhm
    subfr_sz = subfr_sz.astype(int)
        
    if debug and array.shape[0]>20:
        msg = 'Debug with a big array will produce a very long output. '
        msg += 'Try with less than 20 frames in debug mode.'
        raise RuntimeWarning(msg)
    
    pos_x, pos_y = xy
    
    if not isinstance(pos_x,int) or not isinstance(pos_y,int):
        raise TypeError('pos_x and pos_y should be ints')
    
    # If frame size is even we drop a row and a column
    if array.shape[1]%2==0:
        array = array[:,1:,:].copy()
    if array.shape[2]%2==0:
        array = array[:,:,1:].copy()
    
    if verbose:  start_time = time_ini()
    
    cy, cx = frame_center(array[0])
    array_recentered = np.empty_like(array)  
    
    if not nproc:   # Hyper-threading "duplicates" the cores -> cpu_count/2
        nproc = (cpu_count()/2) 
    if nproc==1:
        res = []
        bar = pyprind.ProgBar(n_frames, stream=1, 
                              title='2d Gauss-fitting, looping through frames')
        for i in range(n_frames):
            res.append(_centroid_2dg_frame(array, i, subfr_sz[i], 
                                           pos_y, pos_x, negative, debug, fwhm[i], 
                                           threshold))
            bar.update()
        res = np.array(res)
    elif nproc>1:
        pool = Pool(processes=int(nproc))  
        res = pool.map(EFT, itt.izip(itt.repeat(_centroid_2dg_frame),
                                     itt.repeat(array),
                                     range(n_frames),
                                     subfr_sz, 
                                     itt.repeat(pos_y), 
                                     itt.repeat(pos_x),
                                     itt.repeat(negative),
                                     itt.repeat(debug),
                                     fwhm,
                                     itt.repeat(threshold))) 
        res = np.array(res)
        pool.close()
    y = cy - res[:,0]
    x = cx - res[:,1]
    #return x, y
    
    if offset is not None:
        offx, offy = offset
        y -= offy
        x -= offx
    
    bar2 = pyprind.ProgBar(n_frames, stream=1, title='Shifting the frames')
    for i in range(n_frames):
        if debug:
            print "\nShifts in X and Y"
            print x[i], y[i]
        array_recentered[i] = frame_shift(array[i], y[i], x[i])
        bar2.update()
        
    if verbose:  timing(start_time)

    if save_shifts: 
        np.savetxt('recent_gauss_shifts.txt', np.transpose([y, x]), fmt='%f')
    if full_output:
        return array_recentered, y, x
    else:
        return array_recentered


def cube_recenter_moffat2d_fit(array, pos_y, pos_x, fwhm=4, subi_size=5, 
                               nproc=None, full_output=False, verbose=True, 
                               save_shifts=False, debug=False, 
                               unmoving_star=True, negative=False):
    """ Recenters the frames of a cube. The shifts are found by fitting a 2d 
    moffat to a subimage centered at (pos_x, pos_y). This assumes the frames 
    don't have too large shifts (>5px). The frames are shifted using the 
    function frame_shift() (bicubic interpolation).
    
    Parameters
    ----------
    array : array_like
        Input cube.
    pos_y, pos_x : int or array_like
        Coordinates of the center of the subimage.    
    fwhm : float or array_like
        FWHM size in pixels, either one value (float) that will be the same for
        the whole cube, or an array of floats with the same dimension as the 
        0th dim of array, containing the fwhm for each channel (e.g. in the case
        of an ifs cube, where the fwhm varies with wavelength)
    subi_size : int, optional
        Size of the square subimage sides in terms of FWHM.
    nproc : int or None, optional
        Number of processes (>1) for parallel computing. If 1 then it runs in 
        serial. If None the number of processes will be set to (cpu_count()/2).
    full_output : {False, True}, bool optional
        Whether to return 2 1d arrays of shifts along with the recentered cube 
        or not.
    verbose : {True, False}, bool optional
        Whether to print to stdout the timing or not.
    save_shifts : {False, True}, bool optional
        Whether to save the shifts to a file in disk.
    debug : {False, True}, bool optional
        Whether to print to stdout the shifts or not. 
    unmoving_star : {False, True}, bool optional
        Whether the star centroid is expected to not move a lot within the 
        frames of the input cube. If True, then an additional test is done to 
        be sure the centroid fit returns a reasonable index value (close to the 
        median of the centroid indices in the other frames) - hence not taking 
        noise or a clump of uncorrected bad pixels.
    negative : {False, True}, optional
        If True a negative 2d Moffat fit is performed.
        
    Returns
    -------
    array_recentered : array_like
        The recentered cube. Frames have now odd size.
    If full_output is True:
    y, x : array_like
        1d arrays with the shifts in y and x. 
    
    """    
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array')
    # if not pos_x or not pos_y:
    #     raise ValueError('Missing parameters POS_Y and/or POS_X')
    
    # If frame size is even we drop a row and a column
    if array.shape[1]%2==0:
        array = array[:,1:,:].copy()
    if array.shape[2]%2==0:
        array = array[:,:,1:].copy()
    
    if verbose:  start_time = time_ini()
    
    n_frames = array.shape[0]
    cy, cx = frame_center(array[0])
    array_recentered = np.empty_like(array)  

    if isinstance(fwhm,float) or isinstance(fwhm,int):
        fwhm_scal = fwhm
        fwhm = np.zeros((n_frames))
        fwhm[:] = fwhm_scal
    size = np.zeros(n_frames)    
    for kk in range(n_frames):
        size[kk] = max(2,int(fwhm[kk]*subi_size))
    
    if isinstance(pos_x,int) or isinstance(pos_y,int):
        if isinstance(pos_x,int) and not isinstance(pos_y,int):
            raise ValueError('pos_x and pos_y should have the same shape')
        elif not isinstance(pos_x,int) and isinstance(pos_y,int):
            raise ValueError('pos_x and pos_y should have the same shape')
        pos_x_scal, pos_y_scal = pos_x, pos_y
        pos_x, pos_y = np.zeros((n_frames)),np.zeros((n_frames))
        pos_x[:], pos_y[:] = pos_x_scal, pos_y_scal

    ### Precaution: some frames are dominated by noise and hence cannot be used
    ### to find the star with a Moffat or Gaussian fit.
    ### In that case, just replace the coordinates by the approximate ones
    if unmoving_star:
        star_approx_coords, star_not_present = approx_stellar_position(array,
                                                                       fwhm,
                                                                       True)
        star_approx_coords.tolist()
        star_not_present.tolist()
    else:
        star_approx_coords, star_not_present = [None]*n_frames, [None]*n_frames

    if not nproc:   # Hyper-threading "duplicates" the cores -> cpu_count/2
        nproc = (cpu_count()/2) 
    if nproc==1:
        res = []
        bar = pyprind.ProgBar(n_frames, stream=1, 
                              title='Looping through frames')
        for i in range(n_frames):
            res.append(_centroid_2dm_frame(array, i, size[i], pos_y[i], 
                                           pos_x[i], star_approx_coords[i], 
                                           star_not_present[i], negative, fwhm[i]))
            bar.update()
        res = np.array(res)
    elif nproc>1:
        pool = Pool(processes=int(nproc))  
        res = pool.map(EFT,itt.izip(itt.repeat(_centroid_2dm_frame),
                                    itt.repeat(array), 
                                    range(n_frames),
                                    size.tolist(), 
                                    pos_y.tolist(), 
                                    pos_x.tolist(), 
                                    star_approx_coords,
                                    star_not_present, 
                                    itt.repeat(negative), 
                                    fwhm))
        res = np.array(res)
        pool.close()
    y = cy - res[:,0]
    x = cx - res[:,1]
        
    for i in range(n_frames):
        if debug:
            print "\nShifts in X and Y"
            print x[i], y[i]
        array_recentered[i] = frame_shift(array[i], y[i], x[i])

    if verbose:  timing(start_time)

    if save_shifts: 
        np.savetxt('recent_moffat_shifts.txt', np.transpose([y, x]), fmt='%f')
    if full_output:
        return array_recentered, y, x
    else:
        return array_recentered


def _centroid_2dg_frame(cube, frnum, size, pos_y, pos_x, negative, debug=False, 
                        fwhm=4,threshold=False):
    """ Finds the centroid by using a 2d gaussian fitting in one frame from a 
    cube. To be called from within cube_recenter_gauss2d_fit().
    """
    sub_image, y1, x1 = get_square_robust(cube[frnum], size=size, y=pos_y, 
                                          x=pos_x, position=True)

    # negative gaussian fit
    if negative:  sub_image = -sub_image + np.abs(np.min(-sub_image))
            
    y_i, x_i = fit_2dgaussian(sub_image, crop=False, fwhmx=fwhm, fwhmy=fwhm, 
                              threshold=threshold, sigfactor=1, debug=debug)
    y_i = y1 + y_i
    x_i = x1 + x_i
    return y_i, x_i


def _centroid_2dm_frame(cube, frnum, size, pos_y, pos_x, 
                        star_approx_coords=None, star_not_present=None,
                        negative=False, fwhm=4):
    """ Finds the centroid by using a 2d moffat fitting in one frame from a 
    cube. To be called from within cube_recenter_moffat2d_fit().
    """
    sub_image, y1, x1 = get_square_robust(cube[frnum], size=size+1, y=pos_y, 
                                          x=pos_x,position=True)
    sub_image = sub_image.byteswap().newbyteorder()
    # negative fit
    if negative:  sub_image = -sub_image + np.abs(np.min(-sub_image))
        
    if star_approx_coords is not None and star_not_present is not None:
        if star_not_present:
            y_i,x_i = star_approx_coords
        else:
            y_i, x_i = fit_2dmoffat(sub_image, y1, x1, full_output=False, fwhm=fwhm)
    else:
        y_i, x_i = fit_2dmoffat(sub_image, y1, x1, full_output=False, fwhm=fwhm)
    return y_i, x_i




        
