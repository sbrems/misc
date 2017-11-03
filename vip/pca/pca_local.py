#! /usr/bin/env python

"""
Module with local smart pca (annulus-wise) serial and parallel implementations.
"""

from __future__ import division, print_function

__author__ = 'C. Gomez @ ULg'
__all__ = ['pca_adi_annular',
           'pca_rdi_annular']

import numpy as np
import itertools as itt
from scipy import stats
from multiprocessing import Pool, cpu_count
from ..preproc import cube_derotate, cube_collapse, check_PA_vector
from ..conf import time_ini, timing
from ..conf import eval_func_tuple as EFT 
from ..var import get_annulus_quad, get_annulus
from ..pca.utils_pca import svd_wrapper, matrix_scaling
from ..stats import descriptive_stats



def pca_rdi_annular(cube, angle_list, cube_ref, radius_int=0, asize=1, 
                    ncomp=1, svd_mode='randsvd', min_corr=0.9, fwhm=4, 
                    scaling='temp-standard', collapse='median', 
                    full_output=False, verbose=True, debug=False):
    """ Annular PCA with Reference Library + Correlation + standardization
    
    In the case of having a large number of reference images, e.g. for a survey 
    on a single instrument, we can afford a better selection of the library by 
    constraining the correlation with the median of the science dataset and by 
    working on an annulus-wise way. As with other local PCA algorithms in VIP
    the number of principal components can be automatically adjusted by the
    algorithm by minmizing the residuals in the given patch (a la LOCI). 
    
    Parameters
    ----------
    cube : array_like, 3d
        Input science cube.
    angle_list : array_like, 1d
        Corresponding parallactic angle for each frame.
    cube_ref : array_like, 3d
        Reference library cube. For Reference Star Differential Imaging.
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the 
        central circular area is discarded.
    asize : int, optional
        The size of the annuli, in FWHM. Default is 3.
    ncomp : int, optional
        How many PCs are kept. If none it will be automatically determined.
    svd_mode : {randsvd, eigen, lapack, arpack, opencv}, optional
        Switch for different ways of computing the SVD and principal components.
    min_corr : int, optional
        Level of linear correlation between the library patches and the median 
        of the science. Deafult is 0.9.
    fwhm : float, optional
        Known size of the FHWM in pixels to be used. Deafult is 4.
    scaling : {None, 'temp-mean', 'spat-mean', 'temp-standard', 'spat-standard'}
        With None, no scaling is performed on the input data before SVD. With 
        "temp-mean" then temporal px-wise mean subtraction is done, with 
        "spat-mean" then the spatial mean is subtracted, with "temp-standard" 
        temporal mean centering plus scaling to unit variance is done and with
        "spat-standard" spatial mean centering plus scaling to unit variance is
        performed.  
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    full_output: boolean, optional
        Whether to return the final median combined image only or with other 
        intermediate arrays.  
    verbose : {True, False}, bool optional
        If True prints to stdout intermediate info. 
    debug : {False, True}, bool optional
        Whether to output some intermediate information.
    
    Returns
    -------
    frame : array_like, 2d    
        Median combination of the de-rotated cube.
    If full_output is True:  
    array_out : array_like, 3d 
        Cube of residuals.
    array_der : array_like, 3d
        Cube residuals after de-rotation.    
    
    """
    def define_annuli(angle_list, ann, n_annuli, fwhm, radius_int, annulus_width, 
                      verbose):
        """ Defining the annuli """
        if ann == n_annuli-1:
            inner_radius = radius_int + (ann*annulus_width-1)
        else:                                                                                         
            inner_radius = radius_int + ann*annulus_width
        ann_center = (inner_radius+(annulus_width/2.0))
        
        if verbose:
            msg2 = 'Annulus {:}, Inn radius = {:.2f}, Ann center = {:.2f} '
            print(msg2.format(int(ann+1),inner_radius, ann_center))
        return inner_radius, ann_center
    
    def fr_ref_correlation(vector, matrix):
        """ Getting the correlations """
        lista = []
        for i in range(matrix.shape[0]):
            pears, _ = stats.pearsonr(vector, matrix[i])
            lista.append(pears)
        
        return lista

    def do_pca_annulus(ncomp, matrix, svd_mode, noise_error, data_ref):
        """ Actual PCA for the annulus """
        #V = svd_wrapper(data_ref, svd_mode, ncomp, debug=False, verbose=False)
        V = get_eigenvectors(ncomp, matrix, svd_mode, noise_error=noise_error, 
                             data_ref=data_ref, debug=False)
        # new variables as linear combinations of the original variables in 
        # matrix.T with coefficientes from EV
        transformed = np.dot(V, matrix.T) 
        reconstructed = np.dot(V.T, transformed)
        residuals = matrix - reconstructed.T  
        return residuals, V.shape[0]

    #---------------------------------------------------------------------------
    array = cube
    array_ref = cube_ref
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array.')
    if not array.shape[0] == angle_list.shape[0]:
        raise TypeError('Input vector or parallactic angles has wrong length.')
    
    n, y, _ = array.shape
    if verbose:  start_time = time_ini()
    
    angle_list = check_PA_vector(angle_list)

    annulus_width = asize * fwhm                     # equal size for all annuli
    n_annuli = int(np.floor((y/2-radius_int)/annulus_width))
    if verbose:
        msg = '# annuli = {:}, Ann width = {:}, FWHM = {:.3f}\n'
        print(msg.format(n_annuli, annulus_width, fwhm))
        print('PCA will be done locally per annulus and per quadrant.\n')
     
    cube_out = np.zeros_like(array)
    for ann in range(n_annuli):
        inner_radius, _ = define_annuli(angle_list, ann, n_annuli, fwhm, 
                                        radius_int, annulus_width, verbose) 
        indices = get_annulus(array[0], inner_radius, annulus_width,
                              output_indices=True)
        yy = indices[0]
        xx = indices[1]
                    
        matrix = array[:, yy, xx]                 # shape [nframes x npx_ann] 
        matrix_ref = array_ref[:, yy, xx]
        
        corr = fr_ref_correlation(np.median(matrix, axis=0), matrix_ref)
        indcorr = np.where(np.abs(corr)>=min_corr)
        #print indcorr
        data_ref = matrix_ref[indcorr]
        nfrslib = data_ref.shape[0]
                
        if nfrslib<5:
            msg = 'Too few frames left (<5) fulfill the given correlation level.'
            msg += 'Try decreasing it'
            raise RuntimeError(msg)
        
        matrix = matrix_scaling(matrix, scaling)
        data_ref = matrix_scaling(data_ref, scaling)
        
        residuals, ncomps = do_pca_annulus(ncomp, matrix, svd_mode, 10e-3, data_ref)  
        cube_out[:, yy, xx] = residuals  
            
        if verbose:
            print('# frames in LIB = {}'.format(nfrslib))
            print('# PCs = {}'.format(ncomps))
            print('Done PCA with {:} for current annulus'.format(svd_mode))
            timing(start_time)      
         
    cube_der = cube_derotate(cube_out, angle_list)
    frame = cube_collapse(cube_der, mode=collapse)
    if verbose:
        print('Done derotating and combining.')
        timing(start_time)
    if full_output:
        return cube_out, cube_der, frame 
    else:
        return frame           


def pca_adi_annular(cube, angle_list, radius_int=0, fwhm=4, asize=3, 
                    delta_rot=1, ncomp=1, svd_mode='randsvd', nproc=1,
                    min_frames_pca=10, tol=1e-1, scaling=None, quad=False,
                    collapse='median', full_output=False, verbose=True, 
                    debug=False):
    """ Annular (smart) ADI PCA. The PCA model is computed locally in each
    annulus (optionally quadrants of each annulus). For each annulus we discard
    reference images taking into account a parallactic angle threshold
    (set by ``delta_rot``).
     
    Depending on parameter ``nproc`` the algorithm can work with several cores.
    It's been tested on a Linux and OSX. The ACCELERATE library for linear 
    algebra calcularions, which comes by default in every OSX system, is broken 
    for multiprocessing. Avoid using this function unless you have compiled 
    Python against other linear algebra library. An easy fix is to install 
    latest ANACONDA (2.5 or later) distribution which ships MKL library 
    (replacing the problematic ACCELERATE). On linux with the default 
    LAPACK/BLAS libraries it successfully distributes the processes among all 
    the existing cores. 
    
    Parameters
    ----------
    cube : array_like, 3d
        Input cube.
    angle_list : array_like, 1d
        Corresponding parallactic angle for each frame.
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the 
        central circular area is discarded.
    fwhm : float, optional
        Known size of the FHWM in pixels to be used. Deafult is 4.
    asize : int, optional
        The size of the annuli, in FWHM. Default is 3 which is recommended when 
        ``quad`` is True. Smaller values are valid when ``quad`` is False.  
    delta_rot : int, optional
        Factor for increasing the parallactic angle threshold, expressed in FWHM.
        Default is 1 (excludes 1 FHWM on each side of the considered frame).
        According to Absil+13, a slightly better contrast can be reached for the 
        innermost annuli if we consider a ``delta_rot`` condition as small as 
        0.1 lambda/D. This is because at very small separation, the effect of 
        speckle correlation is more significant than self-subtraction.
    ncomp : int or list or 1d numpy array, optional
        How many PCs are kept. If none it will be automatically determined. If a
        list is provided and it matches the number of annuli then a different
        number of PCs will be used for each annulus (starting with the innermost
        one).
    svd_mode : {randsvd, eigen, lapack, arpack, opencv}, optional
        Switch for different ways of computing the SVD and principal components.
    nproc : None or int, optional
        Number of processes for parallel computing. If None the number of 
        processes will be set to (cpu_count()/2). By default the algorithm works
        in single-process mode.
    min_frames_pca : int, optional 
        Minimum number of frames in the PCA reference library. Be careful, when
        ``min_frames_pca`` <= ``ncomp``, then for certain frames the subtracted 
        low-rank approximation is not optimal (getting a 10 PCs out of 2 frames 
        is not possible so the maximum number of PCs is used = 2). In practice 
        the resulting frame may be more noisy. It is recommended to decrease 
        ``delta_rot`` and have enough frames in the libraries to allow getting 
        ``ncomp`` PCs.    
    tol : float, optional
        Stopping criterion for choosing the number of PCs when ``ncomp`` is None.
        Lower values will lead to smaller residuals and more PCs.
    quad : {False, True}, bool optional
        If False the images are processed in annular fashion. If True, quadrants
        of annulus are used instead.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    full_output: boolean, optional
        Whether to return the final median combined image only or with other 
        intermediate arrays.  
    scaling : {None, 'temp-mean', 'spat-mean', 'temp-standard', 'spat-standard'}
        With None, no scaling is performed on the input data before SVD. With 
        "temp-mean" then temporal px-wise mean subtraction is done, with 
        "spat-mean" then the spatial mean is subtracted, with "temp-standard" 
        temporal mean centering plus scaling to unit variance is done and with
        "spat-standard" spatial mean centering plus scaling to unit variance is
        performed.  
    verbose : {True, False}, bool optional
        If True prints to stdout intermediate info. 
    debug : {False, True}, bool optional
        Whether to output some intermediate information.
     
    Returns
    -------
    frame : array_like, 2d    
        Median combination of the de-rotated cube.
    If full_output is True:  
    array_out : array_like, 3d 
        Cube of residuals.
    array_der : array_like, 3d
        Cube residuals after de-rotation.
     
    """
    array = cube
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array')
    if not array.shape[0] == angle_list.shape[0]:
        raise TypeError('Input vector or parallactic angles has wrong length')

    n, y, _ = array.shape
     
    if verbose:  start_time = time_ini()
    
    angle_list = check_PA_vector(angle_list)

    annulus_width = asize * fwhm                     # equal size for all annuli
    n_annuli = int(np.floor((y/2-radius_int)/annulus_width))
    if verbose:
        msg = '# annuli = {:}, Ann width = {:}, FWHM = {:.3f}'
        print(msg.format(n_annuli, annulus_width, fwhm), '\n')
        print('PCA per annulus (and per quadrant if requested)\n')
     
    if nproc is None:   # Hyper-threading "duplicates" the cores -> cpu_count/2
        nproc = (cpu_count()/2)
    
    #***************************************************************************
    # The annuli are built, and the corresponding PA thresholds for frame 
    # rejection are calculated (at the center of the annulus)
    #***************************************************************************
    cube_out = np.zeros_like(array)
    for ann in range(n_annuli):
        if isinstance(ncomp, list) or isinstance(ncomp, np.ndarray):
            ncomp = list(ncomp)
            if len(ncomp) == n_annuli:
                ncompann = ncomp[ann]
            else:
                msge = 'If ncomp is a list, it must match the number of annuli'
                msg = 'NCOMP : {}, N ANN : {}, ANN SIZE : {}, ANN WIDTH : {}'
                print(msg.format(ncomp, n_annuli, annulus_width, asize))
                raise TypeError(msge)
        else:
            ncompann = ncomp

        pa_thr,inner_radius,ann_center = define_annuli(angle_list, ann, n_annuli, 
                                                       fwhm, radius_int, 
                                                       annulus_width, delta_rot,
                                                       verbose) 
        
        #***********************************************************************
        # Quad Annular ADI-PCA
        #***********************************************************************
        if quad:        
            indices = get_annulus_quad(array[0], inner_radius, annulus_width)
            #*******************************************************************
            # PCA matrix is created for each annular quadrant and scaling if 
            # needed
            #*******************************************************************
            for quadrant in range(4):
                yy = indices[quadrant][0]
                xx = indices[quadrant][1]
                matrix_quad = array[:, yy, xx]      # shape [nframes x npx_quad] 
                 
                matrix_quad = matrix_scaling(matrix_quad, scaling)              
                #matrix_quad = matrix_quad - matrix_quad.mean(axis=0)
                
                #***************************************************************
                # We loop the frames and do the PCA to obtain the residuals cube
                #***************************************************************
                residuals = do_pca_loop(matrix_quad, yy, xx, nproc, angle_list, 
                                        fwhm, pa_thr, scaling, ann_center, 
                                        svd_mode, ncompann, min_frames_pca, tol,
                                        debug, verbose)
                
                for frame in range(n):
                    cube_out[frame][yy, xx] = residuals[frame] 
        
        #***********************************************************************
        # Normal Annular ADI-PCA
        #***********************************************************************
        else:
            indices = get_annulus(array[0], inner_radius, annulus_width, 
                                  output_indices=True)
            yy = indices[0]
            xx = indices[1]
            #*******************************************************************
            # PCA matrix is created for each annular quadrant and scaling if 
            # needed
            #*******************************************************************
            matrix_ann = array[:, yy, xx]
            matrix_ann = matrix_scaling(matrix_ann, scaling)

            #*******************************************************************
            # We loop the frames and do the PCA to obtain the residuals cube
            #*******************************************************************
            residuals = do_pca_loop(matrix_ann, yy, xx, nproc, angle_list, fwhm, 
                                    pa_thr, scaling, ann_center, svd_mode,
                                    ncompann, min_frames_pca, tol, debug, verbose)
            
            for frame in range(n):
                cube_out[frame][yy, xx] = residuals[frame] 

        if verbose:
            print('Done PCA with {:} for current annulus'.format(svd_mode))
            timing(start_time)      
         
    #***************************************************************************
    # Cube is derotated according to the parallactic angle and median combined.
    #***************************************************************************
    cube_der = cube_derotate(cube_out, angle_list)
    frame = cube_collapse(cube_der, mode=collapse)
    if verbose:
        print('Done derotating and combining.')
        timing(start_time)
    if full_output:
        return cube_out, cube_der, frame 
    else:
        return frame 
        
    
    
###*****************************************************************************
### Help functions for encapsulating portions of the main algorithms. They 
### improve readability, debugging and code re-use *****************************
###*****************************************************************************
    
def compute_pa_thresh(ann_center, fwhm, delta_rot=1):
    """ Computes the parallactic angle theshold[degrees]
    Replacing approximation: delta_rot * (fwhm/ann_center) / np.pi * 180
    """
    return np.rad2deg(2*np.arctan(delta_rot*fwhm/(2*ann_center)))
    
    
def define_annuli(angle_list, ann, n_annuli, fwhm, radius_int, annulus_width, 
                  delta_rot, verbose):
    """ Function that defines the annuli geometry using the input parameters.
    Returns the parallactic angle threshold, the inner radius and the annulus
    center for each annulus.
    """
    if ann == n_annuli-1:
        inner_radius = radius_int + (ann*annulus_width-1)
    else:                                                                                         
        inner_radius = radius_int + ann*annulus_width
    ann_center = (inner_radius+(annulus_width/2.0))
    pa_threshold = compute_pa_thresh(ann_center, fwhm, delta_rot) 
     
    mid_range = np.abs(np.amax(angle_list) - np.amin(angle_list))/2
    if pa_threshold >= mid_range - mid_range * 0.1:
        new_pa_th = float(mid_range - mid_range * 0.1)
        if verbose:
            msg = 'PA threshold {:.2f} is too big, will be set to {:.2f}'
            print(msg.format(pa_threshold, new_pa_th))
        pa_threshold = new_pa_th
                         
    if verbose:
        msg2 = 'Annulus {:}, PA thresh = {:.2f}, Inn radius = {:.2f}, Ann center = {:.2f} '
        print(msg2.format(int(ann+1),pa_threshold,inner_radius, ann_center))
    return pa_threshold, inner_radius, ann_center


def find_indices(angle_list, frame, thr, truncate):  
    """ Returns the indices to be left in pca library.  
    
    # TODO: find a more pythonic way to to this!
    """
    n = angle_list.shape[0]
    index_prev = 0 
    index_foll = frame                                  
    for i in range(0, frame):
        if np.abs(angle_list[frame]-angle_list[i]) < thr:
            index_prev = i
            break
        else:
            index_prev += 1
    for k in range(frame, n):
        if np.abs(angle_list[k]-angle_list[frame]) > thr:
            index_foll = k
            break
        else:
            index_foll += 1
    
    half1 = range(0,index_prev)
    half2 = range(index_foll,n)
    
    # This truncation is done on the annuli after 15*FWHM and the goal is to 
    # keep min(num_frames/2, 200) in the library after discarding those based on
    # the PA threshold
    if truncate:
        thr = min(int(n/2), 200)                                                # TODO: 200 is optimal? new parameter? 
        if frame < thr: 
            half1 = range(max(0,index_prev-int(thr/2)), index_prev)
            half2 = range(index_foll, min(index_foll+thr-len(half1),n))
        else:
            half2 = range(index_foll, min(n, int(thr/2+index_foll)))
            half1 = range(max(0,index_prev-thr+len(half2)), index_prev)
    return np.array(half1+half2)



def do_pca_loop(matrix, yy, xx, nproc, angle_list, fwhm, pa_threshold, scaling, 
                ann_center, svd_mode, ncomp, min_frames_pca, tol, debug, verbose):
    """
    """
    matrix_ann = matrix
    n = matrix.shape[0]
    #***************************************************************
    # For each frame we call the subfunction do_pca_patch that will 
    # do PCA on the small matrix, where some frames are discarded 
    # according to the PA threshold, and return the residuals
    #***************************************************************          
    ncomps = []
    nfrslib = []          
    if nproc==1:
        residualarr = []
        for frame in range(n):
            res = do_pca_patch(matrix_ann, frame, angle_list, fwhm,
                               pa_threshold, scaling, ann_center, 
                               svd_mode, ncomp, min_frames_pca, tol,
                               debug)
            residualarr.append(res[0])
            ncomps.append(res[1])
            nfrslib.append(res[2])
        residuals = np.array(residualarr)
        
    elif nproc>1:
        #***********************************************************
        # A multiprocessing pool is created to process the frames in 
        # a parallel way. SVD/PCA is done in do_pca_patch function
        #***********************************************************            
        pool = Pool(processes=int(nproc))
        res = pool.map(EFT, itt.izip(itt.repeat(do_pca_patch), 
                                     itt.repeat(matrix_ann),
                                     range(n), itt.repeat(angle_list),
                                     itt.repeat(fwhm),
                                     itt.repeat(pa_threshold),
                                     itt.repeat(scaling),
                                     itt.repeat(ann_center),
                                     itt.repeat(svd_mode),
                                     itt.repeat(ncomp),
                                     itt.repeat(min_frames_pca),
                                     itt.repeat(tol),
                                     itt.repeat(debug)))
        res = np.array(res)
        residuals = np.array(res[:,0])
        ncomps = res[:,1]
        nfrslib = res[:,2]
        pool.close()                         

    # number of frames in library printed for each annular quadrant
    if verbose:
        descriptive_stats(nfrslib, verbose=verbose, label='Size LIB: ')
    # number of PCs printed for each annular quadrant     
    if ncomp is None and verbose:  
        descriptive_stats(ncomps, verbose=verbose, label='Numb PCs: ')
        
    return residuals


def do_pca_patch(matrix, frame, angle_list, fwhm, pa_threshold, scaling,
                 ann_center, svd_mode, ncomp, min_frames_pca, tol, debug):
    """
    Does the SVD/PCA for each frame patch (small matrix). For each frame we 
    find the frames to be rejected depending on the amount of rotation. The
    library is also truncated on the other end (frames too far or which have 
    rotated more) which are more decorrelated to keep the computational cost 
    lower. This truncation is done on the annuli after 10*FWHM and the goal is
    to keep min(num_frames/2, 200) in the library. 
    """
    if pa_threshold != 0:
        if ann_center > fwhm*10:                                                 # TODO: 10*FWHM optimal? new parameter?
            indices_left = find_indices(angle_list, frame, pa_threshold, True)
        else:
            indices_left = find_indices(angle_list, frame, pa_threshold, False)
         
        data_ref = matrix[indices_left]
        
        if data_ref.shape[0] <= min_frames_pca:
            msg = 'Too few frames left in the PCA library. '
            msg += 'Try decreasing either delta_rot or min_frames_pca.'
            raise RuntimeError(msg)
    else:
        data_ref = matrix
       
    data = data_ref
    #data = data_ref - data_ref.mean(axis=0)
    curr_frame = matrix[frame]                     # current frame
    
    V = get_eigenvectors(ncomp, data, svd_mode, noise_error=tol, debug=False)        
    
    transformed = np.dot(curr_frame, V.T)
    reconstructed = np.dot(transformed.T, V)                        
    residuals = curr_frame - reconstructed     
    return residuals, V.shape[0], data_ref.shape[0]  


# Also used in pca_rdi_annular -------------------------------------------------
def get_eigenvectors(ncomp, data, svd_mode, noise_error=1e-3, max_evs=200, 
                     data_ref=None, debug=False):
    """ Choosing the size of the PCA truncation by Minimizing the residuals
    when ncomp set to None.
    """
    if data_ref is None:
        data_ref = data
    
    if ncomp is None:
        # Defines the number of PCs automatically for each zone (quadrant) by 
        # minimizing the pixel noise (as the pixel STDDEV of the residuals) 
        # decay once per zone         
        ncomp = 0              
        #full_std = np.std(data, axis=0).mean()
        #full_var = np.var(data, axis=0).sum()
        #orig_px_noise = np.mean(np.std(data, axis=1))
        px_noise = []
        px_noise_decay = 1
        # The eigenvectors (SVD/PCA) are obtained once    
        V_big = svd_wrapper(data_ref, svd_mode, min(data_ref.shape[0], max_evs),
                            False, False)
        # noise (stddev of residuals) to be lower than a given thr              
        while px_noise_decay >= noise_error:
            ncomp += 1
            V = V_big[:ncomp]
            transformed = np.dot(data, V.T)
            reconstructed = np.dot(transformed, V)                  
            residuals = data - reconstructed  
            px_noise.append(np.std((residuals)))         
            if ncomp>1: px_noise_decay = px_noise[-2] - px_noise[-1]
            #print 'ncomp {:} {:.4f} {:.4f}'.format(ncomp,px_noise[-1],px_noise_decay)
        
        if debug: print('ncomp', ncomp)
        
    else:
        # Performing SVD/PCA according to "svd_mode" flag
        V = svd_wrapper(data_ref, svd_mode, ncomp, debug=False, verbose=False)   
        
    return V


