#! /usr/bin/env python

"""
Module with contrast curve generation function.
"""

from __future__ import division, print_function

__author__ = 'C. Gomez, O. Absil @ ULg'
__all__ = ['contrast_curve',
           'noise_per_annulus',
           'throughput',
           'aperture_flux']

import numpy as np
import pandas as pd
import photutils
import inspect
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import stats
from scipy.signal import savgol_filter
from skimage.draw import circle
from matplotlib import pyplot as plt
from .fakecomp import inject_fcs_cube, inject_fc_frame, psf_norm
from ..conf import time_ini, timing, sep
from ..var import frame_center, dist



def contrast_curve(cube, angle_list, psf_template, fwhm, pxscale, starphot,
                   algo, sigma=5, nbranch=1, theta=0, inner_rad=1, wedge=(0,360),
                   fc_snr=10.0, student=True, transmission=None, smooth=True,
                   plot=True, dpi=100, imlib='opencv', debug=False, verbose=True,
                   save_plot=None, object_name=None, frame_size=None,
                   fix_y_lim=(), figsize=(8,4), **algo_dict):
    """ Computes the contrast curve for a given SIGMA (*sigma*) level. The
    contrast is calculated as sigma*noise/throughput. This implementation takes
    into account the small sample statistics correction proposed in Mawet et al.
    2014.

    Parameters
    ----------
    cube : array_like
        The input cube without fake companions.
    angle_list : array_like
        Vector with the parallactic angles.
    psf_template : array_like
        Frame with the psf template for the fake companion(s).
        PSF must be centered in array. Normalization is done internally.
    fwhm : float
        FWHM in pixels.
    pxscale : float
        Plate scale or pixel scale of the instrument.
    starphot : int or float or 1d array
        If int or float it corresponds to the aperture photometry of the
        non-coronagraphic PSF which we use to scale the contrast. If a vector
        is given it must contain the photometry correction for each frame.
    algo : callable or function
        The post-processing algorithm, e.g. vip.pca.pca.
    sigma : int
        Sigma level for contrast calculation.
    nbranch : int optional
        Number of branches on which to inject fakes companions. Each branch
        is tested individually.
    theta : float, optional
        Angle in degrees for rotating the position of the first branch that by
        default is located at zero degrees. Theta counts counterclockwise from
        the positive x axis. When working on a wedge, make sure that theta is
        located inside of it.
    inner_rad : int, optional
        Innermost radial distance to be considered in terms of FWHM.
    wedge : tuple of floats, optional
        Initial and Final angles for using a wedge. For example (-90,90) only
        considers the right side of an image.
    fc_snr: float optional
        Signal to noise ratio of injected fake companions
    student : {True, False}, bool optional
        If True uses Student t correction to inject fake companion.
    transmission : tuple of 2 1d arrays, optional
        If not None, then the tuple contains a vector with the factors to be
        applied to the sensitivity and a vector of the radial distances [px]
        where it is sampled (in this order).
    smooth : {True, False}, bool optional
        If True the radial noise curve is smoothed with a Savitzky-Golay filter
        of order 2.
    plot : {True, False}, bool optional
        Whether to plot the final contrast curve or not. True by default.
    dpi : int optional
        Dots per inch for the plots. 100 by default. 300 for printing quality.
    imlib : {'opencv', 'ndimage-fourier', 'ndimage-interp'}, string optional
        Library or method used for image operations (shifts). Opencv is the
        default for being the fastest.
    debug : {False, True}, bool optional
        Whether to print and plot additional info such as the noise, throughput,
        the contrast curve with different X axis and the delta magnitude instead
        of contrast.
    verbose : {True, False, 0, 1, 2} optional
        If True or 1 the function prints to stdout intermediate info and timing,
        if set to 2 more output will be shown. 
    save_plot: string
        If provided, the contrast curve will be saved to this path.
    object_name: string
        Target name, used in the plot title
    frame_size: int
        Frame size used for generating the contrast curve, used in the plot title
    fix_y_lim: tuple
        If provided, the y axis limits will be fixed, for easier comparison between plots
    **algo_dict
        Any other valid parameter of the post-processing algorithms can be
        passed here.

    Returns
    -------
    datafr : pandas dataframe
        Dataframe containing the sensitivity (Gaussian and Student corrected if
        Student parameter is True), the interpolated throughput, the distance in
        pixels, the noise and the sigma corrected (if Student is True).
    """
    if not cube.ndim == 3:
        raise TypeError('The input array is not a cube')
    if not cube.shape[0] == angle_list.shape[0]:
        raise TypeError('Input vector or parallactic angles has wrong length')
    if not psf_template.ndim==2:
        raise TypeError('Template PSF is not a frame')
    if transmission is not None:
        if not isinstance(transmission, tuple) or not len(transmission)==2:
            raise TypeError('transmission must be a tuple with 2 1d vectors')
    if isinstance(starphot, float) or isinstance(starphot, int):  pass
    else:
        if not starphot.shape[0] == cube.shape[0]:
            raise TypeError('Correction vector has bad size')
        cube = cube.copy()
        for i in range(cube.shape[0]):
            cube[i] = cube[i] / starphot[i]

    if verbose:
        start_time = time_ini()
        if isinstance(starphot, float) or isinstance(starphot, int):
            msg0 = 'ALGO : {}, FWHM = {}, # BRANCHES = {}, SIGMA = {},'
            msg0 += ' STARPHOT = {}'
            print(msg0.format(algo.func_name, fwhm, nbranch, sigma, starphot))
        else:
            msg0 = 'ALGO : {}, FWHM = {}, # BRANCHES = {}, SIGMA = {}'
            print(msg0.format(algo.func_name, fwhm, nbranch, sigma))
        print(sep)

    # throughput
    verbose_thru = False
    if verbose==2:  verbose_thru = True
    res_throug = throughput(cube, angle_list, psf_template, fwhm, pxscale,
                            nbranch=nbranch, theta=theta, inner_rad=inner_rad,
                            wedge=wedge, fc_snr=fc_snr, full_output=True, algo=algo,
                            imlib=imlib, verbose=verbose_thru, **algo_dict)
    vector_radd = res_throug[2]
    if res_throug[0].shape[0]>1:  thruput_mean = np.mean(res_throug[0], axis=0)
    else:  thruput_mean = res_throug[0][0]
    frame_nofc = res_throug[5]

    if verbose:
        print('Finished the throughput calculation')
        timing(start_time)

    if thruput_mean[-1]==0:
        thruput_mean = thruput_mean[:-1]
        vector_radd = vector_radd[:-1]

    # noise measured in the empty PP-frame with better sampling, every px
    # starting from 1*FWHM
    noise_samp, rad_samp = noise_per_annulus(frame_nofc, separation=1, fwhm=fwhm,
                                             init_rad=fwhm, wedge=wedge)
    cutin1 = np.where(rad_samp.astype(int)==vector_radd.astype(int).min())[0][0]
    noise_samp = noise_samp[cutin1:]
    rad_samp = rad_samp[cutin1:]
    cutin2 = np.where(rad_samp.astype(int)==vector_radd.astype(int).max())[0][0]
    noise_samp = noise_samp[:cutin2+1]
    rad_samp = rad_samp[:cutin2+1]

    # interpolating the throughput vector, spline order 2
    f = InterpolatedUnivariateSpline(vector_radd, thruput_mean, k=2)
    thruput_interp = f(rad_samp)

    # interpolating the transmission vector, spline order 1
    if transmission is not None:
        trans = transmission[0]
        radvec_trans = transmission[1]
        f2 = InterpolatedUnivariateSpline(radvec_trans, trans, k=1)
        trans_interp = f2(rad_samp)
        thruput_interp *= trans_interp

    if smooth:
        # smoothing the noise vector using a Savitzky-Golay filter
        win = min(noise_samp.shape[0]-2,int(2*fwhm))
        if win%2==0.:  win += 1
        noise_samp_sm = savgol_filter(noise_samp, polyorder=2, mode='nearest',
                                      window_length=win)
    else:
        noise_samp_sm = noise_samp

    if debug:
        plt.rc("savefig", dpi=dpi)
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(vector_radd*pxscale, thruput_mean, '.', label='computed',
                 alpha=0.6)
        plt.plot(rad_samp*pxscale, thruput_interp, ',-', label='interpolated',
                 lw=2, alpha=0.5)
        plt.grid('on', which='both', alpha=0.2, linestyle='solid')
        plt.xlabel('Angular separation [arcsec]')
        plt.ylabel('Throughput')
        plt.legend(loc='best')
        plt.xlim(0, np.max(rad_samp*pxscale))

        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(rad_samp*pxscale, noise_samp, '.', label='computed', alpha=0.6)
        plt.plot(rad_samp*pxscale, noise_samp_sm, ',-', label='noise smoothed',
                 lw=2, alpha=0.5)
        plt.grid('on', alpha=0.2, linestyle='solid')
        plt.xlabel('Angular separation [arcsec]')
        plt.ylabel('Noise')
        plt.legend(loc='best')
        #plt.yscale('log')
        plt.xlim(0, np.max(rad_samp*pxscale))

    # calculating the contrast
    if isinstance(starphot, float) or isinstance(starphot, int):
        cont_curve_samp = ((sigma * noise_samp_sm)/thruput_interp)/starphot
    else:
        cont_curve_samp = ((sigma * noise_samp_sm)/thruput_interp)
    cont_curve_samp[np.where(cont_curve_samp<0)] = 1
    cont_curve_samp[np.where(cont_curve_samp>1)] = 1

    # calculating the Student corrected contrast
    if student:
        n_res_els = np.floor(rad_samp/fwhm*2*np.pi)
        ss_corr = np.sqrt(1 + 1/(n_res_els-1))
        sigma_corr = stats.t.ppf(stats.norm.cdf(sigma), n_res_els)*ss_corr
        if isinstance(starphot, float) or isinstance(starphot, int):
            cont_curve_samp_corr = ((sigma_corr * noise_samp_sm)/thruput_interp)/starphot
        else:
            cont_curve_samp_corr = ((sigma_corr * noise_samp_sm)/thruput_interp)
        cont_curve_samp_corr[np.where(cont_curve_samp_corr<0)] = 1
        cont_curve_samp_corr[np.where(cont_curve_samp_corr>1)] = 1

    # plotting
    if plot or debug:
        if student:
            label = ['Sensitivity (Gaussian)',
                     'Sensitivity (Student-t correction)']
        else:  label = ['Sensitivity (Gaussian)']

        plt.rc("savefig", dpi=dpi)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax1 = fig.add_subplot(111)
        con1, = ax1.plot(rad_samp*pxscale, cont_curve_samp, '-',
                         alpha=0.2, lw=2, color='green')
        con2, = ax1.plot(rad_samp*pxscale, cont_curve_samp, '.',
                         alpha=0.2, color='green')
        if student:
            con3, = ax1.plot(rad_samp*pxscale, cont_curve_samp_corr, '-',
                             alpha=0.4, lw=2, color='blue')
            con4, = ax1.plot(rad_samp*pxscale, cont_curve_samp_corr, '.',
                             alpha=0.4, color='blue')
            lege = [(con1, con2), (con3, con4)]
        else:
            lege = [(con1, con2)]
        plt.legend(lege, label, fancybox=True, fontsize='medium')
        plt.xlabel('Angular separation [arcsec]')
        plt.ylabel(str(sigma)+' sigma contrast')
        plt.grid('on', which='both', alpha=0.2, linestyle='solid')
        ax1.set_yscale('log')
        ax1.set_xlim(0, np.max(rad_samp*pxscale))

        # Give a title to the contrast curve plot
        if object_name != None and frame_size != None:
            # Retrieve ncomp and pca_type info to use in title
            ncomp = algo_dict['ncomp']
            if algo_dict['cube_ref'] == None:
                pca_type = 'ADI'
            else:
                pca_type = 'RDI'
            plt.title(pca_type+' '+object_name+' '+str(ncomp)+'pc '+str(frame_size)+'+'+str(inner_rad),
                      fontsize = 14)

        # Option to fix the y-limit
        if len(fix_y_lim) == 2:
            min_y_lim = min(fix_y_lim[0], fix_y_lim[1])
            max_y_lim = max(fix_y_lim[0], fix_y_lim[1])
            ax1.set_ylim(min_y_lim, max_y_lim)

        # Optionally, save the figure to a path
        if save_plot != None:
            fig.savefig(save_plot, dpi=100)
            
        if debug:
            fig2 = plt.figure(figsize=figsize, dpi=dpi)
            ax3 = fig2.add_subplot(111)
            cc_mags = -2.5*np.log10(cont_curve_samp)
            con4, = ax3.plot(rad_samp*pxscale, cc_mags, '-',
                             alpha=0.2, lw=2, color='green')
            con5, = ax3.plot(rad_samp*pxscale, cc_mags, '.', alpha=0.2,
                             color='green')
            if student:
                cc_mags_corr = -2.5*np.log10(cont_curve_samp_corr)
                con6, = ax3.plot(rad_samp*pxscale, cc_mags_corr, '-',
                                 alpha=0.4, lw=2, color='blue')
                con7, = ax3.plot(rad_samp*pxscale, cc_mags_corr, '.',
                                 alpha=0.4, color='blue')
                lege = [(con4, con5), (con6, con7)]
            else:
                lege = [(con4, con5)]
            plt.legend(lege, label, fancybox=True, fontsize='medium')
            plt.xlabel('Angular separation [arcsec]')
            plt.ylabel('Delta magnitude')
            plt.gca().invert_yaxis()
            plt.grid('on', which='both', alpha=0.2, linestyle='solid')
            ax3.set_xlim(0, np.max(rad_samp*pxscale))
            ax4 = ax3.twiny()
            ax4.set_xlabel('Distance [pixels]')
            ax4.plot(rad_samp, cc_mags, '', alpha=0.)
            ax4.set_xlim(0, np.max(rad_samp))

    if student:
        datafr = pd.DataFrame({'sensitivity (Gauss)': cont_curve_samp,
                               'sensitivity (Student)':cont_curve_samp_corr,
                               'throughput': thruput_interp,
                               'distance': rad_samp, 'noise': noise_samp_sm,
                               'sigma corr':sigma_corr})
    else:
        datafr = pd.DataFrame({'sensitivity (Gauss)': cont_curve_samp,
                               'throughput': thruput_interp,
                               'distance': rad_samp, 'noise': noise_samp_sm})
    return datafr


def throughput(cube, angle_list, psf_template, fwhm, pxscale, algo, nbranch=1,
               theta=0, inner_rad=1, fc_rad_sep=3, wedge=(0,360), fc_snr=10.0,
               full_output=False, imlib='opencv', verbose=True, **algo_dict):
    """ Measures the throughput for chosen algorithm and input dataset. The
    final throughput is the average of the same procedure measured in *nbranch*
    azimutally equidistant branches.

    Parameters
    ----------
    cube : array_like
        The input cube without fake companions.
    angle_list : array_like
        Vector with the parallactic angles.
    psf_template : array_like
        Frame with the psf template for the fake companion(s).
        PSF must be centered in array. Normalization is done internally.
    fwhm : float
        FWHM in pixels.
    pxscale : float
        Plate scale in arcsec/px.
    algo : callable or function
        The post-processing algorithm, e.g. vip.pca.pca. Third party Python
        algorithms can be plugged here. They must have the parameters: 'cube',
        'angle_list' and 'verbose'. Optionally a wrapper function can be used.
    nbranch : int optional
        Number of branches on which to inject fakes companions. Each branch
        is tested individually.
    theta : float, optional
        Angle in degrees for rotating the position of the first branch that by
        default is located at zero degrees. Theta counts counterclockwise from
        the positive x axis.
    inner_rad : int, optional
        Innermost radial distance to be considered in terms of FWHM.
    fc_rad_sep : int optional
        Radial separation between the injected companions (in each of the
        patterns) in FWHM. Must be large enough to avoid overlapping. With the
        maximum possible value, a single fake companion will be injected per
        cube and algorithm post-processing (which greatly affects computation
        time).
    wedge : tuple of floats, optional
        Initial and Final angles for using a wedge. For example (-90,90) only
        considers the right side of an image.
    fc_snr: float optional
        Signal to noise ratio of injected fake companions
    full_output : {False, True}, bool optional
        If True returns intermediate arrays.
    imlib : {'opencv', 'ndimage-fourier', 'ndimage-interp'}, string optional
        Library or method used for image operations (shifts). Opencv is the
        default for being the fastest.
    verbose : {True, False}, bool optional
        If True prints out timing and information.
    **algo_dict
        Parameters of the post-processing algorithms must be passed here.

    Returns
    -------
    thruput_arr : array_like
        2d array whose rows are the annulus-wise throughput values for each
        branch.
    vector_radd : array_like
        1d array with the distances in FWHM (the positions of the annuli).

    If full_output is True then the function returns: thruput_arr, noise,
    vector_radd, cube_fc_all, frame_fc_all, frame_nofc and fc_map_all.

    noise : array_like
        1d array with the noise per annulus.
    cube_fc_all : array_like
        4d array, with the 3 different pattern cubes with the injected fake
        companions.
    frame_fc_all : array_like
        3d array with the 3 frames of the 3 (patterns) processed cubes with
        companions.
    frame_nofc : array_like
        2d array, PCA processed frame without companions.
    fc_map_all : array_like
        3d array with 3 frames containing the position of the companions in the
        3 patterns.

    """
    array = cube
    parangles = angle_list

    if not array.ndim == 3:
        raise TypeError('The input array is not a cube')
    if not array.shape[0] == parangles.shape[0]:
        raise TypeError('Input vector or parallactic angles has wrong length')
    if not psf_template.ndim==2:
        raise TypeError('Template PSF is not a frame or 2d array')
    if not hasattr(algo, '__call__'):
        raise TypeError('Parameter *algo* must be a callable function')
    if not fc_rad_sep>=3 or not fc_rad_sep<=int((array.shape[1]/2.)/fwhm)-1:
        msg = 'Too large separation between companions in the radial patterns. '
        msg += 'Should lie between 3 and {:}'
        raise ValueError(msg.format(int((array.shape[1]/2.)/fwhm)-1))
    if not isinstance(inner_rad, int):
        raise TypeError('inner_rad must be an integer')
    angular_range = wedge[1]-wedge[0]
    if nbranch>1 and angular_range<360:
        msg = 'Only a single branch is allowed when working on a wedge'
        raise RuntimeError(msg)

    if verbose:  start_time = time_ini()
    #***************************************************************************
    # Compute noise in concentric annuli on the "empty frame"
    if 'cube' and 'angle_list' and 'verbose' in inspect.getargspec(algo).args:
        if 'fwhm' in inspect.getargspec(algo).args:
            frame_nofc = algo(cube=array, angle_list=parangles, fwhm=fwhm,
                              verbose=False, **algo_dict)
        else:
            frame_nofc = algo(array, angle_list=parangles, verbose=False,
                              **algo_dict)

    if verbose:
        msg1 = 'Cube without fake companions processed with {:}'
        print(msg1.format(algo.func_name))
        timing(start_time)

    noise, vector_radd = noise_per_annulus(frame_nofc, separation=fwhm,
                                           fwhm=fwhm, wedge=wedge)
    vector_radd = vector_radd[inner_rad-1:]
    noise = noise[inner_rad-1:]
    if verbose:
        print('Measured annulus-wise noise in resulting frame')
        timing(start_time)

    # We crop the PSF and check if PSF has been normalized (so that flux in
    # 1*FWHM aperture = 1) and fix if needed
    psf_template = psf_norm(psf_template, size=3*fwhm, fwhm=fwhm)

    #***************************************************************************
    # Initialize the fake companions
    angle_branch = angular_range / nbranch
    # signal-to-noise ratio of injected fake companions
    snr_level = fc_snr * np.ones_like(noise)
    
    thruput_arr = np.zeros((nbranch, noise.shape[0]))
    fc_map_all = np.zeros((nbranch*fc_rad_sep, array.shape[1], array.shape[2]))
    frame_fc_all = fc_map_all.copy()
    cube_fc_all = np.zeros((nbranch*fc_rad_sep, array.shape[0], array.shape[1],
                            array.shape[2]))
    cy, cx = frame_center(array[0])

    # each branch is computed separately
    for br in range(nbranch):
        # each pattern is computed separately. For each pattern the companions
        # are separated by "fc_rad_sep * fwhm", interleaving the injections
        for irad in range(fc_rad_sep):
            radvec = vector_radd[irad::fc_rad_sep]
            cube_fc = array.copy()
            # filling map with small numbers
            fc_map = np.ones_like(array[0]) * 1e-6
            fcy = []; fcx = []
            for i in range(radvec.shape[0]):
                flux = snr_level[irad+i*fc_rad_sep] * noise[irad+i*fc_rad_sep]
                cube_fc = inject_fcs_cube(cube_fc, psf_template, parangles, flux,
                                          pxscale, rad_dists=[radvec[i]],
                                          theta=br*angle_branch + theta, imlib=imlib,
                                          verbose=False)
                y = cy + radvec[i] * np.sin(np.deg2rad(br*angle_branch + theta))
                x = cx + radvec[i] * np.cos(np.deg2rad(br*angle_branch + theta))
                fc_map = inject_fc_frame(fc_map, psf_template, y, x, flux)
                fcy.append(y); fcx.append(x)

            if verbose:
                msg2 = 'Fake companions injected in branch {:} (pattern {:}/{:})'
                print(msg2.format(br+1, irad+1, fc_rad_sep))
                timing(start_time)

            #*******************************************************************
            if 'cube' and 'angle_list' and 'verbose' in inspect.getargspec(algo).args:
                if 'fwhm' in inspect.getargspec(algo).args:
                    frame_fc = algo(cube=cube_fc, angle_list=parangles,
                                    fwhm=fwhm, verbose=False, **algo_dict)
                else:
                    frame_fc = algo(cube=cube_fc, angle_list=parangles,
                                    verbose=False, **algo_dict)

            if verbose:
                msg3 = 'Cube with fake companions processed with {:}'
                msg3 += '\nMeasuring its annulus-wise throughput'
                print(msg3.format(algo.func_name))
                timing(start_time)

            #*******************************************************************
            injected_flux = aperture_flux(fc_map, fcy, fcx, fwhm, ap_factor=1,
                                          mean=False, verbose=False)
            recovered_flux = aperture_flux((frame_fc - frame_nofc), fcy, fcx,
                                           fwhm, ap_factor=1, mean=False,
                                           verbose=False)
            thruput = (recovered_flux)/injected_flux
            thruput[np.where(thruput<0)] = 0

            thruput_arr[br, irad::fc_rad_sep] = thruput
            fc_map_all[br*fc_rad_sep+irad, :, :] = fc_map
            frame_fc_all[br*fc_rad_sep+irad, :, :] = frame_fc
            cube_fc_all[br*fc_rad_sep+irad, :, :, :] = cube_fc

    if verbose:
        print('Finished measuring the throughput in {:} branches'.format(nbranch))
        timing(start_time)

    if full_output:
        return (thruput_arr, noise, vector_radd, cube_fc_all, frame_fc_all,
                frame_nofc, fc_map_all)
    else:
        return thruput_arr, vector_radd



def noise_per_annulus(array, separation, fwhm, init_rad=None, wedge=(0,360),
                      verbose=False, debug=False):
    """ Measures the noise as the standard deviation of apertures defined in
    each annulus with a given separation.

    Parameters
    ----------
    array : array_like
        Input frame.
    separation : float
        Separation in pixels of the centers of the annuli measured from the
        center of the frame.
    fwhm : float
        FWHM in pixels.
    init_rad : float
        Initial radial distance to be used. If None then the init_rad = FWHM.
    wedge : tuple of floats, optional
        Initial and Final angles for using a wedge. For example (-90,90) only
        considers the right side of an image. Be careful when using small
        wedges, this leads to computing a standard deviation of very small
        samples (<10 values).
    verbose : {False, True}, bool optional
        If True prints information.
    debug : {False, True}, bool optional
        If True plots the positioning of the apertures.

    Returns
    -------
    noise : array_like
        Vector with the noise value per annulus.
    vector_radd : array_like
        Vector with the radial distances values.

    """
    def find_coords(rad, sep, init_angle, fin_angle):
        angular_range = fin_angle-init_angle
        npoints = (np.deg2rad(angular_range)*rad)/sep   #(2*np.pi*rad)/sep
        ang_step = angular_range/npoints   #360/npoints
        x = []
        y = []
        for i in range(int(npoints)):
            newx = rad * np.cos(np.deg2rad(ang_step * i + init_angle))
            newy = rad * np.sin(np.deg2rad(ang_step * i + init_angle))
            x.append(newx)
            y.append(newy)
        return np.array(y), np.array(x)
    #___________________________________________________________________

    if not array.ndim==2:
        raise TypeError('Input array is not a frame or 2d array')
    if not isinstance(wedge, tuple):
        raise TypeError('Wedge must be a tuple with the initial and final angles')

    init_angle, fin_angle = wedge
    centery, centerx = frame_center(array)
    n_annuli = int(np.floor((centery)/separation))

    x = centerx
    y = centery
    noise = []
    vector_radd = []
    if verbose:  print('{} annuli'.format(n_annuli-1))

    if init_rad is None:  init_rad = fwhm

    if debug:
        _, ax = plt.subplots(figsize=(6,6))
        ax.imshow(array, origin='lower', interpolation='nearest',
                  alpha=0.5, cmap='gray')

    for i in range(n_annuli-1):
        y = centery + init_rad + separation*(i)
        rad = dist(centery, centerx, y, x)
        yy, xx = find_coords(rad, fwhm, init_angle, fin_angle)
        yy += centery
        xx += centerx

        apertures = photutils.CircularAperture((xx, yy), fwhm/2.)
        fluxes = photutils.aperture_photometry(array, apertures)
        fluxes = np.array(fluxes['aperture_sum'])

        noise_ann = np.std(fluxes)
        noise.append(noise_ann)
        vector_radd.append(rad)

        if debug:
            for i in range(xx.shape[0]):
                # Circle takes coordinates as (X,Y)
                aper = plt.Circle((xx[i], yy[i]), radius=fwhm/2., color='r',
                              fill=False, alpha=0.8)
                ax.add_patch(aper)
                cent = plt.Circle((xx[i], yy[i]), radius=0.8, color='r',
                              fill=True, alpha=0.5)
                ax.add_patch(cent)

        if verbose:
            print('Radius(px) = {:}, Noise = {:.3f} '.format(rad, noise_ann))

    return np.array(noise), np.array(vector_radd)



def aperture_flux(array, yc, xc, fwhm, ap_factor=1, mean=False, verbose=False):
    """ Returns the sum of pixel values in a circular aperture centered on the
    input coordinates. The radius of the aperture is set as (ap_factor*fwhm)/2.

    Parameters
    ----------
    array : array_like
        Input frame.
    yc, xc : list or 1d arrays
        List of y and x coordinates of sources.
    fwhm : float
        FWHM in pixels.
    ap_factor : int, optional
        Diameter of aperture in terms of the FWHM.

    Returns
    -------
    flux : list of floats
        List of fluxes.

    Note
    ----
    From Photutils documentation, the aperture photometry defines the aperture
    using one of 3 methods:

    'center': A pixel is considered to be entirely in or out of the aperture
              depending on whether its center is in or out of the aperture.
    'subpixel': A pixel is divided into subpixels and the center of each
                subpixel is tested (as above).
    'exact': (default) The exact overlap between the aperture and each pixel is
             calculated.

    """
    n_obj = len(yc)
    flux = np.zeros((n_obj))
    for i, (y, x) in enumerate(zip(yc, xc)):
        if mean:
            ind = circle(y, x,  (ap_factor*fwhm)/2.)
            values = array[ind]
            obj_flux = np.mean(values)
        else:
            aper = photutils.CircularAperture((x, y), (ap_factor*fwhm)/2.)
            obj_flux = photutils.aperture_photometry(array, aper, method='exact')
            obj_flux = np.array(obj_flux['aperture_sum'])
        flux[i] = obj_flux

        if verbose:
            print('Coordinates of object {:} : ({:},{:})'.format(i, y, x))
            print('Object Flux = {:.2f}'.format(flux[i]))

    return flux
