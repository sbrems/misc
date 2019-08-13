import numpy as np
from gaussfitter.gaussfitter import gaussfit


def find_centers(cube, fitpoint,
                 fitmode='static', fwhm=7.):
    '''find the center of the speckle by fitting a gaussian at the
    roughly given position. If fitmode 'static' it always uses
    the same value. Chose 'circular to make it use the last value
    as guess for the next one'''
    cutrad = int(fwhm)  # half size of the frame
    if len(cube.shape) != 3:
        raise ValueError('Invalid data cube shape {}. Should be 3dim'.format(
            cube.shape))

    centers = np.full((cube.shape[0], 2), np.nan)
    sigmas = np.full((cube.shape[0]), np.nan)
    errors = np.full((cube.shape[0], 2), np.nan)
    fitims = np.full((cube.shape[0], 2 * cutrad + 1, 2 * cutrad + 1), np.nan)
    for iframe, frame in enumerate(cube):
        print('Fitting gauss to frame {} of {}'.format(
            iframe, cube.shape[0]), end='\r')
        fitim = frame[fitpoint[0] - cutrad: fitpoint[0] + cutrad + 1,
                      fitpoint[1] - cutrad: fitpoint[1] + cutrad + 1]
        fitims[iframe, ::] = fitim
        height_ini = np.median(fitim)
        amp_ini = np.max(fitim)
        params_ini = (height_ini, amp_ini, cutrad, cutrad,
                      fwhm, fwhm, 0)
        res, err = gaussfit(fitim,
                            params=params_ini,
                            rotation=False,
                            return_error=True)
        centers[iframe, :] = res[2:4] - cutrad + fitpoint
        sigmas[iframe] = res[4]
        errors[iframe, :] = err[2:4]
    print('Done fitting {} gaussians'.format(cube.shape[0]), end='\n')

    return centers, sigmas, errors, fitims
