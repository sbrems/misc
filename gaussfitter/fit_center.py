import numpy as np
from gaussfitter.gaussfitter import gaussfit


def fit_center(imcube, fitpoint, fitmode='circular', fwhm=7.,
               circle=False):
    '''Find the center around fitpoint. If fitmode circular
    use last point as guess for new one. The radius is two FWHM.
    Enter and return in yx-coords.
    Circle=False
    Set to True to force the sigmas in x and y to be the same'''
    assert len(imcube.shape) == 3
    fitradius = int(np.floor(fwhm / 2.) * 4 + 1)  # make sure its odd
    centers = np.full((imcube.shape[0], 2), np.nan)
    sigmas = np.full((imcube.shape[0], 2), np.nan)
    errors = np.full((imcube.shape[0], 2), np.nan)

    for iim, image in enumerate(imcube):
        fitim = image[fitpoint[0] - fitradius:fitpoint[0] + fitradius,
                      fitpoint[1] - fitradius:fitpoint[1] + fitradius]
        amp_ini = np.max(fitim)
        height_ini = np.median(fitim)

        fitparams, parerr = gaussfit(
            fitim,
            params=(height_ini, amp_ini, fitradius, fitradius,
                    fwhm, fwhm, 0.),
            circle=circle,
            return_error=True)
        centers[iim, :] = fitparams[2:4].T + fitpoint - fitradius
        sigmas[iim, :] = fitparams[4:6].T
        errors[iim, :] = parerr[2:4].T

    return centers, sigmas, errors
