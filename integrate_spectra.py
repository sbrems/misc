import numpy as np
from scipy.interpolate import interp1d
from fit_blackbody import get_filter
import astropy.units as u

def filter_flux(tspectrum, filtname, minusepoints=50):
    tfilt = get_filter(filtname)
    # replace invalid errors (nan, inf,..) by the maximum found value
    if np.sum(~np.isfinite(tspectrum['fluxerr'])) > 0:
        if np.sum(~np.isfinite(tspectrum['fluxerr'])) == len(tspectrum):
            print('Only invalid errors found. Assuming they are 0')
            tspectrum['fluxerr'] = 0.0
        else:
            print('Invalid values in error found. Replacing them by the maximum')
            tspectrum['fluxerr'][np.where(~np.isfinite(tspectrum['fluxerr']))] =\
                np.max(tspectrum['fluxerr'][np.where(
                    np.isfinite(tspectrum['fluxerr']))])
    ffilt = interp1d(tfilt['lambda'].to('Angstrom'),
                     tfilt['transmis'],
                     fill_value=0., bounds_error=False)
    fflux = interp1d(tspectrum['wavelength'].to('Angstrom'),
                     tspectrum['flux'].to('erg/(cm2*s*Angstrom)'))
    ffluxerr = interp1d(tspectrum['wavelength'].to('Angstrom'),
                        tspectrum['fluxerr'].to('erg/(cm2*s*Angstrom)'))
    # make sure you use enough points
    if minusepoints > len(tspectrum):
        usepoints = np.linspace(np.min(tspectrum['lambda'].to('Angstrom')),
                                np.max(tspectrum['lambda'].to('Angstrom')),
                                num=minusepoints)
    else:
        usepoints = tspectrum['wavelength'].to('Angstrom')

    ffilt_flux = np.vectorize(lambda wavell: fflux(wavell) * ffilt(wavell))
    tspectrum['flux'] = ffilt_flux(usepoints.to('Angstrom').value)
    ffilt_fluxerr = np.vectorize(lambda wavell: ffluxerr(wavell) *\
                                 ffilt(wavell))
    tspectrum['fluxerr'] = ffilt_fluxerr(usepoints.to('Angstrom').value)

    return tspectrum
