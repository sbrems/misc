import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from fit_blackbody import get_filter, integrate_flux
# import astropy.units as u


def filter_flux(tspectrum, filtname, minusepoints=50):
    tfilt = get_filter(filtname)
    # replace invalid errors (nan, inf,..) by the maximum found value
    if np.sum(~np.isfinite(tspectrum['fluxerr'])) > 0:
        if np.sum(~np.isfinite(tspectrum['fluxerr'])) == len(tspectrum):
            print('Only invalid errors found. Assuming they are 0')
            tspectrum['fluxerr'] = 0.0
        else:
            print(
                'Invalid values in error found. Replacing them by the maximum')
            tspectrum['fluxerr'][np.where(~np.isfinite(tspectrum['fluxerr']))] =\
                np.max(tspectrum['fluxerr'][np.where(
                    np.isfinite(tspectrum['fluxerr']))])
    ffilt = interp1d(tfilt['lambda'].to('Angstrom'),
                     tfilt['transmis'],
                     fill_value=0.,
                     bounds_error=False)
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
    tspectrum['flux'] = ffilt_flux(usepoints.to('Angstrom').value)*tspectrum['flux'].unit
    ffilt_fluxerr = np.vectorize(lambda wavell: ffluxerr(wavell) *
                                 ffilt(wavell))
    tspectrum['fluxerr'] = ffilt_fluxerr(usepoints.to('Angstrom').value)*tspectrum['fluxerr'].unit

    return tspectrum


def integrate_with_error(tspec, bounds=None):
    '''Integrate the spectrum and the errors. Bounds are
    the minimum and maximum wavelength (lam) of shape [min, max]'''
    # check values are sorted and bounds reasonable
    if bounds is None:
        bounds = (np.min(tspec['wavelength']).value, np.max(tspec['wavelength']).value)\
                 * tspec['wavelength'].unit
    bounds = bounds.to('Angstrom')
    assert bounds[0] >= tspec['wavelength'][0]*tspec['wavelength'].unit \
        and bounds[1] <= tspec['wavelength'][-1]*tspec['wavelength'].unit
    # if bounds not present, interpolate it. remove other values
    tfluxfilt = filter_flux(tspec.copy(), filtname='stepfunc_{}_{}'.format(
        bounds[0].to('Angstrom').value,
        bounds[1].to('Angstrom').value), )

    flux, fluxerr = integrate_flux(tfluxfilt['flux'].to('erg/(cm**2*s*Angstrom)').value,
                                   tfluxfilt['wavelength'].to('Angstrom').value,
                                   fluxerr=tfluxfilt['fluxerr'].to('erg/(cm**2*s*Angstrom)').value)
    return [flux, fluxerr]*tfluxfilt['flux'].unit
