import numpy as np
from scipy.interpolate import interp1d
from fit_blackbody import get_filter
import astropy.units as u
from astropy.table import Table
from scipy.ndimage.filters import gaussian_filter1d
from pysynphot import observation
from pysynphot import spectrum


def filter_flux(tspectrum, filtname, minusepoints=50, invert=False):
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
    if minusepoints > 1.5 * len(tspectrum):
        usepoints = np.linspace(np.min(tspectrum['lambda'].to('Angstrom')),
                                np.max(tspectrum['lambda'].to('Angstrom')),
                                num=minusepoints)
    else:
        usepoints = tspectrum['wavelength'].to('Angstrom')

    if invert:
        ffilt_flux = np.vectorize(lambda wavell: fflux(wavell) / ffilt(wavell))
        ffilt_fluxerr = np.vectorize(lambda wavell: ffluxerr(wavell) /
                                     ffilt(wavell))
    else:
        ffilt_flux = np.vectorize(lambda wavell: fflux(wavell) * ffilt(wavell))
        ffilt_fluxerr = np.vectorize(lambda wavell: ffluxerr(wavell) *
                                     ffilt(wavell))
    tspectrum['flux'] = ffilt_flux(usepoints.to('Angstrom').value) * \
                        u.erg/(u.cm**2*u.s*u.Angstrom)
    tspectrum['fluxerr'] = ffilt_fluxerr(usepoints.to('Angstrom').value) * \
                           u.erg/(u.cm**2*u.s*u.Angstrom)

    return tspectrum


def get_spectrum(tspectrum, lambdamin, lambdamax, presolv, mode='number'):
    '''Give a (highres) spectrum, lambdamin/max stop and resolving power.
    Returns the integrated spectrum.
    Set mode='number' to say presolv is the number of resolution elements or
    to 'rpower' to say it is the resolving power which leads to a logarithmic
    distribution of the bins.'''
    if mode == 'rpower':
        wlgrid = make_spectral_grid(lambdamin.to('Angstrom'),
                                    lambdamax.to('Angstrom'),
                                    presolv)
    elif mode == 'number':
        wlgrid = np.linspace(lambdamin.to('Angstrom').value,
                             lambdamax.to('Angstrom').value,
                             presolv+1) * u.Angstrom
    else:
        raise ValueError("Set mode='number' to say presolv is the number \
of resolution elements or \
to 'rpower' to say it is the resolving power which leads to a logarithmic \
distribution of the bins.")
    
    wlgrid = wlgrid.to('Angstrom')
    # define the following function. Outside of the values, NAN is returned
    fflux = interp1d(tspectrum['wavelength'].to('Angstrom'),
                     tspectrum['flux'].to('erg/(cm2*s*Angstrom)'),
                     bounds_error=False)
    ffluxerr = interp1d(tspectrum['wavelength'].to('Angstrom'),
                        tspectrum['fluxerr'].to('erg/(cm2*s*Angstrom)'),
                        bounds_error=False)
    central_wls = np.diff(wlgrid)/2. + wlgrid[:-1]
    fluxes = []
    fluxerrs = []
    for iwl, cwl in enumerate(central_wls):
        usepoints = np.hstack([wlgrid[iwl],
                               tspectrum['wavelength'][np.where(np.logical_and(
                                   tspectrum['wavelength'].to('Angstrom') > wlgrid[iwl],
                                   tspectrum['wavelength'].to('Angstrom') < wlgrid[iwl+1]))[0]].to(
                                       'Angstrom'),
                               wlgrid[iwl+1]])
        usepoints = np.array(usepoints) * u.Angstrom

        fluxes.append(np.trapz(np.vectorize(fflux)([usepoints]),
                               usepoints) /
                      (wlgrid[iwl+1]-wlgrid[iwl]).to('Angstrom'))

        fluxerrs.append(np.trapz(np.vectorize(ffluxerr)([usepoints]),
                                 usepoints) /
                        (wlgrid[iwl+1]-wlgrid[iwl]).to('Angstrom'))
    fluxes = (fluxes * fluxes[0].unit) * u.erg/(u.cm**2*u.s*u.Angstrom)
    fluxerrs = (fluxerrs * fluxerrs[0].unit) * u.erg/(u.cm**2*u.s*u.Angstrom)
    return Table([central_wls, np.diff(wlgrid), fluxes, fluxerrs],
                 names=('central_wavelength', 'binwidth', 'flux', 'fluxerrs'))
    
def make_spectral_grid(start, stop, presolv):
    '''Make an logarithmicly equidistant grid of the spectral points'''
    num = int(np.ceil(np.log10(stop/start) / np.log10(presolv**-1 + 1)))
    wavel = []
    for inum in range(num):
        wavel.append((presolv**-1 + 1)**inum * start)
    return wavel * wavel[0].unit


def binary_filter(val, low, up):
    if val >= low and\
       val <= up:
        return 1
    else:
        return 0

def fold_with_gauss(delta_lambda, tspectrum):
    '''function requires a table with 3 columns: wavelength, flux, fluxerr and a
    delta_lambda which is the sigma of the gaussian it is folded with'''
    delta_lambda = delta_lambda.to(tspectrum['wavelength'].unit)
    # test if linearization is needed. Doing this by comparing first and last
    # stepsize
    if abs((tspectrum['wavelength'][1]  - tspectrum['wavelength'][0]) /
           (tspectrum['wavelength'][-1] - tspectrum['wavelength'][-2]) - 1) \
                                                                   > 0.01:
        print('Linearizing the spectrum first, as it seems not to be linear.')
        units = [tspectrum['wavelength'].unit,
                 tspectrum['flux'].unit,
                 tspectrum['fluxerr'].unit]
        binwidth = delta_lambda/10.
        newwavel = np.arange(np.min(tspectrum['wavelength']),
                             np.max(tspectrum['wavelength']),
                             binwidth.to(tspectrum['wavelength'].unit).value)
        newflux = rebin_spec(tspectrum['wavelength'],
                             tspectrum['flux'],
                             newwavel, keepneg=False)
        newfluxerr = rebin_spec(tspectrum['wavelength'],
                                tspectrum['fluxerr'],
                                newwavel, keepneg=False)
        tspectrum = Table([newwavel, newflux, newfluxerr],
                          names=('wavelength', 'flux', 'fluxerr'))
        tspectrum['wavelength'].unit = units[0]
        tspectrum['flux'].unit = units[1]
        tspectrum['fluxerr'].unit = units[2]
        del units
    else:
        print('The spectrum seems linear already. Not changing it.')
        halfnbins = np.int(np.round(len(tspectrum['wavelength'])/2))
        binwidth = (tspectrum['wavelength'][halfnbins] -
                    tspectrum['wavelength'][halfnbins-1]) \
                    * tspectrum['wavelength'].unit
    # divide by binwidth to convert to pixel units and by 2.355 to convert
    # from FWHM to sigma
    tspectrum['flux'] = gaussian_filter1d(tspectrum['flux'],
                                          delta_lambda / binwidth / 2.355,
                                          mode='reflect') * \
                                          tspectrum['flux'].unit
    tspectrum['fluxerr'] = gaussian_filter1d(tspectrum['fluxerr'],
                                             delta_lambda / binwidth / 2.355,
                                             mode='reflect') * \
                                             tspectrum['fluxerr'].unit
    return tspectrum
    
    
def rebin_spec(wavelength, flux, waveout, keepneg=False):
    spec = spectrum.ArraySourceSpectrum(wave=wavelength, flux=flux,
                                        keepneg=keepneg)
    f = np.ones(len(flux))
    filt = spectrum.ArraySpectralElement(wavelength.to(wavelength.unit).value,
                                         f, waveunits=str(wavelength.unit))
    obs = observation.Observation(spec, filt, binset=waveout, force='taper')

    return obs.binflux
