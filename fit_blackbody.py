import numpy as np
from astropy.modeling.blackbody import blackbody_lambda as bbl
import astropy.units as u
from scipy import integrate
from scipy.optimize import least_squares
from astropy.table import Table
import os
import pickle


# give some properties
_pnfilt = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       'VOSA_filtercurves')  # pn for filters


def fit_flux2temp(fluxes, filtnames, nusepoints=15):
    '''Give lists for fluxes, fluxerrs and filtnames for the points
    you want to fit a blackbody to. It returns the temperature in K'''
    # first get the integrated fluxes of the magnitude bands
    fluxints, fluxerrs = [], []
    for ii in range(len(filtnames)):
        tfilt = get_filter(filtnames[ii])    # make the points where to
        # evaluate the function. It uses the closest
        # filterpoints afterwards
        usepoints = np.linspace(np.min(tfilt['lambda'].to('Angstrom')),
                                np.max(tfilt['lambda'].to('Angstrom')),
                                num=nusepoints)
        new_throughput = np.full(nusepoints, np.nan)
        for ipoint in range(nusepoints):
            new_throughput[ipoint] = tfilt['transmis'][
                find_nearest(tfilt['lambda'],
                             usepoints[ipoint])]
        fluxints.append(integrate_flux(fluxes[0][ii] * new_throughput,
                                       usepoints))
        fluxerrs.append(integrate_flux(fluxes[1][ii] * new_throughput,
                                       usepoints))
    fluxints = fluxints * fluxints[0].unit
    fluxerrs = fluxerrs * fluxerrs[0].unit
    temparea_init = np.array([1000., 1e-17])
    value_changed = False
    while not value_changed:
        fitres = least_squares(residuals, temparea_init,
                               args=[[fluxints, fluxerrs, filtnames,
                                      nusepoints]],
        # use bounds for method trm # bounds=[[0, 1e-40], [20000, 1e-5]],
                               method='lm')
        if np.any(fitres.x == temparea_init):
            print('Fit not worked. Changing initial values')
            temparea_init *= 1.1
        else:
            value_changed = True

    fittemparea = fitres.x
    return fittemparea


def residuals(Temparea, fluxfilts):
    '''The function which is being minimized: BB-flux and obs flux'''
    fluxes, fluxerrs, filtnames, nusepoints = fluxfilts
    skyarea = Temparea[1]*u.sr
    Temp = Temparea[0]*u.K
    bbfluxes = [skyarea * filtered_flux(Temp, filtname,
                                        nusepoints=nusepoints) for
                filtname in filtnames]
    bbfluxes *= bbfluxes[-1].unit

    resid = (bbfluxes - fluxes) / fluxerrs
    # print('Trying Temp = {}, skyarea = {}, resid = {}'.format(
    #    Temp, skyarea, resid))
    # print(bbfluxes, fluxes, bbfluxes-fluxes,resid, sep='\n')
    return resid


def filtered_flux(Temp, filtname, nusepoints=15):
    '''Return the integrated flux of the filter mutliplied with the
    transmission.'''
    tfilt = get_filter(filtname)
    # now integrate with the nearest transmission point
    vecbb_w_filter = np.vectorize(bb_w_filter, excluded=[1, 2])
    # make the points where to evaluate the function. It uses the closest
    # filterpoints afterwards
    usepoints = np.linspace(np.min(tfilt['lambda'].to('Angstrom')),
                            np.max(tfilt['lambda'].to('Angstrom')),
                            num=nusepoints)
    fluxdenses = vecbb_w_filter(usepoints.value,
                                Temp.to('K').value, tfilt)
    fluxdenses = fluxdenses * u.erg / u.cm**2 / u.s / u.Angstrom / u.sr
    flux = integrate_flux(fluxdenses, usepoints)

    return flux


def integrate_flux(fluxdenses, xpoints):
    '''Integrate the flux of a source. If there is only one value given,
    assume a constant fluxdensity.'''
    flux = integrate.trapz(fluxdenses,
                           x=xpoints)

    return flux


def get_filter(filtname):
    '''Reads and returns Astropy table with the filter transmission'''
    pfnfilt = os.path.join(_pnfilt,
                           filtname.replace('/','_') + '.dat')

    tfilt = Table(np.loadtxt(pfnfilt), names=['lambda', 'transmis'])
    tfilt['lambda'].unit = u.Angstrom

    return tfilt


def bb_w_filter(wavel, Temp, tfilt):
    if Temp < 0:
        Temp = 0. * u.K
    flux = bbl(wavel * u.Angstrom, Temp) *\
           tfilt['transmis'][find_nearest(tfilt['lambda'],
                                          wavel)]
    return flux.value


def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx


def loadfiltzeros(filtname):
    '''Load zero filters. If not existent ask for it and add it'''
    pfnfilt = os.path.join(_pnfilt, 'filter_zeropoints.p')
    if os.path.exists(pfnfilt):
        filterzeros = pickle.load(open(pfnfilt, 'rb'))
    else:
        filterzeros = {}

    if filtname in filterzeros.keys():
        zeroflux = filterzeros[filtname]
    elif filtname == 'all':
        zeroflux = filterzeros
    else:
        zeroflux = np.float(eval(input('Please give the Zero point of filter "{}"\
 in the Vega system in (erg/cm2/s/A)'.format(filtname))))
        zeroflux = zeroflux * u.erg / u.cm**2 / u.s / u.Angstrom
        filterzeros[filtname] = zeroflux
        pickle.dump(filterzeros, open(pfnfilt, 'wb'))
    return zeroflux


def mag2flux(mag, filtname):
    '''Load the zero magnitude from the dict file and convert it to
    flux'''
    zeroflux = loadfiltzeros(filtname)
    flux = zeroflux / 10**(mag/2.5)
    return flux


def flux2mag(flux, filtname, nusepoints=15):
    '''Load the zero magnitude from the dict file and convert it to
    flux'''
    zeroflux = loadfiltzeros(filtname)
    tfilt = get_filter(filtname)
    usepoints = np.linspace(np.min(tfilt['lambda'].to('Angstrom')),
                            np.max(tfilt['lambda'].to('Angstrom')),
                            num=nusepoints)
    new_throughput = np.full(nusepoints, np.nan)
    for ipoint in range(nusepoints):
        new_throughput[ipoint] = tfilt['transmis'][
            find_nearest(tfilt['lambda'],
                         usepoints[ipoint])]
        zerofluxint = integrate_flux(zeroflux * new_throughput,
                                     usepoints)
    mag = u.Magnitude((-2.5 * np.log10(flux/zerofluxint)).value)
    return mag
