import numpy as np
import copy
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from astropy.io import fits

from celestial_mechanics.proper_motion import calprowrapper
from spt_magnitude_conversion import split_spt
import fit_blackbody
from fit_blackbody import fit_flux2temp, flux2mag, mag2flux, integrate_flux
from bin_spectra import filter_flux, get_spectrum, fold_with_gauss


class Star():
    '''Save properties of a star. So far available:
    aliases ('auto' from Simbad)
    Coordinates (=['auto',] for Simbad query),
    coord_units: default: ['u.hourangle','u.deg'],
    default coord_frame ='icrs'
    Magnitudes (with the filterband, default filterband = None)
    SpT (also num), get_SpT tries to get it from Simbad
    aliases
    Magnitude bands
    storing and getting pm from simbad (='auto')
    Temperature detemination via a BB fit to these bands
    parallax and conversion to distance
    get_filtered_spectrum (set spectrum first)
    simbad_main_ID returns the Simbad main ID. Usefull to see if stars are aliases
    '''

    def __init__(self, sname,
                 coordinates=[None, [u.hourangle, u.deg], 'icrs'],
                 aliases=None, simbad_main_ID=None,
                 mag=None, filterband=None, magerror=None,
                 SpT=None, SpC=None, temperature=None,
                 mass=None, comp_sep=None, parallax=None,
                 pm=None):
        self.sname = sname
        self.SpT = SpT
        self.temperature = temperature
        self.skyarea = None  # the result when fitting a temperature
        self.mass = mass
        self.comp_sep = comp_sep
        self.parallax = parallax
        self.spectrum = None

        self._pm = pm
        self._aliases = aliases
        self._simbad_main_ID = simbad_main_ID
        self._coordinates = coordinates
        self._mag = mag
        self._filterband = filterband
        self._magerror = magerror
    _kind = 'star'

    def __str__(self):
        return "Starcass object  named {}".format(self.name)

    def SpT_num(self):
        return np.sum(split_spt(self.SpT)[0:2])

    def SpC_num(self):
        return split_spt(self.SpT)[2]

    def get_SpT(self):
        Simbad.add_votable_fields('sptype')
        self.SpT = (Simbad.query_object(self.sname)['SP_TYPE'][0]).decode('utf-8')

    def read_spectrum(self, fpath, wlunit=u.nm,
                      fluxunit=u.erg/u.cm**2/u.s/u.Angstrom):
        data = fits.getdata(fpath)
        print('Assuming dataformat lambda,flux,error with wl-unit \
{} and flux unit {}. Access via self.spectrum'.format(wlunit, fluxunit))
        assert data.shape[1] == 3
        tflux = Table([data[:, 0] * wlunit,
                       data[:, 1] * fluxunit,
                       data[:, 2] * fluxunit],
                      names=('wavelength', 'flux', 'fluxerr'),
                      meta={'origin': fpath})
        self.spectrum = tflux

    def get_filteredspectrum(self, filtname, minusepoints=50, tspectrum=None,
                             invert=False):
        '''Set invert=True to increase the flux, e.g. to get the original flux
        after it had been filtered'''
        if tspectrum is None:
            tspectrum = copy.copy(self.spectrum)
        return filter_flux(tspectrum,
                           filtname,
                           minusepoints=minusepoints, invert=invert)

    def smooth_spectrum(self, delta_lambda, tspectrum=None):
        '''Smooth the spectrum  folding with a gaussian
        assuming a constant delta_lambda (not resolution).
        If no table with a spectrum is given, self.spectrum is used.
        Tests the spectrum. If not linear, it is going to linearize it'''
        if tspectrum is None:
            tspectrum = copy.copy(self.spectrum)
        return fold_with_gauss(delta_lambda, tspectrum)

    def integrate_spectrum(self, tspectrum=None, minusepoints=50):
        '''function requires a table with 3 columns: wavelength, flux, fluxerr.
        If you do not pass it to tspectrum, the algorithm will take the one from
        self.spectrum'''
        if tspectrum is None:
            tspectrum = self.spectrum
        return integrate_flux(tspectrum['flux'],
                              xpoints=tspectrum['wavelength']) * \
                              tspectrum['flux'].unit*tspectrum['wavelength'].unit,\
            integrate_flux(tspectrum['fluxerr'],
                           xpoints=tspectrum['wavelength']) * \
                           tspectrum['fluxerr'].unit*tspectrum['wavelength'].unit

    def get_spectral_grid(self, lambdamin, lambdamax, presolv,
                          tspectrum=None, mode='number'):
        '''Give a (highres) spectrum, lambdamin/max stop and the resolving power.
        Returns the integrated spectrum.
        Set mode='number' to say presolv is the number of resolution elements or
        to 'rpower' to say it is the resolving power which leads to a logarithmic
        distribution of the bins.'''
        if tspectrum is None:
            tspectrum = self.spectrum
        return get_spectrum(tspectrum, lambdamin, lambdamax, presolv,
                            mode=mode)

    def flux(self):
        fluxes = []
        fluxerrs = []
        for mm, msig, fb in zip(self._mag, self._magerror, self._filterband):
            fluxes.append(mag2flux(mm, fb))
            fluxerrs.append(mag2flux(mm + msig, fb))
        fluxes = fluxes * fluxes[-1].unit
        fluxerrs = fluxerrs * fluxerrs[-1].unit
        return fluxes, fluxerrs

    def model_pm(self, tstart, tend):
        tpm = calprowrapper(self.sname, self.distance(),
                            self.coordinates.ra, self.coordinates.dec,
                            self.pm[0], self.pm[1],
                            tstart, tend)
        return tpm

    def distance(self):
        if self.parallax is None:
            print('Please set the parallax to get the distance')
        else:
            return (self.parallax).to(u.pc, equivalencies=u.parallax())

    def get_temp_via_bb(self, nusepoints=15):
        '''Fit a bb to the filtercurves transmission given
        in slef.mag nusepoints gives the number of points used
        in the filtercurves. The second value gives the scaling of
        the flux, e.g. sth similar to the steradian.'''
        print('Getting the temperature of {} using wavebands {}\n\
This may take some time.'.format(
            self.sname, self._filterband))
        temparea = fit_flux2temp(self.flux(), self._filterband,
                                 nusepoints=nusepoints)
        print('Found temperature and offset are: {}'.format(temparea))
        return temparea

    def temp2mag_from_filter(self, filterband, temperature=None,
                             skyarea=None, nusepoints=15):
        '''Get The magnitude in a filterband fitting a BB using the found
        temp and skyarea. Or give a temp and skyarea'''
        if temperature is None:
            temperature = self.temperature
        if skyarea is None:
            skyarea = self.skyarea
        flux = self.skyarea * fit_blackbody.filtered_flux(temperature,
                                                          filterband,
                                                          nusepoints=nusepoints)
        return flux2mag(flux, filterband, nusepoints=nusepoints)

    def show_filterzeros(self):
        return fit_blackbody.loadfiltzeros('all')

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coords):
        if 'auto' in coords:
            coords = Simbad.query_object(self.sname)['RA', 'DEC']
            coords = [coords['RA'][0] + ' ' + coords['DEC'][0], ]
            self._coordinates = SkyCoord(coords, unit=(u.hourangle, u.deg),
                                         frame='icrs')
        else:
            # make sure it is a list
            if len(coords) not in [1, 2, 3]:
                raise ValueError('Coordinates not understood. Set first to auto or provide\
them in a tuple with \
first: ra,dec, second coord_units, third: coord_frame. Two and three are optional.\
E.g. [["2:00:00 -1:00:00"],[u.hourangle, u.deg],"icrs"]')
            crds = coords[0]
            coord_units = self._coordinates[1]
            coord_frame = self._coordinates[2]
            if len(coords) >= 2:
                coord_units = coords[1]
            elif len(coords) == 3:
                coord_frame = coords[2]

            self._coordinates = SkyCoord(crds[0], crds[1], unit=coord_units,
                                         frame=coord_frame)

    @property
    def pm(self):
        return self._pm

    @pm.setter
    def pm(self, pm):
        if isinstance(pm, str):
            if 'auto' in pm.lower():
                Simbad.add_votable_fields('pmra', 'pmdec')
                pm = Simbad.query_object(self.sname)['PMRA', 'PMDEC']
                pm = [pm['PMRA'].to('mas/yr')[0].value,
                      pm['PMDEC'].to('mas/yr')[0].value] * u.mas / u.yr
            else:
                raise ValueError('Did not understand your input "{}" for pm. \
Set it to "auto" to query Simbad or e.g. "[1., 2.,] *u.mas"')
        else:
            # make sure it is a list
            if len(pm) != 2:
                raise ValueError('Dont understand pm format. please give 2 values')
            try:
                pm = pm * pm[0].unit
            except:
                print('No unit specified. Assuming mas/yr')
                pm = pm * u.mas / u.yr
        self._pm = pm

    @property
    def mag(self):
        return [self._mag, self._magerror, self._filterband]

    @mag.setter
    def mag(self, value_filt):
        value_filt = np.array(value_filt)
        if (value_filt.shape[-1] != 3) or\
           (len(value_filt.shape) > 3):
            raise ValueError('Please give a filter name and error to the magnitude\
            value. E.g. star.mag = [[3, 1, "HST_ACS_WFC.F814W_77"]]')
        if self.mag[0] is None:
            self._mag = np.array(value_filt[:, 0], dtype=np.float)
            self._magerror = np.array(value_filt[:, 1], dtype=np.float)
            self._filterband = np.array(value_filt[:, 2])
        else:
            self._mag = np.hstack((self._mag, np.array(value_filt[:, 0],
                                                       dtype=np.float)))
            self._magerror = np.hstack((self._magerror,
                                        np.array(value_filt[:, 1],
                                                 dtype=np.float)))
            self._filterband = np.hstack((self._filterband,
                                          np.array(value_filt[:, 2])))

    @mag.deleter
    def mag(self):
        print('Removing mags, errors and filterbands for {}'.format(self.sname))
        self._mag = None
        self._filterband = None
        self._magerror = None

    @property
    def aliases(self):
        if self._aliases is None:
            self._aliases = list(Simbad.query_objectids(self.sname)['ID'])
        return self._aliases

    @property
    def simbad_main_ID(self):
        if self._simbad_main_ID is None:
            self._simbad_main_ID = Simbad.query_object(
                self.sname)['MAIN_ID'][0].decode('utf-8')
        return self._simbad_main_ID
