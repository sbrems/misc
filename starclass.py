import numpy as np
import copy
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from astropy.io import fits
from astropy.constants import c
from scipy import integrate
from scipy.interpolate import interp1d
import time

from celestial_mechanics.proper_motion import calprowrapper
from spt_magnitude_conversion import split_spt
import fit_blackbody
import read_files
from fit_blackbody import fit_flux2temp, flux2mag, mag2flux, integrate_flux
from bin_spectra import filter_flux, get_spectrum, fold_with_gauss
from integrate_spectra import integrate_with_error


class Star():
    '''Save properties of a star. So far available:
    aliases ('auto' from Simbad)
    Coordinates (=['auto',] for Simbad query),
    coord_units: default: ['u.hourangle','u.deg'],
    default coord_frame ='icrs'
    Magnitudes (with the filterband, default filterband = None)
    SpT (also num), get_SpT tries to get it from Simbad
    Magnitude bands
    storing and getting pm from simbad (='auto')
    Temperature detemination via a BB fit to these bands
    parallax and conversion to distance (withough errors, as mostly you want parallaxerr)
    get_filtered_spectrum (set spectrum first)
    simbad_main_ID returns the Simbad main ID. Usefull to see if stars are aliases
    '''

    @u.quantity_input(
        temperature=u.K,
        parallax=u.mas,
        pm=u.mas / u.yr,
        mass='mass',
    )
    def __init__(self,
                 sname,
                 aliases=None,
                 simbad_main_ID=None,
                 mag=None,
                 filterband=None,
                 magerror=None,
                 SpT=None,
                 SpC=None,
                 temperature=None,
                 mass=None,
                 parallax=None,
                 pm=None,
                 distance=None):
        self.sname = sname
        self.SpT = SpT
        self.temperature = temperature
        self.skyarea = None  # the result when fitting a temperature
        self.mass = mass
        self.spectrum = None

        self._parallax = parallax
        self._pm = pm
        self._distance = distance
        self._aliases = aliases
        self._coordinates = None
        self._simbad_main_ID = simbad_main_ID
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
        self.SpT = (Simbad.query_object(
            self.sname)['SP_TYPE'][0]).decode('utf-8')

    @u.quantity_input(radius=u.cm)
    def read_phoenix_highres(self, filename,
                             radius=1*u.solRad, wlfilename=None):
        '''Read the highres Phoenix spectrum. Scale flux to the stellar
        Radius.'''
        return read_files.read_phoenix_highres(filename,
                                               radius=radius,
                                               wlfilename=wlfilename)

    def get_activity_indicators(self, shiftToVacuum=True):
        '''get the activity indicators for a given spectrum.'''
        tspectrum = self.spectrum.copy()
        if shiftToVacuum:
            print('Shifting wavelength to vacuum')
            tspectrum['wavelength'] = self.vacuumToAir(
                tspectrum['wavelength'])

        # taken from SERVAL/Martins K's email. First is Angstrom, range in km/s
        dactinds = {
            'NaD1': [[5889.950943, -15.0, 15.], [[5885.0, -40., 40.], [5892.94, -200, +300]]],
            'NaD2': [[5895.924237, -15.0, 15.], [[5892.94, -300, +300], [5905.0, -40, +40]]],
            'Halpha': [[6562.808, -40.0, +40.], [[6562.808, -300, -100], [6562.808, +100, +300]]],
            'CaII IRT1': [[8498.02, -15., 15.], [[8492.0, -40, 40], [8504.0, -40., 40.]]],
            'CaII IRT2': [[8542.09, -15., 15.], [[8542.09, -300, -200], [8542.09, +200, +300]]],
            'CaII IRT3': [[8662.14, -15., 15.], [[8662.14, -300, -200], [5905.0, -40, +40]]]
        }
        self.activity_indicators = {}
        for actind, (linerange, compranges) in dactinds.items():
            llims = [linerange[0]*np.sqrt(
                (1-linerange[1]*u.km/u.s/c) /
                (1+linerange[1]*u.km/u.s/c)),
                     linerange[0]*np.sqrt(
                         (1-linerange[2]*u.km/u.s/c) /
                         (1+linerange[2]*u.km/u.s/c))]
            if np.max(llims) > np.max(tspectrum['wavelength']) or \
               np.min(llims) < np.min(tspectrum['wavelength']):
                print('Not determining activity indicator {} because wavelengthrange \
is not covered'.format(actind))
                continue

            lineflux, linefluxerr = self.integrate_flux_with_error(
                bounds=[np.min(llims), np.max(llims)]*u.Angstrom,
                tspectrum=tspectrum)
            lineflux /= np.abs(llims[1] - llims[0])
            refflux, reffluxerr = 0., 0.
            for comp in compranges:
                clims = [comp[0]*np.sqrt(
                    (1-comp[1]*u.km/u.s/c) /
                    (1+comp[1]*u.km/u.s/c)),
                         comp[0]*np.sqrt(
                    (1-comp[2]*u.km/u.s/c) /
                    (1+comp[2]*u.km/u.s/c))]

                refs = self.integrate_flux_with_error(
                    bounds=[np.min(clims), np.max(clims)]*u.Angstrom,
                    tspectrum=tspectrum)
                refs /= np.abs(clims[1] - clims[0])
                refflux += refs[0]
                reffluxerr += refs[1]**2
            reffluxerr = np.sqrt(reffluxerr)

            index = lineflux / refflux / len(compranges)
            indexerr = np.sqrt(index * ((linefluxerr/lineflux)**2 +
                                        (reffluxerr / refflux)**2))
            self.activity_indicators[actind] = [index, indexerr]


    @u.quantity_input(rv=u.km/u.s)
    def read_ceres_spectrum(self, fpath, rv=None):
        '''read the normalized ceres spectrum. Main task is to find the order
        with the highest SNR for a given wavelength. Give the RV to shift the
        lines. If rv=None, it tries the header values'''
        if not fpath.endswith('_sp.fits'):
            print('CAUTION! Usually the FEROS output ends on "_sp.fits". \
Check file {} is correct!'.format(fpath))

        # columns needed are: [wl, flux, fluxerr, SNR]
        spec = fits.getdata(fpath)[[0, 5, 6, 8], :, :]

        # determine which of neighbouring orders has highest SNR
        for io in range(spec.shape[1] - 1):
            flowerord = interp1d(spec[0, io + 1, :],
                                 spec[3, io + 1, :],
                                 assume_sorted=True,
                                 fill_value=-np.inf,
                                 bounds_error=False)
            icut = [((spec[3, io, iwl] >= flowerord(spec[0, io, iwl]))
                     and (spec[3, io, iwl] > 0.))
                    for iwl in range(spec.shape[2])]
            cutwl = spec[0, io, np.min(np.where(icut))]
            # set worse values to NaN
            for idim in range(spec.shape[0]):
                spec[idim, io, spec[0, io, :] < cutwl] = np.nan
                spec[idim, io + 1, spec[0, io + 1, :] >= cutwl] = np.nan
        spec = spec.reshape(4, -1)
        spec = spec[:, np.isfinite(spec[0])]
        spec = spec[:, np.argsort(spec[0, :])]

        tspec = Table([
            spec[0] * u.Angstrom,
            spec[1] * u.erg/(u.cm**2*u.s*u.Angstrom),
            # ceres returns the inverse variance as error:
            np.sqrt(1/spec[2]) * u.erg/(u.cm**2*u.s*u.Angstrom),
            spec[3]
        ],
                      names=('wavelength', 'flux', 'fluxerr', 'SNR'),
                      meta={
                          'origin': fpath,
                          'pipeline': 'CERES'
                      })

        # Doppler shift the spectrum
        if rv is None:
            head = fits.getheader(fpath)
            rv = (head['RV'])*u.km/u.s #+ head['BARYCENTRIC CORRECTION (KM/S)']) * u.km/u.s

        if rv != 0:
            print('Shifting the spectrum by {}'.format(rv))
            tspec['wavelength'] = tspec['wavelength']*np.sqrt((1-(rv/c)) /
                                                              (1+(rv/c)))
        del spec
        self.spectrum = tspec

    def read_spectrum(self,
                      fpath,
                      wlunit=u.nm,
                      fluxunit=u.erg / (u.cm**2 * u.s * u.Angstrom),
                      wlidx=0,
                      flxidx=1,
                      flxerridx=2):
        data = fits.getdata(fpath)
        print('Assuming dataformat lambda,flux,error with wl-unit \
{} and flux unit {}. Access via self.spectrum'.format(wlunit, fluxunit))
        tflux = Table([
            data[:, wlidx] * wlunit, data[:, flxidx] * fluxunit,
            data[:, flxerridx] * fluxunit
        ],
                      names=('wavelength', 'flux', 'fluxerr'),
                      meta={'origin': fpath})

        self.spectrum = tflux

    @u.quantity_input(wavelengths=u.Angstrom)
    def airToVacuum(self, wavelengths=None):
        '''Convert the wavelength from air to vacuum. Following the CERES approach,
        following Edlen 1966'''
        if wavelengths is None:
            wavelengths = self.spectrum['wavelength']
        wls = wavelengths.to('Angstrom').value.copy()
        converged = False
        while not converged:
            wl_new = self._n_Edlen(wls) * wavelengths.to('Angstrom').value
            if np.max(np.abs(wl_new - wls)) < 1e-10:
                converged = True
            wls = wl_new

        return wls

    @u.quantity_input(wavelengths=u.Angstrom)
    def vacuumToAir(self, wavelengths=None):
        if wavelengths is None:
            wavelengths = self.spectrum['wavelength']
        return wavelengths / self._n_Edlen(wavelengths)

    def _n_Edlen(self, wavelength):
        '''Refractive index according to Edlen 1966. Following CERES approach.
        In Angstrom'''
        sigma = 1e4 / wavelength
        sigma2 = sigma**2
        n = 1 + 1e-8 * (8342.13 + 2406030 / (130-sigma2) + 15997/(38.9-sigma2))
        return n

    def get_filteredspectrum(self,
                             filtname,
                             minusepoints=50,
                             tspectrum=None,
                             invert=False):
        '''Set invert=True to increase the flux, e.g. to get the original flux
        after it had been filtered'''
        if tspectrum is None:
            tspectrum = copy.copy(self.spectrum)
        return filter_flux(tspectrum,
                           filtname,
                           minusepoints=minusepoints,
                           invert=invert)

    def integrate_flux_with_error(self, bounds=None, tspectrum=None):
        if tspectrum is None:
            tspectrum = self.spectrum.copy()
        return integrate_with_error(tspectrum, bounds)

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
        self.spectrum. The error is not calculated properly!'''
        if tspectrum is None:
            tspectrum = self.spectrum
        return integrate_flux(tspectrum['flux'],
                              xpoints=tspectrum['wavelength']) * \
                              tspectrum['flux'].unit*tspectrum['wavelength'].unit,\
            integrate_flux(tspectrum['fluxerr'],
                           xpoints=tspectrum['wavelength']) * \
                           tspectrum['fluxerr'].unit*tspectrum['wavelength'].unit

    def get_spectral_grid(self,
                          lambdamin,
                          lambdamax,
                          presolv,
                          tspectrum=None,
                          mode='number'):
        '''Give a (highres) spectrum, lambdamin/max stop and the resolving power.
        Returns the integrated spectrum.
        Set mode='number' to say presolv is the number of resolution elements or
        to 'rpower' to say it is the resolving power which leads to a logarithmic
        distribution of the bins.'''
        if tspectrum is None:
            tspectrum = self.spectrum
        return get_spectrum(tspectrum,
                            lambdamin,
                            lambdamax,
                            presolv,
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
        tpm = calprowrapper(self.sname, self.distance, self.coordinates.ra,
                            self.coordinates.dec, self.pm[0], self.pm[1],
                            tstart, tend)
        return tpm

    def get_temp_via_bb(self, nusepoints=15):
        '''Fit a bb to the filtercurves transmission given
        in slef.mag nusepoints gives the number of points used
        in the filtercurves. The second value gives the scaling of
        the flux, e.g. sth similar to the steradian.'''
        print('Getting the temperature of {} using wavebands {}\n\
This may take some time.'.format(self.sname, self._filterband))
        temparea = fit_flux2temp(self.flux(),
                                 self._filterband,
                                 nusepoints=nusepoints)
        print('Found temperature and offset are: {}'.format(temparea))
        return temparea

    @u.quantity_input(stellar_radius=u.Rsun)
    def get_absolute_magnitude(self, filterband, stellar_radius):
        '''Return the absolute magnitud for the filterband given.
        self.spectrum must be defined before. This function is made
        to work with BT-settl models'''
        intflux = self.integrate_spectrum(
            self.get_filteredspectrum(filterband))[0] / \
            self.calculate_equivalent_width(filterband)
        mag = -2.5 * np.log10(intflux / self.show_filterzeros()[filterband])
        MAG = mag - 5 * np.log10(stellar_radius / u.pc) + 5
        return MAG

    def calculate_equivalent_width(self, filterband):
        tfilter = self.get_filter(filterband)
        return integrate.trapz(tfilter['transmis'],
                               tfilter['lambda']) * tfilter['lambda'].unit

    def get_filter(self, filterband):
        '''Return the filtercuve, if the file exists'''
        return fit_blackbody.get_filter(filterband)

    def temp2mag_from_filter(self,
                             filterband,
                             temperature=None,
                             skyarea=None,
                             nusepoints=15):
        '''Get The magnitude in a filterband fitting a BB using the found
        temp and skyarea. Or give a temp and skyarea'''
        if temperature is None:
            temperature = self.temperature
        if skyarea is None:
            skyarea = self.skyarea
        flux = self.skyarea * fit_blackbody.filtered_flux(
            temperature, filterband, nusepoints=nusepoints)
        return flux2mag(flux, filterband, nusepoints=nusepoints)

    def show_filterzeros(self):
        return fit_blackbody.loadfiltzeros('all')

    @property
    def parallax(self):
        if self._parallax is None:
            print('Parallax is None. To get it from Simbad, \
set parallax = "auto"')
        return self._parallax

    @parallax.setter
    def parallax(self, parallax):
        if isinstance(parallax, str):
            if parallax.lower() == 'auto':
                Simbad.add_votable_fields('parallax')
                plxplxerr = Simbad.query_object(
                    self.sname)['PLX_VALUE', 'PLX_ERROR']
                self._parallax = plxplxerr['PLX_VALUE'].to('mas')[0]
                self.parallaxerr = plxplxerr['PLX_ERROR'].to('mas')[0]
                Simbad.reset_votable_fields()
        else:
            self._parallax = parallax.to('mas')

    @property
    def distance(self):
        if self._distance is None:
            if self._parallax is None:
                print('Give parallax or distance first, \
e.g. by setting parallax="auto"')
            else:
                self._distance = (self.parallax).to(u.pc,
                                                    equivalencies=u.parallax())
        return self._distance

    @distance.setter
    @u.quantity_input(dist=u.pc)
    def distance(self, dist):
        self._distance = dist

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coords):
        if isinstance(coords, str):
            if 'auto' in coords:
                coords = Simbad.query_object(self.sname)['RA', 'DEC']
                coords = [
                    coords['RA'][0] + ' ' + coords['DEC'][0],
                ]
                self._coordinates = SkyCoord(coords,
                                             unit=(u.hourangle, u.deg),
                                             frame='icrs')
        elif coords is None:
            self._coordinates = None
        elif isinstance(coords, SkyCoord):
            self._coordinates = coords
        else:
            # make sure it is a list
            if len(coords) not in [1, 2, 3]:
                raise ValueError(
                    'Coordinates not understood. Set first to auto or provide\
them in a tuple with \
first: ra,dec, second coord_units, third: coord_frame. Two and three are optional.\
E.g. [["2:00:00 -1:00:00"],[u.hourangle, u.deg],"icrs"]')
            crds = coords[0]
            coord_units = [u.deg, u.deg]
            coord_frame = 'icrs'
            if len(coords) >= 2:
                coord_units = coords[1]
            elif len(coords) == 3:
                coord_frame = coords[2]

            self._coordinates = SkyCoord(crds[0],
                                         crds[1],
                                         unit=coord_units,
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
                pm = [
                    pm['PMRA'].to('mas/yr')[0].value,
                    pm['PMDEC'].to('mas/yr')[0].value
                ] * u.mas / u.yr
                Simbad.reset_votable_fields()
            else:
                raise ValueError('Did not understand your input "{}" for pm. \
Set it to "auto" to query Simbad or e.g. "[1., 2.,] *u.mas"')
        else:
            # make sure it is a list
            if len(pm) != 2:
                raise ValueError(
                    'Dont understand pm format. Please give 2 values')
            try:
                pm.unit
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
            raise ValueError(
                'Please give a filter name and error to the magnitude\
            value. E.g. star.mag = [[3, 1, "HST_ACS_WFC.F814W_77"]]')
        if self.mag[0] is None:
            self._mag = np.array(value_filt[:, 0], dtype=np.float)
            self._magerror = np.array(value_filt[:, 1], dtype=np.float)
            self._filterband = np.array(value_filt[:, 2])
        else:
            self._mag = np.hstack(
                (self._mag, np.array(value_filt[:, 0], dtype=np.float)))
            self._magerror = np.hstack(
                (self._magerror, np.array(value_filt[:, 1], dtype=np.float)))
            self._filterband = np.hstack(
                (self._filterband, np.array(value_filt[:, 2])))

    @mag.deleter
    def mag(self):
        print('Removing mags, errors and filterbands for {}'.format(
            self.sname))
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
            try:
                self._simbad_main_ID = Simbad.query_object(
                    self.sname)['MAIN_ID'][0].decode('utf-8')
            except ConnectionError:
                print(
                    'Timeouterror. This usually happens if you send many queries \
and SIMBAD blocks the connection for some time. Trying again in 20seconds')
                time.sleep(20.)
                self._simbad_main_ID = Simbad.query_object(
                    self.sname)['MAIN_ID'][0].decode('utf-8')
        return self._simbad_main_ID
