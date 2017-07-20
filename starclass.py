import numpy as np
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u

from spt_magnitude_conversion import split_spt
import fit_blackbody
from fit_blackbody import fit_temp2flux, flux2mag, mag2flux


class Star():
    '''Save properties of a star. So far available:
    aliases (auto from Simbad)
    Coordinates (=['auto',] for Simbad query),
    coord_units: default: ['u.hourangle','u.deg'],
    default coord_frame ='icrs'
    Magnitudes (with the filterband, default filterband = None)
    SpT (also num)
    aliases
    PM
    Magnitude bands
    Temperature detemination via a BB fit to these bands
    parallax
    '''
    def __init__(self, name,
                 coordinates=[None, [u.hourangle, u.deg], 'icrs'],
                 aliases=None,
                 mag=None, filterband=None, magerror=None,
                 SpT=None, SpC=None, temperature=None):
        self.name = name
        self.SpT = SpT
        self.temperature = temperature
        self.skyarea = None #the result when fitting a temperature
        
        self._aliases = aliases
        self._coordinates = coordinates
        self._mag = mag
        self._filterband = filterband
        self._magerror = magerror
    _kind = 'star'

    def SpT_num(self):
        print('Blubb')
        return np.sum(split_spt(self.SpT)[0:2])

    def SpC_num(self):
        return split_spt(self.SpT)[2]

    def flux(self):
        fluxes = []
        fluxerrs = []
        for mm, msig, fb in zip(self._mag, self._magerror, self._filterband):
            fluxes.append(  mag2flux(mm, fb))
            fluxerrs.append(mag2flux(mm + msig, fb))
        fluxes = fluxes * fluxes[-1].unit
        fluxerrs = fluxerrs * fluxerrs[-1].unit
        return fluxes, fluxerrs

    def get_temp_via_bb(self, nusepoints=15):
        '''Fit a bb to the filtercurves transmission given
        in slef.mag nusepoints gives the number of points used
        in the filtercurves. The second value gives the scaling of 
        the flux, e.g. sth similar to the steradian.'''
        print('Getting the temperature of {} using wavebands {}\n\
This may take some time.'.format(
                self.name, self._filterband))
        return fit_temp2flux(self.flux(), self._filterband,
                             nusepoints=nusepoints)

    def temp2mag_from_filter(self, filterband, nusepoints=15, temperature=None,
                             skyarea = None):
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
            coords = Simbad.query_object(self.name)['RA', 'DEC']
            coords = [coords['RA'][0]+' '+coords['DEC'][0], ]
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
            coord_units = self._coord_units
            coord_frame = self._coord_frame
            if len(coords) >= 2:
                coord_units = coords[1]
            elif len(coords) == 3:
                coord_frame = coords[2]

        self._coordinates = SkyCoord(coords, unit=coord_units,
                                     frame=coord_frame)

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
            self._magerror   = np.array(value_filt[:, 1], dtype=np.float)
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
        print('Removing mags, errors and filterbands for {}'.format(self.name))
        self._mag        = None
        self._filterband = None
        self._magerror   = None

    @property
    def aliases(self):
        if self._aliases is None:
            self._aliases = list(Simbad.query_objectids(self.name)['ID'])
        return self._aliases

            
