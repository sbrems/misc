from .starclass import Star
import numpy as np
import astropy.unit as u
from .celestial_mechanics import make_orbit, calc_point

class Companion(Star):
    '''Try to determine some basic properties of a CC. Such as the Orbit etc.
    All Keywords start with a c, all others are passed to the Starclass.
    cname: Name of CC. removing last letter to give the star the name
    cmass: Mass of the Companion
    ######
    Orbital parameters: (based on Seagers Exoplanets book)
    ######
    cOmega: ascending node
    comega: argument of periapse
    cinclination: inclination
    cphyssep: physical separation. Can also be calculaten from cseparation
    (arcsec) and parallax (Star property)
    cPerdiod: Calculated from orbital parameters or given
    ct0: current Phase of the companion
    '''
    def __init__(self, cname, cmass=None, cseparation=None, cphyssep=None,
                 ceccentricity=0., cPerdiod=None,
                 cinclination=90 * u.deg, comega=0. * u.deg,
                 cOmega=0. * u.deg, ct0=0., **kwargs)

        if cname[-1].isalpha:
            _name = cname[:-1]
        super().__init__(self, _name, kwargs)
        self.cname = cname
        self.cmass = cmass.to('M_jup')
        self.cseparation = cseparation.('arcsec')
        self.ceccentricity = ceccentricity
        self.cinclination = cinclination.to('deg')
        self.comega = comega.to('deg')
        self.cOmega = cOmega.to('deg')
        self.ct0 = ct0

        self._cphyssep = cphyssep
        self._cPerdiod = cPerdiod

    def modelorbit(self, plot=False):
        return make_orbit(self.mass, self.cmass, self.cphyssep,
                          self.eccentricity, self.cPeriod, t0=self.ct0,
                          plot=plot)

    def modelpoint(self, time):
        return calc_point(self.mass, self.cmass, self.cphyssep,
                          self.eccentricity, self.cPeriod, time,
                          t0=self.ct0)

    @property
    def cphyssep(self):
        if self._cphyssep is None:
            print('Assuming circular phase on orbit')
            self._cphyssep = cseparation / u.arcsec * self.distance / u.pc *\
                             u.AU
        return self._cphyssep

    @cphyssep.setter
    def cphyssep(self, physsep):
        self._physsep = physsep.to('AU')

    @property
    def cPeriod(self):
        if self._cPeriod is None:
            print('Getting Period assuming a circular Orbit')
            return np.sqrt(self._physsep**3 / u.AU**3 / self.mass * u.M_sun *
                           u.yr)
        else:
            return self._cPeriod

    @cPerdiod.setter
    def cPerdiod(self, Period):
        self._cPerdiod = Period.to('year')

