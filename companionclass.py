from starclass import Star
import numpy as np
import astropy.units as u
from celestial_mechanics.model_orbit import make_orbit, calc_point, project_orbit


class Companion(Star):
    '''Try to determine some basic properties of a CC. Such as the Orbit etc.
    All Keywords start with a c, all others are passed to the Starclass.
    The values are based on astropy.units (e.g. comp.mass = 1 * u.M_sun).
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
    cPeriod: Calculated from orbital parameters or given
    ct0: time after periastron
    '''

    def __init__(self, cname, cmass=None, cseparation=None, cphyssep=None,
                 ceccentricity=0., cPeriod=None,
                 cinclination=90 * u.deg, comega=0. * u.deg,
                 cOmega=0. * u.deg, ct0=0. * u.yr, **kwargs):
        if cname[-1].isalpha():
            starname = cname[:-1].strip()
        else:
            print('Your Companions name does not end with a letter!? \
Giving star the same name')
            starname = cname
        super().__init__(starname, kwargs)
        self.cname = cname
        self.cmass = cmass
        self.cseparation = cseparation
        self.ceccentricity = ceccentricity
        self.cinclination = cinclination
        self.comega = comega
        self.cOmega = cOmega
        self.ct0 = ct0

        self._cphyssep = cphyssep
        self._cPeriod = cPeriod

    def modelorbit(self, plot=True):
        return project_orbit(self.comega, self.cOmega, self.cinclination,
                             self.mass, self.cmass, self.cphyssep,
                             self.ceccentricity, self.cPeriod, t0=self.ct0,
                             plot=plot)

    def modelpoint(self, time):
        return calc_point(self.mass, self.cmass, self.cphyssep,
                          self.ceccentricity, self.cPeriod, time,
                          t0=self.ct0)

    @property
    def cphyssep(self):
        if self._cphyssep is None:
            print('Assuming circular phase on orbit')
            self._cphyssep = self.cseparation / u.arcsec *\
                self.distance() / u.pc *\
                u.AU
        return self._cphyssep

    @cphyssep.setter
    def cphyssep(self, cphyssep):
        self._cphyssep = cphyssep.to('AU')

    @property
    def cPeriod(self):
        if self._cPeriod is None:
            print('Getting Period assuming a circular Orbit')
            return np.sqrt(self._cphyssep**3 / u.AU**3 / (self.mass +
                                                          self.cmass) *
                           u.M_sun) * u.yr
        else:
            return self._cPeriod

    @cPeriod.setter
    def cPeriod(self, Period):
        self._cPeriod = Period.to('year')
