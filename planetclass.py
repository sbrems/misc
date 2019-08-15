from starclass import Star
from celestial_mechanics import proper_motion
import numpy as np
import astropy.units as u
# from astropy.time import Time
from astropy.table import Table
import dpa2radec
from celestial_mechanics.model_orbit import make_orbit, calc_point
from celestial_mechanics import overpaint_image


class Planet(Star):
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
    cPositions: positions with epochs + errors of directly imaged companions.
Stored in a table. Valid entry would be
    (radec or dpa for RA/DEC or DIST/ParANG): \n
    (dates=Time([2000.0, 2010.0], format='decimalyear'), 'radec', [[200., 10.,],
        [220., 10.]], [[1200., 20.], [1300., 20.]].)
    '''
    @u.quantity_input(cseparation=u.arcsec,
                      cmass='mass',
                      cinclination=u.deg, comega=u.deg,
                      cOmega=u.deg, cphyssep=u.AU)
    def __init__(self, cname, cmass=None, cseparation=None, cphyssep=None,
                 ceccentricity=None, cPerdiod=None,
                 cinclination=90*u.deg, comega=0. * u.deg,
                 cOmega=0. * u.deg, ct0=0.,
                 cPositions=None,
                 **kwargs):

        if cname[-1].isalpha:
            _name = cname[:-1]
        Star.__init__(self, _name, kwargs)
        self.cname = cname
        self.cmass = cmass
        self._cPositions = cPositions
        self._cseparation = cseparation
        self.ceccentricity = ceccentricity
        self.cinclination = cinclination
        self.comega = comega
        self.cOmega = cOmega
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

    @u.quantity_input(pxscale=u.mas, coordinats=u.mas)
    def overpaint_proper_motion(self, image, imdate, refdate, coordinates,
                                xyimstar=None, pxscale=27.02*u.mas,
                                plot_startail=True, pnsave=None,
                                plotlimits=[-1, 2.5]):
        overpaint_image.overpaint(self, image, imdate, refdate,
                                  coordinates, xyimstar=xyimstar,
                                  pxscale=pxscale,
                                  plot_startrail=plot_startail,
                                  pnsave=pnsave, plotlimits=plotlimits)

    def model_proper_motion(self, labels=None, irefdate=0):
        '''Return the appearent positions of the source given in cPositions.
        Set cPositions and proper motion (of the star) before.
        labels = [str,]
        the labels for the plot. None labels them after the dates of the points
        irefdate = 0 (int)
        referencedate to use for pm plot

        returs
        table with the positions of the star as given my calpro
        '''
        if self.coordinates is None:
            print('No coordinates given. Using Simbad ones.')
            self.coordinates = 'auto'
        if self.pm is None:
            print('No proper motion given. Using Simbad one.')
            self.pm = 'auto'
        if self.distance is None:
            print('No distance/parallax given. Using Simbad one.')
            self.parallax = 'auto'
        if labels is None:
            labels = ['{:.2f}'.format(dyr) for dyr in
                      self.cPositions['date'].decimalyear]
        ttimes = proper_motion.pm_plot(self.cname, self.distance,
                                       self.coordinates,
                                       self.pm, self.cPositions, labels,
                                       irefdate=irefdate)
        return ttimes

    @property
    def cPositions(self):
        return self._cPositions

    @cPositions.setter
    def cPositions(self, args):
        '''Set the positions for any CC. Give either RA/DEC \
        (radec as second argument) or the separation/PA \
        (dpa as second argument). Also provide the errors and the dates of the
        observations. A valid input would be f.e.
        (Time([2000.0, 2010.0], format='decimalyear'), 'radec', [[200., 10.,],
        [220., 10.]]*u.mas, [[1200., 20.], [1300., 20.]]*u.mas)
        PA in East of north if given.
        Stores it in an astropyTable'''
        date, coordinatetype, coords1, coords2 = args
        if coordinatetype.lower() == 'radec':
            RA = coords1
            DEC = coords2
            assert len(date) == RA.shape[0] == DEC.shape[0]
            assert RA.shape[1] == 2
            separation = []
            PA = []
            for iobs in range(len(RA)):
                seppa = dpa2radec.inverse(RA[iobs, 0].to('arcsec').value,
                                          RA[iobs, 1].to('arcsec').value,
                                          DEC[iobs, 0].to('arcsec').value,
                                          DEC[iobs, 1].to('arcsec').value)
                separation.append(seppa[0:2])
                PA.append(seppa[2:4])
            separation *= u.arcsec
            PA *= u.deg
        elif coordinatetype.lower() == 'dpa':
            separation = coords1
            PA = coords2
            assert len(date) == separation.shape[0] == PA.shape[0]
            assert separation.shape[1] == 2
            RA = []
            DEC = []
            for iobs in range(len(separation)):
                radec = dpa2radec.do(separation[iobs, 0].to('arcsec').value,
                                     separation[iobs, 1].to('arcsec').value,
                                     PA[iobs, 0].to('deg').value,
                                     PA[iobs, 1].to('deg').value)
                RA.append(radec[0:2])
                DEC.append(radec[2:4])
            RA *= u.arcsec
            DEC *= u.arcsec
        else:
            raise ValueError('Second argument must be radec or dpa \
(distance parallactic angle)')

        self._cPositions = Table([date,
                                 separation[:, 0], separation[:, 1],
                                 PA[:, 0], PA[:, 1],
                                 RA[:, 0], RA[:, 1],
                                 DEC[:, 0], DEC[:, 1]],
                                names=['date', 'separation', 'separationerr',
                                       'PA', 'PAerr',
                                       'RA', 'RAerr', 'DEC', 'DECerr'])

    @property
    def cphyssep(self):
        if self._cphyssep is None:
            # print('Assuming circular phase on orbit')
            self._cphyssep = list(zip(self.cPositions['separation'].to('arcsec') *
                                      self.distance.to('pc') / u.pc * u.AU,
                                      self.cPositions['separationerr'].to('arcsec') *
                                      self.distance.to('pc') / u.pc * u.AU))
        return self._cphyssep

    @cphyssep.setter
    @u.quantity_input(physsep=u.AU)
    def cphyssep(self, physsep):
        if str(physsep).isnumeric():
            print('No error given. Assuming no Error')
            physsep = ([[physsep.to('AU'), 0.]]) * u.AU
        elif len(physsep.shape) != 2 or \
             physsep.shape[1] != 2:
            raise ValueError('Please provide the separation in a nx2 quantity \
matrix.\n\
E.g. [[[[100., 5.], [200, 4.], [344, 5]] * u.AU]] for three measurements \
with their respective errors.')
        self._physsep = physsep.to('AU')

    @property
    def cPeriod(self):
        if self._cPeriod is None:
            print('Getting Period assuming a circular Orbit')
            return np.sqrt(self._physsep**3 / u.AU**3 / self.mass * u.M_sun *
                           u.yr)
        else:
            return self._cPeriod

    @cPeriod.setter
    def cPerdiod(self, Period):
        self._cPerdiod = Period.to('year')

