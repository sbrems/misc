import os
import numpy as np
from astropy.io import fits
import astropy.units as u
from astropy.table import Table


@u.quantity_input(radius=u.cm)
def read_phoenix_highres(filename, radius=1*u.solRad, wlfilename=None):
    '''Reading Phoenix HighRes Spectra and outputting the data in Jy'''
    if wlfilename is None:
        __location__ = os.path.dirname(os.path.realpath(__file__))
        wlfilename = os.path.join(__location__, 'associated_data',
                                  'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')

    data, head = fits.getdata(filename, header=True)
    wavelengths = fits.getdata(wlfilename) * u.AA

    data = data * ((radius/(411.4*u.pc))**2).decompose()
    data = (data*u.erg/(u.s*u.cm**3)).to(
        u.Jy, equivalencies=u.spectral_density(wavelengths))

    return Table([wavelengths.to('micron'), data,
                  np.zeros(len(wavelengths))*data.unit],
                 names=['wavelength', 'flux', 'fluxerr'])
