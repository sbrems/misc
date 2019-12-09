import numpy as np


def do(d, dd, pa, dpa, verbose=False):
    if verbose:
        print('Input:\n Dist: %s+-%s mas/px, PA: %s+-%s deg' % (
            d, dd, pa, dpa))
    pa = pa * np.pi / 180.
    dpa = dpa * np.pi / 180.
    ra = d * np.sin(pa)
    dec = d * np.cos(pa)
    ddec = np.sqrt((d*np.sin(pa) * dpa)**2 + (np.cos(pa)*dd)**2)
    dra  = np.sqrt((d*np.cos(pa) * dpa)**2 + (np.sin(pa)*dd)**2)
    if verbose:
        print('Out:\n RA:%s +- %s mas, DEC: %s +- %s mas' % (
            ra, dra, dec, ddec))
    return ra, dra, dec, ddec


def inverse(ra, dra, dec, ddec, verbose=False):
    if verbose:
        print('Input:\n Ra: {}+-{} mas/px, Dec: {}+-{} mas/px'.format(
            ra, dra, dec, ddec))
    pa = np.arctan2(ra, dec) * 180 / np.pi % 360
    d = np.sqrt(ra**2 + dec**2)
    dpa = np.sqrt(np.square(ra / d**2 * ddec) +
                  np.square(dec / d**2 * dra))
    dd = np.sqrt((dra*np.abs(np.cos(pa)))**2 + (ddec*np.abs(np.sin(pa)))**2)
    if verbose:
        print('Orientation: 0 is up, positive PA to east (pos ra)')
        print('Out:\n Dist: {}+-{} mas/px, PA: {}+-{} deg'.format(
            d, dd, pa, dpa))
    return d, dd, pa, dpa
