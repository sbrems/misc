# a wrapper for the lisp pm-script calpro.pl from yamamoto
import os
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.time import Time
from subprocess import call
from shutil import copyfile


def calprowrapper(sname, dist, RA, DEC, pmRA, pmDEC,
                  tstart, tend,
                  tempdir=None):
    '''Calculate the proper motion from start to enddate.
    Start and enddate should be a astropy time element'''
    # store defdir as we need to cange it
    if tempdir is None:
        tempdir = os.path.join(os.getcwd(), 'temp')
    if not os.path.exists(tempdir):
        os.mkdir(tempdir)
    defdir = os.getcwd()
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    # first make the input file for calpro
    fnout = 'dates{}.dat'.format(sname)
    with open(os.path.join(tempdir, fnout), 'wb') as outfile:
        outfile.write(bytes('star: {}\n\
Dist: {}\n\
RA: {} {} {}\n\
DEC: {} {} {}\n\
pRA: {}\n\
pDEC: {}\n\
===================\
{}\n\
{}\n'.format(
    sname,
    dist.to('pc').value,
    int(RA.hms.h), int(RA.hms.m), float(RA.hms.s),
    int(DEC.dms.d), int(abs(DEC.dms.m)), abs(float(DEC.dms.s)),
    pmRA.to('mas/yr').value,
    pmDEC.to('mas/yr').value,
    tstart.iso.split()[0],
    tend.iso.split()[0]), 'UTF-8')
        )

    # copy the script to the tempdir so it can be executed
    copyfile(os.path.join(scriptdir, 'calpro.pl'),
             os.path.join(tempdir, 'calpro.pl'))
    os.chdir(tempdir)
    call(['perl', 'calpro.pl', fnout])
    os.chdir(defdir)

    # read the results
    tpm = Table.read('{}_orbit.dat'.format(sname), format='ascii',
                     delimiter='\s', comment='#',
                     names=['JD', 'vRA', 'vDEC', 'deltaRA', 'deltaDEC',
                            'RA', 'DEC'])
    # give the units
    tpm['JD'] = Time(tpm['JD'], format='jd')
#    tpm['vRA'] *= u.arcsec / u.d
#    tpm['vDEC'] *= u.mas / u.yr
    tpm['deltaRA'] *= u.arcsec
    tpm['deltaDEC'] *= u.arcsec
    tpm['RA'] *= u.deg
    tpm['DEC'] *= u.deg
    return tpm
