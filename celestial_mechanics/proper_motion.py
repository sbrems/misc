# a wrapper for the lisp pm-script calpro.pl from yamamoto
import os
import matplotlib.pyplot as plt
from matplotlib import rc
# import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.time import Time
from subprocess import call
from shutil import copyfile
import seaborn as sns


def calprowrapper(sname, dist, RA, DEC, pmRA, pmDEC,
                  tstart, tend,
                  tempdir=None):
    '''Calculate the proper motion from start to enddate.
    Start and enddate should be a astropy time element and shoudl be the
    first/last date to consider'''
    # store defdir as we need to cange it
    if tempdir is None:
        tempdir = os.path.join(os.getcwd(), 'temp')
    if not os.path.exists(tempdir):
        os.mkdir(tempdir)
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    defdir = os.getcwd()
    # first make the input file for calpro
    fnout = 'dates{}.dat'.format(sname)
    with open(os.path.join(tempdir, fnout), 'wb') as outfile:
        outfile.write(bytes('star: {}\n\
Dist: {}\n\
RA: {} {} {}\n\
DEC: {} {} {}\n\
pRA: {}\n\
pDEC: {}\n\
===================\n\
{}\n\
{}\n'.format(sname,
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
    tpm = Table.read(os.path.join(tempdir, '{}_orbit.dat'.format(sname)),
                     format='ascii.no_header',
                     delimiter='\s', comment='#',
                     names=['JD', 'vRA', 'vDEC', 'deltaRA', 'deltaDEC',
                            'RA', 'DEC'],)
    # give the units
    tpm['JD'] = Time(tpm['JD'], format='jd')
#    tpm['vRA'] *= u.arcsec / u.d
#    tpm['vDEC'] *= u.mas / u.yr
    tpm['deltaRA'] *= u.arcsec
    tpm['deltaDEC'] *= u.arcsec
    tpm['RA'] *= u.deg
    tpm['DEC'] *= u.deg
    return tpm


@u.quantity_input(distance=u.pc, pm=u.mas/u.yr)
def pm_plot(planetname, distance, starcoords, pm, tpositions,
            labels, includeoffset=True, irefdate=0):
    '''Make a pm plot for the positions found from NaCo archive and ISPY. Assuming
    ISPY is the true position and calculationg backwards.
    Includeoffset=True
    Wether to include the offset of the CC. This is just relabeling the axis.
    positionstable = astropy.table.Table
    a table containing at least date, RA, RAerr, DEC, DECerr for the epochs.
    e.g. planetclass.cPositions provides this.'''
    ttimes = calprowrapper(planetname, distance, starcoords.ra, starcoords.dec,
                           pm[0], pm[1],
                           min(tpositions['date']), max(tpositions['date']),
                           tempdir=os.path.join(os.getcwd(), 'temp'))
    # use the negative value of deltaRA/DEC for the planets pm
    idxoff = _closest_in_list(ttimes['JD'].jd, tpositions['date'][irefdate].jd)
    raoff = -ttimes['deltaRA'].to("mas")[idxoff]
    decoff = -ttimes['deltaDEC'].to("mas")[idxoff]
    raref = tpositions['RA'].to('mas')[irefdate]
    decref = tpositions['DEC'].to('mas')[irefdate]
    # cmap = plt.get_cmap('jet')
    # colors = iter(cmap(np.linspace(0, 1, len(dates))))

    # plot the line of pm
    sns.set_style('darkgrid')
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

    plt.plot((-ttimes['deltaRA'].to('mas')-raoff+raref).to('mas').value,
             (-ttimes['deltaDEC'].to('mas')-decoff+decref).to('mas').value,
             color='k')
    for idate, date in enumerate(tpositions['date']):
        # color = next(colors)
        idx = _closest_in_list(ttimes['JD'].jd, tpositions['date'].jd[idate])
        # the measured positions
        xmeas = tpositions['RA'].to('mas')[idate].value
        ymeas = tpositions['DEC'].to('mas')[idate].value
        plt.errorbar(xmeas,
                     ymeas,
                     xerr=tpositions['RAerr'].to('mas')[idate].value,
                     yerr=tpositions['DECerr'].to('mas')[idate].value,
                     marker='x',
                     label=labels[idate],
                     color='C{}'.format(idate),
                     zorder=2)
        # the calculated positions
        if idate != irefdate:
            xcalc = (-ttimes['deltaRA'].to('mas')[idx]-raoff+raref).to('mas').value
            ycalc = (-ttimes['deltaDEC'].to('mas')[idx]-decoff+decref).to("mas").value
            plt.scatter(
                xcalc,
                ycalc,
                marker='o', s=15,
                color='C{}'.format(idate),
                zorder=2.5)
            # also plot connecting lines
            plt.plot([xmeas, xcalc],
                     [ymeas, ycalc],
                     ls=':', color='C{}'.format(idate))
    plt.ylabel('DEC [mas]')
    plt.xlabel('RA [mas]')
    plt.title('{} proper motion analysis ({} pc)'.format(
        planetname, round(distance.to('pc').value)))
    plt.gca().invert_xaxis()
    plt.gca().set_aspect('equal')
    plt.legend()
    pnsave = os.path.join(os.getcwd(), 'pm_{}.pdf'.format(planetname))
    print('Saving PM plot to {}'.format(pnsave))
    plt.tight_layout()
    plt.savefig(pnsave,
                bbox_inches='tight',
                pad_inches=0)
    plt.close()
    return ttimes

def _closest_in_list(alist, value):
    '''Return the index of the closest value in the list'''
    return min(range(len(alist)), key=lambda i: abs(alist[i] - value))
