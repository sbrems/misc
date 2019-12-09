import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from .proper_motion import calprowrapper

@u.quantity_input(pxscale=u.mas, coordinates=u.mas)
def overpaint(planet, image,
              irefdate=0, xyimstar=None,
              pxscale=27.02*u.mas, plot_startrail=True, pnsave=None,
              plotlimits=None):
    '''Overpaint the position of (multiple) companions.
    Draw the original position and the new calculated one.
    planet:
    a planetclass object containing the stars properties such as
    parallax, pmRA,...
    image:
    a np array containing the image properties

    planet
    an instance of the planetclass. needs coordinates and
    cPositions table with companioninfo to calculate the
    respective positions.
    xyimstar = np.rint([imshape/2., imshape/2.])
    position of the star in the image. Start counting at 0
    pxscale = 27.02* u.mas/u.px
    the pixelscale of the image. An astropy quantity. Assuming image
    is in standard orientation (e.g. east left, north up)
    plot_startrail=True
    plot the movement of the star in the meantime
    plotlimits=[-1, 2.5]
    set to [limlow, limup] to cut the image colors.
    Default is [-1, 2.5]'''
    print('Assuming north up, east left. Pixelscale is {}'.format(pxscale))
    tcPos = planet.cPositions
    imdate = tcPos['date'][-1]
    refdate = tcPos['date'][0]
    print('Assuming first date ({}) is refdate and last date ({}) is the image'.format(
        refdate.isot, imdate.isot))
    ttimes = calprowrapper(planet.cname, planet.distance,
                           planet.coordinates.ra, planet.coordinates.dec,
                           planet.pm[0], planet.pm[1],
                           np.min(tcPos['date']), np.max(tcPos['date']),
                           tempdir=os.path.join(os.getcwd(), 'temp'))
    # remove the offset to the reference date
    refidx = np.argmin(np.abs(ttimes['JD'] - tcPos['date'][irefdate]))
    for col in ['deltaRA', 'deltaDEC']:
        ttimes[col] = ttimes[col] - ttimes[col][refidx]
    ttimes['RA'] = ttimes['deltaRA'] - ttimes['deltaRA'][refidx]
    ttimes['DEC'] = ttimes['deltaDEC'] - ttimes['deltaDEC'][refidx]
    if xyimstar == 'center':
        xyimstar = np.rint(np.array(image.shape[-2:])/2.)
    xyimstar = np.array(xyimstar)
    plt.imshow(image, origin='lower', vmin=plotlimits[0], vmax=plotlimits[1])
    # plot the starposition
    plt.scatter(xyimstar[0], xyimstar[1], marker='*', c='k')
    # get the movement of the star
    xystarmoved = np.array([(ttimes['deltaRA' ].to('mas')[-1] / pxscale).decompose(),
                            (ttimes['deltaDEC'].to('mas')[-1] / pxscale).decompose()])
    if plot_startrail:
        # plot the track of the star
        print('Plotting startrail')
        plt.plot(((ttimes['deltaRA']-tcPos['RA'][irefdate]) / pxscale).decompose()+xyimstar[0],
                 ((-ttimes['deltaDEC']+tcPos['DEC'][irefdate]) / pxscale).decompose()+xyimstar[1],
                 c='orange', linewidth=1, alpha=.7)
    cmap = plt.get_cmap('jet_r')
    refpx = (np.array((-tcPos['RA'][irefdate]*tcPos['RA'].unit.to('arcsec'), \
                       tcPos['DEC'][irefdate]*tcPos['DEC'].unit.to('arcsec'))) *\
             u.arcsec/pxscale).decompose() + \
             (np.array((ttimes['deltaRA'][refidx]*ttimes['deltaRA'].unit.to('arcsec'),
                        -ttimes['deltaDEC'][refidx]*ttimes['deltaDEC'].unit.to('arcsec'))) * \
              u.arcsec/pxscale).decompose() +\
              xyimstar
    plt.errorbar(refpx[0], refpx[1],
                 xerr=(tcPos['RAerr'][irefdate]*tcPos['RAerr'].unit/pxscale).decompose(),
                 yerr=(tcPos['DECerr'][irefdate]*tcPos['DECerr'].unit/pxscale).decompose(),
                 fmt='x', c='k', ecolor='k', zorder=2,
                 ms=3, alpha=.7, lw=.6,
                 label='{:.2f} (ref)'.format(tcPos['date'][irefdate].jyear))
    for iPos, cPos in enumerate(tcPos):
        if iPos == irefdate:
            continue
        idx = np.argmin(np.abs(ttimes['JD'] - cPos['date']))
        pxRaDec = (np.array((-cPos['RA']*tcPos['RA'].unit.to('arcsec'),
                             cPos['DEC']*tcPos['DEC'].unit.to('arcsec')))*u.arcsec/pxscale).decompose() + xyimstar
        dpxRaDec = (np.array((ttimes['deltaRA'][idx]*ttimes['deltaRA'].unit.to('arcsec'),
                              -ttimes['deltaDEC'][idx]*ttimes['deltaDEC'].unit.to('arcsec')))*\
                    u.arcsec/pxscale).decompose()
        plt.errorbar(pxRaDec[0], pxRaDec[1],
                     xerr=(cPos['RAerr']*tcPos['RAerr'].unit/pxscale).decompose(),
                     yerr=(cPos['DECerr']*tcPos['DECerr'].unit/pxscale).decompose(),
                     fmt='x', alpha=0.7,
                     lw=.6,
                     ms=3,
                     c=cmap(0.5*iPos/len(tcPos)),
                     ecolor=cmap(0.5*iPos/len(tcPos)),
                     label='{:.2f} (meas)'.format(cPos['date'].jyear),
                     zorder=10)    
        plt.scatter(refpx[0]+dpxRaDec[0], refpx[1]+dpxRaDec[1],
                    marker='o', edgecolors='none',
                    facecolors=cmap(0.5*iPos/len(tcPos)), alpha=1,
                    label='{:.2f} (pred)'.format(cPos['date'].jyear),
                    linewidths=.5,
                    s=5, zorder=9)
        # plot connecting line
        plt.plot([pxRaDec[0], refpx[0]+dpxRaDec[0]],
                 [pxRaDec[1], refpx[1]+dpxRaDec[1]],
                 lw=1, ls=':', c=cmap(0.5*iPos/len(tcPos)),
                 zorder=1)

    if pnsave is None:
        pnsave = os.path.join(os.getcwd(), 'proper_motion_im.pdf')

    plt.title("{} ({} and {})".format(planet.sname,
                                      imdate.iso[:10],
                                      refdate.iso[:10]))
    plt.legend()
    plt.savefig(pnsave)
    print('Saved fig to {}'.format(pnsave))

    plt.close()
