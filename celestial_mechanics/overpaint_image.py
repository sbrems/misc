import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from .proper_motion import calprowrapper


@u.quantity_input(pxscale=u.mas, coordinates=u.mas)
def overpaint(planet, image, imdate, refdate, coordinates, xyimstar=None,
              pxscale=27.02*u.mas, plot_startrail=True, pnsave=None,
              plotlimits=None):
    '''Overpaint the position of (multiple) companions.
    Draw the original position and the new calculated one.
    planet:
    a planetclass object containing the stars properties such as
    parallax, pmRA,...
    image:
    a np array containing the image properties
    imdate: astropy.time.Time object
    date of the image
    refdate: astropy.time.Time object
    date of the observation of the coordinates
    coordinates:
    Astropy u.mas array of shape nx2. The first entry is the starposition,
    e.g. the relative position
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
    ttimes = calprowrapper(planet.cname, planet.distance,
                           planet.coordinates.ra, planet.coordinates.dec,
                           planet.pm[0], planet.pm[1],
                           imdate, refdate,
                           tempdir=os.path.join(os.getcwd(), 'temp'))
    if xyimstar is None:
        xyimstar = np.rint(np.array(image.shape[-2:])/2.)
    xyimstar = np.array(xyimstar)
    plt.imshow(image, origin='lower', vmin=plotlimits[0], vmax=plotlimits[1])
    # plot the starposition
    plt.scatter(xyimstar[0], xyimstar[1], marker='*', c='k')
    # get the movement of the star
    xystarmoved = np.array([(ttimes['deltaRA'].to('mas')[-1]  / pxscale).decompose(),
                            (ttimes['deltaDEC'].to('mas')[-1] / pxscale).decompose()])
    if imdate.mjd > refdate.mjd:
        xystarmoved = -xystarmoved
    if plot_startrail:
        # plot the track of the star
        print('Plotting startrail')
        if imdate.mjd > refdate.mjd:
            plt.plot((ttimes['deltaRA'] / pxscale).decompose()+xyimstar[0],
                     (ttimes['deltaDEC'] / pxscale).decompose()+xyimstar[1],
                     c='orange', linewidth=0.3)
        else:
            plt.plot((-ttimes['deltaRA'] / pxscale).decompose()+xyimstar[0],
                     (-ttimes['deltaDEC'] / pxscale).decompose()+xyimstar[1],
                     c='orange')

    for coord in coordinates[1:]:
        pxcoords = xyimstar + ((coord - coordinates[0]) / pxscale).decompose()
        plt.scatter(pxcoords[0], pxcoords[1],
                    marker='s', alpha=0.3,
                    linewidths=0.3,
                    edgecolor='red', facecolor='none',
        )
        plt.scatter(pxcoords[0]+xystarmoved[0],
                    pxcoords[1]+xystarmoved[1],
                    marker='o', facecolors='none',
                    edgecolors='r', alpha=0.7,
                    linewidths=0.3)
    if pnsave is None:
        pnsave = os.path.join(os.getcwd(), 'proper_motion_im.pdf')
    plt.title("{} ({} and {})".format(planet.sname,
                                  imdate.iso[:10],
                                  refdate.iso[:10]))
    plt.savefig(pnsave)
    print('Saved fig to {}'.format(pnsave))
    import ipdb;ipdb.set_trace()
    plt.close()
