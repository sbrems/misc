# -*- coding: utf-8 -*-
"""
DACE - download targets

author: Henrik Ruh
"""

from dace import Dace
import os
import numpy as np
from astropy.io import fits
# from astropy import wcs
#from astropy.table import Table
import math
import pyds9

import ipdb


def download(targets):

    # name of txt file with targets list
    if not (isinstance(targets, list) or type(targets) == np.ndarray):
        targets = [
            targets,
        ]
    #  location for saving fits files
    dataloc = os.path.join(
        os.path.expanduser("~"),
        'Documents',
        'NACO',
        'reanalysis2019',
    )

    # create fitsloc if it does not exist
    for tar in targets:
        tarloc = os.path.join(dataloc, tar)
        if not os.path.exists(tarloc):
            os.makedirs(tarloc)

        # retreive info on targets from dace
        targetinfo = Dace.search_observations(pattern=tar)
        datainfo = Dace.retrieve_imaging_dataset_infos(tar)

        # download files

        for fitsname in datainfo['file_rootpath']:
            outputloc = os.path.join(dataloc, tar, fitsname)
            Dace.retrieve_imaging_data(fitsname, "HC", outputloc)

        for fitsname in datainfo['file_rootpath']:

            fitsloc = os.path.join(dataloc, tar, fitsname)
            with fits.open(fitsloc, mode='update') as hdul:

                dat = hdul[0].data
                s = np.shape(dat)

                # expand dimensions of image data, so all images have the same size
                if len(s) == 2:
                    s1 = s[0]
                    s2 = s[1]

                    # create an empty matrix
                    newdat = np.nan * np.ones([1500, 1500])
                    # fill central entries with data
                    newdat[750 - math.ceil(s1 / 2 - 0.5):750 +
                           math.ceil(s1 / 2), 750 -
                           math.ceil(s2 / 2 - 0.5):750 +
                           math.ceil(s2 / 2)] = dat

                else:
                    # if data cube
                    s0 = s[0]
                    s1 = s[1]
                    s2 = s[2]
                    # create an empty matrix
                    newdat = np.nan * np.ones([s[0], 1500, 1500])
                    # fill central part of the matrix with data
                    newdat[:, 750 - math.ceil(s1 / 2 - 0.5):750 +
                           math.ceil(s1 / 2), 750 -
                           math.ceil(s2 / 2 - 0.5):750 +
                           math.ceil(s2 / 2)] = dat

                # save new data in original fits
                hdul[0].data = newdat

                if hdul[0].header['NAXIS1'] != 1500:
                    # change reference pixel to centre
                    na1 = hdul[0].header['NAXIS1']
                    na2 = hdul[0].header['NAXIS2']
                    hdul[0].header['CRPIX1'] = 750 + na1 / 2
                    hdul[0].header['CRPIX2'] = 750 + na2 / 2
                    # change image size in header
                    hdul[0].header['NAXIS1'] = 1500
                    hdul[0].header['NAXIS2'] = 1500

        #%% open files for one target in ds9

        # crude sorting for better display
        fitsnames = [
            os.path.join(dataloc, tar, f)
            for f in sorted(datainfo['file_rootpath'])
        ]

        #ds9str = ds9str + ' -blink interval 0.8'
        ds9 = pyds9.DS9()
        ds9.set('frame delete all')
        for ffn in fitsnames:
            ds9.set('frame new')
            ds9.set("file {}".format(ffn))

            # draw circle
            radii = np.hstack((0.1, np.arange(0.5, 15, 1)))  # radii in arcsec
            for r in radii:
                ds9.set(
                    "regions command {circle 751 751 %.5f # width=1, color=white}"
                    % (r / 0.027))
                # annotate
                ds9.set(
                    'regions ',
                    "image; text 751 %i # text={%.1f''},color=white" %
                    (751 - int(r / 0.027), r))
        ds9.set('single yes')
        ds9.set("frame lock image")
        ds9.set("crosshair lock image")
        print(f'Displaying target {tar}')
        ipdb.set_trace()
