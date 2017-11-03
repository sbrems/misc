import numpy as np
from astropy.io import fits
import os

cam2head = {'red': 1, 'blue': 2}  # for nici data


def read_fits(directory, verbose=True, only_first=False, cam=None,
              fileend='.fits'):
    '''This routine reads all fits files data into a big data cube and all header files
    into a big header cube. The order is the same and is alphabetically.Filenames and headers
    are multiple if it was an imagecube. So all have the same length. So it returns:
    (fits)filenames,datacube,headercube.
    If you give the path of a fits-file,only this is read out.
    Also compatible with NICI data (2 cubes, 3headers)'''
    # to avoid buffer overflows we need the number images first, which is not the same as
    # filenumber, as some images are cubes and some are not
    n_images = 0
    form = []
    if directory.endswith('.fits'):
        files = [os.path.split(directory)[-1]]
        directory = os.path.dirname(directory) + '/'
    else:
        files = sorted(os.listdir(directory))
    for fl in files:
        if fl.endswith(fileend):
            form = fits.getdata(os.path.join(directory, fl)).shape
            if len(form) == 3:  # image cube
                if only_first:
                    n_images += 1
                else:
                    n_images += form[0]
            elif len(form) == 2:  # one image
                n_images += 1
            else:
                raise ValueError('Fits file has unknown format!')
    if verbose:
        print('Found ', n_images, ' frames in ', directory)
    # now make the array
    filenames = []
    headers = []
    # float16 for memory
    all_data = np.full(
        (n_images, form[-2], form[-1]), np.nan, dtype=np.float64)
    n = 0
    for fl in files:
        if fl.endswith(fileend):
            hdulist = fits.open(os.path.join(directory, fl))
            # check hdu shape
            if cam != None:
                if len(hdulist) != 3:  # nici special
                    raise ValueError(
                        'Unknown file format for NICI camera!file:%s' % fl)
                header = hdulist[0].header + hdulist[cam2head[cam]].header
                data = hdulist[cam2head[cam]].data
            else:
                if len(hdulist) != 1:
                    raise ValueError('Unknown file format at file %s' % fl)
                header = hdulist[0].header
                data = hdulist[0].data
            # check data shape
            if (len(data.shape) == 3) & (only_first):
                data = data[0, :, :]
            if len(data.shape) == 3:  # image cube
                all_data[n:n + data.shape[0], :, :] = data
                headers.extend(header for ii in range(data.shape[0]))
                filenames.extend(fl for ii in range(data.shape[0]))
                n += data.shape[0]
            elif len(data.shape) == 2:  # one image
                all_data[n, :, :] = data
                headers.append(header)
                filenames.append(fl)
                n += 1
            else:
                raise ValueError('Fits file has unknown format!')

    return filenames, all_data, headers
