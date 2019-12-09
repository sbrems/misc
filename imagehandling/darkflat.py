import numpy as np
from astropy.stats import sigma_clip
from tqdm import tqdm
from scipy.stats import linregress
from fits_processing import read_fits, filter_header


def sort_darks(darkdir, hexpt, filterheaderkey=None, filterheadervalue=None,
               fileend='.fits'):
    '''returns the median of all same exposures and a list with the
    exposure times (sorted).
    Give filterheadekey and filterheadervalue to filter the files for a
    keyword, e.g. a filtername.'''
    fns, data, headers = read_fits(darkdir, fileend=fileend)
    if filterheaderkey is not None:
        data, headers, usedidz = filter_header(data, headers,
                                               filterheadervalue,
                                               headerkeyword=filterheaderkey,
                                               return_indices=True)
        fns = fns[usedidz]
    exptimes = []
    for head in headers:
        exptimes.append(head[hexpt])
    uexpts = sorted(np.unique(exptimes))
    darks = np.full(np.hstack((len(uexpts), data[0, :, :].shape)), np.nan)
    for ii, uexpt in enumerate(uexpts):
        darks[ii, :, :] = np.median(
            data[np.where(exptimes == uexpt)[0], :, :], axis=0)
    del data
    return darks, uexpts


def subtract_closest_dark(data, dataTexp, ddata, dTexp):
    '''Subtracts darks using the nearest dark exposure time'''
    dTexp = np.array(dTexp)
    for ii, dtime in enumerate(dataTexp):
        idtuse = (np.abs(dTexp - dtime)).argmin()
        data[ii, :, :] -= ddata[idtuse, :, :]
    return data


def divide_closest_flat(data, dataTexp, fdata, fTexp):
    fTexp = np.array(fTexp)
    for ii, time in enumerate(dataTexp):
        iftuse = (np.abs(fTexp - time)).argmin()
        if len(data.shape) == 3:
            for iframe in range(data.shape[0]):
                data[iframe, :, :] /= fdata[iftuse, :, :]
        elif len(data.shape) == 2:
            data /= fdata[iftuse, :, :]
        else:
            raise ValueError('Data has unknown shape')
    return data


def make_masterflat(flats, return_bpm=True, sigma=4):
    '''Assuming the flats are sorted. E.g. in increasing
    or decreasing intensity.
    Make the masterflat by fitting a linear function to all flats.
    If return bpm, then clip the
    Values which are of. Using for loops. So slow.
    Returns a matrix which is 1 where a bp is found, 0 else.'''
    bpm = np.full(flats.shape[1:], 0).astype(int)
    gradmap = np.full(flats.shape[1:], np.nan)
    nflats = flats.shape[0]
    iflats = np.linspace(0, nflats - 1, nflats).astype(int)
    print('Making masterflat (and BPM)')
    for yy in tqdm(range(flats.shape[1])):
        for xx in range(flats.shape[2]):
            # filter cosmic rays
            values = sigma_clip(flats[:, yy, xx], sigma=5)
            gradmap[yy, xx] = linregress(iflats[~values.mask],
                                         values[~values.mask])[0]
    if return_bpm:
        bpm[sigma_clip(gradmap, sigma=sigma).mask] = 1
        return gradmap, bpm
    else:
        return gradmap
