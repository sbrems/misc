import numpy as np


def outer(cube):
    '''Cut the imagecube so all data not being nan is conserved.
    Not necessarily squared.'''
    assert len(cube.shape) == 3
    im = np.abs(np.nanmean(cube, axis=0))
    validcols = np.where(np.nansum(im, axis=0) > 0.)
    validrows = np.where(np.nansum(im, axis=1) > 0.)
    xmin = np.min(validrows)
    xmax = np.max(validcols)
    del validcols
    ymin = np.min(validrows)
    ymax = np.max(validrows)
    del validrows

    return cube[:, ymin:ymax, xmin:xmax]
