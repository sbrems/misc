import numpy as np
from .bin_median import bin_median


def find_max_star(image):
    '''Median smooth an image and find the max pixel.
    The median smoothing helps filter hot pixels and
    cosmic rays. The median is taken by using bin_median
    with a smaller_by_factor=16'''
    image[np.isnan(image)] = np.median(image[~np.isnan(image)])
    binned = bin_median(image, smaller_by_factor=16)
    y, x = np.transpose((binned == binned.max()).nonzero())[0]
    y *= 16
    x *= 16
    while True:
        x0 = max(x - 15, 0)
        y0 = max(y - 15, 0)
        patch = image[y0:min(y + 15, image.shape[0]), x0:min(
            x + 15, image.shape[1])]
        dy, dx = np.transpose(np.where(patch == patch.max()))[0]
        dy -= (y - y0)
        dx -= (x - x0)
        y = min(max(y + dy, 0), image.shape[0] - 1)
        x = min(max(x + dx, 0), image.shape[1] - 1)
        if (dx == 0) and (dy == 0):
            break
    return y, x
