import numpy as np
import os
# from astropy.table import Table
from multiprocessing import Pool
from scipy.ndimage.interpolation import shift
from scipy.signal import fftconvolve
from gaussfitter import gaussfit
from .find_max_star import find_max_star

# from params import *


class subreg:
    def __init__(self, reference):
        self.reference = reference

    def __call__(self, im):
        kernel = self.reference[::-1, ::-1]
        cor = fftconvolve(im, kernel, mode='same')
        y, x = find_max_star(cor)
        g = gaussfit(cor[max(0, y - 40):min(y + 40, cor.shape[0]),
                         max(0, x - 40):min(x + 40, cor.shape[1])])
        shiftx = np.rint(cor.shape[1] / 2.) - max(0, x - 40) - g[2]
        shifty = np.rint(cor.shape[0] / 2.) - max(0, y - 40) - g[3]
        # shifts.append((shifty,shiftx))
        return (shifty, shiftx)


def _get_center(image):
    indices = np.indices(image.shape)
    test = np.where((indices[0] - 150)**2 + (indices[1] - 150)**2 < 10**2)
    m = np.zeros_like(image)
    m[test] = 1
    smoothed = fftconvolve(image, m, 'same')
    y, x = np.where(smoothed == np.max(smoothed))
    return x[0], y[0]


def align_images(images,
                 xystarpositions,
                 #outdir=None,
                 ncpu=1,
                 keepfrac=0.7,
                 alignarea=30,
                 pxhalf=np.inf):
    '''Align images and select the best keepfrac ones.
    NOTE: IT DOES NOT CENTER THE IMAGES, ONLY ALIGN THEM VIA CC
    Returns the aligned images and the indices which were selected.
    images:
    3-dim np.array of the images to be aligned
    xystarpositions:
    (nstars,2)-dimensional array giving the approximate stellar position
    pxhalf=np.inf:
    area which is cut for the alignment. With pxhalf you select the framesize
    remaining
    alignarea:
    px right and left of image. final size 2*pxhalf+1. Set to np.inf to have
    the full frame
    '''
    if not len(images.shape) == 3:
        raise ValueError('Unknown data format {} of images'.format(
            images.shape))
    #if outdir is None:
    #    outdir = os.getcwd()
    if not np.isfinite(pxhalf):
        # also do pxhalf to get a quick alignment. The full image
        # usually takes forever
        fullframe = True
        pxhalf = alignarea
    else:
        fullframe = False
    imagecut = np.full([images.shape[0], 2 * pxhalf + 1, 2 * pxhalf + 1],
                       np.nan)
    for ii, image in enumerate(images):
        starx = xystarpositions[ii, 0]
        stary = xystarpositions[ii, 1]
        imagecut[ii, ::] = image[max(0, stary - pxhalf):min(stary + pxhalf + 1,
                                                            image.shape[1]),
                                 max(0, starx - pxhalf):min(starx + pxhalf + 1,
                                                            image.shape[0]), ]
    if fullframe:
        origims = np.copy(images)
    images = imagecut

    #/////////////////////////////////////////////////////////
    #median combine and first xreg
    #/////////////////////////////////////////////////////////

    print('register number getting medianed: ', len(images))
    print(images[0].size)
    print(images[0].shape)
    print(images[0].dtype)
    first_median = np.median(images, axis=0)

    pool = Pool(ncpu)
    get_shifts = subreg(first_median)
    shifts = pool.map(get_shifts, images)
    first_shifts = np.copy(shifts)
    pool.close()

    for hh in range(len(images)):
        images[hh] = shift(images[hh], shifts[hh], cval=np.nan)

    #/////////////////////////////////////////////////////////
    #keep only the best of images
    #/////////////////////////////////////////////////////////
    if keepfrac < 1:
        cross_reg = []
        for im in images:
            cross_reg.append(np.sum((im - first_median)**2.))

        sorted_cross_reg = np.argsort(cross_reg)
        selected_cross_reg = sorted_cross_reg[0:int(keepfrac * len(images))]
        first_shifts = first_shifts[selected_cross_reg]
        xystarpositions[selected_cross_reg]
        n_selected = len(selected_cross_reg)
        print('Selecting the {} best images'.format(n_selected))

        #/////////////////////////////////////////////////////////
        #median combine and second xreg
        #/////////////////////////////////////////////////////////

        images = np.array(images)[selected_cross_reg, :, :]
        second_median = np.median(images, axis=0)

        print('second subreg')
        pool = Pool(ncpu)
        get_shifts = subreg(second_median)
        shifts = pool.map(get_shifts, images)
        # get center for images. Should be pxhalf. Needed for absolute centering
        #xycen = []
        #for h in range(n_selected):
        #    xycen.append(_get_center(images[h, :, :]))
        #    yxcenshift = np.median(xycen, axis=0)[::-1] - [pxhalf, pxhalf]
        pool.close()

        for h in range(n_selected):
            #shifts[h] += xycenshift
            images[h, :, :] = shift(images[h, :, :], shifts[h], cval=np.nan)
        second_shifts = np.copy(shifts)
    else:
        selected_cross_reg = np.arange(len(images))
        n_selected = len(selected_cross_reg)
        second_shifts = np.full([n_selected, 2], 0.0)

    if fullframe:
        assert origims.shape[2] == origims.shape[1]
        sh = origims.shape[1]
        if sh % 2 == 0:
            addsh = 0.
        else:
            addsh = 0.

        yxfullshifts = np.full([n_selected, 2], np.nan)
        fullims = np.full([n_selected,
                           3*sh, 3*sh],
                          np.nan)
        for ii, im in enumerate(origims):
            yxfullshifts[ii, :] = (-xystarpositions[ii][::-1] +
                                   [sh//2, sh//2] + addsh) + \
                                   first_shifts[ii] + second_shifts[ii]
            yxint = np.array(np.round(yxfullshifts[ii]), dtype=np.int)
            fullims[ii,
                    sh+yxint[0]: 2*sh+yxint[0],
                    sh+yxint[1]: 2*sh+yxint[1]] = shift(
                        im,
                        yxfullshifts[ii] - yxint,
                        cval=np.nan)
            import matplotlib.pyplot as plt
            plt.imshow(fullims[ii])
        images = fullims

    #/////////////////////////////////////////////////////////
    #save
    #/////////////////////////////////////////////////////////
    # images = np.stack(images, axis=0)
    # PAs_sel = PAs[selected_cross_reg]
    # filet_sel = filetable[selected_cross_reg]
    # filet_sel['orig_nr'] = selected_cross_reg
    # fits.writeto(os.path.join(outdir, 'center_im_sat.fits'), images)
    # fits.writeto(os.path.join(outdir, 'rotnth.fits'), PAs_sel)

    # print('Done Aligning stars. SAVED ROTATED IMAGES AND ANGLES IN {}. \
    # Their shape is {}'.format(outdir, images.shape))
    return images, selected_cross_reg
