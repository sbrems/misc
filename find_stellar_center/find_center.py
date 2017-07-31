import numpy as np
import os
from astropy.io import fits
# from gaussfitter.gaussfitter import gaussfit
from read_fits import read_fits
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
from astropy.table import Table
from scipy.stats import sigmaclip
from fft_rotate import fft_3shear_rotate_pad as fftrotate
from .fit_gauss import find_centers

fwhm = 7.
sigma = 5.
# y,x coordinates of the companions
dtar2fitpts = {'HD179218': [[236, 242], [410, 507]],
               'HD100453': [[274, 271], ],
               'HD101412': [[256, 386], [284, 299]],
               'HD259431': [[409, 333], ],
               # HD 72106 looks like a binary. probably centering was very bad. Also nan frame found, rotation angle seems unstable and inner cutout area varies
               'HD72106': [[327, 288], ],
               'HD92536': [[304, 330], ],
               # KKoph has same issue as HD72106
               'KKOph': [[274, 353], ],
               'V1032Cen': [[339, 235], ],
               'V4046Sgr': [[137, 426], ]
               }


def do_all():
    tpos = {}
    for target in dtar2fitpts.keys():
        tpos[target] = do(fitpoints=dtar2fitpts[target],
                          target=target,
                          dir_data=os.path.join(os.getcwd(), target))
    import ipdb
    ipdb.set_trace()


def do(fitpoints=None, fitmode='static', target=None,
       dir_data=None, dir_out=None):
    '''Find the center of a spot. if fitmode 'circular' if fits a circle and
    uses the last center as guess for the new one. if static, it stays at the
    same position and uses the center'''
    if dir_data is None:
        dir_data = os.getcwd()
    if dir_out is None:
        dir_out = os.path.join(dir_data, 'results')
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    try:
        fns, derotcube, headercube = read_fits(
            dir_data, fileend='_derotated.fits')
        print('Found derotated cube')
        derotate = False
    except:
        # no derotated images found
        print('No derotated images found. Derotating them')
        fns, imcube, headercube = read_fits(dir_data, fileend='.fits')
        derotate = True

    if target is None:
        target = fns[0].split('.')[0].split('_')[0]
    print('Processing target {}'.format(target))
    if fitpoints is None:
        fitpoints = dtar2fitpts[target]
    files = os.listdir(dir_data)
    for ff in files:
        if ff.endswith('_parangs.txt'):
            parangs = np.loadtxt(os.path.join(dir_data, ff))
            parangs = - parangs  # for fft
    if derotate:
        # derotate the cube. The center is at shape/2 now!
        print('Derotataing {} frames'.format(imcube.shape[0]))
        derotcube = []
        for iframe, pa, image in zip(range(imcube.shape[0]), parangs, imcube):
            if iframe % 10 == 0:
                print('Derotataing frame {} of {}'.format(
                    iframe, imcube.shape[0]))
            derotcube.append(fftrotate(image, pa))
        derotcube = np.array(derotcube)
        fits.writeto(os.path.join(dir_data, target + '_derotated.fits'),
                     derotcube)
    if fitpoints == [[np.nan, np.nan], ]:
        fitpoints = eval(input('Give the coordinates of the companions,\
for {} e.g. [[274, 271], ]: '.format(target)))
    nfitpoints = len(fitpoints)

    all_centers = []
    all_sigmas = []
    all_errors = []
    for fitpoint in fitpoints:
        print('Finding the centers for point {}'.format(fitpoint))
        centers, sigmas, errors, fitims = find_centers(derotcube, fitpoint,
                                                       fitmode=fitmode,
                                                       fwhm=fwhm)
        all_centers.append(centers)
        all_sigmas.append(sigmas)
        all_errors.append(errors)
    fits.writeto(os.path.join(dir_out, 'cut_cube.fits'), fitims,
                 overwrite=True)
    print('Done finding centers. Plotting them now')
    all_centers = np.array(all_centers)
    all_sigmas = np.array(all_sigmas)
    all_errors = np.array(all_errors)

    tpos = Table(meta={'name': target,
                       'npoints': nfitpoints}, masked=True)
    for ipoint in range(nfitpoints):
        tpos['ycenter_{}'.format(ipoint)] = all_centers[ipoint, :, 0]
        tpos['xcenter_{}'.format(ipoint)] = all_centers[ipoint, :, 1]
        tpos['sigmafit_{}'.format(ipoint)] = all_sigmas[ipoint, :]
        tpos['yerrorfit_{}'.format(ipoint)] = all_errors[ipoint, :, 0]
        tpos['xerrorfit_{}'.format(ipoint)] = all_errors[ipoint, :, 1]

        # do sigma clipping and mask the values
        sigy, miny, maxy = sigmaclip(tpos['ycenter_{}'.format(ipoint)],
                                     low=sigma, high=sigma)
        sigx, minx, maxx = sigmaclip(tpos['xcenter_{}'.format(ipoint)],
                                     low=sigma, high=sigma)
        idz_y = np.where((tpos['ycenter_{}'.format(ipoint)] > miny) &
                         (tpos['ycenter_{}'.format(ipoint)] < maxy))
        idz_x = np.where((tpos['xcenter_{}'.format(ipoint)] > minx) &
                         (tpos['xcenter_{}'.format(ipoint)] < maxx))
        idz_valid = np.intersect1d(idz_x, idz_y)
        mask = np.array([True] * len(tpos))
        mask[idz_valid] = False

        tpos['ycenter_{}'.format(ipoint)].mask = mask
        tpos['xcenter_{}'.format(ipoint)].mask = mask
        tpos['sigmafit_{}'.format(ipoint)].mask = mask
        tpos['yerrorfit_{}'.format(ipoint)].mask = mask
        tpos['xerrorfit_{}'.format(ipoint)].mask = mask

        tpos.meta['std_{}'.format(ipoint)] =\
            np.sqrt(np.std(tpos['ycenter_{}'.format(ipoint)])**2 +
                    np.std(tpos['xcenter_{}'.format(ipoint)])**2)

    # test the movement of the points relative if present
    if nfitpoints == 2:
        print('As there were 2 companions, testing their relative motion')
        tpos['ycenter_diff'] = tpos['ycenter_0'] - tpos['ycenter_1']
        tpos['xcenter_diff'] = tpos['xcenter_0'] - tpos['xcenter_1']
        tpos.meta['std_diff'] = np.sqrt(np.std(tpos['ycenter_diff'])**2 +
                                        np.std(tpos['xcenter_diff'])**2)
    tpos.write(os.path.join(dir_out, 'Companions_positions.csv'),
               format='ascii', delimiter=',', overwrite=True)
    # make the plots
    print('Plotting the results')
    # colors = iter(cm.rainbow(np.linspace(0, 1, 2)))
    for ipoint in range(nfitpoints):
        plt.scatter(tpos['xcenter_{}'.format(ipoint)],
                    tpos['ycenter_{}'.format(ipoint)],
                    s=2, label='Position CC \
nr {} (std={:.4f})'.format(ipoint,
                           tpos.meta['std_{}'.format(ipoint)]),
                    c='red')
        plt.errorbar(np.min(tpos['xcenter_{}'.format(ipoint)]),
                     np.min(tpos['ycenter_{}'.format(ipoint)]),
                     yerr=np.median(tpos['xerrorfit_{}'.format(ipoint)]),
                     xerr=np.median(tpos['yerrorfit_{}'.format(ipoint)]),
                     label='Median errorbar of fit',
                     c='black')
        plt.legend()
        plt.title('Positions of {}.'.format(target))
        plt.xlabel('Position x [px]')
        plt.ylabel('Position y [px]')
        plt.savefig(os.path.join(dir_out, 'scatterplot_src{}.pdf'.format(ipoint)),
                    overwrite=True)
        plt.close('all')

    if nfitpoints == 2:
        plt.scatter(tpos['xcenter_diff'],
                    tpos['ycenter_diff'],
                    s=3, label='2 point difference (std={:.4f})'.format(
                        tpos.meta['std_diff']),
                    c='red')
        plt.errorbar(np.min(tpos['xcenter_diff'.format(ipoint)]),
                     np.min(tpos['ycenter_diff'.format(ipoint)]),
                     yerr=np.median(tpos['xerrorfit_{}'.format(ipoint)]),
                     # XXX: err=np.median(tpos['yerrorfit_{}'.format(ipoint)]),
                     label='Median errorbar of fit',
                     c='black')
        plt.legend()
        plt.title('Positions of {}.'.format(target))
        plt.xlabel('Position x [px]')
        plt.ylabel('Position y [px]')
        plt.savefig(os.path.join(dir_out, 'scatterplot_diff.pdf'.format(ipoint)),
                    overwrite=True)
        plt.close('all')
    return tpos
