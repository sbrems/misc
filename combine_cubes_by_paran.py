from __future__ import print_function,division
import os
import numpy as np
from astropy.io import fits

def combine(dir1,dir2,fn_ang='rotnth.fits',fn_dat='center_im.fits',dir_out=None,clobber=False):
    '''Combines two sets of cubes by sorting them by the paralactic angle. Use header from files 1'''
    if not dir1.endswith('/'):
        dir1 += '/'
    if not dir2.endswith('/'):
        dir2 += '/'
    if dir_out==None:
        dir_out = dir1+'combined/'

    data1,head1 = fits.getdata(dir1+fn_dat,header=True)
    data2,head2 = fits.getdata(dir2+fn_dat,header=True)
    angs1,head_angs1 = fits.getdata(dir1+fn_ang,header=True)
    angs2,head_angs2 = fits.getdata(dir2+fn_ang,header=True)

    angles = np.hstack((angs1,angs2))
    data = np.vstack((data1,data2))
    sorting = np.argsort(angles)
    
    angles = angles[sorting]
    data = data[sorting,:,:]


    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    fits.writeto(dir_out+fn_dat,data,header=head1,clobber=clobber)
    fits.writeto(dir_out+fn_ang,angles,header=head_angs1,clobber=clobber)
    
    print('Merged files from ',dir1,dir2,' to ', dir_out,' successfully.')
