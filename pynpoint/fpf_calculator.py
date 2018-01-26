__author__ = 'Arianna'

from astropy.io import fits
import numpy as np
from numpy import math
import matplotlib.pyplot as plt
from side_functions import angular_coords_float,planets_finder,twoD_Gaussian
from scipy.stats import t
import scipy.optimize as opt
import math
from astropy.table import Table
from photutils import geometry#, aperture_funcs
import photutils as ph


def get_phot_extents(data, positions, extents):
    """
        Get the photometry extents and check if the apertures is fully out of data.
        
        Parameters
        ----------
        data : array_like
        The 2-d array on which to perform photometry.
        
        Returns
        -------
        extents : dict
        The ``extents`` dictionary contains 3 elements:
        
        * ``'ood_filter'``
        A boolean array with `True` elements where the aperture is
        falling out of the data region.
        * ``'pixel_extent'``
        x_min, x_max, y_min, y_max : Refined extent of apertures with
        data coverage.
        * ``'phot_extent'``
        x_pmin, x_pmax, y_pmin, y_pmax: Extent centered to the 0, 0
        positions as required by the `~photutils.geometry` functions.
        """
    
    # Check if an aperture is fully out of data
    ood_filter = np.logical_or(extents[:, 0] >= data.shape[1],
                               extents[:, 1] <= 0)
    np.logical_or(ood_filter, extents[:, 2] >= data.shape[0],
                                             out=ood_filter)
    np.logical_or(ood_filter, extents[:, 3] <= 0, out=ood_filter)
                               
                               # TODO check whether it makes sense to have negative pixel
                               # coordinate, one could imagine a stackes image where the reference
                               # was a bit offset from some of the images? Or in those cases just
                               # give Skycoord to the Aperture and it should deal with the
                               # conversion for the actual case?
    x_min = np.maximum(extents[:, 0], 0)
    x_max = np.minimum(extents[:, 1], data.shape[1])
    y_min = np.maximum(extents[:, 2], 0)
    y_max = np.minimum(extents[:, 3], data.shape[0])
                               
    x_pmin = x_min - positions[:, 0] - 0.5
    x_pmax = x_max - positions[:, 0] - 0.5
    y_pmin = y_min - positions[:, 1] - 0.5
    y_pmax = y_max - positions[:, 1] - 0.5
                               
                               # TODO: check whether any pixel is nan in data[y_min[i]:y_max[i],
                               # x_min[i]:x_max[i])), if yes return something valid rather than nan
                               
    pixel_extent = [x_min, x_max, y_min, y_max]
    phot_extent = [x_pmin, x_pmax, y_pmin, y_pmax]
                               
    return ood_filter, pixel_extent, phot_extent



def fpf_calculator(image,filetype,planet_position,radius,method='exact',subpix=20,plot=False,
                 no_planet_nghbr=True,name='fpf_test',save='False',save_dir='None', j=None):


    ####### PARAMETERS:
    ## image : fits OR 2d-array : (PynPointed) image where to perform the SNR & FPF calculation (as an array or a single fits file)
                                # n.b.: during the whole analysis the image is assumed to be centered on the central star.
    ## filetype : string : format of the image. Possibilities are: 'fits' or 'array'
    ## planet_position : array of two values : rough position of the target in (integer) pixels, as an array: [x,y] (n.b.: if the
                                                #  'method' is 'exact' this position will be assumed to be the best position)
    ## radius : float : desired radius for the apertures (in pixels)
    ## method : string : how to search for the best position given the rough one. Possibilities are:
                        # 'exact' : the given planet_position is assumed to be the best one
                        # 'search' : the best position is found performing a search for the local maximum pixel value
                        # around the given planet_position in a square of side = (2 * r) * 2, rounded up
                        # (nb: the best position is found as integer pixels)
                        # 'fit' : the best position is found fitting a 2D gaussian on the given planet_position. The best
                        # position found in this way allows for floating pixels values.
    ## subpix : integer : how much refined the subpixel grid for the creation of the aperture should be. Each pixel is resampled in
                        #  subpix*subpix other pixels. (Quick test showed that after a value of 10-15, the final results are
                        # quite stable, so the default is 20).
    ## plot : boolean : whether to plot or not the final image together with the created apertures, the SNR and the FPF value
    ## no_planet_nghbr : boolean : whether to consider or not the two background apertures near the signal aperture.
    ## name : string : desired name with which save the final result (if save==True).
    ## save : boolean : whether or not save the results. If true, two images will be saved as fits files: the initial image
                        # with the apertures, and the initial image plus the gaussian fit.
    ## save_dir : string : path to a folder where to save (if save==True) the images as fits files

    ##### OUTPUT:
    ## the function returns:
    # fpf : float : value of the False Probability Fraction evaluated from the t_test value
    # t_test : float : value of the t_test
    # snr : float : value of the snr
    # planet_pos : array of two values : best planet position (evaluated with the desired method) in pixels
    # bckg_aps_cntr : multidimensional array : it contains (in pixel values) all the centers of ALL the apertures created
                    # for the analysis (including the signal aperture and the two apertures nearby). n.b.: the last is the
                    # center position of the signal aperture, i.e.: the best position (found with the desired method)
    # fpf_bckgs : array : it contains the FPF values evaluated for all the background apertures with respect to each other.
                        # n.b.: the signal aperture is ALWAYS excluded, while the background apertures nearby are excluded only if requested

    ## if method == 'fit', it also returns:
    # popt : array : best fit parameters found for the 2D gaussian fit (amplitude, xo, yo, sigma_x, sigma_y, theta, offset)
    # pcov : 2d array : the estimated covariance of popt as returned by the function scipy.optimize.curve_fit

    #############################################


    #Let's import the image:
    if filetype=='fits':
        data=fits.open(image)[0].data #NB: the fits file must consist of a single image (NO datacube)!
    if filetype=='array':
        data=image

    #Get the image size:
    size = len(data[0])

    #Define the center (i.e.: the image is supposed to be centered on the central star):
    center=np.array([size/2.,size/2.])

    #Round up the size of the range of search for the local maximum:
    ros=math.ceil(radius)#*2) #* 2.
    print 'ROS ',ros

    #Let's find the planet position:
    if method =='exact':
        planet_pos=planet_position
    if method == 'search':
        planet_pos=planets_finder(data,'array','local_max',planet_position=planet_position,range_of_search=ros)[0]
    if method == 'fit':
        planet_pos_guess=planets_finder(data,'array','local_max',planet_position=planet_position,range_of_search=ros)[0]
        #Create grid for the fit:
        x = range(size)
        y = range(size)
        x, y = np.meshgrid(x, y)
        sigma = (radius*2)/(2.*np.sqrt(2.*np.log(2.)))
        #create arrays of initial guess:
        #See which quadrant the planet is in, and give accordingly the correct additional factor for the arctan calculation:
        if planet_position[0]>= size/2. and planet_position[1]>=size/2.: #first quadrant
            arctan_fac=0.
            angle_factor=90.
        if planet_position[0]<=size/2. and planet_position[1]>=size/2.:#second quadrant
            arctan_fac=180.
            angle_factor=360.+90.
        if planet_position[0]<=size/2. and planet_position[1]<=size/2.:#third quadrant
              arctan_fac=180.
              angle_factor=270.+180.
        if planet_position[0]>=size/2. and planet_position[1]<=size/2.:#fourth quadrant
            arctan_fac=360.
            angle_factor=180.+270.

        theta=np.deg2rad(angle_factor-(np.degrees(np.arctan((planet_position[1] - size/2.)/(planet_position[0] - size/2.))) + arctan_fac))

        p0=[data[planet_pos_guess[1]][planet_pos_guess[0]],planet_pos_guess[0],planet_pos_guess[1],sigma,sigma,theta,0.]
        print '\n\ninitial guess : '+str(p0)
        popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), data.flatten(),p0=p0)
        planet_pos=np.array([popt[1],popt[2]])
        data_fitted=twoD_Gaussian((x,y),*popt)
        print popt
#[planet_pos_guess[0]-5:planet_pos_guess[0]+5, planet_pos_guess[1]-5:planet_pos_guess[1]+5]
    #let's calculate the distance between the planet and the star (i.e.: ASSUMING IMAGE CENTERING):
    d=np.sqrt((planet_pos[0] - center[0])**2 + (planet_pos[1]-center[1])**2)

    #Now, starting from the position of the planet, let's create a series of background apertures:

    #Let's calculate how many apertures can I put at that distance and given a certain radius of the aperture:
    n_bckg_aps=np.int((2*np.pi*d)/(2.*radius)) #NB: np.int assures that the number is rounded down to avoid overlaps

    #Let's save the center positions of all the background apertures:
    bckg_aps_cntr=np.ones((n_bckg_aps,2))
    angle_i=360./n_bckg_aps
    for i_apertures in range(0,n_bckg_aps):
        bckg_aps_cntr[i_apertures]=angular_coords_float(center,planet_pos,angle_i)
        angle_i=angle_i+(360./(n_bckg_aps))

    #Define the area (given the radius):
    area= np.pi * radius**2

    #Create the apertures and calculate the weighted pixel values in all of them: (the LAST one is the signal aperture)

    #Define some void arrays to be filled with the flux inside each aperture (i.e.: sum of all the weighted pixels),
    # the single weighted pixel values for all the apertures, the fractions for to be multiplied for all the apertures
    # and the not-weighted pixel values:
    # flux = np.zeros(len(bckg_aps_cntr), dtype=np.float)
    fractions=[]
    bckg_values=[]
    bckg_apertures=[]
    signal_aperture=[]

    #Let's define the extention of the region of interest for each aperture
    extents = np.zeros((len(bckg_aps_cntr), 4), dtype=int)

    extents[:, 0] = bckg_aps_cntr[:, 0] - radius + 0.5
    extents[:, 1] = bckg_aps_cntr[:, 0] + radius + 1.5
    extents[:, 2] = bckg_aps_cntr[:, 1] - radius + 0.5
    extents[:, 3] = bckg_aps_cntr[:, 1] + radius + 1.5

    ood_filter, extent, phot_extent = get_phot_extents(data, bckg_aps_cntr,extents)

    x_min, x_max, y_min, y_max = extent
    x_pmin, x_pmax, y_pmin, y_pmax = phot_extent

    if no_planet_nghbr==True:
        print 'Ignore the two apertures near the signal aperture'
    #For each aperture, let's calculate the fraction values given by the intersection between the pixel region of interest
    # and a circle of radius r, then use these fractions to evaluate the final weighted pixel values for each aperture:
    for index in range(len(bckg_aps_cntr)):
        fraction= geometry.circular_overlap_grid(x_pmin[index], x_pmax[index],
                                                  y_pmin[index], y_pmax[index],
                                                  x_max[index] - x_min[index],
                                                  y_max[index]- y_min[index],
                                                  radius, 0, subpix)
        fractions.append(fraction)

        #now, let's return the weighted pixel values but erasing the values ==0. (since the subpixel sampling can result
        # in a fraction =0 and so the pixel value is weighted by 0. and so it is =0. But it is not really 0.)

        #Ignore, if requested, the two apertures near the signal aperture:
        if no_planet_nghbr==True:
            if index != len(bckg_aps_cntr)-1 and index != len(bckg_aps_cntr)-2 and index != 0:
                bckg_values += (filter(lambda a: a !=0.,np.array((data[y_min[index]:y_max[index],x_min[index]:x_max[index]] * fraction)).ravel()))
                bckg_apertures.append(filter(lambda a: a !=0.,np.array((data[y_min[index]:y_max[index],x_min[index]:x_max[index]] * fraction)).ravel()))

            if index==len(bckg_aps_cntr)-1:
                signal_aperture += (filter(lambda a: a !=0.,np.array((data[y_min[index]:y_max[index],x_min[index]:x_max[index]] * fraction)).ravel()))

        else:
            if index != len(bckg_aps_cntr)-1:
                bckg_values += (filter(lambda a: a !=0.,np.array((data[y_min[index]:y_max[index],x_min[index]:x_max[index]] * fraction)).ravel()))
                bckg_apertures.append(filter(lambda a: a !=0.,np.array((data[y_min[index]:y_max[index],x_min[index]:x_max[index]] * fraction)).ravel()))

            else:
                signal_aperture += (filter(lambda a: a !=0.,np.array((data[y_min[index]:y_max[index],x_min[index]:x_max[index]] * fraction)).ravel()))

    signal_aperture=np.array(signal_aperture)
    bckg_values=np.array(bckg_values)
    bckg_apertures=np.array(bckg_apertures)

    ##################################################################################
    ### SNR calculation (as in Meshkat et al. 2014)
    snr=(np.sum(signal_aperture) - (np.mean(bckg_values)) * len(signal_aperture))/\
        (np.std(bckg_values) * np.sqrt(len(signal_aperture)) )

    ##################################################################################

    ### t-test value calculation:
    # Define the sample composed by the means inside the background apertures:
    bckg_apertures_means=[]
    bckg_apertures_fluxes=[]

    for index_mean in range(len(bckg_apertures)):
        bckg_apertures_means.append(np.sum(bckg_apertures[index_mean]) / area)
        bckg_apertures_fluxes.append(np.sum(bckg_apertures[index_mean]))

    bckg_apertures_means=np.array(bckg_apertures_means)
    bckg_apertures_fluxes=np.array(bckg_apertures_fluxes)

    #Calculate t-test value (according to Mawet et al. 2014):
    # In this way, the value that I am assigning to each resolution element is the MEAN of the pixel values:
    t_test=np.abs(((np.sum(signal_aperture)/area) - np.sum(bckg_apertures_means)/len(bckg_apertures_means))/\
           (np.std(bckg_apertures_means) * np.sqrt(1. + 1./len(bckg_apertures_means))))

    # #In this way, the value that I am assigning to each resolution element is the FLUX of the pixel values:
    # t_test=np.abs(((np.sum(signal_aperture)) - np.sum(bckg_apertures_fluxes)/len(bckg_apertures_fluxes))/\
    #        (np.std(bckg_apertures_fluxes) * np.sqrt(1. + 1./len(bckg_apertures_fluxes))))


    #Calculate the t-test value and fpf for all of the other background aperture (but always ignoring the signal one):
    fpf_bckgs=np.zeros(len(bckg_apertures_means))
    for i_t_test in range(len(bckg_apertures_means)):
        bckg_apertures_means_i=np.delete(bckg_apertures_means,i_t_test)
        t_test_i=np.abs((np.sum(bckg_apertures[i_t_test])/area - np.sum(bckg_apertures_means_i)/len(bckg_apertures_means_i))/\
                 (np.std(bckg_apertures_means_i) * np.sqrt(1. + 1./len(bckg_apertures_means_i))))
        fpf_i= 1. - t.cdf(t_test_i,len(bckg_apertures_means_i)-1)
        fpf_bckgs[i_t_test]=fpf_i

    #Given the t-test value, calculate the false alarm probability:
    # nb: define in this way, it is a ONE SIDE TEST!!
    fpf= 1. - t.cdf(t_test,len(bckg_apertures_means)-1)
    # print 'FPF : '+str(fpf)
    ##################################################################################

    #Plot the image together with the apertures, if requested:
    if plot ==True:
        # these are matplotlib.patch.Patch properties:
        props = dict(boxstyle='square', facecolor='white')

        #Let's create all the circles:
        #For the planet aperture:
        signal_aperture=plt.Circle((planet_pos[0],planet_pos[1]),radius,color='Crimson',fill=False,linewidth=1)

        fig = plt.gcf()
        ax=fig.add_subplot(111)
        plt.title(name,size=22)
        plt.imshow(data,origin='lower',alpha=0.5)
        fig.gca().add_artist(signal_aperture)

        plt.hold(True)
        if no_planet_nghbr==False:
            for i_plot in range(n_bckg_aps-1):# the minus 1 is to exclude the planet aperture
                background_aperture=plt.Circle((bckg_aps_cntr[i_plot][0],bckg_aps_cntr[i_plot][1]),radius,color='b',fill=False,linewidth=1)
                fig.gca().add_artist(background_aperture)

        if no_planet_nghbr==True:
            for i_plot in range(n_bckg_aps):
                if i_plot != 0 and i_plot != n_bckg_aps-1 and i_plot != n_bckg_aps-2:
                    background_aperture=plt.Circle((bckg_aps_cntr[i_plot][0],bckg_aps_cntr[i_plot][1]),radius,color='b',fill=False,linewidth=1)
                    fig.gca().add_artist(background_aperture)


        plt.text(0.55,0.8,'FPF = '+str('%s' % float('%.2g' % fpf))+'\npos = ['+str('%s' % float('%.3g' % planet_pos[0]))+' , '+str('%s' % float('%.3g' % planet_pos[1]))+']'
                 ,color='black',size=18,bbox=props,transform=ax.transAxes)

        plt.xlim(0,size)
        plt.ylim(0,size)
        plt.colorbar()
        if save==True:
            plt.savefig(str(save_dir)+str(name)+'%s.pdf'%j,clobber=1)
        plt.show()

        #now, let's plot the gaussian fit:
        if method=='fit':
            fig, ax = plt.subplots(1, 1)
            plt.title(name,size=22)
            ax.hold(True)
            ax.imshow(data,origin='lower',alpha=0.5)
            ax.contour(x, y, data_fitted.reshape(size, size), 5, colors='k')
            plt.text(0.55,0.85,'pos = ['+str('%s' % float('%.3g' % planet_pos[0]))+' , '+
                     str('%s' % float('%.3g' % planet_pos[1]))+']'
                     ,color='black',size=18,bbox=props,transform=ax.transAxes)
            plt.xlim(0,size)
            plt.ylim(0,size)
            if save==True:
                plt.savefig(str(save_dir)+'fit/'+str(name)+'_FIT.pdf')
            plt.show()

    # if method=='fit':
    #     print 'Fit : '+str(popt)


    #Save the result as an ASCII table:
    names=['FPF','t_test','SNR','Planet_pos_x','Planet_pos_y','Method']
    results=Table([[fpf],[t_test],[snr],[planet_pos[0]],[planet_pos[1]],[method]],names=names)
    results_name=save_dir+str(name)+'.txt'

    #Print the results:
    results.pprint(max_lines=-1,max_width=-1,align='^')

    #if requested, save the results
    if save==True:
        results.write(results_name,format='ascii.basic',delimiter='\t')
        print 'The result has been saved in '+str(results_name)+'\n'

    #return the final values:
    if method!='fit':
        return (fpf,t_test,snr,planet_pos,bckg_aps_cntr,fpf_bckgs)
    if method=='fit':
        return (fpf,t_test,snr,planet_pos,bckg_aps_cntr,fpf_bckgs,popt,pcov)

####################################
