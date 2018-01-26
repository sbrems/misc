__author__ = "Gabriele Cugno"

import sys
sys.path.append("/Users/Gabo/Desktop/MasterThesis/PynPoint")

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import math
from scipy.ndimage.interpolation import rotate,zoom
from scipy.stats import t
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
from scipy.ndimage import shift
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d,griddata
from astropy.table import Table
from side_functions import angular_coords_float
from fpf_calculator import fpf_calculator
import matplotlib.pyplot as plt
import PynPoint
from PynPoint.core import Pypeline
from PynPoint.io_modules import ReadFitsCubesDirectory
from PynPoint.core import ProcessingModule
from PynPoint.processing_modules import PSFSubtractionModule
from PynPoint.io_modules import ReadFitsCubesDirectory, WriteAsSingleFitsFile
import photutils as pu



class TwoDmapsModule(ProcessingModule):

    def __init__(self,
                 raw_data_in_tag,
                 psf_in_file,#has to be a fits file
                 working_place_in,
                 input_place_in
                 output_place_in,
                 inner_mask=0.1,
                 pc_number=20,
                 radius=2.0,
                 rough_mag_ext=8.0,
                 stat = 'mean',
                 name_in="detection_limits",
                 cl=0.9999,
                 image_res_in_tag="res_mean",
                 radial_step_pix=5,
                 angular_step_deg=30,
                 mag_step=.2 ,
                 cutting_psf=False,
                 psf_cut_ext=20.,
                 method='exact',
                 fake_planets_out_tag="fake_planets",
                 fake_planets_psf_sub_out_tag="fake_planets_psf_sub",
                 savefolder='None',#non importato come parametro!
                 plot=False,
                 save=False,
                 dir_fake='Planet_fake',
                 mag_default=0.,
                 subpix_precision=1.):

        super(TwoDmapsModule, self).__init__(name_in)

        # Ports

        self.m_science_in_port = self.add_input_port(raw_data_in_tag)
        self.m_result_in_port = self.add_input_port(image_res_in_tag)
        
        self.m_fake_planets_out_port = self.add_output_port(fake_planets_out_tag)
        self.m_fake_planets_psf_sub_out_port = self.add_output_port(fake_planets_psf_sub_out_tag)

        # Parameter
        self.m_pc_number = pc_number
        self.m_radius = radius
        self.m_rough_mag_ext = rough_mag_ext
        self.m_inner_mask = inner_mask
        self.m_psf_in_file = psf_in_file
        self.m_stat = stat
        #self.m_psf_scale = psf_scale
        self.m_cl = cl
        self.m_radial_step_pix = radial_step_pix
        self.m_angular_step_deg = angular_step_deg
        self.m_mag_step = mag_step
        self.m_cutting_psf = cutting_psf
        self.m_psf_cut_ext = psf_cut_ext
        self.m_plot = plot
        self.m_save = save
        self.m_mag_default = mag_default
        self.m_method=method
        self.m_dir_fake_planet=dir_fake
        self.m_raw_data_in_tag = raw_data_in_tag
        self.m_savepath = savefolder
        self.m_subpix_precision = subpix_precision
        self.m_working_place_in = working_place_in
        self.m_input_place_in = input_place_in
        self.m_output_place_in = output_place_in

    


    def create_fake_planet(self,
                           raw_data_in_tag,
                           psf_fits,
                           dir_fake_planets,
                           fake_planet_pos,
                           mag,
                           subpix_precision,
                           negative_flux,
                           pc_number=0.,
                           inner_mask_cent_size=0.,
                           cutting_psf=False,
                           psf_cut=20.,
                           returnim=False,
                           save=False,
                           savefolder='None',
                           psf_type='array'):
    ##################################################################
    
        self.m_raw_in_port = self.add_input_port(raw_data_in_tag)

        #Let's inform the user:
        if negative_flux==True:
            print 'Inserting a fake negative planet at pixel position '+str(fake_planet_pos)+', with a magnitude contrast of '+str(mag)\
                + '\n(The image will be enlarged by a factor of '+str(subpix_precision)+')'
        if negative_flux==False:
            print 'Inserting a fake positive planet at pixel position '+str(fake_planet_pos)+', with a magnitude contrast of '+str(mag)+ '\n(The image will be enlarged by a factor of '+str(subpix_precision)+')'
        
        #import al the images
        image=self.m_raw_in_port.get_all()
    
        #Lets find the size of the image:
        image_size=np.shape(image)[1]
    
        #Let's calculate the sub pixel position, given the subpixel precision:
        sub_pos=np.array([fake_planet_pos[0] * subpix_precision,fake_planet_pos[1]*subpix_precision])

    
        #Calculate the flux reduction:
        if negative_flux==True:
            flux_red=-10**(-mag/2.5)
        if negative_flux==False:
            flux_red=10**(-mag/2.5)
    
        #import the PSF:
        if psf_type=='array':
            psf=psf_fits
        if psf_type=='fits':
            psf=fits.open(psf_fits)[0].data
    
        center=[len(psf)/2.,len(psf)/2.] #Assumes centering already done
        

        
        #Cut the PSF if requested:
        if cutting_psf==True:
            star_psf=psf[int(center[1]-psf_cut/2.):int(center[1]+psf_cut/2.),int(center[0]-psf_cut/2.):int(center[0]+psf_cut/2.)]
        else:
            star_psf=psf

        #Multiply the psf by the reduction factor:
        planet_PSF = flux_red*star_psf
    
        #Let's do subpixel sampling: decide the zoom factor
        return_factor=1./subpix_precision
    
        #Insert the planet PSF:
        #Enlarge the planet psf image:
        planet_PSF_enlarged=zoom(planet_PSF,subpix_precision)

        #Create an image with the same size of the fits files, but filled with zeros (to be used to insert the fake planet):
        fake_image_final = np.zeros([len(image),int(image_size),int(image_size)])
        fake_image= np.zeros([len(image),int(image_size*subpix_precision),int(image_size*subpix_precision)])


        for i in range(len(image)):
            #let's insert the planet at the desired position:
            fake_image[i,int(sub_pos[1]-len(planet_PSF_enlarged[1])/2.):int(sub_pos[1]+len(planet_PSF_enlarged[1])/2.), int(sub_pos[0]-len(planet_PSF_enlarged[0])/2.):int(sub_pos[0]+len(planet_PSF_enlarged[0])/2.)]=planet_PSF_enlarged[:,:]
            #Let's rotate the images:
            fake_image_i=rotate(fake_image[i, :, :],angles[i],reshape=False)
            
            #Let's enlarge the image of the desired zoom factor:
            data_enlarge=np.squeeze(zoom(image[i,:,:],subpix_precision))
                       
            #Let's add the raw data (enlarged) and the void image with the fake planet:
            new_data=data_enlarge+fake_image_i
                       
            #Shrink everything down again:
            fake_image_final[i, :, :]=zoom(new_data,return_factor)

        # Save new images
        self.m_fake_planets_out_port.set_all(fake_image_final)
        self.m_fake_planets_out_port.add_history_information("fake planets", "fake planets")
        self.m_fake_planets_out_port.copy_attributes_from_input_port(self.m_science_in_port)
        self.m_fake_planets_out_port.close_port()

    
        if returnim==True:
        
            # New Pipeline
            pipeline2 = Pypeline(self.m_working_place_in, self.m_input_place_in, self.m_output_place_in)
        
            # Subtract stars PSF
            psf_sub_on_fake_planet = PSFSubtractionModule(pca_number=self.m_pc_number,
                                        cent_size=self.m_inner_mask,
                                       name_in="PSF_subtraction_on_fake_planets",
                                       images_in_tag="fake_planets",
                                       reference_in_tag="fake_planets",
                                       res_mean_tag="res_mean_fake",
                                        res_median_tag="res_median_fake",
                                        res_rot_mean_clip_tag="res_clip_fake"
                                           )
                                           

            pipeline2.add_module(psf_sub_on_fake_planet)
            pipeline2.run()
            
            # get image result
            if self.m_stat == 'mean':
                mean_image = self.add_input_port("res_mean_fake").get_all()
            if self.m_stat == 'median':
                mean_image = self.add_input_port("res_median_fake").get_all()
            if self.m_stat == 'clip':
                mean_image = self.add_input_port("res_clip_fake").get_all()
        
            self.m_fake_planets_psf_sub_out_port.set_all(mean_image)
            self.m_fake_planets_psf_sub_out_port.add_history_information("fake planets psf", "fake planets psf")
            self.m_fake_planets_psf_sub_out_port.copy_attributes_from_input_port(self.m_raw_in_port)
            self.m_fake_planets_psf_sub_out_port.close_port()
            return (mean_image)

    
    
###################################################################################################
    
    

    def run(self):

        #calculate the FPF threshold for the requested Confidence Level (CL) value:
        fpf_threshold=1.-t.cdf(t.interval(self.m_cl,30),30)[1]
        print '\n The FPF threshold relative to a confidence level of '+str(self.m_cl*100)+' %, is '+str(fpf_threshold)+'\n'

        #Import the unsaturated image to be used for the creation of fake planets and normalize it to the value it would have with the same exposure time of our image:
        unsat = fits.open(self.m_psf_in_file)[0].data

        
        if self.m_cutting_psf==True:
            psf_cut=self.m_psf_cut_ext
        if self.m_cutting_psf==False:
            psf_cut=len(unsat)

        # Load image processed by PynPoint
        im_PynPointed = self.m_result_in_port.get_all()

        # Load pixel to arcsec conversion factor
        pix2mas = self.m_science_in_port.get_attribute("ESO INS PIXSCALE") * 1000.
    
        # Size of the image:
        size_image = np.shape(im_PynPointed)[1]
        print 'The size of the image is '+str(size_image)+'\n'
        # center of the image:
        center = np.array([int(size_image / 2.), int(size_image / 2.)])
        # Inner mask in pixels:
        inner_mask_pix = np.ceil(self.m_inner_mask * size_image)

        #Create list of angles given the angular step in degrees:
        angles=[]
        for i_angles in range(int(360/self.m_angular_step_deg)):
            angles.append(i_angles * self.m_angular_step_deg)
        angles=np.array(angles)


        #Calculate how many points are possible, given the size of the inner mask, the size of the psf, the size of the image and the required radial step:
        num_points = int((size_image-1-(center[0]+inner_mask_pix+psf_cut/1.3))/self.m_radial_step_pix)
        print 'There are '+ str(num_points) +' points\n'


        #Create list of positions (given the radial step in pixels and the angles):
        planet_positions = np.zeros((num_points,2))
        positions_all=[]
        for i in range(num_points):
            pos_i_x=int(center[0]+inner_mask_pix+psf_cut/2. + self.m_radial_step_pix*i)
            pos_i_y=int(center[1])
        
            #Check that the points do not fall outside the image boundaries (considering the inner mask and the psf cut):
            while pos_i_x<(0+ psf_cut/2.):
                pos_i_x+=1
            while pos_i_x<(center[0]+inner_mask_pix + psf_cut/2.) and pos_i_x>center[0]:
                pos_i_x+=1
            while pos_i_x>(size_image - psf_cut/2.):
                pos_i_x-=1
            while pos_i_x >(center[0]-inner_mask_pix- psf_cut/2.) and pos_i_x<center[0]:
                pos_i_x-=1
            while pos_i_y <(0+ psf_cut/2.):
                pos_i_y+=1
            while pos_i_y<(center[0]+inner_mask_pix + psf_cut/2.) and pos_i_y>center[1]:
                pos_i_y+=1
            while pos_i_y>(size_image - psf_cut/2.):
                pos_i_y-=1
            while pos_i_y>(center[1]-inner_mask_pix- psf_cut/2.) and pos_i_y<center[1]:
                pos_i_y-=1
        
            planet_positions[i][0]=pos_i_x
            planet_positions[i][1]=pos_i_y

        for m in range(len(angles)):
            for l in range(len(planet_positions)):
                pos_l_x=np.round(angular_coords_float(center,planet_positions[l],angles[m]) [0])
                pos_l_y=np.round(angular_coords_float(center,planet_positions[l],angles[m]) [1])
            
                positions_all.append(np.array([pos_l_x,pos_l_y]))

        positions_all=np.array(positions_all)
    
        print 'I will test '+str(len(positions_all))+' positions:\n',positions_all,' \n'

        #create the empty arrays where to save the max positions, the relative distances (both in pixels and in arcsec),the 5 sigma threshold values for the FPF (for each distance) and the final magnitude contrasts:
        max_positions=np.ones((len(positions_all),2))
        distances=np.zeros((len(positions_all))) # in pixels
        dist_arcsec=np.zeros((len(positions_all))) #in arcsec
        mags=np.zeros((len(positions_all)))


        #Loop on all positions:
        for i_pos in range(len(positions_all)):
    
            rough_mag=self.m_rough_mag_ext #set the initial magnitude
        
            print '\n\n STEP '+str(i_pos+1)+' of '+str(len(positions_all))+'\n'
            
            #set position
            pos=positions_all[i_pos]
        
            #Check if for this position the FPF on the original image is below the threshold (i.e: if there is already a planet there):
            results_original=fpf_calculator(im_PynPointed,'array',pos,self.m_radius,method=self.m_method,no_planet_nghbr=True, plot=False, save=True, save_dir=self.m_savepath, j=i_pos)
            if self.m_method=='fit':
                max_pos_original=np.array([results_original[6][1],results_original[6][2]])
            if self.m_method=='exact' or self.m_method=='search':
                max_pos_original=np.array([results_original[3][0],results_original[3][1]])
            fpf_original= results_original[0]
            dist_original=np.sqrt((np.abs(max_pos_original[0]-center[0]))**2 +(np.abs(max_pos_original[1]-center[1]))**2)# in pixels
            dof_dist_original=(np.int((2*np.pi*dist_original)/(2.*self.m_radius))) -2


            #Evaluate the 5sigma threshold for the FPF at this distance, given the radius:
            if fpf_original<= fpf_threshold:
                print 'For the position '+str(max_pos_original)+' the FPF is already below the threshold for a '+str(self.m_cl*100)+' % confidence level.'
                mags[i_pos]=self.m_mag_default
                max_positions[i_pos][0]=max_pos_original[0]
                max_positions[i_pos][1]=max_pos_original[1]
                distances[i_pos]=dist_original
                dist_arcsec[i_pos]=dist_original*pix2mas * 10**(-3)

            else:
                #Insert a planet with the initial rough magnitude, and check whether the FPF is below the given threshold.
                # If it is below, then start to increase it.
                # If it is above, decrease the initial magnitude and try again:
                im_init= self.create_fake_planet(self.m_raw_data_in_tag,unsat,self.m_dir_fake_planet,pos,self.m_rough_mag_ext,self.m_subpix_precision,
                                False,returnim=True,pc_number=self.m_pc_number,inner_mask_cent_size=self.m_inner_mask,cutting_psf=self.m_cutting_psf,psf_cut=psf_cut)
                results_init=fpf_calculator(im_init,'array',pos,self.m_radius,method=self.m_method,no_planet_nghbr=True)
                if self.m_method=='fit':
                    max_pos_init=np.array([results_original[6][1],results_original[6][2]])
                if self.m_method=='exact' or self.m_method=='search':
                    max_pos_init=np.array([results_original[3][0],results_original[3][1]])
                fpf_init=results_init[0]
                dist_init=np.sqrt((np.abs(max_pos_init[0]-center[0]))**2 +(np.abs(max_pos_init[1]-center[1]))**2)
                dof_dist_init=(np.int((2*np.pi*dist_init)/(2.*self.m_radius))) -2
                    
                
                if fpf_init >= fpf_threshold:
                    #Decrease the magnitude contrast until the FPF is below the 5 sigma threshold:
                    print 'The initial rough magnitude is too high, the magnitude contrast will be decreased until the FPF ' \
                        'drops below the '+str(self.m_cl*100)+' % confidence level threshold.'
                    while fpf_init>=fpf_threshold:
                        if rough_mag<2.:
                            break
                        rough_mag-=1
                        im_init=self.create_fake_planet(self.m_raw_data_in_tag,unsat,self.m_dir_fake_planet,pos,rough_mag,self.m_subpix_precision,
                                               False,returnim=True, pc_number=self.m_pc_number,inner_mask_cent_size=self.m_inner_mask,
                                               cutting_psf=self.m_cutting_psf,psf_cut=psf_cut)
                        results_init=fpf_calculator(im_init,'array',pos,self.m_radius,method=self.m_method,no_planet_nghbr=True)
                        if self.m_method=='fit':
                            max_pos_init=np.array([results_original[6][1],results_original[6][2]])
                        if self.m_method=='exact' or self.m_method=='search':
                            max_pos_init=np.array([results_original[3][0],results_original[3][1]])
                        dist_init=np.sqrt((np.abs(max_pos_init[0]-center[0]))**2 +(np.abs(max_pos_init[1]-center[1]))**2)
                        dof_dist_init=(np.int((2*np.pi*dist_init)/(2.*self.m_radius))) -2
                        fpf_init=results_init[0]
                            
                            
                if fpf_init < fpf_threshold:
                    print 'For the position '+str(pos)+' ,the magnitude contrast of '+str(rough_mag)+' is a good starting point.\n'
                    #Save the max pos and the fpf 5sigma threshold:
                    max_positions[i_pos][0]=max_pos_init[0]
                    max_positions[i_pos][1]=max_pos_init[1]
                    distances[i_pos]=dist_init
                    dist_arcsec[i_pos]=dist_init*pix2mas * 10**(-3)
                    
                    print 'Increasing the magnitude contrast in step of '+str(self.m_mag_step)+'...'
                    mag_init=rough_mag

                    # As long as FPF is smaller than the relative threshold, increase magnitude contrast
                    mag=mag_init
                    fpf=fpf_init
                    max_pos=max_pos_init
            
                    fpf_all=[]
                    mag_all=[]

                    while fpf<=fpf_threshold:
                        fpf_all.append(fpf)
                        mag_all.append(mag)
                        mag+=self.m_mag_step
                        im_loop= self.create_fake_planet(self.m_raw_data_in_tag,unsat,self.m_dir_fake_planet,pos,mag,self.m_subpix_precision,False,returnim=True, pc_number=self.m_pc_number,inner_mask_cent_size=self.m_inner_mask,
                            cutting_psf=self.m_cutting_psf,psf_cut=psf_cut)
                        fpf=fpf_calculator(im_loop,'array',max_pos,self.m_radius,no_planet_nghbr=True, method=self.m_method)[0]
                        print 'For the magnitude contrast of '+str(mag)+' the FPF is '+str(fpf)+'\n'
                        
                    mag_all.append(mag)
                    fpf_all.append(fpf)

                    #Interpolate and find the magnitude at 5 sigma threshold:
                    fpf_l=fpf_all[-2]
                    fpf_r=fpf_all[-1]
                    mag_l=mag_all[-2]
                    mag_r=mag_all[-1]
                    # #interp:
                    coeff=np.polyfit([mag_l,mag_r],[fpf_l,fpf_r],1)
                    polynomial = np.poly1d(coeff)
                    x_interp = np.linspace(mag_l,mag_r,100)
                    y_interp = polynomial(x_interp)
                
                    mag_final=x_interp[np.argmin(np.abs(y_interp-fpf_threshold))]
                    mags[i_pos]=mag_final
                
                    print 'For the position '+str(max_pos)+'at a distance of '+str(dist_arcsec[i_pos])+' the magnitude contrast at '+str(self.m_cl*100)+' % confidence level threshold is '+str(mag_final)+'\n'

        #CONTRAST CURVE:
        # The contrast curve is created as a linear interpolation of the mean values at different radial distance:

        #Mean of the magnitude contrast at the same mean distance:
        mean_mags=np.mean(np.reshape(mags,(len(mags)/num_points,num_points)),0)
        mean_dist=np.mean(np.reshape(dist_arcsec,(len(dist_arcsec)/num_points,num_points)),0)


        #interpolate the values:
        f_interp=interp1d(mean_dist,mean_mags,'linear')
        x_interp_mags = np.linspace(np.min(mean_dist),np.max(mean_dist),100)
        y_interp_mags = f_interp(x_interp_mags)
    
        #Create a table where to save, for each radial distance, the correspondent 5 sigma magnitude contrast limit
        table_contrast_names=['Radial distance','Mag 5-sigma']
        contrast_curves=Table([x_interp_mags,y_interp_mags],names=table_contrast_names)
        contrast_curves_name=self.m_savepath+'Contrast_curve.txt'


        #2D DETECTION MAPS:
        x_2d=positions_all[:,0]
        y_2d=positions_all[:,1]
        #Create grid:
        xi_2d=range(size_image)
        yi_2d=range(size_image)
    
        grid_x, grid_y = np.meshgrid(xi_2d,yi_2d)

        #Interpolate the data:
        z_interp_2d=griddata((x_2d,y_2d),mags, (grid_x,grid_y),method='linear')

        #Erase the center (i.e.: to match with the inner mask):
        radius_erase=inner_mask_pix
        x_mask,y_mask=np.ogrid[-size_image/2.:size_image-size_image/2., -size_image/2.:size_image-size_image/2.]
        mask = x_mask*x_mask + y_mask*y_mask <= radius_erase*radius_erase
        z_interp_2d[mask]=np.nan
        twoD_name=self.m_savepath+'2D_detection_maps.fits'

        #Create a table where to save the original positions and magnitude contrast:
        results_final_names=['Pos x','Pos y','Max pos x','Max pos y','Mag 5-sigma']
        results_final=Table([positions_all[:,0],positions_all[:,1],max_positions[:,0],max_positions[:,1],mags],names=results_final_names)
        results_final_path=self.m_savepath+'2d_maps_cntr_curves.txt'
        
        
        
        #if requested, save the results
        if self.m_save==True:
            contrast_curves.write(contrast_curves_name,format='ascii.basic',delimiter='\t')
            print 'The result for the contrast curve has been saved as a txt file in '+str(contrast_curves_name)+'\n'
            
            twoD=fits.PrimaryHDU(z_interp_2d)
            twoD.writeto(twoD_name)
            print 'The result for the 2D detection map has been saved as a fits file in '+str(twoD_name)
            
            results_final.write(results_final_path,format='ascii.basic',delimiter='\t')
            print 'The original pixel positions where the fake planet have been inserted, together with the max positions, ' \
                ' and final magnitude contrast have been saved as a txt file in '\
                +str(results_final_path)

        #Plot the results:
            #The positions that have been used:
        plt.figure()
        plt.subplot(111)
        plt.title('Positions',size=20)
        plt.imshow(im_PynPointed,origin='lower',alpha=0.5)
        plt.hold(True)
        plt.plot(positions_all[:,0],positions_all[:,1],'Crimson',marker='o',linestyle='')
        plt.xlim(0,size_image)
        plt.ylim(0,size_image)
        if self.m_save==True:
            plt.savefig(self.m_savepath+'Positions.pdf',clobber=1)
        if self.m_plot==True:
            plt.show()


            #The contrast curve:
        plt.figure()
        plt.subplot(111)
        plt.title('Contrast Curve',size=20)
        plt.plot(x_interp_mags,y_interp_mags,'DodgerBlue',linewidth=2.,linestyle='--')
        plt.xlabel('Radial distance [arcsec]',size=15)
        plt.ylabel('Magnitude contrast',size=15)
        plt.ylim(np.max(mean_mags)+2,np.min(mean_mags)-2)
        if self.m_save==True:
            plt.savefig(self.m_savepath+'Contrast_curve.pdf',clobber=1)
        if self.m_plot==True:
            plt.show()


            #The 2D detection map:
        plt.figure()
        plt.subplot(111,adjustable='box', aspect=1)
        plt.title('2D detection map',size=20)
        plt.contour(xi_2d,yi_2d,z_interp_2d,15,linewidths=0.5,colors='k',vmin=np.min(mag),vmax=np.max(mag))
        plt.contourf(xi_2d,yi_2d,z_interp_2d,15,cmap=plt.cm.bone_r,vmin=np.min(mag),vmax=np.max(mag))
        plt.xlim(0,size_image)
        plt.ylim(0,size_image)
        cbar=plt.colorbar()
        cbar.ax.set_title('       $\Delta$mag',size=20)
        if self.m_save==True:
            plt.savefig(self.m_savepath+'2D_detection_map.png',clobber=1)
        if self.m_plot==True:
            plt.show()


        #if requested, save the results
        if self.m_save==True:
            contrast_curves.write(contrast_curves_name,format='ascii.basic',delimiter='\t')
            print 'The result for the contrast curve has been saved as a txt file in '+str(contrast_curves_name)+'\n'
        
            twoD=fits.PrimaryHDU(z_interp_2d)
            twoD.writeto(twoD_name)
            print 'The result for the 2D detection map has been saved as a fits file in '+str(twoD_name)
        
            results_final.write(results_final_path,format='ascii.basic',delimiter='\t')
            print 'The original pixel positions where the fake planet have been inserted, together with the max positions, ' \
                ' and final magnitude contrast have been saved as a txt file in '\
                +str(results_final_path)


